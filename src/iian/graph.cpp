#include "iian/graph.h"
#include "iian/kv_cache.h"
#include "iian/log.h"
#include "paged_attn.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace iian {

static int rope_mode_of(RopeType t) {
    switch (t) {
        case RopeType::NEOX: return GGML_ROPE_TYPE_NEOX;
        case RopeType::MROPE: return GGML_ROPE_TYPE_MROPE;
        case RopeType::VISION: return GGML_ROPE_TYPE_VISION;
        default: return GGML_ROPE_TYPE_NORMAL;
    }
}

GraphContext::GraphContext(ggml_context * ctx, ggml_cgraph * gf_, const Model & model_, const UBatch & ub_, PagedKVCache & kv_, bool fa, bool paged)
    : ctx0(ctx), gf(gf_), model(model_), hp(model_.hparams()), ub(ub_), kv(kv_), flash_attn(fa), paged_attn(paged),
      n_embd(hp.n_embd), n_layer(hp.n_layer), n_tokens(ub_.n_tokens), n_outputs(ub_.n_outputs), n_rot(hp.n_rot),
      n_ctx_orig(hp.n_ctx_orig_yarn), freq_base(hp.rope_freq_base), freq_scale(hp.rope_freq_scale),
      ext_factor(hp.yarn_ext_factor), attn_factor(hp.yarn_attn_factor), beta_fast(hp.yarn_beta_fast), beta_slow(hp.yarn_beta_slow),
      rope_mode(rope_mode_of(hp.rope_type)) {}

GraphContext::~GraphContext() = default;


void GraphContext::cb(ggml_tensor * t, const char * name, int il) {
    if (il >= 0) ggml_format_name(t, "%s-%d", name, il); else ggml_set_name(t, name);
}

ggml_tensor * GraphContext::build_inp_embd(ggml_tensor * tok_embd) {
    inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);
    ggml_tensor * cur = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    cb(cur, "inp_embd", -1);
    return cur;
}

ggml_tensor * GraphContext::build_inp_pos() {
    inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);
    return inp_pos;
}

ggml_tensor * GraphContext::build_inp_out_ids() {
    inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_name(inp_out_ids, "inp_out_ids");
    ggml_set_input(inp_out_ids);
    return inp_out_ids;
}

void GraphContext::build_attn_inputs() {
    inp_k_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
    ggml_set_name(inp_k_idxs, "inp_k_idxs");
    ggml_set_input(inp_k_idxs);
    inp_v_idxs = inp_k_idxs;   // same cell for K and V (non-transposed V)
    if (paged_attn) return;    // the paged kernel walks block tables directly: no masks needed
    const ggml_type mask_type = flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    inp_kq_mask = ggml_new_tensor_2d(ctx0, mask_type, ub.n_kv, n_tokens);
    ggml_set_name(inp_kq_mask, "inp_kq_mask");
    ggml_set_input(inp_kq_mask);

    // Hybrid SWA/global models (gemma3, ...): a second, sliding-window restricted mask that build_attn() uses
    // for layers flagged in hp.is_swa; inp_kq_mask then stays the full causal mask for the global layers.
    if (hp.n_swa > 0 && has_swa_layers()) {
        inp_kq_mask_swa = ggml_new_tensor_2d(ctx0, mask_type, ub.n_kv, n_tokens);
        ggml_set_name(inp_kq_mask_swa, "inp_kq_mask_swa");
        ggml_set_input(inp_kq_mask_swa);
    }
}

bool GraphContext::has_swa_layers() const {
    for (auto f : hp.is_swa) if (f) return true;
    return false;
}

ggml_tensor * GraphContext::build_norm(ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b, NormKind kind, int il) {
    switch (kind) {
        case NormKind::RMS:   cur = ggml_rms_norm(ctx0, cur, hp.f_norm_rms_eps); break;
        case NormKind::LAYER: cur = ggml_norm(ctx0, cur, hp.f_norm_eps); break;
        case NormKind::GROUP: throw std::runtime_error("group norm not supported yet");
    }
    if (w) { cur = ggml_mul(ctx0, cur, w); cb(cur, "norm_w", il); }
    if (b) { cur = ggml_add(ctx0, cur, b); cb(cur, "norm_b", il); }
    return cur;
}

ggml_tensor * GraphContext::build_ffn(ggml_tensor * cur,
                                      ggml_tensor * up, ggml_tensor * up_b,
                                      ggml_tensor * gate, ggml_tensor * gate_b,
                                      ggml_tensor * down, ggml_tensor * down_b,
                                      FfnOp op, FfnGate gate_type, int il) {
    ggml_tensor * tmp = up ? ggml_mul_mat(ctx0, up, cur) : cur;
    cb(tmp, "ffn_up", il);
    if (up_b) { tmp = ggml_add(ctx0, tmp, up_b); cb(tmp, "ffn_up_b", il); }

    if (gate) {
        cur = ggml_mul_mat(ctx0, gate, gate_type == FfnGate::SEQ ? tmp : cur);
        cb(cur, "ffn_gate", il);
        if (gate_b) { cur = ggml_add(ctx0, cur, gate_b); cb(cur, "ffn_gate_b", il); }
    } else {
        cur = tmp;
    }

    switch (op) {
        case FfnOp::SILU:
            if (gate && gate_type == FfnGate::PAR) { cur = ggml_swiglu_split(ctx0, cur, tmp); cb(cur, "ffn_swiglu", il); }
            else { cur = ggml_silu(ctx0, cur); cb(cur, "ffn_silu", il); }
            break;
        case FfnOp::GELU:
            if (gate && gate_type == FfnGate::PAR) { cur = ggml_geglu_split(ctx0, cur, tmp); cb(cur, "ffn_geglu", il); }
            else { cur = ggml_gelu(ctx0, cur); cb(cur, "ffn_gelu", il); }
            break;
        case FfnOp::RELU:
            cur = ggml_relu(ctx0, cur); cb(cur, "ffn_relu", il);
            if (gate && gate_type == FfnGate::PAR) cur = ggml_mul(ctx0, cur, tmp);
            break;
        case FfnOp::RELU_SQR:
            cur = ggml_relu(ctx0, cur); cur = ggml_sqr(ctx0, cur); cb(cur, "ffn_relu_sqr", il);
            if (gate && gate_type == FfnGate::PAR) cur = ggml_mul(ctx0, cur, tmp);
            break;
        case FfnOp::SWIGLU: cur = ggml_swiglu(ctx0, cur); cb(cur, "ffn_swiglu", il); break;
        case FfnOp::GEGLU:  cur = ggml_geglu(ctx0, cur);  cb(cur, "ffn_geglu", il);  break;
    }
    if (gate && gate_type == FfnGate::PAR && (op == FfnOp::SWIGLU || op == FfnOp::GEGLU)) {
        // already fused above; nothing
    }
    if (gate && gate_type == FfnGate::SEQ && op != FfnOp::SWIGLU && op != FfnOp::GEGLU) {
        // sequential gate: activation applied to gate(tmp) only
    }

    if (down) { cur = ggml_mul_mat(ctx0, down, cur); cb(cur, "ffn_down", il); }
    if (down_b) { cur = ggml_add(ctx0, cur, down_b); cb(cur, "ffn_down_b", il); }
    return cur;
}

ggml_tensor * GraphContext::build_moe_ffn(ggml_tensor * cur, ggml_tensor * gate_inp, ggml_tensor * up_exps, ggml_tensor * gate_exps,
                                          ggml_tensor * down_exps, ggml_tensor * exp_probs_b, int64_t n_expert, int64_t n_expert_used,
                                          FfnOp op, bool norm_w, bool scale_w, float w_scale, int il) {
    const int64_t n_tok = cur->ne[1];
    ggml_tensor * logits = ggml_mul_mat(ctx0, gate_inp, cur);   // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);
    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    cb(probs, "ffn_moe_probs", il);
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b) { selection_probs = ggml_add(ctx0, probs, exp_probs_b); cb(selection_probs, "ffn_moe_probs_biased", il); }

    ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, (int) n_expert_used);   // [n_expert_used, n_tokens] I32
    cb(selected_experts, "ffn_moe_topk", il);
    ggml_tensor * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tok), selected_experts);   // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);
    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tok);
        ggml_tensor * sum = ggml_sum_rows(ctx0, weights);
        weights = ggml_div(ctx0, weights, sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tok);
    }
    if (scale_w) weights = ggml_scale(ctx0, weights, w_scale);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tok);
    ggml_tensor * up = ggml_mul_mat_id(ctx0, up_exps, cur, selected_experts);   // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);
    ggml_tensor * experts;
    if (gate_exps) {
        ggml_tensor * gate = ggml_mul_mat_id(ctx0, gate_exps, cur, selected_experts);
        cb(gate, "ffn_moe_gate", il);
        experts = op == FfnOp::GELU ? ggml_geglu_split(ctx0, gate, up) : ggml_swiglu_split(ctx0, gate, up);
    } else {
        experts = op == FfnOp::GELU ? ggml_gelu(ctx0, up) : ggml_silu(ctx0, up);
    }
    experts = ggml_mul_mat_id(ctx0, down_exps, experts, selected_experts);   // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);
    experts = ggml_mul(ctx0, experts, weights);

    // sum experts
    ggml_tensor * moe_out = nullptr;
    for (int64_t i = 0; i < n_expert_used; i++) {
        ggml_tensor * e = ggml_view_2d(ctx0, experts, n_embd, n_tok, experts->nb[2], i * experts->nb[1]);
        moe_out = moe_out ? ggml_add(ctx0, moe_out, e) : e;
    }
    if (n_expert_used == 1) moe_out = ggml_cont(ctx0, moe_out);
    cb(moe_out, "ffn_moe_out", il);
    (void) n_expert;
    return moe_out;
}

GraphContext::QKV GraphContext::build_qkv(const LayerWeights & L, ggml_tensor * cur, int64_t n_embd_head, int64_t n_head, int64_t n_head_kv, int il) {
    ggml_tensor * q, * k, * v;
    const int64_t n_embd_q = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;
    if (L.wqkv) {
        ggml_tensor * qkv = ggml_mul_mat(ctx0, L.wqkv, cur);
        cb(qkv, "wqkv", il);
        if (L.bqkv) { qkv = ggml_add(ctx0, qkv, L.bqkv); cb(qkv, "bqkv", il); }
        if (hp.f_clamp_kqv > 0.0f) qkv = ggml_clamp(ctx0, qkv, -hp.f_clamp_kqv, hp.f_clamp_kqv);
        q = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens, ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        k = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], ggml_row_size(qkv->type, n_embd_q));
        v = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        q = ggml_mul_mat(ctx0, L.wq, cur); cb(q, "Qcur", il);
        if (L.bq) { q = ggml_add(ctx0, q, L.bq); cb(q, "Qcur_b", il); }
        k = ggml_mul_mat(ctx0, L.wk, cur); cb(k, "Kcur", il);
        if (L.bk) { k = ggml_add(ctx0, k, L.bk); cb(k, "Kcur_b", il); }
        v = ggml_mul_mat(ctx0, L.wv, cur); cb(v, "Vcur", il);
        if (L.bv) { v = ggml_add(ctx0, v, L.bv); cb(v, "Vcur_b", il); }
        if (hp.f_clamp_kqv > 0.0f) {
            q = ggml_clamp(ctx0, q, -hp.f_clamp_kqv, hp.f_clamp_kqv);
            k = ggml_clamp(ctx0, k, -hp.f_clamp_kqv, hp.f_clamp_kqv);
            v = ggml_clamp(ctx0, v, -hp.f_clamp_kqv, hp.f_clamp_kqv);
        }
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head,    n_tokens);
        k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head_kv, n_tokens);
    }
    return { q, k, v };
}

ggml_tensor * GraphContext::build_rope(ggml_tensor * cur, ggml_tensor * rope_factors) {
    return build_rope(cur, rope_factors, freq_base, freq_scale);
}

ggml_tensor * GraphContext::build_rope(ggml_tensor * cur, ggml_tensor * rope_factors, float freq_base_l, float freq_scale_l) {
    return ggml_rope_ext(ctx0, cur, inp_pos, rope_factors, (int) n_rot, rope_mode, (int) n_ctx_orig,
                         freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
}

float GraphContext::rope_freq_base_l(int il) const {
    return (hp.n_swa > 0 && hp.is_swa[il]) ? hp.rope_freq_base_swa : freq_base;
}

float GraphContext::rope_freq_scale_l(int il) const {
    return (hp.n_swa > 0 && hp.is_swa[il]) ? hp.rope_freq_scale_swa : freq_scale;
}

uint32_t GraphContext::n_ctx_seq() const {
    return kv.max_seq_len();   // llama.cpp: cparams.n_ctx_seq (per-sequence context), not the whole cache
}

ggml_tensor * GraphContext::rope_factors(const LayerWeights & L) const {
    // ref: llama_model::get_rope_factors — rope_freqs (llama3) wins; else LongRoPE long/short by context size
    if (L.rope_freqs) return L.rope_freqs;
    if (n_ctx_seq() > (uint32_t) n_ctx_orig) return L.rope_long;
    return L.rope_short;
}

ggml_tensor * GraphContext::build_attn_mha(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * kq_mask, float kq_scale, int il) {
    // q: [head_dim, n_head, n_tokens] -> [head_dim, n_tokens, n_head]
    // k: [head_dim, n_head_kv, n_kv]  -> [head_dim, n_kv, n_head_kv]
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    ggml_tensor * cur;
    if (flash_attn) {
        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hp.f_max_alibi_bias, hp.f_attn_logit_softcapping);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        // res: [head_dim_v, n_head, n_tokens]
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);   // [n_kv, n_tokens, n_head]
        cb(kq, "kq", il);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        if (hp.f_attn_logit_softcapping > 0.0f) {
            kq = ggml_scale(ctx0, kq, 1.0f / hp.f_attn_logit_softcapping);
            kq = ggml_tanh(ctx0, kq);
            kq = ggml_scale(ctx0, kq, hp.f_attn_logit_softcapping);
        }
        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hp.f_max_alibi_bias);
        cb(kq, "kq_soft_max", il);
        // v: [head_dim, n_kv, n_head_kv] -> need [n_kv, head_dim, n_head_kv]
        ggml_tensor * vt = ggml_cont(ctx0, ggml_transpose(ctx0, v));
        ggml_tensor * kqv = ggml_mul_mat(ctx0, vt, kq);   // [head_dim, n_tokens, n_head]
        cb(kqv, "kqv", il);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);        // [head_dim, n_head, n_tokens]
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
    }
    ggml_build_forward_expand(gf, cur);
    return cur;
}

ggml_tensor * GraphContext::build_attn(ggml_tensor * wo, ggml_tensor * bo, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float kq_scale, int il) {
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, k);

    // store K/V into the paged cache: k [head_dim, n_head_kv, n_tokens] -> rows [n_embd_gqa, n_tokens]
    {
        const int64_t n_embd_gqa_k = k->ne[0] * k->ne[1];
        const int64_t n_embd_gqa_v = v->ne[0] * v->ne[1];
        ggml_tensor * k_rows = ggml_view_2d(ctx0, k, n_embd_gqa_k, n_tokens, k->nb[2], 0);
        ggml_tensor * v_rows = ggml_view_2d(ctx0, v, n_embd_gqa_v, n_tokens, v->nb[2], 0);
        ggml_tensor * k_written = ggml_set_rows(ctx0, kv.k_full(il), k_rows, inp_k_idxs);
        ggml_tensor * v_written = ggml_set_rows(ctx0, kv.v_full(il), v_rows, inp_v_idxs);
        ggml_build_forward_expand(gf, k_written);
        ggml_build_forward_expand(gf, v_written);
        if (paged_attn) {
            // iian paged attention: each query attends its own block table only (see paged_attn.h)
            auto prm = std::make_unique<PagedAttnParams>();
            prm->ub = &ub; prm->scale = kq_scale; prm->softcap = hp.f_attn_logit_softcapping; prm->causal = hp.causal_attn;
            const bool layer_swa = hp.n_swa > 0 && (hp.is_swa.empty() || !has_swa_layers() || hp.is_swa[il]);
            prm->n_swa = layer_swa ? hp.n_swa : 0;
            paged_params.push_back(std::move(prm));
            ggml_tensor * qc = ggml_is_contiguous(q) ? q : ggml_cont(ctx0, q);
            ggml_tensor * cur = build_paged_attn(ctx0, qc, k_written, v_written, hp.n_head_kv[il], paged_params.back().get());
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
            cb(cur, "kqv_out", il);
            if (wo) { cur = ggml_mul_mat(ctx0, wo, cur); cb(cur, "attn_wo", il); }
            if (bo) { cur = ggml_add(ctx0, cur, bo); cb(cur, "attn_bo", il); }
            return cur;
        }
    }

    ggml_tensor * kc = kv.k_view(ctx0, il, ub.kv_start, ub.n_kv);
    ggml_tensor * vc = kv.v_view(ctx0, il, ub.kv_start, ub.n_kv);
    ggml_tensor * kq_mask = (inp_kq_mask_swa && hp.is_swa[il]) ? inp_kq_mask_swa : inp_kq_mask;
    ggml_tensor * cur = build_attn_mha(q, kc, vc, kq_mask, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) { cur = ggml_mul_mat(ctx0, wo, cur); cb(cur, "attn_wo", il); }
    if (bo) { cur = ggml_add(ctx0, cur, bo); cb(cur, "attn_bo", il); }
    return cur;
}

ggml_tensor * GraphContext::select_outputs(ggml_tensor * cur) {
    return ggml_get_rows(ctx0, cur, inp_out_ids);
}

ggml_tensor * GraphContext::build_lm_head(ggml_tensor * cur, ggml_tensor * output_norm, ggml_tensor * output_norm_b, ggml_tensor * output, ggml_tensor * output_b, NormKind kind) {
    cur = build_norm(cur, output_norm, output_norm_b, kind, -1);
    cb(cur, "result_norm", -1);
    t_embd = cur;
    cur = ggml_mul_mat(ctx0, output, cur);
    if (output_b) cur = ggml_add(ctx0, cur, output_b);
    if (hp.f_final_logit_softcapping > 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hp.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hp.f_final_logit_softcapping);
    }
    if (hp.f_logit_scale != 0.0f) cur = ggml_scale(ctx0, cur, hp.f_logit_scale);
    cb(cur, "result_output", -1);
    t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    return cur;
}

void GraphContext::set_inputs() {
    if (inp_tokens) ggml_backend_tensor_set(inp_tokens, ub.tokens.data(), 0, ub.n_tokens * sizeof(token_t));
    if (inp_pos)    ggml_backend_tensor_set(inp_pos, ub.pos.data(), 0, ub.n_tokens * sizeof(pos_t));
    if (inp_out_ids) ggml_backend_tensor_set(inp_out_ids, ub.out_ids.data(), 0, ub.n_outputs * sizeof(int32_t));
    if (inp_k_idxs) ggml_backend_tensor_set(inp_k_idxs, ub.slots.data(), 0, ub.n_tokens * sizeof(int64_t));

    // Legacy behaviour is preserved: when hp.n_swa > 0 but no layer is flagged in hp.is_swa (no per-layer
    // pattern known), the window applies to every layer through the main mask. Hybrid models get a full causal
    // main mask plus the window-restricted inp_kq_mask_swa for their SWA layers.
    if (inp_kq_mask)     fill_kq_mask(inp_kq_mask, hp.n_swa > 0 && inp_kq_mask_swa == nullptr);
    if (inp_kq_mask_swa) fill_kq_mask(inp_kq_mask_swa, true);
}

void GraphContext::fill_kq_mask(ggml_tensor * mask, bool swa) const {
    const int64_t n_kv = ub.n_kv;
    const bool f16 = mask->type == GGML_TYPE_F16;
    // build the whole [n_kv, n_tokens] mask on the host and upload it once (per-row uploads are far too
    // slow for GPU buffers)
    std::vector<float> full((size_t) n_kv * ub.n_tokens, -INFINITY);
    for (uint32_t i = 0; i < ub.n_tokens; i++) {
        float * row = full.data() + (size_t) i * n_kv;
        const auto & s = ub.seqs[ub.seq_idx[i]];
        const pos_t p1 = ub.pos[i];
        for (size_t c = 0; c < s.cells.size(); c++) {
            if (!hp.causal_attn || s.cell_pos[c] <= p1) {
                // ref: llama_hparams::is_masked_swa (LLAMA_SWA_TYPE_STANDARD): masked when p1 - p0 >= n_swa
                if (swa && p1 - s.cell_pos[c] >= (pos_t) hp.n_swa) continue;
                const int64_t j = s.cells[c] - ub.kv_start;
                if (j >= 0 && j < n_kv) row[j] = hp.f_max_alibi_bias > 0.0f ? -(float) std::abs(p1 - s.cell_pos[c]) : 0.0f;
            }
        }
    }
    if (f16) {
        std::vector<ggml_fp16_t> full16(full.size());
        ggml_fp32_to_fp16_row(full.data(), full16.data(), (int64_t) full.size());
        ggml_backend_tensor_set(mask, full16.data(), 0, full16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(mask, full.data(), 0, full.size() * sizeof(float));
    }
}

} // namespace iian
