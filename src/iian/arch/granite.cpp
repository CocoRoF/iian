// Architecture: granite (IBM Granite 3.x/4.0 dense; also the MoE variant of the same arch id). Llama-like with
// embedding / residual / attention / logit scaling, optional NoPE (rope.scaling.finetuned == false), NORMAL rope.
// Ported from ref/llama.cpp/src/models/granite.cpp (deepstack vision inputs not supported).
#include "iian/arch.h"
#include "iian/graph.h"

#include "ggml.h"

#include <cmath>
#include <stdexcept>

namespace iian {

class ArchGranite : public ArchDef {
public:
    const char * name() const override { return "granite"; }

    void load_hparams(const GgufFile & f, HParams & hp) const override {
        hp.f_norm_rms_eps = f.get_f32("%s.attention.layer_norm_rms_epsilon").value_or(1e-5f);
        auto logit_scale = f.get_f32("%s.logit_scale");
        if (!logit_scale || *logit_scale == 0.0f) throw std::runtime_error("granite: missing required key " + f.expand("%s.logit_scale"));
        hp.f_logit_scale = *logit_scale;
        hp.f_attention_scale = f.get_f32("%s.attention.scale").value_or(0.0f);
        hp.extra["residual_scale"]  = f.get_f32("%s.residual_scale").value_or(0.0f);
        hp.extra["embedding_scale"] = f.get_f32("%s.embedding_scale").value_or(0.0f);
        // Granite uses rope_finetuned as a switch for rope (false -> NoPE), default true
        hp.rope_finetuned = f.get_bool("%s.rope.scaling.finetuned").value_or(true);
        hp.extra["has_rope"] = hp.rope_finetuned ? 1.0f : 0.0f;
        hp.extra["n_ff_shexp"] = (float) f.get_u32("%s.expert_shared_feed_forward_length").value_or(0);
    }

    void create_tensors(Model & m, TensorCreator & tc) const override {
        const HParams & hp = m.hparams();
        const int64_t n_embd = hp.n_embd, n_vocab = hp.n_vocab, n_layer = hp.n_layer;
        const int64_t n_rot = hp.n_rot, n_expert = hp.n_expert;
        const int64_t n_ff_shexp = (int64_t) hp.extra.at("n_ff_shexp");

        m.tok_embd    = tc.global_tensor("token_embd", "weight", {n_embd, n_vocab});
        m.output_norm = tc.global_tensor("output_norm", "weight", {n_embd});
        m.output      = tc.global_tensor("output", "weight", {n_embd, n_vocab}, TensorCreator::NOT_REQUIRED);
        if (!m.output) m.output = tc.global_tensor("token_embd", "weight", {n_embd, n_vocab}, TensorCreator::DUPLICATED);

        for (int il = 0; il < n_layer; il++) {
            auto & L = m.layers[il];
            const int64_t n_head = hp.n_head[il], n_head_kv = hp.n_head_kv[il], n_ff = hp.n_ff[il];
            const int64_t n_embd_head_k = hp.n_embd_head_k, n_embd_head_v = hp.n_embd_head_v;
            const int64_t n_embd_k_gqa = n_embd_head_k * n_head_kv, n_embd_v_gqa = n_embd_head_v * n_head_kv;

            L.attn_norm = tc.layer_tensor("attn_norm", il, "weight", {n_embd});
            L.wqkv = tc.layer_tensor("attn_qkv", il, "weight", {n_embd, n_embd_head_k * n_head + n_embd_k_gqa + n_embd_v_gqa}, TensorCreator::NOT_REQUIRED);
            if (L.wqkv) {
                L.bqkv = tc.layer_tensor("attn_qkv", il, "bias", {n_embd_head_k * n_head + n_embd_k_gqa + n_embd_v_gqa}, TensorCreator::NOT_REQUIRED);
            } else {
                L.wq = tc.layer_tensor("attn_q", il, "weight", {n_embd, n_embd_head_k * n_head});
                L.wk = tc.layer_tensor("attn_k", il, "weight", {n_embd, n_embd_k_gqa});
                L.wv = tc.layer_tensor("attn_v", il, "weight", {n_embd, n_embd_v_gqa});
                L.bq = tc.layer_tensor("attn_q", il, "bias", {n_embd_head_k * n_head}, TensorCreator::NOT_REQUIRED);
                L.bk = tc.layer_tensor("attn_k", il, "bias", {n_embd_k_gqa}, TensorCreator::NOT_REQUIRED);
                L.bv = tc.layer_tensor("attn_v", il, "bias", {n_embd_v_gqa}, TensorCreator::NOT_REQUIRED);
            }
            L.wo = tc.layer_tensor("attn_output", il, "weight", {n_embd_head_k * n_head, n_embd});
            L.bo = tc.layer_tensor("attn_output", il, "bias", {n_embd}, TensorCreator::NOT_REQUIRED);

            L.ffn_norm = tc.layer_tensor("ffn_norm", il, "weight", {n_embd});

            // rope factors: LongRoPE long/short (global tensors) or llama3-style rope_freqs (global tensor)
            if (il == 0) {
                if (hp.rope_scaling == RopeScaling::LONGROPE) {
                    L.rope_long  = tc.global_tensor("rope_factors_long",  "weight", {n_rot / 2}, TensorCreator::NOT_REQUIRED);
                    L.rope_short = tc.global_tensor("rope_factors_short", "weight", {n_rot / 2}, TensorCreator::NOT_REQUIRED);
                } else {
                    L.rope_freqs = tc.global_tensor("rope_freqs", "weight", {n_rot / 2}, TensorCreator::NOT_REQUIRED);
                }
            } else {
                L.rope_long = m.layers[0].rope_long; L.rope_short = m.layers[0].rope_short; L.rope_freqs = m.layers[0].rope_freqs;
            }

            if (n_expert == 0) {
                L.ffn_gate = tc.layer_tensor("ffn_gate", il, "weight", {n_embd, n_ff});
                L.ffn_down = tc.layer_tensor("ffn_down", il, "weight", {n_ff, n_embd});
                L.ffn_up   = tc.layer_tensor("ffn_up",   il, "weight", {n_embd, n_ff});
                L.ffn_gate_b = tc.layer_tensor("ffn_gate", il, "bias", {n_ff}, TensorCreator::NOT_REQUIRED);
                L.ffn_down_b = tc.layer_tensor("ffn_down", il, "bias", {n_embd}, TensorCreator::NOT_REQUIRED);
                L.ffn_up_b   = tc.layer_tensor("ffn_up",   il, "bias", {n_ff}, TensorCreator::NOT_REQUIRED);
            } else {
                L.ffn_gate_inp  = tc.layer_tensor("ffn_gate_inp", il, "weight", {n_embd, n_expert});
                L.ffn_gate_exps = tc.layer_tensor("ffn_gate_exps", il, "weight", {n_embd, n_ff, n_expert}, TensorCreator::NOT_REQUIRED);
                L.ffn_down_exps = tc.layer_tensor("ffn_down_exps", il, "weight", {n_ff, n_embd, n_expert});
                L.ffn_up_exps   = tc.layer_tensor("ffn_up_exps",   il, "weight", {n_embd, n_ff, n_expert});
                if (n_ff_shexp > 0) {
                    L.ffn_gate_shexp = tc.layer_tensor("ffn_gate_shexp", il, "weight", {n_embd, n_ff_shexp});
                    L.ffn_up_shexp   = tc.layer_tensor("ffn_up_shexp",   il, "weight", {n_embd, n_ff_shexp});
                    L.ffn_down_shexp = tc.layer_tensor("ffn_down_shexp", il, "weight", {n_ff_shexp, n_embd});
                }
            }
        }
    }

    void build_graph(GraphContext & g) const override {
        const HParams & hp = g.hp;
        const Model & m = g.model;
        const int64_t n_embd_head = hp.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hp.n_embd_head_k);
        GGML_ASSERT(n_embd_head == g.n_rot);
        const float f_residual_scale  = hp.extra.at("residual_scale");
        const float f_embedding_scale = hp.extra.at("embedding_scale");
        const bool  has_rope = hp.extra.at("has_rope") != 0.0f;
        const bool  has_shexp = hp.extra.at("n_ff_shexp") > 0.0f;

        ggml_tensor * inpL = g.build_inp_embd(m.tok_embd);
        if (f_embedding_scale != 0.0f) { inpL = ggml_scale(g.ctx0, inpL, f_embedding_scale); g.cb(inpL, "inp_scaled", -1); }
        if (has_rope) g.build_inp_pos();
        g.build_inp_out_ids();
        g.build_attn_inputs();
        const float kq_scale = hp.f_attention_scale == 0.0f ? 1.0f / sqrtf((float) n_embd_head) : hp.f_attention_scale;

        for (int il = 0; il < g.n_layer; il++) {
            const auto & L = m.layers[il];
            ggml_tensor * inpSA = inpL;
            ggml_tensor * cur = g.build_norm(inpL, L.attn_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "attn_norm", il);

            // self-attention
            auto [q, k, v] = g.build_qkv(L, cur, n_embd_head, hp.n_head[il], hp.n_head_kv[il], il);
            if (has_rope) {
                ggml_tensor * rope_factors = g.rope_factors(L);
                q = g.build_rope(q, rope_factors);
                k = g.build_rope(k, rope_factors);
            }
            g.cb(q, "Qcur", il); g.cb(k, "Kcur", il); g.cb(v, "Vcur", il);
            cur = g.build_attn(L.wo, L.bo, q, k, v, kq_scale, il);
            g.cb(cur, "attn_out", il);

            if (il == g.n_layer - 1) { cur = g.select_outputs(cur); inpSA = g.select_outputs(inpSA); }

            // ffn (with residual scaling)
            if (f_residual_scale != 0.0f) cur = ggml_scale(g.ctx0, cur, f_residual_scale);
            ggml_tensor * ffn_inp = ggml_add(g.ctx0, cur, inpSA);
            g.cb(ffn_inp, "ffn_inp", il);

            cur = g.build_norm(ffn_inp, L.ffn_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "ffn_norm", il);
            if (!L.ffn_gate_inp) {
                cur = g.build_ffn(cur, L.ffn_up, L.ffn_up_b, L.ffn_gate, L.ffn_gate_b, L.ffn_down, L.ffn_down_b, FfnOp::SILU, FfnGate::PAR, il);
                g.cb(cur, "ffn_out", il);
            } else {
                ggml_tensor * moe_out = g.build_moe_ffn(cur, L.ffn_gate_inp, L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps, nullptr,
                                                        hp.n_expert, hp.n_expert_used, FfnOp::SILU, /*norm_w*/ true, /*scale_w*/ false, 0.0f, il);
                g.cb(moe_out, "ffn_moe_out", il);
                if (has_shexp) {
                    ggml_tensor * ffn_shexp = g.build_ffn(cur, L.ffn_up_shexp, nullptr, L.ffn_gate_shexp, nullptr, L.ffn_down_shexp, nullptr, FfnOp::SILU, FfnGate::PAR, il);
                    g.cb(ffn_shexp, "ffn_shexp", il);
                    cur = ggml_add(g.ctx0, moe_out, ffn_shexp);
                } else {
                    cur = moe_out;
                }
            }
            if (f_residual_scale != 0.0f) cur = ggml_scale(g.ctx0, cur, f_residual_scale);
            cur = ggml_add(g.ctx0, cur, ffn_inp);
            g.cb(cur, "l_out", il);
            inpL = cur;
        }

        // lm head: granite divides logits by f_logit_scale (build_lm_head would multiply), so build it here
        ggml_tensor * cur = g.build_norm(inpL, m.output_norm, nullptr, NormKind::RMS, -1);
        g.cb(cur, "result_norm", -1);
        g.t_embd = cur;
        cur = ggml_mul_mat(g.ctx0, m.output, cur);
        cur = ggml_scale(g.ctx0, cur, 1.0f / hp.f_logit_scale);
        g.cb(cur, "result_output", -1);
        g.t_logits = cur;
        ggml_build_forward_expand(g.gf, cur);
    }

    RopeType rope_type(const HParams &) const override { return RopeType::NORMAL; }
};

IIAN_REGISTER_ARCH("granite", ArchGranite);

} // namespace iian
