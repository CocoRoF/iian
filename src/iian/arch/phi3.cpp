// Architecture: phi3 (Phi-3 / Phi-3.5 mini). Fused QKV (attn_qkv), fused gate/up in ffn_up ([gate; up] halves,
// consumed by ggml_swiglu), LongRoPE factor tensors (rope_factors_long/short), NEOX rope, query pre-scaled.
// Ported from ref/llama.cpp/src/models/phi3.cpp. SWA is disabled like in llama.cpp (see PR #13676).
#include "iian/arch.h"
#include "iian/graph.h"

#include "ggml.h"

#include <cmath>

namespace iian {

class ArchPhi3 : public ArchDef {
public:
    const char * name() const override { return "phi3"; }

    void load_hparams(const GgufFile & f, HParams & hp) const override {
        hp.f_norm_rms_eps = f.get_f32("%s.attention.layer_norm_rms_epsilon").value_or(1e-5f);
        // llama.cpp: "Phi SWA is currently disabled" — the conversion scripts do not populate n_swa correctly
        hp.n_swa = 0;
        hp.is_swa.assign(hp.n_layer, 0);
    }

    void create_tensors(Model & m, TensorCreator & tc) const override {
        const HParams & hp = m.hparams();
        const int64_t n_embd = hp.n_embd, n_vocab = hp.n_vocab, n_layer = hp.n_layer, n_rot = hp.n_rot;

        m.tok_embd      = tc.global_tensor("token_embd", "weight", {n_embd, n_vocab});
        m.output_norm   = tc.global_tensor("output_norm", "weight", {n_embd});
        m.output_norm_b = tc.global_tensor("output_norm", "bias", {n_embd}, TensorCreator::NOT_REQUIRED);
        m.output        = tc.global_tensor("output", "weight", {n_embd, n_vocab}, TensorCreator::NOT_REQUIRED);
        m.output_b      = tc.global_tensor("output", "bias", {n_vocab}, TensorCreator::NOT_REQUIRED);
        if (!m.output) m.output = tc.global_tensor("token_embd", "weight", {n_embd, n_vocab}, TensorCreator::DUPLICATED);

        for (int il = 0; il < n_layer; il++) {
            auto & L = m.layers[il];
            const int64_t n_head = hp.n_head[il], n_head_kv = hp.n_head_kv[il], n_ff = hp.n_ff[il];
            const int64_t n_embd_head = hp.n_embd_head_k;
            const int64_t n_embd_gqa = n_embd_head * n_head_kv;

            L.attn_norm   = tc.layer_tensor("attn_norm", il, "weight", {n_embd});
            L.attn_norm_b = tc.layer_tensor("attn_norm", il, "bias", {n_embd}, TensorCreator::NOT_REQUIRED);
            L.wqkv = tc.layer_tensor("attn_qkv", il, "weight", {n_embd, n_embd_head * n_head + 2 * n_embd_gqa}, TensorCreator::NOT_REQUIRED);
            if (L.wqkv) {
                L.bqkv = tc.layer_tensor("attn_qkv", il, "bias", {n_embd_head * n_head + 2 * n_embd_gqa}, TensorCreator::NOT_REQUIRED);
            } else {
                L.wq = tc.layer_tensor("attn_q", il, "weight", {n_embd, n_embd_head * n_head});
                L.wk = tc.layer_tensor("attn_k", il, "weight", {n_embd, n_embd_gqa});
                L.wv = tc.layer_tensor("attn_v", il, "weight", {n_embd, n_embd_gqa});
                L.bq = tc.layer_tensor("attn_q", il, "bias", {n_embd_head * n_head}, TensorCreator::NOT_REQUIRED);
                L.bk = tc.layer_tensor("attn_k", il, "bias", {n_embd_gqa}, TensorCreator::NOT_REQUIRED);
                L.bv = tc.layer_tensor("attn_v", il, "bias", {n_embd_gqa}, TensorCreator::NOT_REQUIRED);
            }
            L.wo = tc.layer_tensor("attn_output", il, "weight", {n_embd_head * n_head, n_embd});
            L.bo = tc.layer_tensor("attn_output", il, "bias", {n_embd}, TensorCreator::NOT_REQUIRED);

            L.ffn_norm   = tc.layer_tensor("ffn_norm", il, "weight", {n_embd});
            L.ffn_norm_b = tc.layer_tensor("ffn_norm", il, "bias", {n_embd}, TensorCreator::NOT_REQUIRED);
            L.ffn_down = tc.layer_tensor("ffn_down", il, "weight", {n_ff, n_embd});
            L.ffn_up   = tc.layer_tensor("ffn_up",   il, "weight", {n_embd, 2 * n_ff});   // [gate; up]

            // LongRoPE factors are global tensors ("rope_factors_long.weight"); shared by all layers
            if (il == 0) {
                L.rope_long  = tc.global_tensor("rope_factors_long",  "weight", {n_rot / 2}, TensorCreator::NOT_REQUIRED);
                L.rope_short = tc.global_tensor("rope_factors_short", "weight", {n_rot / 2}, TensorCreator::NOT_REQUIRED);
            } else {
                L.rope_long  = m.layers[0].rope_long;
                L.rope_short = m.layers[0].rope_short;
            }
        }
    }

    void build_graph(GraphContext & g) const override {
        const HParams & hp = g.hp;
        const Model & m = g.model;
        const int64_t n_embd_head = hp.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hp.n_embd_head_k);

        ggml_tensor * inpL = g.build_inp_embd(m.tok_embd);
        g.build_inp_pos();
        g.build_inp_out_ids();
        g.build_attn_inputs();

        for (int il = 0; il < g.n_layer; il++) {
            const auto & L = m.layers[il];
            ggml_tensor * residual = inpL;

            // rope freq factors for 128k context (long/short chosen by context size, see GraphContext::rope_factors)
            ggml_tensor * rope_factors = g.rope_factors(L);

            ggml_tensor * cur = g.build_norm(inpL, L.attn_norm, L.attn_norm_b, NormKind::RMS, il);
            g.cb(cur, "attn_norm", il);

            auto [q, k, v] = g.build_qkv(L, cur, n_embd_head, hp.n_head[il], hp.n_head_kv[il], il);
            q = g.build_rope(q, rope_factors);
            k = g.build_rope(k, rope_factors);
            g.cb(q, "Qcur", il); g.cb(k, "Kcur", il); g.cb(v, "Vcur", il);
            q = ggml_scale(g.ctx0, q, 1.0f / sqrtf((float) n_embd_head));
            cur = g.build_attn(L.wo, L.bo, q, k, v, 1.0f, il);

            if (il == g.n_layer - 1) { cur = g.select_outputs(cur); residual = g.select_outputs(residual); }
            cur = ggml_add(g.ctx0, cur, residual);
            residual = cur;

            cur = g.build_norm(cur, L.ffn_norm, L.ffn_norm_b, NormKind::RMS, il);
            g.cb(cur, "ffn_norm", il);
            if (!L.ffn_gate_inp) {
                // ffn_up holds [gate; up]; SWIGLU op splits it (ggml_swiglu) — no separate gate matrix
                cur = g.build_ffn(cur, L.ffn_up, nullptr, nullptr, nullptr, L.ffn_down, nullptr, FfnOp::SWIGLU, FfnGate::SEQ, il);
                g.cb(cur, "ffn_out", il);
            } else {
                cur = g.build_moe_ffn(cur, L.ffn_gate_inp, L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps, nullptr,
                                      hp.n_expert, hp.n_expert_used, FfnOp::SILU, /*norm_w*/ true, /*scale_w*/ false, 0.0f, il);
                g.cb(cur, "ffn_moe_out", il);
            }
            cur = ggml_add(g.ctx0, residual, cur);
            g.cb(cur, "l_out", il);
            inpL = cur;
        }
        g.build_lm_head(inpL, m.output_norm, m.output_norm_b, m.output, m.output_b, NormKind::RMS);
    }

    RopeType rope_type(const HParams &) const override { return RopeType::NEOX; }
};

IIAN_REGISTER_ARCH("phi3", ArchPhi3);

} // namespace iian
