// Architecture: gemma3 (Gemma 3 text). Hybrid sliding-window/global attention (pattern: every Nth layer global,
// default N=6), separate rope base for SWA layers, per-head Q/K RMS norm, pre+post norms around attention and FFN,
// embedding scaled by sqrt(n_embd), GELU(tanh) gated FFN, optional final logit softcap, NEOX rope, head_dim 256.
// Ported from ref/llama.cpp/src/models/gemma3.cpp.
#include "iian/arch.h"
#include "iian/graph.h"

#include "ggml.h"

#include <cmath>

namespace iian {

class ArchGemma3 : public ArchDef {
public:
    const char * name() const override { return "gemma3"; }

    void load_hparams(const GgufFile & f, HParams & hp) const override {
        // n_swa is read by the generic loader from %s.attention.sliding_window
        hp.is_swa.assign(hp.n_layer, 0);
        if (hp.n_swa > 0) {
            // scalar: period (layer il is SWA unless il % period == period - 1); array: explicit per-layer flags
            std::vector<uint32_t> pattern = f.get_arr_u32("%s.attention.sliding_window_pattern");
            if (!pattern.empty()) {
                for (uint32_t il = 0; il < hp.n_layer; il++) hp.is_swa[il] = pattern[il % pattern.size()] != 0;
            } else {
                const uint32_t period = f.get_u32("%s.attention.sliding_window_pattern").value_or(6);
                for (uint32_t il = 0; il < hp.n_layer; il++) hp.is_swa[il] = period == 0 || (il % period < period - 1);
            }
            hp.rope_freq_base_swa  = f.get_f32("%s.rope.freq_base_swa").value_or(10000.0f);
            hp.rope_freq_scale_swa = 1.0f;
        }
        hp.f_final_logit_softcapping = f.get_f32("%s.final_logit_softcapping").value_or(0.0f);
        hp.f_norm_rms_eps = f.get_f32("%s.attention.layer_norm_rms_epsilon").value_or(1e-6f);

        // ref: gemma_pytorch config.py — the 27B model (62 layers) uses query_pre_attn_scalar = n_embd / n_head
        const bool is_27b = hp.n_layer == 62;
        hp.f_attention_scale = is_27b ? 1.0f / sqrtf((float) (hp.n_embd / hp.n_head[0]))
                                      : 1.0f / sqrtf((float) hp.n_embd_head_k);
    }

    void create_tensors(Model & m, TensorCreator & tc) const override {
        const HParams & hp = m.hparams();
        const int64_t n_embd = hp.n_embd, n_vocab = hp.n_vocab, n_layer = hp.n_layer;

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
            if (!L.wqkv) {
                L.wq = tc.layer_tensor("attn_q", il, "weight", {n_embd, n_embd_head_k * n_head});
                L.wk = tc.layer_tensor("attn_k", il, "weight", {n_embd, n_embd_k_gqa});
                L.wv = tc.layer_tensor("attn_v", il, "weight", {n_embd, n_embd_v_gqa});
            }
            L.wo = tc.layer_tensor("attn_output", il, "weight", {n_embd_head_k * n_head, n_embd});

            L.attn_post_norm = tc.layer_tensor("post_attention_norm", il, "weight", {n_embd});
            L.attn_k_norm    = tc.layer_tensor("attn_k_norm", il, "weight", {n_embd_head_k});
            L.attn_q_norm    = tc.layer_tensor("attn_q_norm", il, "weight", {n_embd_head_k});

            L.ffn_norm      = tc.layer_tensor("ffn_norm", il, "weight", {n_embd});
            L.ffn_gate      = tc.layer_tensor("ffn_gate", il, "weight", {n_embd, n_ff});
            L.ffn_up        = tc.layer_tensor("ffn_up",   il, "weight", {n_embd, n_ff});
            L.ffn_down      = tc.layer_tensor("ffn_down", il, "weight", {n_ff, n_embd});
            L.ffn_post_norm = tc.layer_tensor("post_ffw_norm", il, "weight", {n_embd});
        }
    }

    void build_graph(GraphContext & g) const override {
        const HParams & hp = g.hp;
        const Model & m = g.model;
        const int64_t n_embd_head = hp.n_embd_head_k;

        ggml_tensor * inpL = g.build_inp_embd(m.tok_embd);
        inpL = ggml_scale(g.ctx0, inpL, sqrtf((float) g.n_embd));
        g.cb(inpL, "inp_scaled", -1);

        g.build_inp_pos();
        g.build_inp_out_ids();
        g.build_attn_inputs();   // builds inp_kq_mask_swa as well when hp.is_swa has SWA layers

        for (int il = 0; il < g.n_layer; il++) {
            const auto & L = m.layers[il];
            const float freq_base_l  = g.rope_freq_base_l(il);
            const float freq_scale_l = g.rope_freq_scale_l(il);

            ggml_tensor * cur = g.build_norm(inpL, L.attn_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "attn_norm", il);

            auto [q, k, v] = g.build_qkv(L, cur, n_embd_head, hp.n_head[il], hp.n_head_kv[il], il);
            q = g.build_norm(q, L.attn_q_norm, nullptr, NormKind::RMS, il);
            g.cb(q, "Qcur_normed", il);
            q = g.build_rope(q, nullptr, freq_base_l, freq_scale_l);
            k = g.build_norm(k, L.attn_k_norm, nullptr, NormKind::RMS, il);
            g.cb(k, "Kcur_normed", il);
            k = g.build_rope(k, nullptr, freq_base_l, freq_scale_l);
            g.cb(q, "Qcur", il); g.cb(k, "Kcur", il); g.cb(v, "Vcur", il);

            // ref: gemma_pytorch model.py — query scaled before attention, kq_scale = 1
            q = ggml_scale(g.ctx0, q, hp.f_attention_scale);
            cur = g.build_attn(L.wo, nullptr, q, k, v, 1.0f, il);

            if (il == g.n_layer - 1) { cur = g.select_outputs(cur); inpL = g.select_outputs(inpL); }

            cur = g.build_norm(cur, L.attn_post_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "attn_post_norm", il);
            ggml_tensor * sa_out = ggml_add(g.ctx0, cur, inpL);
            g.cb(sa_out, "sa_out", il);

            cur = g.build_norm(sa_out, L.ffn_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "ffn_norm", il);
            cur = g.build_ffn(cur, L.ffn_up, nullptr, L.ffn_gate, nullptr, L.ffn_down, nullptr, FfnOp::GELU, FfnGate::PAR, il);
            g.cb(cur, "ffn_out", il);
            cur = g.build_norm(cur, L.ffn_post_norm, nullptr, NormKind::RMS, -1);
            g.cb(cur, "ffn_post_norm", il);

            cur = ggml_add(g.ctx0, cur, sa_out);
            g.cb(cur, "l_out", il);
            inpL = cur;
        }
        // build_lm_head applies hp.f_final_logit_softcapping when non-zero
        g.build_lm_head(inpL, m.output_norm, nullptr, m.output, nullptr, NormKind::RMS);
    }

    RopeType rope_type(const HParams &) const override { return RopeType::NEOX; }
};

IIAN_REGISTER_ARCH("gemma3", ArchGemma3);

} // namespace iian
