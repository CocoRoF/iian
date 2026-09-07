// Architecture: qwen3 (Qwen3 dense). Like qwen2 but: no QKV biases, per-head RMS norm on Q and K before RoPE (NEOX).
// Ported from ref/llama.cpp/src/models/qwen3.cpp.
#include "iian/arch.h"
#include "iian/graph.h"

#include "ggml.h"

#include <cmath>

namespace iian {

class ArchQwen3 : public ArchDef {
public:
    const char * name() const override { return "qwen3"; }

    void load_hparams(const GgufFile & f, HParams & hp) const override {
        hp.f_norm_rms_eps = f.get_f32("%s.attention.layer_norm_rms_epsilon").value_or(1e-6f);
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
            const int64_t n_embd_head = hp.n_embd_head_k;
            const int64_t n_embd_gqa = n_embd_head * n_head_kv;

            L.attn_norm = tc.layer_tensor("attn_norm", il, "weight", {n_embd});
            // create_tensor_qkv: fused attn_qkv or separate q/k/v; biases optional
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

            L.attn_k_norm = tc.layer_tensor("attn_k_norm", il, "weight", {n_embd_head});
            L.attn_q_norm = tc.layer_tensor("attn_q_norm", il, "weight", {n_embd_head});

            L.ffn_norm = tc.layer_tensor("ffn_norm", il, "weight", {n_embd});
            L.ffn_gate = tc.layer_tensor("ffn_gate", il, "weight", {n_embd, n_ff});
            L.ffn_down = tc.layer_tensor("ffn_down", il, "weight", {n_ff, n_embd});
            L.ffn_up   = tc.layer_tensor("ffn_up",   il, "weight", {n_embd, n_ff});
        }
    }

    void build_graph(GraphContext & g) const override {
        const HParams & hp = g.hp;
        const Model & m = g.model;
        const int64_t n_embd_head = hp.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hp.n_embd_head_k);
        GGML_ASSERT(n_embd_head == g.n_rot);

        ggml_tensor * inpL = g.build_inp_embd(m.tok_embd);
        g.build_inp_pos();
        g.build_inp_out_ids();
        g.build_attn_inputs();
        const float kq_scale = 1.0f / sqrtf((float) n_embd_head);

        for (int il = 0; il < g.n_layer; il++) {
            const auto & L = m.layers[il];
            ggml_tensor * inpSA = inpL;
            ggml_tensor * cur = g.build_norm(inpL, L.attn_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "attn_norm", il);

            auto [q, k, v] = g.build_qkv(L, cur, n_embd_head, hp.n_head[il], hp.n_head_kv[il], il);
            // per-head RMS norm (normalizes along head_dim), then RoPE
            q = g.build_norm(q, L.attn_q_norm, nullptr, NormKind::RMS, il);
            g.cb(q, "Qcur_normed", il);
            q = g.build_rope(q, nullptr);
            k = g.build_norm(k, L.attn_k_norm, nullptr, NormKind::RMS, il);
            g.cb(k, "Kcur_normed", il);
            k = g.build_rope(k, nullptr);
            g.cb(q, "Qcur", il); g.cb(k, "Kcur", il); g.cb(v, "Vcur", il);
            cur = g.build_attn(L.wo, L.bo, q, k, v, kq_scale, il);

            if (il == g.n_layer - 1) { cur = g.select_outputs(cur); inpSA = g.select_outputs(inpSA); }
            ggml_tensor * ffn_inp = ggml_add(g.ctx0, cur, inpSA);
            g.cb(ffn_inp, "ffn_inp", il);

            cur = g.build_norm(ffn_inp, L.ffn_norm, nullptr, NormKind::RMS, il);
            g.cb(cur, "ffn_norm", il);
            cur = g.build_ffn(cur, L.ffn_up, nullptr, L.ffn_gate, nullptr, L.ffn_down, nullptr, FfnOp::SILU, FfnGate::PAR, il);
            g.cb(cur, "ffn_out", il);
            cur = ggml_add(g.ctx0, cur, ffn_inp);
            g.cb(cur, "l_out", il);
            inpL = cur;
        }
        g.build_lm_head(inpL, m.output_norm, nullptr, m.output, nullptr, NormKind::RMS);
    }

    RopeType rope_type(const HParams &) const override { return RopeType::NEOX; }
};

IIAN_REGISTER_ARCH("qwen3", ArchQwen3);

} // namespace iian
