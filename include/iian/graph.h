#pragma once
// GraphContext: helper for building one micro-batch forward graph on ggml.
// Provides the same building blocks as llama.cpp's llm_graph_context (build_norm / build_ffn / build_attn / ...)
// so that architecture code ports almost verbatim, but with paged-KV attention and explicit batch metadata.
#include "iian/hparams.h"
#include "iian/model.h"
#include "iian/types.h"

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace iian {

class PagedKVCache;

// Host-side description of a micro-batch (filled by the engine from SchedulerOutput).
struct UBatch {
    uint32_t n_tokens  = 0;
    uint32_t n_outputs = 0;
    uint32_t n_kv      = 0;     // attention window length (cells [kv_start, kv_start+n_kv))
    uint32_t kv_start  = 0;
    std::vector<token_t>  tokens;      // [n_tokens]
    std::vector<pos_t>    pos;         // [n_tokens]
    std::vector<int32_t>  seq_idx;     // [n_tokens] index into `seqs`
    std::vector<int64_t>  slots;       // [n_tokens] absolute KV cell index for each token
    std::vector<int32_t>  out_ids;     // [n_outputs] token indices whose logits are needed
    // Per-sequence data needed for the mask: for each seq, the cells it owns (absolute indices) with positions.
    struct SeqInfo {
        std::vector<int64_t> cells;     // absolute cell indices of the sequence (computed + this batch's)
        std::vector<pos_t>   cell_pos;  // position of each cell
    };
    std::vector<SeqInfo> seqs;
};

enum class NormKind { RMS, LAYER, GROUP };
enum class FfnOp { SILU, GELU, RELU, RELU_SQR, SWIGLU, GEGLU };
enum class FfnGate { SEQ, PAR };

struct PagedAttnParams;

class GraphContext {
public:
    GraphContext(ggml_context * ctx, ggml_cgraph * gf, const Model & model, const UBatch & ub,
                 PagedKVCache & kv, bool flash_attn, bool paged_attn = false);
    ~GraphContext();   // out of line: PagedAttnParams is only complete inside graph.cpp

    ggml_context * ctx0;
    ggml_cgraph  * gf;
    const Model    & model;
    const HParams  & hp;
    const UBatch   & ub;
    PagedKVCache   & kv;
    const bool     flash_attn;
    const bool     paged_attn;   // use iian's paged-attention kernel instead of the masked window (CPU only)

    // shorthand hparams (like llm_graph_context)
    const int64_t n_embd, n_layer, n_tokens, n_outputs, n_rot, n_ctx_orig;
    const float   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    const int     rope_mode;   // GGML_ROPE_TYPE_*

    // graph inputs (host-filled after allocation)
    ggml_tensor * inp_tokens  = nullptr;   // I32 [n_tokens]
    ggml_tensor * inp_pos     = nullptr;   // I32 [n_tokens]
    ggml_tensor * inp_out_ids = nullptr;   // I32 [n_outputs]
    ggml_tensor * inp_kq_mask = nullptr;   // F16/F32 [n_kv, n_tokens]
    ggml_tensor * inp_kq_mask_swa = nullptr; // same shape, sliding-window restricted; only built when some layer has hp.is_swa[il]
    ggml_tensor * inp_k_idxs  = nullptr;   // I64 [n_tokens]
    ggml_tensor * inp_v_idxs  = nullptr;   // I64 [n_tokens]

    // outputs
    ggml_tensor * t_logits = nullptr;
    ggml_tensor * t_embd   = nullptr;   // final normed hidden states (for embeddings)

    // ---- building blocks ----
    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd);
    ggml_tensor * build_inp_pos();
    ggml_tensor * build_inp_out_ids();
    void          build_attn_inputs();   // creates kq mask + k/v idxs

    ggml_tensor * build_norm(ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b, NormKind kind, int il);
    ggml_tensor * build_ffn(ggml_tensor * cur,
                            ggml_tensor * up,   ggml_tensor * up_b,
                            ggml_tensor * gate, ggml_tensor * gate_b,
                            ggml_tensor * down, ggml_tensor * down_b,
                            FfnOp op, FfnGate gate_type, int il);
    ggml_tensor * build_moe_ffn(ggml_tensor * cur, ggml_tensor * gate_inp, ggml_tensor * up_exps, ggml_tensor * gate_exps,
                                ggml_tensor * down_exps, ggml_tensor * exp_probs_b, int64_t n_expert, int64_t n_expert_used,
                                FfnOp op, bool norm_w, bool scale_w, float w_scale, int il);
    // Q/K/V projections (handles fused wqkv or separate; adds biases). Output shapes [head_dim, n_head, n_tokens].
    struct QKV { ggml_tensor * q, * k, * v; };
    QKV build_qkv(const LayerWeights & L, ggml_tensor * cur, int64_t n_embd_head, int64_t n_head, int64_t n_head_kv, int il);
    ggml_tensor * build_rope(ggml_tensor * cur, ggml_tensor * rope_factors);
    // Same with explicit per-layer rope base/scale (hybrid SWA/global models use a different base for SWA layers).
    ggml_tensor * build_rope(ggml_tensor * cur, ggml_tensor * rope_factors, float freq_base_l, float freq_scale_l);
    // Per-layer rope base/scale: SWA layers of hybrid models use hp.rope_freq_base_swa / rope_freq_scale_swa.
    float rope_freq_base_l(int il) const;
    float rope_freq_scale_l(int il) const;
    // Rope frequency factors for a layer, mirroring llama.cpp's llama_model::get_rope_factors():
    // rope_freqs if present (llama3), else LongRoPE long/short factors chosen by n_ctx_seq() vs n_ctx_orig.
    ggml_tensor * rope_factors(const LayerWeights & L) const;
    // Stand-in for llama.cpp's cparams.n_ctx_seq: the engine does not pass a per-sequence context to the graph,
    // so the total KV capacity (an upper bound on any sequence length) is used.
    uint32_t n_ctx_seq() const;
    // Writes k/v into the paged cache and computes attention over the batch window; applies wo/bo.
    ggml_tensor * build_attn(ggml_tensor * wo, ggml_tensor * bo, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                             float kq_scale, int il);
    // Selects only output rows at the last layer (call before the final residual add of the last layer).
    ggml_tensor * select_outputs(ggml_tensor * cur);
    ggml_tensor * build_lm_head(ggml_tensor * cur, ggml_tensor * output_norm, ggml_tensor * output_norm_b, ggml_tensor * output, ggml_tensor * output_b, NormKind kind);

    void cb(ggml_tensor * t, const char * name, int il);

    // per-layer parameter blocks for the paged-attention custom op (stable addresses across graph reuse)
    std::vector<std::unique_ptr<PagedAttnParams>> paged_params;

    // Host-side: fill all input tensors from the UBatch (after graph allocation).
    void set_inputs();

private:
    ggml_tensor * build_attn_mha(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * kq_mask, float kq_scale, int il);
    bool has_swa_layers() const;
    void fill_kq_mask(ggml_tensor * mask, bool swa) const;
};

} // namespace iian
