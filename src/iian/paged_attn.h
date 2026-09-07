#pragma once
// iian paged-attention kernel (CPU): attention over a request's own KV blocks only.
//
// The masked/flash path attends every query over the batch-wide cell window [kv_start, kv_start+n_kv)
// and masks other sequences out: O(n_tokens x window). This kernel walks each query's own cell list
// (its block table), so the cost is O(sum of context lengths) — vLLM's paged attention, on ggml.
// Implemented as a ggml custom op (runs on the CPU backend; F16/F32 K/V; GQA; sliding window; softcap).
#include "iian/graph.h"

#include "ggml.h"

#include <cstdint>

namespace iian {

struct PagedAttnParams {
    const UBatch * ub = nullptr;   // per-token sequence index / position and per-sequence cells (stable across graph reuse)
    float    scale = 1.0f;
    float    softcap = 0.0f;       // 0 = off
    bool     causal = true;
    uint32_t n_swa = 0;            // 0 = full attention for this layer
};

// Returns true if the kernel supports these cache types / hparams (F16 or F32 K/V, no ALiBi).
bool paged_attn_supported(ggml_type type_k, ggml_type type_v, float max_alibi_bias);

// q: [D, n_head, n_tokens] F32 (after RoPE). k_cache/v_cache: [n_embd_gqa, n_cells] (pass the set_rows
// results so the op is ordered after the KV writes). Result: [Dv, n_head, n_tokens] F32, contiguous.
ggml_tensor * build_paged_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k_cache, ggml_tensor * v_cache,
                               int64_t n_head_kv, const PagedAttnParams * params);

} // namespace iian
