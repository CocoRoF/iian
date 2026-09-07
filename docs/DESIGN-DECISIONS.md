# Design decisions (ADR log)

Each entry: context, decision, consequences. Newest last.

## ADR-001: ggml as the compute substrate

**Context.** We need GGUF + every quantization format + CPU/CUDA/Metal/Vulkan/SYCL/... today.
Rewriting kernels is a multi-year effort; vLLM's kernels are torch-bound and not portable.

**Decision.** Vendor ggml verbatim (`third_party/ggml`, MIT) and build only our own layers above it.
Backend options pass straight through CMake (`-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, ...).

**Consequences.** We inherit ggml's op set and its performance characteristics. Upstream updates are a
directory sync. Custom kernels (paged attention) must be written as ggml ops per backend.

## ADR-002: our own model layer instead of wrapping libllama

**Context.** `llama_decode` already batches sequences, but its cache is cell-scanned, its scheduling is
per-slot and it owns the sampling/output path. Building vLLM-style scheduling on top would mean fighting
the abstraction (block tables, preemption, prefix hashing all need to see the cells).

**Decision.** Re-implement model loading, graph building, KV cache, scheduler, sampler and engine.
Port llama.cpp's *helpers* (graph building blocks) so that architecture code stays a near-verbatim port,
and copy the pieces that are orthogonal to inference performance (tokenizer, Jinja).

**Consequences.** We must add architectures ourselves (one file each; see `docs/models.md`).
We can guarantee correctness against llama.cpp by token-exact greedy comparison, which we do in CI.

## ADR-003: mask-based fused attention first, paged-attention kernel second

**Context.** ggml has no paged-attention op. Its `flash_attn_ext` takes strided K/V views and a mask.

**Decision.** Store K/V in a flat cell pool, write with `set_rows` + slot indices, attend over a padded
window with a per-token mask. Keep the allocator address-compact so the window stays small. Isolate
attention behind `GraphContext::build_attn()`.

**Consequences.** Correct on every backend today; per-step attention cost is `O(n_tokens x window)`
rather than `O(sum of sequence lengths)`. A dedicated paged-attention op (CPU first, then CUDA/Metal) is
the main planned performance item; it will not change architecture code.

## ADR-004: block hashes are verified

**Context.** vLLM keys the prefix cache by hash only (sha256 for cross-process safety).

**Decision.** Use a fast 64-bit chained hash *and* store the block's token ids; a hit requires equal
tokens. Single-process only (no cross-process cache sharing yet).

**Consequences.** Zero risk of serving wrong KV on collision; `block_size * 4` bytes per block extra.

## ADR-005: preemption is recompute-only

Same as vLLM v1. Swap-to-CPU would be a second memory subsystem for little gain once prefix caching
exists: a preempted request re-hits its own blocks on resume unless they were evicted.

## ADR-006: one engine thread, no process boundary

vLLM separates the API server and engine core into processes (ZMQ) to escape the GIL. We have no GIL:
HTTP workers and the engine thread share one process; hand-off is a mutex-protected queue and a per-request
output queue. Multi-model serving will be one process per model behind a router (like llama-server's
router mode), which also isolates crashes.

## ADR-007: shared library for the core

Architectures register themselves through static initializers. A static archive would let the linker drop
unreferenced arch objects; `libiian.so` keeps them all. (A static build would need `--whole-archive`.)

## ADR-008: CLI is a daemon manager, not just a launcher

`vllm serve` cannot be listed, stopped or inspected. `iian serve -d` writes a record under
`~/.iian/run/`, `iian ps/stop/logs/status` operate on it; models live in `~/.iian/models` and are managed
with `pull/ls/rm`. Foreground servers write the same record so `ps` sees them too.

## ADR-009: logging is structured from day one

Every line: timestamp, level, subsystem tag, optional request id; pretty or JSON (`--log-format json`);
file tee for daemons; a periodic one-line throughput summary like vLLM's, but only while there is traffic.

## ADR-010: OpenAI compatibility is strict, extensions are explicit

Response shapes, SSE framing (`role` chunk, deltas, `finish_reason`, usage chunk, `[DONE]`), error object
and finish reasons follow OpenAI; vLLM-style extensions (`top_k`, `min_p`, `repetition_penalty`,
`stop_token_ids`, `min_tokens`, `chat_template_kwargs`, `cache_salt`, `priority`, ...) are accepted with the
same names so existing clients keep working. Engine-control endpoints require `--enable-admin`.
