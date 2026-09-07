# iian architecture

`iian` ("Inference Is All You Need") is a from-scratch LLM inference engine written in C++20 on top of
[ggml](https://github.com/ggml-org/ggml). It combines the two halves that today live in different projects:

| | llama.cpp | vLLM | iian |
|---|---|---|---|
| Runtime | C/C++ | Python + CUDA | C++ |
| Weights | GGUF, 40+ quant types | safetensors (GGUF via plugin) | GGUF native |
| Backends | CPU / CUDA / Metal / Vulkan / ... (ggml) | CUDA / ROCm / ... (torch) | everything ggml supports |
| KV cache | unified cell cache, per-slot | paged blocks, prefix cache | paged blocks, prefix cache |
| Scheduling | per-slot continuous batching | token-budget scheduler, chunked prefill, preemption | same as vLLM |
| Serving | llama-server | OpenAI server | OpenAI server |
| CLI | many binaries | `vllm serve` only | one `iian` binary with daemon/model mgmt |

## Layering

```
 tools/iian            CLI: serve / run / ps / stop / logs / pull / ls / info / bench ...
 src/iian/cli          argument parsing, daemon records (~/.iian/run), model cache (~/.iian/models)
 src/iian/server       OpenAI-compatible HTTP (cpp-httplib), SSE streaming, metrics
 ───────────────────────────────────────────────────────────────────────────────────────
 src/iian/engine.cpp   Engine: step loop thread, micro-batch construction, graph reuse, output processing
 src/iian/scheduler.cpp vLLM-v1 scheduler (token budget, chunked prefill, preemption = recompute)
 src/iian/kv_cache.cpp PagedKVCache + BlockPool (ref-counted blocks, LRU free list, prefix-cache hash table)
 src/iian/sampler.cpp  per-request CPU sampler (penalties, top-k/p, min-p, logit bias, logprobs)
 src/iian/graph.cpp    GraphContext: ggml graph building blocks (norm / ffn / moe / qkv / rope / attention)
 src/iian/arch/*.cpp   one file per model architecture (llama, qwen2, qwen3, gemma3, ...)
 src/iian/model_loader.cpp GGUF -> device placement -> zero-copy mmap or upload to backend buffers
 src/iian/tokenizer    llama.cpp vocabulary port (all tokenizer types, 59 pre-tokenizers), standalone
 src/iian/chat         llama.cpp Jinja engine port + ChatTemplate facade
 ───────────────────────────────────────────────────────────────────────────────────────
 third_party/ggml      vendored ggml (tensor ops, quant formats, GGUF, CPU/CUDA/Metal/Vulkan/... backends)
```

Everything above the line is ours. `tokenizer/` and `chat/` are faithful ports of llama.cpp code (MIT), made
standalone: they read GGUF metadata directly and have no dependency on libllama.

## Request lifecycle

1. **HTTP** thread parses the request, renders the chat template (Jinja) and tokenizes.
2. `Engine::submit()` creates a `Request` (prompt tokens, `SamplingParams`, per-request RNG, incremental
   detokenizer, output queue) and pushes it to the engine thread.
3. **Engine thread** (`Engine::step()`):
   1. ingest new requests / aborts
   2. `Scheduler::schedule()` picks `(request, n_new_tokens)` pairs under the token budget
   3. builds a `UBatch`: token ids, positions, KV slot per token, per-sequence cell lists, output rows
   4. builds (or reuses) the ggml graph for the architecture, allocates it through `ggml_backend_sched`,
      fills the input tensors (tokens, positions, K/V write indices, attention mask), computes
   5. copies logits for the rows that need them, samples one token per finished-prefill/decoding request
   6. detokenizes incrementally, checks stop conditions, pushes `OutputChunk`s to each request's queue
   7. commits full KV blocks to the prefix cache, frees finished requests
4. The HTTP thread pops chunks and streams SSE deltas (or accumulates for non-streaming).

The single invariant that drives everything (borrowed from vLLM v1): every request has
`num_computed_tokens`, and each step schedules work so that it catches up with `tokens.size()`.
There is no separate prefill/decode phase: a fresh request whose prompt is 1000 tokens gets scheduled
as 512 + 488 tokens (chunked prefill) and then 1 token per step (decode). A prefix-cache hit simply
starts `num_computed_tokens` at the number of cached tokens.

## Paged KV cache

`PagedKVCache` owns, per layer, two ggml tensors `K[n_embd_k_gqa, n_cells]` and `V[n_embd_v_gqa, n_cells]`
on the layer's device. Cells are grouped into blocks of `block_size` (default 16). A request owns a
*block table* (`KVRequestState::block_ids`); token `i` lives in cell
`block_ids[i / block_size] * block_size + i % block_size`.

`BlockPool` is a port of vLLM's `BlockPool`:

* flat `std::vector<Block>` with ref counts, an intrusive doubly-linked **free list** (O(1) removal when a
  cached-but-free block is re-hit), and a `hash -> block` multimap;
* `touch()` takes a reference and removes the block from the free list;
* `free_blocks()` returns blocks to the free list: uncached blocks at the **front** (reused first),
  cached blocks at the **back** (kept for prefix reuse; LRU eviction);
* requests free their blocks in reverse order so the tail of a chain is evicted first.

**Prefix caching**: full blocks get a chained 64-bit hash `H_i = hash(H_{i-1}, tokens_i, salt)`,
computed incrementally as tokens are appended. On admission the scheduler walks the request's hashes and
takes the longest run of cached blocks (at most `n_tokens - 1` tokens so the last token is always
recomputed to obtain logits). Unlike vLLM we also store the block's token ids and verify them on a hit,
so a hash collision can never return wrong KV data. Shared prefixes are shared physically
(ref-counted), never copied.

**Preemption** is recompute-only, like vLLM v1: a preempted request frees its blocks and goes back to
the front of the waiting queue with `num_computed_tokens = 0`; on resume, the prefix cache usually
serves most of it again.

## Attention over paged KV on ggml

K/V writes use `ggml_set_rows(cache, rows, slot_ids)`: an I64 index tensor maps each token of the batch to
its cell — the same primitive llama.cpp uses, and equivalent to vLLM's `slot_mapping`.

Attention reads a **window** `[kv_start, kv_start + n_kv)` of cells (padded to 256 so graph topology is
stable and can be reused) and a `[n_kv, n_tokens]` mask that is `0` for cells the query token may attend
(same sequence, causal, inside the sliding window) and `-inf` elsewhere. `ggml_flash_attn_ext` (or the
`mul_mat` + `soft_max_ext` fallback) then runs one fused kernel for the whole batch across all sequences.

This is exactly how llama.cpp's unified cache works and it runs on every ggml backend today. Its cost is
`O(n_tokens x window)`: with many concurrent sequences the window covers cells that belong to other
sequences (masked out). To keep the window tight, the block allocator prefers low addresses (the free list is
LIFO for uncached blocks).

**Paged attention kernel** (`src/iian/paged_attn.cpp`). On CPU-only runs the engine instead uses iian's own
kernel, a ggml custom op that receives the batch's per-sequence cell lists (block tables) and, for every
(query token, head) pair, computes a two-pass softmax over that sequence's own cells only:
`O(sum of context lengths)`. F16/F32 K/V, GQA, sliding window, logit softcap; AVX2+F16C SIMD dot/axpy with a
scalar fallback. The engine picks it per step (`EngineConfig::attention = auto|masked|paged`): tiny decode
batches on models with few heads (`n_tokens x n_head < 2 x threads`) stay on ggml's fused kernel, everything
else goes paged; the choice is part of the graph-reuse key. Numerically it sits within the spread of ggml's
own two attention paths (see `docs/models.md`). GPU backends still use the masked path until a CUDA/Metal
kernel lands; both paths live behind `GraphContext::build_attn()`, so architecture code never changes.

## Scheduler

`Scheduler::schedule()` is a direct port of vLLM v1's algorithm:

* **Phase 1 – running queue**: every running request gets `min(remaining, budget)` tokens. If KV
  allocation fails, the *last* running request (lowest priority under FCFS; max priority value under
  PRIORITY) is preempted and the allocation is retried. Requests already scheduled this step are refunded
  if they are the victim.
* **Phase 2 – waiting queue**: only if nothing was preempted this step. Prefix-cache lookup, allocation,
  admission until the budget or `max_num_seqs` is exhausted. Allocation failure here stops admission
  (no preemption from the waiting queue), which guarantees forward progress.
* `update_after_step()` advances `num_computed_tokens` and commits full blocks to the prefix cache,
  capped at the number of real tokens (so draft tokens can never poison the cache once speculative
  decoding is added).

Knobs (`SchedulerConfig`): `max_num_seqs`, `max_num_batched_tokens`, `long_prefill_token_threshold`,
`enable_chunked_prefill`, `policy` (FCFS / PRIORITY).

## Graph building and graph reuse

`GraphContext` mirrors llama.cpp's `llm_graph_context` closely so that model code ports almost verbatim:
`build_inp_embd`, `build_norm`, `build_ffn` (SiLU/GELU/... with fused `swiglu_split`), `build_moe_ffn`
(`ggml_top_k` + `ggml_mul_mat_id`), `build_qkv` (fused or separate, biases, clamp), `build_rope`,
`build_attn`, `select_outputs` (compute the LM head only for rows that need logits), `build_lm_head`
(final norm, output projection, logit softcap/scale).

The engine keys the graph on `(n_tokens, n_outputs, n_kv, kv_start)`. Steady-state decode (same batch
shape) reuses the previous graph and its allocation and only refills the input tensors, which removes the
graph-build/alloc overhead from the per-token critical path.

## Model loading

`ModelLoader::load()`:

1. parse GGUF metadata (`GgufFile`), generic hparams, then the architecture's `load_hparams()`;
2. build the tokenizer from the metadata;
3. choose devices (all GPUs ggml reports, else CPU) and assign layers (`n_gpu_layers`);
4. the architecture's `create_tensors()` declares the weights it expects (names, shapes, required or
   optional); every tensor is validated against the file — a wrong shape or a missing required tensor
   fails loading with a message naming the tensor;
5. tensors on the CPU device are **zero-copy**: their ggml buffer points into the `mmap`'d file
   (`ggml_backend_dev_buffer_from_host_ptr`); GPU tensors are uploaded with `ggml_backend_tensor_set`.

## Sampling

`SamplerState::sample()` runs per request on the CPU with vLLM semantics: logprobs are computed from the
raw distribution; then `allowed_token_ids`, `logit_bias`, `bad_words`, repetition/presence/frequency
penalties; then greedy or temperature + top-k + top-p + min-p sampling with the request's own RNG (`seed`).
`min_tokens` masks EOS/stop tokens in the engine. Stop strings are checked on the detokenized text
with a hold-back of `max(len(stop)) - 1` bytes so a partial stop string is never streamed.

## Speculative decoding (prompt lookup)

With `spec_ngram = k`, after each accepted token the engine looks for the most recent earlier occurrence of
the request's last `n` tokens (`n` from 4 down to 1) and proposes the `k` tokens that followed it. The
scheduler appends the drafts to the request's real tokens for the next step (KV slots included), the graph
returns logits for the last real token and every draft position, and verification samples from the target
distribution at each position: a draft is accepted only when the sample equals it, and the first mismatch's
sample is the recovered token. For one-hot drafts this *is* exact rejection sampling, so greedy and stochastic
outputs are distributed exactly as without speculation (`test-batching ... auto 4` checks bit-identical
greedy output). Rejected drafts roll `num_computed_tokens` back; their KV cells are overwritten next step.
Drafts are skipped for grammar-constrained requests. Acceptance rate is reported in the stats line and
`/v1/engine/stats` (`spec_drafted`, `spec_accepted`).

**Draft model** (`spec_draft_model`, `src/iian/draft.cpp`): a small model with the same tokenizer keeps its
own paged KV cache per request. After every target step the draft model is fed the tokens it has not seen
(batched across requests, chunked), then runs `n_draft` greedy decode rounds (one batched micro-batch per
round) to produce the proposals; accepted drafts already sit in its cache at the right positions, so only
the recovered/bonus token needs to be fed next time. Verification is the same exact one-hot rejection
sampling as for n-gram drafts. Preempted or finished requests release their draft blocks.

## Embeddings

`Engine::submit_embedding()` runs a prompt through the same scheduler (chunked prefill included) with every
token of the chunk as an output row, pools the final normed hidden states (`t_embd`) across chunks (mean /
last / cls), optionally L2-normalizes and returns the vector in the finishing `OutputChunk`.

## Threading model

* one **engine thread** runs the step loop; it is the only thread touching ggml compute, the scheduler and
  the KV cache;
* HTTP worker threads only call `Engine::submit/abort/stats` (mutex-protected queues) and block on a
  request's `OutputQueue`;
* the CPU backend's compute threads are ggml's own thread pool (`--threads`).

## Adding a model architecture

One file in `src/iian/arch/` (globbed by CMake), three methods, one macro:

```cpp
class ArchFoo : public iian::ArchDef {
    const char * name() const override { return "foo"; }
    void load_hparams(const GgufFile & f, HParams & hp) const override { ... }   // arch-specific keys
    void create_tensors(Model & m, TensorCreator & tc) const override { ... }    // declare weights
    void build_graph(GraphContext & g) const override { ... }                    // forward graph
    RopeType rope_type(const HParams &) const override { return RopeType::NEOX; }
};
IIAN_REGISTER_ARCH("foo", ArchFoo);
```

No enums, no switch statements, no CMake edits. See `docs/models.md` for the verification workflow
(token-exact comparison against llama.cpp).

## What was copied, what was written

* **Copied (MIT, llama.cpp/ggml)**: `third_party/ggml` (verbatim), `src/iian/tokenizer` (llama-vocab +
  unicode, adapted to read GGUF directly), `src/iian/chat/jinja` (llama.cpp's Jinja engine, adapted),
  `third_party/cpp-httplib`, `third_party/nlohmann`.
* **Ported as algorithms (Apache-2.0, vLLM)**: scheduler, block pool / prefix caching, request lifecycle,
  stop handling, sampler semantics, OpenAI protocol shapes, metrics names. No vLLM code is included.
* **Written for iian**: everything else (GGUF loader, model loader, arch plug-in system, graph context,
  paged KV cache on ggml, engine, sampler, server, CLI/daemon).
