# Roadmap

Status legend: ✅ done · 🚧 in progress · ⬜ planned

## Phase 0 — foundation ✅
- ✅ vendored ggml, CMake build, CPU backend (GPU backends via ggml flags)
- ✅ GGUF loader, mmap zero-copy weights, device placement
- ✅ architecture plug-in system (`src/iian/arch/*.cpp`), llama + qwen2 verified token-exact vs llama.cpp
- ✅ standalone tokenizer (all llama.cpp vocab types), verified id-exact vs `llama-tokenize`
- ✅ Jinja chat templates (llama.cpp engine port)
- ✅ paged KV cache: block pool, ref counting, LRU, verified prefix caching
- ✅ vLLM-v1 scheduler: token budget, chunked prefill, preemption (recompute), FCFS/priority
- ✅ engine loop, graph reuse, incremental detokenization, stop strings/tokens, min/max tokens, logprobs
- ✅ CPU sampler: temperature, top-k, top-p, min-p, penalties, logit bias, bad words, seeds
- ✅ e2e tests: batching == sequential, prefix cache == no cache, preemption == no preemption

## Phase 1 — serving ✅
- ✅ OpenAI server: /v1/chat/completions (+SSE), /v1/completions, /v1/models, /health, /metrics, /tokenize
- ✅ `iian` CLI: serve (foreground/daemon), ps, stop, logs, status, run, complete, pull, ls, rm, info, bench
- ✅ more architectures (see docs/models.md): qwen3, gemma3 (sliding window), llama-3.x rope factors, phi3, granite, smollm3
- ✅ structured output: GBNF grammar sampler (llama-grammar port) + JSON schema → grammar + regex → grammar; `response_format` json_object/json_schema, guided_json/choice/grammar/regex
- ✅ tool calling: hermes/qwen, llama3, mistral, generic formats; `tool_choice` required/named via grammar; streaming deltas
- ✅ embeddings endpoint (`/v1/embeddings`, mean/last/cls pooling, base64, dimensions); ⬜ reranking
- ✅ multi-model serving: `iian serve a.gguf b.gguf --models-dir D --models-max N --lazy`; per-request routing, lazy load, LRU unload, admin load/unload

## Phase 2 — performance
- ✅ **paged attention kernel** (CPU, `src/iian/paged_attn.cpp`): ggml custom op walking per-sequence block tables,
  O(sum of sequence lengths); AVX2/F16C SIMD; auto-selected on CPU-only runs (`--attention auto|masked|paged`)
- ⬜ paged attention for CUDA/Metal backends
- ✅ speculative decoding: n-gram/prompt-lookup drafts (`--spec-ngram N`, exact one-hot rejection sampling, greedy output unchanged); ✅ draft-model drafts (`--spec-draft small.gguf`, batched draft KV; 89% acceptance with a Q8 SmolLM2 draft for the F16 target, output bit-identical)
- ⬜ sliding-window KV eviction (free out-of-window blocks, like vLLM's SlidingWindowManager)
- ⬜ quantized KV cache validation on GPU backends (q8_0/q4_0 K/V already selectable)
- ⬜ CUDA graphs for steady-state decode; pipeline parallel across GPUs (ggml sched supports it)
- ⬜ prefill/decode budget tuning, long-prefill threshold defaults per device
- ⬜ benchmark suite (`iian bench`, serving benchmark vs llama-server and vLLM)

## Phase 3 — ecosystem
- ⬜ Python bindings (pybind11) mirroring the engine API
- ⬜ HF → GGUF conversion integration (`iian convert`, reusing llama.cpp's `convert_hf_to_gguf.py`)
- ⬜ quantization (`iian quantize`, ggml_quantize_chunk + imatrix)
- ⬜ multimodal (vision/audio encoders, mmproj GGUF)
- ⬜ LoRA adapters (GGUF LoRA, hot-swap)
- ⬜ Anthropic Messages API and Responses API compatibility
- ⬜ KV cache offload / disaggregated prefill
