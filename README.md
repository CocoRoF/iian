# iian — Inference Is All You Need

**A from-scratch LLM inference engine in C++ that combines llama.cpp's foundations with vLLM's serving
architecture.**

* **GGUF native, every quantization**: built on [ggml](https://github.com/ggml-org/ggml) (vendored), so all
  GGUF models and quant formats (Q4_K_M, Q8_0, IQ*, MXFP4, ...) and all ggml backends (CPU, CUDA, Metal,
  Vulkan, SYCL, ...) work out of the box.
* **vLLM-class serving**: paged KV cache with automatic prefix caching, continuous batching with chunked
  prefill and preemption, a CPU paged-attention kernel, prompt-lookup speculative decoding, structured output
  (JSON schema / GBNF), tool calling, embeddings, per-request sampling, OpenAI-compatible API with SSE streaming.
* **A CLI that actually manages servers**: `iian serve -d`, `iian ps`, `iian stop`, `iian logs -f`,
  `iian status`, plus `pull / ls / rm / info / run / complete / bench`. Structured logs (pretty or JSON).
* **Adding a model is one file**: an architecture is a self-registering plug-in
  (`load_hparams` / `create_tensors` / `build_graph`), verified token-exactly against llama.cpp.

## Quick start

```bash
export PATH="$HOME/.local/bin:$PATH"     # cmake/ninja (installed via `uv tool install cmake ninja`)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release    # add -DGGML_CUDA=ON, -DGGML_METAL=ON, ...
cmake --build build -j
alias iian=$PWD/build/tools/iian/iian

iian pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m   # -> ~/.iian/models/...
iian serve qwen2.5-0.5b -d --port 9931 --enable-admin  # daemon; logs in ~/.iian/logs
# several models in one server: iian serve a.gguf b.gguf --models-max 1 -d   (routing by the `model` field)
# speculative decoding: iian serve big.gguf --spec-draft small.gguf     (or --spec-ngram 4 without a draft model)
iian ps                                                # NAME PID MODEL ENDPOINT STATUS UPTIME REQS
iian run qwen2.5-0.5b                                  # interactive chat over the API
curl localhost:9931/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
iian status qwen2.5-0.5b && iian logs -f qwen2.5-0.5b
iian stop all
```

Run a model without a server: `iian run models/model.gguf` (interactive) or
`iian complete models/model.gguf -p "The capital of France is" -n 32 --temp 0`.

## API

`POST /v1/chat/completions` (tools/tool_choice, response_format json_object/json_schema, guided_json/choice/grammar),
`POST /v1/completions`, `POST /v1/embeddings` (vLLM-style extensions: `top_k`, `min_p`, `repetition_penalty`,
`stop_token_ids`, `min_tokens`, `ignore_eos`, `chat_template_kwargs`, `cache_salt`, `priority`, ...), `GET /v1/models`, `POST /tokenize`, `POST /detokenize`, `GET /health`, `GET /metrics`
(Prometheus), `GET /props`, and with `--enable-admin`: `GET /v1/engine/stats`, `POST /v1/engine/abort`.
Errors use the OpenAI `{"error": {...}}` shape; `--api-key` enables bearer auth. See `docs/api.md`.

## Supported architectures

`llama` (Llama 1/2/3.x, Mistral, SmolLM2), `qwen2`, `qwen3`, `gemma3`, `phi3`, `granite`, `smollm3`.
See `docs/models.md` for the verification status and how to add one (`src/iian/arch/*.cpp`).

## Layout

```
include/iian/   public headers          src/iian/arch/      architectures (one file each)
src/iian/       engine, scheduler, kv   src/iian/tokenizer  llama.cpp vocab port (standalone)
src/iian/server OpenAI HTTP server      src/iian/chat       Jinja chat templates (llama.cpp port)
src/iian/cli    CLI + daemon manager    third_party/ggml    vendored ggml
tests/          unit / e2e / tokenizer / chat / server / archs
docs/           ARCHITECTURE.md, DESIGN-DECISIONS.md, ROADMAP.md, models.md, cli.md, api.md
```

## Tests

```bash
cd build && ctest --output-on-failure      # unit, tokenizer (id-exact vs llama-tokenize), chat templates
                                           # (374 cases vs llama.cpp), batching e2e, server e2e
tests/archs/run.sh                         # greedy output vs llama.cpp for every model in models/
```

`tests/e2e/test-batching` proves the serving path does not change results: concurrent batched requests,
prefix-cache hits and preempted requests all produce exactly the same tokens as isolated runs.

## Status and roadmap

Phases 0–1 (engine, serving, CLI) and most of Phase 2 (CPU paged attention, structured output, tool calling,
embeddings, prompt-lookup speculative decoding) are done; next: multi-model router, GPU paged attention,
draft-model speculation. See `docs/ROADMAP.md`.

## License

MIT. Includes ggml/llama.cpp code (MIT) and re-implements vLLM algorithms (Apache-2.0); see
`THIRD_PARTY_NOTICES.md`.
