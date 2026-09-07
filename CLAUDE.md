# iian — developer notes

C++20 LLM inference engine on vendored ggml: GGUF/quantization from llama.cpp, paged KV cache +
continuous-batching scheduler from vLLM, OpenAI server, daemon-style CLI. Read `docs/ARCHITECTURE.md`
first, then `docs/DESIGN-DECISIONS.md`. `ref/` holds read-only upstream checkouts (llama.cpp, vLLM) used
for porting and for token-exact verification; never edit them.

## Build / test
```
export PATH="$HOME/.local/bin:$PATH"          # cmake + ninja were installed with `uv tool install`
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # add -DGGML_CUDA=ON / -DGGML_METAL=ON etc.
cmake --build build -j4     # shared dev box: keep -j small, never run two model jobs at once
cd build && ctest --output-on-failure            # unit + tokenizer + chat + e2e (needs models/)
./build/tests/test-generate models/SmolLM2-135M-Instruct-F16.gguf "The capital of France is" 32
./build/tests/test-batching models/SmolLM2-135M-Instruct-F16.gguf            # [attn auto|masked|paged] [spec_ngram] [draft.gguf]
python3 tests/server/test-server.py build/tools/iian/iian models/SmolLM2-135M-Instruct-F16.gguf   # 81 checks incl. router
IIAN_DUMP_TENSORS=1 ./build/tests/test-generate m.gguf "prompt" 1 --attn masked --ctx 2048       # layer dump vs llama-eval-callback
```
Reference llama.cpp is built in `ref/llama.cpp/build/bin/` (`llama-completion -m ... -p ... -n N --temp 0 -no-cnv`
gives the greedy baseline every arch must match token-exactly).

## Layout
- `include/iian/` public headers (one per subsystem) · `src/iian/` implementation
- `src/iian/arch/*.cpp` one architecture per file, self-registering (`IIAN_REGISTER_ARCH`), globbed by CMake
- `src/iian/tokenizer`, `src/iian/chat` ports of llama.cpp (keep diffs to upstream minimal)
- `src/iian/server`, `src/iian/cli`, `tools/iian` HTTP server and CLI
- `tests/unit`, `tests/e2e`, `tests/tokenizer`, `tests/chat`, `tests/server`, `tests/archs`
- `models/` test GGUFs (not committed) · `docs/` design docs

## Resource rules (shared machine)
- Do not run verification/benchmark jobs in the background or in parallel; one model process at a time, <= 8 threads.
- Models >1 GB (Phi-3.5, SmolLM3, granite) need a quiet machine; do not download more models without asking.

## Conventions
- Correctness bar: greedy output identical to llama.cpp for every supported model (see `tests/archs/run.sh`).
- New engine features must keep `test-batching` green (batched == sequential, cached == uncached, preempted == not preempted).
- Logging via `LOG_INF("tag", ...)` from `iian/log.h`; tags: model, kv, sched, engine, http, cli.
- Public headers are the contract between subsystems; extend additively.
