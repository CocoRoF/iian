# Third-party notices

iian is licensed under the MIT License (see `LICENSE`). It includes or derives from the following works.

## ggml / llama.cpp — MIT
Copyright (c) 2023-2026 The ggml authors. https://github.com/ggml-org/llama.cpp

* `third_party/ggml/` — verbatim copy of the ggml library (tensor library, quantization, GGUF, backends).
* `src/iian/tokenizer/` — derived from `src/llama-vocab.{h,cpp}`, `src/unicode*.{h,cpp}`; adapted to be
  standalone (reads GGUF metadata directly; `iian` namespace; `iian::Tokenizer` interface).
* `src/iian/chat/jinja/` — derived from `common/jinja/` (llama.cpp's Jinja template engine).
* `src/iian/grammar/` — derived from `src/llama-grammar.{h,cpp}` and `common/json-schema-to-grammar.{h,cpp}`.
* `src/iian/graph.cpp` — the attention / FFN / MoE graph building blocks follow `src/llama-graph.cpp`.
* `src/iian/arch/*.cpp` — model graph definitions follow `src/models/*.cpp`.

## cpp-httplib — MIT
Copyright (c) 2025 Yuji Hirose. `third_party/cpp-httplib/`.

## nlohmann/json — MIT
Copyright (c) 2013-2025 Niels Lohmann. `third_party/nlohmann/`.

## vLLM — Apache License 2.0
Copyright contributors to the vLLM project. https://github.com/vllm-project/vllm

No vLLM source code is included. The following components re-implement algorithms and interface
conventions documented in vLLM v1 (scheduler, KV cache manager / block pool / prefix caching, request
lifecycle, sampler semantics, OpenAI protocol shapes, metric names): `src/iian/scheduler.cpp`,
`src/iian/kv_cache.cpp`, `src/iian/engine.cpp`, `src/iian/sampler.cpp`, `src/iian/server/`.

- `src/iian/vec_math.h`: the AVX2 `expf8` routine is adapted from ggml's `ggml-cpu/vec.h` (`ggml_v_expf`), itself
  adapted from the Arm Limited optimized routines (MIT).
