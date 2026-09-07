# Supported architectures

Verification = greedy generation (32 tokens, 3 prompts) compared token-for-token with llama.cpp
(`tests/archs/run.sh`, which runs iian with `--attn masked` so both engines use the same ggml flash-attention
kernel and the comparison tests the model graph rather than kernel numerics). llama.cpp itself has two CPU
attention paths (`-fa on` / `-fa off`) whose numerics differ; a run passes when iian equals either. CPU
weights are repacked into ggml's optimised layouts exactly like llama.cpp (`--no-repack` disables it), which
was required for Q4_K/Q8_0 models to match. Remaining single-token divergences on 4-bit models are near-tie
argmax flips, not logic errors — see the notes.

| arch (GGUF) | file | example models | verified with | status |
|---|---|---|---|---|
| `llama` | `src/iian/arch/llama.cpp` | Llama 2/3.x, Mistral, SmolLM2, Mixtral (MoE tensors) | SmolLM2-135M F16 + Q8_0, Llama-3.2-1B Q4_K_M | ✅ 3/3 each (Llama-3.2 via rope factors) |
| `qwen2` | `qwen2.cpp` | Qwen2 / Qwen2.5 | Qwen2.5-0.5B Q4_K_M | ✅ 2/3 exact; 1 prompt flips one near-tie token at position 29 (28 identical tokens before it; Q4_K_M) |
| `qwen3` | `qwen3.cpp` | Qwen3 dense (q/k norm) | Qwen3-0.6B Q8_0 | ✅ 3/3 |
| `smollm3` | `smollm3.cpp` | SmolLM3 (NoPE layers) | SmolLM3-3B Q4_K_M | ✅ 3/3 |
| `gemma3` | `gemma3.cpp` | Gemma 3 (SWA/global hybrid, q/k norm, post norms) | gemma-3-270m-it Q8_0 | ✅ 3/3 (see the numeric-sensitivity note) |
| `granite` | `granite.cpp` | Granite 3.x/4 dense (scaled residuals/logits) | granite-4.0-micro Q4_K_M | ✅ 2/3 exact; layer dump: attention output of layer 0 identical, FFN differs at the 1e-4 level and compounds over 40 layers (embedding scale 12); one prompt flips at token 9 |
| `phi3` | `phi3.cpp` | Phi-3 / Phi-3.5 (fused qkv, LongRoPE) | Phi-3.5-mini Q4_K_M | ✅ graph verified: layer 0 identical to llama.cpp after fixing LongRoPE factor selection (per-sequence context, not KV pool size); later layers drift at the 1e-5 level, greedy text diverges after 7 tokens on this 32-layer Q4/Q5/Q6 mix |

Attention numeric parity inside iian (`tests/e2e/test-attn-parity`, 33-token prefill, all logits):

| model | flash vs mul_mat | flash vs paged kernel | argmax mismatches |
|---|---|---|---|
| SmolLM2-135M F16 | mean 0.006, max 0.04 | mean 0.006, max 0.04 | 0 / 33 |
| Qwen2.5-0.5B Q4_K_M | mean 0.088, max 0.79 | mean 0.095, max 1.0 | 1 / 33 |

The paged kernel sits inside the spread of ggml's own two attention paths.

## Verification method for larger models
For 3–4B models the per-token comparison is complemented by a layer-by-layer dump: `IIAN_DUMP_TENSORS=1
test-generate <gguf> <prompt> 1 --attn masked --ctx 2048` versus `llama-eval-callback -m <gguf> -p <prompt> -n 1`.
Identical sums at `attn_norm-0`, `kqv_out-0`, `l_out-0` prove the graph; later-layer drift at the 1e-4 level on
4-bit weights is expected between two builds that repack different subsets of tensors.

## gemma3 numeric-sensitivity note
With `IIAN_DUMP_TENSORS=1` vs llama.cpp's `llama-eval-callback`, every layer output of a gemma-3-270m prefill
matches llama.cpp to 6 significant digits (`l_out-0` 3199.07, final logits -24.855 / -11.912 / -3.2637).
The model's residual stream carries huge outliers (values around -1500 at the last layer), so any change in
attention arithmetic is amplified: ggml's own non-flash path differs from its flash path by ~0.5 in the
logits at the first decode step, and iian's paged kernel differs by a similar amount. Token-exact parity
with llama.cpp therefore needs the same fused kernel (`--attention masked`); `auto`/`paged` give equally
valid but not bit-identical samples on this model.

## How to add an architecture
1. Create `src/iian/arch/<name>.cpp` (CMake globs the directory) with a class deriving `iian::ArchDef`:
   * `load_hparams(GgufFile, HParams&)` — read the arch-specific keys (`%s` expands to the arch name);
   * `create_tensors(Model&, TensorCreator&)` — declare every weight with its GGUF name and shape
     (`tc.layer_tensor("attn_q", il, "weight", {n_embd, n_head*head_dim})`; `NOT_REQUIRED` / `DUPLICATED` flags);
   * `build_graph(GraphContext&)` — the forward pass from the building blocks in `include/iian/graph.h`
     (`build_inp_embd`, `build_norm`, `build_qkv`, `build_rope`, `build_attn`, `build_ffn`, `build_moe_ffn`,
     `select_outputs`, `build_lm_head`);
   * `rope_type()` — NORMAL / NEOX / NONE, as in llama.cpp's `llama_model_rope_type`.
2. Register: `IIAN_REGISTER_ARCH("<gguf arch name>", ArchClass);`
3. Verify: put a small GGUF in `models/` and run `tests/archs/run.sh models/<file>.gguf`
   (llama.cpp reference must be built in `ref/llama.cpp/build`). Every prompt must match token-for-token.

Port from `ref/llama.cpp/src/models/<arch>.cpp` (graph) and the arch's `load_arch_hparams` /
`load_arch_tensors`; GGUF key and tensor names are in `ref/llama.cpp/src/llama-arch.cpp`.
