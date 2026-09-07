# GPU development workflow

Code is edited locally and pushed to the public GitHub repository `CocoRoF/iian`; builds and GPU tests run on
the GPU server (2 x RTX 5090, CUDA 13.0, reached through a jump host), which pulls from GitHub over plain
https into `~/iian` (no credentials or keys on the server).

```
# local
git commit -am "..." && git push origin main
# remote (~/iian)
scripts/gpu-test.sh          # git pull, CUDA build (sm_120), generate/batching/spec/server tests, bench
scripts/gpu-test.sh full     # + greedy comparison against llama.cpp built with CUDA (~/ref/llama.cpp)
```

Build flags used on the server: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc` (the `nvcc` on PATH there is CUDA 12.0, which does not
support Blackwell). cmake/ninja come from `uv tool install`.

On a GPU, `--attention auto` uses ggml's fused flash-attention over the masked window (the CPU paged kernel is
CPU-only); layers are split across all visible GPUs (`--device CUDA0 --device CUDA1` to pin, `-ngl N` to keep
some layers on the CPU). The KV cache lives on each layer's device and is auto-sized to a quarter of the free
GPU memory unless `--kv-cache-gb` / `--kv-cache-tokens` is given.

## Notes from the RTX 5090 runs

- **Device selection.** `IIAN_DEVICES=CUDA1` (comma-separated names) overrides the device list for every
  tool and test, so a shared box can keep one GPU for other users; `scripts/gpu-test.sh` picks the GPU with the
  most free memory automatically. CUDA reports host (UMA) memory as "free", so the engine clamps the KV budget
  to the device's total memory before taking its quarter.
- **Batch-shape numerics.** CUDA chooses different matmul / flash-attention kernels depending on the batch
  size, so the same token computed inside a 47-token prefill and inside a 15-token suffix batch on a cached
  prefix differs by ~1e-2 nats. When the top-2 candidates of a greedy step are within a few 1e-3 nats this flips
  the token (observed with SmolLM2-135M F16: "Germany" -1.2831 vs "France" -1.2866). `test-batching` requests
  top-2 logprobs and accepts a divergence only when both runs show a near tie (< 0.1 nats) and each run's pick
  is the other's runner-up; any other mismatch is still a failure. CPU runs are bit-exact and never need this.
- **Non-flash attention is broken on sm_120 with CUDA 13 in the vendored ggml**: the CUDA softmax kernel fails
  with `CUDA error: invalid argument` (llama.cpp's own `llama-completion -fa off -ngl 99` fails the same way on
  the server). Keep flash attention on (the default). `IIAN_NO_FLASH_ATTN=1` exists only as a debugging switch
  for the mul_mat + softmax path on backends where it works.
