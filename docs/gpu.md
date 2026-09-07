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

## Performance on an RTX 5090 (SmolLM2-135M-Instruct F16, CUDA 13, 2026-09-07)

`tests/bench-batch` (256-token prompts, 128 generated tokens, greedy) against `llama-batched-bench` from the same
ggml commit built with CUDA, same GPU (`-ngl 99 -fa on`):

| concurrency | iian prompt tok/s | iian gen tok/s | iian TTFT | iian TPOT | llama.cpp PP tok/s | llama.cpp TG tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 16983 | 1036 | 15 ms  | 0.9 ms  | 7497  | 1163 |
| 8  | 32513 | 3345 | 63 ms  | 1.9 ms  | 24240 | 2275 |
| 32 | 29329 | 2546 | 170 ms | 11.2 ms | 25069 | 1874 |
| 64 | 30264 | 3780 | 300 ms | 14.5 ms | 25134 | 2451 |

iian's gen tok/s counts the whole run including the prefill phase (llama.cpp's TG excludes it), so the
single-stream figures are close to parity; batched decode is ahead of llama.cpp's unified KV cache from
concurrency 8 up because gather attention keeps every sequence's attention O(its own length). The batch-32 dip is
ggml-cuda's own F16 matmul dispatch (the tensor-core `mmf` kernel serves n <= 16, cuBLAS above) and shows in
llama.cpp too.

What it took (each item was found with `IIAN_PROFILE=1`, which prints per-phase step timings, and
`IIAN_PROFILE=ops`, which times every graph node):

- **CUDA graphs**: standalone ggml defaults them off; iian's CMake turns them on (`GGML_CUDA_GRAPHS`).
- **KV cell count aligned to 256**: ggml's CUDA flash attention only takes its fast kernels when the KV length is
  a multiple of 256 (`FATTN_KQ_STRIDE`); a 400-cell cache ran 2.5x slower.
- **Sampler**: vectorised argmax / log-sum-exp / top-k, `min_p` as a filter instead of a sort, bucket sort for
  `top_p`, logits rows sampled in place and step buffers kept allocated (a >128 KiB vector per step is mmap'd and
  page-faulted every time): 209 -> 31 us per token.
- **Warmup on the engine thread**: the engine runs a small prefill + decode on its own thread at construction.
  CUDA loads kernels lazily (`CUDA_MODULE_LOADING=EAGER` takes two minutes for ggml's kernel set) and the first
  CUDA work on a fresh thread costs ~90 ms, so a warmup on any other thread does not help the serving thread.
  `--no-warmup` skips it; `IIAN_WARMUP_TOKENS=N` changes the prefill size.
- **Gather only when needed**: a single contiguous sequence uses the plain masked flash-attention path (no K/V
  gathers); multi-sequence steps use gather attention.

Remaining known gaps: the first request of each new batch-size class still pays 10-90 ms of lazy kernel loading
(inherent to CUDA lazy loading; llama.cpp behaves the same), and single-stream decode is a few percent behind
llama.cpp (per-step CPU overhead, two scheduler splits for the host-resident token embedding).
