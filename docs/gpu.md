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
