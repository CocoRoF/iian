#!/usr/bin/env bash
# Runs on the GPU server: pull the latest commit, build with CUDA, run the test suite on the GPU.
# Usage: scripts/gpu-test.sh [quick|full]   (quick = build + generate/batching/server; full = + arch verification)
set -uo pipefail
MODE=${1:-quick}
FAILS=0
run() { # run <label> <command...>: print filtered output, count non-zero exits
  local label=$1; shift
  echo "== $label"
  if ! "$@"; then echo "!! $label FAILED"; FAILS=$((FAILS+1)); fi
}
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
git pull -q --ff-only || exit 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=$CUDACXX > build-cmake.log 2>&1
cmake --build build -j 20 > build.log 2>&1 || { tail -30 build.log; exit 1; }
M=models/SmolLM2-135M-Instruct-F16.gguf
# use the GPU with the most free memory (the box is shared)
BEST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export IIAN_DEVICES="CUDA${BEST:-0}"
echo "== using $IIAN_DEVICES"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
echo "== devices"; ./build/tools/iian/iian version | tail -4
B='^\[|PASSED|FAIL|note|ref:|got:|top:|speculative|numerics'
run "generate (GPU)"        sh -c "./build/tests/test-generate $M 'The capital of France is' 48 --threads 8 2>&1 | grep -E 'TEXT|tok/s|backends|ready' | cut -c1-200"
run "batching (GPU)"        sh -c "./build/tests/test-batching $M 2>&1 | grep -E '$B'"
run "batching + spec-ngram" sh -c "./build/tests/test-batching $M auto 4 2>&1 | grep -E '$B'"
run "batching + draft"      sh -c "./build/tests/test-batching $M auto 0 models/SmolLM2-135M-Instruct-Q8_0.gguf 2>&1 | grep -E '$B'"
run "batching, KV q8_0"     sh -c "./build/tests/test-batching $M auto 0 - q8_0 2>&1 | grep -E '$B'"
run "generate, KV q4_0"     sh -c "./build/tests/test-generate $M 'The capital of France is' 32 --threads 8 --kv q4_0 2>&1 | grep -E 'TEXT|tok/s' | cut -c1-200"
run "server e2e"            sh -c "python3 tests/server/test-server.py build/tools/iian/iian $M 2>&1 | grep -E 'FAIL|checks'"
run "bench"                 sh -c "./build/tests/bench-batch $M --concurrency 1,8,32,64 --prompt 256 --gen 128 --threads 8 2>&1 | grep -v '^\x1b\[90m'"
if [ "$MODE" = full ]; then
  echo "== arch verification vs llama.cpp (CUDA)"
  LLAMA_BIN=$HOME/ref/llama.cpp/build/bin
  for m in models/*.gguf; do
    for p in "The capital of France is" "def quicksort(arr):" "Once upon a time"; do
      ours=$(./build/tests/test-generate "$m" "$p" 32 --threads 8 --attn masked 2>/dev/null | sed -n 's/^TEXT: //p' | sed 's/\\n/ /g' | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')
      ref=$($LLAMA_BIN/llama-completion -m "$m" -p "$p" -n 32 --temp 0 -no-cnv --no-warmup -ngl 99 2>/dev/null | tr -d '\r' | sed 's/\[end of text\]//' | tr -s '[:space:]' ' ' | sed 's/^ *//; s/ *$//')
      if [ "$ours" = "$ref" ]; then echo "PASS  $(basename $m) | $p"; else echo "FAIL  $(basename $m) | $p"; echo "   iian : $ours"; echo "   llama: $ref"; FAILS=$((FAILS+1)); fi
    done
  done
fi
echo "== done: $FAILS failing step(s)"
exit $((FAILS > 0))
