#!/usr/bin/env bash
# Token-exact verification of every architecture against llama.cpp (greedy, 32 tokens, 3 prompts).
# Usage: tests/archs/run.sh [model.gguf ...]   (default: all models in models/)
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEN="$ROOT/build/tests/test-generate"
LLAMA="$ROOT/ref/llama.cpp/build/bin/llama-completion"
N=${N:-32}
THREADS=${THREADS:-8}
GEN_FLAGS=${GEN_FLAGS:---attn masked --ctx 2048}   # same ggml FA kernel as llama.cpp: verifies the model graph, not kernel numerics
LLAMA_FLAGS=${LLAMA_FLAGS:--c 8192}
PROMPTS=("The capital of France is" "def quicksort(arr):" "Once upon a time")
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then MODELS=("$ROOT"/models/*.gguf); fi
fail=0; pass=0
for m in "${MODELS[@]}"; do
  name=$(basename "$m")
  for p in "${PROMPTS[@]}"; do
    ours=$("$GEN" "$m" "$p" "$N" --threads "$THREADS" $GEN_FLAGS 2>/dev/null | sed -n 's/^TEXT: //p' | sed 's/\\n/ /g; s/[[:space:]]*$//')
    # llama.cpp has two attention paths (-fa on/off) whose numerics differ slightly; a greedy run can
    # legitimately match either one. We accept a match against either path.
    ref_on=$("$LLAMA" -m "$m" -p "$p" -n "$N" --temp 0 -no-cnv --no-warmup -t "$THREADS" -fa on -c 2048 $LLAMA_FLAGS 2>/dev/null | tr -d '\r' | sed 's/\[end of text\]//' | sed 's/[[:space:]]*$//')
    ref_off=$("$LLAMA" -m "$m" -p "$p" -n "$N" --temp 0 -no-cnv --no-warmup -t "$THREADS" -fa off -c 2048 $LLAMA_FLAGS 2>/dev/null | tr -d '\r' | sed 's/\[end of text\]//' | sed 's/[[:space:]]*$//')
    ours_n=$(printf '%s' "$ours" | tr -s '[:space:]' ' ' | sed 's/^ *//')
    ref_on_n=$(printf '%s' "$ref_on" | tr -s '[:space:]' ' ' | sed 's/^ *//')
    ref_off_n=$(printf '%s' "$ref_off" | tr -s '[:space:]' ' ' | sed 's/^ *//')
    if [ -n "$ours_n" ] && { [ "$ours_n" = "$ref_on_n" ] || [ "$ours_n" = "$ref_off_n" ]; }; then
      pass=$((pass+1)); echo "PASS  $name  | $p $([ "$ours_n" = "$ref_on_n" ] && echo '(=fa on)' || echo '(=fa off)')"
    else
      fail=$((fail+1)); echo "FAIL  $name  | $p"; echo "   iian    : $ours_n"; echo "   llama on : $ref_on_n"; echo "   llama off: $ref_off_n"
    fi
  done
done
echo "archs: $pass passed, $fail failed"
[ $fail -eq 0 ]
