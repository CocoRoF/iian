#!/usr/bin/env bash
# Generate expected token ids for tests/tokenizer/corpus.txt using llama.cpp's llama-tokenize.
#
# Output: tests/tokenizer/expected/<model>.parse.txt   (parse_special = true,  llama-tokenize default)
#         tests/tokenizer/expected/<model>.noparse.txt (parse_special = false, --no-parse-special)
# One line per corpus line, in llama-tokenize --ids form: "[1, 2, 3]" (or "[]").
# BOS is added iff the model's tokenizer.ggml.add_bos_token is true (llama-tokenize behaviour); the test mirrors
# this by calling encode(text, special().add_bos, parse_special).
#
# Each corpus line is fed through a temp file with -f so the bytes are preserved exactly; llama-tokenize applies
# the same escape processing (\n \r \t \' \" \\ \xNN) as the test's unescape().
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF_BUILD="${LLAMA_BUILD_DIR:-$ROOT/ref/llama.cpp/build}"
TOKENIZE="$REF_BUILD/bin/llama-tokenize"
CORPUS="$ROOT/tests/tokenizer/corpus.txt"
OUT_DIR="$ROOT/tests/tokenizer/expected"

export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$TOKENIZE" ]; then
    echo "building llama-tokenize in $REF_BUILD ..." >&2
    cmake --build "$REF_BUILD" --target llama-tokenize -j 8
fi

MODELS=(
    "$ROOT/models/SmolLM2-135M-Instruct-F16.gguf"
    "$ROOT/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
)
if [ $# -gt 0 ]; then
    MODELS=("$@")
fi

mkdir -p "$OUT_DIR"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

for model in "${MODELS[@]}"; do
    name="$(basename "$model" .gguf)"
    for mode in parse noparse; do
        flag=()
        if [ "$mode" = noparse ]; then
            flag=(--no-parse-special)
        fi
        outfile="$OUT_DIR/$name.$mode.txt"
        : > "$outfile"
        n=0
        while IFS= read -r line || [ -n "$line" ]; do
            printf '%s' "$line" > "$tmp"
            ids="$("$TOKENIZE" -m "$model" -f "$tmp" --ids "${flag[@]}" 2>/dev/null | grep -E '^\[([0-9]+(, [0-9]+)*)?\]$' || true)"
            if [ -z "$ids" ]; then
                echo "error: llama-tokenize produced no ids for corpus line $((n + 1)) ($name, $mode)" >&2
                exit 1
            fi
            printf '%s\n' "$ids" >> "$outfile"
            n=$((n + 1))
        done < "$CORPUS"
        echo "wrote $outfile ($n lines)" >&2
    done
done
