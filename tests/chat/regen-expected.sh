#!/usr/bin/env bash
# Regenerates tests/chat/templates/*.expected (and gguf-*.expected) using llama.cpp's reference implementation.
# The .expected files are the ground truth the test compares against; rerun this after updating ref/llama.cpp.
#
#   LLAMA_TEST_CHAT_TEMPLATE  path to llama.cpp's test-chat-template (build with -DLLAMA_BUILD_TESTS=ON)
#   IIAN_TEST_CHAT_TEMPLATE   path to this project's test-chat-template (used for --dump-gguf)
#
# NOTE: templates using dates are rendered by llama.cpp with the *current* time; keep "now_iso8601" in the
#       fixture .json files in sync with the day the .expected files were generated.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LLAMA_BIN="${LLAMA_TEST_CHAT_TEMPLATE:-$ROOT/ref/llama.cpp/build-tests/bin/test-chat-template}"
IIAN_BIN="${IIAN_TEST_CHAT_TEMPLATE:-$ROOT/build-chat/tests/chat/test-chat-template}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

render() { # <template file> <input json> <output file>
    "$LLAMA_BIN" "$1" --json "$2" --output "$3" > "$TMP/log.txt" 2>&1 || { cat "$TMP/log.txt"; echo "FAILED: $1"; exit 1; }
}

for tmpl in "$HERE"/templates/*.jinja; do
    name="$(basename "$tmpl" .jinja)"
    input="$HERE/templates/$name.json"
    [ -f "$input" ] || { echo "skip $name (no input json)"; continue; }
    render "$tmpl" "$input" "$HERE/templates/$name.expected"
    echo "wrote templates/$name.expected"
done

for model in SmolLM2-135M-Instruct-F16 Qwen2.5-0.5B-Instruct-Q4_K_M; do
    gguf="$ROOT/models/$model.gguf"
    [ -f "$gguf" ] || { echo "skip $model (missing gguf)"; continue; }
    "$IIAN_BIN" --dump-gguf "$gguf" > "$TMP/dump.txt"
    python3 - "$TMP/dump.txt" "$TMP/$model.jinja" "$TMP/$model.json" <<'PY'
import json, sys
dump, tmpl_path, json_path = sys.argv[1:]
text = open(dump, encoding="utf-8").read()
head, tmpl = text.split("--- template ---\n", 1)
tmpl = tmpl[:-1] if tmpl.endswith("\n") else tmpl   # --dump-gguf appends one newline
bos = json.loads(head.split("bos_token: ",1)[1].split("\n",1)[0])
eos = json.loads(head.split("eos_token: ",1)[1].split("\n",1)[0])
open(tmpl_path, "w", encoding="utf-8").write(tmpl)
inp = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    "bos_token": bos, "eos_token": eos, "add_generation_prompt": True,
}
json.dump(inp, open(json_path, "w", encoding="utf-8"), ensure_ascii=False)
PY
    render "$TMP/$model.jinja" "$TMP/$model.json" "$HERE/templates/gguf-$model.expected"
    echo "wrote templates/gguf-$model.expected"
done
