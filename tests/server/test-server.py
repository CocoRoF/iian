#!/usr/bin/env python3
"""End-to-end test of the iian OpenAI-compatible server and the CLI daemon commands.

Usage: test-server.py <iian-binary> <model.gguf>
Only the Python standard library is used.
"""
import http.client
import json
import re
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

BIN = sys.argv[1]
MODEL = sys.argv[2]
API_KEY = "test-key-123"
HOME = tempfile.mkdtemp(prefix="iian-test-")
ENV = dict(os.environ, IIAN_HOME=HOME, NO_COLOR="1")
MODEL_NAME = os.path.basename(MODEL)
if MODEL_NAME.endswith(".gguf"):
    MODEL_NAME = MODEL_NAME[:-5]

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -> " + str(detail)[:300]))
    return bool(cond)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


PORT = free_port()


def request(method, path, body=None, headers=None, key=API_KEY, timeout=120):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    if headers:
        h.update(headers)
    conn.request(method, path, body=None if body is None else (body if isinstance(body, (str, bytes)) else json.dumps(body)), headers=h)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    try:
        j = json.loads(data.decode("utf-8"))
    except Exception:
        j = None
    return r.status, j, data.decode("utf-8", "replace"), dict(r.getheaders())


def sse(path, body, key=API_KEY):
    """Returns (status, list of parsed data payloads, done_seen)."""
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = conn.getresponse()
    if r.status != 200:
        data = r.read().decode()
        conn.close()
        return r.status, [], False, data
    events, done, buf = [], False, b""
    while True:
        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data:"):
                    payload = line[5:].strip().decode()
                    if payload == "[DONE]":
                        done = True
                    else:
                        events.append(json.loads(payload))
    conn.close()
    return 200, events, done, ""


def run_cli(*args, timeout=60):
    p = subprocess.run([BIN, *args], env=ENV, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def wait_ready(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st, j, _, _ = request("GET", "/health", key=None, timeout=3)
            if st == 200 and j and j.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def main():
    global PORT
    # ---- start the server as a daemon through the CLI ----
    rc, out, err = run_cli("serve", MODEL, "-d", "--name", "e2e", "--port", str(PORT), "--enable-admin", "--api-key", API_KEY, "--max-model-len", "2048", "--max-num-seqs", "8", "--threads", "4")
    check("serve -d starts", rc == 0, err)
    check("server becomes healthy", wait_ready(), "timeout waiting for /health")
    try:
        # ---- basic endpoints ----
        st, j, _, _ = request("GET", "/v1/models")
        check("/v1/models", st == 200 and j["object"] == "list" and j["data"][0]["id"] == MODEL_NAME, (st, j))
        st, j, _, _ = request("GET", "/v1/models", key=None)
        check("401 without api key", st == 401 and j and "error" in j, (st, j))
        st, j, _, _ = request("GET", "/version")
        check("/version", st == 200 and "version" in json.dumps(j), (st, j))

        # ---- chat completion (greedy, deterministic) ----
        body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Write one sentence about the sea."}], "max_tokens": 32, "temperature": 0}
        st, j1, _, hdr = request("POST", "/v1/chat/completions", body)
        ok = st == 200 and j1["object"] == "chat.completion" and j1["choices"][0]["message"]["role"] == "assistant"
        check("chat completion shape", ok, (st, j1))
        text1 = j1["choices"][0]["message"]["content"] if ok else ""
        check("chat content non-empty", bool(text1.strip()), j1)
        u = j1.get("usage", {})
        check("usage counts", u.get("prompt_tokens", 0) > 0 and u.get("completion_tokens", 0) > 0 and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], u)
        check("finish_reason valid", j1["choices"][0]["finish_reason"] in ("stop", "length"), j1["choices"][0])
        st, j2, _, _ = request("POST", "/v1/chat/completions", body)
        check("greedy is deterministic", j2["choices"][0]["message"]["content"] == text1, (text1, j2["choices"][0]["message"]["content"]))
        check("X-Request-Id header present", any(k.lower() == "x-request-id" for k in hdr), hdr)

        # ---- streaming equals non-streaming ----
        sbody = dict(body, stream=True, stream_options={"include_usage": True})
        st, events, done, raw = sse("/v1/chat/completions", sbody)
        check("stream 200 + [DONE]", st == 200 and done, raw)
        deltas = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e.get("choices"))
        check("stream first chunk has role", events and events[0]["choices"][0]["delta"].get("role") == "assistant", events[:1])
        check("stream text == non-stream text", deltas == text1, (deltas, text1))
        finishes = [e["choices"][0]["finish_reason"] for e in events if e.get("choices") and e["choices"][0].get("finish_reason")]
        check("stream has finish_reason", finishes and finishes[-1] in ("stop", "length"), finishes)
        usage_chunks = [e for e in events if e.get("usage") and not e.get("choices")]
        check("stream usage chunk (choices=[])", len(usage_chunks) == 1 and usage_chunks[0]["usage"]["completion_tokens"] > 0, usage_chunks)
        check("stream chunks are chat.completion.chunk", all(e["object"] == "chat.completion.chunk" for e in events), set(e["object"] for e in events))

        # ---- completions ----
        cb = {"model": MODEL_NAME, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, "echo": True, "logprobs": 2}
        st, j, _, _ = request("POST", "/v1/completions", cb)
        ok = st == 200 and j["object"] == "text_completion" and j["choices"][0]["text"].startswith("The capital of France is")
        check("completion + echo", ok, (st, j))
        lp = j["choices"][0].get("logprobs") if ok else None
        check("completion logprobs", lp and len(lp["tokens"]) == len(lp["token_logprobs"]) and len(lp["top_logprobs"]) == len(lp["tokens"]) and all(len(t) <= 2 for t in lp["top_logprobs"] if t), lp)
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": ["Hello", "Bonjour"], "max_tokens": 4, "temperature": 0})
        check("completion with prompt list -> 2 choices", st == 200 and len(j["choices"]) == 2 and [c["index"] for c in j["choices"]] == [0, 1], (st, j))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": [1, 2, 3, 4], "max_tokens": 2})
        check("completion with token ids", st == 200, (st, j))

        # ---- stop strings / max_tokens / logit_bias / min_tokens ----
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "The capital of France is", "max_tokens": 64, "temperature": 0, "stop": ["."]})
        t = j["choices"][0]["text"]
        check("stop string honoured", st == 200 and "." not in t and j["choices"][0]["finish_reason"] == "stop" and j["choices"][0].get("stop_reason") == ".", j["choices"][0])
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "Once upon a time", "max_tokens": 5, "temperature": 0, "ignore_eos": True})
        check("max_tokens -> length", st == 200 and j["choices"][0]["finish_reason"] == "length" and j["usage"]["completion_tokens"] == 5, j)
        st, tj, _, _ = request("POST", "/tokenize", {"prompt": " Paris"})
        paris = tj["tokens"][-1] if st == 200 else None
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "The capital of France is", "max_tokens": 1, "temperature": 0, "logit_bias": {str(paris): -100}})
        check("logit_bias suppresses token", st == 200 and j["choices"][0]["text"] != " Paris", j["choices"][0])
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "Hi", "max_tokens": 20, "min_tokens": 20, "temperature": 0})
        check("min_tokens", st == 200 and j["usage"]["completion_tokens"] == 20, j.get("usage"))

        # ---- structured outputs ----
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(body, response_format={"type": "json_object"}, max_tokens=64,
                                                                     messages=[{"role": "user", "content": "Give me a JSON object with keys name and age."}]))
        txt = j["choices"][0]["message"]["content"] if st == 200 else ""
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = None
        check("response_format json_object -> valid JSON object", st == 200 and isinstance(parsed, dict), (st, txt[:200]))
        schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"], "additionalProperties": False}
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(body, response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}, max_tokens=64,
                                                                     messages=[{"role": "user", "content": "Describe Paris."}]))
        txt = j["choices"][0]["message"]["content"] if st == 200 else ""
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = None
        check("response_format json_schema -> conforming JSON", st == 200 and isinstance(parsed, dict) and isinstance(parsed.get("city"), str) and isinstance(parsed.get("population"), int) and set(parsed) <= {"city", "population"}, (st, txt[:200]))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "Is the sky blue? Answer:", "max_tokens": 8, "temperature": 0, "guided_choice": [" yes", " no", " maybe"]})
        check("guided_choice", st == 200 and j["choices"][0]["text"] in (" yes", " no", " maybe"), (st, j))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "Count:", "max_tokens": 12, "temperature": 0, "grammar": 'root ::= " " [0-9]+ ("," [0-9]+)*'})
        t = j["choices"][0]["text"] if st == 200 else ""
        check("GBNF grammar honoured", st == 200 and t and all(c in " 0123456789," for c in t), (st, t))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "My phone number is", "max_tokens": 16, "temperature": 0, "guided_regex": " [0-9]{3}-[0-9]{4}"})
        t = j["choices"][0]["text"] if st == 200 else ""
        check("guided_regex honoured", st == 200 and re.fullmatch(r" [0-9]{3}-[0-9]{4}", t) is not None, (st, t))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "Answer:", "max_tokens": 8, "temperature": 0, "structured_outputs": {"regex": " (yes|no)"}})
        t = j["choices"][0]["text"] if st == 200 else ""
        check("structured_outputs.regex honoured", st == 200 and t in (" yes", " no"), (st, t))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "x", "max_tokens": 2, "guided_regex": "(unbalanced"})
        check("400 on invalid regex", st == 400, (st, j))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "x", "max_tokens": 2, "grammar": "root ::= (unterminated"})
        check("400 on invalid grammar", st == 400, (st, j))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "x", "max_tokens": 2, "guided_choice": ["a"], "grammar": "root ::= \"a\""})
        check("400 on two constraints", st == 400, (st, j))

        # ---- tool calls ----
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"], "additionalProperties": False}}},
                 {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"], "additionalProperties": False}}}]
        tbody = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "What is the weather in Paris?"}], "tools": tools, "tool_choice": "required", "parallel_tool_calls": False, "max_tokens": 96, "temperature": 0}
        st, j, _, _ = request("POST", "/v1/chat/completions", tbody)
        tc = j["choices"][0]["message"].get("tool_calls", []) if st == 200 else []
        ok = st == 200 and tc and tc[0]["type"] == "function" and tc[0]["function"]["name"] in ("get_weather", "get_time") and tc[0]["id"].startswith("call_")
        try:
            args = json.loads(tc[0]["function"]["arguments"]) if ok else None
        except Exception:
            args = None
        check("tool_choice=required -> tool_calls", ok and isinstance(args, dict) and j["choices"][0]["finish_reason"] == "tool_calls" and j["choices"][0]["message"]["content"] is None, (st, j.get("choices")))
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(tbody, tool_choice={"type": "function", "function": {"name": "get_time"}}))
        tc = j["choices"][0]["message"].get("tool_calls", []) if st == 200 else []
        check("tool_choice=named function", st == 200 and len(tc) == 1 and tc[0]["function"]["name"] == "get_time" and "tz" in json.loads(tc[0]["function"]["arguments"]), (st, j.get("choices")))
        st, events, done, raw = sse("/v1/chat/completions", dict(tbody, stream=True))
        deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
        tc_deltas = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        finishes = [e["choices"][0]["finish_reason"] for e in events if e.get("choices") and e["choices"][0].get("finish_reason")]
        check("streamed tool_calls delta + finish_reason", st == 200 and done and tc_deltas and tc_deltas[0][0]["index"] == 0 and tc_deltas[0][0]["function"]["name"] in ("get_weather", "get_time") and finishes and finishes[-1] == "tool_calls", (st, deltas[-3:], finishes))
        check("no raw tool-call text leaked into content", all("{" not in d.get("content", "") for d in deltas), deltas)
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(tbody, tool_choice="none", max_tokens=8))
        check("tool_choice=none -> plain content", st == 200 and j["choices"][0]["message"]["tool_calls"] == [] and isinstance(j["choices"][0]["message"]["content"], str), (st, j.get("choices")))
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(tbody, tool_choice={"type": "function", "function": {"name": "nope"}}))
        check("400 unknown forced tool", st == 400, (st, j))

        # ---- embeddings ----
        import base64, struct, math
        st, j, _, _ = request("POST", "/v1/embeddings", {"model": MODEL_NAME, "input": "The quick brown fox"})
        ok = st == 200 and j["object"] == "list" and len(j["data"]) == 1 and j["data"][0]["object"] == "embedding"
        vec = j["data"][0]["embedding"] if ok else []
        norm = math.sqrt(sum(x * x for x in vec)) if vec else 0
        check("/v1/embeddings single", ok and len(vec) == 576 and abs(norm - 1.0) < 1e-3 and j["usage"]["prompt_tokens"] > 0, (st, str(j)[:200]))
        st, j2, _, _ = request("POST", "/v1/embeddings", {"model": MODEL_NAME, "input": ["The quick brown fox", "Hello"]})
        check("/v1/embeddings batch + deterministic", st == 200 and len(j2["data"]) == 2 and [d["index"] for d in j2["data"]] == [0, 1] and all(abs(a - b) < 1e-5 for a, b in zip(j2["data"][0]["embedding"], vec)), (st, str(j2)[:200]))
        st, j3, _, _ = request("POST", "/v1/embeddings", {"model": MODEL_NAME, "input": "The quick brown fox", "encoding_format": "base64"})
        raw = base64.b64decode(j3["data"][0]["embedding"]) if st == 200 else b""
        dec = list(struct.unpack("<%df" % (len(raw) // 4), raw)) if raw else []
        check("/v1/embeddings base64", st == 200 and len(dec) == 576 and all(abs(a - b) < 1e-6 for a, b in zip(dec, vec)), (st, len(dec)))
        st, j4, _, _ = request("POST", "/v1/embeddings", {"model": MODEL_NAME, "input": "The quick brown fox", "dimensions": 64})
        v4 = j4["data"][0]["embedding"] if st == 200 else []
        check("/v1/embeddings dimensions=64", st == 200 and len(v4) == 64 and abs(math.sqrt(sum(x * x for x in v4)) - 1.0) < 1e-3, (st, len(v4)))
        st, j5, _, _ = request("POST", "/v1/embeddings", {"model": MODEL_NAME, "input": [[1, 2, 3, 4]], "pooling": "last"})
        check("/v1/embeddings token ids + last pooling", st == 200 and len(j5["data"][0]["embedding"]) == 576, (st, str(j5)[:120]))
        st, j6, _, _ = request("POST", "/v1/embeddings", {"model": MODEL_NAME, "input": ""})
        check("400 empty embedding input", st == 400, (st, j6))

        # ---- chat logprobs, n>1, seed ----
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(body, logprobs=True, top_logprobs=3, max_tokens=4))
        lp = j["choices"][0].get("logprobs") if st == 200 else None
        check("chat logprobs", lp and lp["content"] and all(len(c["top_logprobs"]) == 3 for c in lp["content"]), (st, lp))
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(body, n=2, max_tokens=6, temperature=1.0, seed=7))
        check("n=2 returns 2 choices", st == 200 and len(j["choices"]) == 2 and [c["index"] for c in j["choices"]] == [0, 1], (st, j))

        # ---- concurrency: batched results equal sequential ----
        # Greedy outputs must match token for token. Like tests/e2e/test-batching, a divergence is tolerated only
        # at a near tie (top-2 within 0.25 nats in both runs, each pick the other's runner-up): GPU backends choose
        # different kernels per batch shape and their rounding differs by up to ~0.1 nats.
        prompts = ["The capital of France is", "Once upon a time", "def add(a, b):", "Water boils at"]
        def tops(j):
            lp = j["choices"][0]["logprobs"]
            return [sorted(t.items(), key=lambda kv: -kv[1])[:2] for t in lp["top_logprobs"]], lp["tokens"]
        def same_or_near_tie(a, b):
            (ta, toka), (tb, tokb) = tops(a), tops(b)
            if toka == tokb:
                return True
            i = 0
            while i < min(len(toka), len(tokb)) and toka[i] == tokb[i]:
                i += 1
            if i >= min(len(ta), len(tb)) or len(ta[i]) < 2 or len(tb[i]) < 2:
                return False
            gap_a, gap_b = ta[i][0][1] - ta[i][1][1], tb[i][0][1] - tb[i][1][1]
            return 0 <= gap_a < 0.25 and 0 <= gap_b < 0.25 and ta[i][1][0] == tokb[i] and tb[i][1][0] == toka[i]
        seq = {}
        for p in prompts:
            st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": p, "max_tokens": 16, "temperature": 0, "logprobs": 2})
            seq[p] = j
        conc = {}
        def worker(p):
            st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": p, "max_tokens": 16, "temperature": 0, "logprobs": 2})
            conc[p] = j if st == 200 else {"error": (st, j)}
        ths = [threading.Thread(target=worker, args=(p,)) for p in prompts * 2]
        [t.start() for t in ths]
        [t.join() for t in ths]
        check("8 concurrent requests == sequential", all("error" not in conc[p] and same_or_near_tie(seq[p], conc[p]) for p in prompts),
              {p: (seq[p]["choices"][0]["text"], conc[p]["choices"][0]["text"] if "error" not in conc[p] else conc[p]) for p in prompts})

        # ---- prefix caching ----
        longp = "This is a long shared prefix. " * 12 + "The capital of France is"
        st, j1, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": longp, "max_tokens": 4, "temperature": 0})
        st, j2, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": longp, "max_tokens": 4, "temperature": 0})
        cached = j2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        check("prefix cache hit reported", cached > 0 and same_or_near_tie(j1, j2), (j1.get("usage"), j2.get("usage"), j1["choices"][0]["text"], j2["choices"][0]["text"]))

        # ---- tokenize / detokenize ----
        st, tj, _, _ = request("POST", "/tokenize", {"prompt": "Hello, world! 안녕하세요"})
        check("/tokenize", st == 200 and tj["count"] == len(tj["tokens"]) > 0, (st, tj))
        st, dj, _, _ = request("POST", "/detokenize", {"tokens": tj["tokens"]})
        check("/detokenize round trip", st == 200 and dj.get("prompt") == "Hello, world! 안녕하세요", (st, dj))
        st, tj2, _, _ = request("POST", "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})
        check("/tokenize with messages", st == 200 and tj2["count"] > 2, (st, tj2))

        # ---- errors ----
        st, j, _, _ = request("POST", "/v1/chat/completions", "{not json")
        check("400 on bad JSON", st == 400 and j["error"]["type"] == "invalid_request_error", (st, j))
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(body, temperature="hot"))
        check("400 on wrong field type", st == 400 and j["error"].get("param") == "temperature", (st, j))
        st, j, _, _ = request("POST", "/v1/chat/completions", dict(body, model="nope"))
        check("404 unknown model", st == 404 and j["error"]["type"] == "not_found_error", (st, j))
        st, j, _, _ = request("POST", "/v1/chat/completions", {"model": MODEL_NAME, "messages": []})
        check("400 empty messages", st == 400, (st, j))
        st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "x " * 3000, "max_tokens": 1})
        check("400 prompt too long", st == 400 and ("context length" in json.dumps(j) or "max-model-len" in json.dumps(j)), (st, j))
        st, j, _, _ = request("GET", "/nope")
        check("404 unknown route", st == 404, (st, j))

        # ---- metrics / admin ----
        st, _, txt, _ = request("GET", "/metrics")
        check("/metrics prometheus", st == 200 and "iian:generation_tokens_total" in txt and "iian:num_requests_running" in txt, txt[:200])
        st, j, _, _ = request("GET", "/v1/engine/stats")
        check("/v1/engine/stats (admin)", st == 200 and j["stats"]["requests_finished"] > 0, (st, j))
        st, j, _, _ = request("GET", "/props")
        check("/props", st == 200 and "chat_template" in json.dumps(j), (st, j))

        # ---- CLI management ----
        rc, out, err = run_cli("ps")
        check("iian ps lists server", rc == 0 and "e2e" in out and "running" in out, out + err)
        rc, out, err = run_cli("status", "e2e")
        check("iian status", rc == 0 and "health" in out and "requests" in out, out + err)
        rc, out, err = run_cli("logs", "e2e", "-n", "3")
        check("iian logs", rc == 0 and len(out.strip().splitlines()) >= 1, out + err)
        rc, out, err = run_cli("run", "e2e", "-p", "Say hi", "-n", "8", "--temp", "0", timeout=120)
        check("iian run against server", rc == 0 and out.strip(), out + err)
    finally:
        rc, out, err = run_cli("stop", "e2e")
        check("iian stop", rc == 0 and "stopped" in out, out + err)
        rc, out, err = run_cli("ps")
        check("ps empty after stop", rc == 0 and "e2e" not in out, out)

    # ---- multi-model router: two models, max 1 loaded, lazy ----
    MODEL2 = os.path.join(os.path.dirname(MODEL), "SmolLM2-135M-Instruct-Q8_0.gguf")
    if os.path.exists(MODEL2):
        NAME2 = os.path.basename(MODEL2)[:-5]
        PORT = free_port()
        rc, out, err = run_cli("serve", MODEL, MODEL2, "-d", "--name", "e2e-multi", "--port", str(PORT), "--enable-admin", "--api-key", API_KEY,
                               "--max-model-len", "1024", "--max-num-seqs", "4", "--threads", "4", "--models-max", "1", "--lazy")
        check("multi-model serve -d (lazy)", rc == 0, err)
        check("router healthy before any load", wait_ready(60), "timeout")
        try:
            st, j, _, _ = request("GET", "/v1/models")
            ids = {d["id"]: d["status"] for d in j["data"]} if st == 200 else {}
            check("/v1/models lists both (unloaded)", st == 200 and ids == {MODEL_NAME: "unloaded", NAME2: "unloaded"}, (st, ids))
            st, j, _, _ = request("POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 2})
            check("400 without model when several are served", st == 400, (st, j))
            st, j, _, _ = request("POST", "/v1/completions", {"model": MODEL_NAME, "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0}, timeout=300)
            check("lazy-loads model A on first request", st == 200 and j["model"] == MODEL_NAME, (st, j))
            st, j, _, _ = request("GET", "/v1/models")
            ids = {d["id"]: d["status"] for d in j["data"]}
            check("A ready, B unloaded", ids == {MODEL_NAME: "ready", NAME2: "unloaded"}, ids)
            st, j, _, _ = request("POST", "/v1/completions", {"prompt": "Hello", "max_tokens": 2, "temperature": 0})
            check("single loaded model used when 'model' omitted", st == 200 and j["model"] == MODEL_NAME, (st, j))
            st, jb, _, _ = request("POST", "/v1/chat/completions", {"model": NAME2, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0}, timeout=300)
            check("lazy-loads model B", st == 200 and jb["model"] == NAME2, (st, jb))
            st, j, _, _ = request("GET", "/v1/models")
            ids = {d["id"]: d["status"] for d in j["data"]}
            check("LRU unloaded A (models-max 1)", ids == {MODEL_NAME: "unloaded", NAME2: "ready"}, ids)
            st, j, _, _ = request("POST", "/v1/completions", {"model": "nope", "prompt": "x", "max_tokens": 1})
            check("404 unknown model on router", st == 404, (st, j))
            st, j, _, _ = request("POST", "/v1/models/load", {"model": MODEL_NAME}, timeout=300)
            check("admin load A", st == 200 and j["status"] == "ready", (st, j))
            st, j, _, _ = request("GET", "/v1/models")
            ids = {d["id"]: d["status"] for d in j["data"]}
            check("load A evicted B", ids == {MODEL_NAME: "ready", NAME2: "unloaded"}, ids)
            st, j, _, _ = request("POST", "/v1/models/unload", {"model": MODEL_NAME})
            check("admin unload A", st == 200 and j["status"] == "unloaded", (st, j))
            st, _, txt, _ = request("GET", "/metrics")
            check("/metrics on router", st == 200 and "iian:models_loaded" in txt, txt[:100])
            rc, out, err = run_cli("ps")
            check("ps shows both model names", rc == 0 and MODEL_NAME in out and NAME2 in out, out)
        finally:
            rc, out, err = run_cli("stop", "e2e-multi")
            check("stop multi-model server", rc == 0, out + err)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n%d checks, %d failed" % (len(results), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
