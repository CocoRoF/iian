# `iian` command line

One binary, subcommands grouped by purpose. Every command has `--help` with sectioned options; unknown
flags get a "did you mean" suggestion. Values resolve as **CLI > `IIAN_<FLAG>` env > `--config file.json` > default**.
State lives under `$IIAN_HOME` (default `~/.iian`): `run/` (server records), `logs/`, `models/`.

## Serve
```
iian serve <model> [<model> ...] [options]
```
Each `<model>` is a `.gguf` path, `hf:org/repo[:quant]` (downloaded into `~/.iian/models` on first use), or a
cached model name (`iian ls`). Several models can be served by one process: requests pick one with the
`model` field (required when more than one is loaded). `--models-dir DIR` adds every GGUF in a directory,
loaded on first use; `--models-max N` keeps at most N loaded and unloads the least recently used; `--lazy`
defers loading of the listed models too. `/v1/models` shows `ready | loading | unloaded | failed` per model,
and with `--enable-admin` `POST /v1/models/load|unload {"model": name}` control them.

| Group | Flags |
|---|---|
| Multi-model | `--models-dir DIR` `--models-max N` (0 = unlimited) `--lazy` |
| Server | `--host` (127.0.0.1) `-p/--port` (9931) `--unix-socket PATH` `--api-key KEY` `--served-model-name NAME` `--chat-template FILE` `--allow-request-chat-template` `--enable-admin` `--threads-http N` `-d/--detach` `--name NAME` `--config FILE` |
| Model | `-t/--threads N` `--threads-batch N` `-ngl/--n-gpu-layers N` `--device NAME` (repeatable) `--no-mmap` `--mlock` |
| Engine / KV cache | `--spec-ngram N` (prompt-lookup speculative decoding, 0 = off) `--spec-draft MODEL --spec-draft-n N` (draft-model speculation; same tokenizer) `--attention auto|masked|paged` `--no-repack` `-c/--max-model-len N` (default: min(training ctx, 8192)) `--kv-cache-tokens N` `--kv-cache-gb G` `--kv-dtype f16|bf16|f32|q8_0|q4_0|q4_1|q5_0|q5_1` `--block-size N` (16) `--no-prefix-caching` `--no-flash-attn` `--seed N` |
| Scheduler | `--max-num-seqs N` (32) `--max-num-batched-tokens N` (1024) `--no-chunked-prefill` `--long-prefill-token-threshold N` `--scheduling-policy fcfs|priority` |
| Logging | `--log-level trace|debug|info|warn|error|off` `--log-format pretty|json` `--log-file PATH` `--no-color` `-v` `-q` |

Foreground: Ctrl-C shuts down gracefully. Daemon (`-d`): double-fork, logs to `~/.iian/logs/<name>.log`,
record in `~/.iian/run/<name>.json` (removed on exit). Foreground servers write the record too, so `ps`
sees them. The KV cache is auto-sized to `max_model_len x min(max_num_seqs, 4)` cells, capped at half of the
free memory of the device holding it; if even one context does not fit, `serve` fails with a message that
says which flag to change.

## Manage
```
iian ps [--prune] [--json]         NAME PID MODEL ENDPOINT STATUS UPTIME REQS(run/wait)
iian stop <name|pid|all> [--timeout 10]
iian restart <name>
iian logs <name> [-f] [-n 100]
iian status <name|http://host:port> [--json]
```
`status` and the REQS column read `/v1/engine/stats`, which needs `--enable-admin` on the server.

## Use
```
iian run <model|server> [-p TEXT] [-s SYSTEM] [sampling flags] [--no-stats]
iian complete <model|server> -p TEXT [-n N] [--temp T] [--json]
iian embed <model|server> [text ...] [--pooling mean|last|cls] [--similarity] [--json]
```
If the argument names a running server, requests go over HTTP (streaming); otherwise the model is loaded
in-process (no server needed). `run` is an interactive chat with `/exit`, `/clear`, `/regen`,
`/system <text>`; `-p` makes it single-turn. Sampling flags: `-n/--max-tokens`, `--temp`, `--top-p`,
`--top-k`, `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`, `--seed`, `--stop`
(repeatable), `--grammar GBNF|file`, `--json-schema SCHEMA|file` (structured output; `'{}'` = any JSON).

## Models
```
iian pull hf:org/repo[:quant] | <url>   [--dir DIR] [--force] [-q]
iian ls [--dir DIR] [--json]
iian rm <name|path> [-y]
iian info <model> [--full] [--all] [--tensors] [--json]
```
`pull` lists the repo's GGUF files through the Hugging Face API, picks the requested quant (or Q4_K_M >
Q8_0 > F16), downloads with a progress bar (TLS via OpenSSL, `curl` fallback) and stores
`~/.iian/models/<org>__<repo>__<file>`. `info` reads only the metadata (no weights): architecture,
hyper-parameters, tokenizer, chat template, tensor types; `--all` dumps every GGUF key.

## Other
```
iian bench <model> [--concurrency 1,4,16] [--prompt-tokens 256] [--gen-tokens 128] [--json]
iian version
```
`bench` submits N concurrent requests with distinct random prompts (so the prefix cache cannot help) and
prints prompt/generation/total tokens per second, mean time-to-first-token and per-token latency.

## Config file
`--config file.json` takes a flat object of flag names to values:
```json
{"port": 8000, "max-model-len": 4096, "max-num-seqs": 64, "kv-dtype": "q8_0", "enable-admin": true}
```
Environment overrides use the flag name upper-cased with `-` -> `_`: `IIAN_PORT=8000`, `IIAN_LOG_FORMAT=json`.
`IIAN_DEVICES=CUDA1` (or `CUDA0,CUDA1`) restricts model placement when `--device` is not given.
