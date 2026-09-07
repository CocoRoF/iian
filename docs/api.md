# HTTP API

OpenAI-compatible; extensions follow vLLM's field names so existing clients keep working.
All JSON. Authentication (if `--api-key` is set): `Authorization: Bearer <key>` or `X-Api-Key: <key>`;
`/health`, `/ready`, `/version` are exempt. Every response carries `X-Request-Id` (echoed if the client sent one).

## POST /v1/chat/completions
Request fields:
* OpenAI: `model` (must match the served name, or omit), `messages` (string content or `[{"type":"text","text":...}]` parts),
  `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `n` (1..16), `stop` (string or array),
  `stream`, `stream_options.include_usage`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`
  (`{"<token id>": bias}`), `logprobs` + `top_logprobs`, `user`, `response_format` (`{"type":"text"}` only, others 400),
  `tools` / `tool_choice` (`none` | `auto` | `required` | `{"type":"function","function":{"name":..}}`), `parallel_tool_calls`.
* Extensions: `top_k`, `min_p`, `repetition_penalty`, `stop_token_ids`, `include_stop_str_in_output`, `ignore_eos`,
  `min_tokens`, `skip_special_tokens`, `add_generation_prompt`, `chat_template_kwargs` (object passed to the template,
  e.g. `{"enable_thinking": false}`), `chat_template` (only with `--allow-request-chat-template`), `priority`
  (lower = sooner under `--scheduling-policy priority`), `cache_salt` (isolates prefix-cache entries), `request_id`.

Response: `{"id":"chatcmpl-...","object":"chat.completion","created":..,"model":..,
"choices":[{"index":0,"message":{"role":"assistant","content":".."},"logprobs":null|{"content":[...]},
"finish_reason":"stop"|"length","stop_reason":null|"<stop string>"|<token id>}],
"usage":{"prompt_tokens":..,"completion_tokens":..,"total_tokens":..,"prompt_tokens_details":{"cached_tokens":..}}}`.

Streaming (`text/event-stream`): `data: {chat.completion.chunk}` frames — first chunk has `delta.role`,
then `delta.content` deltas, the last content chunk carries `finish_reason`; with `include_usage` one more
chunk with `"choices": []` and `usage`; then `data: [DONE]`. Errors after the stream started are sent as a
`data: {"error": {...}}` frame. Disconnecting the client aborts generation.

### Tool calling
Tools are rendered by the model's chat template. The response format is detected from the template
(`hermes` `<tool_call>{json}</tool_call>` for Qwen/Hermes-style models, `llama3` JSON `{"name","parameters"}`,
`mistral` `[TOOL_CALLS][...]`, or a generic JSON object). Generated calls are returned as
`message.tool_calls[{id:"call_..", type:"function", function:{name, arguments}}]` with `finish_reason:
"tool_calls"` and `content: null` when the message is only calls. `tool_choice: "required"` or a named function
forces a well-formed call through the grammar sampler (built from each tool's `parameters` schema), so even
small models produce valid calls. Streaming holds back text from the point a call may start and emits one
`delta.tool_calls` chunk per call at the end, followed by `finish_reason: "tool_calls"`.

### Structured outputs (all generation endpoints)
Exactly one of: `response_format: {"type":"json_object"}` (any JSON object), `response_format:
{"type":"json_schema","json_schema":{"name":..,"schema":{...}}}` (JSON conforming to the schema),
`guided_json` (schema object or string), `guided_choice` (array of strings: the output is one of them),
`guided_grammar` / `grammar` (a GBNF grammar in llama.cpp syntax, root rule `root`), or a vLLM-style
`structured_outputs: {json|grammar|choice}` object. Schemas are converted to GBNF (llama.cpp's converter) and
enforced token-by-token by the grammar sampler; an invalid grammar or schema is a 400. `regex` is not supported yet.

## POST /v1/completions
`prompt` (string, array of strings, or array of token ids), `suffix` (400: not supported), `echo`,
`logprobs` (int: top-k per token; response has `tokens`, `token_logprobs`, `top_logprobs`, `text_offset`),
`max_tokens` (default 16), `best_of` (must equal `n`), plus the same sampling fields and extensions as chat.
Response object is `text_completion`; streaming frames are `text_completion` chunks with `text` deltas.

## POST /v1/embeddings
`input`: string, array of strings, array of token ids, or array of token-id arrays (up to 256 inputs, batched
by the engine); `encoding_format`: `float` (default) | `base64` (little-endian float32); `dimensions` truncates
(and re-normalizes); extensions: `pooling` (`mean` default | `last` | `cls`), `normalize` (default true),
`add_special_tokens`. Response: `{"object":"list","data":[{"object":"embedding","index":i,"embedding":[...]}],
"model":..,"usage":{"prompt_tokens","total_tokens"}}`. Works with any supported decoder model (the pooled
final hidden state); dedicated embedding architectures (BERT-style) are not yet supported.

## Other endpoints
| Endpoint | Purpose |
|---|---|
| `GET /v1/models`, `GET /v1/models/{id}` | one card per served model with `status` (`ready`/`loading`/`unloaded`/`failed`), `max_model_len`, `meta{arch, ftype, n_params}`. With several models the `model` field is required in requests (400 otherwise, unless exactly one is loaded); an unknown name is 404; an unloaded model is loaded on first use. |
| `POST /v1/models/load`, `POST /v1/models/unload` (admin) | `{"model": name}`; unload is refused (409) while requests are in flight; loading beyond `--models-max` unloads the least recently used idle model |
| `POST /tokenize` | `{"prompt": str}` or `{"messages": [...]}` (chat template applied), optional `add_special_tokens` → `{"tokens","count","max_model_len"}` |
| `POST /detokenize` | `{"tokens": [...]}` → `{"prompt": str}` |
| `GET /health`, `/ready` | `{"status":"ok"|"loading"}` (503 while loading) |
| `GET /version` | version + build info |
| `GET /metrics` | Prometheus text: `iian:num_requests_running/waiting`, `iian:kv_cache_usage_perc`, `iian:prompt_tokens_total`, `iian:generation_tokens_total`, `iian:request_success_total{finished_reason=..}`, `iian:prefix_cache_{queries,hits}_total`, `iian:num_preemptions_total`, latency summaries |
| `GET /props` | served model, chat template source and origin, bos/eos, context size, build info |
| `GET /v1/engine/stats` (admin) | JSON of `EngineStats`, HTTP counters and the engine config |
| `POST /v1/engine/abort` (admin) | `{"request_id": "chatcmpl-..."}` aborts an in-flight request |

Admin endpoints return 403 unless the server runs with `--enable-admin`.

## Errors
```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "temperature", "code": null}}
```
Types: `invalid_request_error` (400), `authentication_error` (401), `permission_error` (403),
`not_found_error` (404), `service_unavailable` (503), `internal_error` (500). Messages say what to change
(e.g. a too-long prompt reports the model's context length and suggests `--max-model-len`).
