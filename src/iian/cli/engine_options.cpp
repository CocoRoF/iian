#include "engine_options.h"

#include "paths.h"

#include "iian/gguf.h"
#include "iian/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <thread>

namespace iian::cli {

void add_model_flags(ArgParser & p) {
    p.group("Model");
    p.add("-t,--threads", "CPU threads for generation (default: half the cores)", "N");
    p.add("--threads-batch", "CPU threads for prompt processing (default: same as --threads)", "N");
    p.add("-ngl,--n-gpu-layers", "Layers to offload to the GPU (-1 = all when a GPU backend is available)", "N", "-1");
    p.add_repeatable("--device", "Backend device(s) to place layers on, in order (e.g. CUDA0, Vulkan0, CPU); repeatable", "NAME");
    p.add("--no-mmap", "Read the model into memory instead of mmap-ing it");
    p.add("--mlock", "Lock model memory (prevent swapping)");
    p.add("--no-repack", "Do not repack CPU weights into ggml's optimised layouts (keeps weights mmap-shared, slower matmul)");
}

void add_engine_flags(ArgParser & p) {
    p.group("Engine / KV cache");
    p.add("-c,--max-model-len", "Context length per request, prompt + output (0 = model's training context)", "N", "0");
    p.add("--kv-cache-tokens", "Total KV cache capacity in tokens (0 = max-model-len x min(max-num-seqs, 4))", "N", "0");
    p.add("--kv-cache-gb", "Cap the KV cache at this many GiB (overrides --kv-cache-tokens when smaller)", "G");
    p.add("--kv-dtype", "KV cache element type: f16|bf16|f32|q8_0|q4_0|q4_1|q5_0|q5_1", "T", "f16");
    p.add("--block-size", "KV cache block size in tokens (prefix-cache granularity)", "N", "16");
    p.add("--no-prefix-caching", "Disable automatic prefix caching");
    p.add("--no-flash-attn", "Disable flash attention");
    p.add("--attention", "auto | masked | paged (paged = iian kernel, CPU only; auto picks per step)", "MODE", "auto");
    p.add("--seed", "Base RNG seed for sampling (0 = per-request random)", "N", "0");
    p.add("--spec-ngram", "Speculative decoding via prompt lookup: draft tokens per step (0 = off; try 4)", "N", "0");
    p.add("--spec-draft", "Speculative decoding with a draft model (GGUF path / cached name; same tokenizer as the target)", "MODEL");
    p.add("--spec-draft-n", "Draft tokens per step with --spec-draft", "N", "5");
    p.group("Scheduler");
    p.add("--max-num-seqs", "Max concurrently running requests", "N", "32");
    p.add("--max-num-batched-tokens", "Token budget per engine step (chunked-prefill granularity)", "N", "1024");
    p.add("--no-chunked-prefill", "Process prompts whole instead of in chunks");
    p.add("--long-prefill-token-threshold", "Cap a single request's prefill chunk (0 = off)", "N", "0");
    p.add("--scheduling-policy", "fcfs | priority (honours the request 'priority' field)", "P", "fcfs");
}

void add_logging_flags(ArgParser & p) {
    p.group("Logging");
    p.add("--log-level", "trace|debug|info|warn|error|off", "L", "info");
    p.add("--log-format", "pretty | json", "F", "pretty");
    p.add("--log-file", "Also append logs to this file", "PATH");
    p.add("--no-color", "Disable ANSI colours in logs");
    p.add("-v,--verbose", "Shortcut for --log-level debug");
    p.add("-q,--quiet", "Shortcut for --log-level warn");
}

void add_sampling_flags(ArgParser & p) {
    p.group("Sampling");
    p.add("-n,--max-tokens", "Max tokens to generate (-1 = until the context is full)", "N", "-1");
    p.add("--temp", "Temperature (0 = greedy)", "T", "0.7");
    p.add("--top-p", "Nucleus sampling", "P", "1.0");
    p.add("--top-k", "Top-k sampling (0 = off)", "K", "0");
    p.add("--min-p", "Min-p sampling", "P", "0");
    p.add("--repeat-penalty", "Repetition penalty (1 = off)", "R", "1.0");
    p.add("--presence-penalty", "Presence penalty [-2, 2]", "X", "0");
    p.add("--frequency-penalty", "Frequency penalty [-2, 2]", "X", "0");
    p.add("--seed", "Sampling seed", "N");
    p.add_repeatable("--stop", "Stop string (repeatable)", "S");
    p.add("--grammar", "Constrain output with a GBNF grammar (file path or literal)", "GBNF");
    p.add("--json-schema", "Constrain output to a JSON schema (file path or literal; '{}' = any JSON object)", "SCHEMA");
    p.add("--regex", "Constrain output to a regular expression (anchored; converted to a grammar)", "RE");
}

DeviceConfig device_config_from_args(const ArgParser & p) {
    DeviceConfig d;
    if (p.has("threads")) d.n_threads = (int) p.get_int("threads");
    if (p.has("threads-batch")) d.n_threads_batch = (int) p.get_int("threads-batch");
    if (p.has("n-gpu-layers")) d.n_gpu_layers = (int) p.get_int("n-gpu-layers");
    d.devices = p.get_all("device");
    d.use_mmap = !p.get_bool("no-mmap");
    d.use_mlock = p.get_bool("mlock");
    d.use_extra_bufts = !p.get_bool("no-repack");
    return d;
}

EngineConfig engine_config_from_args(const ArgParser & p) {
    EngineConfig e;
    e.device = device_config_from_args(p);
    e.n_threads = e.device.n_threads;
    e.n_threads_batch = e.device.n_threads_batch;
    e.max_model_len = (uint32_t) p.get_int("max-model-len");
    e.kv_cache_tokens = (uint32_t) p.get_int("kv-cache-tokens");
    if (p.has("kv-cache-gb")) e.kv_cache_bytes = (size_t) (p.get_double("kv-cache-gb") * 1073741824.0);
    e.kv_dtype = p.get("kv-dtype");
    e.block_size = (uint32_t) p.get_int("block-size");
    e.enable_prefix_caching = !p.get_bool("no-prefix-caching");
    e.flash_attn = !p.get_bool("no-flash-attn");
    e.attention = p.get("attention");
    e.seed = (uint64_t) p.get_int("seed");
    e.spec_ngram = (uint32_t) p.get_int("spec-ngram");
    if (p.has("spec-draft")) e.spec_draft_model = p.get("spec-draft");
    e.spec_draft_n = (uint32_t) p.get_int("spec-draft-n");
    e.sched.max_num_seqs = (uint32_t) p.get_int("max-num-seqs");
    e.sched.max_num_batched_tokens = (uint32_t) p.get_int("max-num-batched-tokens");
    e.sched.enable_chunked_prefill = !p.get_bool("no-chunked-prefill");
    e.sched.long_prefill_token_threshold = (uint32_t) p.get_int("long-prefill-token-threshold");
    e.sched.policy = p.get("scheduling-policy") == "priority" ? SchedulingPolicy::PRIORITY : SchedulingPolicy::FCFS;
    e.sched.max_model_len = e.max_model_len;
    return e;
}

bool setup_logging(const ArgParser & p, std::string & err) {
    Logger & lg = Logger::instance();
    std::string lvl = p.get("log-level");
    if (p.get_bool("verbose")) lvl = "debug";
    if (p.get_bool("quiet")) lvl = "warn";
    LogLevel l;
    if (!Logger::parse_level(lvl, l)) { err = "unknown --log-level '" + lvl + "' (use trace|debug|info|warn|error|off)"; return false; }
    lg.set_level(l);
    std::string fmt = p.get("log-format");
    if (fmt == "json") lg.set_format(LogFormat::JSON);
    else if (fmt == "pretty" || fmt.empty()) lg.set_format(LogFormat::PRETTY);
    else { err = "unknown --log-format '" + fmt + "' (use pretty|json)"; return false; }
    if (p.get_bool("no-color") || getenv("NO_COLOR") || !is_tty(stderr)) lg.set_color(false);
    if (p.has("log-file")) {
        std::string f = expand_user(p.get("log-file"));
        lg.set_file(f);
    }
    return true;
}

bool load_config_file(ArgParser & p, const std::string & path, std::string & err) {
    auto s = read_file(expand_user(path));
    if (!s) { err = "cannot read --config file " + path; return false; }
    nlohmann::json j = nlohmann::json::parse(*s, nullptr, false);
    if (j.is_discarded() || !j.is_object()) { err = "--config file " + path + " must contain a JSON object of flag -> value (e.g. {\"port\": 9931, \"max-model-len\": 4096})"; return false; }
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string k = it.key();
        std::replace(k.begin(), k.end(), '_', '-');
        if (!p.find(k)) {
            std::string sug = p.suggest(k);
            err = "unknown option '" + it.key() + "' in " + path + (sug.empty() ? "" : " (did you mean '" + sug + "'?)");
            return false;
        }
    }
    p.set_config(j);
    return true;
}

std::string model_banner(const Model & m) {
    const HParams & hp = m.hparams();
    char buf[512];
    snprintf(buf, sizeof buf, "%s | %s | %s params | %s | n_layer=%u n_embd=%u n_ctx_train=%u vocab=%u",
             hp.name.empty() ? strip_gguf_ext(basename_of(m.path())).c_str() : hp.name.c_str(), hp.arch.c_str(), m.size_label().c_str(),
             m.ftype_name().c_str(), hp.n_layer, hp.n_embd, hp.n_ctx_train, hp.n_vocab);
    return buf;
}

} // namespace iian::cli
