// Prometheus text exposition + JSON stats view.
#include "server_context.h"

#include "iian/version.h"

#include <cstdio>
#include <string>

namespace iian::server {

namespace {

struct MetricWriter {
    std::string out;
    std::string labels;
    void gauge(const char * name, const char * help, double v)   { emit(name, help, "gauge", v); }
    void counter(const char * name, const char * help, double v) { emit(name, help, "counter", v); }
    void header(const char * name, const char * help, const char * type) {
        out += "# HELP "; out += name; out += ' '; out += help; out += '\n';
        out += "# TYPE "; out += name; out += ' '; out += type; out += '\n';
    }
    void sample(const char * name, const std::string & extra_labels, double v) {
        char buf[64];
        snprintf(buf, sizeof buf, "%.17g", v);
        std::string lbl = labels;
        if (!extra_labels.empty()) { if (!lbl.empty()) lbl += ','; lbl += extra_labels; }
        out += name;
        if (!lbl.empty()) { out += '{'; out += lbl; out += '}'; }
        out += ' '; out += buf; out += '\n';
    }
    void emit(const char * name, const char * help, const char * type, double v) { header(name, help, type); sample(name, "", v); }
};

std::string escape_label(const std::string & s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}

} // namespace

static void render_one(MetricWriter & w, const ServerContext & ctx) {
    w.labels = "model_name=\"" + escape_label(ctx.model.name) + "\"";
    const Engine * e = ctx.engine.load();
    EngineStats s;
    if (e) s = e->stats();
    const Counters & c = ctx.counters;

    w.gauge("iian:num_requests_running", "Number of requests currently running on the engine.", (double) s.num_running);
    w.gauge("iian:num_requests_waiting", "Number of requests waiting to be processed.", (double) s.num_waiting);
    w.gauge("iian:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", s.kv_usage);
    w.counter("iian:prompt_tokens_total", "Number of prefill tokens processed.", (double) s.prompt_tokens);
    w.counter("iian:generation_tokens_total", "Number of generation tokens processed.", (double) s.generation_tokens);
    w.counter("iian:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried tokens.", (double) s.prefix_cache_queries);
    w.counter("iian:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens.", (double) s.prefix_cache_hits);
    w.counter("iian:num_preemptions_total", "Cumulative number of preemptions from the engine.", (double) s.preemptions);
    w.counter("iian:engine_steps_total", "Cumulative number of engine steps.", (double) s.steps);
    w.counter("iian:requests_total", "Requests submitted to the engine.", (double) s.requests_total);
    w.header("iian:request_success_total", "Count of successfully processed requests, by finish reason.", "counter");
    w.sample("iian:request_success_total", "finished_reason=\"stop\"", (double) c.finished_stop.load());
    w.sample("iian:request_success_total", "finished_reason=\"length\"", (double) c.finished_length.load());
    w.sample("iian:request_success_total", "finished_reason=\"abort\"", (double) c.finished_abort.load());
    w.sample("iian:request_success_total", "finished_reason=\"error\"", (double) c.finished_error.load());
    w.gauge("iian:prompt_tokens_per_second", "Prefill throughput over the last stats window.", s.prompt_tps);
    w.gauge("iian:generation_tokens_per_second", "Generation throughput over the last stats window.", s.generation_tps);
    w.header("iian:time_to_first_token_seconds", "Time to first token (summary: sum/count).", "summary");
    w.sample("iian:time_to_first_token_seconds_sum", "", s.ttft_ms_sum / 1000.0);
    w.sample("iian:time_to_first_token_seconds_count", "", (double) s.ttft_samples);
    w.header("iian:e2e_request_latency_seconds", "End-to-end request latency (summary: sum/count).", "summary");
    w.sample("iian:e2e_request_latency_seconds_sum", "", (double) c.e2e_latency_us_sum.load() / 1e6);
    w.sample("iian:e2e_request_latency_seconds_count", "", (double) c.e2e_latency_count.load());
    w.counter("iian:http_requests_total", "HTTP requests served.", (double) c.http_requests.load());
    w.counter("iian:http_errors_total", "HTTP requests answered with an error status.", (double) c.http_errors.load());
    w.gauge("iian:http_requests_in_flight", "HTTP generation requests currently in flight.", (double) c.in_flight.load());
    w.gauge("iian:ready", "1 when the model is loaded and serving.", e ? 1.0 : 0.0);
}

std::string render_metrics(const ServerContext & ctx) {
    MetricWriter w;
    render_one(w, ctx);
    w.labels.clear();
    w.gauge("iian:uptime_seconds", "Server uptime.", ctx.uptime_s());
    return w.out;
}

std::string render_metrics_all(const ServerContext & router, const std::vector<const ServerContext *> & models) {
    // one block per loaded model (each sample carries model_name); router-level totals at the end
    MetricWriter w;
    for (const ServerContext * m : models) render_one(w, *m);
    w.labels.clear();
    w.gauge("iian:models_loaded", "Number of models currently loaded.", (double) models.size());
    w.counter("iian:http_requests_received_total", "HTTP requests received by the server (all models).", (double) router.counters.http_requests.load());
    w.counter("iian:http_errors_total_all", "HTTP requests answered with an error status (all models).", (double) router.counters.http_errors.load());
    w.gauge("iian:uptime_seconds", "Server uptime.", router.uptime_s());
    return w.out;
}

nlohmann::json engine_stats_json(const ServerContext & ctx) {
    nlohmann::json j;
    j["model"] = ctx.model.name;
    j["ready"] = ctx.ready();
    j["uptime_s"] = ctx.uptime_s();
    j["version"] = iian::version();
    const Engine * e = ctx.engine.load();
    if (!e) return j;
    EngineStats s = e->stats();
    j["stats"] = {
        {"requests_total", s.requests_total}, {"requests_finished", s.requests_finished}, {"requests_aborted", s.requests_aborted},
        {"prompt_tokens", s.prompt_tokens}, {"generation_tokens", s.generation_tokens}, {"cached_prompt_tokens", s.cached_prompt_tokens},
        {"steps", s.steps}, {"preemptions", s.preemptions}, {"spec_drafted", s.spec_drafted}, {"spec_accepted", s.spec_accepted},
        {"num_running", s.num_running}, {"num_waiting", s.num_waiting}, {"kv_usage", s.kv_usage},
        {"prefix_cache_queries", s.prefix_cache_queries}, {"prefix_cache_hits", s.prefix_cache_hits},
        {"prompt_tps", s.prompt_tps}, {"generation_tps", s.generation_tps},
        {"ttft_ms_avg", s.ttft_ms_avg}, {"tpot_ms_avg", s.tpot_ms_avg}, {"ttft_samples", s.ttft_samples},
    };
    const Counters & c = ctx.counters;
    j["http"] = {
        {"requests", c.http_requests.load()}, {"errors", c.http_errors.load()}, {"in_flight", c.in_flight.load()},
        {"finished", {{"stop", c.finished_stop.load()}, {"length", c.finished_length.load()}, {"abort", c.finished_abort.load()}, {"error", c.finished_error.load()}}},
        {"streamed", c.requests_streamed.load()},
        {"e2e_latency_ms_avg", c.e2e_latency_count.load() ? (double) c.e2e_latency_us_sum.load() / 1000.0 / (double) c.e2e_latency_count.load() : 0.0},
    };
    const EngineConfig & cfg = e->config();
    const PagedKVCache & kv = e->kv();
    j["config"] = {
        {"max_model_len", e->max_model_len()}, {"kv_cache_tokens", kv.num_cells()}, {"kv_cache_blocks", kv.num_blocks()},
        {"kv_cache_bytes", kv.bytes()}, {"block_size", kv.block_size()}, {"kv_dtype", cfg.kv_dtype},
        {"flash_attn", cfg.flash_attn}, {"enable_prefix_caching", cfg.enable_prefix_caching},
        {"n_threads", cfg.n_threads}, {"seed", cfg.seed},
        {"max_num_seqs", cfg.sched.max_num_seqs}, {"max_num_batched_tokens", cfg.sched.max_num_batched_tokens},
        {"enable_chunked_prefill", cfg.sched.enable_chunked_prefill},
        {"policy", cfg.sched.policy == SchedulingPolicy::PRIORITY ? "priority" : "fcfs"},
        {"backends", e->backend_summary()},
    };
    return j;
}

} // namespace iian::server
