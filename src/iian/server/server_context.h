#pragma once
// Shared server state: engine pointer, served model info, chat template, HTTP-side counters.
#include "http_server.h"
#include "tool_calls.h"

#include "iian/chat_template.h"
#include "iian/engine.h"
#include "iian/request.h"
#include "iian/types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace iian::server {

struct ModelInfo {
    std::string name;          // served model name
    std::string path;          // GGUF path
    std::string arch, ftype, size_label, model_name_meta;
    uint64_t    n_params = 0;
    uint64_t    file_bytes = 0;
    int64_t     created = 0;   // unix time of the model file (mtime)
    uint32_t    max_model_len = 0;
};

struct Counters {
    std::atomic<uint64_t> http_requests{0};
    std::atomic<uint64_t> http_errors{0};
    std::atomic<uint64_t> finished_stop{0}, finished_length{0}, finished_abort{0}, finished_error{0};
    std::atomic<uint64_t> requests_streamed{0};
    std::atomic<uint64_t> e2e_latency_us_sum{0}, e2e_latency_count{0};
    std::atomic<uint64_t> in_flight{0};
};

// external id ("chatcmpl-...") -> engine request ids, for /v1/engine/abort and introspection.
struct ActiveRequests {
    std::mutex mtx;
    std::unordered_map<std::string, std::vector<request_id_t>> by_external;
    void add(const std::string & ext, request_id_t id) { std::lock_guard<std::mutex> lk(mtx); by_external[ext].push_back(id); }
    void remove(const std::string & ext) { std::lock_guard<std::mutex> lk(mtx); by_external.erase(ext); }
    std::vector<request_id_t> find(const std::string & ext) { std::lock_guard<std::mutex> lk(mtx); auto it = by_external.find(ext); return it == by_external.end() ? std::vector<request_id_t>{} : it->second; }
    size_t size() { std::lock_guard<std::mutex> lk(mtx); return by_external.size(); }
    nlohmann::json list() {
        std::lock_guard<std::mutex> lk(mtx);
        nlohmann::json a = nlohmann::json::array();
        for (auto & [k, v] : by_external) a.push_back({{"request_id", k}, {"engine_ids", v}});
        return a;
    }
};

struct ServerContext {
    ServerConfig cfg;
    ActiveRequests active;
    std::atomic<Engine *> engine{nullptr};
    std::atomic<bool> stopping{false};
    ModelInfo model;
    Counters counters;
    std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();

    // chat template (default one from --chat-template or the GGUF, else ChatML)
    std::unique_ptr<ChatTemplate> chat_template;
    std::string chat_template_origin;   // "override" | "model" | "chatml"
    ToolFormat tool_format = ToolFormat::NONE;   // detected from the template source
    std::string bos_text, eos_text;
    std::mutex template_mtx;

    // Throws 503 while the engine is not attached yet.
    Engine & require_engine() const;
    bool ready() const { return engine.load() != nullptr; }

    // Called by set_engine: fills ModelInfo and loads the chat template.
    void attach(Engine * e);

    // Checks that `requested` (may be empty) names the served model; throws 404 otherwise.
    void check_model_name(const std::string & requested) const;

    // Render OpenAI-format messages with the chat template. `override_src` (if non-empty) must be
    // allowed by cfg.allow_request_chat_template.
    std::string render_chat(const nlohmann::json & messages, const nlohmann::json & tools, bool add_generation_prompt,
                            const std::map<std::string, nlohmann::json> & kwargs, const std::string & override_src);

    // Tokenize a rendered chat prompt: BOS is added only if the template did not already emit it.
    std::vector<token_t> tokenize_rendered(const std::string & rendered) const;
    // Tokenize a raw completion prompt (add_special = vocab default unless overridden).
    std::vector<token_t> tokenize_raw(const std::string & text, std::optional<bool> add_special) const;
    // 400 if the prompt cannot fit.
    void check_prompt_fits(size_t n_prompt, int32_t max_tokens) const;

    double uptime_s() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count(); }
};

// Prometheus text exposition of engine + server metrics (metrics.cpp).
std::string render_metrics(const ServerContext & ctx);
std::string render_metrics_all(const ServerContext & router, const std::vector<const ServerContext *> & models);
// JSON view of EngineStats + config (used by /v1/engine/stats and the CLI).
nlohmann::json engine_stats_json(const ServerContext & ctx);

} // namespace iian::server
