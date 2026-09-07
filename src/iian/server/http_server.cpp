#include "http_server.h"

#include "api_error.h"
#include "generation.h"
#include "json_util.h"
#include "model_registry.h"
#include "openai.h"
#include "server_context.h"

#include "iian/engine.h"
#include "iian/log.h"
#include "iian/tokenizer.h"
#include "iian/version.h"

#include "cpp-httplib/httplib.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <csignal>
#include <thread>

namespace iian::server {

namespace {

// Per-thread request bookkeeping for the access log (httplib runs a request end-to-end on one thread).
struct ReqLog {
    std::chrono::steady_clock::time_point start;
    std::string request_id;
    uint64_t prompt_tokens = 0, gen_tokens = 0;
};
thread_local ReqLog tl_log;

void send_json(httplib::Response & res, int status, const json & j) {
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

void send_error(httplib::Response & res, const ApiError & e) {
    send_json(res, e.status, e.to_json());
}

bool is_public_path(const std::string & p) {
    return p == "/health" || p == "/ready" || p == "/version" || p == "/" || p == "/v1/health";
}

} // namespace

struct HttpServer::Impl {
    ServerContext ctx;              // router-level context: config, HTTP counters, uptime, stopping flag
    ModelRegistry registry;
    httplib::Server svr;
    std::thread thread;
    std::thread preload_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> preloading{false};
    std::string endpoint;

    explicit Impl(ServerConfig cfg) : registry(cfg, cfg.models_max) {
        ctx.cfg = cfg;
        if (ctx.cfg.models.empty() && !ctx.cfg.model_path.empty()) ctx.cfg.models.push_back({ctx.cfg.model_path, ctx.cfg.model_name.empty() ? basename_no_ext_of(ctx.cfg.model_path) : ctx.cfg.model_name});
        for (auto & [path, name] : ctx.cfg.models) registry.add(path, name);
        ctx.model.name = ctx.cfg.models.empty() ? ctx.cfg.model_name : ctx.cfg.models[0].second;
        ctx.model.path = ctx.cfg.models.empty() ? ctx.cfg.model_path : ctx.cfg.models[0].first;
    }

    static std::string basename_no_ext_of(const std::string & path) {
        std::string b = path.substr(path.find_last_of('/') == std::string::npos ? 0 : path.find_last_of('/') + 1);
        if (b.size() > 5 && b.compare(b.size() - 5, 5, ".gguf") == 0) b.resize(b.size() - 5);
        return b;
    }

    // Per-request model resolution (the lease keeps the model loaded while the request runs).
    ModelRegistry::Lease lease_for(const json & body) { return registry.acquire(json_string(body, "model").value_or("")); }

    using Handler = std::function<void(const httplib::Request &, httplib::Response &)>;

    // Uniform exception -> OpenAI error mapping.
    httplib::Server::Handler wrap(Handler fn) {
        return [this, fn = std::move(fn)](const httplib::Request & req, httplib::Response & res) {
            try {
                fn(req, res);
            } catch (const ApiError & e) {
                send_error(res, e);
            } catch (const json::exception & e) {
                send_error(res, ApiError::bad_request(std::string("invalid JSON in request: ") + e.what()));
            } catch (const std::invalid_argument & e) {
                send_error(res, ApiError::bad_request(e.what()));
            } catch (const std::exception & e) {
                LOG_ERR("http", "%s %s: unhandled exception: %s", req.method.c_str(), req.path.c_str(), e.what());
                send_error(res, ApiError::internal(e.what()));
            }
            if (res.status >= 400) ctx.counters.http_errors++;
        };
    }

    httplib::Server::Handler admin(Handler fn) {
        return wrap([this, fn = std::move(fn)](const httplib::Request & req, httplib::Response & res) {
            if (!ctx.cfg.enable_admin) throw ApiError::forbidden("engine admin endpoints are disabled; start the server with --enable-admin");
            fn(req, res);
        });
    }

    void setup_middleware() {
        svr.set_default_headers({
            {"Access-Control-Allow-Origin", "*"},
            {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
            {"Access-Control-Allow-Headers", "Content-Type, Authorization, X-Api-Key, X-Request-Id"},
            {"Server", std::string("iian/") + iian::version()},
        });
        svr.set_pre_routing_handler([this](const httplib::Request & req, httplib::Response & res) {
            ctx.counters.http_requests++;
            tl_log = ReqLog{};
            tl_log.start = std::chrono::steady_clock::now();
            tl_log.request_id = req.get_header_value("X-Request-Id");
            if (tl_log.request_id.empty()) tl_log.request_id = random_id(16);
            res.set_header("X-Request-Id", tl_log.request_id);
            if (req.method == "OPTIONS") { res.status = 204; return httplib::Server::HandlerResponse::Handled; }
            if (!ctx.cfg.api_key.empty() && !is_public_path(req.path)) {
                std::string key = req.get_header_value("X-Api-Key");
                if (key.empty()) {
                    std::string auth = req.get_header_value("Authorization");
                    if (auth.rfind("Bearer ", 0) == 0) key = auth.substr(7);
                }
                if (key != ctx.cfg.api_key) {
                    ctx.counters.http_errors++;
                    send_error(res, ApiError::unauthorized("invalid or missing API key; send it as 'Authorization: Bearer <key>' or 'X-Api-Key: <key>'"));
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        svr.set_exception_handler([this](const httplib::Request & req, httplib::Response & res, std::exception_ptr ep) {
            std::string msg = "unknown error";
            try { if (ep) std::rethrow_exception(ep); } catch (const std::exception & e) { msg = e.what(); } catch (...) {}
            LOG_ERR("http", "%s %s: exception: %s", req.method.c_str(), req.path.c_str(), msg.c_str());
            ctx.counters.http_errors++;
            send_error(res, ApiError::internal(msg));
        });
        svr.set_error_handler([this](const httplib::Request & req, httplib::Response & res) {
            if (res.body.empty()) {   // unmatched route / method
                ctx.counters.http_errors++;
                if (res.status == 404)
                    send_error(res, ApiError(404, "no route for " + req.method + " " + req.path + " (see GET / for the endpoint list)", "not_found_error"));
                else
                    send_error(res, ApiError(res.status, "HTTP " + std::to_string(res.status), res.status >= 500 ? "internal_error" : "invalid_request_error"));
            }
        });
        svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl_log.start).count();
            if (tl_log.prompt_tokens || tl_log.gen_tokens)
                LOG_INF("http", "%s %s %d %.1fms rid=%s prompt=%llu gen=%llu %s", req.method.c_str(), req.path.c_str(), res.status, ms, tl_log.request_id.c_str(),
                        (unsigned long long) tl_log.prompt_tokens, (unsigned long long) tl_log.gen_tokens, req.remote_addr.c_str());
            else
                LOG_INF("http", "%s %s %d %.1fms rid=%s %s", req.method.c_str(), req.path.c_str(), res.status, ms, tl_log.request_id.c_str(), req.remote_addr.c_str());
        });
        svr.set_read_timeout(ctx.cfg.read_timeout_s, 0);
        svr.set_write_timeout(ctx.cfg.write_timeout_s, 0);
        svr.set_payload_max_length(256u * 1024 * 1024);
        svr.set_keep_alive_max_count(1000);
        int n = ctx.cfg.n_threads_http > 0 ? ctx.cfg.n_threads_http : std::max(8, (int) std::thread::hardware_concurrency());
        svr.new_task_queue = [n] { return new httplib::ThreadPool((size_t) n, (size_t) n + 1024); };
    }

    // ---- generation ---------------------------------------------------------------------------

    void handle_generate(const httplib::Request & req, httplib::Response & res, bool chat) {
        json body = parse_json_body(req.body);
        auto lease = std::make_shared<ModelRegistry::Lease>(lease_for(body));
        ServerContext & mctx = lease->ctx();
        GenerationRequest gr = chat ? parse_chat_request(mctx, body, req.get_header_value("X-Request-Id"))
                                    : parse_completion_request(mctx, body, req.get_header_value("X-Request-Id"));
        auto run = std::make_shared<GenerationRun>(mctx, std::move(gr));
        run->submit();
        tl_log.prompt_tokens = run->prompt_tokens();
        tl_log.request_id = run->request().id;
        res.set_header("X-Request-Id", run->request().id);

        if (!run->request().stream) {
            auto closed = req.is_connection_closed;
            bool ok = run->collect([&] { return closed && closed(); });
            tl_log.gen_tokens = run->completion_tokens();
            if (!ok && ctx.stopping.load()) throw ApiError::unavailable("server is shutting down; the request was aborted");
            json out = run->response_json();
            run->record();
            send_json(res, 200, out);
            return;
        }

        mctx.counters.requests_streamed++;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        auto provider = [this, run, lease](size_t, httplib::DataSink & sink) -> bool {
            if (ctx.stopping.load()) run->abort_all("server shutting down");
            std::string out;
            const bool more = run->next_sse(out, std::chrono::milliseconds(100));
            if (!out.empty()) {
                if (!sink.write(out.data(), out.size())) { run->abort_all("client disconnected (write failed)"); return false; }
            } else if (more && !sink.is_writable()) {
                run->abort_all("client disconnected");
                return false;
            }
            if (!more) { tl_log.gen_tokens = run->completion_tokens(); sink.done(); }
            return true;
        };
        auto on_complete = [run, lease](bool success) {
            if (!success) run->abort_all("stream closed");
            run->record();
        };
        res.set_chunked_content_provider("text/event-stream", provider, on_complete);
    }

    // ---- misc endpoints -------------------------------------------------------------------------

    json model_json(ModelSlot & sl) {
        const bool loaded = sl.loaded();
        json m = {
            {"id", sl.name}, {"object", "model"}, {"created", loaded ? sl.ctx.model.created : sl.created}, {"owned_by", "iian"},
            {"root", sl.path}, {"parent", nullptr}, {"permission", json::array()},
            {"status", sl.state == ModelSlot::State::LOADED ? "ready" : sl.state == ModelSlot::State::LOADING ? "loading" : sl.state == ModelSlot::State::FAILED ? "failed" : "unloaded"},
            {"meta", {{"arch", loaded ? sl.ctx.model.arch : sl.arch}, {"ftype", loaded ? sl.ctx.model.ftype : sl.ftype}, {"n_params", loaded ? sl.ctx.model.n_params : sl.n_params},
                      {"size_label", loaded ? sl.ctx.model.size_label : sl.size_label}, {"file_bytes", sl.file_bytes}, {"name", loaded ? sl.ctx.model.model_name_meta : sl.meta_name}}},
        };
        if (loaded) m["max_model_len"] = sl.ctx.model.max_model_len;
        if (!sl.error.empty()) m["error"] = sl.error;
        return m;
    }

    void handle_embeddings(const httplib::Request & req, httplib::Response & res) {
        json body = parse_json_body(req.body);
        auto lease = lease_for(body);
        ServerContext & ctx = lease.ctx();
        Engine & engine = ctx.require_engine();
        auto it = body.find("input");
        if (it == body.end() || it->is_null()) throw ApiError::bad_request("'input' is required (string, array of strings, or token ids)", "input");
        std::optional<bool> add_special = json_bool(body, "add_special_tokens");
        std::vector<std::vector<token_t>> inputs;
        std::vector<size_t> n_tokens_in;
        auto add_text = [&](const std::string & t) { if (t.empty()) throw ApiError::bad_request("'input' strings must not be empty", "input"); inputs.push_back(ctx.tokenize_raw(t, add_special)); };
        auto add_ids = [&](const json & arr) {
            std::vector<token_t> v;
            const int32_t n_vocab = (int32_t) engine.model().hparams().n_vocab;
            for (auto & e : arr) { if (!e.is_number_integer()) throw ApiError::bad_request("'input' token arrays must contain integers", "input"); int64_t t = e.get<int64_t>(); if (t < 0 || t >= n_vocab) throw ApiError::bad_request("token id " + std::to_string(t) + " is out of range", "input"); v.push_back((token_t) t); }
            if (v.empty()) throw ApiError::bad_request("'input' token arrays must not be empty", "input");
            inputs.push_back(std::move(v));
        };
        if (it->is_string()) add_text(it->get<std::string>());
        else if (it->is_array()) {
            if (it->empty()) throw ApiError::bad_request("'input' must not be empty", "input");
            if ((*it)[0].is_number_integer()) add_ids(*it);
            else for (auto & e : *it) { if (e.is_string()) add_text(e.get<std::string>()); else if (e.is_array()) add_ids(e); else json_type_error("input", "strings or token-id arrays", e); }
        } else json_type_error("input", "a string or an array", *it);
        if (inputs.size() > 256) throw ApiError::bad_request("at most 256 inputs per request", "input");

        std::string fmt = json_string(body, "encoding_format").value_or("float");
        if (fmt != "float" && fmt != "base64") throw ApiError::bad_request("encoding_format must be 'float' or 'base64'", "encoding_format");
        std::string pooling_s = json_string(body, "pooling").value_or("mean");
        Engine::Pooling pooling = Engine::Pooling::MEAN;
        if (pooling_s == "last") pooling = Engine::Pooling::LAST; else if (pooling_s == "cls") pooling = Engine::Pooling::CLS;
        else if (pooling_s != "mean") throw ApiError::bad_request("pooling must be 'mean', 'last' or 'cls'", "pooling");
        const bool normalize = json_bool(body, "normalize").value_or(true);
        int64_t dims = json_integer(body, "dimensions").value_or(0);
        if (dims < 0 || dims > (int64_t) engine.n_embd()) throw ApiError::bad_request("dimensions must be between 1 and " + std::to_string(engine.n_embd()), "dimensions");
        for (auto & v : inputs) ctx.check_prompt_fits(v.size(), 0);

        // submit all, then collect (the engine batches them)
        std::vector<Engine::Handle> hs;
        for (auto & v : inputs) { n_tokens_in.push_back(v.size()); hs.push_back(engine.submit_embedding(v, pooling, normalize)); }
        json data = json::array();
        uint64_t total_tokens = 0;
        for (size_t i = 0; i < hs.size(); i++) {
            OutputChunk c;
            while (true) {
                if (hs[i].out->pop(c, std::chrono::milliseconds(200))) { if (c.finished) break; continue; }
                if (!res_writable(req)) { for (auto & h : hs) engine.abort(h.id); throw ApiError(499, "client disconnected", "internal_error"); }
            }
            if (!c.error.empty()) throw ApiError(500, "embedding failed: " + c.error, "internal_error");
            std::vector<float> v = std::move(c.embedding);
            if (dims > 0 && (int64_t) v.size() > dims) {
                v.resize((size_t) dims);
                if (normalize) { double n = 0; for (float x : v) n += (double) x * x; float inv = n > 0 ? (float) (1.0 / std::sqrt(n)) : 0.0f; for (auto & x : v) x *= inv; }
            }
            total_tokens += n_tokens_in[i];
            json emb;
            if (fmt == "base64") emb = base64_encode(reinterpret_cast<const unsigned char *>(v.data()), v.size() * sizeof(float));
            else emb = v;
            data.push_back({{"object", "embedding"}, {"index", (int) i}, {"embedding", emb}});
        }
        ctx.counters.http_requests++;
        send_json(res, 200, json{{"object", "list"}, {"data", data}, {"model", ctx.model.name}, {"usage", {{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}}}});
    }

    static bool res_writable(const httplib::Request &) { return true; }

    static std::string base64_encode(const unsigned char * data, size_t len) {
        static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out; out.reserve((len + 2) / 3 * 4);
        for (size_t i = 0; i < len; i += 3) {
            uint32_t v = (uint32_t) data[i] << 16 | (i + 1 < len ? (uint32_t) data[i + 1] << 8 : 0) | (i + 2 < len ? data[i + 2] : 0);
            out += tbl[(v >> 18) & 63]; out += tbl[(v >> 12) & 63];
            out += i + 1 < len ? tbl[(v >> 6) & 63] : '=';
            out += i + 2 < len ? tbl[v & 63] : '=';
        }
        return out;
    }

    void handle_tokenize(const httplib::Request & req, httplib::Response & res) {
        json body = parse_json_body(req.body);
        auto lease = lease_for(body);
        ServerContext & ctx = lease.ctx();
        Engine & engine = ctx.require_engine();
        const Tokenizer & tok = engine.model().tokenizer();
        std::vector<token_t> toks;
        std::optional<bool> add_special = json_bool(body, "add_special_tokens");
        if (body.contains("messages")) {
            const json & msgs = body["messages"];
            if (!msgs.is_array()) json_type_error("messages", "an array", msgs);
            std::string rendered = ctx.render_chat(msgs, nullptr, json_bool(body, "add_generation_prompt").value_or(true), {}, "");
            toks = ctx.tokenize_rendered(rendered);
        } else if (auto p = json_string(body, "prompt")) {
            toks = ctx.tokenize_raw(*p, add_special.value_or(true));
        } else if (auto c = json_string(body, "content")) {   // llama.cpp compatibility
            toks = ctx.tokenize_raw(*c, add_special.value_or(true));
        } else {
            throw ApiError::bad_request("provide 'prompt' (string) or 'messages' (array)", "prompt");
        }
        json out = {{"tokens", toks}, {"count", toks.size()}, {"max_model_len", engine.max_model_len()}};
        if (json_bool(body, "with_pieces").value_or(false)) {
            json pieces = json::array();
            for (auto t : toks) pieces.push_back(tok.token_to_piece(t, true));
            out["pieces"] = pieces;
        }
        send_json(res, 200, out);
    }

    void handle_detokenize(const httplib::Request & req, httplib::Response & res) {
        json body = parse_json_body(req.body);
        auto lease = lease_for(body);
        ServerContext & ctx = lease.ctx();
        Engine & engine = ctx.require_engine();
        const Tokenizer & tok = engine.model().tokenizer();
        std::vector<int32_t> toks = json_int_list(body, "tokens");
        if (!body.contains("tokens")) throw ApiError::bad_request("'tokens' (array of integers) is required", "tokens");
        const int32_t n_vocab = (int32_t) engine.model().hparams().n_vocab;
        for (auto t : toks) if (t < 0 || t >= n_vocab) throw ApiError::bad_request("token id " + std::to_string(t) + " is out of range [0, " + std::to_string(n_vocab) + ")", "tokens");
        bool skip_special = json_bool(body, "skip_special_tokens").value_or(false);
        std::string text = tok.decode(toks, skip_special, !skip_special);
        send_json(res, 200, json{{"prompt", text}, {"content", text}});
    }

    ModelSlot * default_slot() {
        auto v = registry.all();
        for (auto * sl : v) if (sl->loaded()) return sl;
        return v.empty() ? nullptr : v[0];
    }

    bool any_ready() { return registry.n_loaded() > 0; }

    json props_json() {
        ModelSlot * sl = default_slot();
        if (!sl) return json{{"version", iian::version()}, {"build_info", ctx.cfg.build_info}, {"ready", false}, {"models", json::array()}};
        ServerContext & ctx = sl->ctx;
        json j = {
            {"model_name", ctx.model.name}, {"model_path", ctx.model.path},
            {"chat_template", ctx.chat_template ? ctx.chat_template->source() : ""},
            {"chat_template_origin", ctx.chat_template_origin},
            {"bos_token", ctx.bos_text}, {"eos_token", ctx.eos_text},
            {"n_ctx", ctx.model.max_model_len}, {"max_model_len", ctx.model.max_model_len},
            {"version", iian::version()}, {"build_info", ctx.cfg.build_info},
            {"ready", ctx.ready()},
        };
        if (Engine * e = ctx.engine.load()) {
            j["backends"] = e->backend_summary();
            j["total_slots"] = e->config().sched.max_num_seqs;
            j["arch"] = ctx.model.arch;
            j["ftype"] = ctx.model.ftype;
            j["n_params"] = ctx.model.n_params;
            j["tokenizer"] = e->model().tokenizer().type_name();
            j["chat_template_caps"] = {
                {"supports_system_role", ctx.chat_template->caps().supports_system_role},
                {"supports_tools", ctx.chat_template->caps().supports_tools},
                {"supports_tool_calls", ctx.chat_template->caps().supports_tool_calls},
                {"requires_typed_content", ctx.chat_template->caps().requires_typed_content},
                {"supports_enable_thinking", ctx.chat_template->caps().supports_enable_thinking},
            };
            j["tool_call_format"] = tool_format_name(ctx.tool_format);
        }
        json models = json::array();
        for (auto * m : registry.all()) models.push_back({{"name", m->name}, {"path", m->path}, {"loaded", m->loaded()}});
        j["models"] = models;
        j["models_max"] = registry.max_loaded();
        return j;
    }

    void setup_routes() {
        svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });

        svr.Get("/", wrap([this](const httplib::Request &, httplib::Response & res) {
            send_json(res, 200, json{{"name", "iian"}, {"version", iian::version()}, {"model", ctx.model.name}, {"ready", any_ready()}, {"models", registry.size()},
                                     {"endpoints", {"POST /v1/chat/completions", "POST /v1/completions", "GET /v1/models", "GET /v1/models/{id}",
                                                    "GET /health", "GET /ready", "GET /version", "GET /metrics", "POST /tokenize", "POST /detokenize",
                                                    "GET /props", "GET /v1/engine/stats (admin)", "POST /v1/engine/abort (admin)"}}});
        }));
        auto health = wrap([this](const httplib::Request &, httplib::Response & res) {
            const size_t n_loaded = registry.n_loaded();
            if (ctx.stopping.load()) send_json(res, 503, json{{"status", "stopping"}});
            else if (n_loaded > 0 || (ctx.cfg.lazy_load && !preloading.load()))
                send_json(res, 200, json{{"status", "ok"}, {"model", ctx.model.name}, {"models_loaded", n_loaded}, {"models", registry.size()}, {"uptime_s", ctx.uptime_s()}});
            else send_json(res, 503, json{{"status", "loading"}, {"model", ctx.model.name}, {"message", "model is loading"}});
        });
        svr.Get("/health", health);
        svr.Get("/v1/health", health);
        svr.Get("/ready", health);
        svr.Get("/version", wrap([this](const httplib::Request &, httplib::Response & res) {
            send_json(res, 200, json{{"version", iian::version()}, {"build_info", ctx.cfg.build_info}, {"httplib", CPPHTTPLIB_VERSION}});
        }));
        svr.Get("/metrics", wrap([this](const httplib::Request &, httplib::Response & res) {
            std::vector<const ServerContext *> ctxs;
            for (auto * sl : registry.all()) if (sl->loaded()) ctxs.push_back(&sl->ctx);
            res.set_content(render_metrics_all(ctx, ctxs), "text/plain; version=0.0.4; charset=utf-8");
        }));
        svr.Get("/v1/models", wrap([this](const httplib::Request &, httplib::Response & res) {
            json data = json::array();
            for (auto * sl : registry.all()) data.push_back(model_json(*sl));
            send_json(res, 200, json{{"object", "list"}, {"data", data}});
        }));
        svr.Get(R"(/v1/models/(.+))", wrap([this](const httplib::Request & req, httplib::Response & res) {
            std::string id = req.matches[1];
            ModelSlot * sl = registry.find(id);
            if (!sl) throw ApiError::not_found("The model '" + id + "' does not exist (GET /v1/models lists the available models)", "model");
            send_json(res, 200, model_json(*sl));
        }));
        svr.Post("/v1/models/load", admin([this](const httplib::Request & req, httplib::Response & res) {
            json body = parse_json_body(req.body);
            std::string name = json_string(body, "model").value_or("");
            ModelSlot * sl = registry.find(name);
            if (!sl) throw ApiError::not_found("unknown model '" + name + "'", "model");
            std::string err;
            if (!registry.load(*sl, err)) throw ApiError::unavailable("failed to load '" + name + "': " + err);
            send_json(res, 200, model_json(*sl));
        }));
        svr.Post("/v1/models/unload", admin([this](const httplib::Request & req, httplib::Response & res) {
            json body = parse_json_body(req.body);
            std::string name = json_string(body, "model").value_or("");
            ModelSlot * sl = registry.find(name);
            if (!sl) throw ApiError::not_found("unknown model '" + name + "'", "model");
            std::string err;
            if (!registry.unload(*sl, err)) throw ApiError(409, "cannot unload '" + name + "': " + err, "invalid_request_error");
            send_json(res, 200, model_json(*sl));
        }));
        svr.Post("/v1/chat/completions", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_generate(req, res, true); }));
        svr.Post("/chat/completions", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_generate(req, res, true); }));
        svr.Post("/v1/completions", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_generate(req, res, false); }));
        svr.Post("/completions", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_generate(req, res, false); }));
        svr.Post("/v1/embeddings", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_embeddings(req, res); }));
        svr.Post("/embeddings", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_embeddings(req, res); }));
        svr.Post("/tokenize", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_tokenize(req, res); }));
        svr.Post("/detokenize", wrap([this](const httplib::Request & req, httplib::Response & res) { handle_detokenize(req, res); }));
        svr.Get("/props", wrap([this](const httplib::Request &, httplib::Response & res) { send_json(res, 200, props_json()); }));

        svr.Get("/v1/engine/stats", admin([this](const httplib::Request & req, httplib::Response & res) {
            std::string name = req.get_param_value("model");
            ModelSlot * sl = name.empty() ? default_slot() : registry.find(name);
            if (!sl) throw ApiError::not_found("unknown model '" + name + "'", "model");
            json j = engine_stats_json(sl->ctx);
            j["active_requests"] = sl->ctx.active.list();
            json models = json::array();
            for (auto * m : registry.all()) models.push_back({{"name", m->name}, {"loaded", m->loaded()}, {"in_flight", m->in_flight.load()}});
            j["models"] = models;
            send_json(res, 200, j);
        }));
        svr.Post("/v1/engine/abort", admin([this](const httplib::Request & req, httplib::Response & res) {
            json body = parse_json_body(req.body);
            auto lease = lease_for(body);
            ServerContext & ctx = lease.ctx();
            Engine & engine = ctx.require_engine();
            std::string rid;
            if (auto s = json_string(body, "request_id")) rid = *s;
            else if (auto n = json_integer(body, "request_id")) rid = std::to_string(*n);
            else throw ApiError::bad_request("'request_id' is required (the response id, e.g. \"chatcmpl-...\", or an engine request id)", "request_id");
            std::vector<request_id_t> ids = ctx.active.find(rid);
            if (ids.empty() && !rid.empty() && std::all_of(rid.begin(), rid.end(), ::isdigit)) ids.push_back((request_id_t) std::stoull(rid));
            for (auto id : ids) engine.abort(id);
            send_json(res, 200, json{{"request_id", rid}, {"aborted", ids.size()}});
        }));
    }

    bool bind(std::string & err) {
        if (!ctx.cfg.unix_socket.empty()) {
            ::unlink(ctx.cfg.unix_socket.c_str());
            svr.set_address_family(AF_UNIX);
            if (!svr.bind_to_port(ctx.cfg.unix_socket, 80)) { err = "failed to bind unix socket " + ctx.cfg.unix_socket + " (check the directory exists and is writable)"; return false; }
            endpoint = "unix://" + ctx.cfg.unix_socket;
            return true;
        }
        if (!svr.bind_to_port(ctx.cfg.host, ctx.cfg.port)) {
            err = "failed to bind " + ctx.cfg.host + ":" + std::to_string(ctx.cfg.port) + " (port in use? pick another with --port, or run `iian ps` to see running servers)";
            return false;
        }
        std::string h = ctx.cfg.host == "0.0.0.0" || ctx.cfg.host == "::" ? "127.0.0.1" : ctx.cfg.host;
        if (h.find(':') != std::string::npos) h = "[" + h + "]";
        endpoint = "http://" + h + ":" + std::to_string(ctx.cfg.port);
        return true;
    }
};

HttpServer::HttpServer(ServerConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {
    impl_->setup_middleware();
    impl_->setup_routes();
}

HttpServer::~HttpServer() { stop(); }

bool HttpServer::start(std::string & err) {
    if (impl_->running.load()) return true;
    ::signal(SIGPIPE, SIG_IGN);
    if (!impl_->bind(err)) return false;
    impl_->running = true;
    impl_->thread = std::thread([this] {
        impl_->svr.listen_after_bind();
        impl_->running = false;
    });
    if (!impl_->ctx.cfg.lazy_load && impl_->registry.size() > 0) {
        impl_->preloading = true;
        impl_->preload_thread = std::thread([this] {
            for (ModelSlot * sl : impl_->registry.all()) {
                if (impl_->ctx.stopping.load()) break;
                std::string err;
                impl_->registry.load(*sl, err);
            }
            impl_->preloading = false;
        });
    }
    LOG_INF("http", "listening on %s (threads=%d, admin=%s, auth=%s)", impl_->endpoint.c_str(),
            impl_->ctx.cfg.n_threads_http > 0 ? impl_->ctx.cfg.n_threads_http : std::max(8, (int) std::thread::hardware_concurrency()),
            impl_->ctx.cfg.enable_admin ? "on" : "off", impl_->ctx.cfg.api_key.empty() ? "off" : "on");
    return true;
}

void HttpServer::set_loader(std::function<void(ModelSlot &)> loader) { impl_->registry.set_loader(std::move(loader)); }
ModelRegistry & HttpServer::registry() { return impl_->registry; }

void HttpServer::set_engine(Engine * engine) {
    // compatibility path: adopt an externally created engine as the first (or only) model
    auto slots = impl_->registry.all();
    ModelSlot * sl = slots.empty() ? &impl_->registry.add(impl_->ctx.cfg.model_path, impl_->ctx.cfg.model_name.empty() ? "model" : impl_->ctx.cfg.model_name) : slots[0];
    std::lock_guard<std::mutex> lk(sl->mtx);
    if (engine) { sl->ctx.attach(engine); sl->state = ModelSlot::State::LOADED; sl->loaded_at = sl->last_used = std::chrono::steady_clock::now(); }
    else { sl->ctx.engine.store(nullptr); sl->state = ModelSlot::State::UNLOADED; }
}

void HttpServer::wait_preload() {
    if (impl_->preload_thread.joinable()) impl_->preload_thread.join();
}

void HttpServer::stop() {
    if (!impl_) return;
    impl_->ctx.stopping = true;
    if (impl_->preload_thread.joinable()) impl_->preload_thread.join();
    impl_->registry.shutdown();
    if (impl_->thread.joinable()) {
        impl_->svr.stop();
        impl_->thread.join();
        LOG_INF("http", "server stopped");
    }
    if (!impl_->ctx.cfg.unix_socket.empty()) ::unlink(impl_->ctx.cfg.unix_socket.c_str());
}

bool HttpServer::running() const { return impl_->running.load(); }
std::string HttpServer::endpoint() const { return impl_->endpoint; }
const ServerConfig & HttpServer::config() const { return impl_->ctx.cfg; }
ServerContext & HttpServer::context() { return impl_->ctx; }

} // namespace iian::server
