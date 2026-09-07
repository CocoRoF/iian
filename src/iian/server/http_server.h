#pragma once
// iian HTTP server: OpenAI-compatible REST API (chat/completions, completions, models, tokenize, ...)
// plus health, metrics and admin endpoints, on top of cpp-httplib.
//
// Lifecycle: construct with a ServerConfig -> start() binds and serves immediately (everything but
// /health, /ready, /version answers 503 "loading") -> set_engine() once the model is ready -> stop().
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace iian {
class Engine;
namespace server {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int         port = 9931;
    std::string unix_socket;                 // if set, listen on this AF_UNIX path instead of host:port
    std::string api_key;                     // empty = no auth
    std::string model_name;                  // served model name (empty -> basename of the GGUF)
    std::string model_path;                  // shown while loading (set_engine fills the rest)
    std::string chat_template;               // template source override (already read from file if any)
    bool        allow_request_chat_template = false;
    bool        enable_admin = false;        // /v1/engine/* endpoints
    int         n_threads_http = -1;         // -1 = auto
    std::string build_info;                  // shown by /version and /props
    int         read_timeout_s = 600;
    int         write_timeout_s = 600;
    // multi-model: (path, served name) pairs; the first is the default. models_max <= 0 = unlimited.
    std::vector<std::pair<std::string, std::string>> models;
    int         models_max = 0;
    bool        lazy_load = false;           // load models on first use instead of at startup
};

struct ServerContext;
struct ModelSlot;
class ModelRegistry;

class HttpServer {
public:
    explicit HttpServer(ServerConfig cfg);
    ~HttpServer();
    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    // Bind + start serving on a background thread. Returns false (with `err`) if the bind fails.
    bool start(std::string & err);
    // Model loading is done by the server through this callback (see model_registry.h). Must be set before start().
    void set_loader(std::function<void(ModelSlot &)> loader);
    // Registered models (from ServerConfig::models) are preloaded on start() unless lazy_load; returns when all
    // preloads finished (or immediately in lazy mode). Failures are logged; check registry().
    void wait_preload();
    ModelRegistry & registry();
    // Single-model compatibility: attach an already-created engine as the (only) model.
    void set_engine(Engine * engine);
    // Graceful stop: in-flight streams are aborted, the listener is joined. Idempotent.
    void stop();
    bool running() const;

    std::string endpoint() const;           // "http://host:port" or "unix://path"
    const ServerConfig & config() const;
    ServerContext & context();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace server
} // namespace iian
