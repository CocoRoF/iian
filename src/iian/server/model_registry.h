#pragma once
// Multi-model registry: one ServerContext + Engine per served model, lazy loading, LRU unloading.
#include "server_context.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace iian::server {

struct ModelSlot {
    enum class State { UNLOADED, LOADING, LOADED, FAILED };
    std::string name, path;
    ServerContext ctx;                       // per-model context (cfg copy with model_name/model_path set)
    std::shared_ptr<Model> model;
    std::unique_ptr<Engine> engine;
    State state = State::UNLOADED;
    std::string error;
    std::atomic<int> in_flight{0};
    std::chrono::steady_clock::time_point last_used = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point loaded_at{};
    std::mutex mtx;
    std::condition_variable cv;
    // cheap GGUF-header metadata for /v1/models while unloaded
    std::string arch, ftype, size_label, meta_name;
    uint64_t n_params = 0, file_bytes = 0;
    int64_t created = 0;
    bool loaded() const { return state == State::LOADED; }
};

// Loads the model for a slot: must set slot.model, slot.engine (started) and call slot.ctx.attach(engine).
using LoadFn = std::function<void(ModelSlot &)>;

class ModelRegistry {
public:
    ModelRegistry(const ServerConfig & base_cfg, int max_loaded);   // max_loaded <= 0: unlimited
    void set_loader(LoadFn f) { loader_ = std::move(f); }
    int max_loaded() const { return max_loaded_; }

    ModelSlot & add(const std::string & path, const std::string & name);   // throws std::invalid_argument on duplicate name
    ModelSlot * find(const std::string & name);
    std::vector<ModelSlot *> all();
    size_t size();
    size_t n_loaded();
    bool loading_any();

    // RAII in-flight lease: while held the model cannot be unloaded.
    struct Lease {
        ModelSlot * slot = nullptr;
        Lease() = default;
        explicit Lease(ModelSlot * s) : slot(s) { if (slot) slot->in_flight++; }
        Lease(Lease && o) noexcept : slot(o.slot) { o.slot = nullptr; }
        Lease & operator=(Lease && o) noexcept { if (this != &o) { release(); slot = o.slot; o.slot = nullptr; } return *this; }
        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;
        ~Lease() { release(); }
        void release() { if (slot) { slot->in_flight--; slot = nullptr; } }
        ServerContext & ctx() { return slot->ctx; }
    };
    // Resolves `name` ("" = the only model, or the only loaded one) and loads it if needed (blocking).
    // Throws ApiError (404 unknown, 400 ambiguous, 503 load failure / shutting down).
    Lease acquire(const std::string & name);

    bool load(ModelSlot & s, std::string & err);      // synchronous; evicts idle LRU slots beyond max_loaded
    bool unload(ModelSlot & s, std::string & err);    // refuses while requests are in flight
    void shutdown();                                  // stops all engines

private:
    void evict_beyond_max(ModelSlot * keep);
    std::mutex mtx_;
    std::vector<std::unique_ptr<ModelSlot>> slots_;
    LoadFn loader_;
    ServerConfig base_;
    int max_loaded_;
    std::atomic<bool> stopping_{false};
};

} // namespace iian::server
