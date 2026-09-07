#include "model_registry.h"

#include "api_error.h"

#include "iian/log.h"
#include "iian/model.h"

#include <algorithm>
#include <sys/stat.h>

namespace iian::server {

ModelRegistry::ModelRegistry(const ServerConfig & base_cfg, int max_loaded) : base_(base_cfg), max_loaded_(max_loaded) {}

ModelSlot & ModelRegistry::add(const std::string & path, const std::string & name) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto & s : slots_) if (s->name == name) throw std::invalid_argument("duplicate model name '" + name + "' (use --served-model-name or rename the file)");
    auto s = std::make_unique<ModelSlot>();
    s->name = name; s->path = path;
    s->ctx.cfg = base_;
    s->ctx.cfg.model_name = name;
    s->ctx.cfg.model_path = path;
    s->ctx.model.name = name;
    s->ctx.model.path = path;
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) { s->file_bytes = (uint64_t) st.st_size; s->created = (int64_t) st.st_mtime; }
    try {
        auto md = ModelLoader::load_metadata(path);
        s->arch = md->hparams().arch; s->ftype = md->ftype_name(); s->size_label = md->size_label(); s->n_params = md->n_params(); s->meta_name = md->hparams().name;
    } catch (const std::exception & e) {
        LOG_WRN("http", "cannot read metadata of %s: %s", path.c_str(), e.what());
    }
    slots_.push_back(std::move(s));
    return *slots_.back();
}

ModelSlot * ModelRegistry::find(const std::string & name) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto & s : slots_) if (s->name == name) return s.get();
    return nullptr;
}

std::vector<ModelSlot *> ModelRegistry::all() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<ModelSlot *> v;
    for (auto & s : slots_) v.push_back(s.get());
    return v;
}

size_t ModelRegistry::size() { std::lock_guard<std::mutex> lk(mtx_); return slots_.size(); }
size_t ModelRegistry::n_loaded() { std::lock_guard<std::mutex> lk(mtx_); size_t n = 0; for (auto & s : slots_) if (s->loaded()) n++; return n; }
bool ModelRegistry::loading_any() { std::lock_guard<std::mutex> lk(mtx_); for (auto & s : slots_) if (s->state == ModelSlot::State::LOADING) return true; return false; }

ModelRegistry::Lease ModelRegistry::acquire(const std::string & name) {
    if (stopping_) throw ApiError::unavailable("server is shutting down");
    ModelSlot * slot = nullptr;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (slots_.empty()) throw ApiError::unavailable("no models are configured on this server");
        if (name.empty()) {
            if (slots_.size() == 1) slot = slots_[0].get();
            else {
                ModelSlot * only = nullptr; int n = 0;
                for (auto & s : slots_) if (s->loaded()) { only = s.get(); n++; }
                if (n == 1) slot = only;
                else {
                    std::string names;
                    for (auto & s : slots_) names += (names.empty() ? "" : ", ") + s->name;
                    throw ApiError::bad_request("'model' is required: this server serves several models (" + names + ")", "model");
                }
            }
        } else {
            for (auto & s : slots_) if (s->name == name) { slot = s.get(); break; }
            if (!slot) {
                std::string names;
                for (auto & s : slots_) names += (names.empty() ? "" : ", ") + s->name;
                throw ApiError::not_found("The model '" + name + "' does not exist. Available: " + names + " (GET /v1/models)", "model");
            }
        }
    }
    for (int attempt = 0; attempt < 3; attempt++) {
        std::string err;
        if (!slot->loaded() && !load(*slot, err)) throw ApiError::unavailable("model '" + slot->name + "' could not be loaded: " + err);
        std::lock_guard<std::mutex> lk(mtx_);
        if (slot->loaded()) { slot->last_used = std::chrono::steady_clock::now(); return Lease(slot); }
    }
    throw ApiError::unavailable("model '" + slot->name + "' keeps being unloaded; raise --models-max");
}

bool ModelRegistry::load(ModelSlot & s, std::string & err) {
    {
        std::unique_lock<std::mutex> lk(s.mtx);
        s.cv.wait(lk, [&] { return s.state != ModelSlot::State::LOADING; });
        if (s.state == ModelSlot::State::LOADED) return true;
        if (!loader_) { err = "no model loader configured"; return false; }
        s.state = ModelSlot::State::LOADING;
        s.error.clear();
    }
    LOG_INF("http", "loading model '%s' (%s)", s.name.c_str(), s.path.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    try {
        loader_(s);
        if (!s.engine) throw std::runtime_error("loader did not create an engine");
    } catch (const std::exception & e) {
        ok = false; err = e.what();
        s.engine.reset(); s.model.reset();
        s.ctx.engine.store(nullptr);
        LOG_ERR("http", "failed to load model '%s': %s", s.name.c_str(), e.what());
    }
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        s.state = ok ? ModelSlot::State::LOADED : ModelSlot::State::FAILED;
        s.error = err;
        s.loaded_at = std::chrono::steady_clock::now();
        s.last_used = s.loaded_at;
    }
    s.cv.notify_all();
    if (ok) {
        LOG_INF("http", "model '%s' ready in %.1fs (ctx %u, kv cache %.0f MiB)", s.name.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), s.engine->max_model_len(), s.engine->kv().bytes() / 1048576.0);
        evict_beyond_max(&s);
    }
    return ok;
}

void ModelRegistry::evict_beyond_max(ModelSlot * keep) {
    if (max_loaded_ <= 0) return;
    while (true) {
        ModelSlot * victim = nullptr;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            size_t loaded = 0;
            for (auto & s : slots_) if (s->loaded()) loaded++;
            if ((int) loaded <= max_loaded_) return;
            for (auto & s : slots_) {
                if (!s->loaded() || s.get() == keep || s->in_flight.load() > 0) continue;
                if (!victim || s->last_used < victim->last_used) victim = s.get();
            }
        }
        if (!victim) { LOG_WRN("http", "%zu models loaded (> --models-max %d) but all are busy; not unloading", n_loaded(), max_loaded_); return; }
        std::string err;
        if (!unload(*victim, err)) return;
        LOG_INF("http", "unloaded model '%s' (least recently used) to stay within --models-max %d", victim->name.c_str(), max_loaded_);
    }
}

bool ModelRegistry::unload(ModelSlot & s, std::string & err) {
    std::unique_lock<std::mutex> lk(s.mtx);
    if (s.state == ModelSlot::State::LOADING) { err = "model is loading"; return false; }
    if (s.state != ModelSlot::State::LOADED) return true;
    if (s.in_flight.load() > 0) { err = "model has requests in flight"; return false; }
    s.ctx.engine.store(nullptr);
    if (s.engine) s.engine->stop();
    s.engine.reset();
    s.model.reset();
    s.state = ModelSlot::State::UNLOADED;
    return true;
}

void ModelRegistry::shutdown() {
    stopping_ = true;
    for (ModelSlot * s : all()) {
        std::unique_lock<std::mutex> lk(s->mtx);
        s->cv.wait(lk, [&] { return s->state != ModelSlot::State::LOADING; });
        s->ctx.stopping = true;
        s->ctx.engine.store(nullptr);
        if (s->engine) s->engine->stop();
        s->engine.reset();
        s->model.reset();
        s->state = ModelSlot::State::UNLOADED;
    }
}

} // namespace iian::server
