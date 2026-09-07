#include "iian/engine.h"
#include "iian/arch.h"
#include "iian/graph.h"
#include "iian/log.h"
#include "iian/tokenizer.h"
#include "paged_attn.h"
#include "grammar/grammar.h"
#include "grammar/json_schema.h"
#include "draft.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace iian {

struct Engine::GraphState {
    std::vector<uint8_t> meta;   // ggml context memory for graph nodes
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::unique_ptr<GraphContext> gc;
    UBatch ub;                   // the batch the current graph was built for (inputs are re-set every step)
    // topology key: if unchanged, the graph (and its allocation) is reused and only the inputs are refilled
    struct Key { uint32_t n_tokens = 0, n_outputs = 0, n_kv = 0, kv_start = 0; bool paged = false; bool valid = false; uint64_t layout = 0; bool gather = false; } key;
    uint64_t n_reused = 0, n_built = 0;
    // IIAN_PROFILE=1: per-phase step timings (microseconds), reported every 500 steps and at shutdown
    struct Prof { bool on = false; uint64_t steps = 0, tokens = 0, us_sched = 0, us_batch = 0, us_build = 0, us_inputs = 0, us_compute = 0, us_outputs = 0, us_sample = 0; int n_splits = 0, n_copies = 0; } prof;
    ~GraphState() { if (ctx) ggml_free(ctx); }
    void report() const {
        if (!prof.on || prof.steps == 0) return;
        const double n = (double) prof.steps;
        fprintf(stderr, "iian profile: %llu steps, %.1f tokens/step | per step: schedule %.0f us, batch %.0f us, graph build %.0f us (%llu builds, %llu reuses), inputs %.0f us, compute+logits %.0f us (%d splits, %d copies), sampling/outputs %.0f us (sampler %.0f us)\n",
                (unsigned long long) prof.steps, prof.tokens / n, prof.us_sched / n, prof.us_batch / n, prof.us_build / n,
                (unsigned long long) n_built, (unsigned long long) n_reused, prof.us_inputs / n, prof.us_compute / n, prof.n_splits, prof.n_copies, prof.us_outputs / n, prof.us_sample / n);
    }
};

static ggml_type kv_type_from_string(const std::string & s) {
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    throw std::runtime_error("unknown kv cache dtype: " + s + " (use f16|bf16|f32|q8_0|q4_0|q4_1|q5_0|q5_1)");
}

// IIAN_DUMP_TENSORS=1: print name, shape and checksum of every graph node (debug aid to compare layer
// outputs against llama.cpp's `llama-eval-callback`). Only F32/F16 tensors are summarised.
// IIAN_PROFILE=ops: time every graph node (the scheduler synchronises after each node when a callback asks
// for it) and report the heaviest ops at shutdown. Keys are "<op>:<tensor base name>" with the layer suffix stripped.
struct OpProfile { double us = 0; uint64_t n = 0; };
static std::map<std::string, OpProfile> g_op_prof;
static int64_t g_op_t0 = 0;
static bool profile_op_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) { g_op_t0 = ggml_time_us(); return true; }
    std::string name = ggml_get_name(t);
    size_t dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size() && std::all_of(name.begin() + dash + 1, name.end(), ::isdigit)) name.resize(dash);
    auto & e = g_op_prof[std::string(ggml_op_name(t->op)) + ":" + name];
    e.us += (double) (ggml_time_us() - g_op_t0); e.n++;
    return true;
}
static void report_op_profile() {
    if (g_op_prof.empty()) return;
    std::vector<std::pair<std::string, OpProfile>> v(g_op_prof.begin(), g_op_prof.end());
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.us > b.second.us; });
    double total = 0; for (auto & e : v) total += e.second.us;
    fprintf(stderr, "iian op profile (per-node synchronised, %.1f ms total):\n", total / 1000.0);
    for (size_t i = 0; i < v.size() && i < 20; i++)
        fprintf(stderr, "  %5.1f%%  %9.1f ms  %8llu calls  %6.1f us/call  %s\n", 100.0 * v[i].second.us / total, v[i].second.us / 1000.0,
                (unsigned long long) v[i].second.n, v[i].second.us / v[i].second.n, v[i].first.c_str());
}

static bool dump_tensor_cb(ggml_tensor * t, bool ask, void *) {
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) return true;
    const int64_t n = ggml_nelements(t);
    std::vector<float> buf((size_t) n);
    if (t->type == GGML_TYPE_F32) ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    else { std::vector<ggml_fp16_t> h((size_t) n); ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(ggml_fp16_t)); ggml_fp16_to_fp32_row(h.data(), buf.data(), n); }
    double sum = 0, asum = 0;
    for (int64_t i = 0; i < n; i++) { sum += buf[i]; asum += std::fabs(buf[i]); }
    fprintf(stderr, "iian_dump %-24s %-8s [%lld,%lld,%lld,%lld] sum=%.6g abs=%.6g first=[%.5g %.5g %.5g]\n", ggml_get_name(t), ggml_op_desc(t),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], sum, asum, buf[0], n > 1 ? buf[1] : 0.f, n > 2 ? buf[2] : 0.f);
    return true;
}

Engine::Engine(std::shared_ptr<Model> model, const EngineConfig & cfg) : model_(std::move(model)), cfg_(cfg) {
    dump_tensors_ = getenv("IIAN_DUMP_TENSORS") != nullptr;
    if (getenv("IIAN_NO_FLASH_ATTN")) cfg_.flash_attn = false;   // debugging aid: force the mul_mat+softmax attention path
    const HParams & hp = model_->hparams();
    // default context: the model's training context, capped at 8k unless the user asks for more
    // (a 128k default would allocate tens of GiB of KV cache before anyone typed a prompt)
    max_model_len_ = cfg.max_model_len ? cfg.max_model_len : std::min<uint32_t>(hp.n_ctx_train, 8192);
    if (max_model_len_ > hp.n_ctx_train) LOG_WRN("engine", "max_model_len %u exceeds the model's training context %u", max_model_len_, hp.n_ctx_train);
    if (cfg.sched.max_model_len == 0) cfg_.sched.max_model_len = max_model_len_;

    // ---- backends ----
    for (auto * dev : model_->devices()) {
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) throw std::runtime_error(std::string("failed to init backend ") + ggml_backend_dev_name(dev));
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) backend_cpu_ = b;
        backends_.push_back(b);
    }
    // CPU last (sched prefers earlier backends for ops)
    std::stable_partition(backends_.begin(), backends_.end(), [&](ggml_backend_t b) { return b != backend_cpu_; });
    {
        int nt = cfg.n_threads > 0 ? cfg.n_threads : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu_));
        auto set_n_threads = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_n_threads");
        if (set_n_threads) set_n_threads(backend_cpu_, nt);
        cfg_.n_threads = nt;
    }

    // ---- KV cache ----
    KVCacheConfig kcfg;
    kcfg.block_size = cfg.block_size;
    kcfg.type_k = kcfg.type_v = kv_type_from_string(cfg.kv_dtype);
    kcfg.enable_prefix_caching = cfg.enable_prefix_caching;
    size_t per_cell = 0;
    for (uint32_t il = 0; il < hp.n_layer; il++) if (hp.has_kv(il)) per_cell += ggml_row_size(kcfg.type_k, hp.n_embd_k_gqa(il)) + ggml_row_size(kcfg.type_v, hp.n_embd_v_gqa(il));
    per_cell = std::max<size_t>(1, per_cell);
    uint32_t cells = cfg.kv_cache_tokens;
    if (cells == 0) {
        cells = max_model_len_ * std::max(1u, std::min(cfg_.sched.max_num_seqs, 4u));
        // auto budget: never take more than half of the free memory of the device holding the cache
        size_t budget = cfg.kv_cache_bytes;
        if (budget == 0) {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(model_->dev_layer(hp.n_layer - 1), &free_b, &total_b);
            // some backends report host (UMA) memory as "free"; never trust more than the device's total
            if (total_b && free_b > total_b) free_b = total_b;
            if (free_b) budget = free_b / 4;   // conservative: weights, compute buffers and other tenants share this memory
        }
        if (budget) cells = (uint32_t) std::min<size_t>(cells, budget / per_cell);
    } else if (cfg.kv_cache_bytes) {
        cells = (uint32_t) std::min<size_t>(cells, cfg.kv_cache_bytes / per_cell);
    }
    // ggml's CUDA flash-attention kernels take their fast paths only when the KV length is a multiple of 256
    // (FATTN_KQ_STRIDE); the attention window is padded to 256 but clipped at the cache end, so keep the cell count
    // a multiple of lcm(block_size, 256) (at most 255 extra cells).
    const uint32_t align = std::lcm<uint32_t>(std::max<uint32_t>(1, kcfg.block_size), 256u);
    cells = (cells + align - 1) / align * align;
    if (cells < max_model_len_) {
        char b[256];
        snprintf(b, sizeof b, "KV cache too small: %u cells (%.2f GiB) but max_model_len is %u; lower --max-model-len, use --kv-dtype q8_0, or raise --kv-cache-gb",
                 cells, (double) cells * per_cell / 1073741824.0, max_model_len_);
        throw std::runtime_error(b);
    }
    kcfg.max_cells = cells;
    kv_ = std::make_unique<PagedKVCache>(*model_, kcfg);
    kv_->set_max_seq_len(max_model_len_);
    sched_ = std::make_unique<Scheduler>(cfg_.sched, *kv_);

    // ---- graph scheduler ----
    const size_t n_tensors = model_->layers.size() * 24 + 16;
    max_nodes_ = std::max<size_t>(8192, 8 * n_tensors);
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto * b : backends_) bufts.push_back(ggml_backend_get_default_buffer_type(b));
    gsched_ = ggml_backend_sched_new(backends_.data(), bufts.data(), (int) backends_.size(), max_nodes_, false, true);
    graph_ = std::make_unique<GraphState>();
    graph_->prof.on = getenv("IIAN_PROFILE") != nullptr;
    profile_ops_ = getenv("IIAN_PROFILE") && std::string(getenv("IIAN_PROFILE")) == "ops";
    graph_->meta.resize(ggml_tensor_overhead() * max_nodes_ + ggml_graph_overhead_custom(max_nodes_, false));

    // attention path: iian's paged kernel is a CPU custom op; use it when everything runs on the CPU
    {
        const bool cpu_only = backends_.size() == 1 && backends_[0] == backend_cpu_;
        const bool supported = paged_attn_supported(kv_->type_k(), kv_->type_v(), hp.f_max_alibi_bias);
        // gather: per-sequence batched flash attention (any backend; the GPU default). Requires flash attention.
        if (cfg.attention == "paged") {
            if (!cpu_only || !supported) throw std::runtime_error("attention=paged requires CPU-only execution with f16/f32 KV cache and no ALiBi");
            paged_attn_ = true;
            cfg_.attention_force_paged = true;
        } else if (cfg.attention == "masked") {
            paged_attn_ = false;
        } else if (cfg.attention == "gather") {
            if (!cfg_.flash_attn) throw std::runtime_error("attention=gather requires flash attention");
            gather_attn_ = true;
        } else if (cfg.attention == "auto") {
            paged_attn_ = cpu_only && supported;
            gather_attn_ = !cpu_only && cfg_.flash_attn;
        } else {
            throw std::runtime_error("unknown attention mode '" + cfg.attention + "' (auto|masked|paged|gather)");
        }
        cfg_.attention = paged_attn_ ? "paged" : gather_attn_ ? "gather" : "masked";
    }

    if (!cfg.spec_draft_model.empty()) {
        DraftConfig dc;
        dc.model_path = cfg.spec_draft_model;
        dc.n_draft = std::max<uint32_t>(1, cfg.spec_draft_n);
        dc.n_threads = cfg_.n_threads;
        dc.device = cfg.device;
        draft_ = std::make_unique<DraftModel>(*model_, dc, kv_->num_cells(), max_model_len_);
    }

    t_last_stats_ = std::chrono::steady_clock::now();
    LOG_INF("engine", "ready: max_model_len=%u kv_cells=%u (%.1f MiB) block_size=%u prefix_cache=%s attention=%s%s threads=%d max_num_seqs=%u max_batched_tokens=%u",
            max_model_len_, kv_->num_cells(), kv_->bytes() / 1048576.0, kv_->block_size(), cfg.enable_prefix_caching ? "on" : "off",
            cfg_.attention.c_str(), (!paged_attn_ && !gather_attn_ && cfg.flash_attn) ? "(flash)" : "", cfg_.n_threads, cfg_.sched.max_num_seqs, cfg_.sched.max_num_batched_tokens);
}

Engine::~Engine() {
    stop();
    draft_.reset();
    if (gsched_) ggml_backend_sched_free(gsched_);
    graph_.reset();
    kv_.reset();
    for (auto * b : backends_) ggml_backend_free(b);
}

std::string Engine::backend_summary() const {
    std::string s;
    for (auto * b : backends_) { if (!s.empty()) s += ", "; s += ggml_backend_name(b); }
    return s;
}

void Engine::start() {
    if (running_.exchange(true)) return;
    if (cfg_.warmup) warmup();
    thread_ = std::thread([this] { run_loop(); });
}

// One small prefill batch plus a decode step before serving: loads the backend kernels (CUDA lazy module loading),
// initialises cuBLAS and captures the first CUDA graph, so the first real request does not pay for it (the same
// idea as llama.cpp's start-up warmup). Runs synchronously on the caller's thread before the engine thread exists.
void Engine::warmup() {
    const int64_t t0 = ggml_time_us();
    const uint32_t n = std::max<uint32_t>(1, std::min<uint32_t>({64u, max_model_len_ > 4 ? max_model_len_ - 4 : 1u, cfg_.sched.max_num_batched_tokens}));
    const auto & sp = model_->tokenizer().special();
    const token_t t = sp.bos != TOKEN_NULL ? sp.bos : (sp.eos != TOKEN_NULL ? sp.eos : 0);
    std::vector<token_t> toks(n, t);
    SamplingParams params; params.temperature = 0.0f; params.max_tokens = 2; params.ignore_eos = true;
    RequestOptions opts; opts.cache_salt = 0x7761726d75702e69ULL;   // private prefix-cache namespace
    try {
        Handle h = submit(std::move(toks), params, opts);
        int steps = 0;
        for (; steps < 64; steps++) { if (!step()) break; }
        for (Request * r : sched_->all_requests()) finish(r, RequestStatus::FINISHED_ABORTED, FinishReason::ABORT, "warmup");
        bool finished = false; OutputChunk c; while (h.out->try_pop(c)) finished |= c.finished;
        LOG_INF("engine", "warmup: %u-token prefill + decode, %d steps%s in %.0f ms", n, steps, finished ? "" : " (request not finished)", (ggml_time_us() - t0) / 1000.0);
    } catch (const std::exception & e) {
        LOG_WRN("engine", "warmup failed (%s); continuing", e.what());
    }
}

void Engine::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (graph_) graph_->report();
    if (profile_ops_) report_op_profile();
    // abort everything left
    for (Request * r : sched_->all_requests()) finish(r, RequestStatus::FINISHED_ABORTED, FinishReason::ABORT, "engine shutting down");
}

Engine::Handle Engine::submit(std::vector<token_t> prompt, SamplingParams params, RequestOptions opts) {
    if (prompt.empty()) throw std::invalid_argument("prompt must not be empty");
    params.validate(model_->hparams().n_vocab);
    for (auto t : prompt) if (t < 0 || t >= (token_t) model_->hparams().n_vocab) throw std::invalid_argument("prompt token id out of range: " + std::to_string(t));
    if (prompt.size() >= max_model_len_) throw std::invalid_argument("prompt is too long: " + std::to_string(prompt.size()) + " tokens, max_model_len is " + std::to_string(max_model_len_));

    auto req = std::make_shared<Request>();
    req->id = next_id_++;
    req->opts = std::move(opts);
    req->params = std::move(params);
    req->n_prompt = (uint32_t) prompt.size();
    req->tokens = std::move(prompt);
    req->sampler = SamplerState(req->params, cfg_.seed + req->id * 0x9E3779B97F4A7C15ULL);
    req->detok = std::make_unique<IncrementalDetokenizer>(model_->tokenizer(), req->params.skip_special_tokens);
    req->out = std::make_shared<OutputQueue>();
    if (!req->params.grammar.empty() || !req->params.json_schema.empty()) {
        std::string gbnf = req->params.grammar;
        if (!req->params.json_schema.empty()) {
            nlohmann::ordered_json schema = nlohmann::ordered_json::parse(req->params.json_schema, nullptr, false);
            if (schema.is_discarded()) throw std::invalid_argument("json_schema is not valid JSON");
            try { gbnf = json_schema_to_grammar(schema); }
            catch (const std::exception & e) { throw std::invalid_argument(std::string("json_schema cannot be converted to a grammar: ") + e.what()); }
        }
        llama_grammar * g = llama_grammar_init_impl(&model_->tokenizer(), gbnf.c_str(), req->params.grammar_root.c_str(), false, nullptr, 0, nullptr, 0);
        if (!g) throw std::invalid_argument("invalid grammar (see the server log for the parse error)");
        req->grammar = std::shared_ptr<llama_grammar>(g, [](llama_grammar * p) { llama_grammar_free_impl(p); });
    }
    Handle h{req->id, req->out};
    {
        std::lock_guard<std::mutex> lk(mtx_);
        new_requests_.push_back(std::move(req));
        std::lock_guard<std::mutex> sl(stats_mtx_);
        stats_.requests_total++;
    }
    cv_.notify_all();
    return h;
}

uint32_t Engine::n_embd() const { return model_->hparams().n_embd; }

Engine::Handle Engine::submit_embedding(std::vector<token_t> prompt, Pooling pooling, bool normalize, RequestOptions opts) {
    if (prompt.empty()) throw std::invalid_argument("input must not be empty");
    for (auto t : prompt) if (t < 0 || t >= (token_t) model_->hparams().n_vocab) throw std::invalid_argument("input token id out of range: " + std::to_string(t));
    if (prompt.size() > max_model_len_) throw std::invalid_argument("input is too long: " + std::to_string(prompt.size()) + " tokens, max_model_len is " + std::to_string(max_model_len_));
    auto req = std::make_shared<Request>();
    req->id = next_id_++;
    req->opts = std::move(opts);
    req->n_prompt = (uint32_t) prompt.size();
    req->tokens = std::move(prompt);
    req->is_embedding = true;
    req->pooling = (int) pooling;
    req->normalize = normalize;
    req->params.max_tokens = 1;
    req->out = std::make_shared<OutputQueue>();
    Handle h{req->id, req->out};
    {
        std::lock_guard<std::mutex> lk(mtx_);
        new_requests_.push_back(std::move(req));
        std::lock_guard<std::mutex> sl(stats_mtx_);
        stats_.requests_total++;
    }
    cv_.notify_all();
    return h;
}

void Engine::abort(request_id_t id) {
    std::lock_guard<std::mutex> lk(mtx_);
    aborts_.push_back(id);
    cv_.notify_all();
}

EngineStats Engine::stats() const {
    std::lock_guard<std::mutex> lk(stats_mtx_);
    EngineStats s = stats_;
    s.num_running = sched_->num_running();
    s.num_waiting = sched_->num_waiting();
    s.kv_usage = kv_->usage();
    s.prefix_cache_queries = kv_->pool().stats().cache_queries;
    s.prefix_cache_hits = kv_->pool().stats().cache_hits;
    s.preemptions = sched_->num_preemptions();
    s.ttft_ms_avg = s.ttft_samples ? s.ttft_ms_sum / s.ttft_samples : 0.0;
    s.tpot_ms_avg = s.tpot_samples ? s.tpot_ms_sum / s.tpot_samples : 0.0;
    return s;
}

void Engine::run_loop() {
    LOG_DBG("engine", "engine thread started");
    while (running_) {
        bool did = false;
        try {
            did = step();
        } catch (const std::exception & e) {
            LOG_ERR("engine", "step failed: %s", e.what());
            for (Request * r : sched_->all_requests()) finish(r, RequestStatus::FINISHED_ERROR, FinishReason::ERROR, e.what());
        }
        if (!did) {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(200), [&] { return !running_ || !new_requests_.empty() || !aborts_.empty(); });
        }
        log_stats();
    }
    LOG_DBG("engine", "engine thread exiting");
}

void Engine::finish(Request * r, RequestStatus st, FinishReason why, const std::string & err) {
    if (r->is_finished()) return;
    r->finish_reason = why;
    r->t_finished = std::chrono::steady_clock::now();
    r->status = st;   // mark before the scheduler drops its (owning) reference
    OutputChunk c;
    c.finished = true;
    c.finish_reason = why;
    c.stop_reason = r->stop_reason;
    c.error = err;
    // flush held-back text
    std::string tail = r->detok ? r->detok->finish() : "";
    r->text += tail;
    if (r->text_sent < r->text.size()) { c.text = r->text.substr(r->text_sent); r->text_sent = r->text.size(); }
    c.tokens = std::move(r->pending_tokens);
    c.logprobs = std::move(r->pending_logprobs);
    c.n_prompt_tokens = r->n_prompt;
    c.n_output_tokens = r->n_output();
    c.n_cached_tokens = r->num_cached_tokens;
    if (r->is_embedding) c.embedding = std::move(r->embd_acc);
    r->out->push(std::move(c));
    LOG_DBG("engine", "request %llu finished: %s (%u prompt, %u generated, %u cached)", (unsigned long long) r->id,
            finish_reason_str(why), r->n_prompt, r->n_output(), r->num_cached_tokens);
    if (draft_) draft_->release(*r);
    sched_->finish_request(r, st);   // frees KV and destroys the request: r is dangling from here on
    {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.requests_finished++;
        if (why == FinishReason::ABORT) stats_.requests_aborted++;
    }
}

void Engine::deliver(Request * r, bool final) {
    (void) final;
    // stop-string holdback: never stream the tail that could be the start of a stop string
    size_t holdback = 0;
    for (auto & s : r->params.stop) holdback = std::max(holdback, s.size() > 0 ? s.size() - 1 : 0);
    size_t send_to = r->text.size() > holdback ? r->text.size() - holdback : 0;
    if (send_to <= r->text_sent && r->pending_tokens.empty()) return;
    OutputChunk c;
    if (send_to > r->text_sent) { c.text = r->text.substr(r->text_sent, send_to - r->text_sent); r->text_sent = send_to; }
    c.tokens = std::move(r->pending_tokens);
    c.logprobs = std::move(r->pending_logprobs);
    r->pending_tokens.clear();
    r->pending_logprobs.clear();
    c.n_prompt_tokens = r->n_prompt;
    c.n_output_tokens = r->n_output();
    r->out->push(std::move(c));
}

bool Engine::step() {
    // ---- ingest ----
    {
        std::vector<std::shared_ptr<Request>> fresh;
        std::vector<request_id_t> aborts;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            fresh.swap(new_requests_);
            aborts.swap(aborts_);
        }
        for (auto & r : fresh) sched_->add_request(std::move(r));
        for (auto id : aborts) {
            for (Request * r : sched_->all_requests()) if (r->id == id) { finish(r, RequestStatus::FINISHED_ABORTED, FinishReason::ABORT); break; }
        }
    }
    if (!sched_->has_requests()) return false;

    // ---- schedule ----
    const int64_t t_sched0 = ggml_time_us();
    SchedulerOutput so = sched_->schedule();
    const int64_t t_sched1 = ggml_time_us();
    if (so.scheduled.empty()) {
        if (!so.preempted.empty()) return true;
        // nothing schedulable: requests waiting cannot fit even alone -> fail them
        for (Request * r : sched_->all_requests()) {
            if (r->status == RequestStatus::WAITING && sched_->num_running() == 0) {
                finish(r, RequestStatus::FINISHED_ERROR, FinishReason::ERROR, "request does not fit in the KV cache");
            }
        }
        return false;
    }

    // ---- build micro-batch ----
    // gather attention groups sequences with equal token counts into contiguous token ranges of the batch
    if (gather_attn_) std::stable_sort(so.scheduled.begin(), so.scheduled.end(), [](const ScheduledRequest & a, const ScheduledRequest & b) { return a.n_new_tokens > b.n_new_tokens; });
    UBatch ub;
    ub.n_tokens = so.total_tokens;
    ub.tokens.reserve(ub.n_tokens); ub.pos.reserve(ub.n_tokens); ub.seq_idx.reserve(ub.n_tokens); ub.slots.reserve(ub.n_tokens);
    std::vector<int32_t> logit_rows;   // scheduled index -> output row (-1 if none)
    std::vector<std::pair<int32_t, int32_t>> embd_rows;   // scheduled index -> (first output row, count) for embedding requests
    int64_t cell_min = INT64_MAX, cell_max = -1;
    for (size_t si = 0; si < so.scheduled.size(); si++) {
        const auto & s = so.scheduled[si];
        Request * r = s.req;
        UBatch::SeqInfo seq;
        const uint32_t n_ctx_tokens = r->num_computed_tokens + s.n_new_tokens;
        seq.cells.reserve(n_ctx_tokens); seq.cell_pos.reserve(n_ctx_tokens);
        for (uint32_t i = 0; i < n_ctx_tokens; i++) {
            const int64_t c = kv_->cell_of(r->kv, i);
            seq.cells.push_back(c); seq.cell_pos.push_back((pos_t) i);
            cell_min = std::min(cell_min, c); cell_max = std::max(cell_max, c);
        }
        for (uint32_t i = r->num_computed_tokens; i < n_ctx_tokens; i++) {
            ub.tokens.push_back(i < r->n_tokens() ? r->tokens[i] : r->spec_tokens[i - r->n_tokens()]);
            ub.pos.push_back((pos_t) i);
            ub.seq_idx.push_back((int32_t) si);
            ub.slots.push_back(kv_->cell_of(r->kv, i));
        }
        if (r->is_embedding) {
            // embeddings need the hidden state of every token of this chunk
            embd_rows.push_back({(int32_t) ub.out_ids.size(), (int32_t) s.n_new_tokens});
            for (uint32_t i = 0; i < s.n_new_tokens; i++) ub.out_ids.push_back((int32_t) ub.tokens.size() - s.n_new_tokens + i);
            logit_rows.push_back(-1);
        } else {
            embd_rows.push_back({-1, 0});
            const bool needs_logits = n_ctx_tokens >= r->n_tokens();
            if (needs_logits) {
                // one row for the last real token plus one per draft token (verification)
                logit_rows.push_back((int32_t) ub.out_ids.size());
                const int32_t first = (int32_t) ub.tokens.size() - 1 - (int32_t) s.n_spec;
                for (uint32_t k = 0; k <= s.n_spec; k++) ub.out_ids.push_back(first + (int32_t) k);
            } else logit_rows.push_back(-1);
        }
        ub.seqs.push_back(std::move(seq));
    }
    ub.n_outputs = (uint32_t) ub.out_ids.size();
    if (ub.n_outputs == 0) { ub.out_ids.push_back(0); ub.n_outputs = 1; }   // keep graph topology valid
    // attention window: [kv_start, kv_start + n_kv), padded to 256 for graph stability
    ub.kv_start = (uint32_t) ((cell_min / 256) * 256);
    ub.n_kv = std::min<uint32_t>(kv_->num_cells() - ub.kv_start, (uint32_t) (((cell_max + 1 - ub.kv_start + 255) / 256) * 256));

    // ---- build (or reuse) + compute graph ----
    const int64_t t0 = ggml_time_us();
    GraphState & g = *graph_;
    // per-step attention path: the paged kernel parallelises over (token, head) pairs, so tiny decode batches
    // on models with few heads run faster through ggml's fused kernel; everything else goes paged.
    const bool paged_step = paged_attn_ && (cfg_.attention_force_paged ||
                            (uint64_t) ub.n_tokens * model_->hparams().n_head_max() >= 2ull * (uint64_t) cfg_.n_threads);
    // gather attention pays per-layer K/V gathers to keep each sequence's attention O(its own length); a step with
    // a single contiguous sequence already has a window of exactly that sequence, so it takes the plain masked
    // path (no gathers). Multiple sequences, or one sequence spread over a wide window, use gather.
    bool gather_step = false;
    if (gather_attn_) {
        uint32_t n_active = 0, sum_len = 0;
        for (const auto & sq : ub.seqs) if (!sq.cells.empty()) { n_active++; sum_len += (uint32_t) sq.cells.size(); }
        gather_step = n_active > 1 || ub.n_kv > sum_len + 256;
    }
    const uint64_t layout = gather_step ? attn_layout_hash(attn_groups(ub)) : 0;
    const bool reuse = g.key.valid && g.key.n_tokens == ub.n_tokens && g.key.n_outputs == ub.n_outputs
                       && g.key.n_kv == ub.n_kv && g.key.kv_start == ub.kv_start && g.key.paged == paged_step && g.key.layout == layout
                       && g.key.gather == gather_step;
    g.ub = std::move(ub);
    if (!reuse) {
        if (g.ctx) { ggml_free(g.ctx); g.ctx = nullptr; }
        ggml_init_params ip = { g.meta.size(), g.meta.data(), /*no_alloc*/ true };
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, max_nodes_, false);
        g.gc = std::make_unique<GraphContext>(g.ctx, g.gf, *model_, g.ub, *kv_, cfg_.flash_attn, paged_step, gather_step);
        model_->arch().build_graph(*g.gc);
        ggml_backend_sched_reset(gsched_);
        if (!ggml_backend_sched_alloc_graph(gsched_, g.gf)) throw std::runtime_error("failed to allocate compute graph");
        g.key = { g.ub.n_tokens, g.ub.n_outputs, g.ub.n_kv, g.ub.kv_start, paged_step, true, layout, gather_step };
        g.n_built++;
    } else {
        g.n_reused++;
    }
    const int64_t t_built = ggml_time_us();
    g.gc->set_inputs();
    const int64_t t_inputs = ggml_time_us();
    // IIAN_DUMP_TENSORS=1: print name/shape/sum of every graph node (debugging aid, compare with llama-eval-callback)
    static const bool dump_tensors = getenv("IIAN_DUMP_TENSORS") != nullptr;
    if (dump_tensors) {
        ggml_backend_sched_set_eval_callback(gsched_, [](ggml_tensor * t, bool ask, void *) -> bool {
            if (ask) return true;
            if (!ggml_backend_buffer_is_host(t->buffer)) return true;
            std::vector<float> tmp;
            const int64_t n = ggml_nelements(t);
            const float * data = nullptr;
            if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) data = (const float *) t->data;
            else if (ggml_is_contiguous(t) && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 || ggml_get_type_traits(t->type)->to_float)) {
                tmp.resize(n); ggml_get_type_traits(t->type)->to_float(t->data, tmp.data(), n); data = tmp.data();
            }
            if (!data) { fprintf(stderr, "iian_debug: %-24s %-8s [%lld,%lld,%lld,%lld] (skipped)\n", ggml_get_name(t), ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]); return true; }
            double sum = 0; for (int64_t i = 0; i < n; i++) sum += data[i];
            fprintf(stderr, "iian_debug: %-24s %-8s [%lld,%lld,%lld,%lld] sum = %.6f first = %.6f %.6f %.6f\n", ggml_get_name(t), ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], sum, data[0], n > 1 ? data[1] : 0.0f, n > 2 ? data[2] : 0.0f);
            return true;
        }, nullptr);
    }
    if (dump_tensors_) ggml_backend_sched_set_eval_callback(gsched_, dump_tensor_cb, nullptr);
    else if (profile_ops_) ggml_backend_sched_set_eval_callback(gsched_, profile_op_cb, nullptr);
    ggml_status st = ggml_backend_sched_graph_compute(gsched_, g.gf);
    if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed with status " + std::to_string((int) st));

    // ---- fetch logits ----
    const uint32_t n_vocab = model_->hparams().n_vocab;
    std::vector<float> & logits = logits_buf_;   // reused: a fresh >128 KiB vector per step would be mmap'd and page-faulted every time
    logits.resize((size_t) g.ub.n_outputs * n_vocab);
    ggml_backend_tensor_get(g.gc->t_logits, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_sched_synchronize(gsched_);
    const int64_t t1 = ggml_time_us();
    LOG_TRC("engine", "step: %u tokens, %zu seqs, n_kv=%u, %u outputs, %.1f ms%s", g.ub.n_tokens, so.scheduled.size(), g.ub.n_kv, g.ub.n_outputs, (t1 - t0) / 1000.0, reuse ? " (graph reused)" : "");

    // ---- sample + outputs ----
    sched_->update_after_step(so);
    process_embeddings(so, embd_rows);
    process_outputs(so, logits, logit_rows);
    {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.steps++;
    }
    if (g.prof.on) {
        const int64_t t2 = ggml_time_us();
        auto & pf = g.prof;
        pf.steps++; pf.tokens += g.ub.n_tokens;
        pf.us_sched += t_sched1 - t_sched0; pf.us_batch += t0 - t_sched1; pf.us_build += t_built - t0;
        pf.us_inputs += t_inputs - t_built; pf.us_compute += t1 - t_inputs; pf.us_outputs += t2 - t1;
        pf.n_splits = ggml_backend_sched_get_n_splits(gsched_); pf.n_copies = ggml_backend_sched_get_n_copies(gsched_);
        if (pf.steps % 500 == 0) g.report();
    }
    return true;
}

void Engine::process_outputs(const SchedulerOutput & so, std::vector<float> & logits_all, const std::vector<int32_t> & logit_rows) {
    const uint32_t n_vocab = model_->hparams().n_vocab;
    const Tokenizer & tok = model_->tokenizer();
    const auto now = std::chrono::steady_clock::now();
    uint64_t n_prompt_done = 0, n_gen = 0;

    for (size_t si = 0; si < so.scheduled.size(); si++) {
        const auto & s = so.scheduled[si];
        Request * r = s.req;
        if (r->is_finished()) continue;
        const bool was_prefill = r->num_computed_tokens - s.n_new_tokens < r->n_prompt;
        if (was_prefill) n_prompt_done += std::min<uint32_t>(s.n_new_tokens, r->n_prompt - (r->num_computed_tokens - s.n_new_tokens));
        if (r->is_embedding || logit_rows[si] < 0) continue;   // embedding request / prefill chunk: no token yet

        const SamplingParams & p = r->params;
        const uint32_t n_spec = s.n_spec;
        uint32_t n_accepted = 0;
        bool stop = false;
        FinishReason why = FinishReason::NONE;
        for (uint32_t k = 0; k <= n_spec && !stop; k++) {
        // each output row is sampled exactly once, so the sampler works on it in place (no per-token copy)
        float * scratch = logits_all.data() + (size_t) (logit_rows[si] + k) * n_vocab;
        // min_tokens: forbid EOS / stop tokens
        if ((int32_t) r->n_output() < p.min_tokens) {
            for (uint32_t t = 0; t < n_vocab; t++) if (tok.is_eog((token_t) t)) scratch[t] = -INFINITY;
            for (auto t : p.stop_token_ids) scratch[t] = -INFINITY;
        }
        const token_t * prompt_view = r->tokens.data();
        const size_t    n_prompt    = r->n_prompt;
        const token_t * output_view = r->tokens.data() + r->n_prompt;
        const size_t    n_output    = r->tokens.size() - r->n_prompt;
        SampledToken st;
        const int64_t t_s0 = graph_->prof.on ? ggml_time_us() : 0;
        if (r->grammar) {
            // llama.cpp strategy: sample first and only pay for the full-vocabulary grammar mask when the
            // sampled token is not acceptable
            std::vector<float> backup(scratch, scratch + n_vocab);
            st = r->sampler.sample(scratch, (int32_t) n_vocab, prompt_view, n_prompt, output_view, n_output);
            if (!llama_grammar_would_accept(*r->grammar, st.token)) {
                llama_grammar_apply_impl(*r->grammar, backup.data(), (int32_t) n_vocab);
                st = r->sampler.sample(backup.data(), (int32_t) n_vocab, prompt_view, n_prompt, output_view, n_output);
            }
            llama_grammar_accept_impl(*r->grammar, st.token);
        } else {
            st = r->sampler.sample(scratch, (int32_t) n_vocab, prompt_view, n_prompt, output_view, n_output);
        }
        if (graph_->prof.on) graph_->prof.us_sample += ggml_time_us() - t_s0;
        // speculative verification: the sample at a draft position is kept only if it equals the draft;
        // a mismatch ends the step (the sampled token is the recovered token)
        const bool verifying = k < n_spec;
        const bool accepted = verifying && st.token == r->spec_tokens[k];
        if (accepted) n_accepted++;
        r->tokens.push_back(st.token);
        r->pending_tokens.push_back(st.token);
        if (p.logprobs >= 0) r->pending_logprobs.push_back(st);
        n_gen++;

        // timings
        if (r->n_output() == 1) {
            r->t_first_token = now;
            std::lock_guard<std::mutex> lk(stats_mtx_);
            stats_.ttft_ms_sum += std::chrono::duration<double, std::milli>(now - r->opts.arrival).count();
            stats_.ttft_samples++;
        }

        // ---- stop checks (token-level) ----
        const bool is_eog = tok.is_eog(st.token);
        if (is_eog && !p.ignore_eos) { stop = true; why = FinishReason::STOP; }
        if (!stop && std::find(p.stop_token_ids.begin(), p.stop_token_ids.end(), st.token) != p.stop_token_ids.end()) {
            stop = true; why = FinishReason::STOP; r->stop_reason = std::to_string(st.token);
        }
        // detokenize (skip the stop token's text unless include_stop_str_in_output)
        if (!(stop && !p.include_stop_str_in_output && !is_eog) && !(is_eog && p.skip_special_tokens)) {
            r->text += r->detok->push(st.token);
        } else if (!is_eog) {
            r->text += r->detok->push(st.token);
        }
        // ---- stop strings ----
        if (!stop && !p.stop.empty() && (int32_t) r->n_output() >= p.min_tokens) {
            size_t best_end = std::string::npos, best_start = 0; std::string best;
            for (const auto & sstr : p.stop) {
                if (sstr.empty()) continue;
                size_t from = r->stop_check_offset > sstr.size() ? r->stop_check_offset - sstr.size() : 0;
                size_t pos = r->text.find(sstr, from);
                if (pos != std::string::npos && pos + sstr.size() < best_end) { best_end = pos + sstr.size(); best_start = pos; best = sstr; }
            }
            if (best_end != std::string::npos) {
                stop = true; why = FinishReason::STOP; r->stop_reason = best;
                r->text.resize(p.include_stop_str_in_output ? best_end : best_start);
                r->detok->finish();
            }
            r->stop_check_offset = r->text.size();
        }
        if (!stop && p.max_tokens > 0 && (int32_t) r->n_output() >= p.max_tokens) { stop = true; why = FinishReason::LENGTH; }
        if (!stop && r->n_tokens() >= max_model_len_) { stop = true; why = FinishReason::LENGTH; }
        if (verifying && !accepted) break;   // recovered token appended; remaining drafts are rejected
        }   // rows
        if (n_spec > 0) {
            // rejected drafts occupy KV cells past the accepted length: roll the computed count back so
            // they are recomputed (overwritten) next step
            const uint32_t n_rejected = n_spec - n_accepted;
            r->num_computed_tokens -= n_rejected;
            std::lock_guard<std::mutex> lk(stats_mtx_);
            stats_.spec_drafted += n_spec;
            stats_.spec_accepted += n_accepted;
        }
        r->spec_tokens.clear();
        if (!stop && !draft_ && cfg_.spec_ngram > 0 && !r->grammar) r->spec_tokens = ngram_draft(r->tokens);

        if (stop) finish(r, why == FinishReason::LENGTH ? RequestStatus::FINISHED_LENGTH : RequestStatus::FINISHED_STOPPED, why);
        else deliver(r, false);
    }
    if (draft_) {
        std::vector<Request *> cands;
        for (const auto & s : so.scheduled) if (!s.req->is_finished() && !s.req->is_embedding && !s.req->grammar && s.req->num_computed_tokens + 1 == s.req->n_tokens()) cands.push_back(s.req);
        if (!cands.empty()) {
            try { draft_->draft(cands); }
            catch (const std::exception & e) { LOG_WRN("spec", "draft step failed (%s); continuing without drafts", e.what()); for (auto * r : cands) r->spec_tokens.clear(); }
        }
    }
    std::lock_guard<std::mutex> lk(stats_mtx_);
    stats_.prompt_tokens += n_prompt_done;
    stats_.generation_tokens += n_gen;
}

std::vector<token_t> Engine::ngram_draft(const std::vector<token_t> & toks) const {
    std::vector<token_t> out;
    const size_t n_tok = toks.size();
    const size_t k = cfg_.spec_ngram;
    const size_t search_from = n_tok > 8192 ? n_tok - 8192 : 0;   // bound the scan for very long contexts
    for (size_t n = cfg_.spec_ngram_max_n; n >= std::max<uint32_t>(1, cfg_.spec_ngram_min_n); n--) {
        if (n_tok < n + 1) continue;
        const token_t * suffix = toks.data() + n_tok - n;
        // most recent earlier occurrence of the suffix n-gram
        for (size_t i = n_tok - n; i-- > search_from;) {
            if (memcmp(toks.data() + i, suffix, n * sizeof(token_t)) != 0) continue;
            const size_t start = i + n;
            for (size_t j = start; j < n_tok && out.size() < k; j++) out.push_back(toks[j]);
            if (!out.empty()) return out;
        }
    }
    return out;
}

void Engine::process_embeddings(const SchedulerOutput & so, const std::vector<std::pair<int32_t, int32_t>> & embd_rows) {
    bool any = false;
    for (auto & e : embd_rows) if (e.first >= 0) { any = true; break; }
    if (!any) return;
    GraphState & g = *graph_;
    ggml_tensor * t = g.gc->t_embd;
    const int64_t n_embd = t->ne[0];
    std::vector<float> rows((size_t) t->ne[0] * t->ne[1]);
    ggml_backend_tensor_get(t, rows.data(), 0, rows.size() * sizeof(float));
    for (size_t si = 0; si < so.scheduled.size(); si++) {
        Request * r = so.scheduled[si].req;
        if (!r->is_embedding || r->is_finished()) continue;
        const auto [first, count] = embd_rows[si];
        if (r->embd_acc.empty()) r->embd_acc.assign((size_t) n_embd, 0.0f);
        const bool first_chunk = r->embd_count == 0;
        for (int32_t k = 0; k < count; k++) {
            const float * row = rows.data() + (size_t) (first + k) * n_embd;
            switch ((Pooling) r->pooling) {
                case Pooling::MEAN: for (int64_t d = 0; d < n_embd; d++) r->embd_acc[d] += row[d]; break;
                case Pooling::LAST: std::copy(row, row + n_embd, r->embd_acc.begin()); break;
                case Pooling::CLS:  if (first_chunk && k == 0) std::copy(row, row + n_embd, r->embd_acc.begin()); break;
            }
        }
        r->embd_count += (uint32_t) count;
        if (r->num_computed_tokens >= r->n_tokens()) {
            if ((Pooling) r->pooling == Pooling::MEAN && r->embd_count) for (auto & v : r->embd_acc) v /= (float) r->embd_count;
            if (r->normalize) {
                double n = 0; for (float v : r->embd_acc) n += (double) v * v;
                const float inv = n > 0 ? (float) (1.0 / std::sqrt(n)) : 0.0f;
                for (auto & v : r->embd_acc) v *= inv;
            }
            {
                std::lock_guard<std::mutex> lk(stats_mtx_);
                stats_.prompt_tokens += r->n_prompt;
            }
            finish(r, RequestStatus::FINISHED_STOPPED, FinishReason::STOP);
        }
    }
}

void Engine::log_stats() {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - t_last_stats_).count();
    if (dt < 10.0) return;
    t_last_stats_ = now;
    EngineStats s = stats();
    const double ptps = (s.prompt_tokens - stats_prompt_tokens_prev_) / dt;
    const double gtps = (s.generation_tokens - stats_gen_tokens_prev_) / dt;
    stats_prompt_tokens_prev_ = s.prompt_tokens;
    stats_gen_tokens_prev_ = s.generation_tokens;
    {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.prompt_tps = ptps;
        stats_.generation_tps = gtps;
    }
    if (s.num_running == 0 && s.num_waiting == 0 && ptps == 0 && gtps == 0) return;
    const double hit = s.prefix_cache_queries ? 100.0 * s.prefix_cache_hits / s.prefix_cache_queries : 0.0;
    const double acc = s.spec_drafted ? 100.0 * s.spec_accepted / s.spec_drafted : 0.0;
    LOG_INF("engine", "prompt %.1f tok/s | gen %.1f tok/s | running %zu | waiting %zu | kv %.1f%% | prefix-hit %.1f%% | preempt %llu | ttft %.0fms%s",
            ptps, gtps, s.num_running, s.num_waiting, 100.0 * s.kv_usage, hit, (unsigned long long) s.preemptions, s.ttft_ms_avg,
            (cfg_.spec_ngram || draft_) ? (" | spec-accept " + std::to_string((int) acc) + "%").c_str() : "");
}

} // namespace iian
