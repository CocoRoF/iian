#pragma once
// Engine: owns the model, backends, paged KV cache and scheduler; runs the step loop on its own thread.
#include "iian/kv_cache.h"
#include "iian/model.h"
#include "iian/request.h"
#include "iian/scheduler.h"
#include "iian/types.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ggml_backend;
struct ggml_backend_sched;

namespace iian { class DraftModel; }

namespace iian {

struct EngineConfig {
    uint32_t max_model_len = 0;          // 0 -> model's training context
    uint32_t kv_cache_tokens = 0;        // total KV cells; 0 -> max_model_len * max_num_seqs (capped by kv_cache_bytes)
    size_t   kv_cache_bytes = 0;         // optional memory budget for KV (0 = no cap)
    uint32_t block_size = 16;
    std::string kv_dtype = "f16";        // f16 | bf16 | q8_0 | q4_0 | f32
    bool     flash_attn = true;
    std::string attention = "auto";     // auto | masked | paged | gather  (paged = CPU kernel; gather = per-sequence batched flash attention)
    bool attention_force_paged = false;  // set when attention == "paged": use the kernel even for tiny batches
    bool warmup = true;                  // run a small prefill + decode inside start() so kernels, cuBLAS and CUDA graphs are initialised before the first request
    bool     enable_prefix_caching = true;
    uint64_t seed = 0;
    // speculative decoding via prompt lookup (n-gram): propose up to spec_ngram tokens that followed the most recent
    // occurrence of the current n-gram suffix (n in [spec_ngram_min_n, spec_ngram_max_n]) in the request's own
    // tokens. Exact: accepted only where the target sample equals the draft (one-hot rejection sampling).
    uint32_t spec_ngram = 0;           // draft tokens per step; 0 = off
    uint32_t spec_ngram_min_n = 1;
    uint32_t spec_ngram_max_n = 4;
    // speculative decoding with a draft model (same tokenizer); overrides spec_ngram when set
    std::string spec_draft_model;
    uint32_t spec_draft_n = 5;
    int      n_threads = -1;
    int      n_threads_batch = -1;
    SchedulerConfig sched;
    DeviceConfig device;
};

struct EngineStats {
    uint64_t requests_total = 0, requests_finished = 0, requests_aborted = 0;
    uint64_t prompt_tokens = 0, generation_tokens = 0, cached_prompt_tokens = 0;
    uint64_t steps = 0, preemptions = 0;
    uint64_t spec_drafted = 0, spec_accepted = 0;
    size_t   num_running = 0, num_waiting = 0;
    float    kv_usage = 0.0f;
    uint64_t prefix_cache_queries = 0, prefix_cache_hits = 0;
    double   prompt_tps = 0.0, generation_tps = 0.0;   // rolling since last stats window
    double   ttft_ms_avg = 0.0, tpot_ms_avg = 0.0;
    uint64_t ttft_samples = 0, tpot_samples = 0;
    double   ttft_ms_sum = 0.0, tpot_ms_sum = 0.0;
};

class Engine {
public:
    Engine(std::shared_ptr<Model> model, const EngineConfig & cfg);
    ~Engine();

    void start();   // spawns the engine thread
    void stop();    // aborts everything and joins

    // Submit a tokenized prompt. Returns the output queue; the id is available via handle.
    struct Handle { request_id_t id; std::shared_ptr<OutputQueue> out; };
    Handle submit(std::vector<token_t> prompt, SamplingParams params, RequestOptions opts = {});
    // Embeddings: run the prompt through the model and pool the final hidden states (no generation).
    enum class Pooling { MEAN = 0, LAST = 1, CLS = 2 };
    Handle submit_embedding(std::vector<token_t> prompt, Pooling pooling, bool normalize, RequestOptions opts = {});
    uint32_t n_embd() const;
    void abort(request_id_t id);

    const Model & model() const { return *model_; }
    const EngineConfig & config() const { return cfg_; }
    const PagedKVCache & kv() const { return *kv_; }
    uint32_t max_model_len() const { return max_model_len_; }
    EngineStats stats() const;
    std::string backend_summary() const;

    // Run one scheduler/compute step synchronously (used by tests and by the thread loop).
    bool step();
    void warmup();

private:
    struct GraphState;
    void run_loop();
    void process_outputs(const SchedulerOutput & so, std::vector<float> & logits, const std::vector<int32_t> & logit_rows);   // logits rows are consumed in place
    std::vector<token_t> ngram_draft(const std::vector<token_t> & toks) const;
    void process_embeddings(const SchedulerOutput & so, const std::vector<std::pair<int32_t, int32_t>> & embd_rows);
    void finish(Request * r, RequestStatus st, FinishReason why, const std::string & err = "");
    void deliver(Request * r, bool final);
    void log_stats();

    std::shared_ptr<Model> model_;
    EngineConfig cfg_;
    uint32_t max_model_len_ = 0;
    std::unique_ptr<PagedKVCache> kv_;
    std::unique_ptr<Scheduler> sched_;
    std::vector<ggml_backend *> backends_;
    ggml_backend * backend_cpu_ = nullptr;
    ggml_backend_sched * gsched_ = nullptr;
    std::unique_ptr<GraphState> graph_;
    size_t max_nodes_ = 0;
    bool paged_attn_ = false;
    bool gather_attn_ = false;
    std::vector<float> logits_buf_;   // per-step logits, kept allocated across steps (rows are sampled in place)
    bool dump_tensors_ = false;
    bool profile_ops_ = false;
    std::unique_ptr<DraftModel> draft_;

    // inbound queues
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<std::shared_ptr<Request>> new_requests_;
    std::vector<request_id_t> aborts_;
    std::atomic<request_id_t> next_id_{1};
    std::atomic<bool> running_{false};
    std::thread thread_;

    mutable std::mutex stats_mtx_;
    EngineStats stats_;
    std::chrono::steady_clock::time_point t_last_stats_;
    uint64_t stats_prompt_tokens_prev_ = 0, stats_gen_tokens_prev_ = 0;
};

} // namespace iian
