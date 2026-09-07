#pragma once
// vLLM-v1-style scheduler: no prefill/decode phases; each step assigns tokens to requests so that
// num_computed_tokens catches up with tokens.size(). Chunked prefill, prefix caching and preemption
// (recompute) fall out of that single invariant.
#include "iian/kv_cache.h"
#include "iian/request.h"

#include <deque>
#include <memory>
#include <vector>

namespace iian {

enum class SchedulingPolicy { FCFS, PRIORITY };

struct SchedulerConfig {
    uint32_t max_num_seqs = 32;             // max concurrently running requests
    uint32_t max_num_batched_tokens = 1024; // token budget per step (chunked prefill granularity)
    uint32_t max_model_len = 0;             // per-request context limit (prompt + output)
    uint32_t long_prefill_token_threshold = 0;   // 0 = off; caps one request's prefill chunk
    bool     enable_chunked_prefill = true;
    SchedulingPolicy policy = SchedulingPolicy::FCFS;
};

struct ScheduledRequest {
    Request * req;
    uint32_t  n_new_tokens;     // tokens computed this step for this request (real + draft)
    uint32_t  n_spec = 0;       // how many of them are draft tokens (verified after the step)
    bool      resumed;          // came back from preemption this step
};

struct SchedulerOutput {
    std::vector<ScheduledRequest> scheduled;
    std::vector<Request *> preempted;
    uint32_t total_tokens = 0;
};

class Scheduler {
public:
    Scheduler(const SchedulerConfig & cfg, PagedKVCache & kv);

    void add_request(std::shared_ptr<Request> req);
    // Called by the engine after a request finished or was aborted; frees KV and removes it.
    void finish_request(Request * req, RequestStatus status);
    SchedulerOutput schedule();
    // Advance num_computed_tokens for scheduled requests and commit prefix-cache blocks.
    void update_after_step(const SchedulerOutput & out);

    size_t num_waiting() const { return waiting_.size(); }
    size_t num_running() const { return running_.size(); }
    bool has_requests() const { return !waiting_.empty() || !running_.empty(); }
    const SchedulerConfig & config() const { return cfg_; }
    uint64_t num_preemptions() const { return n_preemptions_; }
    std::vector<Request *> all_requests() const;

private:
    void preempt(Request * req);
    SchedulerConfig cfg_;
    PagedKVCache & kv_;
    std::deque<std::shared_ptr<Request>> waiting_;
    std::vector<std::shared_ptr<Request>> running_;
    uint64_t n_preemptions_ = 0;
};

} // namespace iian
