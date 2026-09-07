#include "iian/scheduler.h"
#include "iian/log.h"

#include <algorithm>

namespace iian {

const char * finish_reason_str(FinishReason r) {
    switch (r) {
        case FinishReason::STOP: return "stop";
        case FinishReason::LENGTH: return "length";
        case FinishReason::ABORT: return "abort";
        case FinishReason::ERROR: return "error";
        case FinishReason::TOOL_CALLS: return "tool_calls";
        default: return "";
    }
}

void OutputQueue::push(OutputChunk c) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (c.finished) finished_ = true;
    q_.push_back(std::move(c));
    cv_.notify_all();
}
bool OutputQueue::pop(OutputChunk & out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (!cv_.wait_for(lk, timeout, [&] { return !q_.empty(); })) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}
bool OutputQueue::try_pop(OutputChunk & out) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
}

Scheduler::Scheduler(const SchedulerConfig & cfg, PagedKVCache & kv) : cfg_(cfg), kv_(kv) {}

void Scheduler::add_request(std::shared_ptr<Request> req) {
    req->status = RequestStatus::WAITING;
    if (cfg_.policy == SchedulingPolicy::PRIORITY) {
        auto pos = std::upper_bound(waiting_.begin(), waiting_.end(), req, [](const auto & a, const auto & b) {
            return std::make_pair(a->opts.priority, a->opts.arrival) < std::make_pair(b->opts.priority, b->opts.arrival);
        });
        waiting_.insert(pos, std::move(req));
    } else {
        waiting_.push_back(std::move(req));
    }
}

std::vector<Request *> Scheduler::all_requests() const {
    std::vector<Request *> out;
    for (auto & r : running_) out.push_back(r.get());
    for (auto & r : waiting_) out.push_back(r.get());
    return out;
}

void Scheduler::finish_request(Request * req, RequestStatus status) {
    req->status = status;
    kv_.free(req->kv);
    auto it = std::find_if(running_.begin(), running_.end(), [&](auto & p) { return p.get() == req; });
    if (it != running_.end()) { running_.erase(it); return; }
    auto wt = std::find_if(waiting_.begin(), waiting_.end(), [&](auto & p) { return p.get() == req; });
    if (wt != waiting_.end()) waiting_.erase(wt);
}

void Scheduler::preempt(Request * req) {
    kv_.free(req->kv);
    req->status = RequestStatus::PREEMPTED;
    req->num_computed_tokens = 0;
    req->num_preemptions++;
    n_preemptions_++;
    auto it = std::find_if(running_.begin(), running_.end(), [&](auto & p) { return p.get() == req; });
    std::shared_ptr<Request> sp = *it;
    running_.erase(it);
    if (cfg_.policy == SchedulingPolicy::PRIORITY) { add_request(sp); sp->status = RequestStatus::PREEMPTED; }
    else waiting_.push_front(sp);
    LOG_WRN("sched", "preempted request %llu (%u tokens computed) due to KV pressure", (unsigned long long) req->id, req->n_tokens());
}

SchedulerOutput Scheduler::schedule() {
    SchedulerOutput out;
    uint32_t budget = cfg_.max_num_batched_tokens;

    // ---------------- phase 1: running requests (decodes + in-flight prefill chunks) ----------------
    size_t idx = 0;
    while (idx < running_.size() && budget > 0) {
        Request * req = running_[idx].get();
        uint32_t n_new = req->n_tokens() - req->num_computed_tokens;
        if (cfg_.long_prefill_token_threshold > 0) n_new = std::min(n_new, cfg_.long_prefill_token_threshold);
        n_new = std::min(n_new, budget);
        if (n_new == 0) { idx++; continue; }
        // speculative drafts ride along only when the whole (real + draft) window fits the budget and the context
        uint32_t n_spec = 0;
        if (!req->spec_tokens.empty() && n_new == req->n_tokens() - req->num_computed_tokens) {
            n_spec = (uint32_t) req->spec_tokens.size();
            if (n_new + n_spec > budget) n_spec = budget - n_new;
            if (req->num_computed_tokens + n_new + n_spec + 1 > cfg_.max_model_len) n_spec = std::min<uint32_t>(n_spec, cfg_.max_model_len > req->num_computed_tokens + n_new + 1 ? cfg_.max_model_len - req->num_computed_tokens - n_new - 1 : 0);
            n_new += n_spec;
        }

        bool ok;
        while (true) {
            ok = kv_.allocate_slots(req->kv, req->num_computed_tokens + n_new, {});
            if (ok) break;
            // OOM: preempt the lowest-priority running request (last in the list)
            Request * victim = running_.back().get();
            if (victim == req) { preempt(req); out.preempted.push_back(req); break; }
            // refund victim if already scheduled this step
            auto sit = std::find_if(out.scheduled.begin(), out.scheduled.end(), [&](auto & s) { return s.req == victim; });
            if (sit != out.scheduled.end()) { budget += sit->n_new_tokens; out.total_tokens -= sit->n_new_tokens; out.scheduled.erase(sit); }
            preempt(victim);
            out.preempted.push_back(victim);
        }
        if (!ok) break;   // req itself was preempted: stop scheduling

        out.scheduled.push_back({req, n_new, n_spec, false});
        out.total_tokens += n_new;
        budget -= n_new;
        idx++;
    }

    // ---------------- phase 2: waiting requests (never in a step that preempted) ----------------
    if (out.preempted.empty()) {
        while (!waiting_.empty() && budget > 0 && running_.size() < cfg_.max_num_seqs) {
            std::shared_ptr<Request> sp = waiting_.front();
            Request * req = sp.get();

            std::vector<int32_t> computed_blocks;
            uint32_t n_computed = req->num_computed_tokens;
            if (n_computed == 0) {
                kv_.update_hashes(req->block_hashes, req->tokens, req->opts.cache_salt);
                n_computed = kv_.get_computed_blocks(req->tokens, req->block_hashes, computed_blocks);
                kv_.pool().stat_query((uint32_t) (req->tokens.size() / kv_.block_size()), n_computed / kv_.block_size());
            }
            uint32_t n_new = req->n_tokens() - n_computed;
            if (cfg_.long_prefill_token_threshold > 0) n_new = std::min(n_new, cfg_.long_prefill_token_threshold);
            if (!cfg_.enable_chunked_prefill && n_new > budget) break;
            n_new = std::min(n_new, budget);
            if (n_new == 0) break;

            if (!kv_.allocate_slots(req->kv, n_computed + n_new, computed_blocks)) break;   // no preemption from the waiting queue

            waiting_.pop_front();
            const bool resumed = req->status == RequestStatus::PREEMPTED;
            req->status = RequestStatus::RUNNING;
            req->num_computed_tokens = n_computed;
            if (!resumed) req->num_cached_tokens = n_computed;
            req->t_scheduled = std::chrono::steady_clock::now();
            running_.push_back(sp);
            out.scheduled.push_back({req, n_new, 0, resumed});
            out.total_tokens += n_new;
            budget -= n_new;
        }
    }
    return out;
}

void Scheduler::update_after_step(const SchedulerOutput & out) {
    for (const auto & s : out.scheduled) {
        Request * r = s.req;
        if (r->is_finished()) continue;
        r->num_computed_tokens += s.n_new_tokens;
        // commit full blocks to the prefix cache (never beyond real tokens)
        kv_.update_hashes(r->block_hashes, r->tokens, r->opts.cache_salt);
        kv_.cache_blocks(r->kv, r->tokens, std::min<uint32_t>(r->num_computed_tokens, r->n_tokens()), r->block_hashes);
    }
}

} // namespace iian
