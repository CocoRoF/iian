#pragma once
#include "iian/kv_cache.h"
#include "iian/sampling.h"
#include "iian/tokenizer.h"
#include "iian/types.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace iian {

struct llama_grammar;   // GBNF grammar state (src/iian/grammar/grammar.h)

// Ordering matters: is_finished() == status > PREEMPTED (vLLM convention)
enum class RequestStatus { WAITING = 0, RUNNING, PREEMPTED, FINISHED_STOPPED, FINISHED_LENGTH, FINISHED_ABORTED, FINISHED_ERROR };
enum class FinishReason { NONE, STOP, LENGTH, ABORT, ERROR, TOOL_CALLS };
const char * finish_reason_str(FinishReason r);   // "stop" | "length" | "abort" | "error" | "tool_calls"

// One streaming delta delivered to the client.
struct OutputChunk {
    std::vector<token_t>      tokens;      // newly generated tokens in this chunk
    std::string               text;        // newly decoded text (may lag tokens for partial UTF-8 / stop holdback)
    std::vector<SampledToken> logprobs;    // parallel to tokens, if requested
    bool         finished = false;
    FinishReason finish_reason = FinishReason::NONE;
    std::string  stop_reason;              // matched stop string, or "<token id>" for stop_token_ids, or ""
    std::string  error;
    // usage snapshot (valid when finished)
    uint32_t n_prompt_tokens = 0;
    uint32_t n_output_tokens = 0;
    uint32_t n_cached_tokens = 0;          // prompt tokens served from prefix cache
    std::vector<float> embedding;          // embedding requests: the pooled (optionally normalized) vector, on finish
};

// Thread-safe queue handed to the submitter; the engine pushes chunks, the server pops them.
class OutputQueue {
public:
    void push(OutputChunk c);
    // Blocks until a chunk is available or timeout; returns false on timeout.
    bool pop(OutputChunk & out, std::chrono::milliseconds timeout);
    bool try_pop(OutputChunk & out);
    bool finished() const { return finished_; }
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<OutputChunk> q_;
    bool finished_ = false;
};

struct RequestOptions {
    std::string external_id;   // client-visible id (e.g. "chatcmpl-...")
    int32_t priority = 0;      // lower = higher priority (priority policy only)
    uint64_t cache_salt = 0;   // isolates prefix cache entries (0 = shared)
    std::chrono::steady_clock::time_point arrival = std::chrono::steady_clock::now();
};

// Engine-internal request state.
struct Request {
    request_id_t id = 0;
    RequestOptions opts;
    SamplingParams params;
    std::vector<token_t> tokens;   // prompt + generated (all_token_ids)
    uint32_t n_prompt = 0;
    RequestStatus status = RequestStatus::WAITING;
    FinishReason finish_reason = FinishReason::NONE;
    std::string stop_reason;

    uint32_t num_computed_tokens = 0;    // vLLM invariant: catches up to tokens.size()
    uint32_t num_cached_tokens = 0;      // prefix-cache hit length at admission
    uint32_t num_preemptions = 0;
    KVRequestState kv;
    std::vector<uint64_t> block_hashes;  // chained hashes of full blocks

    SamplerState sampler;
    std::shared_ptr<llama_grammar> grammar;   // null = unconstrained
    // embedding requests (no sampling): pooled hidden states of the prompt
    bool     is_embedding = false;
    int      pooling = 0;                     // Engine::Pooling
    bool     normalize = true;
    std::vector<float> embd_acc;              // running pooled vector (sum for MEAN, last row for LAST, first row for CLS)
    uint32_t embd_count = 0;
    // speculative decoding: draft tokens proposed for the next step (prompt-lookup / n-gram)
    std::vector<token_t> spec_tokens;
    KVRequestState draft_kv;                  // draft model's KV blocks for this request
    uint32_t draft_computed = 0;              // real tokens fed to the draft model
    uint32_t draft_epoch = 0;                 // num_preemptions the draft state corresponds to
    std::unique_ptr<IncrementalDetokenizer> detok;
    std::string text;                    // all decoded text so far
    size_t text_sent = 0;                // bytes of `text` already delivered
    size_t stop_check_offset = 0;
    std::vector<SampledToken> pending_logprobs;
    std::vector<token_t> pending_tokens;

    std::shared_ptr<OutputQueue> out;
    std::chrono::steady_clock::time_point t_first_token{};
    std::chrono::steady_clock::time_point t_scheduled{};
    std::chrono::steady_clock::time_point t_finished{};

    uint32_t n_tokens() const { return (uint32_t) tokens.size(); }
    uint32_t n_output() const { return (uint32_t) tokens.size() - n_prompt; }
    bool is_finished() const { return status > RequestStatus::PREEMPTED; }
    bool is_prefill() const { return num_computed_tokens < n_tokens(); }
};

} // namespace iian
