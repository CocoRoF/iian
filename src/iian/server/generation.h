#pragma once
// Drives one API request through the engine: fans out choices, accumulates OutputChunks,
// and renders either a full response or a sequence of SSE frames.
#include "openai.h"
#include "server_context.h"

#include "iian/engine.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace iian::server {

class GenerationRun {
public:
    GenerationRun(ServerContext & ctx, GenerationRequest req);
    ~GenerationRun();
    GenerationRun(const GenerationRun &) = delete;
    GenerationRun & operator=(const GenerationRun &) = delete;

    // Submit every (prompt x n) choice to the engine. Throws ApiError on rejection.
    void submit();
    // Abort all unfinished choices (idempotent).
    void abort_all(const char * why);
    bool all_finished() const;

    // Non-streaming: block until every choice is finished or `client_gone()` returns true.
    // Returns false if the client went away (choices were aborted).
    bool collect(const std::function<bool()> & client_gone);
    // Streaming: append the next SSE frames to `out` (may be none within `wait`); returns false once
    // the stream is complete (the final "[DONE]" frame has been appended).
    bool next_sse(std::string & out, std::chrono::milliseconds wait);

    // Build the non-streaming JSON response. Throws ApiError if a choice failed.
    json response_json();

    const GenerationRequest & request() const { return req_; }
    const std::vector<ChoiceResult> & choices() const { return results_; }
    uint64_t prompt_tokens() const;
    uint64_t completion_tokens() const;
    uint64_t cached_tokens() const;
    // Update server counters once (finish reasons, latency).
    void record();

private:
    struct Choice {
        Engine::Handle handle{};
        bool submitted = false;
        bool finished = false;
        bool role_sent = false;      // chat stream: first delta carries the role
        bool echo_sent = false;      // completion stream: first delta carries the echoed prompt
        size_t lp_offset = 0;        // completion logprobs text offset
        size_t content_sent = 0;     // chat + tools: bytes of the accumulated text already streamed as content
    };
    struct Event { size_t choice; OutputChunk chunk; };
    std::vector<Event> pump(std::chrono::milliseconds wait);
    void apply(size_t ci, const OutputChunk & c);
    std::string sse_for(size_t ci, const OutputChunk & c);

    ServerContext & ctx_;
    GenerationRequest req_;
    std::vector<Choice> choices_;
    std::vector<ChoiceResult> results_;
    std::chrono::steady_clock::time_point t_start_ = std::chrono::steady_clock::now();
    bool done_sent_ = false;
    bool recorded_ = false;
    bool aborted_ = false;
};

inline std::string sse_frame(const json & j) { return "data: " + j.dump() + "\n\n"; }

} // namespace iian::server
