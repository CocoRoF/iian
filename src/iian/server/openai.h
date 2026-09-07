#pragma once
// OpenAI-compatible protocol: request parsing (chat + completions) and response formatting.
#include "server_context.h"
#include "tool_calls.h"

#include "iian/request.h"
#include "iian/sampling.h"
#include "iian/types.h"

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace iian::server {

using json = nlohmann::json;

struct GenerationRequest {
    enum class Kind { CHAT, COMPLETION };
    Kind        kind = Kind::CHAT;
    std::string id;                               // "chatcmpl-..." | "cmpl-..."
    std::string model;                            // name echoed in the response
    std::vector<std::vector<token_t>> prompts;    // one per prompt (completions accept a list)
    std::vector<std::string> prompt_texts;        // for `echo`
    SamplingParams params;
    int  n = 1;
    bool stream = false;
    bool include_usage = false;
    bool echo = false;
    bool want_logprobs = false;
    int  top_logprobs = 0;                        // entries to show per token
    int32_t  priority = 0;
    uint64_t cache_salt = 0;
    std::string rendered_prompt;                  // chat only (debug)
    bool parse_tool_calls = false;                // tools were given (and tool_choice != none)
    ToolFormat tool_format = ToolFormat::NONE;
    int64_t created = 0;
    size_t total_choices() const { return prompts.size() * (size_t) n; }
};

// Accumulated output of one choice.
struct ChoiceResult {
    int    index = 0;
    size_t prompt_index = 0;
    std::string text;
    std::vector<token_t> tokens;
    std::vector<SampledToken> logprobs;
    FinishReason finish = FinishReason::NONE;
    std::string stop_reason;
    std::string error;
    uint32_t n_prompt = 0, n_output = 0, n_cached = 0;
};

GenerationRequest parse_chat_request(ServerContext & ctx, const json & body, const std::string & request_id_hdr);
GenerationRequest parse_completion_request(ServerContext & ctx, const json & body, const std::string & request_id_hdr);

std::string random_id(size_t n = 29);
json finish_reason_json(FinishReason r);
json stop_reason_json(const std::string & s);
json usage_json(uint64_t n_prompt, uint64_t n_completion, uint64_t n_cached);

// logprobs
json chat_logprobs_json(const Tokenizer & tok, const std::vector<SampledToken> & lps, int top_n);
json completion_logprobs_json(const Tokenizer & tok, const std::vector<SampledToken> & lps, int top_n, size_t text_offset_base, size_t & running_offset);

// full responses
json chat_response_json(ServerContext & ctx, const GenerationRequest & req, const std::vector<ChoiceResult> & choices);
json completion_response_json(ServerContext & ctx, const GenerationRequest & req, const std::vector<ChoiceResult> & choices);
// streaming chunks (finish is FinishReason::NONE for intermediate chunks)
json chat_chunk_json(const GenerationRequest & req, int index, const json & delta, const json & logprobs, FinishReason finish, const std::string & stop_reason);
json completion_chunk_json(const GenerationRequest & req, int index, const std::string & text, const json & logprobs, FinishReason finish, const std::string & stop_reason);
json usage_chunk_json(const GenerationRequest & req, uint64_t n_prompt, uint64_t n_completion, uint64_t n_cached);

} // namespace iian::server
