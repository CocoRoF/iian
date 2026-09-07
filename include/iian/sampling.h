#pragma once
// Sampling parameters (vLLM SamplingParams semantics, OpenAI-compatible naming) and the CPU sampler.
#include "iian/types.h"

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace iian {

struct SamplingParams {
    int32_t n = 1;
    float temperature = 1.0f;        // 0 -> greedy
    float top_p = 1.0f;
    int32_t top_k = 0;               // 0 or -1 -> disabled
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    int32_t repetition_last_n = -1;  // window for repetition penalty (-1 = all output tokens)
    std::optional<uint64_t> seed;
    int32_t max_tokens = -1;         // -1 -> until context is full
    int32_t min_tokens = 0;
    bool ignore_eos = false;
    std::vector<std::string> stop;          // stop strings
    std::vector<token_t> stop_token_ids;
    bool include_stop_str_in_output = false;
    bool skip_special_tokens = true;
    int32_t logprobs = -1;                  // -1 = none, 0 = sampled token only, k = top-k + sampled
    bool prompt_logprobs = false;
    std::map<token_t, float> logit_bias;
    std::vector<token_t> allowed_token_ids;
    std::vector<std::vector<token_t>> bad_words;   // token sequences that may not be produced
    bool echo = false;
    // structured output: a GBNF grammar (llama.cpp syntax) and/or a JSON schema (converted to GBNF at submit time)
    std::string grammar;
    std::string grammar_root = "root";
    std::string json_schema;          // JSON text of the schema; empty = none

    bool is_greedy() const { return temperature < 1e-5f; }
    void validate(int32_t n_vocab) const;   // throws std::invalid_argument
};

struct TokenLogprob {
    token_t token;
    float   logprob;
};

struct SampledToken {
    token_t token;
    float   logprob = 0.0f;                 // logprob of the sampled token (raw distribution), if requested
    std::vector<TokenLogprob> top;          // top-k logprobs, if requested
};

// Per-request sampler state (RNG + token history for penalties).
class SamplerState {
public:
    SamplerState() = default;
    explicit SamplerState(const SamplingParams & p, uint64_t default_seed);
    // Sample one token from logits[n_vocab] (logits are modified in place as scratch).
    SampledToken sample(float * logits, int32_t n_vocab, const std::vector<token_t> & prompt, const std::vector<token_t> & output);
    const SamplingParams & params() const { return params_; }
private:
    SamplingParams params_;
    std::mt19937_64 rng_;
    std::vector<int32_t> ids_;      // scratch: candidate ids
    std::vector<float> probs_;      // scratch
};

} // namespace iian
