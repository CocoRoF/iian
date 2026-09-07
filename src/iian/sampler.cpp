#include "iian/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace iian {

void SamplingParams::validate(int32_t n_vocab) const {
    if (n < 1 || n > 128) throw std::invalid_argument("n must be in [1, 128]");
    if (temperature < 0.0f) throw std::invalid_argument("temperature must be >= 0");
    if (top_p <= 0.0f || top_p > 1.0f) throw std::invalid_argument("top_p must be in (0, 1]");
    if (top_k < -1) throw std::invalid_argument("top_k must be -1, 0 or positive");
    if (min_p < 0.0f || min_p > 1.0f) throw std::invalid_argument("min_p must be in [0, 1]");
    if (presence_penalty < -2.0f || presence_penalty > 2.0f) throw std::invalid_argument("presence_penalty must be in [-2, 2]");
    if (frequency_penalty < -2.0f || frequency_penalty > 2.0f) throw std::invalid_argument("frequency_penalty must be in [-2, 2]");
    if (repetition_penalty <= 0.0f) throw std::invalid_argument("repetition_penalty must be > 0");
    if (max_tokens < -1 || max_tokens == 0) throw std::invalid_argument("max_tokens must be >= 1");
    if (min_tokens < 0) throw std::invalid_argument("min_tokens must be >= 0");
    if (max_tokens > 0 && min_tokens > max_tokens) throw std::invalid_argument("min_tokens must be <= max_tokens");
    if (logprobs > 20) throw std::invalid_argument("logprobs must be <= 20");
    for (auto & kv : logit_bias) if (kv.first < 0 || kv.first >= n_vocab) throw std::invalid_argument("logit_bias token id out of range: " + std::to_string(kv.first));
    for (auto t : stop_token_ids) if (t < 0 || t >= n_vocab) throw std::invalid_argument("stop_token_ids out of range: " + std::to_string(t));
}

SamplerState::SamplerState(const SamplingParams & p, uint64_t default_seed) : params_(p) {
    rng_.seed(p.seed.value_or(default_seed));
}

static inline float log_sum_exp(const float * v, int32_t n) {
    float mx = -INFINITY;
    for (int32_t i = 0; i < n; i++) mx = std::max(mx, v[i]);
    double s = 0.0;
    for (int32_t i = 0; i < n; i++) s += std::exp((double) (v[i] - mx));
    return (float) (mx + std::log(s));
}

SampledToken SamplerState::sample(float * logits, int32_t n_vocab, const std::vector<token_t> & prompt, const std::vector<token_t> & output) {
    const SamplingParams & p = params_;
    SampledToken out;

    // --- logprobs from the raw distribution (vLLM semantics), captured before any processing ---
    const bool want_logprobs = p.logprobs >= 0;
    float raw_lse = 0.0f;
    std::vector<float> raw;
    if (want_logprobs) {
        raw.assign(logits, logits + n_vocab);
        raw_lse = log_sum_exp(raw.data(), n_vocab);
    }

    // --- logits processors (non argmax-invariant) ---
    if (!p.allowed_token_ids.empty()) {
        std::vector<float> keep(n_vocab, -INFINITY);
        for (auto t : p.allowed_token_ids) if (t >= 0 && t < n_vocab) keep[t] = logits[t];
        std::copy(keep.begin(), keep.end(), logits);
    }
    for (auto & kv : p.logit_bias) logits[kv.first] += kv.second;
    if (!p.bad_words.empty()) {
        for (const auto & bw : p.bad_words) {
            if (bw.empty()) continue;
            const size_t pre = bw.size() - 1;
            if (output.size() < pre) continue;
            bool match = true;
            for (size_t i = 0; i < pre; i++) if (output[output.size() - pre + i] != bw[i]) { match = false; break; }
            if (match && bw.back() >= 0 && bw.back() < n_vocab) logits[bw.back()] = -INFINITY;
        }
    }
    if ((int32_t) output.size() < p.min_tokens) {
        // handled by caller (EOS masking) since it needs vocab knowledge; nothing here
    }
    // penalties
    if (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f || p.frequency_penalty != 0.0f) {
        std::unordered_map<token_t, int32_t> counts;
        const size_t n_out = output.size();
        const size_t start = p.repetition_last_n < 0 ? 0 : (n_out > (size_t) p.repetition_last_n ? n_out - p.repetition_last_n : 0);
        for (size_t i = start; i < n_out; i++) counts[output[i]]++;
        if (p.repetition_penalty != 1.0f) {
            // vLLM: repetition penalty applies to prompt tokens as well
            for (auto t : prompt) if (!counts.count(t)) counts.emplace(t, 0);
        }
        for (auto & kv : counts) {
            const token_t t = kv.first;
            if (t < 0 || t >= n_vocab) continue;
            float & l = logits[t];
            if (p.repetition_penalty != 1.0f) l = l > 0 ? l / p.repetition_penalty : l * p.repetition_penalty;
            if (kv.second > 0) l -= p.frequency_penalty * kv.second + p.presence_penalty;
        }
    }

    // --- greedy ---
    if (p.is_greedy()) {
        int32_t best = 0;
        for (int32_t i = 1; i < n_vocab; i++) if (logits[i] > logits[best]) best = i;
        out.token = best;
    } else {
        // temperature
        const float inv_t = 1.0f / p.temperature;
        for (int32_t i = 0; i < n_vocab; i++) logits[i] *= inv_t;

        // candidate set
        ids_.resize(n_vocab);
        std::iota(ids_.begin(), ids_.end(), 0);
        int32_t n_cand = n_vocab;
        const int32_t k = (p.top_k > 0 && p.top_k < n_vocab) ? p.top_k : n_vocab;
        if (k < n_vocab) {
            std::partial_sort(ids_.begin(), ids_.begin() + k, ids_.end(), [&](int a, int b) { return logits[a] > logits[b]; });
            n_cand = k;
        } else if (p.top_p < 1.0f || p.min_p > 0.0f) {
            std::sort(ids_.begin(), ids_.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        }
        // softmax over candidates
        probs_.resize(n_cand);
        float mx = -INFINITY;
        for (int32_t i = 0; i < n_cand; i++) mx = std::max(mx, logits[ids_[i]]);
        double sum = 0.0;
        for (int32_t i = 0; i < n_cand; i++) { probs_[i] = std::exp(logits[ids_[i]] - mx); sum += probs_[i]; }
        for (int32_t i = 0; i < n_cand; i++) probs_[i] = (float) (probs_[i] / sum);
        // min_p (argmax-invariant): drop candidates below min_p * max_prob
        if (p.min_p > 0.0f && n_cand > 1) {
            const float thr = p.min_p * probs_[0];   // sorted when min_p is used
            int32_t keep = 0;
            for (int32_t i = 0; i < n_cand; i++) if (probs_[i] >= thr) keep++;
            n_cand = std::max(1, keep);
        }
        // top_p
        if (p.top_p < 1.0f && n_cand > 1) {
            float cum = 0.0f;
            int32_t keep = 0;
            for (int32_t i = 0; i < n_cand; i++) { cum += probs_[i]; keep++; if (cum >= p.top_p) break; }
            n_cand = keep;
        }
        // renormalize + sample
        double s2 = 0.0;
        for (int32_t i = 0; i < n_cand; i++) s2 += probs_[i];
        std::uniform_real_distribution<double> U(0.0, s2);
        double r = U(rng_);
        int32_t pick = n_cand - 1;
        for (int32_t i = 0; i < n_cand; i++) { r -= probs_[i]; if (r <= 0) { pick = i; break; } }
        out.token = ids_[pick];
    }

    if (want_logprobs) {
        out.logprob = raw[out.token] - raw_lse;
        if (p.logprobs > 0) {
            const int32_t k = std::min(p.logprobs, n_vocab);
            std::vector<int32_t> idx(n_vocab);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return raw[a] > raw[b]; });
            out.top.reserve(k);
            for (int32_t i = 0; i < k; i++) out.top.push_back({idx[i], raw[idx[i]] - raw_lse});
        }
    }
    return out;
}

} // namespace iian
