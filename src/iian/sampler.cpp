#include "iian/sampling.h"

#include "vec_math.h"

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

// Descending sort of candidate ids by logit. Large sets (full-vocabulary top_p) use a bucket pass first, like
// llama.cpp's partial sort: O(n) bucketing by value, then a std::sort inside each bucket, top bucket first.
static void sort_desc(std::vector<int32_t> & ids, const float * v) {
    const auto desc = [&](int32_t a, int32_t b) { return v[a] > v[b] || (v[a] == v[b] && a < b); };
    if (ids.size() < 4096) { std::sort(ids.begin(), ids.end(), desc); return; }
    float lo = INFINITY, hi = -INFINITY;
    for (int32_t i : ids) { const float x = v[i]; if (x > -INFINITY) { lo = std::min(lo, x); hi = std::max(hi, x); } }
    if (!(hi > lo)) { std::sort(ids.begin(), ids.end(), desc); return; }
    constexpr int NB = 256;
    const float scale = (NB - 1) / (hi - lo);
    std::vector<int32_t> counts(NB + 1, 0), bucket(ids.size());
    for (size_t j = 0; j < ids.size(); j++) {
        const float x = v[ids[j]];
        int b = x > -INFINITY ? (int) ((x - lo) * scale) : 0;
        b = std::min(std::max(b, 0), NB - 1);
        bucket[j] = NB - 1 - b;   // 0 = highest values
        counts[bucket[j] + 1]++;
    }
    for (int b = 0; b < NB; b++) counts[b + 1] += counts[b];
    std::vector<int32_t> sorted(ids.size());
    { std::vector<int32_t> pos(counts.begin(), counts.end() - 1); for (size_t j = 0; j < ids.size(); j++) sorted[pos[bucket[j]]++] = ids[j]; }
    for (int b = 0; b < NB; b++) if (counts[b + 1] - counts[b] > 1) std::sort(sorted.begin() + counts[b], sorted.begin() + counts[b + 1], desc);
    ids.swap(sorted);
}

SampledToken SamplerState::sample(float * logits, int32_t n_vocab, const token_t * prompt, size_t n_prompt, const token_t * output, size_t n_output) {
    const SamplingParams & p = params_;
    SampledToken out;

    // --- logprobs from the raw distribution (vLLM semantics), captured before any processing ---
    const bool want_logprobs = p.logprobs >= 0;
    float raw_lse = 0.0f;
    std::vector<float> & raw = raw_;
    if (want_logprobs) {
        raw.assign(logits, logits + n_vocab);
        raw_lse = vec::log_sum_exp(raw.data(), n_vocab);
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
            if (n_output < pre) continue;
            bool match = true;
            for (size_t i = 0; i < pre; i++) if (output[n_output - pre + i] != bw[i]) { match = false; break; }
            if (match && bw.back() >= 0 && bw.back() < n_vocab) logits[bw.back()] = -INFINITY;
        }
    }
    if ((int32_t) n_output < p.min_tokens) {
        // handled by caller (EOS masking) since it needs vocab knowledge; nothing here
    }
    // penalties
    if (p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f || p.frequency_penalty != 0.0f) {
        std::unordered_map<token_t, int32_t> counts;
        const size_t n_out = n_output;
        const size_t start = p.repetition_last_n < 0 ? 0 : (n_out > (size_t) p.repetition_last_n ? n_out - p.repetition_last_n : 0);
        for (size_t i = start; i < n_out; i++) counts[output[i]]++;
        if (p.repetition_penalty != 1.0f) {
            // vLLM: repetition penalty applies to prompt tokens as well
            for (size_t i = 0; i < n_prompt; i++) if (!counts.count(prompt[i])) counts.emplace(prompt[i], 0);
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
        out.token = vec::argmax(logits, n_vocab);
    } else {
        // temperature
        const float inv_t = 1.0f / p.temperature;
        for (int32_t i = 0; i < n_vocab; i++) logits[i] *= inv_t;
        const float mx = vec::max_value(logits, n_vocab);

        // candidate set (ids_), sorted by descending logit only when a truncation needs the order
        ids_.clear();
        if (p.min_p > 0.0f) {
            // min_p keeps prob >= min_p * max_prob, i.e. logit >= mx + log(min_p): a filter, no sort needed
            const float thr = mx + logf(p.min_p);
            for (int32_t i = 0; i < n_vocab; i++) if (logits[i] >= thr) ids_.push_back(i);
            if (ids_.empty()) ids_.push_back(vec::argmax(logits, n_vocab));
        } else {
            ids_.resize(n_vocab);
            std::iota(ids_.begin(), ids_.end(), 0);
        }
        const auto desc = [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
        if (p.top_k > 0 && p.top_k < (int32_t) ids_.size()) {
            std::partial_sort(ids_.begin(), ids_.begin() + p.top_k, ids_.end(), desc);
            ids_.resize(p.top_k);
        } else if (p.top_p < 1.0f && ids_.size() > 1) {
            sort_desc(ids_, logits);
        }
        int32_t n_cand = (int32_t) ids_.size();
        // softmax over candidates (vectorised exp over the whole row when the candidate set is large)
        probs_.resize(n_cand);
        if (n_cand > 2048) {
            exp_.resize(n_vocab);
            size_t i = 0;
#ifdef IIAN_VEC_AVX2
            const __m256 vmx = _mm256_set1_ps(mx);
            for (; i + 8 <= (size_t) n_vocab; i += 8) _mm256_storeu_ps(exp_.data() + i, vec::expf8(_mm256_sub_ps(_mm256_loadu_ps(logits + i), vmx)));
#endif
            for (; i < (size_t) n_vocab; i++) exp_[i] = expf(logits[i] - mx);
            for (int32_t c = 0; c < n_cand; c++) probs_[c] = exp_[ids_[c]];
        } else {
            for (int32_t c = 0; c < n_cand; c++) probs_[c] = expf(logits[ids_[c]] - mx);
        }
        double sum = 0.0;
        for (int32_t c = 0; c < n_cand; c++) sum += probs_[c];
        const float inv_sum = (float) (1.0 / sum);
        for (int32_t c = 0; c < n_cand; c++) probs_[c] *= inv_sum;
        // top_p: cumulative mass in descending order (ids_ are sorted whenever top_p < 1)
        if (p.top_p < 1.0f && n_cand > 1) {
            float cum = 0.0f;
            int32_t keep = 0;
            for (int32_t c = 0; c < n_cand; c++) { cum += probs_[c]; keep++; if (cum >= p.top_p) break; }
            n_cand = keep;
        }
        // renormalize + sample
        double s2 = 0.0;
        for (int32_t c = 0; c < n_cand; c++) s2 += probs_[c];
        std::uniform_real_distribution<double> U(0.0, s2);
        double r = U(rng_);
        int32_t pick = n_cand - 1;
        for (int32_t c = 0; c < n_cand; c++) { r -= probs_[c]; if (r <= 0) { pick = c; break; } }
        out.token = ids_[pick];
    }

    if (want_logprobs) {
        out.logprob = raw[out.token] - raw_lse;
        if (p.logprobs > 0) {
            const int32_t k = std::min(p.logprobs, n_vocab);
            vec::top_k(raw.data(), n_vocab, k, topk_);
            out.top.reserve(k);
            for (const auto & e : topk_) out.top.push_back({e.first, e.second - raw_lse});
        }
    }
    return out;
}

} // namespace iian
