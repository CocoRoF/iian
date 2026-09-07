// Small SIMD helpers for the CPU-side sampler: vectorised argmax, log-sum-exp and top-k over a vocabulary-sized
// row. AVX2+FMA paths with scalar fallbacks; iian is built with -march=native so the fast paths are used on every
// x86 machine that has them.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define IIAN_VEC_AVX2 1
#endif

namespace iian::vec {

#ifdef IIAN_VEC_AVX2
// expf for 8 lanes; adapted from ggml (ggml-cpu/vec.h, itself adapted from the Arm optimized routines).
// max error 1.45358 + 0.5 ulp; inputs above 88.38 flush to +inf, below -103.97 to 0.
inline __m256 expf8(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f), _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                     _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
                                     u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)), _mm256_set1_epi32(0x82000000u));
    const __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
                        _mm256_andnot_ps(_mm256_castsi256_ps(d),
                                         _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                                                      _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

inline float hmax8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif

// maximum value of v[0..n) (n >= 1); -inf entries are fine, NaNs are treated as smaller than anything
inline float max_value(const float * v, size_t n) {
    size_t i = 0;
    float mx = -INFINITY;
#ifdef IIAN_VEC_AVX2
    if (n >= 8) {
        __m256 m = _mm256_loadu_ps(v);
        for (i = 8; i + 8 <= n; i += 8) m = _mm256_max_ps(m, _mm256_loadu_ps(v + i));
        mx = hmax8(m);
    }
#endif
    for (; i < n; i++) if (v[i] > mx) mx = v[i];
    return mx;
}

// index of the first maximum (greedy argmax); ties resolve to the lowest index like a scalar scan
inline int32_t argmax(const float * v, size_t n) {
    const float mx = max_value(v, n);
    for (size_t i = 0; i < n; i++) if (v[i] == mx) return (int32_t) i;
    return 0;
}

// log(sum(exp(v))) in float with a vectorised exp; matches the double-precision scalar version to ~1e-6
inline float log_sum_exp(const float * v, size_t n) {
    const float mx = max_value(v, n);
    if (!(mx > -INFINITY)) return mx;   // all -inf (or empty)
    size_t i = 0;
    float sum = 0.0f;
#ifdef IIAN_VEC_AVX2
    if (n >= 8) {
        const __m256 vmx = _mm256_set1_ps(mx);
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (; i + 16 <= n; i += 16) {
            acc0 = _mm256_add_ps(acc0, expf8(_mm256_sub_ps(_mm256_loadu_ps(v + i), vmx)));
            acc1 = _mm256_add_ps(acc1, expf8(_mm256_sub_ps(_mm256_loadu_ps(v + i + 8), vmx)));
        }
        for (; i + 8 <= n; i += 8) acc0 = _mm256_add_ps(acc0, expf8(_mm256_sub_ps(_mm256_loadu_ps(v + i), vmx)));
        sum = hsum8(_mm256_add_ps(acc0, acc1));
    }
#endif
    for (; i < n; i++) sum += expf(v[i] - mx);
    return mx + logf(sum);
}

// the k largest entries of v[0..n) as (index, value), best first; ties keep the lower index first.
// One pass with a small insertion buffer: O(n k), for the k <= 20 the API allows this beats a partial_sort of n ids.
inline void top_k(const float * v, size_t n, int32_t k, std::vector<std::pair<int32_t, float>> & out) {
    out.clear();
    if (k <= 0) return;
    out.reserve((size_t) k);
    for (size_t i = 0; i < n; i++) {
        const float x = v[i];
        if ((int32_t) out.size() == k && !(x > out.back().second)) continue;
        // insert keeping descending order; equal values go after existing ones (lower index first)
        size_t pos = out.size();
        while (pos > 0 && x > out[pos - 1].second) pos--;
        out.insert(out.begin() + (std::ptrdiff_t) pos, {(int32_t) i, x});
        if ((int32_t) out.size() > k) out.pop_back();
    }
}

} // namespace iian::vec
