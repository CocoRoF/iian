#include "paged_attn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#define IIAN_PA_AVX2 1
#endif

namespace iian {

bool paged_attn_supported(ggml_type type_k, ggml_type type_v, float max_alibi_bias) {
    auto ok = [](ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_F32; };
    return ok(type_k) && ok(type_v) && max_alibi_bias == 0.0f;
}

namespace {

// ---- dot / axpy kernels over one K/V row segment (D elements) ----
template <typename T> struct Kern;

template <> struct Kern<float> {
    static inline float dot(const float * q, const float * k, int D) {
        float s = 0.0f;
#ifdef IIAN_PA_AVX2
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), _mm256_loadu_ps(k + i + 8), a1);
        }
        for (; i + 8 <= D; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), a0);
        a0 = _mm256_add_ps(a0, a1);
        float tmp[8]; _mm256_storeu_ps(tmp, a0);
        s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < D; i++) s += q[i] * k[i];
#else
        for (int i = 0; i < D; i++) s += q[i] * k[i];
#endif
        return s;
    }
    static inline void axpy(float * acc, float p, const float * v, int D) {
#ifdef IIAN_PA_AVX2
        const __m256 pp = _mm256_set1_ps(p);
        int i = 0;
        for (; i + 8 <= D; i += 8) _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(pp, _mm256_loadu_ps(v + i), _mm256_loadu_ps(acc + i)));
        for (; i < D; i++) acc[i] += p * v[i];
#else
        for (int i = 0; i < D; i++) acc[i] += p * v[i];
#endif
    }
};

template <> struct Kern<ggml_fp16_t> {
    static inline float dot(const float * q, const ggml_fp16_t * k, int D) {
        float s = 0.0f;
#ifdef IIAN_PA_AVX2
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (k + i))), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (k + i + 8))), a1);
        }
        for (; i + 8 <= D; i += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (k + i))), a0);
        a0 = _mm256_add_ps(a0, a1);
        float tmp[8]; _mm256_storeu_ps(tmp, a0);
        s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (; i < D; i++) s += q[i] * ggml_fp16_to_fp32(k[i]);
#else
        for (int i = 0; i < D; i++) s += q[i] * ggml_fp16_to_fp32(k[i]);
#endif
        return s;
    }
    static inline void axpy(float * acc, float p, const ggml_fp16_t * v, int D) {
#ifdef IIAN_PA_AVX2
        const __m256 pp = _mm256_set1_ps(p);
        int i = 0;
        for (; i + 8 <= D; i += 8) _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(pp, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (v + i))), _mm256_loadu_ps(acc + i)));
        for (; i < D; i++) acc[i] += p * ggml_fp16_to_fp32(v[i]);
#else
        for (int i = 0; i < D; i++) acc[i] += p * ggml_fp16_to_fp32(v[i]);
#endif
    }
};

template <typename TK, typename TV>
void paged_attn_impl(ggml_tensor * dst, int ith, int nth, const PagedAttnParams & P) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const UBatch & ub = *P.ub;

    const int64_t D  = q->ne[0];
    const int64_t Dv = dst->ne[0];
    const int64_t n_head = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_head_kv = k->ne[0] / D;
    const int64_t gqa = n_head / n_head_kv;

    // work items: (token, head) pairs, contiguous chunks per thread
    const int64_t n_items = n_tokens * n_head;
    const int64_t per = (n_items + nth - 1) / nth;
    const int64_t i0 = std::min<int64_t>(n_items, (int64_t) ith * per), i1 = std::min<int64_t>(n_items, i0 + per);

    thread_local std::vector<float> scores;
    thread_local std::vector<float> acc;
    if ((int64_t) acc.size() < Dv) acc.resize(Dv);

    for (int64_t it = i0; it < i1; it++) {
        const int64_t t = it / n_head, h = it % n_head, hk = h / gqa;
        const float * qv = (const float *) ((const char *) q->data + t * q->nb[2] + h * q->nb[1]);
        const UBatch::SeqInfo & seq = ub.seqs[ub.seq_idx[t]];
        const pos_t p1 = ub.pos[t];
        const size_t n_cells = seq.cells.size();
        if (scores.size() < n_cells) scores.resize(n_cells);

        // pass 1: scores + max over the cells this query may attend
        float mx = -INFINITY;
        for (size_t c = 0; c < n_cells; c++) {
            const pos_t p0 = seq.cell_pos[c];
            if ((P.causal && p0 > p1) || (P.n_swa > 0 && p1 - p0 >= (pos_t) P.n_swa)) { scores[c] = -INFINITY; continue; }
            const TK * kp = (const TK *) ((const char *) k->data + seq.cells[c] * k->nb[1]) + hk * D;
            float s = Kern<TK>::dot(qv, kp, (int) D) * P.scale;
            if (P.softcap > 0.0f) s = P.softcap * tanhf(s / P.softcap);
            scores[c] = s;
            mx = std::max(mx, s);
        }
        float * out = (float *) ((char *) dst->data + t * dst->nb[2] + h * dst->nb[1]);
        if (mx == -INFINITY) { std::fill(out, out + Dv, 0.0f); continue; }
        // pass 2: softmax + weighted V sum
        std::fill(acc.begin(), acc.begin() + Dv, 0.0f);
        float sum = 0.0f;
        for (size_t c = 0; c < n_cells; c++) {
            if (scores[c] == -INFINITY) continue;
            const float p = expf(scores[c] - mx);
            sum += p;
            const TV * vp = (const TV *) ((const char *) v->data + seq.cells[c] * v->nb[1]) + hk * Dv;
            Kern<TV>::axpy(acc.data(), p, vp, (int) Dv);
        }
        const float inv = 1.0f / sum;
        for (int64_t i = 0; i < Dv; i++) out[i] = acc[i] * inv;
    }
}

void paged_attn_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const PagedAttnParams & P = *static_cast<const PagedAttnParams *>(userdata);
    const ggml_type tk = dst->src[1]->type, tv = dst->src[2]->type;
    if (tk == GGML_TYPE_F16 && tv == GGML_TYPE_F16)      paged_attn_impl<ggml_fp16_t, ggml_fp16_t>(dst, ith, nth, P);
    else if (tk == GGML_TYPE_F32 && tv == GGML_TYPE_F32) paged_attn_impl<float, float>(dst, ith, nth, P);
    else if (tk == GGML_TYPE_F16 && tv == GGML_TYPE_F32) paged_attn_impl<ggml_fp16_t, float>(dst, ith, nth, P);
    else                                                 paged_attn_impl<float, ggml_fp16_t>(dst, ith, nth, P);
}

} // namespace

ggml_tensor * build_paged_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k_cache, ggml_tensor * v_cache,
                               int64_t n_head_kv, const PagedAttnParams * params) {
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(q->nb[0] == sizeof(float));   // head vectors must be contiguous
    const int64_t Dv = v_cache->ne[0] / n_head_kv;
    ggml_tensor * args[3] = { q, k_cache, v_cache };
    ggml_tensor * out = ggml_custom_4d(ctx, GGML_TYPE_F32, Dv, q->ne[1], q->ne[2], 1, args, 3, paged_attn_op, GGML_N_TASKS_MAX,
                                       const_cast<PagedAttnParams *>(params));
    ggml_set_name(out, "paged_attn");
    return out;
}

} // namespace iian
