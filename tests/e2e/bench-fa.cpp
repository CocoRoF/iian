// Flash-attention microbenchmark for the layouts iian uses: one batched call over the sequence dimension
// (gather attention: q [hd, T, nh, S], k/v [hd, L, nhkv, S] permuted views, mask [L, T, 1, S]), the same work
// as S separate calls (ne3 = 1 each), and the old masked window (one call over S*L cells, mask [S*L, S*T]).
// Usage: bench-fa [--hd 64] [--nh 32] [--nhkv 8] [--T 1] [--S 64] [--L 512] [--len 384] [--iters 50]
//        device: IIAN_DEVICES=CUDA1 (default: first non-CPU device, else CPU)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_dev_t pick_device() {
    ggml_backend_load_all();
    if (const char * e = getenv("IIAN_DEVICES")) {
        std::string s(e); s = s.substr(0, s.find(','));
        if (auto * d = ggml_backend_dev_by_name(s.c_str())) return d;
        fprintf(stderr, "device %s not found\n", s.c_str()); exit(1);
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) return d;
    }
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

struct Shape { int hd, nh, nhkv, T, S, L, len; };

// mode 0: batched (ne3 = S); mode 1: S separate calls; mode 2: masked window (ne3 = 1, n_kv = S*L)
static double run(ggml_backend_t backend, const Shape & sh, int mode, int iters, float * checksum) {
    const int n_calls = mode == 1 ? sh.S : 1;
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (16 + 8 * n_calls) + ggml_graph_overhead_custom(16 + 8 * n_calls, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> outs;
    ggml_tensor * q, * k, * v, * mask;
    if (mode == 2) {
        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.hd, (int64_t) sh.T * sh.S, sh.nh, 1);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.hd, sh.nhkv, (int64_t) sh.L * sh.S, 1);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.hd, sh.nhkv, (int64_t) sh.L * sh.S, 1);
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, (int64_t) sh.L * sh.S, (int64_t) sh.T * sh.S, 1, 1);
    } else {
        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.hd, sh.T, sh.nh, sh.S);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.hd, sh.nhkv, sh.L, sh.S);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.hd, sh.nhkv, sh.L, sh.S);
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, sh.L, sh.T, 1, sh.S);
    }
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(mask);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16 + 8 * n_calls, false);
    const float scale = 1.0f / sqrtf((float) sh.hd);
    for (int c = 0; c < n_calls; c++) {
        ggml_tensor * qc = q, * kc = k, * vc = v, * mc = mask;
        if (mode == 1) {
            qc = ggml_view_4d(ctx, q, sh.hd, sh.T, sh.nh, 1, q->nb[1], q->nb[2], q->nb[3], (size_t) c * q->nb[3]);
            kc = ggml_view_4d(ctx, k, sh.hd, sh.nhkv, sh.L, 1, k->nb[1], k->nb[2], k->nb[3], (size_t) c * k->nb[3]);
            vc = ggml_view_4d(ctx, v, sh.hd, sh.nhkv, sh.L, 1, v->nb[1], v->nb[2], v->nb[3], (size_t) c * v->nb[3]);
            mc = ggml_view_4d(ctx, mask, sh.L, sh.T, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3], (size_t) c * mask->nb[3]);
        }
        kc = ggml_permute(ctx, kc, 0, 2, 1, 3);   // [hd, L, nhkv, S]
        vc = ggml_permute(ctx, vc, 0, 2, 1, 3);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, qc, kc, vc, mc, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
        outs.push_back(o);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    // deterministic inputs
    std::vector<float> qf((size_t) ggml_nelements(q));
    for (size_t i = 0; i < qf.size(); i++) qf[i] = 0.01f * (float) ((i * 7919) % 97) - 0.5f;
    ggml_backend_tensor_set(q, qf.data(), 0, ggml_nbytes(q));
    std::vector<ggml_fp16_t> kf((size_t) ggml_nelements(k));
    for (size_t i = 0; i < kf.size(); i++) kf[i] = ggml_fp32_to_fp16(0.01f * (float) ((i * 104729) % 89) - 0.4f);
    ggml_backend_tensor_set(k, kf.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, kf.data(), 0, ggml_nbytes(v));
    // mask: sequence s has `len` valid cells, query t at position len - T + t (causal)
    std::vector<ggml_fp16_t> mf((size_t) ggml_nelements(mask), ggml_fp32_to_fp16(-INFINITY));
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f);
    for (int s = 0; s < sh.S; s++) for (int t = 0; t < sh.T; t++) {
        const int p1 = sh.len - sh.T + t;
        for (int j = 0; j < sh.len && j <= p1; j++) {
            if (mode == 2) mf[((size_t) s * sh.T + t) * ((size_t) sh.L * sh.S) + (size_t) s * sh.L + j] = z;
            else mf[((size_t) s * sh.T + t) * sh.L + j] = z;
        }
    }
    ggml_backend_tensor_set(mask, mf.data(), 0, ggml_nbytes(mask));
    // warmup
    ggml_backend_graph_compute(backend, gf); ggml_backend_synchronize(backend);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    double sum = 0;
    for (auto * o : outs) { std::vector<float> of((size_t) ggml_nelements(o)); ggml_backend_tensor_get(o, of.data(), 0, ggml_nbytes(o)); for (float x : of) sum += x; }
    *checksum = (float) sum;
    ggml_gallocr_free(galloc); ggml_free(ctx);
    return us;
}

int main(int argc, char ** argv) {
    Shape sh { 64, 32, 8, 1, 64, 512, 384 };
    int iters = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i]; const int v = atoi(argv[i + 1]);
        if (a == "--hd") sh.hd = v; else if (a == "--nh") sh.nh = v; else if (a == "--nhkv") sh.nhkv = v;
        else if (a == "--T") sh.T = v; else if (a == "--S") sh.S = v; else if (a == "--L") sh.L = v;
        else if (a == "--len") sh.len = v; else if (a == "--iters") iters = v;
        else { fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
    }
    if (sh.len > sh.L) sh.len = sh.L;
    ggml_backend_dev_t dev = pick_device();
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("device %s | hd=%d nh=%d nhkv=%d T=%d S=%d L=%d len=%d\n", ggml_backend_dev_name(dev), sh.hd, sh.nh, sh.nhkv, sh.T, sh.S, sh.L, sh.len);
    const char * names[] = { "batched (ne3=S)", "S separate calls", "masked window (S*L)" };
    for (int mode = 0; mode < 3; mode++) {
        float cs = 0;
        const double us = run(backend, sh, mode, iters, &cs);
        printf("  %-22s %9.1f us/step  (checksum %.4f)\n", names[mode], us, cs);
    }
    ggml_backend_free(backend);
    return 0;
}
