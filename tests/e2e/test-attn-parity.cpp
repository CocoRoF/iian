// Numeric parity of the attention paths: masked flash-attn vs masked mul_mat/softmax vs iian paged kernel.
// Runs one prefill over a prompt with logits for every position and reports max/mean abs differences
// and argmax agreement. Usage: test-attn-parity <model.gguf> [prompt] [--threads N]
#include "iian/arch.h"
#include "iian/arch.h"
#include "iian/engine.h"
#include "iian/graph.h"
#include "iian/kv_cache.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace iian;

static std::vector<float> run(std::shared_ptr<Model> model, const std::vector<token_t> & toks, bool flash, bool paged, int threads) {
    const HParams & hp = model->hparams();
    KVCacheConfig kc; kc.block_size = 16; kc.max_cells = 512; kc.type_k = kc.type_v = GGML_TYPE_F16;
    PagedKVCache kv(*model, kc);
    KVRequestState st;
    if (!kv.allocate_slots(st, (uint32_t) toks.size(), {})) throw std::runtime_error("alloc failed");

    UBatch ub;
    ub.n_tokens = (uint32_t) toks.size();
    UBatch::SeqInfo seq;
    for (uint32_t i = 0; i < ub.n_tokens; i++) {
        ub.tokens.push_back(toks[i]); ub.pos.push_back((pos_t) i); ub.seq_idx.push_back(0);
        ub.slots.push_back(kv.cell_of(st, i)); ub.out_ids.push_back((int32_t) i);
        seq.cells.push_back(kv.cell_of(st, i)); seq.cell_pos.push_back((pos_t) i);
    }
    ub.seqs.push_back(seq);
    ub.n_outputs = ub.n_tokens;
    ub.kv_start = 0;
    ub.n_kv = std::min<uint32_t>(kv.num_cells(), ((uint32_t) (kv.cell_of(st, ub.n_tokens - 1) + 1 + 255) / 256) * 256);

    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto set_nt = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_n_threads");
    if (set_nt) set_nt(cpu, threads);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu);
    const size_t max_nodes = 16384;
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, &buft, 1, max_nodes, false, true);
    std::vector<uint8_t> meta(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    ggml_init_params ip = { meta.size(), meta.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    GraphContext gc(ctx, gf, *model, ub, kv, flash, paged);
    model->arch().build_graph(gc);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) throw std::runtime_error("alloc graph failed");
    gc.set_inputs();
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    std::vector<float> logits((size_t) ub.n_outputs * hp.n_vocab);
    ggml_backend_tensor_get(gc.t_logits, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return logits;
}

static void compare(const char * a, const char * b, const std::vector<float> & x, const std::vector<float> & y, uint32_t n_tok, uint32_t n_vocab) {
    double mx = 0, sum = 0; int argmax_diff = 0;
    for (uint32_t t = 0; t < n_tok; t++) {
        int ax = 0, ay = 0;
        for (uint32_t v = 0; v < n_vocab; v++) {
            const size_t i = (size_t) t * n_vocab + v;
            double d = std::fabs((double) x[i] - y[i]); mx = std::max(mx, d); sum += d;
            if (x[i] > x[(size_t) t * n_vocab + ax]) ax = v;
            if (y[i] > y[(size_t) t * n_vocab + ay]) ay = v;
        }
        if (ax != ay) argmax_diff++;
    }
    printf("%-28s vs %-28s max|d|=%.4g mean|d|=%.3g argmax mismatches=%d/%u\n", a, b, mx, sum / ((double) n_tok * n_vocab), argmax_diff, n_tok);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [prompt] [--threads N]\n", argv[0]); return 1; }
    std::string prompt = "The capital of France is Paris. It is the largest city in Europe and the second largest in the world. It is also the capital of the department of Paris.";
    int threads = 8;
    for (int i = 2; i < argc; i++) { if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]); else prompt = argv[i]; }
    Logger::instance().set_level(LogLevel::WARN);
    std::shared_ptr<Model> model = ModelLoader::load(argv[1], DeviceConfig{});
    auto toks = model->tokenizer().encode(prompt, true, true);
    const uint32_t n_vocab = model->hparams().n_vocab;
    auto fa = run(model, toks, true, false, threads);
    auto nofa = run(model, toks, false, false, threads);
    auto pg = run(model, toks, true, true, threads);
    printf("%zu tokens\n", toks.size());
    compare("masked flash-attn", "masked mul_mat+softmax", fa, nofa, (uint32_t) toks.size(), n_vocab);
    compare("masked flash-attn", "paged kernel", fa, pg, (uint32_t) toks.size(), n_vocab);
    compare("masked mul_mat+softmax", "paged kernel", nofa, pg, (uint32_t) toks.size(), n_vocab);
    return 0;
}
