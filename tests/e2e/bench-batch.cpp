// Batched throughput micro-benchmark: N concurrent requests, each `prompt_len` prompt tokens (distinct
// prompts so the prefix cache does not help) and `gen` generated tokens; reports aggregate throughput.
// Usage: bench-batch <model.gguf> [--concurrency 1,4,16] [--prompt 256] [--gen 64] [--threads 8] [--attn masked|paged]
#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace iian;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [--concurrency 1,4,16] [--prompt 256] [--gen 64] [--threads N] [--attn masked|paged]\n", argv[0]); return 1; }
    std::vector<int> conc = {1, 4, 16};
    int prompt_len = 256, gen = 64, threads = -1;
    std::string attn = "";
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--concurrency") && i + 1 < argc) { conc.clear(); std::string s = argv[++i]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); conc.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--attn") && i + 1 < argc) attn = argv[++i];
    }
    Logger::instance().set_level(LogLevel::WARN);
    std::shared_ptr<Model> model = ModelLoader::load(argv[1], DeviceConfig{});
    const uint32_t n_vocab = model->hparams().n_vocab;
    int max_c = 1; for (int c : conc) max_c = std::max(max_c, c);

    printf("model=%s prompt=%d gen=%d threads=%d attn=%s\n", model->hparams().name.c_str(), prompt_len, gen, threads, attn.empty() ? "default" : attn.c_str());
    printf("%-12s %12s %12s %12s %10s %10s\n", "concurrency", "prompt tok/s", "gen tok/s", "total tok/s", "TTFT ms", "TPOT ms");
    for (int c : conc) {
        EngineConfig ec;
        ec.n_threads = threads;
        ec.sched.max_num_seqs = (uint32_t) c;
        ec.sched.max_num_batched_tokens = 2048;
        ec.max_model_len = (uint32_t) (prompt_len + gen + 16);
        ec.kv_cache_tokens = ec.max_model_len * c;
        ec.enable_prefix_caching = false;
        if (!attn.empty()) ec.attention = attn;
        Engine eng(model, ec);
        std::mt19937 rng(1234);
        SamplingParams sp; sp.temperature = 0.0f; sp.max_tokens = gen; sp.ignore_eos = true;
        std::vector<Engine::Handle> hs;
        for (int i = 0; i < c; i++) {
            std::vector<token_t> p(prompt_len);
            for (auto & t : p) t = (token_t) (rng() % (n_vocab - 10) + 5);
            hs.push_back(eng.submit(p, sp));
        }
        auto t0 = std::chrono::steady_clock::now();
        eng.start();
        double ttft_sum = 0; int n_first = 0; long n_gen = 0;
        std::vector<bool> got_first(c, false);
        std::vector<int> done(c, 0);
        int n_done = 0;
        std::vector<double> t_last(c, 0.0), tpot_sum(c, 0.0); std::vector<int> tpot_n(c, 0);
        while (n_done < c) {
            for (int i = 0; i < c; i++) {
                if (done[i]) continue;
                OutputChunk ch;
                if (!hs[i].out->try_pop(ch)) continue;
                double now = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (!got_first[i] && !ch.tokens.empty()) { got_first[i] = true; ttft_sum += now; n_first++; t_last[i] = now; }
                else if (!ch.tokens.empty()) { tpot_sum[i] += (now - t_last[i]) / ch.tokens.size(); tpot_n[i]++; t_last[i] = now; }
                n_gen += (long) ch.tokens.size();
                if (ch.finished) { done[i] = 1; n_done++; }
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        double tpot = 0; int tn = 0; for (int i = 0; i < c; i++) if (tpot_n[i]) { tpot += tpot_sum[i] / tpot_n[i]; tn++; }
        eng.stop();
        printf("%-12d %12.1f %12.1f %12.1f %10.0f %10.1f\n", c, (double) c * prompt_len / secs, (double) n_gen / secs,
               ((double) c * prompt_len + n_gen) / secs, n_first ? ttft_sum / n_first : 0.0, tn ? tpot / tn : 0.0);
    }
    return 0;
}
