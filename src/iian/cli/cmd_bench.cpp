// iian bench: in-process throughput benchmark at several concurrency levels.
#include "args.h"
#include "commands.h"
#include "engine_options.h"
#include "model_resolve.h"

#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <thread>

namespace iian::cli {

int cmd_bench(const Args & args) {
    ArgParser p("iian bench", "iian bench <model> [options]",
                "Measure prompt and generation throughput. Each run submits N concurrent requests with distinct random prompts\n"
                "(so the prefix cache cannot help) and reports aggregate tokens/s, time-to-first-token and per-token latency.");
    p.positional("model", "GGUF path, hf: spec or cached model name");
    p.group("Benchmark");
    p.add("--concurrency", "Comma-separated concurrency levels", "LIST", "1,4,16");
    p.add("--prompt-tokens", "Prompt length per request", "N", "256");
    p.add("--gen-tokens", "Tokens to generate per request", "N", "128");
    p.add("--json", "Machine-readable output");
    add_model_flags(p);
    add_engine_flags(p);
    add_logging_flags(p);
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::string err;
    if (!setup_logging(p, err)) return fail(err);
    if (!p.has("log-level") && !p.get_bool("verbose")) Logger::instance().set_level(LogLevel::WARN);

    std::vector<int> conc;
    { std::string s = p.get("concurrency"); size_t i = 0; while (i <= s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); if (j > i) conc.push_back(atoi(s.substr(i, j - i).c_str())); i = j + 1; } }
    const int prompt_len = (int) p.get_int("prompt-tokens"), gen = (int) p.get_int("gen-tokens");
    std::shared_ptr<Model> model;
    try { model = ModelLoader::load(resolve_model(p.positional(0), false, true).path, device_config_from_args(p)); }
    catch (const std::exception & e) { return fail(e.what()); }
    const uint32_t n_vocab = model->hparams().n_vocab;
    const bool js = p.get_bool("json");
    nlohmann::json out = nlohmann::json::array();
    if (!js) {
        printf("%s %s | prompt %d tok | gen %d tok | threads %s\n", bold("bench").c_str(), model_banner(*model).c_str(), prompt_len, gen, p.has("threads") ? p.get("threads").c_str() : "auto");
        printf("%-12s %14s %14s %14s %10s %10s %10s\n", "concurrency", "prompt tok/s", "gen tok/s", "total tok/s", "TTFT ms", "TPOT ms", "wall s");
    }
    for (int c : conc) {
        EngineConfig ec = engine_config_from_args(p);
        ec.sched.max_num_seqs = (uint32_t) c;
        if (!p.has("max-model-len")) ec.max_model_len = (uint32_t) (prompt_len + gen + 16);
        if (!p.has("kv-cache-tokens")) ec.kv_cache_tokens = ec.max_model_len * c;
        ec.enable_prefix_caching = false;
        std::unique_ptr<Engine> eng;
        try { eng = std::make_unique<Engine>(model, ec); } catch (const std::exception & e) { return fail(e.what()); }
        std::mt19937 rng(1234);
        SamplingParams sp; sp.temperature = 0.0f; sp.max_tokens = gen; sp.ignore_eos = true;
        std::vector<Engine::Handle> hs;
        for (int i = 0; i < c; i++) {
            std::vector<token_t> pr(prompt_len);
            for (auto & t : pr) t = (token_t) (rng() % (n_vocab - 10) + 5);
            hs.push_back(eng->submit(pr, sp));
        }
        auto t0 = std::chrono::steady_clock::now();
        eng->start();
        std::vector<bool> got_first(c, false), done(c, false);
        std::vector<double> t_last(c, 0.0), tpot_sum(c, 0.0); std::vector<int> tpot_n(c, 0);
        double ttft_sum = 0; int n_first = 0, n_done = 0; long n_gen = 0;
        while (n_done < c) {
            bool any = false;
            for (int i = 0; i < c; i++) {
                if (done[i]) continue;
                OutputChunk ch;
                if (!hs[i].out->try_pop(ch)) continue;
                any = true;
                double now = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (!got_first[i] && !ch.tokens.empty()) { got_first[i] = true; ttft_sum += now; n_first++; t_last[i] = now; }
                else if (!ch.tokens.empty()) { tpot_sum[i] += (now - t_last[i]) / ch.tokens.size(); tpot_n[i]++; t_last[i] = now; }
                n_gen += (long) ch.tokens.size();
                if (ch.finished) { done[i] = true; n_done++; }
            }
            if (!any) std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double tpot = 0; int tn = 0; for (int i = 0; i < c; i++) if (tpot_n[i]) { tpot += tpot_sum[i] / tpot_n[i]; tn++; }
        eng->stop();
        double ptps = (double) c * prompt_len / secs, gtps = (double) n_gen / secs;
        double ttft = n_first ? ttft_sum / n_first : 0.0, tp = tn ? tpot / tn : 0.0;
        if (js) out.push_back({{"concurrency", c}, {"prompt_tps", ptps}, {"gen_tps", gtps}, {"total_tps", ptps + gtps}, {"ttft_ms", ttft}, {"tpot_ms", tp}, {"wall_s", secs}});
        else printf("%-12d %14.1f %14.1f %14.1f %10.0f %10.1f %10.1f\n", c, ptps, gtps, ptps + gtps, ttft, tp, secs);
    }
    if (js) printf("%s\n", out.dump(2).c_str());
    return 0;
}

} // namespace iian::cli
