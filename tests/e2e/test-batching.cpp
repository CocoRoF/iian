// Correctness of continuous batching, prefix caching and preemption:
// outputs of concurrently-served greedy requests must equal sequential single-request outputs.
#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace iian;

static std::vector<SampledToken> g_last_lp;   // logprobs of the last collect() call (when requested)

static std::vector<token_t> collect(Engine::Handle & h, std::string & text, uint32_t & cached, std::string & err) {
    std::vector<token_t> toks;
    g_last_lp.clear();
    while (true) {
        OutputChunk c;
        if (!h.out->pop(c, std::chrono::seconds(300))) { err = "timeout"; return toks; }
        text += c.text;
        toks.insert(toks.end(), c.tokens.begin(), c.tokens.end());
        g_last_lp.insert(g_last_lp.end(), c.logprobs.begin(), c.logprobs.end());
        if (!c.error.empty()) err = c.error;
        if (c.finished) { cached = c.n_cached_tokens; break; }
    }
    return toks;
}

// On a mismatch, show where the two runs diverge and how close the top-2 logits were there:
// a gap of a few 1e-2 nats is backend numerics (e.g. CUDA flash-attention tile shapes), a large gap is a bug.
static void explain_divergence(const Tokenizer & tok, const std::vector<token_t> & a, const std::vector<SampledToken> & lpa,
                               const std::vector<token_t> & b, const std::vector<SampledToken> & lpb) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
    printf("  first divergence at output token %zu:\n", i);
    auto show = [&](const char * tag, const std::vector<SampledToken> & lp) {
        if (i >= lp.size()) { printf("    %s: (no logprobs)\n", tag); return; }
        printf("    %s: sampled %d '%s' logprob %.4f; top:", tag, lp[i].token, tok.token_to_piece(lp[i].token, true).c_str(), lp[i].logprob);
        for (const auto & t : lp[i].top) printf(" %d('%s')=%.4f", t.token, tok.token_to_piece(t.token, true).c_str(), t.logprob);
        printf("\n");
    };
    show("ref", lpa); show("got", lpb);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 1; }
    Logger::instance().set_level(LogLevel::WARN);
    const std::string attn = argc > 2 ? argv[2] : "auto";
    const uint32_t spec = argc > 3 ? (uint32_t) atoi(argv[3]) : 0;
    const std::string draft = argc > 4 ? argv[4] : "";
    std::shared_ptr<Model> model = ModelLoader::load(argv[1], DeviceConfig{});
    const auto & tok = model->tokenizer();

    std::vector<std::string> prompts = {
        "The capital of France is",
        "Once upon a time, in a small village",
        "def fibonacci(n):\n    ",
        "The three primary colors are",
        "Q: What is 2+2?\nA:",
        "Water boils at",
        "In 1969, humans first",
        "The quick brown fox",
    };
    const int n_gen = 24;
    SamplingParams sp; sp.temperature = 0.0f; sp.max_tokens = n_gen;
    const SamplingParams sp_base = sp;
    int failures = 0;

    // ---- 1. sequential reference (no batching, no prefix cache) ----
    std::map<std::string, std::vector<token_t>> ref;
    {
        EngineConfig ec; ec.attention = attn; ec.spec_ngram = spec; ec.spec_draft_model = draft; ec.enable_prefix_caching = false; ec.sched.max_num_seqs = 1; ec.max_model_len = 512;
        Engine eng(model, ec); eng.start();
        for (auto & p : prompts) {
            auto h = eng.submit(tok.encode(p, true, true), sp);
            std::string text, err; uint32_t cached = 0;
            ref[p] = collect(h, text, cached, err);
            if (!err.empty()) { printf("FAIL ref %s: %s\n", p.c_str(), err.c_str()); failures++; }
        }
        eng.stop();
    }
    printf("[1] sequential reference: %zu prompts ok\n", ref.size());

    // ---- 2. all prompts concurrently (continuous batching, chunked prefill with small budget) ----
    {
        EngineConfig ec; ec.attention = attn; ec.spec_ngram = spec; ec.spec_draft_model = draft; ec.sched.max_num_seqs = 8; ec.sched.max_num_batched_tokens = 7; ec.max_model_len = 512;
        Engine eng(model, ec);
        std::vector<Engine::Handle> hs;
        for (auto & p : prompts) hs.push_back(eng.submit(tok.encode(p, true, true), sp));
        eng.start();
        int bad = 0;
        for (size_t i = 0; i < prompts.size(); i++) {
            std::string text, err; uint32_t cached = 0;
            auto got = collect(hs[i], text, cached, err);
            if (got != ref[prompts[i]]) { printf("FAIL batched mismatch for '%s'\n  got: %s\n", prompts[i].c_str(), text.c_str()); bad++; }
        }
        failures += bad;
        eng.stop();
        printf("[2] concurrent batching (budget=7 tokens/step): %s\n", bad ? "FAIL" : "ok");
    }

    // ---- 3. prefix caching: same prompt twice + shared long prefix ----
    {
        EngineConfig ec; ec.attention = attn; ec.spec_ngram = spec; ec.spec_draft_model = draft; ec.sched.max_num_seqs = 4; ec.max_model_len = 1024; ec.block_size = 16;
        Engine eng(model, ec); eng.start();
        SamplingParams sp = sp_base; sp.logprobs = 2;   // top-2 logprobs let a mismatch be explained (near-tie vs bug)
        std::string longp = "This is a long shared system prompt that should be cached across requests. It talks about many things, including the weather, the economy, and cats. ";
        for (int i = 0; i < 3; i++) longp += "Repeat " + std::to_string(i) + ". ";
        std::string p1 = longp + "The capital of France is";
        std::string p2 = longp + "The capital of Germany is";
        auto ta = tok.encode(p1, true, true);
        auto h1 = eng.submit(ta, sp);
        std::string t1, e1; uint32_t c1 = 0; auto r1 = collect(h1, t1, c1, e1); auto lp1 = g_last_lp;
        auto h2 = eng.submit(ta, sp);
        std::string t2, e2; uint32_t c2 = 0; auto r2 = collect(h2, t2, c2, e2); auto lp2 = g_last_lp;
        auto h3 = eng.submit(tok.encode(p2, true, true), sp);
        std::string t3, e3; uint32_t c3 = 0; auto r3 = collect(h3, t3, c3, e3); auto lp3 = g_last_lp;
        // reference without cache
        EngineConfig ec2 = ec; ec2.enable_prefix_caching = false;
        Engine eng2(model, ec2); eng2.start();
        auto hr = eng2.submit(ta, sp); std::string tr, er; uint32_t cr = 0; auto rr = collect(hr, tr, cr, er); auto lpr = g_last_lp;
        auto hr3 = eng2.submit(tok.encode(p2, true, true), sp); std::string tr3, er3; uint32_t cr3 = 0; auto rr3 = collect(hr3, tr3, cr3, er3); auto lpr3 = g_last_lp;
        eng2.stop();
        printf("[3] prefix cache: prompt=%zu tokens; cached tokens run1=%u run2=%u shared-prefix=%u\n", ta.size(), c1, c2, c3);
        if (c1 != 0) { printf("FAIL: first request should have 0 cached tokens\n"); failures++; }
        if (c2 == 0 || c2 + 16 <= (uint32_t) ta.size() - 16) { printf("FAIL: second identical request should hit the cache (got %u of %zu)\n", c2, ta.size()); failures++; }
        if (c3 == 0) { printf("FAIL: shared prefix should hit the cache\n"); failures++; }
        if (r1 != rr || r2 != rr) {
            printf("FAIL: cached outputs differ from reference\n  ref: %s\n  r1: %s\n  r2: %s\n", tr.c_str(), t1.c_str(), t2.c_str()); failures++;
            explain_divergence(tok, rr, lpr, r1 != rr ? r1 : r2, r1 != rr ? lp1 : lp2);
        }
        if (r3 != rr3) {
            printf("FAIL: shared-prefix output differs from reference\n  ref: %s\n  got: %s\n", tr3.c_str(), t3.c_str()); failures++;
            explain_divergence(tok, rr3, lpr3, r3, lp3);
        }
        eng.stop();
    }

    // ---- 4. preemption: KV cache too small for all requests at once ----
    {
        EngineConfig ec; ec.attention = attn; ec.spec_ngram = spec; ec.spec_draft_model = draft; ec.sched.max_num_seqs = 8; ec.max_model_len = 256; ec.kv_cache_tokens = 256; ec.block_size = 16;
        Engine eng(model, ec);
        std::vector<Engine::Handle> hs;
        SamplingParams sp2 = sp; sp2.max_tokens = 40;
        for (auto & p : prompts) hs.push_back(eng.submit(tok.encode(p, true, true), sp2));
        eng.start();
        // reference with plenty of memory
        EngineConfig ecr; ecr.attention = attn; ecr.enable_prefix_caching = false; ecr.sched.max_num_seqs = 1; ecr.max_model_len = 256;
        Engine engr(model, ecr); engr.start();
        int bad = 0;
        for (size_t i = 0; i < prompts.size(); i++) {
            std::string text, err; uint32_t cached = 0;
            auto got = collect(hs[i], text, cached, err);
            auto hr = engr.submit(tok.encode(prompts[i], true, true), sp2);
            std::string tr, er; uint32_t cr = 0; auto r = collect(hr, tr, cr, er);
            if (got != r || !err.empty()) { bad++; printf("  mismatch '%s': %s\n   ref: %s\n", prompts[i].c_str(), err.empty() ? text.c_str() : err.c_str(), tr.c_str()); }
        }
        auto st = eng.stats();
        printf("[4] preemption (256-cell cache, 8 x 40-token requests): preemptions=%llu mismatches=%d %s\n",
               (unsigned long long) st.preemptions, bad, bad ? "FAIL" : "ok");
        if (spec || !draft.empty()) printf("    speculative: drafted=%llu accepted=%llu (%.0f%%)\n", (unsigned long long) st.spec_drafted, (unsigned long long) st.spec_accepted,
                         st.spec_drafted ? 100.0 * st.spec_accepted / st.spec_drafted : 0.0);
        if (bad) failures++;
        if (st.preemptions == 0) printf("  (warning: no preemption occurred; test not exercised)\n");
        eng.stop(); engr.stop();
    }

    // ---- 5. abort mid-generation + stop strings ----
    {
        EngineConfig ec; ec.attention = attn; ec.spec_ngram = spec; ec.spec_draft_model = draft; ec.max_model_len = 512;
        Engine eng(model, ec); eng.start();
        SamplingParams s3 = sp; s3.max_tokens = 200; s3.stop = {"."};
        auto h = eng.submit(tok.encode("The capital of France is", true, true), s3);
        std::string text, err; uint32_t c = 0; collect(h, text, c, err);
        if (text.find('.') != std::string::npos || text.empty()) { printf("FAIL stop string: '%s'\n", text.c_str()); failures++; }
        else printf("[5] stop string ok: '%s'\n", text.c_str());
        auto h2 = eng.submit(tok.encode("Once upon a time", true, true), sp);
        OutputChunk ch; h2.out->pop(ch, std::chrono::seconds(30));
        eng.abort(h2.id);
        std::string t2, e2; uint32_t c2 = 0; collect(h2, t2, c2, e2);
        printf("[5] abort ok\n");
        eng.stop();
    }

    printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
