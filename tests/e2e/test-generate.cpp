// End-to-end: load GGUF, run greedy generation through the engine, print text.
// Usage: test-generate <model.gguf> [prompt] [n_tokens] [--chat]
#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [prompt] [n] [--temp T] [--threads N] [--attn auto|masked|paged] [--kv f16|q8_0|q4_0] [--ctx N]\n", argv[0]); return 1; }
    std::string prompt = argc > 2 ? argv[2] : "The capital of France is";
    int n = argc > 3 ? atoi(argv[3]) : 16;
    float temp = 0.0f; int threads = -1; bool fa = true; std::string attn = "auto"; int spec = 0; std::string draft; int ctx = 0; std::string kv_dtype = "f16";
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-fa")) fa = false;
        else if (!strcmp(argv[i], "--attn") && i + 1 < argc) attn = argv[++i];
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--draft") && i + 1 < argc) draft = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv") && i + 1 < argc) kv_dtype = argv[++i];
    }
    using namespace iian;
    DeviceConfig dc;
    std::shared_ptr<Model> model = ModelLoader::load(argv[1], dc);
    EngineConfig ec;
    ec.n_threads = threads;
    ec.flash_attn = fa;
    ec.attention = attn;
    ec.spec_ngram = (uint32_t) spec;
    ec.spec_draft_model = draft;
    ec.kv_dtype = kv_dtype;
    if (ctx > 0) ec.max_model_len = (uint32_t) ctx;
    ec.sched.max_num_seqs = 4;
    ec.sched.max_num_batched_tokens = 512;
    Engine engine(model, ec);
    auto toks = model->tokenizer().encode(prompt, /*add_special*/ true, /*parse_special*/ true);
    printf("prompt tokens (%zu):", toks.size());
    for (auto t : toks) printf(" %d", t);
    printf("\n");
    SamplingParams sp;
    sp.temperature = temp;
    sp.max_tokens = n;
    sp.logprobs = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto h = engine.submit(toks, sp);
    engine.start();
    std::string text;
    int n_out = 0;
    std::vector<token_t> out_toks;
    while (true) {
        OutputChunk c;
        if (!h.out->pop(c, std::chrono::seconds(120))) { fprintf(stderr, "timeout\n"); return 2; }
        text += c.text;
        n_out += (int) c.tokens.size();
        out_toks.insert(out_toks.end(), c.tokens.begin(), c.tokens.end());
        if (!c.error.empty()) { fprintf(stderr, "error: %s\n", c.error.c_str()); return 3; }
        if (c.finished) { printf("finish_reason=%s\n", finish_reason_str(c.finish_reason)); break; }
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("output tokens:");
    for (auto t : out_toks) printf(" %d", t);
    std::string full = prompt + text;
    for (size_t i = 0; i < full.size(); i++) if (full[i] == '\n') full.replace(i, 1, "\\n"), i++;
    printf("\nTEXT: %s\n", full.c_str());
    printf("%d tokens in %.2fs (%.1f tok/s)\n", n_out, std::chrono::duration<double>(t1 - t0).count(), n_out / std::chrono::duration<double>(t1 - t0).count());
    engine.stop();
    return 0;
}
