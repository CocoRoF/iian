#include "commands.h"

#include "args.h"

#include "iian/version.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>

namespace iian::cli {

const std::vector<Command> & commands() {
    static const std::vector<Command> cmds = {
        {"serve",    "Start an OpenAI-compatible server for a model (foreground or -d daemon)", cmd_serve,    "Serve"},
        {"ps",       "List running servers",                                                     cmd_ps,       "Manage"},
        {"stop",     "Stop a running server (name, pid or 'all')",                              cmd_stop,     "Manage"},
        {"restart",  "Restart a running server with the same arguments",                       cmd_restart,  "Manage"},
        {"logs",     "Show (and follow) a server's log",                                        cmd_logs,     "Manage"},
        {"status",   "Detailed status of a server (health, model, engine stats)",              cmd_status,   "Manage"},
        {"run",      "Chat with a model interactively (server or in-process)",                 cmd_run,      "Use"},
        {"complete", "One-shot raw text completion",                                            cmd_complete, "Use"},
        {"embed",    "Embeddings for texts (vectors or a cosine-similarity matrix)",                cmd_embed,    "Use"},
        {"pull",     "Download a GGUF from Hugging Face (hf:org/repo[:quant]) or a URL",       cmd_pull,     "Models"},
        {"ls",       "List local models",                                                       cmd_ls,       "Models"},
        {"rm",       "Delete a local model",                                                    cmd_rm,       "Models"},
        {"info",     "Inspect a GGUF file (architecture, hparams, tokenizer, tensors)",         cmd_info,     "Models"},
        {"bench",    "Benchmark prompt/generation throughput at several concurrency levels",    cmd_bench,    "Other"},
        {"version",  "Print version and build information",                                    cmd_version,  "Other"},
    };
    return cmds;
}

const Command * find_command(const std::string & name) {
    for (const auto & c : commands()) if (name == c.name) return &c;
    return nullptr;
}

int print_global_help() {
    printf("%s — LLM inference engine (GGUF, paged KV cache, continuous batching, OpenAI API)\n\n", bold("iian").c_str());
    printf("%s iian <command> [options]\n\n", bold("Usage:").c_str());
    const char * sections[] = {"Serve", "Manage", "Use", "Models", "Other"};
    for (const char * sec : sections) {
        printf("%s\n", bold(sec).c_str());
        for (const auto & c : commands()) if (!strcmp(c.section, sec)) printf("  %-10s %s\n", c.name, c.summary);
        printf("\n");
    }
    printf("%s\n", bold("Examples:").c_str());
    printf("  iian serve hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m -d      # download + serve in the background\n");
    printf("  iian run qwen2.5-0.5b-instruct                               # chat with the running server\n");
    printf("  iian ps && iian logs -f qwen2.5-0.5b-instruct && iian stop all\n");
    printf("  iian bench models/model.gguf --concurrency 1,8,32\n\n");
    printf("Run 'iian <command> --help' for command options. Environment: IIAN_HOME (default ~/.iian), IIAN_<FLAG> overrides.\n");
    return 0;
}

std::string build_info_string() {
    std::string s = std::string("iian ") + iian::version();
    s += std::string(" (ggml ") + ggml_version() + ")";
    return s;
}

int fail(const std::string & msg) {
    fprintf(stderr, "%s %s\n", color("error:", 31).c_str(), msg.c_str());
    return 1;
}

int cmd_version(const Args & args) {
    (void) args;
    printf("%s\n", build_info_string().c_str());
    ggml_backend_load_all();
    printf("backends:");
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) printf(" %s", ggml_backend_reg_name(ggml_backend_reg_get(i)));
    printf("\ndevices:\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * d = ggml_backend_dev_get(i);
        size_t f = 0, t = 0; ggml_backend_dev_memory(d, &f, &t);
        printf("  %-10s %s (%s free / %s)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d), human_bytes(f).c_str(), human_bytes(t).c_str());
    }
    return 0;
}

} // namespace iian::cli
