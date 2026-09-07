// iian pull / ls / rm / info
#include "args.h"
#include "commands.h"
#include "engine_options.h"
#include "model_resolve.h"
#include "paths.h"

#include "iian/arch.h"
#include "iian/gguf.h"
#include "iian/hparams.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include "ggml.h"

#include <cstdio>
#include <iostream>
#include <map>
#include <unistd.h>

namespace iian::cli {

static bool arch_supported(const std::string & arch) {
    register_builtin_archs();
    return ArchRegistry::instance().get(arch) != nullptr;
}

int cmd_pull(const Args & args) {
    ArgParser p("iian pull", "iian pull <hf:org/repo[:quant]|url> [options]",
                "Download a GGUF into $IIAN_HOME/models. Without :quant, prefers Q4_K_M, then Q8_0, then F16.");
    p.positional("spec", "hf:org/repo[:quant], org/repo[:quant], or an http(s) URL");
    p.add("--dir", "Destination directory", "DIR");
    p.add("--force", "Re-download even if the file exists");
    p.add("-q,--quiet", "No progress output");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    PullOptions o; o.force = p.get_bool("force"); o.quiet = p.get_bool("quiet");
    if (p.has("dir")) o.dir = expand_user(p.get("dir"));
    try {
        std::string spec = p.positional(0), path;
        if (spec.rfind("http://", 0) == 0 || spec.rfind("https://", 0) == 0) {
            if (auto hs = parse_hf_spec(spec)) path = hf_pull(*hs, o); else path = url_pull(spec, o);
        } else if (auto hs = parse_hf_spec(spec)) {
            path = hf_pull(*hs, o);
        } else {
            return fail("unrecognised spec '" + spec + "' (expected hf:org/repo[:quant] or a URL)");
        }
        printf("%s\n", path.c_str());
    } catch (const std::exception & e) { return fail(e.what()); }
    return 0;
}

int cmd_ls(const Args & args) {
    ArgParser p("iian ls", "iian ls [options]", "List local GGUF models ($IIAN_HOME/models and --dir).");
    p.add_repeatable("--dir", "Additional directory to scan", "DIR");
    p.add("--json", "Machine-readable output");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::vector<std::string> dirs{models_dir()};
    for (auto & d : p.get_all("dir")) dirs.push_back(expand_user(d));
    auto models = list_local_models(dirs);
    nlohmann::json out = nlohmann::json::array();
    if (models.empty() && !p.get_bool("json")) { printf("no local models in %s (use `iian pull hf:org/repo:quant`)\n", models_dir().c_str()); return 0; }
    if (!p.get_bool("json")) printf("%-44s %-9s %-10s %-8s %-7s %s\n", "NAME", "SIZE", "ARCH", "QUANT", "PARAMS", "PATH");
    for (auto & m : models) {
        std::string arch = "?", ftype = "?", params = "?";
        try {
            auto md = ModelLoader::load_metadata(m.path);
            arch = md->hparams().arch; ftype = md->ftype_name(); params = md->size_label();
        } catch (...) {}
        if (p.get_bool("json")) out.push_back({{"name", m.name}, {"path", m.path}, {"bytes", m.bytes}, {"arch", arch}, {"ftype", ftype}, {"params", params}});
        else printf("%-44s %-9s %-10s %-8s %-7s %s\n", pad(m.name, 44).c_str(), human_bytes(m.bytes).c_str(), arch.c_str(), ftype.c_str(), params.c_str(), m.path.c_str());
    }
    if (p.get_bool("json")) printf("%s\n", out.dump(2).c_str());
    return 0;
}

int cmd_rm(const Args & args) {
    ArgParser p("iian rm", "iian rm <name|path> [-y]", "Delete a local model file.");
    p.positional("model", "Cached model name (see `iian ls`) or a path");
    p.add("-y,--yes", "Do not ask for confirmation");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::string path;
    try { path = resolve_model(p.positional(0), false, false).path; } catch (const std::exception & e) { return fail(e.what()); }
    if (!p.get_bool("yes")) {
        printf("delete %s (%s)? [y/N] ", path.c_str(), human_bytes(file_size(path).value_or(0)).c_str());
        fflush(stdout);
        std::string a; std::getline(std::cin, a);
        if (a != "y" && a != "Y" && a != "yes") { printf("aborted\n"); return 0; }
    }
    if (::unlink(path.c_str()) != 0) return fail("cannot delete " + path);
    printf("deleted %s\n", path.c_str());
    return 0;
}

int cmd_info(const Args & args) {
    ArgParser p("iian info", "iian info <model> [options]", "Inspect a GGUF: architecture, hyper-parameters, tokenizer, chat template, tensors.");
    p.positional("model", "GGUF path or cached model name");
    p.add("--full", "Print the whole chat template");
    p.add("--all", "Dump all metadata keys");
    p.add("--tensors", "List every tensor with shape and type");
    p.add("--json", "Machine-readable output");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::string path;
    try { path = resolve_model(p.positional(0), false, false).path; } catch (const std::exception & e) { return fail(e.what()); }
    std::unique_ptr<Model> m;
    try { m = ModelLoader::load_metadata(path); } catch (const std::exception & e) { return fail(e.what()); }
    const HParams & hp = m->hparams();
    const GgufFile & f = m->gguf();
    std::map<std::string, size_t> type_counts; std::map<std::string, uint64_t> type_bytes;
    for (auto & t : f.tensors()) { type_counts[ggml_type_name(t.type)]++; type_bytes[ggml_type_name(t.type)] += t.nbytes; }
    const Tokenizer & tok = m->tokenizer();
    std::string tmpl = tok.chat_template();
    const bool supported = arch_supported(hp.arch);

    if (p.get_bool("json")) {
        nlohmann::json j = {{"path", path}, {"arch", hp.arch}, {"name", hp.name}, {"ftype", m->ftype_name()}, {"params", m->n_params()}, {"size_label", m->size_label()},
                            {"file_bytes", file_size(path).value_or(0)}, {"n_layer", hp.n_layer}, {"n_embd", hp.n_embd}, {"n_ctx_train", hp.n_ctx_train}, {"n_vocab", hp.n_vocab},
                            {"n_head", hp.n_head.empty() ? 0 : hp.n_head[0]}, {"n_head_kv", hp.n_head_kv.empty() ? 0 : hp.n_head_kv[0]}, {"head_dim", hp.n_embd_head_k},
                            {"n_ff", hp.n_ff.empty() ? 0 : hp.n_ff[0]}, {"n_expert", hp.n_expert}, {"rope_freq_base", hp.rope_freq_base}, {"n_swa", hp.n_swa},
                            {"tokenizer", {{"type", tok.type_name()}, {"pre", tok.pre_type_name()}, {"n_tokens", tok.n_tokens()}, {"bos", tok.special().bos}, {"eos", tok.special().eos}, {"add_bos", tok.special().add_bos}}},
                            {"chat_template", tmpl}, {"tensor_types", type_counts}, {"supported", supported}};
        if (p.get_bool("all")) { nlohmann::json md; for (auto & kv : f.metadata_dump()) md[kv.first] = kv.second; j["metadata"] = md; }
        printf("%s\n", j.dump(2).c_str());
        return 0;
    }
    printf("%s %s\n", bold("file:      ").c_str(), path.c_str());
    printf("%s %s (%s)\n", bold("size:      ").c_str(), human_bytes(file_size(path).value_or(0)).c_str(), m->ftype_name().c_str());
    printf("%s %s%s\n", bold("arch:      ").c_str(), hp.arch.c_str(), supported ? "" : color("  (not supported by this build of iian)", 33).c_str());
    printf("%s %s\n", bold("name:      ").c_str(), hp.name.c_str());
    printf("%s %s (%llu)\n", bold("params:    ").c_str(), m->size_label().c_str(), (unsigned long long) m->n_params());
    printf("%s %s\n", bold("hparams:   ").c_str(), hp.summary().c_str());
    if (hp.n_expert) printf("%s %u experts, %u used\n", bold("moe:       ").c_str(), hp.n_expert, hp.n_expert_used);
    if (hp.n_swa) printf("%s window %u\n", bold("swa:       ").c_str(), hp.n_swa);
    printf("%s %s/%s, %d tokens, bos=%d eos=%d eot=%d add_bos=%d\n", bold("tokenizer: ").c_str(), tok.type_name().c_str(), tok.pre_type_name().c_str(), tok.n_tokens(),
           tok.special().bos, tok.special().eos, tok.special().eot, (int) tok.special().add_bos);
    printf("%s ", bold("tensors:   ").c_str());
    bool first = true;
    for (auto & kv : type_counts) { printf("%s%s x%zu (%s)", first ? "" : ", ", kv.first.c_str(), kv.second, human_bytes(type_bytes[kv.first]).c_str()); first = false; }
    printf("\n");
    if (tmpl.empty()) printf("%s (none; ChatML will be used)\n", bold("template:  ").c_str());
    else if (p.get_bool("full")) printf("%s\n%s\n", bold("template:  ").c_str(), tmpl.c_str());
    else { std::string t = tmpl.substr(0, 160); for (auto & c : t) if (c == '\n') c = ' '; printf("%s %s%s\n", bold("template:  ").c_str(), t.c_str(), tmpl.size() > 160 ? dim(" ... (--full)").c_str() : ""); }
    if (p.get_bool("tensors")) {
        printf("\n%-40s %-8s %s\n", "TENSOR", "TYPE", "SHAPE");
        for (auto & t : f.tensors()) {
            std::string shape = "[";
            for (int d = 0; d < t.n_dims; d++) shape += (d ? ", " : "") + std::to_string(t.ne[d]);
            printf("%-40s %-8s %s]\n", t.name.c_str(), ggml_type_name(t.type), shape.c_str());
        }
    }
    if (p.get_bool("all")) {
        printf("\n%s\n", bold("metadata:").c_str());
        for (auto & kv : f.metadata_dump()) printf("  %-50s %s\n", kv.first.c_str(), kv.second.c_str());
    }
    return 0;
}

} // namespace iian::cli
