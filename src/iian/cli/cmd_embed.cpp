// iian embed: embeddings from a running server or an in-process model.
#include "args.h"
#include "client.h"
#include "commands.h"
#include "engine_options.h"
#include "model_resolve.h"
#include "paths.h"
#include "state.h"

#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <iostream>

namespace iian::cli {

int cmd_embed(const Args & args) {
    ArgParser p("iian embed", "iian embed <model|server> [text ...] [options]", "Compute embeddings (one vector per input text). Reads stdin lines when no text is given.");
    p.rest("text", "Input text(s)");
    p.add("--pooling", "mean | last | cls", "P", "mean");
    p.add("--no-normalize", "Skip L2 normalization");
    p.add("--dimensions", "Truncate vectors to N dimensions", "N");
    p.add("--json", "Print a JSON array of vectors (default: one line per input, space separated)");
    p.add("--similarity", "Print the cosine-similarity matrix between the inputs instead of the vectors");
    add_model_flags(p);
    add_engine_flags(p);
    add_logging_flags(p);
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::string err;
    if (!setup_logging(p, err)) return fail(err);
    if (p.positionals().empty()) return fail("missing <model|server>");
    std::vector<std::string> texts(p.positionals().begin() + 1, p.positionals().end());
    if (texts.empty()) { std::string l; while (std::getline(std::cin, l)) if (!l.empty()) texts.push_back(l); }
    if (texts.empty()) return fail("no input text (pass texts as arguments or on stdin)");

    std::vector<std::vector<float>> vecs;
    try {
        ResolvedModel r = resolve_model(p.positional(0), true, true);
        nlohmann::json body = {{"input", texts}, {"pooling", p.get("pooling")}, {"normalize", !p.get_bool("no-normalize")}};
        if (p.has("dimensions")) body["dimensions"] = p.get_int("dimensions");
        if (!r.server_name.empty()) {
            ServerClient c(*load_record(r.server_name), std::chrono::seconds(600));
            body["model"] = load_record(r.server_name)->model_name.substr(0, load_record(r.server_name)->model_name.find(','));
            auto res = c.post_json("/v1/embeddings", body);
            if (!res.ok()) return fail(res.error_message());
            for (auto & d : res.json()["data"]) vecs.push_back(d["embedding"].get<std::vector<float>>());
        } else {
            if (!p.has("log-level") && !p.get_bool("verbose")) Logger::instance().set_level(LogLevel::WARN);
            std::shared_ptr<Model> model = ModelLoader::load(r.path, device_config_from_args(p));
            EngineConfig ec = engine_config_from_args(p);
            Engine eng(model, ec); eng.start();
            Engine::Pooling pool = p.get("pooling") == "last" ? Engine::Pooling::LAST : p.get("pooling") == "cls" ? Engine::Pooling::CLS : Engine::Pooling::MEAN;
            std::vector<Engine::Handle> hs;
            for (auto & t : texts) hs.push_back(eng.submit_embedding(model->tokenizer().encode(t, true, true), pool, !p.get_bool("no-normalize")));
            for (auto & h : hs) {
                OutputChunk c;
                while (true) { if (!h.out->pop(c, std::chrono::seconds(600))) return fail("timeout"); if (c.finished) break; }
                if (!c.error.empty()) return fail(c.error);
                auto v = std::move(c.embedding);
                if (p.has("dimensions") && (int64_t) v.size() > p.get_int("dimensions")) v.resize((size_t) p.get_int("dimensions"));
                vecs.push_back(std::move(v));
            }
            eng.stop();
        }
    } catch (const std::exception & e) { return fail(e.what()); }

    if (p.get_bool("similarity")) {
        auto dot = [](const std::vector<float> & a, const std::vector<float> & b) { double s = 0, na = 0, nb = 0; for (size_t i = 0; i < a.size() && i < b.size(); i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return na > 0 && nb > 0 ? s / std::sqrt(na * nb) : 0.0; };
        printf("%-8s", "");
        for (size_t j = 0; j < texts.size(); j++) printf(" %7zu", j);
        printf("\n");
        for (size_t i = 0; i < texts.size(); i++) {
            printf("%-8zu", i);
            for (size_t j = 0; j < texts.size(); j++) printf(" %7.4f", dot(vecs[i], vecs[j]));
            printf("  %s\n", texts[i].substr(0, 40).c_str());
        }
        return 0;
    }
    if (p.get_bool("json")) { printf("%s\n", nlohmann::json(vecs).dump().c_str()); return 0; }
    for (auto & v : vecs) { for (size_t i = 0; i < v.size(); i++) printf("%s%.6g", i ? " " : "", v[i]); printf("\n"); }
    return 0;
}

} // namespace iian::cli
