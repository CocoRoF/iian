// iian run (interactive chat) and iian complete (raw completion), against a running server or in-process.
#include "args.h"
#include "client.h"
#include "commands.h"
#include "engine_options.h"
#include "model_resolve.h"
#include "paths.h"
#include "state.h"

#include "iian/grammar/json_schema.h"
#include "iian/chat_template.h"
#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace iian::cli {

namespace {

SamplingParams sampling_from_args(const ArgParser & p) {
    SamplingParams sp;
    sp.max_tokens = (int32_t) p.get_int("max-tokens");
    sp.temperature = (float) p.get_double("temp");
    sp.top_p = (float) p.get_double("top-p");
    sp.top_k = (int32_t) p.get_int("top-k");
    sp.min_p = (float) p.get_double("min-p");
    sp.repetition_penalty = (float) p.get_double("repeat-penalty");
    sp.presence_penalty = (float) p.get_double("presence-penalty");
    sp.frequency_penalty = (float) p.get_double("frequency-penalty");
    if (p.has("seed")) sp.seed = (uint64_t) p.get_int("seed");
    sp.stop = p.get_all("stop");
    auto file_or_literal = [](const std::string & v) { auto f = read_file(expand_user(v)); return f ? *f : v; };
    if (p.has("grammar")) sp.grammar = file_or_literal(p.get("grammar"));
    if (p.has("json-schema")) sp.json_schema = file_or_literal(p.get("json-schema"));
    if (p.has("regex")) sp.grammar = regex_to_grammar(p.get("regex"));
    return sp;
}

nlohmann::json sampling_json(const SamplingParams & sp) {
    nlohmann::json j = {{"temperature", sp.temperature}, {"top_p", sp.top_p}, {"top_k", sp.top_k}, {"min_p", sp.min_p},
                        {"repetition_penalty", sp.repetition_penalty}, {"presence_penalty", sp.presence_penalty}, {"frequency_penalty", sp.frequency_penalty}};
    if (!sp.grammar.empty()) j["grammar"] = sp.grammar;
    if (!sp.json_schema.empty()) { auto sc = nlohmann::json::parse(sp.json_schema, nullptr, false); if (!sc.is_discarded()) j["guided_json"] = sc; }
    if (sp.max_tokens > 0) j["max_tokens"] = sp.max_tokens;
    if (sp.seed) j["seed"] = *sp.seed;
    if (!sp.stop.empty()) j["stop"] = sp.stop;
    return j;
}

using TextSink = std::function<void(const std::string &)>;

// Generation backend abstraction: HTTP server or in-process engine.
struct Backend {
    virtual ~Backend() = default;
    // Streams text to on_text; returns (n_tokens, finish_reason) or throws.
    virtual std::pair<int, std::string> chat(const nlohmann::json & messages, const SamplingParams & sp, const TextSink & on_text) = 0;
    virtual std::pair<int, std::string> complete(const std::string & prompt, const SamplingParams & sp, const TextSink & on_text) = 0;
    virtual std::string name() const = 0;
};

struct HttpBackend : Backend {
    ServerClient client;
    std::string model;
    explicit HttpBackend(const RunRecord & rec) : client(rec, std::chrono::seconds(600)), model(rec.model_name) {}
    std::pair<int, std::string> stream(const std::string & path, nlohmann::json body, bool chat, const TextSink & on_text) {
        body["model"] = model; body["stream"] = true; body["stream_options"] = {{"include_usage", true}};
        int n = 0; std::string finish, error;
        auto res = client.post_sse(path, body, [&](const std::string & data) {
            if (data == "[DONE]") return false;
            auto j = nlohmann::json::parse(data, nullptr, false);
            if (j.is_discarded()) return true;
            if (j.contains("error")) { error = j["error"].value("message", "server error"); return false; }
            for (auto & ch : j.value("choices", nlohmann::json::array())) {
                std::string t;
                if (chat) { auto & d = ch["delta"]; if (d.contains("content") && d["content"].is_string()) t = d["content"].get<std::string>(); }
                else if (ch.contains("text") && ch["text"].is_string()) t = ch["text"].get<std::string>();
                if (!t.empty()) on_text(t);
                if (ch.contains("finish_reason") && ch["finish_reason"].is_string()) finish = ch["finish_reason"].get<std::string>();
            }
            if (j.contains("usage") && j["usage"].is_object()) n = j["usage"].value("completion_tokens", 0);
            return true;
        });
        if (!error.empty()) throw std::runtime_error(error);
        if (res.status == 0 && !res.error.empty()) throw std::runtime_error("connection failed: " + res.error);
        if (res.status >= 400) throw std::runtime_error(res.error_message());
        return {n, finish};
    }
    std::pair<int, std::string> chat(const nlohmann::json & messages, const SamplingParams & sp, const TextSink & on_text) override {
        nlohmann::json body = sampling_json(sp); body["messages"] = messages;
        return stream("/v1/chat/completions", body, true, on_text);
    }
    std::pair<int, std::string> complete(const std::string & prompt, const SamplingParams & sp, const TextSink & on_text) override {
        nlohmann::json body = sampling_json(sp); body["prompt"] = prompt;
        return stream("/v1/completions", body, false, on_text);
    }
    std::string name() const override { return model + " @ " + client.endpoint(); }
};

struct LocalBackend : Backend {
    std::shared_ptr<Model> model;
    std::unique_ptr<Engine> engine;
    std::unique_ptr<ChatTemplate> tmpl;
    std::string bos;
    LocalBackend(const std::string & path, const ArgParser & p) {
        model = ModelLoader::load(path, device_config_from_args(p));
        EngineConfig ec = engine_config_from_args(p);
        if (!p.has("max-num-seqs")) ec.sched.max_num_seqs = 1;
        engine = std::make_unique<Engine>(model, ec);
        engine->start();
        const Tokenizer & tok = model->tokenizer();
        const auto & sp = tok.special();
        bos = sp.bos != TOKEN_NULL ? tok.token_text(sp.bos) : "";
        const std::string eos = sp.eos != TOKEN_NULL ? tok.token_text(sp.eos) : "";
        std::string src = tok.chat_template().empty() ? chatml_template_source() : tok.chat_template();
        try { tmpl = ChatTemplate::parse(src, bos, eos); }
        catch (const std::exception & e) { LOG_WRN("cli", "model chat template failed to parse (%s); using ChatML", e.what()); tmpl = ChatTemplate::parse(chatml_template_source(), bos, eos); }
    }
    std::pair<int, std::string> generate(std::vector<token_t> toks, const SamplingParams & sp, const TextSink & on_text) {
        auto h = engine->submit(std::move(toks), sp);
        int n = 0; std::string finish;
        while (true) {
            OutputChunk c;
            if (!h.out->pop(c, std::chrono::seconds(600))) throw std::runtime_error("generation timed out");
            if (!c.text.empty()) on_text(c.text);
            n += (int) c.tokens.size();
            if (!c.error.empty()) throw std::runtime_error(c.error);
            if (c.finished) { finish = finish_reason_str(c.finish_reason); break; }
        }
        return {n, finish};
    }
    std::pair<int, std::string> chat(const nlohmann::json & messages, const SamplingParams & sp, const TextSink & on_text) override {
        ChatTemplateInputs in; in.messages = messages; in.add_generation_prompt = true;
        std::string prompt = tmpl->apply(in);
        const Tokenizer & tok = model->tokenizer();
        const bool has_bos = !bos.empty() && prompt.rfind(bos, 0) == 0;
        return generate(tok.encode(prompt, /*add_special*/ !has_bos && tok.special().add_bos, /*parse_special*/ true), sp, on_text);
    }
    std::pair<int, std::string> complete(const std::string & prompt, const SamplingParams & sp, const TextSink & on_text) override {
        return generate(model->tokenizer().encode(prompt, true, true), sp, on_text);
    }
    std::string name() const override { return model->hparams().name.empty() ? basename_of(model->path()) : model->hparams().name; }
};

std::unique_ptr<Backend> make_backend(const std::string & spec, const ArgParser & p, std::string & err) {
    try {
        ResolvedModel r = resolve_model(spec, /*allow_server*/ true, /*pull*/ true);
        if (!r.server_name.empty()) return std::make_unique<HttpBackend>(*load_record(r.server_name));
        if (!p.has("log-level") && !p.get_bool("verbose")) Logger::instance().set_level(LogLevel::WARN);
        return std::make_unique<LocalBackend>(r.path, p);
    } catch (const std::exception & e) { err = e.what(); return nullptr; }
}

} // namespace

int cmd_run(const Args & args) {
    ArgParser p("iian run", "iian run <model|server> [options]",
                "Chat with a model. <model|server> is a running server name (see `iian ps`), a .gguf path, hf:org/repo[:quant] or a cached model.\n"
                "Interactive commands: /exit, /clear (reset conversation), /regen (regenerate last reply), /system <text>.");
    p.positional("model", "Server name, GGUF path, hf: spec or cached model name");
    p.group("Chat");
    p.add("-p,--prompt", "Single-turn mode: send this message and exit", "TEXT");
    p.add("-s,--system", "System prompt", "TEXT");
    p.add("--no-stats", "Do not print [tokens, tok/s] after replies");
    add_sampling_flags(p);
    add_model_flags(p);
    add_engine_flags(p);
    add_logging_flags(p);
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::string err;
    if (!setup_logging(p, err)) return fail(err);
    auto be = make_backend(p.positional(0), p, err);
    if (!be) return fail(err);
    SamplingParams sp;
    try { sp = sampling_from_args(p); } catch (const std::exception & e) { return fail(e.what()); }

    nlohmann::json messages = nlohmann::json::array();
    if (p.has("system")) messages.push_back({{"role", "system"}, {"content", p.get("system")}});
    const bool single = p.has("prompt");
    const bool tty = is_tty(stdin) && is_tty(stdout);
    if (!single && tty) printf("%s %s  %s\n", color("iian", 36).c_str(), be->name().c_str(), dim("(/exit, /clear, /regen, /system <text>)").c_str());

    auto turn = [&](const std::string & user) -> int {
        if (!user.empty()) messages.push_back({{"role", "user"}, {"content", user}});
        std::string reply;
        auto t0 = std::chrono::steady_clock::now();
        try {
            auto [n, finish] = be->chat(messages, sp, [&](const std::string & t) { fputs(t.c_str(), stdout); fflush(stdout); reply += t; });
            printf("\n");
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!p.get_bool("no-stats") && tty) printf("%s\n", dim("[" + std::to_string(n) + " tok, " + std::to_string((int) (s > 0 ? n / s : 0)) + " tok/s" + (finish == "length" ? ", cut by max_tokens" : "") + "]").c_str());
        } catch (const std::exception & e) { printf("\n"); return fail(e.what()); }
        messages.push_back({{"role", "assistant"}, {"content", reply}});
        return 0;
    };

    if (single) return turn(p.get("prompt"));
    std::string line;
    while (true) {
        if (tty) { printf("%s ", color(">", 32).c_str()); fflush(stdout); }
        if (!std::getline(std::cin, line)) { if (tty) printf("\n"); break; }
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/clear") { messages = nlohmann::json::array(); if (p.has("system")) messages.push_back({{"role", "system"}, {"content", p.get("system")}}); printf("%s\n", dim("conversation cleared").c_str()); continue; }
        if (line.rfind("/system ", 0) == 0) { messages = nlohmann::json::array(); messages.push_back({{"role", "system"}, {"content", line.substr(8)}}); printf("%s\n", dim("system prompt set, conversation cleared").c_str()); continue; }
        if (line == "/regen") {
            if (messages.empty() || messages.back()["role"] != "assistant") { printf("%s\n", dim("nothing to regenerate").c_str()); continue; }
            messages.erase(messages.end() - 1);
            if (turn("")) return 1;
            continue;
        }
        if (turn(line)) return 1;
    }
    return 0;
}

int cmd_complete(const Args & args) {
    ArgParser p("iian complete", "iian complete <model|server> -p <prompt> [options]", "Raw text completion (no chat template).");
    p.positional("model", "Server name, GGUF path, hf: spec or cached model name");
    p.add("-p,--prompt", "Prompt text ('-' reads stdin)", "TEXT");
    p.add("--json", "Print a JSON object with the text, token count and finish reason");
    add_sampling_flags(p);
    add_model_flags(p);
    add_engine_flags(p);
    add_logging_flags(p);
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    if (!p.has("prompt")) return fail("missing -p/--prompt");
    std::string err;
    if (!setup_logging(p, err)) return fail(err);
    std::string prompt = p.get("prompt");
    if (prompt == "-") { std::string all, l; while (std::getline(std::cin, l)) all += l + "\n"; prompt = all; }
    auto be = make_backend(p.positional(0), p, err);
    if (!be) return fail(err);
    SamplingParams sp;
    try { sp = sampling_from_args(p); } catch (const std::exception & e) { return fail(e.what()); }
    if (!p.has("max-tokens")) sp.max_tokens = 128;
    std::string text;
    const bool js = p.get_bool("json");
    try {
        auto t0 = std::chrono::steady_clock::now();
        auto [n, finish] = be->complete(prompt, sp, [&](const std::string & t) { text += t; if (!js) { fputs(t.c_str(), stdout); fflush(stdout); } });
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (js) printf("%s\n", nlohmann::json({{"prompt", prompt}, {"text", text}, {"completion_tokens", n}, {"finish_reason", finish}, {"tokens_per_second", s > 0 ? n / s : 0.0}}).dump(2).c_str());
        else printf("\n");
    } catch (const std::exception & e) { return fail(e.what()); }
    return 0;
}

} // namespace iian::cli
