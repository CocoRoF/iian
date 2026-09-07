#include "server_context.h"

#include "api_error.h"

#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include <sys/stat.h>

#include <filesystem>

namespace iian::server {

Engine & ServerContext::require_engine() const {
    Engine * e = engine.load();
    if (!e) throw ApiError::unavailable("model is still loading; retry in a moment (GET /health reports readiness)");
    if (stopping.load()) throw ApiError::unavailable("server is shutting down");
    return *e;
}

static std::string basename_no_ext(const std::string & path) {
    std::filesystem::path p(path);
    std::string stem = p.filename().string();
    if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".gguf") stem.resize(stem.size() - 5);
    return stem;
}

void ServerContext::attach(Engine * e) {
    const Model & m = e->model();
    const Tokenizer & tok = m.tokenizer();
    model.path = m.path();
    model.name = cfg.model_name.empty() ? basename_no_ext(m.path()) : cfg.model_name;
    model.arch = m.hparams().arch;
    model.ftype = m.ftype_name();
    model.size_label = m.size_label();
    model.model_name_meta = m.hparams().name;
    model.n_params = m.n_params();
    model.max_model_len = e->max_model_len();
    struct stat st{};
    if (::stat(m.path().c_str(), &st) == 0) { model.file_bytes = (uint64_t) st.st_size; model.created = (int64_t) st.st_mtime; }
    else model.created = (int64_t) std::time(nullptr);

    const SpecialTokens & sp = tok.special();
    bos_text = sp.bos != TOKEN_NULL ? tok.token_text(sp.bos) : "";
    eos_text = sp.eos != TOKEN_NULL ? tok.token_text(sp.eos) : "";

    std::string src;
    if (!cfg.chat_template.empty()) { src = cfg.chat_template; chat_template_origin = "override"; }
    else if (!tok.chat_template().empty()) { src = tok.chat_template(); chat_template_origin = "model"; }
    else { src = chatml_template_source(); chat_template_origin = "chatml"; }
    try {
        chat_template = ChatTemplate::parse(src, bos_text, eos_text);
    } catch (const std::exception & ex) {
        LOG_WRN("http", "chat template (%s) failed to parse, falling back to ChatML: %s", chat_template_origin.c_str(), ex.what());
        chat_template = ChatTemplate::parse(chatml_template_source(), bos_text, eos_text);
        chat_template_origin = "chatml (fallback)";
    }
    tool_format = detect_tool_format(src);
    LOG_INF("http", "chat template: %s; tool-call format: %s", chat_template_origin.c_str(), tool_format_name(tool_format));
    engine.store(e);
}

void ServerContext::check_model_name(const std::string & requested) const {
    if (requested.empty() || requested == model.name) return;
    throw ApiError::not_found("The model '" + requested + "' does not exist. This server serves '" + model.name +
                              "' (GET /v1/models lists it); pass that name or omit 'model'.", "model");
}

std::string ServerContext::render_chat(const nlohmann::json & messages, const nlohmann::json & tools, bool add_generation_prompt,
                                       const std::map<std::string, nlohmann::json> & kwargs, const std::string & override_src) {
    ChatTemplateInputs in;
    in.messages = messages;
    in.tools = tools;
    in.add_generation_prompt = add_generation_prompt;
    for (const auto & [k, v] : kwargs) in.extra_context[k] = v;
    auto it = kwargs.find("enable_thinking");
    if (it != kwargs.end() && it->second.is_boolean()) in.enable_thinking = it->second.get<bool>();

    std::unique_ptr<ChatTemplate> local;
    const ChatTemplate * tmpl = chat_template.get();
    if (!override_src.empty()) {
        if (!cfg.allow_request_chat_template)
            throw ApiError::forbidden("per-request 'chat_template' is disabled; start the server with --allow-request-chat-template to enable it");
        try {
            local = ChatTemplate::parse(override_src, bos_text, eos_text);
        } catch (const std::exception & ex) {
            throw ApiError::bad_request(std::string("'chat_template' failed to parse: ") + ex.what(), "chat_template");
        }
        tmpl = local.get();
    }
    try {
        std::lock_guard<std::mutex> lk(template_mtx);
        return tmpl->apply(in);
    } catch (const ApiError &) {
        throw;
    } catch (const std::exception & ex) {
        throw ApiError::bad_request(std::string("chat template rendering failed: ") + ex.what() +
                                    " (check message roles/content; use --chat-template to override the model's template)", "messages");
    }
}

std::vector<token_t> ServerContext::tokenize_rendered(const std::string & rendered) const {
    const Tokenizer & tok = require_engine().model().tokenizer();
    // llama.cpp behaviour: templates usually emit the BOS text themselves; never double it.
    bool add_special = tok.special().add_bos;
    if (!bos_text.empty() && rendered.compare(0, bos_text.size(), bos_text) == 0) add_special = false;
    return tok.encode(rendered, add_special, /*parse_special*/ true);
}

std::vector<token_t> ServerContext::tokenize_raw(const std::string & text, std::optional<bool> add_special) const {
    const Tokenizer & tok = require_engine().model().tokenizer();
    bool add = add_special.value_or(tok.special().add_bos);
    if (add && !bos_text.empty() && text.compare(0, bos_text.size(), bos_text) == 0) add = false;
    return tok.encode(text, add, /*parse_special*/ true);
}

void ServerContext::check_prompt_fits(size_t n_prompt, int32_t max_tokens) const {
    const uint32_t max_len = model.max_model_len;
    if (n_prompt == 0) throw ApiError::bad_request("prompt is empty after tokenization; provide at least one token");
    if (n_prompt >= max_len)
        throw ApiError::bad_request("This model's maximum context length is " + std::to_string(max_len) + " tokens. However, your prompt has " +
                                    std::to_string(n_prompt) + " tokens; shorten the prompt or start the server with a larger --max-model-len.", "messages");
    if (max_tokens > 0 && n_prompt + (size_t) max_tokens > max_len)
        throw ApiError::bad_request("This model's maximum context length is " + std::to_string(max_len) + " tokens. However, you requested " +
                                    std::to_string(n_prompt + (size_t) max_tokens) + " tokens (" + std::to_string(n_prompt) + " in the prompt; " +
                                    std::to_string(max_tokens) + " for the completion). Reduce max_tokens or the prompt length.", "max_tokens");
}

} // namespace iian::server
