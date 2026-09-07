#include "openai.h"

#include "iian/grammar/json_schema.h"

#include "api_error.h"
#include "json_util.h"

#include "iian/log.h"
#include "iian/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <mutex>
#include <random>

namespace iian::server {

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

std::string random_id(size_t n) {
    static std::mutex mtx;
    static std::mt19937_64 rng{std::random_device{}()};
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::lock_guard<std::mutex> lk(mtx);
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; i++) s += alphabet[rng() % (sizeof(alphabet) - 1)];
    return s;
}

static uint64_t fnv1a(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h ? h : 1;
}

json finish_reason_json(FinishReason r) {
    switch (r) {
        case FinishReason::NONE: return nullptr;
        case FinishReason::STOP: return "stop";
        case FinishReason::LENGTH: return "length";
        case FinishReason::ABORT: return "abort";
        case FinishReason::ERROR: return "error";
        case FinishReason::TOOL_CALLS: return "tool_calls";
    }
    return nullptr;
}

json stop_reason_json(const std::string & s) {
    if (s.empty()) return nullptr;
    // stop_token_ids matches are reported as the integer token id (vLLM convention)
    if (!s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }) && s.size() < 10) return std::stoi(s);
    return s;
}

json usage_json(uint64_t n_prompt, uint64_t n_completion, uint64_t n_cached) {
    return json{
        {"prompt_tokens", n_prompt},
        {"completion_tokens", n_completion},
        {"total_tokens", n_prompt + n_completion},
        {"prompt_tokens_details", {{"cached_tokens", n_cached}}},
    };
}

static json bytes_json(const std::string & piece) {
    json a = json::array();
    for (unsigned char c : piece) a.push_back((int) c);
    return a;
}

static float clean_logprob(float lp) {
    if (std::isnan(lp)) return -9999.0f;
    if (std::isinf(lp)) return lp < 0 ? -9999.0f : 0.0f;
    return lp;
}

json chat_logprobs_json(const Tokenizer & tok, const std::vector<SampledToken> & lps, int top_n) {
    json content = json::array();
    for (const auto & st : lps) {
        std::string piece = tok.token_to_piece(st.token, true);
        json top = json::array();
        for (size_t i = 0; i < st.top.size() && (int) i < top_n; i++) {
            std::string p = tok.token_to_piece(st.top[i].token, true);
            top.push_back({{"token", p}, {"logprob", clean_logprob(st.top[i].logprob)}, {"bytes", bytes_json(p)}});
        }
        content.push_back({{"token", piece}, {"logprob", clean_logprob(st.logprob)}, {"bytes", bytes_json(piece)}, {"top_logprobs", top}});
    }
    return json{{"content", content}, {"refusal", nullptr}};
}

json completion_logprobs_json(const Tokenizer & tok, const std::vector<SampledToken> & lps, int top_n, size_t text_offset_base, size_t & running_offset) {
    json tokens = json::array(), token_logprobs = json::array(), top_logprobs = json::array(), text_offset = json::array();
    for (const auto & st : lps) {
        std::string piece = tok.token_to_piece(st.token, true);
        tokens.push_back(piece);
        token_logprobs.push_back(clean_logprob(st.logprob));
        json top = json::object();
        top[piece] = clean_logprob(st.logprob);
        for (size_t i = 0; i < st.top.size() && (int) i < top_n; i++) top[tok.token_to_piece(st.top[i].token, true)] = clean_logprob(st.top[i].logprob);
        top_logprobs.push_back(top);
        text_offset.push_back(text_offset_base + running_offset);
        running_offset += piece.size();
    }
    return json{{"tokens", tokens}, {"token_logprobs", token_logprobs}, {"top_logprobs", top_logprobs}, {"text_offset", text_offset}};
}

// ---------------------------------------------------------------------------------------------
// shared sampling-parameter parsing (OpenAI names + vLLM extensions)
// ---------------------------------------------------------------------------------------------

static void parse_sampling(const json & body, GenerationRequest & r, int32_t n_vocab) {
    SamplingParams & p = r.params;
    if (auto v = json_number(body, "temperature")) p.temperature = (float) *v;
    if (auto v = json_number(body, "top_p")) p.top_p = (float) *v;
    if (auto v = json_integer(body, "top_k")) p.top_k = (int32_t) *v;
    if (auto v = json_number(body, "min_p")) p.min_p = (float) *v;
    if (auto v = json_number(body, "presence_penalty")) p.presence_penalty = (float) *v;
    if (auto v = json_number(body, "frequency_penalty")) p.frequency_penalty = (float) *v;
    if (auto v = json_number(body, "repetition_penalty")) p.repetition_penalty = (float) *v;
    if (auto v = json_integer(body, "seed")) p.seed = (uint64_t) *v;
    if (auto v = json_integer(body, "min_tokens")) p.min_tokens = (int32_t) *v;
    if (auto v = json_bool(body, "ignore_eos")) p.ignore_eos = *v;
    if (auto v = json_bool(body, "include_stop_str_in_output")) p.include_stop_str_in_output = *v;
    if (auto v = json_bool(body, "skip_special_tokens")) p.skip_special_tokens = *v;
    p.stop = json_string_or_list(body, "stop");
    for (const auto & s : p.stop) if (s.empty()) throw ApiError::bad_request("'stop' strings must not be empty", "stop");
    if (p.stop.size() > 32) throw ApiError::bad_request("at most 32 'stop' strings are supported", "stop");
    p.stop_token_ids = json_int_list(body, "stop_token_ids");
    if (auto v = json_integer(body, "n")) {
        if (*v < 1 || *v > 16) throw ApiError::bad_request("'n' must be between 1 and 16", "n");
        r.n = (int) *v;
    }
    if (auto v = json_integer(body, "priority")) r.priority = (int32_t) *v;
    if (auto v = json_string(body, "cache_salt")) r.cache_salt = v->empty() ? 0 : fnv1a(*v);
    if (auto v = json_bool(body, "stream")) r.stream = *v;
    if (const json * so = json_object(body, "stream_options")) {
        if (auto v = json_bool(*so, "include_usage")) r.include_usage = *v;
        if (r.include_usage && !r.stream) throw ApiError::bad_request("'stream_options' is only allowed when 'stream' is true", "stream_options");
    }
    if (const json * lb = json_object(body, "logit_bias")) {
        for (auto it = lb->begin(); it != lb->end(); ++it) {
            token_t id;
            try { id = (token_t) std::stol(it.key()); } catch (...) { throw ApiError::bad_request("'logit_bias' keys must be token ids as strings, got '" + it.key() + "'", "logit_bias"); }
            if (id < 0 || id >= n_vocab) throw ApiError::bad_request("'logit_bias' token id " + it.key() + " is out of range [0, " + std::to_string(n_vocab) + ")", "logit_bias");
            if (!it.value().is_number()) json_type_error("logit_bias." + it.key(), "a number", it.value());
            float b = it.value().get<float>();
            if (b < -100.0f || b > 100.0f) throw ApiError::bad_request("'logit_bias' values must be in [-100, 100]", "logit_bias");
            p.logit_bias[id] = b;
        }
    }
    if (body.contains("best_of") && !body["best_of"].is_null()) {
        if (!body["best_of"].is_number_integer() || body["best_of"].get<int>() != r.n)
            throw ApiError::not_supported("'best_of' different from 'n' is not supported yet", "best_of");
    }
    if (body.contains("user") && !body["user"].is_null() && !body["user"].is_string()) json_type_error("user", "a string", body["user"]);
}

static void validate_params(GenerationRequest & r, int32_t n_vocab) {
    try {
        r.params.validate(n_vocab);
    } catch (const std::invalid_argument & e) {
        throw ApiError::bad_request(e.what());
    }
}

static std::string make_id(const json & body, const std::string & hdr, const char * prefix) {
    if (auto v = json_string(body, "request_id"); v && !v->empty()) return std::string(prefix) + *v;
    if (!hdr.empty()) return std::string(prefix) + hdr;
    return std::string(prefix) + random_id();
}

// ---------------------------------------------------------------------------------------------
// chat completions
// ---------------------------------------------------------------------------------------------

static void validate_messages(const json & messages) {
    if (!messages.is_array()) throw ApiError::bad_request("'messages' must be an array of {role, content} objects", "messages");
    if (messages.empty()) throw ApiError::bad_request("'messages' must contain at least one message", "messages");
    size_t i = 0;
    for (const auto & m : messages) {
        std::string at = "messages[" + std::to_string(i++) + "]";
        if (!m.is_object()) throw ApiError::bad_request(at + " must be an object", "messages");
        auto role = m.find("role");
        if (role == m.end() || !role->is_string()) throw ApiError::bad_request(at + ".role must be a string (system|user|assistant|tool)", "messages");
        auto c = m.find("content");
        if (c == m.end() || c->is_null()) {
            if (*role != "assistant" || !m.contains("tool_calls"))
                throw ApiError::bad_request(at + ".content is required (a string or an array of {type:'text', text} parts)", "messages");
            continue;
        }
        if (c->is_string()) continue;
        if (!c->is_array()) throw ApiError::bad_request(at + ".content must be a string or an array of content parts, got " + json_type_name(*c), "messages");
        size_t j = 0;
        for (const auto & part : *c) {
            std::string pat = at + ".content[" + std::to_string(j++) + "]";
            if (!part.is_object()) throw ApiError::bad_request(pat + " must be an object with a 'type'", "messages");
            auto t = part.find("type");
            if (t == part.end() || !t->is_string()) throw ApiError::bad_request(pat + ".type must be a string", "messages");
            if (*t != "text") throw ApiError::not_supported(pat + ": content part type '" + t->get<std::string>() + "' is not supported (text only)", "messages");
            auto txt = part.find("text");
            if (txt == part.end() || !txt->is_string()) throw ApiError::bad_request(pat + ".text must be a string", "messages");
        }
    }
}


// Structured outputs. Accepts OpenAI `response_format` ({"type":"json_object"} | {"type":"json_schema","json_schema":{"schema":...}}),
// vLLM-style `guided_json` / `guided_grammar` / `guided_choice` / `structured_outputs: {json|grammar|choice}` and the
// iian extension `grammar` (GBNF). At most one constraint may be given.
static void parse_structured_outputs(const json & body, SamplingParams & params) {
    int n = 0;
    auto set_schema = [&](const json & schema, const char * param) {
        if (schema.is_string()) params.json_schema = schema.get<std::string>();
        else if (schema.is_object()) params.json_schema = schema.dump();
        else throw ApiError::bad_request(std::string("'") + param + "' must be a JSON schema object or string", param);
        n++;
    };
    auto set_choice = [&](const json & choices, const char * param) {
        if (!choices.is_array() || choices.empty()) throw ApiError::bad_request(std::string("'") + param + "' must be a non-empty array of strings", param);
        std::string g = "root ::= ";
        for (size_t i = 0; i < choices.size(); i++) {
            if (!choices[i].is_string()) throw ApiError::bad_request(std::string("'") + param + "' entries must be strings", param);
            if (i) g += " | ";
            g += gbnf_format_literal(choices[i].get<std::string>());
        }
        params.grammar = g;
        n++;
    };
    if (const json * rf = json_object(body, "response_format")) {
        std::string t = json_string(*rf, "type").value_or("text");
        if (t == "text") {
        } else if (t == "json_object") {
            params.json_schema = "{\"type\":\"object\"}";
            n++;
        } else if (t == "json_schema") {
            const json * js = json_object(*rf, "json_schema");
            if (!js) throw ApiError::bad_request("response_format.json_schema must be an object with a 'schema' field", "response_format");
            auto it = js->find("schema");
            if (it == js->end()) throw ApiError::bad_request("response_format.json_schema.schema is required", "response_format");
            set_schema(*it, "response_format");
        } else {
            throw ApiError::not_supported("response_format type '" + t + "' is not supported (text | json_object | json_schema)", "response_format");
        }
    }
    if (auto it = body.find("guided_json"); it != body.end() && !it->is_null()) set_schema(*it, "guided_json");
    if (auto it = body.find("guided_grammar"); it != body.end() && !it->is_null()) { if (!it->is_string()) throw ApiError::bad_request("'guided_grammar' must be a GBNF string", "guided_grammar"); params.grammar = it->get<std::string>(); n++; }
    if (auto it = body.find("grammar"); it != body.end() && !it->is_null()) { if (!it->is_string()) throw ApiError::bad_request("'grammar' must be a GBNF string", "grammar"); params.grammar = it->get<std::string>(); n++; }
    if (auto it = body.find("guided_choice"); it != body.end() && !it->is_null()) set_choice(*it, "guided_choice");
    if (const json * so = json_object(body, "structured_outputs")) {
        if (auto it = so->find("json"); it != so->end() && !it->is_null()) set_schema(*it, "structured_outputs.json");
        if (auto it = so->find("grammar"); it != so->end() && !it->is_null()) { params.grammar = it->get<std::string>(); n++; }
        if (auto it = so->find("choice"); it != so->end() && !it->is_null()) set_choice(*it, "structured_outputs.choice");
        if (auto it = so->find("regex"); it != so->end() && !it->is_null()) throw ApiError::not_supported("regex-constrained output is not supported yet (use a grammar)", "structured_outputs.regex");
    }
    if (n > 1) throw ApiError::bad_request("only one structured-output constraint may be given (response_format / guided_json / guided_grammar / guided_choice / grammar)", "response_format");
}

GenerationRequest parse_chat_request(ServerContext & ctx, const json & body, const std::string & request_id_hdr) {
    Engine & engine = ctx.require_engine();
    const Tokenizer & tok = engine.model().tokenizer();
    const int32_t n_vocab = (int32_t) engine.model().hparams().n_vocab;

    GenerationRequest r;
    r.kind = GenerationRequest::Kind::CHAT;
    r.created = (int64_t) std::time(nullptr);
    r.id = make_id(body, request_id_hdr, "chatcmpl-");
    r.model = json_string(body, "model").value_or("");
    ctx.check_model_name(r.model);
    r.model = ctx.model.name;

    auto msgs = body.find("messages");
    if (msgs == body.end()) throw ApiError::bad_request("'messages' is required", "messages");
    validate_messages(*msgs);

    parse_sampling(body, r, n_vocab);
    r.params.max_tokens = -1;
    if (auto v = json_integer(body, "max_completion_tokens")) r.params.max_tokens = (int32_t) *v;
    else if (auto v2 = json_integer(body, "max_tokens")) r.params.max_tokens = (int32_t) *v2;
    if (r.params.max_tokens == 0 || r.params.max_tokens < -1) throw ApiError::bad_request("'max_tokens' must be >= 1", "max_tokens");

    if (auto v = json_bool(body, "logprobs")) r.want_logprobs = *v;
    if (auto v = json_integer(body, "top_logprobs")) {
        if (*v < 0 || *v > 20) throw ApiError::bad_request("'top_logprobs' must be between 0 and 20", "top_logprobs");
        if (!r.want_logprobs) throw ApiError::bad_request("'top_logprobs' requires 'logprobs': true", "top_logprobs");
        r.top_logprobs = (int) *v;
    }
    r.params.logprobs = r.want_logprobs ? r.top_logprobs : -1;

    parse_structured_outputs(body, r.params);

    // tools: rendered into the prompt by the template; tool-call parsing is not done (raw text returned)
    json tools = nullptr;
    if (const json * t = json_array(body, "tools")) tools = *t;
    std::string tool_choice = "auto", forced_name;
    auto tc = body.find("tool_choice");
    if (tc != body.end() && !tc->is_null()) {
        if (tc->is_string()) tool_choice = tc->get<std::string>();
        else if (tc->is_object()) {
            tool_choice = "function";
            if (tc->contains("function") && (*tc)["function"].is_object()) forced_name = (*tc)["function"].value("name", "");
            if (forced_name.empty()) throw ApiError::bad_request("tool_choice object must be {\"type\":\"function\",\"function\":{\"name\":...}}", "tool_choice");
        } else json_type_error("tool_choice", "a string or an object", *tc);
        if (tool_choice != "auto" && tool_choice != "none" && tool_choice != "required" && tool_choice != "function")
            throw ApiError::bad_request("tool_choice must be 'none', 'auto', 'required' or a function object", "tool_choice");
    }
    if (tool_choice == "none") tools = nullptr;
    if (!tools.is_null() && !tools.empty()) {
        r.parse_tool_calls = true;
        r.tool_format = ctx.tool_format == ToolFormat::NONE ? ToolFormat::GENERIC : ctx.tool_format;
        if (tool_choice == "required" || tool_choice == "function") {
            if (!r.params.grammar.empty() || !r.params.json_schema.empty())
                throw ApiError::bad_request("tool_choice=required cannot be combined with another structured-output constraint", "tool_choice");
            const bool parallel = json_bool(body, "parallel_tool_calls").value_or(true) && tool_choice != "function";
            try { r.params.grammar = tool_call_grammar(nlohmann::ordered_json(tools), r.tool_format, parallel, forced_name); }
            catch (const std::exception & e) { throw ApiError::bad_request(e.what(), "tools"); }
        }
    }

    bool add_generation_prompt = json_bool(body, "add_generation_prompt").value_or(true);
    std::map<std::string, json> kwargs;
    if (const json * kw = json_object(body, "chat_template_kwargs")) for (auto it = kw->begin(); it != kw->end(); ++it) kwargs[it.key()] = it.value();
    std::string tmpl_override = json_string(body, "chat_template").value_or("");

    r.rendered_prompt = ctx.render_chat(*msgs, tools, add_generation_prompt, kwargs, tmpl_override);
    r.prompts.push_back(ctx.tokenize_rendered(r.rendered_prompt));
    r.prompt_texts.push_back(r.rendered_prompt);
    ctx.check_prompt_fits(r.prompts[0].size(), r.params.max_tokens);
    validate_params(r, n_vocab);
    (void) tok;
    return r;
}

// ---------------------------------------------------------------------------------------------
// completions
// ---------------------------------------------------------------------------------------------

GenerationRequest parse_completion_request(ServerContext & ctx, const json & body, const std::string & request_id_hdr) {
    Engine & engine = ctx.require_engine();
    const Tokenizer & tok = engine.model().tokenizer();
    const int32_t n_vocab = (int32_t) engine.model().hparams().n_vocab;

    GenerationRequest r;
    r.kind = GenerationRequest::Kind::COMPLETION;
    r.created = (int64_t) std::time(nullptr);
    r.id = make_id(body, request_id_hdr, "cmpl-");
    r.model = json_string(body, "model").value_or("");
    ctx.check_model_name(r.model);
    r.model = ctx.model.name;

    parse_sampling(body, r, n_vocab);
    parse_structured_outputs(body, r.params);
    r.params.max_tokens = 16;
    if (auto v = json_integer(body, "max_tokens")) r.params.max_tokens = (int32_t) *v;
    if (r.params.max_tokens == 0 || r.params.max_tokens < -1) throw ApiError::bad_request("'max_tokens' must be >= 1", "max_tokens");
    r.echo = json_bool(body, "echo").value_or(false);
    if (auto v = json_integer(body, "logprobs")) {
        if (*v < 0 || *v > 20) throw ApiError::bad_request("'logprobs' must be between 0 and 20", "logprobs");
        r.want_logprobs = true;
        r.top_logprobs = (int) *v;
        r.params.logprobs = (int32_t) *v;
    }
    if (body.contains("suffix") && !body["suffix"].is_null()) throw ApiError::not_supported("'suffix' (fill-in-the-middle) is not supported yet", "suffix");
    std::optional<bool> add_special = json_bool(body, "add_special_tokens");

    auto pr = body.find("prompt");
    if (pr == body.end() || pr->is_null()) throw ApiError::bad_request("'prompt' is required (a string, a list of strings, or a list of token ids)", "prompt");
    auto add_text = [&](const std::string & s) {
        r.prompts.push_back(ctx.tokenize_raw(s, add_special));
        r.prompt_texts.push_back(s);
    };
    auto add_tokens = [&](const json & arr) {
        std::vector<token_t> t;
        for (const auto & e : arr) {
            if (!e.is_number_integer()) throw ApiError::bad_request("'prompt' token list must contain integers only", "prompt");
            token_t id = e.get<token_t>();
            if (id < 0 || id >= n_vocab) throw ApiError::bad_request("'prompt' token id " + std::to_string(id) + " is out of range [0, " + std::to_string(n_vocab) + ")", "prompt");
            t.push_back(id);
        }
        r.prompt_texts.push_back(tok.decode(t, /*remove_special*/ false, /*unparse_special*/ true));
        r.prompts.push_back(std::move(t));
    };
    if (pr->is_string()) add_text(pr->get<std::string>());
    else if (pr->is_array()) {
        if (pr->empty()) throw ApiError::bad_request("'prompt' must not be empty", "prompt");
        if ((*pr)[0].is_number_integer()) add_tokens(*pr);
        else {
            for (const auto & e : *pr) {
                if (e.is_string()) add_text(e.get<std::string>());
                else if (e.is_array()) add_tokens(e);
                else throw ApiError::bad_request("'prompt' list entries must be strings or token-id lists", "prompt");
            }
        }
    } else json_type_error("prompt", "a string, a list of strings, or a list of token ids", *pr);
    if (r.prompts.size() > 32) throw ApiError::bad_request("at most 32 prompts per request", "prompt");
    for (const auto & p : r.prompts) ctx.check_prompt_fits(p.size(), r.params.max_tokens);
    validate_params(r, n_vocab);
    return r;
}

// ---------------------------------------------------------------------------------------------
// responses
// ---------------------------------------------------------------------------------------------

static void sum_usage(const GenerationRequest & req, const std::vector<ChoiceResult> & choices, uint64_t & np, uint64_t & nc, uint64_t & ncached) {
    np = nc = ncached = 0;
    std::vector<bool> seen(req.prompts.size(), false);
    for (const auto & c : choices) {
        nc += c.n_output;
        if (c.prompt_index < seen.size() && !seen[c.prompt_index]) { seen[c.prompt_index] = true; np += c.n_prompt; ncached += c.n_cached; }
    }
}

json chat_response_json(ServerContext & ctx, const GenerationRequest & req, const std::vector<ChoiceResult> & choices) {
    const Tokenizer & tok = ctx.require_engine().model().tokenizer();
    json arr = json::array();
    for (const auto & c : choices) {
        json content = c.text, calls = json::array(), finish = finish_reason_json(c.finish);
        if (req.parse_tool_calls) {
            ParsedAssistant pa = parse_tool_calls(c.text, req.tool_format);
            if (!pa.tool_calls.empty()) {
                calls = tool_calls_json(pa.tool_calls);
                content = pa.content.empty() ? json(nullptr) : json(pa.content);
                if (c.finish == FinishReason::STOP) finish = "tool_calls";
            }
        }
        json choice = {
            {"index", c.index},
            {"message", {{"role", "assistant"}, {"content", content}, {"tool_calls", calls}, {"refusal", nullptr}}},
            {"logprobs", req.want_logprobs ? chat_logprobs_json(tok, c.logprobs, req.top_logprobs) : json(nullptr)},
            {"finish_reason", finish},
            {"stop_reason", stop_reason_json(c.stop_reason)},
        };
        arr.push_back(std::move(choice));
    }
    uint64_t np, nc, ncached;
    sum_usage(req, choices, np, nc, ncached);
    return json{
        {"id", req.id}, {"object", "chat.completion"}, {"created", req.created}, {"model", req.model},
        {"choices", arr}, {"usage", usage_json(np, nc, ncached)},
    };
}

json completion_response_json(ServerContext & ctx, const GenerationRequest & req, const std::vector<ChoiceResult> & choices) {
    const Tokenizer & tok = ctx.require_engine().model().tokenizer();
    json arr = json::array();
    for (const auto & c : choices) {
        std::string prefix = req.echo && c.prompt_index < req.prompt_texts.size() ? req.prompt_texts[c.prompt_index] : "";
        json lp = nullptr;
        if (req.want_logprobs) {
            size_t off = 0;
            lp = completion_logprobs_json(tok, c.logprobs, req.top_logprobs, prefix.size(), off);
        }
        arr.push_back({
            {"index", c.index}, {"text", prefix + c.text}, {"logprobs", lp},
            {"finish_reason", finish_reason_json(c.finish)}, {"stop_reason", stop_reason_json(c.stop_reason)},
        });
    }
    uint64_t np, nc, ncached;
    sum_usage(req, choices, np, nc, ncached);
    return json{
        {"id", req.id}, {"object", "text_completion"}, {"created", req.created}, {"model", req.model},
        {"choices", arr}, {"usage", usage_json(np, nc, ncached)},
    };
}

json chat_chunk_json(const GenerationRequest & req, int index, const json & delta, const json & logprobs, FinishReason finish, const std::string & stop_reason) {
    json choice = {{"index", index}, {"delta", delta}, {"logprobs", logprobs}, {"finish_reason", finish_reason_json(finish)}};
    if (finish != FinishReason::NONE) choice["stop_reason"] = stop_reason_json(stop_reason);
    return json{{"id", req.id}, {"object", "chat.completion.chunk"}, {"created", req.created}, {"model", req.model}, {"choices", json::array({choice})}};
}

json completion_chunk_json(const GenerationRequest & req, int index, const std::string & text, const json & logprobs, FinishReason finish, const std::string & stop_reason) {
    json choice = {{"index", index}, {"text", text}, {"logprobs", logprobs}, {"finish_reason", finish_reason_json(finish)}};
    if (finish != FinishReason::NONE) choice["stop_reason"] = stop_reason_json(stop_reason);
    return json{{"id", req.id}, {"object", "text_completion"}, {"created", req.created}, {"model", req.model}, {"choices", json::array({choice})}};
}

json usage_chunk_json(const GenerationRequest & req, uint64_t n_prompt, uint64_t n_completion, uint64_t n_cached) {
    return json{
        {"id", req.id}, {"object", req.kind == GenerationRequest::Kind::CHAT ? "chat.completion.chunk" : "text_completion"},
        {"created", req.created}, {"model", req.model}, {"choices", json::array()}, {"usage", usage_json(n_prompt, n_completion, n_cached)},
    };
}

} // namespace iian::server
