// ChatTemplate: OpenAI-style messages/tools -> prompt string via the iian::jinja engine.
//
// The input pre-processing mirrors llama.cpp's common/chat.cpp (common_chat_msgs_parse_oaicompat,
// common_chat_msg::to_json_oaicompat, messages_inp_normalizer, the generic `workaround::` helpers and
// common_chat_template_direct_apply_impl) so that rendering is identical to llama.cpp for the same inputs.

// NOTE: include/iian/chat_template.h only forward-declares nlohmann::json but stores it by value,
//       so the full definition must be visible before the header is included.
#include "nlohmann/json.hpp"

#include "iian/chat_template.h"

#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/utils.h"
#include "jinja/value.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace iian {

namespace {

// ordered_json keeps key insertion order, which matters for `tojson` output (llama.cpp's common_json is ordered too)
using ojson = nlohmann::ordered_json;

//
// string helpers
//

// port of llama.cpp trim_whitespace (chat-auto-parser-helpers.cpp)
std::string trim_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }
    if (start == str.length()) {
        return "";
    }
    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }
    return str.substr(start, end - start + 1);
}

//
// time helpers
//

// Parse a subset of ISO-8601: YYYY-MM-DD[THH:MM[:SS[.fff]]][Z|+HH:MM|-HH:MM]
// Without a zone designator the time is interpreted as local time (same as std::mktime).
bool parse_iso8601(const std::string & s, std::time_t & out) {
    int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
    int n = 0;
    if (std::sscanf(s.c_str(), "%4d-%2d-%2d%n", &year, &month, &day, &n) < 3 || n != 10) {
        return false;
    }
    size_t pos = static_cast<size_t>(n);
    if (pos < s.size() && (s[pos] == 'T' || s[pos] == ' ')) {
        int m = 0;
        if (std::sscanf(s.c_str() + pos + 1, "%2d:%2d%n", &hour, &minute, &m) < 2 || m != 5) {
            return false;
        }
        pos += 1 + static_cast<size_t>(m);
        if (pos < s.size() && s[pos] == ':') {
            if (std::sscanf(s.c_str() + pos + 1, "%2d%n", &second, &m) < 1 || m != 2) {
                return false;
            }
            pos += 1 + static_cast<size_t>(m);
            if (pos < s.size() && (s[pos] == '.' || s[pos] == ',')) {
                ++pos;
                while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
                    ++pos;
                }
            }
        }
    }
    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon  = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min  = minute;
    tm.tm_sec  = second;
    tm.tm_isdst = -1;
    if (pos >= s.size()) {
        out = std::mktime(&tm);
        return out != static_cast<std::time_t>(-1);
    }
    long offset_sec = 0;
    if (s[pos] == 'Z' || s[pos] == 'z') {
        ++pos;
    } else if (s[pos] == '+' || s[pos] == '-') {
        int sign = s[pos] == '-' ? -1 : 1;
        int oh = 0, om = 0, m = 0;
        if (std::sscanf(s.c_str() + pos + 1, "%2d%n", &oh, &m) < 1 || m != 2) {
            return false;
        }
        pos += 1 + static_cast<size_t>(m);
        if (pos < s.size() && s[pos] == ':') {
            ++pos;
        }
        if (pos < s.size()) {
            if (std::sscanf(s.c_str() + pos, "%2d%n", &om, &m) < 1 || m != 2) {
                return false;
            }
            pos += static_cast<size_t>(m);
        }
        offset_sec = sign * (oh * 3600L + om * 60L);
    } else {
        return false;
    }
    if (pos != s.size()) {
        return false;
    }
#if defined(_WIN32)
    std::time_t utc = _mkgmtime(&tm);
#else
    std::time_t utc = timegm(&tm);
#endif
    if (utc == static_cast<std::time_t>(-1)) {
        return false;
    }
    out = utc - offset_sec;
    return true;
}

// port of llama.cpp format_time (chat.cpp): local time, strftime format
std::string format_time(std::time_t time, const char * format) {
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    char buf[128];
    size_t len = std::strftime(buf, sizeof(buf), format, &local_time);
    return std::string(buf, len);
}

//
// template source patches (port of common_chat_templates_init)
//

std::string patch_template_source(std::string src) {
    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (src.find("<|channel|>") != std::string::npos
        // search for the error message and patch it
        && src.find("in message.content or") != std::string::npos) {
        jinja::string_replace_all(src,
            "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
            "\"<|channel|>final<|message|>\" in message.content %}",
            "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (src.find("[TOOL_CALLS]") != std::string::npos
        // search for the error message and patch it
        && src.find("if (message['content'] is none or") != std::string::npos) {
        jinja::string_replace_all(src,
            "{%- if (message['content'] is none or message['content'] == '' or "
            "message['content']|length == 0) and (message['tool_calls'] is not defined or "
            "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
            "{%- if false %}");
    }
    return src;
}

//
// OpenAI-compatible message model (port of common_chat_msg)
//

struct chat_msg_content_part {
    std::string type;
    std::string text;
};

struct chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;
};

struct chat_msg {
    std::string                        role;
    std::string                        content;
    std::vector<chat_msg_content_part> content_parts;
    std::vector<chat_tool_call>        tool_calls;
    std::string                        reasoning_content;
    std::string                        tool_name;
    std::string                        tool_call_id;

    bool contains_media() const {
        for (const auto & part : content_parts) {
            if (part.type == "media_marker") {
                return true;
            }
        }
        return false;
    }

    // port of common_chat_msg::to_json_oaicompat
    ojson to_json_oaicompat(bool concat_typed_text) const {
        if (!content.empty() && !content_parts.empty()) {
            throw std::runtime_error("Cannot specify both content and content_parts");
        }
        ojson jmsg = ojson::object();
        jmsg["role"] = role;
        if (!content.empty()) {
            jmsg["content"] = content;
        } else if (!content_parts.empty()) {
            if (concat_typed_text || contains_media()) {
                std::string text;
                bool last_was_media_marker = false;
                // join parts with newline, do not add newline before or after media markers
                for (const auto & part : content_parts) {
                    bool add_new_line = true;
                    if (part.type == "text") {
                        add_new_line = !last_was_media_marker && !text.empty();
                        last_was_media_marker = false;
                    } else if (part.type == "media_marker") {
                        add_new_line = false;
                        last_was_media_marker = true;
                    } else {
                        continue; // ignore unknown content part types
                    }
                    if (add_new_line) {
                        text += '\n';
                    }
                    text += part.text;
                }
                jmsg["content"] = text;
            } else {
                ojson parts = ojson::array();
                for (const auto & part : content_parts) {
                    ojson p = ojson::object();
                    p["type"] = part.type;
                    p["text"] = part.text;
                    parts.push_back(std::move(p));
                }
                jmsg["content"] = std::move(parts);
            }
        } else {
            jmsg["content"] = "";
        }
        if (!reasoning_content.empty()) {
            jmsg["reasoning_content"] = reasoning_content;
        }
        if (!tool_name.empty()) {
            jmsg["name"] = tool_name;
        }
        if (!tool_call_id.empty()) {
            jmsg["tool_call_id"] = tool_call_id;
        }
        if (!tool_calls.empty()) {
            ojson jtool_calls = ojson::array();
            for (const auto & tool_call : tool_calls) {
                ojson fn = ojson::object();
                fn["name"] = tool_call.name;
                fn["arguments"] = tool_call.arguments;
                ojson tc = ojson::object();
                tc["type"] = "function";
                tc["function"] = std::move(fn);
                if (!tool_call.id.empty()) {
                    tc["id"] = tool_call.id;
                }
                jtool_calls.push_back(std::move(tc));
            }
            jmsg["tool_calls"] = std::move(jtool_calls);
        }
        return jmsg;
    }
};

// port of common_chat_msgs_parse_oaicompat
std::vector<chat_msg> parse_messages_oaicompat(const ojson & messages) {
    std::vector<chat_msg> msgs;
    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }
        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }
            chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role").get<std::string>();

            bool has_content    = message.contains("content");
            bool has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content.get<std::string>();
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        chat_msg_content_part msg_part;
                        msg_part.type = type.get<std::string>();
                        msg_part.text = part.at("text").get<std::string>();
                        msg.content_parts.push_back(std::move(msg_part));
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " + content.dump());
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name = fc.at("name").get<std::string>();
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args.get<std::string>();
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id").get<std::string>();
                    }
                    msg.tool_calls.push_back(std::move(tc));
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument("Expected 'content' or 'tool_calls' in message: " + message.dump());
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content").get<std::string>();
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name").get<std::string>();
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id").get<std::string>();
            }
            msgs.push_back(std::move(msg));
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }
    return msgs;
}

struct chat_tool {
    std::string name;
    std::string description;
    std::string parameters; // JSON text
};

// port of common_chat_tools_parse_oaicompat
std::vector<chat_tool> parse_tools_oaicompat(const ojson & tools) {
    std::vector<chat_tool> result;
    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }
                const auto & function = tool.at("function");
                chat_tool t;
                t.name        = function.at("name").get<std::string>();
                t.description = function.value("description", std::string());
                t.parameters  = function.value("parameters", ojson::object()).dump();
                result.push_back(std::move(t));
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }
    return result;
}

// port of common_chat_tools_to_json_oaicompat (null when there are no tools)
ojson tools_to_json_oaicompat(const std::vector<chat_tool> & tools) {
    if (tools.empty()) {
        return ojson();
    }
    ojson result = ojson::array();
    for (const auto & tool : tools) {
        ojson fn = ojson::object();
        fn["name"]        = tool.name;
        fn["description"] = tool.description;
        fn["parameters"]  = ojson::parse(tool.parameters);
        ojson t = ojson::object();
        t["type"]     = "function";
        t["function"] = std::move(fn);
        result.push_back(std::move(t));
    }
    return result;
}

//
// content normalization (port of messages_inp_normalizer)
//

// join parts with newline, do not add newline before or after media markers
std::string concat_content_parts(const ojson & parts) {
    std::string text;
    bool last_was_media_marker = false;
    for (const auto & part : parts) {
        std::string type = part.value("type", std::string());
        bool add_new_line = true;
        if (type == "text") {
            add_new_line = !last_was_media_marker && !text.empty();
            last_was_media_marker = false;
        } else if (type == "media_marker") {
            add_new_line = false;
            last_was_media_marker = true;
        } else {
            continue; // ignore unknown content part types
        }
        if (add_new_line) {
            text += '\n';
        }
        text += part.value("text", std::string());
    }
    return text;
}

// handle supports_string_content / supports_typed_content
// if string=true and array=false, convert array to string
// if string=false and array=true, convert string to array
// if both are true, do nothing
ojson normalize_content(const ojson & messages, const jinja::caps & caps) {
    bool only_string = caps.supports_string_content && !caps.supports_typed_content;
    bool only_typed  = !caps.supports_string_content && caps.supports_typed_content;
    if ((!only_string && !only_typed) || !messages.is_array()) {
        return messages;
    }
    ojson normalized = ojson::array();
    for (const auto & msg : messages) {
        ojson copy = msg;
        if (copy.contains("content")) {
            ojson & it = copy.at("content");
            if (only_typed && it.is_string()) {
                ojson part = ojson::object();
                part["type"] = "text";
                part["text"] = it.get<std::string>();
                it = ojson::array({ std::move(part) });
            } else if (only_string && it.is_array()) {
                it = concat_content_parts(it);
            }
        }
        normalized.push_back(std::move(copy));
    }
    return normalized;
}

//
// generic input workarounds (port of chat.cpp `namespace workaround`)
//

namespace workaround {

void map_developer_role_to_system(ojson & messages) {
    for (auto & message : messages) {
        if (message.contains("role") && message["role"] == "developer") {
            message["role"] = "system";
        }
    }
}

// if first message is system and template does not support it, merge it with next message
// (llama.cpp only handles string content here; typed content is merged into the first text part)
void system_message_not_supported(ojson & messages) {
    if (messages.empty() || messages.front().at("role") != "system") {
        return;
    }
    if (messages.size() > 1) {
        const ojson & first_content = messages.front().at("content");
        std::string system_text = first_content.is_array() ? concat_content_parts(first_content)
                                                            : first_content.get<std::string>();
        ojson & second_content = messages[1].at("content");
        if (second_content.is_string()) {
            second_content = system_text + "\n" + second_content.get<std::string>();
        } else if (second_content.is_array()) {
            bool merged = false;
            for (auto & part : second_content) {
                if (part.value("type", std::string()) == "text") {
                    part["text"] = system_text + "\n" + part.value("text", std::string());
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                ojson part = ojson::object();
                part["type"] = "text";
                part["text"] = system_text + "\n";
                second_content.insert(second_content.begin(), std::move(part));
            }
        } else {
            second_content = system_text + "\n";
        }
    }
    // else: drop the system prompt, the template cannot render it
    messages.erase(0);
}

void requires_non_null_content(ojson & messages) {
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

void func_args_not_string(ojson & messages) {
    for (auto & message : messages) {
        if (!message.contains("tool_calls")) {
            continue;
        }
        for (auto & tool_call : message["tool_calls"]) {
            if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                auto & args = tool_call["function"]["arguments"];
                if (args.is_string()) {
                    try {
                        args = ojson::parse(args.get<std::string>());
                    } catch (const std::exception & e) {
                        throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                    }
                }
            }
        }
    }
}

// Trim leading/trailing whitespace from message contents before rendering (StepFun templates)
void trim_all_content(std::vector<chat_msg> & messages) {
    for (auto & message : messages) {
        message.content           = trim_whitespace(message.content);
        message.reasoning_content = trim_whitespace(message.reasoning_content);
        for (auto & part : message.content_parts) {
            if (part.type == "text") {
                part.text = trim_whitespace(part.text);
            }
        }
    }
}

} // namespace workaround

//
// ChatTemplate implementation
//

class ChatTemplateImpl final : public ChatTemplate {
public:
    ChatTemplateImpl(const std::string & source, const std::string & bos_token, const std::string & eos_token)
        : bos_(bos_token), eos_(eos_token) {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(patch_template_source(source));
        prog_ = jinja::parse_from_tokens(lexer_res);
        src_  = lexer_res.source;
        jcaps_ = jinja::caps_get(prog_);

        caps_.supports_system_role         = jcaps_.supports_system_role;
        caps_.supports_tools               = jcaps_.supports_tools;
        caps_.supports_tool_calls          = jcaps_.supports_tool_calls;
        caps_.supports_tool_responses      = jcaps_.supports_tool_responses;
        caps_.supports_parallel_tool_calls = jcaps_.supports_parallel_tool_calls;
        caps_.requires_typed_content       = !jcaps_.supports_string_content && jcaps_.supports_typed_content;
        caps_.supports_enable_thinking     = jcaps_.supports_enable_thinking;
    }

    const std::string & source() const override { return src_; }
    const ChatTemplateCaps & caps() const override { return caps_; }

    std::string apply(const ChatTemplateInputs & inputs) const override {
        try {
            return apply_impl(inputs);
        } catch (const std::runtime_error &) {
            throw;
        } catch (const std::exception & e) {
            // jinja::raised_exception / rethrown_exception derive from std::exception only
            throw std::runtime_error(e.what());
        }
    }

private:
    std::string apply_impl(const ChatTemplateInputs & inputs) const {
        // 1. parse OpenAI-style inputs (validation identical to llama.cpp)
        std::vector<chat_msg>  msgs  = parse_messages_oaicompat(inputs.messages);
        std::vector<chat_tool> tools = parse_tools_oaicompat(inputs.tools);

        if (src_.find("You have access to the following functions in JSONSchema format") != std::string::npos) {
            // StepFun: trim message contents (including typed content parts) before rendering,
            // otherwise leftover whitespace drives the model into reasoning loops (llama.cpp issue #24181)
            workaround::trim_all_content(msgs);
        }

        // 2. render messages to JSON and normalize string/typed content according to the template caps
        ojson messages = ojson::array();
        for (const auto & msg : msgs) {
            messages.push_back(msg.to_json_oaicompat(/* concat_typed_text= */ false));
        }
        messages = normalize_content(messages, jcaps_);

        ojson jtools = tools_to_json_oaicompat(tools);

        // 3. generic workarounds (same conditions as common_chat_templates_apply_jinja)
        if (src_.find("<|channel|>") == std::string::npos) {
            // map developer to system for all models except for GPT-OSS
            workaround::map_developer_role_to_system(messages);
        }
        if (!jcaps_.supports_system_role) {
            workaround::system_message_not_supported(messages);
        }
        if (jcaps_.supports_tool_calls) {
            // some templates require the content field in tool call messages to be non-null
            workaround::requires_non_null_content(messages);
        }
        if (jcaps_.supports_object_arguments) {
            workaround::func_args_not_string(messages);
        }

        // 4. time
        std::time_t now = std::time(nullptr);
        if (!inputs.now_iso8601.empty() && !parse_iso8601(inputs.now_iso8601, now)) {
            throw std::runtime_error("ChatTemplate: invalid now_iso8601 value: '" + inputs.now_iso8601 + "'");
        }

        // 5. build the jinja globals (port of common_chat_template_direct_apply_impl)
        ojson user_globals = ojson::object(); // strings are marked as user input
        ojson tmpl_globals = ojson::object(); // strings are NOT marked as user input

        user_globals["messages"]      = std::move(messages);
        tmpl_globals["bos_token"]      = bos_;
        tmpl_globals["eos_token"]      = eos_;
        tmpl_globals["enable_thinking"] = inputs.enable_thinking;
        if (jtools.is_array()) {
            user_globals["tools"] = std::move(jtools);
        }
        // extra context: llama.cpp always provides datetime / date_string, then chat_template_kwargs
        tmpl_globals["datetime"]    = format_time(now, "%b %d %Y");
        tmpl_globals["date_string"] = format_time(now, "%d %b %Y");
        for (const auto & [key, val] : inputs.extra_context) {
            // user supplied kwargs override everything (matching llama.cpp's overwrite semantics)
            tmpl_globals.erase(key);
            user_globals.erase(key);
            user_globals[key] = ojson(val);
        }
        if (inputs.add_generation_prompt) {
            tmpl_globals.erase("add_generation_prompt");
            user_globals.erase("add_generation_prompt");
            tmpl_globals["add_generation_prompt"] = true;
        }

        jinja::context ctx(src_);
        ctx.current_time = now;

        auto lookup = [&](const char * key) -> const ojson * {
            if (user_globals.contains(key)) return &user_globals.at(key);
            if (tmpl_globals.contains(key)) return &tmpl_globals.at(key);
            return nullptr;
        };
        if (const ojson * v = lookup("preserve_reasoning"); v && v->is_boolean()) {
            jinja::caps_apply_preserve_reasoning(ctx, v->get<bool>());
        }
        if (const ojson * v = lookup("reasoning_effort"); v && v->is_string() && !v->get<std::string>().empty()) {
            jinja::caps_apply_reasoning_effort(ctx, v->get<std::string>());
        }

        jinja::global_from_json(ctx, tmpl_globals, /* mark_input= */ false);
        jinja::global_from_json(ctx, user_globals, /* mark_input= */ true);

        // 6. render
        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(prog_);
        auto parts = jinja::runtime::gather_string_parts(results);
        return parts->as_string().str();
    }

    jinja::program prog_; // note: AST nodes carry transient state during execution, apply() is not reentrant
    std::string src_;
    std::string bos_;
    std::string eos_;
    jinja::caps jcaps_;
    ChatTemplateCaps caps_;
};

} // namespace

std::unique_ptr<ChatTemplate> ChatTemplate::parse(const std::string & source, const std::string & bos_token, const std::string & eos_token) {
    try {
        return std::make_unique<ChatTemplateImpl>(source, bos_token, eos_token);
    } catch (const std::runtime_error &) {
        throw;
    } catch (const std::exception & e) {
        throw std::runtime_error(e.what());
    }
}

const char * chatml_template_source() {
    // same as llama.cpp CHATML_TEMPLATE_SRC
    return
        "{%- for message in messages -%}\n"
        "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "  {{- '<|im_start|>assistant\n' -}}\n"
        "{%- endif -%}";
}

} // namespace iian
