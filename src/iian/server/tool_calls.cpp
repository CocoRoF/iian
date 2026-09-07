#include "tool_calls.h"

#include "iian/grammar/grammar.h"
#include "iian/grammar/json_schema.h"
#include "openai.h"

#include <algorithm>
#include <cstring>

namespace iian::server {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

const char * tool_format_name(ToolFormat f) {
    switch (f) {
        case ToolFormat::HERMES: return "hermes";
        case ToolFormat::LLAMA3: return "llama3";
        case ToolFormat::MISTRAL: return "mistral";
        case ToolFormat::GENERIC: return "generic";
        default: return "none";
    }
}

ToolFormat detect_tool_format(const std::string & src) {
    if (src.find("<tool_call>") != std::string::npos) return ToolFormat::HERMES;
    if (src.find("[TOOL_CALLS]") != std::string::npos) return ToolFormat::MISTRAL;
    if (src.find("<|python_tag|>") != std::string::npos || (src.find("\"parameters\"") != std::string::npos && src.find("tool_calls") != std::string::npos)) return ToolFormat::LLAMA3;
    if (src.find("tools") != std::string::npos && src.find("tool_calls") != std::string::npos) return ToolFormat::GENERIC;
    return ToolFormat::NONE;
}

namespace {

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Accepts {"name": "...", "arguments"|"parameters": {...}|"..."} -> ToolCall; false if not a tool call.
bool tool_call_from_json(const json & j, ToolCall & out) {
    if (!j.is_object() || !j.contains("name") || !j["name"].is_string()) return false;
    out.name = j["name"].get<std::string>();
    const json * args = nullptr;
    if (j.contains("arguments")) args = &j["arguments"];
    else if (j.contains("parameters")) args = &j["parameters"];
    if (!args) out.arguments = "{}";
    else if (args->is_string()) out.arguments = args->get<std::string>();
    else out.arguments = args->dump();
    out.id = "call_" + random_id(24);
    return true;
}

// Parses the JSON value starting at text[pos] (object or array). Returns end position or npos.
size_t parse_json_at(const std::string & text, size_t pos, json & out) {
    // fast bracket matching (strings aware) so trailing text after the value is tolerated
    if (pos >= text.size() || (text[pos] != '{' && text[pos] != '[')) return std::string::npos;
    int depth = 0; bool in_str = false, esc = false;
    for (size_t i = pos; i < text.size(); i++) {
        char c = text[i];
        if (in_str) { if (esc) esc = false; else if (c == '\\') esc = true; else if (c == '"') in_str = false; continue; }
        if (c == '"') in_str = true;
        else if (c == '{' || c == '[') depth++;
        else if (c == '}' || c == ']') { if (--depth == 0) {
            out = json::parse(text.substr(pos, i + 1 - pos), nullptr, false);
            return out.is_discarded() ? std::string::npos : i + 1;
        } }
    }
    return std::string::npos;
}

bool collect_calls(const json & v, std::vector<ToolCall> & calls) {
    if (v.is_array()) {
        bool any = false;
        for (auto & e : v) { ToolCall tc; if (tool_call_from_json(e, tc)) { calls.push_back(tc); any = true; } }
        return any;
    }
    ToolCall tc;
    if (tool_call_from_json(v, tc)) { calls.push_back(tc); return true; }
    return false;
}

} // namespace

ParsedAssistant parse_tool_calls(const std::string & text, ToolFormat fmt) {
    ParsedAssistant out;
    if (fmt == ToolFormat::HERMES) {
        std::string content;
        size_t pos = 0;
        while (true) {
            size_t s = text.find("<tool_call>", pos);
            if (s == std::string::npos) { content += text.substr(pos); break; }
            content += text.substr(pos, s - pos);
            size_t body = s + strlen("<tool_call>");
            size_t e = text.find("</tool_call>", body);
            std::string inner = trim(text.substr(body, e == std::string::npos ? std::string::npos : e - body));
            json v = json::parse(inner, nullptr, false);
            if (v.is_discarded() || !collect_calls(v, out.tool_calls)) content += text.substr(s, e == std::string::npos ? std::string::npos : e + strlen("</tool_call>") - s);
            if (e == std::string::npos) break;
            pos = e + strlen("</tool_call>");
        }
        out.content = trim(content);
        return out;
    }
    if (fmt == ToolFormat::MISTRAL) {
        size_t s = text.find("[TOOL_CALLS]");
        if (s != std::string::npos) {
            json v;
            size_t start = text.find_first_of("[{", s + strlen("[TOOL_CALLS]"));
            size_t e = start == std::string::npos ? std::string::npos : parse_json_at(text, start, v);
            if (e != std::string::npos && collect_calls(v, out.tool_calls)) { out.content = trim(text.substr(0, s) + text.substr(e)); return out; }
        }
        out.content = trim(text);
        return out;
    }
    // LLAMA3 / GENERIC / NONE: a JSON object or array at the start of the message (after an optional python tag)
    std::string t = trim(text);
    const char * tag = "<|python_tag|>";
    if (t.rfind(tag, 0) == 0) t = trim(t.substr(strlen(tag)));
    if (!t.empty() && (t[0] == '{' || t[0] == '[')) {
        // llama3 may emit several objects separated by ';'
        size_t pos = 0; bool any = false;
        while (pos < t.size()) {
            json v;
            size_t e = parse_json_at(t, pos, v);
            if (e == std::string::npos || !collect_calls(v, out.tool_calls)) break;
            any = true;
            pos = e;
            while (pos < t.size() && (t[pos] == ';' || t[pos] == ',' || isspace((unsigned char) t[pos]))) pos++;
            if (pos < t.size() && t[pos] != '{' && t[pos] != '[') break;
        }
        if (any) { out.content = trim(t.substr(pos)); return out; }
    }
    out.content = trim(text);
    return out;
}

size_t streamable_prefix(const std::string & partial, ToolFormat fmt) {
    auto held_for_marker = [&](const char * marker) -> size_t {
        size_t s = partial.find(marker);
        if (s != std::string::npos) return s;
        // longest suffix of `partial` that is a prefix of `marker`
        const size_t mlen = strlen(marker);
        for (size_t k = std::min(mlen - 1, partial.size()); k > 0; k--) {
            if (partial.compare(partial.size() - k, k, marker, k) == 0) return partial.size() - k;
        }
        return partial.size();
    };
    switch (fmt) {
        case ToolFormat::HERMES: return held_for_marker("<tool_call>");
        case ToolFormat::MISTRAL: return held_for_marker("[TOOL_CALLS]");
        default: {
            // a message that starts with '{', '[' or the python tag may be a tool call: hold everything
            size_t a = partial.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return 0;   // only whitespace so far: wait
            if (partial[a] == '{' || partial[a] == '[' || partial[a] == '<') return 0;
            return partial.size();
        }
    }
}

std::string tool_call_grammar(const ojson & tools, ToolFormat fmt, bool parallel, const std::string & only) {
    // one JSON schema per tool: {"name": <const>, "arguments": <parameters schema>}
    ojson alternatives = ojson::array();
    const char * args_key = fmt == ToolFormat::LLAMA3 ? "parameters" : "arguments";
    for (const auto & t : tools) {
        if (!t.is_object() || !t.contains("function")) continue;
        const ojson & f = t["function"];
        std::string name = f.value("name", "");
        if (name.empty() || (!only.empty() && name != only)) continue;
        ojson params = f.contains("parameters") && f["parameters"].is_object() ? f["parameters"] : ojson{{"type", "object"}};
        ojson schema = {{"type", "object"}, {"properties", {{"name", {{"const", name}}}, {args_key, params}}}, {"required", ojson::array({"name", args_key})}, {"additionalProperties", false}};
        alternatives.push_back(schema);
    }
    if (alternatives.empty()) throw std::invalid_argument(only.empty() ? "no usable tool definitions" : "tool_choice names a function that is not in 'tools': " + only);
    ojson call_schema = alternatives.size() == 1 ? alternatives[0] : ojson{{"anyOf", alternatives}};
    // wrap with the format markers by building a small GBNF around the schema grammar (root renamed)
    std::string inner = json_schema_to_grammar(call_schema);
    // rename the schema's root rule to `call`
    size_t rp = inner.find("root ::=");
    if (rp == std::string::npos) throw std::invalid_argument("schema grammar has no root rule");
    inner.replace(rp, 4, "call");
    std::string root;
    switch (fmt) {
        case ToolFormat::HERMES:
            root = parallel ? "root ::= (\"<tool_call>\\n\" call \"\\n</tool_call>\" \"\\n\"?)+" : "root ::= \"<tool_call>\\n\" call \"\\n</tool_call>\"";
            break;
        case ToolFormat::MISTRAL:
            root = parallel ? "root ::= \"[TOOL_CALLS][\" call (\",\" call)* \"]\"" : "root ::= \"[TOOL_CALLS][\" call \"]\"";
            break;
        case ToolFormat::LLAMA3:
            root = parallel ? "root ::= call (\";\" call)*" : "root ::= call";
            break;
        default:
            root = parallel ? "root ::= \"[\" call (\",\" call)* \"]\"" : "root ::= call";
            break;
    }
    return root + "\n" + inner;
}

json tool_calls_json(const std::vector<ToolCall> & calls) {
    json a = json::array();
    for (const auto & c : calls) a.push_back({{"id", c.id}, {"type", "function"}, {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
    return a;
}

json tool_calls_delta_json(const std::vector<ToolCall> & calls) {
    json a = json::array();
    for (size_t i = 0; i < calls.size(); i++) a.push_back({{"index", (int) i}, {"id", calls[i].id}, {"type", "function"}, {"function", {{"name", calls[i].name}, {"arguments", calls[i].arguments}}}});
    return a;
}

} // namespace iian::server
