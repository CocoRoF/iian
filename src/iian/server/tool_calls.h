#pragma once
// Tool-call support: detect the format a chat template expects, parse tool calls out of generated text,
// and build a grammar that forces a well-formed tool call (tool_choice = required / a named function).
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace iian::server {

enum class ToolFormat {
    NONE,      // template has no tool support: fall back to GENERIC parsing when tools are given
    HERMES,    // <tool_call>{"name":..,"arguments":{..}}</tool_call>   (Qwen 2.5/3, Hermes, many ChatML models)
    LLAMA3,    // {"name":..,"parameters":{..}} optionally after <|python_tag|>  (Llama 3.x)
    MISTRAL,   // [TOOL_CALLS][{"name":..,"arguments":{..}}]
    GENERIC,   // a bare JSON object {"name":..,"arguments"|"parameters":{..}} (or an array of them)
};

const char * tool_format_name(ToolFormat f);
ToolFormat detect_tool_format(const std::string & template_source);

struct ToolCall {
    std::string id;          // "call_..."
    std::string name;
    std::string arguments;   // JSON text
};

struct ParsedAssistant {
    std::string content;             // text outside tool calls (trimmed)
    std::vector<ToolCall> tool_calls;
};

// Parses `text` (a complete assistant message). Never throws; unparseable calls stay in `content`.
ParsedAssistant parse_tool_calls(const std::string & text, ToolFormat fmt);

// Number of leading bytes of a partial message that can be streamed as plain content without risking
// cutting into a tool call (the rest is held back until the message is complete).
size_t streamable_prefix(const std::string & partial, ToolFormat fmt);

// GBNF grammar forcing one tool call (or, with `parallel`, one or more) chosen from `tools`
// (OpenAI tools array). `only` restricts to a single function name (tool_choice = {function:{name}}).
std::string tool_call_grammar(const nlohmann::ordered_json & tools, ToolFormat fmt, bool parallel, const std::string & only);

nlohmann::json tool_calls_json(const std::vector<ToolCall> & calls);          // OpenAI message.tool_calls
nlohmann::json tool_calls_delta_json(const std::vector<ToolCall> & calls);    // OpenAI delta.tool_calls (with index)

} // namespace iian::server
