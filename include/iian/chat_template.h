#pragma once
// Jinja chat template rendering (OpenAI-style messages -> prompt string).
// The Jinja engine is derived from llama.cpp's common/jinja (MIT), made standalone under namespace iian::jinja.
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "nlohmann/json.hpp"

namespace iian {

struct ChatTemplateCaps {
    bool supports_system_role  = true;
    bool supports_tools        = false;
    bool supports_tool_calls   = false;
    bool supports_tool_responses = false;
    bool supports_parallel_tool_calls = false;
    bool requires_typed_content = false;   // content must be [{"type":"text","text":...}]
    bool supports_enable_thinking = false; // template honors enable_thinking kwarg
};

struct ChatTemplateInputs {
    // OpenAI-format messages: [{"role": "...", "content": "..." | [...], "tool_calls": [...], ...}]
    // ordered_json preserves key order so `tojson` output of tool parameters/arguments matches llama.cpp.
    // (assigning a plain nlohmann::json converts implicitly, with keys sorted)
    nlohmann::ordered_json messages;
    nlohmann::ordered_json tools;          // OpenAI tools array (may be null/empty)
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::map<std::string, nlohmann::ordered_json> extra_context;  // chat_template_kwargs
    std::string now_iso8601;               // for strftime_now; empty -> current time
};

class ChatTemplate {
public:
    // source: raw Jinja template text. bos/eos: token strings exposed as bos_token / eos_token variables.
    static std::unique_ptr<ChatTemplate> parse(const std::string & source, const std::string & bos_token, const std::string & eos_token);
    virtual ~ChatTemplate() = default;

    virtual const std::string & source() const = 0;
    virtual const ChatTemplateCaps & caps() const = 0;
    // Render. Throws std::runtime_error with a template-source-traced message on error.
    virtual std::string apply(const ChatTemplateInputs & inputs) const = 0;
};

// Built-in fallback template (ChatML) used when a model has no embedded template.
const char * chatml_template_source();

} // namespace iian
