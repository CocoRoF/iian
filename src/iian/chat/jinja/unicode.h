#pragma once

// Minimal UTF-8 helpers used by the jinja engine (derived from llama.cpp common/unicode.h, MIT).

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace iian::jinja {

struct utf8_parse_result {
    uint32_t codepoint;      // Decoded codepoint (only valid if status == SUCCESS)
    size_t bytes_consumed;   // How many bytes this codepoint uses (1-4)
    enum status { SUCCESS, INCOMPLETE, INVALID } status;

    utf8_parse_result(enum status s, uint32_t cp = 0, size_t bytes = 0)
        : codepoint(cp), bytes_consumed(bytes), status(s) {}
};

// Parse a single UTF-8 codepoint from input at the given byte offset
utf8_parse_result parse_utf8_codepoint(std::string_view input, size_t offset);

} // namespace iian::jinja
