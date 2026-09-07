#pragma once
// Derived from llama.cpp (src/unicode-data.h) - MIT License, see THIRD_PARTY_NOTICES.md.
// Ported into the standalone iian tokenizer: everything lives in namespace iian and
// no libllama headers are required.

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace iian {

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::initializer_list<range_nfd> unicode_ranges_nfd;

} // namespace iian
