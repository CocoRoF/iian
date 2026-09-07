// iian::IncrementalDetokenizer - streaming-safe detokenization that only ever emits complete UTF-8 sequences.
#include "iian/tokenizer.h"

#include <cstdint>

namespace iian {

namespace {

// Expected sequence length for a UTF-8 lead byte, or 0 for continuation / invalid lead bytes.
inline size_t utf8_seq_len(uint8_t c) {
    if (c < 0x80)           return 1;
    if ((c & 0xE0) == 0xC0) return (c >= 0xC2) ? 2 : 0; // C0/C1 are always invalid (overlong)
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return (c <= 0xF4) ? 4 : 0; // > U+10FFFF is invalid
    return 0;
}

inline bool is_cont(uint8_t c) {
    return (c & 0xC0) == 0x80;
}

// Returns the length of the longest prefix of `s` that ends on a UTF-8 sequence boundary.
// Invalid bytes are treated as complete single-byte units (emitted raw, never held back);
// only a *plausible* incomplete trailing sequence is held back.
size_t utf8_complete_prefix(const std::string & s) {
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const uint8_t c = (uint8_t) s[i];
        size_t len = utf8_seq_len(c);
        if (len == 0) {
            // stray continuation byte or invalid lead byte: pass through as-is
            i += 1;
            continue;
        }
        if (i + len > n) {
            // possibly incomplete: hold back only if every byte we have so far is a valid continuation
            bool plausible = true;
            for (size_t j = i + 1; j < n; ++j) {
                if (!is_cont((uint8_t) s[j])) {
                    plausible = false;
                    break;
                }
            }
            if (plausible) {
                return i;
            }
            i += 1;
            continue;
        }
        bool valid = true;
        for (size_t j = 1; j < len; ++j) {
            if (!is_cont((uint8_t) s[i + j])) {
                valid = false;
                break;
            }
        }
        i += valid ? len : 1;
    }
    return n;
}

} // namespace

IncrementalDetokenizer::IncrementalDetokenizer(const Tokenizer & tok, bool skip_special)
    : tok_(tok), skip_special_(skip_special) {}

std::string IncrementalDetokenizer::push(token_t tok) {
    // special=false hides control/unknown tokens (returns ""), matching llama_token_to_piece(..., special)
    pending_ += tok_.token_to_piece(tok, /*special=*/!skip_special_);

    const size_t n_complete = utf8_complete_prefix(pending_);
    if (n_complete == 0) {
        return "";
    }
    std::string out = pending_.substr(0, n_complete);
    pending_.erase(0, n_complete);
    out_ += out;
    return out;
}

std::string IncrementalDetokenizer::finish() {
    // whatever is left is an incomplete sequence: emit the raw bytes rather than dropping them
    std::string out;
    out.swap(pending_);
    out_ += out;
    return out;
}

} // namespace iian
