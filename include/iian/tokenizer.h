#pragma once
// iian tokenizer: GGUF-native vocabulary + tokenization.
// Implementation is derived from llama.cpp's llama-vocab (MIT, see THIRD_PARTY_NOTICES.md)
// but is fully standalone: it reads the GGUF metadata directly and has no dependency on libllama.
#include "iian/types.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct gguf_context;

namespace iian {

enum class VocabType { NONE, SPM, BPE, WPM, UGM, RWKV, PLAMO2 };

enum TokenAttr : uint32_t {
    TOKEN_ATTR_UNDEFINED    = 0,
    TOKEN_ATTR_UNKNOWN      = 1 << 0,
    TOKEN_ATTR_UNUSED       = 1 << 1,
    TOKEN_ATTR_NORMAL       = 1 << 2,
    TOKEN_ATTR_CONTROL      = 1 << 3,
    TOKEN_ATTR_USER_DEFINED = 1 << 4,
    TOKEN_ATTR_BYTE         = 1 << 5,
    TOKEN_ATTR_NORMALIZED   = 1 << 6,
    TOKEN_ATTR_LSTRIP       = 1 << 7,
    TOKEN_ATTR_RSTRIP       = 1 << 8,
    TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
};

struct SpecialTokens {
    token_t bos = TOKEN_NULL;
    token_t eos = TOKEN_NULL;
    token_t eot = TOKEN_NULL;
    token_t eom = TOKEN_NULL;
    token_t unk = TOKEN_NULL;
    token_t sep = TOKEN_NULL;
    token_t pad = TOKEN_NULL;
    token_t nl  = TOKEN_NULL;
    token_t mask = TOKEN_NULL;
    token_t fim_pre = TOKEN_NULL, fim_suf = TOKEN_NULL, fim_mid = TOKEN_NULL;
    token_t fim_pad = TOKEN_NULL, fim_rep = TOKEN_NULL, fim_sep = TOKEN_NULL;
    bool add_bos = false;
    bool add_eos = false;
    bool add_sep = false;
};

class Tokenizer {
public:
    // Build from an already-opened GGUF context (metadata only; no tensor data needed).
    static std::unique_ptr<Tokenizer> from_gguf(const gguf_context * ctx);
    virtual ~Tokenizer() = default;

    virtual VocabType type() const = 0;
    virtual std::string type_name() const = 0;   // e.g. "BPE", "SPM"
    virtual std::string pre_type_name() const = 0; // e.g. "llama3", "qwen2"
    virtual int32_t n_tokens() const = 0;

    // text -> tokens.  add_special: prepend BOS (if the vocab wants it) / append EOS.
    // parse_special: treat special-token text in the input as special tokens (true for trusted prompts,
    // false for untrusted user text to prevent special-token injection).
    virtual std::vector<token_t> encode(std::string_view text, bool add_special, bool parse_special) const = 0;

    // Single-token piece (raw bytes as stored in the vocab, byte-tokens converted to bytes).
    // special=false hides control tokens (returns "").
    virtual std::string token_to_piece(token_t tok, bool special) const = 0;

    // tokens -> text (remove_special strips BOS/EOS; unparse_special renders control tokens as text)
    virtual std::string decode(const std::vector<token_t> & toks, bool remove_special, bool unparse_special) const = 0;

    virtual const SpecialTokens & special() const = 0;
    virtual bool is_eog(token_t tok) const = 0;      // end-of-generation (EOS, EOT, EOM, ...)
    virtual bool is_control(token_t tok) const = 0;
    virtual bool is_byte(token_t tok) const = 0;
    virtual const std::string & token_text(token_t tok) const = 0;
    virtual TokenAttr token_attr(token_t tok) const = 0;
    virtual token_t text_to_token(std::string_view text) const = 0; // exact vocab lookup, TOKEN_NULL if absent

    // Raw chat template string embedded in the GGUF (tokenizer.chat_template), or "" if none.
    virtual std::string chat_template(const std::string & variant = "") const = 0;
};

// Incremental (streaming) detokenizer: feeds tokens one by one and emits only complete UTF-8 text,
// holding back partial multi-byte sequences until they complete.
class IncrementalDetokenizer {
public:
    explicit IncrementalDetokenizer(const Tokenizer & tok, bool skip_special = true);
    // Returns the newly produced text for this token ("" if held back).
    std::string push(token_t tok);
    // Flush any held-back bytes (e.g. at end of generation).
    std::string finish();
    // All text emitted so far.
    const std::string & text() const { return out_; }
private:
    const Tokenizer & tok_;
    bool skip_special_;
    std::string pending_;
    std::string out_;
};

} // namespace iian
