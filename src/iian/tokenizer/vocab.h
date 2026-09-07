#pragma once
// Derived from llama.cpp (src/llama-vocab.h) - MIT License, see THIRD_PARTY_NOTICES.md.
// Internal vocabulary implementation behind iian::Tokenizer. Not part of the public API.
#include "iian/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;

namespace iian {

// pre-tokenization types
enum VocabPreType {
    VOCAB_PRE_TYPE_DEFAULT           = 0,
    VOCAB_PRE_TYPE_LLAMA3            = 1,
    VOCAB_PRE_TYPE_DEEPSEEK_LLM      = 2,
    VOCAB_PRE_TYPE_DEEPSEEK_CODER    = 3,
    VOCAB_PRE_TYPE_FALCON            = 4,
    VOCAB_PRE_TYPE_MPT               = 5,
    VOCAB_PRE_TYPE_STARCODER         = 6,
    VOCAB_PRE_TYPE_GPT2              = 7,
    VOCAB_PRE_TYPE_REFACT            = 8,
    VOCAB_PRE_TYPE_COMMAND_R         = 9,
    VOCAB_PRE_TYPE_STABLELM2         = 10,
    VOCAB_PRE_TYPE_QWEN2             = 11,
    VOCAB_PRE_TYPE_OLMO              = 12,
    VOCAB_PRE_TYPE_DBRX              = 13,
    VOCAB_PRE_TYPE_SMAUG             = 14,
    VOCAB_PRE_TYPE_PORO              = 15,
    VOCAB_PRE_TYPE_CHATGLM3          = 16,
    VOCAB_PRE_TYPE_CHATGLM4          = 17,
    VOCAB_PRE_TYPE_VIKING            = 18,
    VOCAB_PRE_TYPE_JAIS              = 19,
    VOCAB_PRE_TYPE_TEKKEN            = 20,
    VOCAB_PRE_TYPE_SMOLLM            = 21,
    VOCAB_PRE_TYPE_CODESHELL         = 22,
    VOCAB_PRE_TYPE_BLOOM             = 23,
    VOCAB_PRE_TYPE_GPT3_FINNISH      = 24,
    VOCAB_PRE_TYPE_EXAONE            = 25,
    VOCAB_PRE_TYPE_CHAMELEON         = 26,
    VOCAB_PRE_TYPE_MINERVA           = 27,
    VOCAB_PRE_TYPE_DEEPSEEK3_LLM     = 28,
    VOCAB_PRE_TYPE_GPT4O             = 29,
    VOCAB_PRE_TYPE_SUPERBPE          = 30,
    VOCAB_PRE_TYPE_TRILLION          = 31,
    VOCAB_PRE_TYPE_BAILINGMOE        = 32,
    VOCAB_PRE_TYPE_LLAMA4            = 33,
    VOCAB_PRE_TYPE_PIXTRAL           = 34,
    VOCAB_PRE_TYPE_SEED_CODER        = 35,
    VOCAB_PRE_TYPE_HUNYUAN           = 36,
    VOCAB_PRE_TYPE_KIMI_K2           = 37,
    VOCAB_PRE_TYPE_HUNYUAN_DENSE     = 38,
    VOCAB_PRE_TYPE_GROK_2            = 39,
    VOCAB_PRE_TYPE_GRANITE_DOCLING   = 40,
    VOCAB_PRE_TYPE_MINIMAX_M2        = 41,
    VOCAB_PRE_TYPE_AFMOE             = 42,
    VOCAB_PRE_TYPE_SOLAR_OPEN        = 43,
    VOCAB_PRE_TYPE_YOUTU             = 44,
    VOCAB_PRE_TYPE_EXAONE_MOE        = 45,
    VOCAB_PRE_TYPE_QWEN35            = 46,
    VOCAB_PRE_TYPE_TINY_AYA          = 47,
    VOCAB_PRE_TYPE_JOYAI_LLM         = 48,
    VOCAB_PRE_TYPE_JAIS2             = 49,
    VOCAB_PRE_TYPE_GEMMA4            = 50,
    VOCAB_PRE_TYPE_SARVAM_MOE        = 51,
    VOCAB_PRE_TYPE_MINICPM5          = 52,
    VOCAB_PRE_TYPE_WHITESPACE        = 53,
    VOCAB_PRE_TYPE_GRANITE_EMB_MULTI = 54,
    VOCAB_PRE_TYPE_MELLUM2           = 55,
    VOCAB_PRE_TYPE_LAGUNA            = 56,
    VOCAB_PRE_TYPE_HY_V4             = 57,
    VOCAB_PRE_TYPE_SPARK2_5          = 58,
};

struct Vocab {
    struct token_data {
        std::string text;
        float       score;
        TokenAttr   attr;
    };

    struct normalizer_options {
        bool lowercase     = true;
        bool strip_accents = true;
        // TODO: clean_text, handle_chinese_chars
    };

    Vocab();
    ~Vocab();

    // build from GGUF metadata only (throws std::runtime_error on malformed / unsupported vocabularies)
    void load(const gguf_context * ctx);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    VocabType    get_type()     const;
    VocabPreType get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (token_t id) const;
    bool is_unknown     (token_t id) const;
    bool is_control     (token_t id) const;
    bool is_byte        (token_t id) const;
    bool is_user_defined(token_t id) const;
    bool is_unused      (token_t id) const;
    bool is_eog         (token_t id) const;

    uint8_t token_to_byte(token_t id) const;
    token_t byte_to_token(uint8_t ch) const;

    token_t text_to_token(const std::string & text) const;

    const token_data & get_token_data(token_t id) const;

    const char * token_get_text (token_t id) const;
    float        token_get_score(token_t id) const;
    TokenAttr    token_get_attr (token_t id) const;

    token_t token_bos() const;
    token_t token_eos() const;
    token_t token_eot() const;
    token_t token_eom() const;
    token_t token_unk() const;
    token_t token_sep() const;
    token_t token_nl () const;
    token_t token_pad() const;
    token_t token_mask() const;

    token_t token_prefix() const;
    token_t token_middle() const;
    token_t token_suffix() const;

    token_t token_fim_pre() const;
    token_t token_fim_suf() const;
    token_t token_fim_mid() const;
    token_t token_fim_pad() const;
    token_t token_fim_rep() const;
    token_t token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;
    const normalizer_options & get_normalizer_opts() const;

    const std::vector<token_t> & get_suppress_tokens() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                      token_t * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<token_t> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                      token_t   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(token_t token) const;

    int32_t detokenize(
                const token_t * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<token_t> & tokens,
                                  bool   special) const;

    // "tokenizer.chat_template" (variant == "") or "tokenizer.chat_template.<variant>"; "" if absent
    std::string chat_template(const std::string & variant = "") const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace iian
