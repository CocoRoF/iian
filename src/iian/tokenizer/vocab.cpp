// Derived from llama.cpp (src/llama-vocab.cpp) - MIT License, see THIRD_PARTY_NOTICES.md.
// Ported into the standalone iian tokenizer: everything lives in namespace iian and
// no libllama headers are required.
#include "vocab.h"

#include "ggml.h"
#include "gguf.h"
#include "tokenizer_log.h"

#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <climits>
#include <stdexcept>
#include <string_view>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <forward_list>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

namespace iian {

//
// small local helpers (replacing llama-impl.h / llama-model-loader.h)
//

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    const int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// GGUF metadata keys used by the vocabulary (spellings from llama.cpp src/llama-arch.cpp, LLM_KV_NAMES)
static constexpr const char * KV_GENERAL_ARCHITECTURE             = "general.architecture";
static constexpr const char * KV_GENERAL_NAME                     = "general.name";
static constexpr const char * KV_TOKENIZER_MODEL                  = "tokenizer.ggml.model";
static constexpr const char * KV_TOKENIZER_PRE                    = "tokenizer.ggml.pre";
static constexpr const char * KV_TOKENIZER_LIST                   = "tokenizer.ggml.tokens";
static constexpr const char * KV_TOKENIZER_TOKEN_TYPE             = "tokenizer.ggml.token_type";
static constexpr const char * KV_TOKENIZER_TOKEN_TYPE_COUNT       = "tokenizer.ggml.token_type_count";
static constexpr const char * KV_TOKENIZER_SCORES                 = "tokenizer.ggml.scores";
static constexpr const char * KV_TOKENIZER_MERGES                 = "tokenizer.ggml.merges";
static constexpr const char * KV_TOKENIZER_BOS_ID                 = "tokenizer.ggml.bos_token_id";
static constexpr const char * KV_TOKENIZER_EOS_ID                 = "tokenizer.ggml.eos_token_id";
static constexpr const char * KV_TOKENIZER_EOT_ID                 = "tokenizer.ggml.eot_token_id";
static constexpr const char * KV_TOKENIZER_EOM_ID                 = "tokenizer.ggml.eom_token_id";
static constexpr const char * KV_TOKENIZER_UNK_ID                 = "tokenizer.ggml.unknown_token_id";
static constexpr const char * KV_TOKENIZER_SEP_ID                 = "tokenizer.ggml.seperator_token_id";
static constexpr const char * KV_TOKENIZER_PAD_ID                 = "tokenizer.ggml.padding_token_id";
static constexpr const char * KV_TOKENIZER_MASK_ID                = "tokenizer.ggml.mask_token_id";
static constexpr const char * KV_TOKENIZER_ADD_BOS                = "tokenizer.ggml.add_bos_token";
static constexpr const char * KV_TOKENIZER_ADD_EOS                = "tokenizer.ggml.add_eos_token";
static constexpr const char * KV_TOKENIZER_ADD_SEP                = "tokenizer.ggml.add_sep_token";
static constexpr const char * KV_TOKENIZER_ADD_PREFIX             = "tokenizer.ggml.add_space_prefix";
static constexpr const char * KV_TOKENIZER_REMOVE_EXTRA_WS        = "tokenizer.ggml.remove_extra_whitespaces";
static constexpr const char * KV_TOKENIZER_PRECOMPILED_CHARSMAP   = "tokenizer.ggml.precompiled_charsmap";
static constexpr const char * KV_TOKENIZER_CHAT_TEMPLATE          = "tokenizer.chat_template";
static constexpr const char * KV_TOKENIZER_NORMALIZER_LOWERCASE   = "tokenizer.ggml.normalizer.lowercase";
static constexpr const char * KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS = "tokenizer.ggml.normalizer.strip_accents";
static constexpr const char * KV_TOKENIZER_FIM_PRE_ID             = "tokenizer.ggml.fim_pre_token_id";
static constexpr const char * KV_TOKENIZER_FIM_SUF_ID             = "tokenizer.ggml.fim_suf_token_id";
static constexpr const char * KV_TOKENIZER_FIM_MID_ID             = "tokenizer.ggml.fim_mid_token_id";
static constexpr const char * KV_TOKENIZER_FIM_PAD_ID             = "tokenizer.ggml.fim_pad_token_id";
static constexpr const char * KV_TOKENIZER_FIM_REP_ID             = "tokenizer.ggml.fim_rep_token_id";
static constexpr const char * KV_TOKENIZER_FIM_SEP_ID             = "tokenizer.ggml.fim_sep_token_id";
static constexpr const char * KV_TOKENIZER_SUPPRESS_TOKENS        = "tokenizer.ggml.suppress_tokens";
// deprecated FIM keys
static constexpr const char * KV_TOKENIZER_PREFIX_ID              = "tokenizer.ggml.prefix_token_id";
static constexpr const char * KV_TOKENIZER_SUFFIX_ID              = "tokenizer.ggml.suffix_token_id";
static constexpr const char * KV_TOKENIZER_MIDDLE_ID              = "tokenizer.ggml.middle_token_id";

// token types as stored in the GGUF "tokenizer.ggml.token_type" array (llama_token_type in llama.h)
enum gguf_token_type {
    GGUF_TOKEN_TYPE_UNDEFINED    = 0,
    GGUF_TOKEN_TYPE_NORMAL       = 1,
    GGUF_TOKEN_TYPE_UNKNOWN      = 2,
    GGUF_TOKEN_TYPE_CONTROL      = 3,
    GGUF_TOKEN_TYPE_USER_DEFINED = 4,
    GGUF_TOKEN_TYPE_UNUSED       = 5,
    GGUF_TOKEN_TYPE_BYTE         = 6,
};

// ---- scalar GGUF accessors -------------------------------------------------------------------------
// Each returns true if the key exists (and stores its value), false if absent. A key of the wrong type is an
// error (throws), except that integer-typed values are accepted for any integer target and bool accepts
// integer types as well (a small leniency over llama.cpp's strict type check; it never changes results for
// files produced by convert_hf_to_gguf.py).

static bool gguf_get_str(const gguf_context * ctx, const char * key, std::string & out, bool required = false) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key));
        }
        return false;
    }
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
            key, gguf_type_name(gguf_get_kv_type(ctx, kid)), gguf_type_name(GGUF_TYPE_STRING)));
    }
    out = gguf_get_val_str(ctx, kid);
    return true;
}

static bool gguf_get_i64_any(const gguf_context * ctx, const char * key, int64_t & out, bool required, bool allow_bool) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key));
        }
        return false;
    }
    const gguf_type t = gguf_get_kv_type(ctx, kid);
    switch (t) {
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8 (ctx, kid); return true;
        case GGUF_TYPE_INT8:   out = gguf_get_val_i8 (ctx, kid); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx, kid); return true;
        case GGUF_TYPE_INT16:  out = gguf_get_val_i16(ctx, kid); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx, kid); return true;
        case GGUF_TYPE_INT32:  out = gguf_get_val_i32(ctx, kid); return true;
        case GGUF_TYPE_UINT64: out = (int64_t) gguf_get_val_u64(ctx, kid); return true;
        case GGUF_TYPE_INT64:  out = gguf_get_val_i64(ctx, kid); return true;
        case GGUF_TYPE_BOOL:
            if (allow_bool) {
                out = gguf_get_val_bool(ctx, kid) ? 1 : 0;
                return true;
            }
            break;
        default:
            break;
    }
    throw std::runtime_error(format("key %s has wrong type %s but expected an integer type", key, gguf_type_name(t)));
}

static bool gguf_get_u32(const gguf_context * ctx, const char * key, uint32_t & out, bool required = false) {
    int64_t v = 0;
    if (!gguf_get_i64_any(ctx, key, v, required, /*allow_bool=*/false)) {
        return false;
    }
    out = (uint32_t) v;
    return true;
}

static bool gguf_get_bool(const gguf_context * ctx, const char * key, bool & out, bool required = false) {
    int64_t v = 0;
    if (!gguf_get_i64_any(ctx, key, v, required, /*allow_bool=*/true)) {
        return false;
    }
    out = v != 0;
    return true;
}


//
// helpers
//

struct naive_trie {
    naive_trie() : has_value(false), value(0) {
    }
    void insert(const char * key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->has_value = true;
            this->value = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        } else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }
    std::pair<const char *, size_t> get_longest_prefix(const char * key, size_t len, size_t offset = 0) const {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        }

        return std::make_pair(key, offset);
    }
    const struct naive_trie * traverse(const char c) const {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        }

        return NULL;
    }
    std::map<char, struct naive_trie> children;
    bool has_value;
    token_t value;
};

//
// tokenizers
//

struct llm_tokenizer {
    llm_tokenizer() {}
    virtual ~llm_tokenizer() = default;
};

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

//
// SPM tokenizer
// original implementation:
// https://github.com/ggml-org/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
//

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm : llm_tokenizer {
    llm_tokenizer_spm(const Vocab & /*vocab*/) {}
};

struct llm_tokenizer_spm_session {
    llm_tokenizer_spm_session(const Vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<token_t> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (int i = 1; i < (int) symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //IIAN_TOK_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<token_t> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.text_to_token(text);

        // Do we need to support is_unused?
        if (token != TOKEN_NULL) {
            output.push_back(token);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                token_t id = vocab.byte_to_token(symbol.text[j]);
                output.push_back(id);
            }
            return;
        }

        resegment(symbols[p->second.first], output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.text_to_token(text);

        if (token == TOKEN_NULL) {
            return;
        }

        if (static_cast<uint32_t>(token) >= vocab.n_tokens()) {
            return;
        }

        const auto & tok_data = vocab.get_token_data(token);

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const Vocab & vocab;
    // currently unused
    // const llm_tokenizer_spm * spm_tokenizer;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
    std::map<std::string, std::pair<int, int>> rev_merge;
};

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class iian_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() =  delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = iian_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe : llm_tokenizer {
    llm_tokenizer_bpe(const Vocab & vocab) {
        GGML_ASSERT(vocab.get_type() == VocabType::BPE);
        switch (vocab.get_pre_type()) {
            case VOCAB_PRE_TYPE_LLAMA3:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                    // adapted: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2080233989
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_JAIS2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",

                    // adapted: same as llama3 but with cascading whitespace pattern
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
                };
                break;
            case VOCAB_PRE_TYPE_DBRX:
            case VOCAB_PRE_TYPE_SMAUG:
                regex_exprs = {
                    // same as llama3
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_DEEPSEEK_LLM:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                    "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                    "\\s+$",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}+",
                };
                break;
            case VOCAB_PRE_TYPE_DEEPSEEK3_LLM:
            case VOCAB_PRE_TYPE_HUNYUAN_DENSE:
            case VOCAB_PRE_TYPE_JOYAI_LLM:
            case VOCAB_PRE_TYPE_HY_V4:
                regex_exprs = {
                    "\\p{N}{1,3}",
                    "[一-龥぀-ゟ゠-ヿ]+",
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_SPARK2_5:
                regex_exprs = {
                    "\\p{N}{1,3}",
                    "[一-龥぀-ゟ゠-ヿ]+",
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
                    "\\p{N}",
                };
                break;
            case VOCAB_PRE_TYPE_YOUTU:
                regex_exprs = {
                    "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_DEEPSEEK_CODER:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?\\p{L}+",
                    "\\s?\\p{P}+",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}",
                };
                break;
            case VOCAB_PRE_TYPE_FALCON:
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "[0-9][0-9][0-9]",
                };
                break;
            case VOCAB_PRE_TYPE_STARCODER:
            case VOCAB_PRE_TYPE_REFACT:
            case VOCAB_PRE_TYPE_COMMAND_R:
            case VOCAB_PRE_TYPE_SMOLLM:
            case VOCAB_PRE_TYPE_CODESHELL:
            case VOCAB_PRE_TYPE_EXAONE:
            case VOCAB_PRE_TYPE_MINERVA:
            case VOCAB_PRE_TYPE_MELLUM2:
                regex_exprs = {
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case VOCAB_PRE_TYPE_GPT2:
            case VOCAB_PRE_TYPE_MPT:
            case VOCAB_PRE_TYPE_OLMO:
            case VOCAB_PRE_TYPE_JAIS:
            case VOCAB_PRE_TYPE_TRILLION:
            case VOCAB_PRE_TYPE_GRANITE_DOCLING:
                regex_exprs = {
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case VOCAB_PRE_TYPE_STABLELM2:
            case VOCAB_PRE_TYPE_QWEN2:
            case VOCAB_PRE_TYPE_HUNYUAN:
            case VOCAB_PRE_TYPE_SOLAR_OPEN:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_QWEN35:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_PORO:
            case VOCAB_PRE_TYPE_BLOOM:
            case VOCAB_PRE_TYPE_GPT3_FINNISH:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                };
                break;
            case VOCAB_PRE_TYPE_CHATGLM4:
                regex_exprs = {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_VIKING:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                    "\\p{N}",
                };
                break;
            case VOCAB_PRE_TYPE_TEKKEN:
                // original regex from tokenizer.json
                // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                regex_exprs = {
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_CHAMELEON:
                // Note: in theory, the special token (sentinel and image token) regex_exprs below
                // are unnecessary, as they are split in `tokenizer_st_partition` anyway.
                // However, since the upstream pre-tokenizer uses them, they are also
                // included here (see https://huggingface.co/facebook/chameleon-7b).
                regex_exprs = {
                    "<sentinel:[0-9]+>",  // Sentinel tokens
                    "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",  // Image tokens
                    "([\\t\\n]|    |  )",  // directly from tokenizer.json
                    "\\p{N}", // Individual digits
                    "[\\p{P}!-/:-@\\[-`{-~]",  // Punctuation, Isolated
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case VOCAB_PRE_TYPE_GPT4O:
            case VOCAB_PRE_TYPE_MINIMAX_M2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_GRANITE_EMB_MULTI:
                // Same lookaheads as GPT4O but with \p{M} added so combining marks
                // (diacritics) attach to their base letters. Avoids excessive
                // backtracking on scripts that use them heavily (Bengali, Hindi,
                // Telugu, Thai, ...). See PR #22716 for benchmarks.
                regex_exprs = {
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_TINY_AYA:
                regex_exprs = {
                    // original regex from tokenizer.json: "\\d{1,3}(?=(?:\\d{3})*\\b)"
                    "\\d{1,3}(?=(?:\\d{3})*\\b)",
                    // original regex from tokenizer.json: "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_KIMI_K2:
                regex_exprs = {
                    // K2 trigger pattern - this will activate the custom K2 handler in unicode.cpp
                    // The custom handler implements all K2 patterns with proper Han character exclusion
                    "\\p{Han}+",
                };
                break;
            case VOCAB_PRE_TYPE_SUPERBPE:
                regex_exprs = {
                    "\\p{N}+",
                    "(?=(\\d{3})+(?!\\d))",
                };
                break;
            case VOCAB_PRE_TYPE_BAILINGMOE:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
                    // FIXME? Changed possessive quantifiers (?+ and ++) to greedy to avoid errors and imatrix hanging (tried atomic grouping but it's not supported?)
                    "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_SEED_CODER:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_GROK_2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_AFMOE:
                regex_exprs = {
                    // Digit handling - uses custom implementation in unicode.cpp
                    // Groups digits with leading 1-2 based on total length modulo 3
                    "\\p{AFMoE_digits}",
                    // CJK and Asian scripts (using direct Unicode literals)
                    "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
                    // Main BPE pattern
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_LAGUNA:
                regex_exprs = {
                    "[^\\n]+|[\\n]+",
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_EXAONE_MOE:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_GEMMA4:
                // Gemma4 uses SPM-style BPE: spaces are replaced with ▁ by the
                // normalizer, then BPE merges run on the whole text without
                // word-level pre-splitting. We only need to split on newlines
                // since BPE merge lookup asserts no newlines in tokens.
                regex_exprs = {
                    "[^\\n]+|[\\n]+",
                };
                byte_encode = false; // uses raw UTF-8, not GPT-2 byte encoding
                break;
            case VOCAB_PRE_TYPE_SARVAM_MOE:
                // Sarvam uses SPM-style BPE (same shape as Gemma4): spaces replaced with U+2581
                // by the normalizer, BPE merges over the whole text on raw UTF-8.
                regex_exprs = {
                    "[^\\n]+|[\\n]+",
                };
                byte_encode = false;
                break;
            case VOCAB_PRE_TYPE_MINICPM5:
                regex_exprs = {
                    // original regex from tokenizer.json (openbmb/MiniCPM5-1B)
                    "\\p{N}{1,3}",
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case VOCAB_PRE_TYPE_WHITESPACE:
                // whitespace pre-tokenizer (jinaai/jina-embeddings-v2-base-zh)
                regex_exprs = {
                    "\\S+",
                };
                byte_encode = false;
                break;
            default:
                // default regex for BPE tokenization pre-processing
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]",
                };
                break;
        }
    }

    std::vector<std::string> regex_exprs;
    bool byte_encode = true; // GPT-2 byte encoding; false for SPM-style BPE (raw UTF-8)
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const Vocab & vocab, const llm_tokenizer_bpe & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    virtual ~llm_tokenizer_bpe_session() = default;

    static void append(const token_t token_id, std::vector<token_t> & output)  {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<token_t> & output) const {
        if (vocab.get_add_bos()) {
            GGML_ASSERT(vocab.token_bos() != TOKEN_NULL);
            output.push_back(vocab.token_bos());
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<token_t> & output) const {
        if (vocab.get_add_eos()) {
            GGML_ASSERT(vocab.token_eos() != TOKEN_NULL);
            output.push_back(vocab.token_eos());
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<token_t> & output) const {
        if (vocab.get_add_bos() && output.size() >= 2 && output[1] == vocab.token_bos()) {
            IIAN_TOK_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.get_add_eos() && output.size() >= 2 && *(output.end()-2) == vocab.token_eos()) {
            IIAN_TOK_LOG_WARN(
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    virtual void tokenize(const std::string & text, std::vector<token_t> & output) {
        int final_prev_index = -1;
        const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs, tokenizer.byte_encode);

        symbols_final.clear();
        auto tok_pre = vocab.get_pre_type();

        for (const auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
            if (vocab.get_ignore_merges() && vocab.text_to_token(word) != TOKEN_NULL) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            } else if (tok_pre == VOCAB_PRE_TYPE_GEMMA4 && word.find_first_not_of('\n') == std::string::npos) {
                // fix for gemma 4, ref: https://github.com/ggml-org/llama.cpp/pull/21343
                auto tok = vocab.text_to_token(word);
                if (tok != TOKEN_NULL) {
                    symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                    offset = word.size();
                }
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (int i = 1; i < (int) symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.text_to_token(str);

                if (token == TOKEN_NULL) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        token_t token_multibyte = TOKEN_NULL;
                        if (tokenizer.byte_encode) {
                            std::string byte_str(1, *j);
                            token_multibyte = vocab.text_to_token(byte_str);
                        } else {
                            // For non-byte-encoded BPE (e.g. gemma-4), byte tokens use <0xXX> format
                            static const char * hex = "0123456789ABCDEF";
                            const uint8_t ch = (uint8_t)*j;
                            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
                            token_multibyte = vocab.text_to_token(buf);
                        }
                        if (token_multibyte != TOKEN_NULL) {
                            output.push_back(token_multibyte);
                        }
                    }
                } else {
                    output.push_back(token);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    const Vocab & vocab;
    const llm_tokenizer_bpe & tokenizer;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};

//
// WPM tokenizer
//

struct llm_tokenizer_wpm : llm_tokenizer {
    llm_tokenizer_wpm(const Vocab & /*vocab*/) {}
};

struct llm_tokenizer_wpm_session {
    llm_tokenizer_wpm_session(const Vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<token_t> & output) {
        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text, vocab.get_normalizer_opts());
        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string & word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len() + 1); j > i; j--) {
                    auto id = vocab.text_to_token(word1.substr(i, j - i));
                    if (id != TOKEN_NULL) {
                        output.push_back(id);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.token_unk());
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    static std::vector<std::string> preprocess(const std::string & text, const Vocab::normalizer_options & normalizer_opts)  {
        std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
        if (normalizer_opts.strip_accents) {
            cpts = unicode_cpts_normalize_nfd(cpts);
        }
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts) {
            const auto flags = unicode_cpt_flags_from_cpt(cpt);

            if (flags.is_whitespace) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            assert (!flags.is_separator);
            if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
                continue;
            }

            if (normalizer_opts.strip_accents && flags.is_accent_mark) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(normalizer_opts.lowercase ? unicode_tolower(cpt) : cpt);
            if (flags.is_punctuation || ( cpt < 0x7F && flags.is_symbol ) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }

    static bool is_chinese_char(uint32_t cpt) {
        return
            (cpt >= 0x04E00 && cpt <= 0x09FFF) ||
            (cpt >= 0x03400 && cpt <= 0x04DBF) ||
            (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
            (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
            (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
            (cpt >= 0x2B920 && cpt <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (cpt >= 0x0F900 && cpt <= 0x0FAFF) ||
            (cpt >= 0x2F800 && cpt <= 0x2FA1F);
            //(cpt >= 0x3000  && cpt <= 0x303F)  ||
            //(cpt >= 0xFF00  && cpt <= 0xFFEF);
    }

private:
    const Vocab & vocab;
    // currently unused
    // const llm_tokenizer_wpm * wpm_tokenizer;
};

//
// UGM tokenizer
//

struct llm_tokenizer_ugm : llm_tokenizer {
    llm_tokenizer_ugm(const Vocab & vocab, const std::vector<char> & precompiled_charsmap) {
        if (precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t *) &precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);

            // Next xcda_blob_size bytes contain entries of XOR-compressed compact
            // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
            xcda_array = (const uint32_t *) &precompiled_charsmap[charsmap_offset];
            xcda_array_size = xcda_blob_size / sizeof(uint32_t);
            charsmap_offset += xcda_blob_size;

            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = precompiled_charsmap.size() - charsmap_offset;
        }

        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            const auto & token_data = vocab.get_token_data(id);

            if (vocab.is_normal(id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (vocab.is_normal(id) ||
                vocab.is_user_defined(id) ||
                vocab.is_unused(id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (vocab.is_user_defined(id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }

    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";

    const char * prefix_replacements = NULL;
    size_t prefix_replacements_size = 0;

    const uint32_t * xcda_array = NULL;
    size_t xcda_array_size = 0;

    struct naive_trie user_defined_token_matcher;

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;

    float unknown_token_score_penalty = 10.0;
    float unknown_token_score;

    struct naive_trie token_matcher;
};

struct llm_tokenizer_ugm_session {
    llm_tokenizer_ugm_session(const Vocab & vocab, const llm_tokenizer_ugm & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
    void tokenize(const std::string & text, std::vector<token_t> & output) {
        // get current size of output (for reversal later)
        size_t output_size = output.size();

        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }

        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.token_unk(), 0, -DBL_MAX});
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.token_unk(), 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization & current_best = tokenization_results[input_offset];
            const struct naive_trie * node = tokenizer.token_matcher.traverse(normalized[prefix_offset++]);

            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    token_t token_id = node->value;
                    const auto & token_data = vocab.get_token_data(token_id);

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = vocab.is_user_defined(token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if (!single_codepoint_token_found) {
                const double challenger_score = current_best.score_sum + tokenizer.unknown_token_score;
                prefix_offset = input_offset + n_utf8_code_units;
                struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    struct best_tokenization challenger = { vocab.token_unk(), input_offset, challenger_score };
                    current_champ = challenger;
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        bool is_prev_unknown = false;
        for (struct best_tokenization & tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
            bool is_unknown = tokenization.token_id == vocab.token_unk();
            if (!(is_prev_unknown && is_unknown)) {
                output.push_back(tokenization.token_id);
            }
            if (tokenization.input_offset == 0) {
                break;
            }
            is_prev_unknown = is_unknown;
        }

        // reverse the output since we added tokens starting from the end of the input
        std::reverse(output.begin() + output_size, output.end());
    }

private:

    // helper structure for returning normalization results
    struct normalization_result {
        const char * normalized;
        size_t normalized_len;
        size_t consumed_input;
    };

    void normalize(const std::string& input, std::string * normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.get_escape_whitespaces() ? tokenizer.escaped_space : " ";

        const bool shall_prepend_space = !vocab.get_treat_whitespace_as_suffix() && vocab.get_add_space_prefix();
        const bool shall_append_space  =  vocab.get_treat_whitespace_as_suffix() && vocab.get_add_space_prefix();
        const bool shall_merge_spaces  =  vocab.get_remove_extra_whitespaces();

        bool is_space_prepended = false;
        bool processing_non_ws = false;

        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                } else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }

            input_offset += norm_res.consumed_input;
        }

        if (shall_append_space) {
            normalized->append(space);
        }
    }

    /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Each bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
    struct xcda_array_view {
    public:
        xcda_array_view(const uint32_t * xcda_array, size_t xcda_array_size) : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {
        }
        uint32_t get_base(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
        }
        uint32_t get_lcheck(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) | 0xff);
        }
        bool get_leaf(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 8) & 1;
        }
        uint32_t get_value(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) - 1);
        }
    private:
        uint32_t get_node(size_t index) {
            if (index >= xcda_array_size) {
                throw std::runtime_error("Index out of array bounds in XCDA array!");
            }
            return xcda_array[index];
        }
        const uint32_t * xcda_array;
        size_t xcda_array_size;
    };

    // this structure stores the best tokenization so far at input_offset
    struct best_tokenization {
        token_t token_id;
        size_t input_offset;
        double score_sum;
    };

    struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match =
           tokenizer.user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }

        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (tokenizer.xcda_array_size > 0) {
            struct xcda_array_view xcda_view(tokenizer.xcda_array, tokenizer.xcda_array_size);

            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;
            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);
            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
        }

        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= tokenizer.prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char * prefix_replacement = &(tokenizer.prefix_replacements)[longest_prefix_offset];
            size_t max_len = tokenizer.prefix_replacements_size - longest_prefix_offset;
            size_t repl_len = 0;
            while (repl_len < max_len && prefix_replacement[repl_len] != '\0') {
                repl_len++;
            }
            if (repl_len == max_len) {
                throw std::runtime_error("Unterminated string in precompiled charsmap!");
            }
            return { prefix_replacement, repl_len, longest_prefix_length };
        }

        // check if the input prefix contains a valid sequence of UTF-8 code units
        try {
            // if yes, return this sequence unmodified
            size_t prefix_offset = input_offset;
            unicode_cpt_from_utf8(input, prefix_offset);
            return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
        } catch (std::invalid_argument & /*ex*/) {
            // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
            return { "\xEF\xBF\xBD", 3, 1 };
        }
    }

    const Vocab & vocab;
    const llm_tokenizer_ugm & tokenizer;
};

//
// RWKV tokenizer
//

static std::vector<uint8_t> iian_unescape_rwkv_token(const std::string & escaped) {
    std::vector<uint8_t> output;
    output.reserve(escaped.size());

    // Parser state
    bool escaping = false;
    uint8_t hex_remaining = 0;
    uint8_t hex_acc = 0;

    // Step through characters, performing parsing
    for (const char & c : escaped) {
        // If we're parsing a hex code, interpret the next character
        if (hex_remaining != 0) {
            uint8_t value = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            hex_acc = (hex_acc << 4) + value;

            hex_remaining -= 1;
            if (hex_remaining == 0) {
                output.push_back(hex_acc);
                hex_acc = 0;
            }

            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            if (c == 't') {
                output.push_back('\t');
            } else if (c == 'n') {
                output.push_back('\n');
            } else if (c == 'r') {
                output.push_back('\r');
            } else if (c == 'x') {
                hex_remaining = 2;
            } else {
                output.push_back(c);
            }

            escaping = false;
            continue;
        }

        if (c == '\\') {
            escaping = true;
            continue;
        }

        output.push_back(c);
    }

    return output;
}

struct llm_tokenizer_rwkv : llm_tokenizer {
    llm_tokenizer_rwkv(const Vocab & vocab) {
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.

        // build trie
        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            const auto & data = vocab.get_token_data(id);
            const auto text = iian_unescape_rwkv_token(data.text);
            token_matcher.insert((const char *) text.data(), text.size(), id);
        }
    }

    struct naive_trie token_matcher;
};

struct llm_tokenizer_rwkv_session {
    llm_tokenizer_rwkv_session(const Vocab & vocab, const llm_tokenizer_rwkv & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<token_t> & output) {
        uint32_t position = 0;
        while (position < text.size()) {
            const struct naive_trie * node = tokenizer.token_matcher.traverse(text[position]);
            if (node == NULL) {
                // no matching token found, add unknown token
                output.push_back(vocab.token_unk());
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            uint32_t token_id = 0;
            uint32_t token_length = 0;
            while (node != NULL) {
                if (node->has_value) {
                    token_id = node->value;
                    token_length = position + 1;
                }
                if (position + 1 >= text.size()) {
                    break;
                }
                node = node->traverse(text[++position]);
            }

            // add the longest matching token
            output.push_back(token_id);
            position = token_length;
        }
    }

private:
    const Vocab & vocab;
    const llm_tokenizer_rwkv & tokenizer;
};

struct llm_tokenizer_plamo2 : llm_tokenizer {
    llm_tokenizer_plamo2(const Vocab & vocab) {
        build(vocab);
    }

    void build(const Vocab & vocab) {
        // Reset internal structures
        tokens_.clear();
        bytes_.assign(256, 0);
        to_suffix_id_.clear();
        table_.clear();

        // Build token list and byte mapping
        std::unordered_map<std::string, float> suffix_to_score;
        std::unordered_map<std::string, token_t> token_to_id;

        for (size_t token_id = 0; token_id < vocab.n_tokens(); ++token_id) {
            const auto & entry = vocab.get_token_data(token_id);
            tokens_.push_back(entry.text);
            token_to_id[entry.text] = static_cast<token_t>(token_id);

            // Handle byte tokens
            if (vocab.is_byte(token_id)) {
                if (entry.text.length() == 6 && entry.text.substr(0, 3) == "<0x" && entry.text.back() == '>') {
                    std::string hex_str = entry.text.substr(3, 2);
                    if (std::isxdigit(static_cast<unsigned char>(hex_str[0])) && std::isxdigit(static_cast<unsigned char>(hex_str[1]))) {
                        int byte_val = std::stoi(hex_str, nullptr, 16);
                        bytes_[byte_val] = static_cast<token_t>(token_id);
                    }
                }
                continue;
            }

            // Add token and all its suffixes to suffix_to_score
            suffix_to_score[entry.text] = entry.score;

            // Extract suffixes character by character (UTF-8 aware)
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(entry.text);
            for (size_t i = 1; i < cpts.size(); ++i) {
                std::string suffix;
                for (size_t j = i; j < cpts.size(); ++j) {
                    suffix += unicode_cpt_to_utf8(cpts[j]);
                }
                if (suffix_to_score.find(suffix) == suffix_to_score.end()) {
                    suffix_to_score[suffix] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        // Check that all byte tokens are set
        for (int i = 0; i < 256; ++i) {
            if (bytes_[i] == 0) {
                throw std::runtime_error("Byte token for <0x" + std::to_string(i) + "> is not set");
            }
        }

        // Build suffix list in lexicographical order of reversed strings
        std::vector<std::string> suffixes;
        suffixes.reserve(suffix_to_score.size() + 1);
        for (const auto & pair : suffix_to_score) {
            suffixes.push_back(pair.first);
        }
        suffixes.push_back("");  // Empty suffix

        std::sort(suffixes.begin(), suffixes.end(), [](const std::string & a, const std::string & b) {
            std::string rev_a(a.rbegin(), a.rend());
            std::string rev_b(b.rbegin(), b.rend());
            return rev_a < rev_b;
        });

        // Build suffix_to_id and to_suffix_id_
        std::unordered_map<std::string, int32_t> suffix_to_id;
        int32_t num_pieces = 0;

        for (const auto & suffix : suffixes) {
            suffix_to_id[suffix] = num_pieces;
            if (!suffix.empty()) {
                std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);

                std::string remaining;
                for (size_t i = 1; i < cpts.size(); ++i) {
                    remaining += unicode_cpt_to_utf8(cpts[i]);
                }

                int64_t piece_code = (static_cast<int64_t>(cpts[0]) << 32) | suffix_to_id[remaining];
                to_suffix_id_[piece_code] = num_pieces;

                // Count number of pieces for this suffix
                int32_t pieces_for_suffix = 1; // sentinel row
                for (int32_t piece_length = static_cast<int32_t>(cpts.size()); piece_length > 0; --piece_length) {
                    std::string piece;
                    for (int32_t i = 0; i < piece_length; ++i) {
                        piece += unicode_cpt_to_utf8(cpts[i]);
                    }
                    if (suffix_to_score.find(piece) != suffix_to_score.end()) {
                        pieces_for_suffix++;
                    }
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces++;  // Empty suffix contributes one piece (sentinel row)
            }
        }

        // Build flattened table
        table_.resize(num_pieces, std::vector<int32_t>(4, 0));
        int32_t table_idx = 0;

        for (const auto & suffix : suffixes) {
            // Add all prefixes of the suffix to the table (in decreasing order of length)
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);
            for (int32_t piece_length = static_cast<int32_t>(cpts.size()); piece_length > 0; --piece_length) {
                std::string piece;
                for (int32_t i = 0; i < piece_length; ++i) {
                    piece += unicode_cpt_to_utf8(cpts[i]);
                }

                auto score_it = suffix_to_score.find(piece);
                if (score_it == suffix_to_score.end()) {
                    continue;
                }

                table_[table_idx][TABLE_PIECE_LENGTH] = piece_length;
                auto token_it = token_to_id.find(piece);
                table_[table_idx][TABLE_TOKEN_ID] = (token_it != token_to_id.end()) ? token_it->second : -1;

                float score = score_it->second;
                table_[table_idx][TABLE_SCORE] = std::isfinite(score) ?
                    static_cast<int32_t>(std::round(score * 1e4)) : INVALID_SCORE;
                table_[table_idx][TABLE_PIECE_ID] = suffix_to_id[piece];

                table_idx++;
            }

            // Add sentinel row
            table_[table_idx][TABLE_PIECE_LENGTH] = 1;
            table_[table_idx][TABLE_TOKEN_ID] = -1;
            table_[table_idx][TABLE_SCORE] = UNKNOWN_SCORE;
            table_idx++;
        }
    }

    std::vector<token_t> encode(const std::string & text) const {
        std::vector<uint32_t> unicode_data = unicode_cpts_from_utf8(text);
        // Skip the first code point if it is a BOM (Byte Order Mark)
        if (!unicode_data.empty() && unicode_data[0] == 0xFEFF) {
            unicode_data.erase(unicode_data.begin());
        }

        if (unicode_data.empty()) {
            return {};
        }

        const size_t data_len = unicode_data.size();

        // Initialize scores array (dynamic programming)
        std::vector<int64_t> scores(data_len + 1, static_cast<int64_t>(1) << 60);
        scores[data_len] = 0;

        // Path array to track best tokenization
        std::vector<std::vector<int32_t>> path(data_len + 1, std::vector<int32_t>(3, 0));

        int32_t suffix_id = 0;

        // Process from end to beginning
        for (int i = static_cast<int>(data_len) - 1; i >= 0; --i) {
            uint32_t c = unicode_data[i];

            // Find next suffix ID
            for (size_t p = suffix_id; p < table_.size(); ++p) {
                int64_t piece_code = (static_cast<int64_t>(c) << 32) | table_[p][TABLE_PIECE_ID];
                auto it = to_suffix_id_.find(piece_code);
                suffix_id = (it != to_suffix_id_.end()) ? it->second : 0;

                if (suffix_id > 0 || table_[p][TABLE_SCORE] == UNKNOWN_SCORE) {
                    break;
                }
            }

            // Update best path
            for (size_t p = suffix_id; p < table_.size(); ++p) {
                int32_t score = table_[p][TABLE_SCORE];
                if (score > INVALID_SCORE) {
                    int32_t piece_length = table_[p][TABLE_PIECE_LENGTH];
                    int64_t s = scores[i + piece_length] - score;

                    if (s < scores[i]) {
                        scores[i] = s;
                        path[i][PATH_TOKEN_LENGTH] = piece_length;
                        path[i][PATH_TOKEN_ID] = table_[p][TABLE_TOKEN_ID];
                        path[i][PATH_NUM_TOKENS] = path[i + piece_length][PATH_NUM_TOKENS] + 1;

                        if (score == UNKNOWN_SCORE) {
                            // Add UTF-8 byte count
                            path[i][PATH_NUM_TOKENS] += (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
                        }
                    }
                }

                if (score == UNKNOWN_SCORE) {
                    break;
                }
            }
        }

        // Decode the best path
        std::vector<token_t> token_ids;
        token_ids.reserve(path[0][PATH_NUM_TOKENS]);

        int pos = 0;
        while (pos < static_cast<int>(data_len)) {
            if (path[pos][PATH_TOKEN_ID] >= 0) {
                token_ids.push_back(path[pos][PATH_TOKEN_ID]);
            } else {
                // Fall back to byte tokens
                uint32_t c = unicode_data[pos];
                int s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);

                for (int i = 0; i < s; ++i) {
                    uint8_t b;
                    if (s == 1) {
                        b = c;
                    } else {
                        if (i == 0) {
                            b = (0xF00 >> s) & 0xFF;
                        } else {
                            b = 0x80;
                        }
                    }
                    token_ids.push_back(bytes_[b | ((c >> ((s - i - 1) * 6)) & 0x3F)]);
                }
            }

            assert(path[pos][PATH_TOKEN_LENGTH] > 0);
            pos += path[pos][PATH_TOKEN_LENGTH];
        }

        return token_ids;
    }
private:
    // Constants for table structure
    static constexpr int32_t TABLE_PIECE_LENGTH = 0;
    static constexpr int32_t TABLE_TOKEN_ID     = 1;
    static constexpr int32_t TABLE_SCORE        = 2;
    static constexpr int32_t TABLE_PIECE_ID     = 3;

    // Constants for path array
    static constexpr int32_t PATH_TOKEN_LENGTH  = 0;
    static constexpr int32_t PATH_TOKEN_ID      = 1;
    static constexpr int32_t PATH_NUM_TOKENS    = 2;

    // Score constants
    static constexpr int32_t INVALID_SCORE = -20000000;
    static constexpr int32_t UNKNOWN_SCORE = -10000000;

    // List of tokens in the vocabulary
    std::vector<std::string> tokens_;

    // Mapping from byte code point to token ID (for byte fallback)
    std::vector<token_t> bytes_;

    // Mapping from piece code to suffix ID
    std::unordered_map<int64_t, int32_t> to_suffix_id_;

    // Flattened table representing the Trie structure
    // Each row contains: [piece_length, token_id, score, piece_id]
    std::vector<std::vector<int32_t>> table_;
};

struct llm_tokenizer_plamo2_session {
    llm_tokenizer_plamo2_session(const llm_tokenizer_plamo2 & tokenizer) : tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<token_t> & output) {
        std::vector<token_t> tokens = tokenizer.encode(text);
        output.insert(output.end(), tokens.begin(), tokens.end());
    }

private:
    const llm_tokenizer_plamo2 & tokenizer;
};

// reserved suffix (U+E000) that keeps DNA k-mers distinct from identical
// base-vocab BPE tokens (e.g. CCCCCC) in token_to_id; erased from id_to_token
// text at load
static const std::string dna_kmer_marker = "\xee\x80\x80";

struct llm_tokenizer_hybriddna_session : llm_tokenizer_bpe_session {
    llm_tokenizer_hybriddna_session(const Vocab & vocab, const llm_tokenizer_bpe & tokenizer) : llm_tokenizer_bpe_session{vocab, tokenizer}, vocab{vocab} {}

    void tokenize(const std::string & text, std::vector<token_t> & output) override {
        static const std::string open_tag  = "<dna>";
        static const std::string close_tag = "</dna>";

        const auto dna_begin_id = vocab.text_to_token(open_tag);
        const auto dna_end_id   = vocab.text_to_token(close_tag);
        const auto dna_oov_id   = vocab.text_to_token("<oov>");

        // Fall back to plain BPE if the DNA pieces aren't in the vocab.
        if (dna_begin_id == TOKEN_NULL || dna_end_id == TOKEN_NULL || dna_oov_id == TOKEN_NULL) {
            llm_tokenizer_bpe_session::tokenize(text, output);
            return;
        }

        const size_t k = 6;
        size_t pos = 0;

        while (pos < text.size()) {
            const size_t start = text.find(open_tag, pos);
            if (start == std::string::npos) {
                if (pos < text.size()) {
                    llm_tokenizer_bpe_session::tokenize(text.substr(pos), output);
                }
                break;
            }
            if (start > pos) {
                llm_tokenizer_bpe_session::tokenize(text.substr(pos, start - pos), output);
            }
            output.push_back(dna_begin_id);

            const size_t content_start = start + open_tag.size();
            const size_t end           = text.find(close_tag, content_start);
            const size_t content_end   = (end == std::string::npos) ? text.size() : end;

            emit_dna_kmers(text.substr(content_start, content_end - content_start), k, dna_oov_id, output);

            if (end == std::string::npos) {
                break;
            }
            output.push_back(dna_end_id);
            pos = end + close_tag.size();
        }
    }

private:
    void emit_dna_kmers(const std::string & raw, size_t k, token_t oov_id, std::vector<token_t> & output) {
        std::string seq = raw;
        for (char & c : seq) {
            if (c >= 'a' && c <= 'z') {
                c = char(c - 32);
            }
        }

        // k-mers carry the reserved marker suffix; a non-ACGT k-mer simply
        // isn't in the vocab and falls back to <oov>
        auto kmer_token = [&](const std::string & kmer) {
            const auto tok = vocab.text_to_token(kmer + dna_kmer_marker);
            return tok != TOKEN_NULL ? tok : oov_id;
        };

        size_t i = 0;
        for (; i + k <= seq.size(); i += k) {
            output.push_back(kmer_token(seq.substr(i, k)));
        }
        if (i < seq.size()) {
            std::string kmer = seq.substr(i);
            kmer.append(k - kmer.size(), 'A');
            output.push_back(kmer_token(kmer));
        }
    }

    const Vocab & vocab;
};

struct llm_tokenizer_whitespace_session : llm_tokenizer_bpe_session {
    llm_tokenizer_whitespace_session(const Vocab & vocab, const llm_tokenizer_bpe & tokenizer) : llm_tokenizer_bpe_session{vocab, tokenizer}, vocab{vocab} {}

    void tokenize(const std::string & text, std::vector<token_t> & output) override {
        const bool lowercase = vocab.get_normalizer_opts().lowercase;

        std::string segment;
        auto flush = [&]() {
            if (!segment.empty()) {
                llm_tokenizer_bpe_session::tokenize(segment, output);
                segment.clear();
            }
        };

        for (uint32_t cpt : unicode_cpts_from_utf8(text)) {
            // drop whitespace
            if (unicode_cpt_flags_from_cpt(cpt).is_whitespace) {
                flush();
            } else {
                segment += unicode_cpt_to_utf8(lowercase ? unicode_tolower(cpt) : cpt);
            }
        }
        flush();
    }

private:
    const Vocab & vocab;
};

//
// impl
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(token_t _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((token_t) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT(_offset >= 0);
            GGML_ASSERT(_length >= 1);
            GGML_ASSERT(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const token_t token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

struct Vocab::impl {
    uint32_t n_token_types = 0; // for BERT-style token types

    std::string tokenizer_model;
    std::string tokenizer_pre;

    VocabType     type     = VocabType::SPM;
    VocabPreType pre_type = VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    // default LLaMA special tokens
    // TODO: should we set all of these to TOKEN_NULL?
    token_t special_bos_id  = 1;
    token_t special_eos_id  = 2;
    token_t special_eot_id  = TOKEN_NULL;
    token_t special_eom_id  = TOKEN_NULL;
    token_t special_unk_id  = 0;
    token_t special_sep_id  = TOKEN_NULL;
    token_t special_pad_id  = TOKEN_NULL;
    token_t special_mask_id = TOKEN_NULL;

    token_t linefeed_id = 13;

    // fim tokens
    token_t special_fim_pre_id = TOKEN_NULL;
    token_t special_fim_suf_id = TOKEN_NULL;
    token_t special_fim_mid_id = TOKEN_NULL;
    token_t special_fim_pad_id = TOKEN_NULL;
    token_t special_fim_rep_id = TOKEN_NULL; // repo
    token_t special_fim_sep_id = TOKEN_NULL; // file separator

    // tokenizer flags
    bool add_space_prefix           = false;
    bool add_bos                    = false;
    bool add_eos                    = false;
    bool add_sep                    = false;
    bool ignore_merges              = false;
    bool clean_spaces               = false;  // clean_up_tokenization_spaces
    bool remove_extra_whitespaces   = false;
    bool escape_whitespaces         = true;
    bool treat_whitespace_as_suffix = false;

    // BertNormalizer options
    Vocab::normalizer_options normalizer_opts;

    std::unordered_map<std::string, token_t> token_to_id;
    std::vector<token_data>                      id_to_token;

    std::vector<token_t> cache_special_tokens;
    std::vector<std::string> cache_token_to_piece; // llama_token_to_piece(special = true);
    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string> & p) const {
            return std::hash<std::string>{}(p.first) ^  //create some hash for pair
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;

    // set of all tokens that cause "end of generation"
    std::set<token_t> special_eog_ids;

    std::vector<token_t> suppress_tokens;

    std::unique_ptr<llm_tokenizer> tokenizer;

    std::vector<char> precompiled_charsmap;

    // "tokenizer.chat_template" (key "") and "tokenizer.chat_template.<variant>" (key "<variant>")
    std::map<std::string, std::string> chat_templates;

    impl(const Vocab & vocab) : vocab(vocab) {
    }

    ~impl() = default;

    void load(const gguf_context * ctx);

    VocabType get_type() const;

    std::string type_name() const;

    bool is_normal      (token_t id) const;
    bool is_unknown     (token_t id) const;
    bool is_control     (token_t id) const;
    bool is_byte        (token_t id) const;
    bool is_user_defined(token_t id) const;
    bool is_unused      (token_t id) const;
    bool is_eog         (token_t id) const;

    uint8_t token_to_byte(token_t id) const;

    TokenAttr token_get_attr(token_t id) const;

    void init_tokenizer(VocabType type);

    void tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const;

    std::string token_to_piece_for_cache(
                  token_t   token,
                         bool   special) const;


    std::vector<token_t> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  token_t * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

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

    void print_info() const;

private:
    const Vocab & vocab;
};

void Vocab::impl::load(const gguf_context * ctx) {
    // chat templates (read up-front: they are independent of the vocab type, even "no_vocab" models may carry one)
    {
        const std::string prefix = KV_TOKENIZER_CHAT_TEMPLATE;
        const int64_t n_kv = gguf_get_n_kv(ctx);
        for (int64_t i = 0; i < n_kv; ++i) {
            const std::string key = gguf_get_key(ctx, i);
            if (key.compare(0, prefix.size(), prefix) != 0 || gguf_get_kv_type(ctx, i) != GGUF_TYPE_STRING) {
                continue;
            }
            if (key.size() == prefix.size()) {
                chat_templates[""] = gguf_get_val_str(ctx, i);
            } else if (key[prefix.size()] == '.' && key.size() > prefix.size() + 1) {
                chat_templates[key.substr(prefix.size() + 1)] = gguf_get_val_str(ctx, i);
            }
        }
    }


    // determine vocab type
    {
        gguf_get_str(ctx, KV_TOKENIZER_MODEL, tokenizer_model, /*required=*/true);
        gguf_get_str(ctx, KV_TOKENIZER_PRE,   tokenizer_pre);

        gguf_get_u32(ctx, KV_TOKENIZER_TOKEN_TYPE_COUNT, n_token_types);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
            type = VocabType::NONE;

            // default special tokens
            special_bos_id  = TOKEN_NULL;
            special_eos_id  = TOKEN_NULL;
            special_unk_id  = TOKEN_NULL;
            special_sep_id  = TOKEN_NULL;
            special_pad_id  = TOKEN_NULL;
            special_mask_id = TOKEN_NULL;
            linefeed_id     = TOKEN_NULL;

            // read vocab size from metadata
            uint32_t n_tokens = 0;
            std::string general_arch;
            gguf_get_str(ctx, KV_GENERAL_ARCHITECTURE, general_arch);
            if (!general_arch.empty() && gguf_get_u32(ctx, (general_arch + ".vocab_size").c_str(), n_tokens)) {
                IIAN_TOK_LOG_WARN("%s: adding %u dummy tokens\n", __func__, n_tokens);
                id_to_token.resize(n_tokens);
            }

            return;
        }

        if (tokenizer_model == "llama") {
            type = VocabType::SPM;

            // default special tokens
            special_bos_id  = 1;
            special_eos_id  = 2;
            special_unk_id  = 0;
            special_sep_id  = TOKEN_NULL;
            special_pad_id  = TOKEN_NULL;
            special_mask_id = TOKEN_NULL;
        } else if (tokenizer_model == "bert") {
            type = VocabType::WPM;

            // default special tokens
            special_bos_id  = 101;
            special_eos_id  = TOKEN_NULL;
            special_unk_id  = 100;
            special_sep_id  = 102;
            special_pad_id  = 0;
            special_mask_id = 103;

            add_sep = true;
        } else if (tokenizer_model == "gpt2" || tokenizer_model == "hybriddna" || tokenizer_model == "whitespace") {
            type = VocabType::BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, KV_TOKENIZER_MERGES);
            // Kimi-K2 uses custom tokenization without traditional BPE merges
            const bool is_kimi_k2 = (tokenizer_pre == "kimi-k2");

            if (merges_keyidx == -1) {
                if (!is_kimi_k2) {
                    throw std::runtime_error("cannot find tokenizer merges in model file\n");
                }
                // Kimi-K2 doesn't need merges, skip
                IIAN_TOK_LOG_INFO("%s: Kimi-K2 tokenizer detected, skipping BPE merges\n", __func__);
            } else {
                if (gguf_get_kv_type(ctx, merges_keyidx) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(ctx, merges_keyidx) != GGUF_TYPE_STRING) {
                    throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_MERGES));
                }
                const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
                for (int i = 0; i < n_merges; i++) {
                    const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                    //GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                    std::string first;
                    std::string second;

                    const size_t pos = word.find(' ', 1);

                    if (pos != std::string::npos) {
                        first  = word.substr(0, pos);
                        second = word.substr(pos + 1);
                    }

                    bpe_ranks.emplace(std::make_pair(first, second), i);
                }
            }

            // default special tokens
            special_bos_id  = 11;
            special_eos_id  = 11;
            special_unk_id  = TOKEN_NULL;
            special_sep_id  = TOKEN_NULL;
            special_pad_id  = TOKEN_NULL;
            special_mask_id = TOKEN_NULL;
        } else if (tokenizer_model == "t5") {
            type = VocabType::UGM;

            // default special tokens
            special_bos_id  = TOKEN_NULL;
            special_eos_id  = 1;
            special_unk_id  = 2;
            special_sep_id  = TOKEN_NULL;
            special_pad_id  = 0;
            special_mask_id = TOKEN_NULL;

            const int precompiled_charsmap_keyidx = gguf_find_key(ctx, KV_TOKENIZER_PRECOMPILED_CHARSMAP);
            if (precompiled_charsmap_keyidx != -1) {
                if (gguf_get_kv_type(ctx, precompiled_charsmap_keyidx) != GGUF_TYPE_ARRAY) {
                    throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_PRECOMPILED_CHARSMAP));
                }
                const gguf_type pc_type = gguf_get_arr_type(ctx, precompiled_charsmap_keyidx);
                if (pc_type != GGUF_TYPE_INT8 && pc_type != GGUF_TYPE_UINT8) {
                    throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_PRECOMPILED_CHARSMAP));
                }

                const size_t n_precompiled_charsmap = gguf_get_arr_n(ctx, precompiled_charsmap_keyidx);
                const char * pc = (const char *) gguf_get_arr_data(ctx, precompiled_charsmap_keyidx);
                precompiled_charsmap.assign(pc, pc + n_precompiled_charsmap);
                if (precompiled_charsmap.size() < sizeof(uint32_t)) {
                    throw std::runtime_error("precompiled_charsmap too small for xcda_blob_size header!");
                }
                uint32_t * xcda_blob_size = (uint32_t *) &precompiled_charsmap[0];
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
                *xcda_blob_size = __builtin_bswap32(*xcda_blob_size);
#endif
                if (*xcda_blob_size + sizeof(uint32_t) >= precompiled_charsmap.size()) {
                    throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
                }
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
                // correct endianness of data in precompiled_charsmap binary blob
                size_t xcda_array_size = *xcda_blob_size / sizeof(uint32_t);
                uint32_t * xcda_array = (uint32_t *) &precompiled_charsmap[sizeof(uint32_t)];
                for (size_t i = 0; i < xcda_array_size; ++i) {
                    xcda_array[i] = __builtin_bswap32(xcda_array[i]);
                }
#endif
            }
        } else if (tokenizer_model == "rwkv") {
            type = VocabType::RWKV;

            // default special tokens
            special_bos_id = TOKEN_NULL;
            special_eos_id = TOKEN_NULL;
            special_unk_id = TOKEN_NULL;
            special_sep_id = TOKEN_NULL;
            special_pad_id = TOKEN_NULL;
        } else if (tokenizer_model == "plamo2") {
            type = VocabType::PLAMO2;

            // PLaMo-2 default special tokens (these will be overridden by model config)
            special_bos_id = 1;  // <|plamo:bos|>
            special_eos_id = 2;  // <|plamo:eos|>
            special_unk_id = 0;  // <|plamo:unk|>
            special_sep_id = TOKEN_NULL;
            special_pad_id = 3;  // <|plamo:pad|>
            special_mask_id = TOKEN_NULL;
        } else if (tokenizer_model == "gemma4") {
            type = VocabType::BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, KV_TOKENIZER_MERGES);
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }
            {
                if (gguf_get_kv_type(ctx, merges_keyidx) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(ctx, merges_keyidx) != GGUF_TYPE_STRING) {
                    throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_MERGES));
                }
                const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
                for (int i = 0; i < n_merges; i++) {
                    const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);

                    std::string first;
                    std::string second;

                    const size_t pos = word.find(' ', 1);

                    if (pos != std::string::npos) {
                        first  = word.substr(0, pos);
                        second = word.substr(pos + 1);
                    }

                    bpe_ranks.emplace(std::make_pair(first, second), i);
                }
            }

            // default special tokens (to be read from GGUF)
            special_bos_id  = TOKEN_NULL;
            special_eos_id  = TOKEN_NULL;
            special_unk_id  = TOKEN_NULL;
            special_sep_id  = TOKEN_NULL;
            special_pad_id  = TOKEN_NULL;
            special_mask_id = TOKEN_NULL;

            tokenizer_pre = "gemma4";
        } else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (type == VocabType::BPE) {
            add_space_prefix = false;
            escape_whitespaces = false;
            clean_spaces = true;
            if (tokenizer_pre.empty()) {
                IIAN_TOK_LOG_WARN("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                IIAN_TOK_LOG_WARN("%s:                                             \n", __func__);
                IIAN_TOK_LOG_WARN("%s: ************************************        \n", __func__);
                IIAN_TOK_LOG_WARN("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                IIAN_TOK_LOG_WARN("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                IIAN_TOK_LOG_WARN("%s: ************************************        \n", __func__);
                IIAN_TOK_LOG_WARN("%s:                                             \n", __func__);
                pre_type = VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                pre_type = VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "minicpm5") {
                pre_type = VOCAB_PRE_TYPE_MINICPM5;
                ignore_merges = true;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3"  ||
                    tokenizer_pre == "falcon-h1" ||
                    tokenizer_pre == "pixtral"  ||
                    tokenizer_pre == "midm-2.0" ||
                    tokenizer_pre == "lfm2"     ||
                    tokenizer_pre == "jina-v5-nano") {
                pre_type = VOCAB_PRE_TYPE_LLAMA3;
                ignore_merges = true;
                add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                pre_type = VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                pre_type = VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-v3") {
                pre_type = VOCAB_PRE_TYPE_DEEPSEEK3_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "spark2_5") {
                pre_type = VOCAB_PRE_TYPE_SPARK2_5;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "youtu") {
                pre_type = VOCAB_PRE_TYPE_YOUTU;
                clean_spaces = false;
                ignore_merges = true;
            } else if (
                    tokenizer_pre == "falcon") {
                pre_type = VOCAB_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                pre_type = VOCAB_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                pre_type = VOCAB_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "gigachat"   ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "a.x-4.0" ||
                    tokenizer_pre == "mellum"  ||
                    tokenizer_pre == "modern-bert") {
                pre_type = VOCAB_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "jais-2") {
                pre_type = VOCAB_PRE_TYPE_JAIS2;
            } else if (
                    tokenizer_pre == "gemma4" ||
                    tokenizer_pre == "granite-embed-multi-311m") {
                pre_type = VOCAB_PRE_TYPE_GEMMA4;
                escape_whitespaces = true;
            } else if (
                    tokenizer_pre == "sarvam-moe") {
                pre_type = VOCAB_PRE_TYPE_SARVAM_MOE;
                escape_whitespaces = true;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "jina-v1-en" ||
                    tokenizer_pre == "jina-v2-code" ||
                    tokenizer_pre == "roberta-bpe") {
                pre_type = VOCAB_PRE_TYPE_GPT2;
                add_sep = true;
            } else if (
                    tokenizer_pre == "whitespace") {
                pre_type = VOCAB_PRE_TYPE_WHITESPACE;
                normalizer_opts.lowercase = false;
            } else if (
                    tokenizer_pre == "refact") {
                pre_type = VOCAB_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                pre_type = VOCAB_PRE_TYPE_COMMAND_R;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "qwen2" ||
                    tokenizer_pre == "deepseek-r1-qwen" ||
                    tokenizer_pre == "kormo" ||
                    tokenizer_pre == "f2llmv2") {
                pre_type = VOCAB_PRE_TYPE_QWEN2;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "qwen35") {
                pre_type = VOCAB_PRE_TYPE_QWEN35;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                pre_type = VOCAB_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                pre_type = VOCAB_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                pre_type = VOCAB_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                pre_type = VOCAB_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                pre_type = VOCAB_PRE_TYPE_PORO;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "glm4" ||
                tokenizer_pre == "chatglm-bpe") {
                pre_type = VOCAB_PRE_TYPE_CHATGLM4;
                special_bos_id = TOKEN_NULL;
            } else if (
                tokenizer_pre == "viking") {
                pre_type = VOCAB_PRE_TYPE_VIKING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                pre_type = VOCAB_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                pre_type = VOCAB_PRE_TYPE_TEKKEN;
                clean_spaces = false;
                ignore_merges = true;
                add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                pre_type = VOCAB_PRE_TYPE_SMOLLM;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                pre_type = VOCAB_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                pre_type = VOCAB_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                pre_type = VOCAB_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                pre_type = VOCAB_PRE_TYPE_EXAONE;
            } else if (
                tokenizer_pre == "exaone4") {
                pre_type = VOCAB_PRE_TYPE_GPT2;
            } else if (
                tokenizer_pre == "exaone-moe") {
                pre_type = VOCAB_PRE_TYPE_EXAONE_MOE;
            } else if (
                tokenizer_pre == "chameleon") {
                pre_type = VOCAB_PRE_TYPE_CHAMELEON;
                add_bos = true;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minerva-7b") {
                pre_type = VOCAB_PRE_TYPE_MINERVA;
            } else if (
                tokenizer_pre == "megrez") {
                pre_type = VOCAB_PRE_TYPE_QWEN2;
            } else if (
                tokenizer_pre == "gpt-4o" ||
                tokenizer_pre == "llama4" ||
                tokenizer_pre == "kanana2" ||
                tokenizer_pre == "talkie") {
                pre_type = VOCAB_PRE_TYPE_GPT4O;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "granite-embed-multi-97m") {
                pre_type = VOCAB_PRE_TYPE_GRANITE_EMB_MULTI;
                clean_spaces = false;
                ignore_merges = true;
            } else if (
                tokenizer_pre == "tiny_aya" ||
                tokenizer_pre == "cohere2moe") {
                pre_type = VOCAB_PRE_TYPE_TINY_AYA;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "superbpe") {
                pre_type = VOCAB_PRE_TYPE_SUPERBPE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "trillion") {
                pre_type = VOCAB_PRE_TYPE_TRILLION;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "granite-docling") {
                pre_type = VOCAB_PRE_TYPE_GRANITE_DOCLING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "bailingmoe" ||
                tokenizer_pre == "bailingmoe2" ||
                tokenizer_pre == "llada-moe") {
                pre_type = VOCAB_PRE_TYPE_BAILINGMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "seed-coder") {
                pre_type = VOCAB_PRE_TYPE_SEED_CODER;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan") {
                pre_type = VOCAB_PRE_TYPE_HUNYUAN;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan-dense") {
                pre_type = VOCAB_PRE_TYPE_HUNYUAN_DENSE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hy_v4") {
                pre_type = VOCAB_PRE_TYPE_HY_V4;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "joyai-llm") {
                pre_type = VOCAB_PRE_TYPE_JOYAI_LLM;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "kimi-k2") {
                pre_type = VOCAB_PRE_TYPE_KIMI_K2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "grok-2") {
                pre_type = VOCAB_PRE_TYPE_GROK_2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "afmoe") {
                pre_type = VOCAB_PRE_TYPE_AFMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "laguna") {
                pre_type = VOCAB_PRE_TYPE_LAGUNA;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minimax-m2") {
                pre_type = VOCAB_PRE_TYPE_MINIMAX_M2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "solar-open") {
                pre_type = VOCAB_PRE_TYPE_SOLAR_OPEN;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "mellum2") {
                pre_type = VOCAB_PRE_TYPE_MELLUM2;
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
        } else if (type == VocabType::SPM) {
            pre_type = VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = true;
            clean_spaces = false;
            add_bos = true;
            add_eos = false;
        } else if (type == VocabType::WPM) {
            pre_type = VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = true;
            add_bos = true;
            add_eos = false;
            add_sep = true;
        } else if (type == VocabType::UGM) {
            pre_type = VOCAB_PRE_TYPE_DEFAULT;
            add_bos = false;
            add_eos = true;
        } else if (type == VocabType::RWKV) {
            pre_type = VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = false;
            add_bos = false;
            add_eos = false;
        } else {
            pre_type = VOCAB_PRE_TYPE_DEFAULT;
        }

        gguf_get_bool(ctx, KV_TOKENIZER_ADD_PREFIX,      add_space_prefix);
        gguf_get_bool(ctx, KV_TOKENIZER_REMOVE_EXTRA_WS, remove_extra_whitespaces);
    }

    const int token_idx = gguf_find_key(ctx, KV_TOKENIZER_LIST);
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    if (gguf_get_kv_type(ctx, token_idx) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, token_idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_LIST));
    }

    const uint32_t n_tokens = gguf_get_arr_n(ctx, token_idx);

    const float * scores = nullptr;
    const int * iscores = nullptr;
    const int score_idx = gguf_find_key(ctx, KV_TOKENIZER_SCORES);
    if (score_idx != -1) {
        const gguf_type kv_type = gguf_get_kv_type(ctx, score_idx);
        const gguf_type arr_type = kv_type == GGUF_TYPE_ARRAY ? gguf_get_arr_type(ctx, score_idx) : GGUF_TYPE_COUNT;
        if (arr_type != GGUF_TYPE_INT32 &&
            arr_type != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_SCORES));
        }
        const uint32_t n_scores = gguf_get_arr_n(ctx, score_idx);
        if (n_scores < n_tokens) {
            throw std::runtime_error("Index out of array bounds for scores (" + std::to_string(n_scores) + " < " + std::to_string(n_tokens) + ")\n");
        }
        if (arr_type == GGUF_TYPE_INT32) {
            iscores = (const int *) gguf_get_arr_data(ctx, score_idx);
        } else {
            scores = (const float * ) gguf_get_arr_data(ctx, score_idx);
        }
    }

    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, KV_TOKENIZER_TOKEN_TYPE);
    if (toktype_idx != -1) {
        if (gguf_get_kv_type(ctx, toktype_idx) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx, toktype_idx) != GGUF_TYPE_INT32) {
            throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_TOKEN_TYPE));
        }
        const uint32_t n_toktypes = gguf_get_arr_n(ctx, toktype_idx);
        if (n_toktypes < n_tokens) {
            throw std::runtime_error("Index out of array bounds for toktypes (" + std::to_string(n_toktypes) + " < " + std::to_string(n_tokens) + ")\n");
        }
        toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);
    }

    id_to_token.resize(n_tokens);

    for (uint32_t i = 0; i < n_tokens; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        if (word.empty()) {
            IIAN_TOK_LOG_WARN("%s: empty token at index %u\n", __func__, i);
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        token_to_id[word] = i;
        max_token_len = std::max(max_token_len, (int) word.size());

        auto & token_data = id_to_token[i];
        token_data.text  = std::move(word);
        if (scores) {
            token_data.score = scores[i];
        } else if (iscores) {
            token_data.score = static_cast<float>(iscores[i]);
        } else {
            token_data.score = 0.0f;
        }
        token_data.attr  = TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch(toktypes[i]) {
                case GGUF_TOKEN_TYPE_UNKNOWN:      token_data.attr = TOKEN_ATTR_UNKNOWN;      break;
                case GGUF_TOKEN_TYPE_UNUSED:       token_data.attr = TOKEN_ATTR_UNUSED;       break;
                case GGUF_TOKEN_TYPE_NORMAL:       token_data.attr = TOKEN_ATTR_NORMAL;       break;
                case GGUF_TOKEN_TYPE_CONTROL:      token_data.attr = TOKEN_ATTR_CONTROL;      break;
                case GGUF_TOKEN_TYPE_USER_DEFINED: token_data.attr = TOKEN_ATTR_USER_DEFINED; break;
                case GGUF_TOKEN_TYPE_BYTE:         token_data.attr = TOKEN_ATTR_BYTE;         break;
                case GGUF_TOKEN_TYPE_UNDEFINED:    token_data.attr = TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    GGML_ASSERT(id_to_token.size() == token_to_id.size());

    // hybriddna: the marker suffix kept k-mer ids distinct in token_to_id; erase
    // it from id_to_token so the k-mers detokenize to the bare DNA sequence. The
    // k-mers are the block right after <oov>, so only scan from there.
    if (tokenizer_model == "hybriddna") {
        const auto idx = token_to_id.find("<oov>");
        if (idx != token_to_id.end()) {
            auto it = id_to_token.begin() + idx->second + 1;
            for (; it != id_to_token.end(); ++it) {
                std::string & text = it->text;
                if (text.size() > dna_kmer_marker.size()
                        && text.compare(text.size() - dna_kmer_marker.size(), dna_kmer_marker.size(), dna_kmer_marker) == 0) {
                    text.erase(text.size() - dna_kmer_marker.size());
                }
            }
        }
    }

    init_tokenizer(type);

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (type == VocabType::SPM) {
        try {
            linefeed_id = vocab.byte_to_token('\n');
        } catch (const std::exception & e) {
            IIAN_TOK_LOG_WARN("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            linefeed_id = special_pad_id;
        }
    } else if (type == VocabType::WPM) {
        linefeed_id = special_pad_id;
    } else if (type == VocabType::RWKV) {
        const std::vector<int> ids = tokenize("\n", false);
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        linefeed_id = ids[0];
    } else {
        const std::vector<int> ids = tokenize("\n", false);

        //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        if (ids.empty()) {
            IIAN_TOK_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            linefeed_id = special_pad_id;
        } else {
            linefeed_id = ids[0];
        }
    }

    // special tokens
    {
        const std::vector<std::pair<const char *, int32_t &>> special_token_types = {
            { KV_TOKENIZER_BOS_ID,     special_bos_id     },
            { KV_TOKENIZER_EOS_ID,     special_eos_id     },
            { KV_TOKENIZER_EOT_ID,     special_eot_id     },
            { KV_TOKENIZER_EOM_ID,     special_eom_id     },
            { KV_TOKENIZER_UNK_ID,     special_unk_id     },
            { KV_TOKENIZER_SEP_ID,     special_sep_id     },
            { KV_TOKENIZER_PAD_ID,     special_pad_id     },
            { KV_TOKENIZER_MASK_ID,    special_mask_id    },
            { KV_TOKENIZER_FIM_PRE_ID, special_fim_pre_id },
            { KV_TOKENIZER_FIM_SUF_ID, special_fim_suf_id },
            { KV_TOKENIZER_FIM_MID_ID, special_fim_mid_id },
            { KV_TOKENIZER_FIM_PAD_ID, special_fim_pad_id },
            { KV_TOKENIZER_FIM_REP_ID, special_fim_rep_id },
            { KV_TOKENIZER_FIM_SEP_ID, special_fim_sep_id },

            // deprecated
            { KV_TOKENIZER_PREFIX_ID, special_fim_pre_id },
            { KV_TOKENIZER_SUFFIX_ID, special_fim_suf_id },
            { KV_TOKENIZER_MIDDLE_ID, special_fim_mid_id },
        };

        for (const auto & it : special_token_types) {
            const char * key = std::get<0>(it);
            int32_t & id = std::get<1>(it);

            if (id >= 0 && static_cast<size_t>(id) >= id_to_token.size()) {
                IIAN_TOK_LOG_WARN("%s: default special token '%s' = %d out of vocab range, disabling\n",
                    __func__, key, id);
                id = TOKEN_NULL;
            }

            uint32_t new_id;
            if (!gguf_get_u32(ctx, key, new_id)) {
                continue;
            }
            if (new_id >= id_to_token.size()) {
                IIAN_TOK_LOG_WARN("%s: bad special token: '%s' = %u, using default id %d\n",
                    __func__, key, new_id, id);
            } else {
                id = new_id;
            }
        }

        // Handle add_bos, add_eos and add_sep
        {
            bool temp = true;

            if (gguf_get_bool(ctx, KV_TOKENIZER_ADD_BOS, temp)) {
                add_bos = temp;
            }
            if (gguf_get_bool(ctx, KV_TOKENIZER_ADD_EOS, temp)) {
                add_eos = temp;
            }
            if (gguf_get_bool(ctx, KV_TOKENIZER_ADD_SEP, temp)) {
                add_sep = temp;
            }

            // workaround for Gemma 4
            // ref: https://github.com/ggml-org/llama.cpp/pull/21500
            if (pre_type == VOCAB_PRE_TYPE_GEMMA4 && !add_bos) {
                add_bos = true;

                IIAN_TOK_LOG_WARN("%s: override '%s' to 'true' for Gemma4\n", __func__, KV_TOKENIZER_ADD_BOS);
            }
        }

        // BertNormalizer options
        gguf_get_bool(ctx, KV_TOKENIZER_NORMALIZER_LOWERCASE,     normalizer_opts.lowercase);
        normalizer_opts.strip_accents = normalizer_opts.lowercase;
        gguf_get_bool(ctx, KV_TOKENIZER_NORMALIZER_STRIP_ACCENTS, normalizer_opts.strip_accents);

        // suppress tokens
        {
            const int suppress_idx = gguf_find_key(ctx, KV_TOKENIZER_SUPPRESS_TOKENS);
            if (suppress_idx != -1) {
                if (gguf_get_kv_type(ctx, suppress_idx) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(ctx, suppress_idx) != GGUF_TYPE_INT32) {
                    throw std::runtime_error(format("invalid gguf type for %s", KV_TOKENIZER_SUPPRESS_TOKENS));
                }
                const int n = gguf_get_arr_n(ctx, suppress_idx);
                const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, suppress_idx);
                // drop out-of-range ids
                suppress_tokens.reserve(n);
                for (int i = 0; i < n; ++i) {
                    const int32_t id = data[i];
                    if (id >= 0 && id < (int) id_to_token.size()) {
                        suppress_tokens.push_back(id);
                    }
                }
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text

        for (const auto & t : token_to_id) {
            auto & attr = id_to_token[t.second].attr;

            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if (special_eot_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|eot_id|>"
                        || t.first == "<|im_end|>"
                        || t.first == "<|end|>"
                        || t.first == "<end_of_turn>"
                        || t.first == "<|endoftext|>"
                        || t.first == "<|end_of_text|>" // granite
                        || t.first == "<EOT>"
                        || t.first == "_<EOT>"
                        || t.first == "[EOT]" // Kimi-K2
                        || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                        || t.first == "<end_of_utterance>" // smoldocling
                   ) {
                    special_eot_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find EOM token: "<|eom_id|>"
            if (special_eom_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|eom_id|>"
                        ) {
                    special_eom_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if (special_fim_pre_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_prefix|>"  // Qwen
                        || t.first == "<fim-prefix>"
                        || t.first == "<fim_prefix>"    // Granite
                        || t.first == "<｜fim▁begin｜>" // DeepSeek
                        || t.first == "<PRE>"
                        || t.first == "▁<PRE>"          // CodeLlama
                        || t.first == "<|code_prefix|>" // GLM-4.5
                        || t.first == "<|prefix|>"      // Falcon-H1-Tiny-Coder
                        ) {
                    special_fim_pre_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if (special_fim_suf_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_suffix|>" // Qwen
                        || t.first == "<fim-suffix>"
                        || t.first == "<fim_suffix>"   // Granite
                        || t.first == "<｜fim▁hole｜>" // DeepSeek
                        || t.first == "<SUF>"
                        || t.first == "▁<SUF>"         // CodeLlama
                        || t.first == "<|code_suffix|>" // GLM-4.5
                        || t.first == "<|suffix|>"      // Falcon-H1-Tiny-Coder
                        ) {
                    special_fim_suf_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if (special_fim_mid_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_middle|>" // Qwen
                        || t.first == "<fim-middle>"
                        || t.first == "<fim_middle>"   // Granite
                        || t.first == "<｜fim▁end｜>"  // DeepSeek
                        || t.first == "<MID>"
                        || t.first == "▁<MID>"         // CodeLlama
                        || t.first == "<|code_middle|>" // GLM-4.5
                        || t.first == "<|middle|>"      // Falcon-H1-Tiny-Coder
                        ) {
                    special_fim_mid_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if (special_fim_pad_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_pad|>" // Qwen
                        || t.first == "<fim-pad>"
                        || t.first == "<fim_pad>"   // Granite
                        || t.first == "<PAD>"
                        || t.first == "[PAD]" // Kimi-K2
                        ) {
                    special_fim_pad_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if (special_fim_rep_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_repo|>"  // Qwen
                        || t.first == "<|repo_name|>"
                        || t.first == "<fim-repo>"
                        || t.first == "<REPO>"
                        || t.first == "<reponame>"    // Granite
                        ) {
                    special_fim_rep_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if (special_fim_sep_id == TOKEN_NULL) {
                if (false
                        || t.first == "<|file_sep|>" // Qwen
                        ) {
                    special_fim_sep_id = t.second;
                    if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                        IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                    }
                }
            }
        }

        // auto-detect unused tokens: e.g. control tokens with the word "unused"
        // ideally, these tokens should be marked as unused during conversion
        {
            uint32_t n_unused = 0;

            for (const auto & t : token_to_id) {
                auto & attr = id_to_token[t.second].attr;

                if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                    continue;
                }

                if ((attr & TOKEN_ATTR_UNUSED) == 0) {
                    if (strstr(t.first.c_str(), "unused") != NULL) {
                        attr = (TokenAttr) (attr | TOKEN_ATTR_UNUSED);
                    }
                }

                if (attr & TOKEN_ATTR_UNUSED) {
                    n_unused++;
                }
            }

            IIAN_TOK_LOG_INFO("%s: %u unused tokens\n", __func__, n_unused);
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggml-org/llama.cpp/issues/9606
        special_eog_ids.clear();

        if (special_fim_pad_id != TOKEN_NULL && special_eog_ids.count(special_fim_pad_id) == 0) {
            special_eog_ids.insert(special_fim_pad_id);
        }

        if (special_fim_rep_id != TOKEN_NULL && special_eog_ids.count(special_fim_rep_id) == 0) {
            special_eog_ids.insert(special_fim_rep_id);
        }

        if (special_fim_sep_id != TOKEN_NULL && special_eog_ids.count(special_fim_sep_id) == 0) {
            special_eog_ids.insert(special_fim_sep_id);
        }

        for (const auto & t : token_to_id) {
            auto & attr = id_to_token[t.second].attr;

            if (false
                    || t.first == "<|eot_id|>"
                    || t.first == "<|im_end|>"
                    || t.first == "<|end|>"
                    || t.first == "<|return|>" // o200k_harmony
                    || t.first == "<|call|>"   // o200k_harmony
                    || t.first == "<|flush|>"  // solar-open
                    || t.first == "<|calls|>"  // solar-open
                    || t.first == "<end_of_turn>"
                    || t.first == "<|endoftext|>"
                    || t.first == "</s>"      // paddleocr
                    || t.first == "<|eom_id|>"
                    || t.first == "<EOT>"
                    || t.first == "_<EOT>"
                    || t.first == "[EOT]" // Kimi-K2
                    || t.first == "[EOS]" // Kimi-K2
                    || t.first == "<|end_of_text|>"
                    || t.first == "<end_of_utterance>" // smoldocling
                    || t.first == "<eos>"            // gemma4
                    || t.first == "<turn|>"          // gemma4
                    || t.first == "<|tool_response>" // gemma4
                    || t.first == "<｜end▁of▁sentence｜>" // deepseek-ocr
                    || t.first == "[e~[" // minimax-m2/m3
               ) {
                special_eog_ids.insert(t.second);
                if ((attr & TOKEN_ATTR_CONTROL) == 0) {
                    IIAN_TOK_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                    attr = (TokenAttr) (attr | TOKEN_ATTR_CONTROL);
                }
            } else {
                if (attr & TOKEN_ATTR_CONTROL && !(attr & TOKEN_ATTR_UNUSED)) {
                    // token is control, but not marked as EOG -> print a debug log
                    if (special_eog_ids.count(t.second) == 0) {
                        IIAN_TOK_LOG_DEBUG("%s: control token: %6d '%s' is not marked as EOG\n",
                                __func__, t.second, t.first.c_str());
                    }
                }
            }
        }

        // @ngxson : quick hack for gpt-oss, always render these tokens
        for (const auto & t : token_to_id) {
            auto & attr = id_to_token[t.second].attr;

            if (t.first == "<|channel|>" || t.first == "<|message|>" || t.first == "<|start|>" || t.first == "<|constrain|>") {
                IIAN_TOK_LOG_WARN("%s: setting token '%s' (%d) attribute to USER_DEFINED (%u), old attributes: %u\n",
                        __func__, t.first.c_str(), t.second, TOKEN_ATTR_USER_DEFINED, attr);

                attr = TOKEN_ATTR_USER_DEFINED;
            }
        }

        // sanity checks
        if (special_eos_id != TOKEN_NULL && special_eog_ids.count(special_eos_id) == 0) {
            special_eog_ids.insert(special_eos_id);
            IIAN_TOK_LOG_WARN("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eot_id != TOKEN_NULL && special_eog_ids.count(special_eot_id) == 0) {
            special_eog_ids.insert(special_eot_id);
            IIAN_TOK_LOG_WARN("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eom_id != TOKEN_NULL && special_eog_ids.count(special_eom_id) == 0) {
            special_eog_ids.insert(special_eom_id);
            IIAN_TOK_LOG_WARN("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        // TODO: workaround for o200k_harmony and solar-open tokenizer: the "<|end|>" token should not be EOG
        //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens ("<|calls|>" and "<|flush|>" for solar-open),
        //       we remove the "<|end|>" token from the EOG list
        {
            bool has_return = false;
            bool has_call   = false;
            bool has_end    = false;
            bool has_flush  = false;

            token_t end_id = TOKEN_NULL;

            IIAN_TOK_LOG_INFO("%s: printing all EOG tokens:\n", __func__);
            for (auto tid : special_eog_ids) {
                if (tid < 0 || tid >= (token_t) id_to_token.size()) {
                    IIAN_TOK_LOG_WARN("%s: EOG token id %d is out of range (vocab size %zu), skipping\n",
                            __func__, tid, id_to_token.size());
                    continue;
                }
                auto & text = id_to_token[tid].text;

                IIAN_TOK_LOG_INFO("%s:   - %d ('%s')\n", __func__, tid, text.c_str());

                if (text == "<|return|>") {
                    has_return = true;
                } else if (text == "<|call|>" || text == "<|calls|>") {
                    has_call = true;
                } else if (text == "<|flush|>") {
                    has_flush = true;
                } else if (text == "<|end|>") {
                    has_end = true;
                    end_id = tid;
                }
            }

            if ((has_return && has_call && has_end) || (has_call && has_flush && has_end)) {
                special_eog_ids.erase(end_id);

                auto & attr = id_to_token[end_id].attr;
                attr = TOKEN_ATTR_USER_DEFINED;

                IIAN_TOK_LOG_WARN("%s: special_eog_ids contains both '<|return|>' and '<|call|>', or '<|calls|>' and '<|flush|>' tokens, removing '<|end|>' token from EOG list\n", __func__);
            }
        }

        // workaround for gemma4 and paddleocr: do not include </s> as an eog token
        {
            bool has_tool_response = false;
            bool has_s = false;

            token_t s_id = TOKEN_NULL;

            for (auto tid : special_eog_ids) {
                if (tid < 0 || tid >= (token_t) id_to_token.size()) {
                    continue;
                }
                const auto & text = id_to_token[tid].text;
                if (text == "<|tool_response>") {
                    has_tool_response = true;
                } else if (text == "</s>") {
                    has_s = true;
                    s_id = tid;
                }
            }

            if (has_tool_response && has_s) {
                special_eog_ids.erase(s_id);

                auto & attr = id_to_token[s_id].attr;
                attr = TOKEN_ATTR_NORMAL;

                IIAN_TOK_LOG_WARN("%s: special_eog_ids contains '<|tool_response>', removing '</s>' token from EOG list\n", __func__);
            }
        }
    }

    // build special tokens cache
    {
        for (token_t id = 0; id < (token_t) n_tokens; ++id) {
            if (id_to_token[id].attr & (TOKEN_ATTR_CONTROL | TOKEN_ATTR_USER_DEFINED | TOKEN_ATTR_UNKNOWN)) {
                cache_special_tokens.push_back(id);
            }
        }

        std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
            [&] (const token_t a, const token_t b) {
                return id_to_token[a].text.size() > id_to_token[b].text.size();
            }
        );

        IIAN_TOK_LOG_INFO("%s: special tokens cache size = %u\n", __func__, (uint32_t) cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<std::string> cache(n_tokens);

        for (uint32_t id = 0; id < n_tokens; ++id) {
            cache[id] = token_to_piece_for_cache(id, true);

            size_cache += cache[id].size();
        }

        std::swap(cache_token_to_piece, cache);

        IIAN_TOK_LOG_INFO("%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the GGUF file.
    //TODO: Extract attributes from GGUF file.
    {
        auto _contains_any = [] (const std::string & str, const std::vector<std::string_view> & substrs) -> bool {
            for (const auto & substr : substrs) {
                if (str.find(substr) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };

        auto _set_tokenid_attr = [&] (const token_t id, TokenAttr attr, bool value) {
            uint32_t current = id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            id_to_token[id].attr = (TokenAttr) current;
        };

        auto _set_token_attr = [&] (const std::string & token, TokenAttr attr, bool value) {
            _set_tokenid_attr(token_to_id.at(token), attr, value);
        };

        std::string model_name;
        std::string tokenizer_pre;
        std::string general_arch;

        gguf_get_str(ctx, KV_GENERAL_NAME,  model_name);
        gguf_get_str(ctx, KV_TOKENIZER_PRE, tokenizer_pre);
        gguf_get_str(ctx, KV_GENERAL_ARCHITECTURE, general_arch);

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [] (const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer/architecture name
        if (false
                || _contains_any(tokenizer_pre, {"jina-v2-de", "jina-v2-es", "jina-v2-code"})
                || _contains_any(general_arch, {"nomic-bert-moe", "jina-bert-v3"})
           ) {
            if (token_to_id.count("<mask>") == 0) {
                IIAN_TOK_LOG_WARN("%s: Mask token is missing in vocab, please reconvert model!\n", __func__);
            } else {
                _set_token_attr("<mask>", TOKEN_ATTR_LSTRIP, true);
            }
        } else if (_contains_any(model_name, {"phi-3", "phi3"})) {
            for (auto id : cache_special_tokens) {
                _set_tokenid_attr(id, TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"</s>"}) {
                _set_token_attr(token, TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"<unk>", "<s>", "<|endoftext|>"}) {
                _set_token_attr(token, TOKEN_ATTR_RSTRIP, false);
            }
        } else if (_contains_any(model_name, {"modern-bert"})) {
            if (token_to_id.count("[MASK]") == 0 ) {
                IIAN_TOK_LOG_WARN("%s: Mask token missing in vocab!\n", __func__);
            }
            else {
                _set_token_attr("[MASK]", TOKEN_ATTR_LSTRIP, true);
            }
        }
    }
}

VocabType Vocab::impl::get_type() const {
    return type;
}

std::string Vocab::impl::type_name() const{
    switch (type) {
        case VocabType::NONE:   return "no vocab";
        case VocabType::SPM:    return "SPM";
        case VocabType::BPE:    return "BPE";
        case VocabType::WPM:    return "WPM";
        case VocabType::UGM:    return "UGM";
        case VocabType::RWKV:   return "RWKV";
        case VocabType::PLAMO2: return "PLaMo2";
        default:                      return "unknown";
    }
}

bool Vocab::impl::is_normal(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token[id].attr & TOKEN_ATTR_NORMAL;
}

bool Vocab::impl::is_unknown(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token[id].attr & TOKEN_ATTR_UNKNOWN;
}

bool Vocab::impl::is_control(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token[id].attr & TOKEN_ATTR_CONTROL;
}

bool Vocab::impl::is_byte(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token[id].attr & TOKEN_ATTR_BYTE;
}

bool Vocab::impl::is_user_defined(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token[id].attr & TOKEN_ATTR_USER_DEFINED;
}

bool Vocab::impl::is_unused(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token[id].attr & TOKEN_ATTR_UNUSED;
}

bool Vocab::impl::is_eog(token_t id) const {
    return id != TOKEN_NULL && special_eog_ids.count(id) > 0;
}

uint8_t Vocab::impl::token_to_byte(token_t id) const {
    GGML_ASSERT(get_type() != VocabType::NONE);
    GGML_ASSERT(is_byte(id));
    const auto & token_data = id_to_token.at(id);
    switch (get_type()) {
        case VocabType::SPM:
        case VocabType::UGM: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case VocabType::BPE: {
            // Gemma4 uses BPE with SPM-style byte fallback tokens (<0xXX>)
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case VocabType::WPM: {
            GGML_ABORT("fatal error");
        }
        default:
            GGML_ABORT("fatal error");
    }
}

TokenAttr Vocab::impl::token_get_attr(token_t id) const {
    GGML_ASSERT(type != VocabType::NONE);
    return id_to_token.at(id).attr;
}

void Vocab::impl::init_tokenizer(VocabType type) {
    IIAN_TOK_LOG_DEBUG("%s: initializing tokenizer for type %d\n", __func__, (int) type);

    switch (type) {
        case VocabType::SPM:
            tokenizer = std::make_unique<llm_tokenizer_spm>(vocab);
            break;
        case VocabType::BPE:
            tokenizer = std::make_unique<llm_tokenizer_bpe>(vocab);
            break;
        case VocabType::WPM:
            tokenizer = std::make_unique<llm_tokenizer_wpm>(vocab);
            break;
        case VocabType::UGM:
            tokenizer = std::make_unique<llm_tokenizer_ugm>(vocab, precompiled_charsmap);
            break;
        case VocabType::RWKV:
            tokenizer = std::make_unique<llm_tokenizer_rwkv>(vocab);
            break;
        case VocabType::PLAMO2:
            tokenizer = std::make_unique<llm_tokenizer_plamo2>(vocab);
            break;
        default:
            GGML_ABORT("unsupported vocab type");
    }
}

//
// (de-) tokenize
//

// #define PRETOKENIZERDEBUG

void Vocab::impl::tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const {
    // for each special token
    for (const token_t special_id : cache_special_tokens) {
        const auto & data = vocab.get_token_data(special_id);
        const auto & text = data.text;

        if (!parse_special && (data.attr & (TOKEN_ATTR_CONTROL | TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    //  string_view begins at pos 0 for the same reason
                    auto match = std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(text, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

#ifdef PRETOKENIZERDEBUG
                    IIAN_TOK_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + text.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        if (data.attr & TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

// NOTE: avoid ever using this except for building the token_to_piece caches
std::string Vocab::impl::token_to_piece_for_cache(token_t token, bool special) const {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

static void iian_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void iian_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

static std::string iian_decode_text(const std::string & text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

std::vector<token_t> Vocab::impl::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call Vocab::init_tokenizer() first.");

    std::vector<token_t> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer, parse_special);
    }

    switch (get_type()) {
        case VocabType::SPM:
            {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []

                bool is_prev_special = true;  // prefix with space if first token

                if (add_special && add_bos) {
                    GGML_ASSERT(special_bos_id != TOKEN_NULL);
                    output.push_back(special_bos_id);
                    is_prev_special = true;
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text;

                        // prefix with space if previous is special
                        if (add_space_prefix && is_prev_special) {
                            text = ' ';
                        }

                        text += fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        iian_escape_whitespace(text);
                        llm_tokenizer_spm_session session(vocab);
                        session.tokenize(text, output);
                        is_prev_special = false;
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                        is_prev_special = true;
                    }
                }

                if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
                    IIAN_TOK_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && add_eos) {
                    GGML_ASSERT(special_eos_id != TOKEN_NULL);
                    output.push_back(special_eos_id);
                }
            } break;
        case VocabType::BPE:
            {
                // it calls some other methods that are not exist in llm_tokenizer,
                // here just cast it to bpe tokenizer object
                const llm_tokenizer_bpe * tok_bpe = static_cast<const llm_tokenizer_bpe *>(tokenizer.get());

                std::unique_ptr<llm_tokenizer_bpe_session> session;
                if (vocab.get_tokenizer_model() == "hybriddna") {
                    session = std::make_unique<llm_tokenizer_hybriddna_session>(vocab, *tok_bpe);
                } else if (vocab.get_tokenizer_model() == "whitespace") {
                    session = std::make_unique<llm_tokenizer_whitespace_session>(vocab, *tok_bpe);
                } else {
                    session = std::make_unique<llm_tokenizer_bpe_session>(vocab, *tok_bpe);
                }

                if (add_special) {
                    session->append_bos(output);
                }
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

                        if (escape_whitespaces) {
                            iian_escape_whitespace(text);
                        }

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session->tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        session->append(fragment.token, output);
                    }
                }

                if (add_special) {
                    session->append_eos(output);
                    session->check_double_bos_eos(output);
                }
            } break;
        case VocabType::WPM:
            {
                if (add_special) {
                    GGML_ASSERT(special_bos_id != TOKEN_NULL);
                    output.push_back(special_bos_id);
                }

                llm_tokenizer_wpm_session session(vocab);

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special) {
                    GGML_ASSERT(special_sep_id != TOKEN_NULL);
                    output.push_back(special_sep_id);
                }
            } break;
        case VocabType::UGM:
            {
                if (add_special && add_bos) {
                    GGML_ASSERT(special_bos_id != TOKEN_NULL);
                    output.push_back(special_bos_id);
                }
                llm_tokenizer_ugm_session session(vocab, *static_cast<const llm_tokenizer_ugm *>(tokenizer.get()));

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);
#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
                    IIAN_TOK_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && add_eos) {
                    GGML_ASSERT(special_eos_id != TOKEN_NULL);
                    output.push_back(special_eos_id);
                }
            } break;
        case VocabType::RWKV:
            {
                llm_tokenizer_rwkv_session session(vocab, *static_cast<const llm_tokenizer_rwkv *>(tokenizer.get()));
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case VocabType::PLAMO2:
            {
                llm_tokenizer_plamo2_session session(*static_cast<const llm_tokenizer_plamo2 *>(tokenizer.get()));
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        IIAN_TOK_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case VocabType::NONE:
            GGML_ABORT("fatal error");
    }

    return output;
}

int32_t Vocab::impl::token_to_piece(token_t token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    // ref: https://github.com/ggml-org/llama.cpp/pull/7587#discussion_r1620983843
    static const int attr_special = TOKEN_ATTR_UNKNOWN | TOKEN_ATTR_CONTROL;
    const TokenAttr attr = token_get_attr(token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=] (const char * token, size_t size) -> int32_t {
        if (size >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            GGML_ABORT("invalid token size: %zu exceeds int32_t limit", size);
        }

        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t) size;
        }
        memcpy(buf, token, size);
        return (int32_t) size;
    };

    // if we have a cache - use it
    {
        const auto & cache = cache_token_to_piece;

        if (!cache.empty()) {
            const auto & result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t) id_to_token.size()) {
        const std::string & token_text = id_to_token[token].text;
        switch (get_type()) {
            case VocabType::WPM:
            case VocabType::SPM:
            case VocabType::UGM: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & TOKEN_ATTR_NORMAL) {
                    std::string result = token_text;
                    iian_unescape_whitespace(result);
                    return _try_copy(result.data(), result.size());
                }
                if (attr & TOKEN_ATTR_BYTE) {
                    char byte = (char) token_to_byte(token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case VocabType::BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & TOKEN_ATTR_NORMAL) {
                    if (escape_whitespaces) {
                        // SPM-style BPE: tokens contain ▁ for spaces
                        std::string result = token_text;
                        iian_unescape_whitespace(result);
                        return _try_copy(result.data(), result.size());
                    }
                    std::string result = iian_decode_text(token_text);
                    return _try_copy(result.data(), result.size());
                }
                if (attr & TOKEN_ATTR_BYTE) {
                    char byte = (char) token_to_byte(token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case VocabType::RWKV: {
                std::vector<uint8_t> result = iian_unescape_rwkv_token(token_text);

                // If we don't have enough space, return an error
                if (result.size() > (size_t)length) {
                    return -(int)result.size();
                }

                memcpy(buf, result.data(), result.size());
                return (int)result.size();
            }
            case VocabType::PLAMO2: {
                // PLaMo-2 uses similar token handling as BPE/SPM
                if (vocab.is_byte(token)) {
                    // Handle byte tokens like <0xXX>
                    if (token_text.length() == 6 && token_text.substr(0, 3) == "<0x" && token_text.back() == '>') {
                        std::string hex_str = token_text.substr(3, 2);
                        if (std::isxdigit(static_cast<unsigned char>(hex_str[0])) && std::isxdigit(static_cast<unsigned char>(hex_str[1]))) {
                            int hex_val = std::stoi(hex_str, nullptr, 16);
                            if (length < 1) {
                                return -1;
                            }
                            buf[0] = static_cast<char>(hex_val);
                            return 1;
                        }
                    }
                }

                // Normal token - just copy the text
                std::string result = token_text;
                return _try_copy(result.data(), result.size());
            }
            default:
                GGML_ABORT("fatal error");
        }
    }

    return 0;
}

const std::string & Vocab::impl::token_to_piece(token_t token) const {
    return cache_token_to_piece.at(token);
}

int32_t Vocab::impl::detokenize(
               const token_t * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    if (type == VocabType::NONE) {
        return 0;
    }

    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call Vocab::init_tokenizer() first.");

    int32_t avail = text_len_max;
    int32_t total = 0;

    // remove the leading space
    bool remove_space = add_space_prefix;

    if (remove_special && add_bos) {
        if (n_tokens > 0 && tokens[0] == special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && add_eos) {
        if (n_tokens > 0 && tokens[n_tokens - 1] == special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(avail >= 0);
        int32_t n_chars = token_to_piece(tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) {
            avail = 0;
            total -= n_chars;
        } else if (n_chars > 0) {
            avail -= n_chars;
            text  += n_chars;
            total += n_chars;
        }
    }

    if (total > text_len_max) {
        return -total;
    }

    if (clean_spaces) {
        text -= total;  // restart text

        // first pass: characters ?!.,  //TODO: where do these characters come from?
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '?' || x == '!' || x == '.' || x == ',') {  // " ?", " !", " .", " ,"
                    total--;  // remove space
                }
            }
            text[total++] = x;
        }

        // second pass: strip single apostrophe between spaces
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {  // " ' "
                total--;           // remove prev space
                text[++i] = '\0';  // remove next space
            }
            text[total++] = x;
        }

        // third pass: apostrophe contractions  //NOTE: this makes sense?
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '\'' && i + 1 < total3) {
                    const char x1 = text[i + 1];
                    if (x1 == 't' || x1 == 'd') {  // " 't", " 'd"
                        //total--;  // remove space
                    } else if (x1 == 's' || x1 == 'm') {  // " 's", " 'm"
                        total--;  // remove space
                    } else if (i + 2 < total3) {
                        const char x2 = text[i + 2];
                        if ((x1 == 'l' && x2 == 'l')) {  // " 'll"
                            //total--;  // remove space
                        } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {  // " 're", " 've"
                            total--;  // remove space
                        } else {
                            //total--;  // remove space
                        }
                    } else {
                        //total--;  // remove space
                    }
                }
            }
            text[total++] = x;
        }
    }

    return total <= text_len_max ? total : -total;
}

void Vocab::impl::print_info() const {
    IIAN_TOK_LOG_INFO("%s: vocab type            = %s\n",     __func__, type_name().c_str());
    IIAN_TOK_LOG_INFO("%s: n_vocab               = %u\n",     __func__, vocab.n_tokens());
    IIAN_TOK_LOG_INFO("%s: n_merges              = %u\n",     __func__, (uint32_t) bpe_ranks.size());

    // special tokens
    if (special_bos_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: BOS token             = %d '%s'\n", __func__, special_bos_id,     id_to_token.at(special_bos_id).text.c_str() );  }
    if (special_eos_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: EOS token             = %d '%s'\n", __func__, special_eos_id,     id_to_token.at(special_eos_id).text.c_str() );  }
    if (special_eot_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: EOT token             = %d '%s'\n", __func__, special_eot_id,     id_to_token.at(special_eot_id).text.c_str() );  }
    if (special_eom_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: EOM token             = %d '%s'\n", __func__, special_eom_id,     id_to_token.at(special_eom_id).text.c_str() );  }
    if (special_unk_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: UNK token             = %d '%s'\n", __func__, special_unk_id,     id_to_token.at(special_unk_id).text.c_str() );  }
    if (special_sep_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: SEP token             = %d '%s'\n", __func__, special_sep_id,     id_to_token.at(special_sep_id).text.c_str() );  }
    if (special_pad_id  != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: PAD token             = %d '%s'\n", __func__, special_pad_id,     id_to_token.at(special_pad_id).text.c_str() );  }
    if (special_mask_id != TOKEN_NULL)    { IIAN_TOK_LOG_INFO( "%s: MASK token            = %d '%s'\n", __func__, special_mask_id,    id_to_token.at(special_mask_id).text.c_str() ); }

    if (linefeed_id != TOKEN_NULL)        { IIAN_TOK_LOG_INFO( "%s: LF token              = %d '%s'\n", __func__, linefeed_id,        id_to_token.at(linefeed_id).text.c_str() ); }

    if (special_fim_pre_id != TOKEN_NULL) { IIAN_TOK_LOG_INFO( "%s: FIM PRE token         = %d '%s'\n", __func__, special_fim_pre_id, id_to_token.at(special_fim_pre_id).text.c_str() ); }
    if (special_fim_suf_id != TOKEN_NULL) { IIAN_TOK_LOG_INFO( "%s: FIM SUF token         = %d '%s'\n", __func__, special_fim_suf_id, id_to_token.at(special_fim_suf_id).text.c_str() ); }
    if (special_fim_mid_id != TOKEN_NULL) { IIAN_TOK_LOG_INFO( "%s: FIM MID token         = %d '%s'\n", __func__, special_fim_mid_id, id_to_token.at(special_fim_mid_id).text.c_str() ); }
    if (special_fim_pad_id != TOKEN_NULL) { IIAN_TOK_LOG_INFO( "%s: FIM PAD token         = %d '%s'\n", __func__, special_fim_pad_id, id_to_token.at(special_fim_pad_id).text.c_str() ); }
    if (special_fim_rep_id != TOKEN_NULL) { IIAN_TOK_LOG_INFO( "%s: FIM REP token         = %d '%s'\n", __func__, special_fim_rep_id, id_to_token.at(special_fim_rep_id).text.c_str() ); }
    if (special_fim_sep_id != TOKEN_NULL) { IIAN_TOK_LOG_INFO( "%s: FIM SEP token         = %d '%s'\n", __func__, special_fim_sep_id, id_to_token.at(special_fim_sep_id).text.c_str() ); }

    for (const auto & id : special_eog_ids) {
        IIAN_TOK_LOG_INFO( "%s: EOG token             = %d '%s'\n", __func__, id, id_to_token.at(id).text.c_str() );
    }

    IIAN_TOK_LOG_INFO("%s: max token length      = %d\n", __func__, max_token_len);
}

Vocab::Vocab() : pimpl(new impl(*this)) {
}

Vocab::~Vocab() = default;

void Vocab::load(const gguf_context * ctx) {
    pimpl->load(ctx);
}

std::string Vocab::get_tokenizer_model() const {
    return pimpl->tokenizer_model;
}

std::string Vocab::get_tokenizer_pre() const {
    return pimpl->tokenizer_pre;
}

VocabType Vocab::get_type() const {
    return pimpl->type;
}

VocabPreType Vocab::get_pre_type() const {
    return pimpl->pre_type;
}

uint32_t Vocab::n_tokens() const {
    return (uint32_t) pimpl->id_to_token.size();
}

uint32_t Vocab::n_token_types() const {
    return (uint32_t) pimpl->n_token_types;
}

std::string Vocab::type_name() const{
    return pimpl->type_name();
}

bool Vocab::is_normal(token_t id) const {
    return pimpl->is_normal(id);
}

bool Vocab::is_unknown(token_t id) const {
    return pimpl->is_unknown(id);
}

bool Vocab::is_control(token_t id) const {
    return pimpl->is_control(id);
}

bool Vocab::is_byte(token_t id) const {
    return pimpl->is_byte(id);
}

bool Vocab::is_user_defined(token_t id) const {
    return pimpl->is_user_defined(id);
}

bool Vocab::is_unused(token_t id) const {
    return pimpl->is_unused(id);
}

bool Vocab::is_eog(token_t id) const {
    return pimpl->is_eog(id);
}

uint8_t Vocab::token_to_byte(token_t id) const {
    return pimpl->token_to_byte(id);
}

token_t Vocab::byte_to_token(uint8_t ch) const {
    GGML_ASSERT(get_type() != VocabType::NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (get_type()) {
        case VocabType::SPM:
        case VocabType::UGM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = pimpl->token_to_id.find(buf);
            if (token != pimpl->token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return pimpl->token_to_id.at(buf2);
        }
        case VocabType::WPM:
        case VocabType::BPE: {
            return pimpl->token_to_id.at(unicode_byte_to_utf8(ch));
        }
        case VocabType::PLAMO2: {
            // PLaMo-2 uses byte tokens in format <0xXX>
            char hex_str[8];
            snprintf(hex_str, sizeof(hex_str), "<0x%02X>", ch);
            return pimpl->token_to_id.at(hex_str);
        }
        default:
            GGML_ABORT("fatal error");
    }
}

token_t Vocab::text_to_token(const std::string & text) const {
    GGML_ASSERT(pimpl->type != VocabType::NONE);
    auto it = pimpl->token_to_id.find(text);
    if (it != pimpl->token_to_id.end()) {
        return (*it).second;
    }
    return TOKEN_NULL;
}

const Vocab::token_data & Vocab::get_token_data(token_t id) const {
    GGML_ASSERT(pimpl->type != VocabType::NONE);
    return pimpl->id_to_token.at(id);
}

const char * Vocab::token_get_text(token_t id) const {
    GGML_ASSERT(pimpl->type != VocabType::NONE);
    return pimpl->id_to_token.at(id).text.c_str();
}

float Vocab::token_get_score(token_t id) const {
    GGML_ASSERT(pimpl->type != VocabType::NONE);
    return pimpl->id_to_token.at(id).score;
}

TokenAttr Vocab::token_get_attr(token_t id) const {
    return pimpl->token_get_attr(id);
}

token_t Vocab::token_bos() const {
    return pimpl->special_bos_id;
}

token_t Vocab::token_eos() const {
    return pimpl->special_eos_id;
}

token_t Vocab::token_eot() const {
    return pimpl->special_eot_id;
}

token_t Vocab::token_eom() const {
    return pimpl->special_eom_id;
}

token_t Vocab::token_unk() const {
    return pimpl->special_unk_id;
}

token_t Vocab::token_sep() const {
    return pimpl->special_sep_id;
}

token_t Vocab::token_nl() const {
    return pimpl->linefeed_id;
}

token_t Vocab::token_pad() const {
    return pimpl->special_pad_id;
}

token_t Vocab::token_prefix() const {
    return pimpl->special_fim_pre_id;
}

token_t Vocab::token_middle() const {
    return pimpl->special_fim_mid_id;
}

token_t Vocab::token_suffix() const {
    return pimpl->special_fim_suf_id;
}

token_t Vocab::token_fim_pre() const {
    return pimpl->special_fim_pre_id;
}

token_t Vocab::token_fim_suf() const {
    return pimpl->special_fim_suf_id;
}

token_t Vocab::token_fim_mid() const {
    return pimpl->special_fim_mid_id;
}

token_t Vocab::token_fim_pad() const {
    return pimpl->special_fim_pad_id;
}

token_t Vocab::token_fim_rep() const {
    return pimpl->special_fim_rep_id;
}

token_t Vocab::token_fim_sep() const {
    return pimpl->special_fim_sep_id;
}

token_t Vocab::token_mask() const {
    return pimpl->special_mask_id;
}

bool Vocab::get_add_space_prefix() const {
    return pimpl->add_space_prefix;
}

bool Vocab::get_add_bos() const {
    return pimpl->add_bos;
}

bool Vocab::get_add_eos() const {
    return pimpl->add_eos;
}

bool Vocab::get_add_sep() const {
    return pimpl->add_sep;
}

bool Vocab::get_ignore_merges() const {
    return pimpl->ignore_merges;
}

bool Vocab::get_clean_spaces() const {
    return pimpl->clean_spaces;
}

bool Vocab::get_remove_extra_whitespaces() const {
    return pimpl->remove_extra_whitespaces;
}

bool Vocab::get_escape_whitespaces() const {
    return pimpl->escape_whitespaces;
}

bool Vocab::get_treat_whitespace_as_suffix() const {
    return pimpl->treat_whitespace_as_suffix;
}

const Vocab::normalizer_options & Vocab::get_normalizer_opts() const {
    return pimpl->normalizer_opts;
}

const std::vector<token_t> & Vocab::get_suppress_tokens() const {
    return pimpl->suppress_tokens;
}

int Vocab::max_token_len() const {
    return pimpl->max_token_len;
}

int Vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    GGML_ASSERT(token_left.find(' ')   == std::string::npos);
    GGML_ASSERT(token_right.find(' ')  == std::string::npos);

    auto it = pimpl->bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == pimpl->bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

std::vector<std::string> Vocab::get_bpe_merges() const {
    int max_rank = -1;
    for (const auto & pair : pimpl->bpe_ranks) {
        max_rank = std::max(max_rank, pair.second);
    }
    std::vector<std::string> result(max_rank + 1);

    for (const auto & pair : pimpl->bpe_ranks) {
        result[pair.second] = pair.first.first + " " + pair.first.second;
    }

    return result;
}

std::vector<char> Vocab::get_precompiled_charsmap() const {
    return pimpl->precompiled_charsmap;
}

int32_t Vocab::tokenize(
                  const char * text,
                     int32_t   text_len,
                 token_t * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) const {
    auto res = tokenize(std::string(text, text_len), add_special, parse_special);
    if (res.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        IIAN_TOK_LOG_ERROR("%s: tokenization result size %zu exceeds int32_t limit\n", __func__, res.size());
        return std::numeric_limits<int32_t>::min();
    }

    if (n_tokens_max < (int) res.size()) {
        // IIAN_TOK_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

std::vector<token_t> Vocab::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    return pimpl->tokenize(raw_text, add_special, parse_special);
}

const std::string & Vocab::token_to_piece(token_t token) const {
    return pimpl->token_to_piece(token);
}

int32_t Vocab::token_to_piece(token_t token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    return pimpl->token_to_piece(token, buf, length, lstrip, special);
}

int32_t Vocab::detokenize(
               const token_t * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    return pimpl->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}

std::string Vocab::detokenize(const std::vector<token_t> & tokens, bool special) const {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

void Vocab::print_info() const {
    pimpl->print_info();
}

//
// chat templates
//

std::string Vocab::chat_template(const std::string & variant) const {
    const auto it = pimpl->chat_templates.find(variant);
    if (it == pimpl->chat_templates.end()) {
        return "";
    }
    return it->second;
}

//
// public interface: iian::Tokenizer implemented on top of Vocab
//

class VocabTokenizer final : public Tokenizer {
public:
    explicit VocabTokenizer(std::unique_ptr<Vocab> vocab) : vocab_(std::move(vocab)) {
        const Vocab & v = *vocab_;
        special_.bos     = v.token_bos();
        special_.eos     = v.token_eos();
        special_.eot     = v.token_eot();
        special_.eom     = v.token_eom();
        special_.unk     = v.token_unk();
        special_.sep     = v.token_sep();
        special_.pad     = v.token_pad();
        special_.nl      = v.token_nl();
        special_.mask    = v.token_mask();
        special_.fim_pre = v.token_fim_pre();
        special_.fim_suf = v.token_fim_suf();
        special_.fim_mid = v.token_fim_mid();
        special_.fim_pad = v.token_fim_pad();
        special_.fim_rep = v.token_fim_rep();
        special_.fim_sep = v.token_fim_sep();
        special_.add_bos = v.get_add_bos();
        special_.add_eos = v.get_add_eos();
        special_.add_sep = v.get_add_sep();
    }

    VocabType   type()          const override { return vocab_->get_type(); }
    std::string type_name()     const override { return vocab_->type_name(); }
    std::string pre_type_name() const override { return vocab_->get_tokenizer_pre(); }
    int32_t     n_tokens()      const override { return (int32_t) vocab_->n_tokens(); }

    std::vector<token_t> encode(std::string_view text, bool add_special, bool parse_special) const override {
        if (vocab_->get_type() == VocabType::NONE) {
            throw std::runtime_error("iian::Tokenizer::encode: model has no vocabulary");
        }
        return vocab_->tokenize(std::string(text), add_special, parse_special);
    }

    // llama_token_to_piece(vocab, tok, buf, len, /*lstrip=*/0, special)
    std::string token_to_piece(token_t tok, bool special) const override {
        if (!in_range(tok)) {
            return "";
        }
        std::string piece;
        piece.resize(piece.capacity());
        int32_t n_chars = vocab_->token_to_piece(tok, &piece[0], (int32_t) piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            const int32_t check = vocab_->token_to_piece(tok, &piece[0], (int32_t) piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
            n_chars = -n_chars;
        }
        piece.resize(n_chars);
        return piece;
    }

    // llama_detokenize(vocab, toks, n, buf, len, remove_special, unparse_special)
    std::string decode(const std::vector<token_t> & toks, bool remove_special, bool unparse_special) const override {
        std::string text;
        text.resize(std::max(text.capacity(), toks.size()));
        int32_t n_chars = vocab_->detokenize(toks.data(), (int32_t) toks.size(), &text[0], (int32_t) text.size(), remove_special, unparse_special);
        if (n_chars < 0) {
            text.resize(-n_chars);
            n_chars = vocab_->detokenize(toks.data(), (int32_t) toks.size(), &text[0], (int32_t) text.size(), remove_special, unparse_special);
            GGML_ASSERT(n_chars <= (int32_t) text.size());  // whitespace trimming is performed after per-token detokenization
        }
        text.resize(n_chars);
        return text;
    }

    const SpecialTokens & special() const override { return special_; }

    bool is_eog(token_t tok) const override { return vocab_->is_eog(tok); }

    bool is_control(token_t tok) const override {
        return in_range(tok) && vocab_->get_type() != VocabType::NONE && vocab_->is_control(tok);
    }

    bool is_byte(token_t tok) const override {
        return in_range(tok) && vocab_->get_type() != VocabType::NONE && vocab_->is_byte(tok);
    }

    const std::string & token_text(token_t tok) const override {
        return vocab_->get_token_data(tok).text; // throws std::out_of_range for bad ids (like llama_vocab_get_text)
    }

    TokenAttr token_attr(token_t tok) const override {
        return vocab_->token_get_attr(tok);
    }

    token_t text_to_token(std::string_view text) const override {
        if (vocab_->get_type() == VocabType::NONE) {
            return TOKEN_NULL;
        }
        return vocab_->text_to_token(std::string(text));
    }

    std::string chat_template(const std::string & variant) const override {
        return vocab_->chat_template(variant);
    }

    const Vocab & vocab() const { return *vocab_; }

private:
    bool in_range(token_t tok) const {
        return tok >= 0 && (uint32_t) tok < vocab_->n_tokens();
    }

    std::unique_ptr<Vocab> vocab_;
    SpecialTokens special_;
};

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const gguf_context * ctx) {
    if (ctx == nullptr) {
        throw std::invalid_argument("iian::Tokenizer::from_gguf: null gguf_context");
    }
    auto vocab = std::make_unique<Vocab>();
    vocab->load(ctx);
    return std::make_unique<VocabTokenizer>(std::move(vocab));
}

} // namespace iian
