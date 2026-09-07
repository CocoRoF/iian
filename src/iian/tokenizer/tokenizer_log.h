#pragma once
// Minimal logging shim for the tokenizer module (stand-in for llama.cpp's LLAMA_LOG_*).
// Header-only on purpose: the project logger will replace this later; only the IIAN_TOK_LOG_* macros
// are used by the tokenizer sources.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace iian {

enum class TokLogLevel : int {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3,
    NONE  = 4,
};

namespace detail {
inline TokLogLevel & tok_log_level_ref() {
    // default: WARN; override once via env IIAN_TOK_LOG_LEVEL=debug|info|warn|error|none
    static TokLogLevel level = [] {
        const char * env = std::getenv("IIAN_TOK_LOG_LEVEL");
        if (env == nullptr) return TokLogLevel::WARN;
        if (!std::strcmp(env, "debug")) return TokLogLevel::DEBUG;
        if (!std::strcmp(env, "info"))  return TokLogLevel::INFO;
        if (!std::strcmp(env, "warn"))  return TokLogLevel::WARN;
        if (!std::strcmp(env, "error")) return TokLogLevel::ERROR;
        if (!std::strcmp(env, "none"))  return TokLogLevel::NONE;
        return TokLogLevel::WARN;
    }();
    return level;
}
} // namespace detail

inline TokLogLevel tok_log_level() { return detail::tok_log_level_ref(); }
inline void tok_log_set_level(TokLogLevel level) { detail::tok_log_level_ref() = level; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
inline void tok_log(TokLogLevel level, const char * fmt, ...) {
    if (static_cast<int>(level) < static_cast<int>(tok_log_level())) {
        return;
    }
    const char * tag = "";
    switch (level) {
        case TokLogLevel::DEBUG: tag = "[tokenizer:debug] "; break;
        case TokLogLevel::INFO:  tag = "[tokenizer:info] ";  break;
        case TokLogLevel::WARN:  tag = "[tokenizer:warn] ";  break;
        case TokLogLevel::ERROR: tag = "[tokenizer:error] "; break;
        default: break;
    }
    std::fputs(tag, stderr);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fflush(stderr);
}

} // namespace iian

#define IIAN_TOK_LOG_DEBUG(...) ::iian::tok_log(::iian::TokLogLevel::DEBUG, __VA_ARGS__)
#define IIAN_TOK_LOG_INFO(...)  ::iian::tok_log(::iian::TokLogLevel::INFO,  __VA_ARGS__)
#define IIAN_TOK_LOG_WARN(...)  ::iian::tok_log(::iian::TokLogLevel::WARN,  __VA_ARGS__)
#define IIAN_TOK_LOG_ERROR(...) ::iian::tok_log(::iian::TokLogLevel::ERROR, __VA_ARGS__)
