#pragma once
// Structured logging for iian.
//
// Design goals (vs. vLLM's noisy text logs and llama.cpp's flat stderr):
//   * every line carries level, timestamp, subsystem tag and (optionally) a request id
//   * two sinks: pretty (colored, human) and JSON-lines (machine); switchable at runtime
//   * cheap when disabled: level check before formatting
//   * per-thread request context so deep code doesn't need to thread request ids through
#include <cstdarg>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace iian {

enum class LogLevel : int { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4, OFF = 5 };

enum class LogFormat { PRETTY, JSON };

struct LogRecord {
    LogLevel    level;
    int64_t     time_us;       // wall clock, microseconds since epoch
    const char * tag;          // subsystem ("engine", "sched", "kv", "http", "model", ...)
    std::string  msg;
    uint64_t    request_id;    // 0 = none
    const char * file;
    int          line;
};

class Logger {
public:
    static Logger & instance();

    void set_level(LogLevel lvl);
    LogLevel level() const { return level_; }
    bool enabled(LogLevel lvl) const { return static_cast<int>(lvl) >= static_cast<int>(level_); }

    void set_format(LogFormat fmt);
    void set_color(bool on);
    void set_timestamps(bool on);
    // Redirect output (default: stderr). Also used by the daemon to tee into a log file.
    void set_file(const std::string & path);          // append; "" = stderr only
    void set_sink(std::function<void(const LogRecord &)> sink); // extra sink (e.g. in-memory ring for `iian logs`)

    void log(LogLevel lvl, const char * tag, const char * file, int line, const char * fmt, ...) __attribute__((format(printf, 6, 7)));
    void vlog(LogLevel lvl, const char * tag, const char * file, int line, const char * fmt, va_list ap);

    // Per-thread request context.
    static void     set_request_context(uint64_t request_id);
    static uint64_t request_context();

    // Parses "trace|debug|info|warn|error|off" (case-insensitive). Returns false on bad input.
    static bool parse_level(std::string_view s, LogLevel & out);
    static const char * level_name(LogLevel lvl);

private:
    Logger();
    LogLevel  level_ = LogLevel::INFO;
    LogFormat format_ = LogFormat::PRETTY;
    bool      color_ = true;
    bool      timestamps_ = true;
    void *    impl_ = nullptr;
};

struct RequestLogScope {
    uint64_t prev;
    explicit RequestLogScope(uint64_t id) : prev(Logger::request_context()) { Logger::set_request_context(id); }
    ~RequestLogScope() { Logger::set_request_context(prev); }
};

} // namespace iian

#define IIAN_LOG(lvl, tag, ...) \
    do { if (::iian::Logger::instance().enabled(lvl)) ::iian::Logger::instance().log(lvl, tag, __FILE__, __LINE__, __VA_ARGS__); } while (0)

#define LOG_TRC(tag, ...) IIAN_LOG(::iian::LogLevel::TRACE, tag, __VA_ARGS__)
#define LOG_DBG(tag, ...) IIAN_LOG(::iian::LogLevel::DEBUG, tag, __VA_ARGS__)
#define LOG_INF(tag, ...) IIAN_LOG(::iian::LogLevel::INFO,  tag, __VA_ARGS__)
#define LOG_WRN(tag, ...) IIAN_LOG(::iian::LogLevel::WARN,  tag, __VA_ARGS__)
#define LOG_ERR(tag, ...) IIAN_LOG(::iian::LogLevel::ERROR, tag, __VA_ARGS__)
