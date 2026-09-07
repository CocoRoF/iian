#include "iian/log.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

namespace iian {

namespace {

struct LoggerImpl {
    std::mutex mtx;
    FILE * file = nullptr;
    std::function<void(const LogRecord &)> sink;
    thread_local static uint64_t request_ctx;
};
thread_local uint64_t LoggerImpl::request_ctx = 0;

const char * level_color(LogLevel lvl) {
    switch (lvl) {
        case LogLevel::TRACE: return "\x1b[90m";
        case LogLevel::DEBUG: return "\x1b[36m";
        case LogLevel::INFO:  return "\x1b[32m";
        case LogLevel::WARN:  return "\x1b[33m";
        case LogLevel::ERROR: return "\x1b[31m";
        default:              return "";
    }
}

void json_escape(std::string & out, std::string_view s) {
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += (char) c;
        }
    }
}

} // namespace

Logger & Logger::instance() {
    static Logger l;
    return l;
}

Logger::Logger() : impl_(new LoggerImpl()) {
    if (const char * e = getenv("IIAN_LOG_LEVEL")) { LogLevel l; if (parse_level(e, l)) level_ = l; }
    if (const char * e = getenv("IIAN_LOG_FORMAT")) { if (strcmp(e, "json") == 0) format_ = LogFormat::JSON; }
    if (const char * e = getenv("NO_COLOR")) { (void) e; color_ = false; }
}

void Logger::set_level(LogLevel lvl) { level_ = lvl; }
void Logger::set_format(LogFormat fmt) { format_ = fmt; }
void Logger::set_color(bool on) { color_ = on; }
void Logger::set_timestamps(bool on) { timestamps_ = on; }

void Logger::set_file(const std::string & path) {
    auto * im = static_cast<LoggerImpl *>(impl_);
    std::lock_guard<std::mutex> lk(im->mtx);
    if (im->file) { fclose(im->file); im->file = nullptr; }
    if (!path.empty()) {
        im->file = fopen(path.c_str(), "a");
        if (im->file) setvbuf(im->file, nullptr, _IOLBF, 0);
    }
}

void Logger::set_sink(std::function<void(const LogRecord &)> sink) {
    auto * im = static_cast<LoggerImpl *>(impl_);
    std::lock_guard<std::mutex> lk(im->mtx);
    im->sink = std::move(sink);
}

void Logger::set_request_context(uint64_t id) { LoggerImpl::request_ctx = id; }
uint64_t Logger::request_context() { return LoggerImpl::request_ctx; }

const char * Logger::level_name(LogLevel lvl) {
    switch (lvl) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default:              return "OFF";
    }
}

bool Logger::parse_level(std::string_view s, LogLevel & out) {
    std::string l(s);
    for (auto & c : l) c = (char) tolower((unsigned char) c);
    if (l == "trace") out = LogLevel::TRACE;
    else if (l == "debug") out = LogLevel::DEBUG;
    else if (l == "info") out = LogLevel::INFO;
    else if (l == "warn" || l == "warning") out = LogLevel::WARN;
    else if (l == "error") out = LogLevel::ERROR;
    else if (l == "off" || l == "none") out = LogLevel::OFF;
    else return false;
    return true;
}

void Logger::log(LogLevel lvl, const char * tag, const char * file, int line, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlog(lvl, tag, file, line, fmt, ap);
    va_end(ap);
}

void Logger::vlog(LogLevel lvl, const char * tag, const char * file, int line, const char * fmt, va_list ap) {
    char stack_buf[1024];
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(stack_buf, sizeof stack_buf, fmt, ap);
    std::string msg;
    if (n < 0) { msg = fmt; }
    else if ((size_t) n < sizeof stack_buf) { msg.assign(stack_buf, n); }
    else { msg.resize(n + 1); vsnprintf(msg.data(), n + 1, fmt, ap2); msg.resize(n); }
    va_end(ap2);

    const auto now = std::chrono::system_clock::now();
    LogRecord rec{lvl, std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count(),
                  tag, std::move(msg), LoggerImpl::request_ctx, file, line};

    std::string out;
    out.reserve(rec.msg.size() + 96);
    if (format_ == LogFormat::JSON) {
        char ts[64];
        time_t secs = rec.time_us / 1000000;
        struct tm tmv; gmtime_r(&secs, &tmv);
        strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tmv);
        char tsbuf[96]; snprintf(tsbuf, sizeof tsbuf, "%s.%06lldZ", ts, (long long)(rec.time_us % 1000000));
        out += "{\"ts\":\""; out += tsbuf; out += "\",\"level\":\""; out += level_name(lvl);
        out += "\",\"tag\":\""; json_escape(out, tag ? tag : "");
        if (rec.request_id) { out += "\",\"req\":"; out += std::to_string(rec.request_id); out += ",\"msg\":\""; }
        else out += "\",\"msg\":\"";
        json_escape(out, rec.msg);
        out += "\"}\n";
    } else {
        if (timestamps_) {
            time_t secs = rec.time_us / 1000000;
            struct tm tmv; localtime_r(&secs, &tmv);
            char ts[32]; strftime(ts, sizeof ts, "%H:%M:%S", &tmv);
            char buf[48]; snprintf(buf, sizeof buf, "%s.%03lld ", ts, (long long)((rec.time_us / 1000) % 1000));
            if (color_) out += "\x1b[90m";
            out += buf;
            if (color_) out += "\x1b[0m";
        }
        if (color_) out += level_color(lvl);
        char lb[8]; snprintf(lb, sizeof lb, "%-5s", level_name(lvl));
        out += lb;
        if (color_) out += "\x1b[0m";
        out += ' ';
        if (color_) out += "\x1b[1m";
        char tb[16]; snprintf(tb, sizeof tb, "%-6s", tag ? tag : "");
        out += tb;
        if (color_) out += "\x1b[0m";
        if (rec.request_id) { out += " [req "; out += std::to_string(rec.request_id); out += "]"; }
        out += ' ';
        out += rec.msg;
        if (out.empty() || out.back() != '\n') out += '\n';
    }

    auto * im = static_cast<LoggerImpl *>(impl_);
    std::lock_guard<std::mutex> lk(im->mtx);
    fputs(out.c_str(), stderr);
    if (im->file) fputs(out.c_str(), im->file);
    if (im->sink) im->sink(rec);
}

} // namespace iian
