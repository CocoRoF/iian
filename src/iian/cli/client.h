#pragma once
// Small HTTP client for talking to a running iian server (JSON + SSE), built on cpp-httplib.
#include "state.h"

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "nlohmann/json.hpp"

namespace httplib { class Client; }

namespace iian::cli {

struct HttpResult {
    int status = 0;                 // 0 = transport error
    std::string body;
    std::string error;              // transport error text
    bool ok() const { return status >= 200 && status < 300; }
    nlohmann::json json() const;    // parses body (discarded -> null)
    std::string error_message() const;   // OpenAI error message or transport error
};

class ServerClient {
public:
    explicit ServerClient(const RunRecord & rec, std::chrono::milliseconds timeout = std::chrono::seconds(5));
    ServerClient(const std::string & endpoint, const std::string & api_key, std::chrono::milliseconds timeout = std::chrono::seconds(5));
    ~ServerClient();

    void set_read_timeout(std::chrono::milliseconds t);
    HttpResult get(const std::string & path);
    HttpResult post_json(const std::string & path, const nlohmann::json & body);
    // POST + parse text/event-stream; `on_data` receives each "data:" payload (without the prefix).
    // Returns the final status (0 on transport error); stops early if on_data returns false.
    HttpResult post_sse(const std::string & path, const nlohmann::json & body, const std::function<bool(const std::string &)> & on_data);
    const std::string & endpoint() const { return endpoint_; }

private:
    void init(const std::string & endpoint, std::chrono::milliseconds timeout);
    std::string endpoint_, api_key_;
    std::unique_ptr<httplib::Client> cli_;
};

// Probe helpers used by ps/status/run.
struct HealthInfo { bool reachable = false; int status = 0; std::string state; };   // state: ok|loading|stopping|down
HealthInfo probe_health(const RunRecord & rec, std::chrono::milliseconds timeout = std::chrono::milliseconds(800));

} // namespace iian::cli
