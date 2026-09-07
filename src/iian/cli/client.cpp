#include "client.h"

#include "cpp-httplib/httplib.h"

#include <sys/socket.h>

namespace iian::cli {

using json = nlohmann::json;

json HttpResult::json() const {
    auto j = nlohmann::json::parse(body, nullptr, false);
    return j.is_discarded() ? nlohmann::json(nullptr) : j;
}

std::string HttpResult::error_message() const {
    if (status == 0) return error.empty() ? "connection failed" : error;
    auto j = json();
    if (j.is_object() && j.contains("error") && j["error"].is_object() && j["error"].contains("message")) return j["error"]["message"].get<std::string>();
    if (j.is_object() && j.contains("message")) return j["message"].dump();
    return "HTTP " + std::to_string(status) + (body.empty() ? "" : ": " + body.substr(0, 200));
}

ServerClient::ServerClient(const RunRecord & rec, std::chrono::milliseconds timeout) : api_key_(rec.api_key) { init(rec.endpoint(), timeout); }
ServerClient::ServerClient(const std::string & endpoint, const std::string & api_key, std::chrono::milliseconds timeout) : api_key_(api_key) { init(endpoint, timeout); }
ServerClient::~ServerClient() = default;

void ServerClient::init(const std::string & endpoint, std::chrono::milliseconds timeout) {
    endpoint_ = endpoint;
    if (endpoint.rfind("unix://", 0) == 0) {
        cli_ = std::make_unique<httplib::Client>(endpoint.substr(7), 80);
        cli_->set_address_family(AF_UNIX);
    } else {
        cli_ = std::make_unique<httplib::Client>(endpoint);
    }
    cli_->set_connection_timeout(timeout);
    cli_->set_read_timeout(timeout);
    cli_->set_write_timeout(timeout);
    cli_->set_keep_alive(false);
    httplib::Headers h;
    if (!api_key_.empty()) h.emplace("Authorization", "Bearer " + api_key_);
    cli_->set_default_headers(h);
}

void ServerClient::set_read_timeout(std::chrono::milliseconds t) { cli_->set_read_timeout(t); }

static HttpResult from_result(const httplib::Result & r) {
    HttpResult out;
    if (!r) { out.error = httplib::to_string(r.error()); return out; }
    out.status = r->status;
    out.body = r->body;
    return out;
}

HttpResult ServerClient::get(const std::string & path) { return from_result(cli_->Get(path)); }

HttpResult ServerClient::post_json(const std::string & path, const json & body) {
    return from_result(cli_->Post(path, body.dump(), "application/json"));
}

HttpResult ServerClient::post_sse(const std::string & path, const json & body, const std::function<bool(const std::string &)> & on_data) {
    HttpResult out;
    std::string buf, non_sse_body;
    bool is_sse = false;
    // httplib's Post has no ResponseHandler overload: detect SSE lazily from the first bytes
    // ("data:" prefix) since the server only ever streams text/event-stream on this path.
    bool first = true;
    auto res = cli_->Post(path, httplib::Headers{}, body.dump(), "application/json",
        [&](const char * data, size_t len) {
            if (first) { first = false; is_sse = len >= 5 && std::string(data, 5) == "data:"; }
            if (!is_sse) { non_sse_body.append(data, len); return true; }
            buf.append(data, len);
            size_t pos;
            while ((pos = buf.find("\n\n")) != std::string::npos) {
                std::string ev = buf.substr(0, pos);
                buf.erase(0, pos + 2);
                size_t p = 0;
                while (p < ev.size()) {
                    size_t nl = ev.find('\n', p);
                    std::string line = ev.substr(p, nl == std::string::npos ? std::string::npos : nl - p);
                    p = nl == std::string::npos ? ev.size() : nl + 1;
                    if (line.rfind("data:", 0) == 0) {
                        std::string payload = line.substr(5);
                        if (!payload.empty() && payload[0] == ' ') payload.erase(0, 1);
                        if (!on_data(payload)) return false;
                    }
                }
            }
            return true;
        });
    if (res) out.status = res->status;
    else if (out.status == 0 && res.error() != httplib::Error::Canceled) out.error = httplib::to_string(res.error());
    if (!is_sse) out.body = non_sse_body;
    return out;
}

HealthInfo probe_health(const RunRecord & rec, std::chrono::milliseconds timeout) {
    HealthInfo h;
    try {
        ServerClient c(rec, timeout);
        HttpResult r = c.get("/health");
        h.status = r.status;
        if (r.status == 0) { h.state = "down"; return h; }
        h.reachable = true;
        auto j = r.json();
        h.state = j.is_object() && j.contains("status") && j["status"].is_string() ? j["status"].get<std::string>() : (r.ok() ? "ok" : "error");
    } catch (...) {
        h.state = "down";
    }
    return h;
}

} // namespace iian::cli
