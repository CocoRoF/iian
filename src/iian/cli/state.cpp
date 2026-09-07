#include "state.h"

#include "paths.h"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>

namespace iian::cli {

using json = nlohmann::json;

std::string RunRecord::endpoint() const {
    if (!unix_socket.empty()) return "unix://" + unix_socket;
    std::string h = host == "0.0.0.0" || host == "::" || host.empty() ? "127.0.0.1" : host;
    if (h.find(':') != std::string::npos) h = "[" + h + "]";
    return "http://" + h + ":" + std::to_string(port);
}

json RunRecord::to_json() const {
    return json{
        {"name", name}, {"pid", pid}, {"host", host}, {"port", port}, {"unix_socket", unix_socket},
        {"model_path", model_path}, {"model_name", model_name}, {"started_at", started_at},
        {"log_file", log_file}, {"args", args}, {"admin", admin}, {"detached", detached}, {"api_key", api_key},
    };
}

RunRecord RunRecord::from_json(const json & j) {
    RunRecord r;
    r.name = j.value("name", "");
    r.pid = j.value("pid", (int64_t) 0);
    r.host = j.value("host", "127.0.0.1");
    r.port = j.value("port", 0);
    r.unix_socket = j.value("unix_socket", "");
    r.model_path = j.value("model_path", "");
    r.model_name = j.value("model_name", "");
    r.started_at = j.value("started_at", (int64_t) 0);
    r.log_file = j.value("log_file", "");
    if (j.contains("args") && j["args"].is_array()) for (const auto & a : j["args"]) if (a.is_string()) r.args.push_back(a.get<std::string>());
    r.admin = j.value("admin", false);
    r.detached = j.value("detached", false);
    r.api_key = j.value("api_key", "");
    return r;
}

std::string record_path(const std::string & name) { return run_dir() + "/" + name + ".json"; }

bool save_record(const RunRecord & r) { return write_file_atomic(record_path(r.name), r.to_json().dump(2) + "\n", 0600); }
bool remove_record(const std::string & name) { return ::unlink(record_path(name).c_str()) == 0; }

std::optional<RunRecord> load_record(const std::string & name) {
    auto s = read_file(record_path(name));
    if (!s) return std::nullopt;
    json j = json::parse(*s, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::nullopt;
    return RunRecord::from_json(j);
}

std::vector<RunRecord> list_records() {
    std::vector<RunRecord> out;
    for (const auto & f : list_dir(run_dir())) {
        if (f.size() < 6 || f.compare(f.size() - 5, 5, ".json") != 0) continue;
        if (auto r = load_record(f.substr(0, f.size() - 5))) out.push_back(*r);
    }
    std::sort(out.begin(), out.end(), [](const RunRecord & a, const RunRecord & b) { return a.started_at < b.started_at; });
    return out;
}

bool pid_alive(int64_t pid) {
    if (pid <= 0) return false;
    return ::kill((pid_t) pid, 0) == 0 || errno == EPERM;
}

std::vector<RunRecord> find_records(const std::string & selector) {
    std::vector<RunRecord> all = list_records(), out;
    if (selector == "all") return all;
    bool numeric = !selector.empty() && std::all_of(selector.begin(), selector.end(), ::isdigit);
    for (const auto & r : all) {
        if (r.name == selector || (numeric && r.pid == std::stoll(selector))) out.push_back(r);
    }
    return out;
}

std::string default_server_name(const std::string & model_path) {
    std::string n = strip_gguf_ext(basename_of(model_path));
    std::string out;
    for (char c : n) {
        if (isalnum((unsigned char) c) || c == '-' || c == '_' || c == '.') out += c;
        else out += '-';
    }
    if (out.empty()) out = "iian";
    return out;
}

} // namespace iian::cli
