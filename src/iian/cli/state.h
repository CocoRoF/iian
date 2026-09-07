#pragma once
// Run records: one JSON file per server under $IIAN_HOME/run/<name>.json (written by `serve`,
// read by ps/stop/logs/status/run/complete).
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace iian::cli {

struct RunRecord {
    std::string name;
    int64_t     pid = 0;
    std::string host;
    int         port = 0;
    std::string unix_socket;
    std::string model_path;
    std::string model_name;
    int64_t     started_at = 0;       // unix seconds
    std::string log_file;
    std::vector<std::string> args;
    bool        admin = false;
    bool        detached = false;
    std::string api_key;              // stored 0600 so local tools can talk to the server

    std::string endpoint() const;     // http://host:port | unix://path
    nlohmann::json to_json() const;
    static RunRecord from_json(const nlohmann::json & j);
};

std::string record_path(const std::string & name);
bool save_record(const RunRecord & r);
bool remove_record(const std::string & name);
std::optional<RunRecord> load_record(const std::string & name);
std::vector<RunRecord> list_records();
bool pid_alive(int64_t pid);
// name, pid or "all" -> matching records
std::vector<RunRecord> find_records(const std::string & selector);
// Turn a model spec into a sane default server name (basename without extension, lowercase-safe chars).
std::string default_server_name(const std::string & model_path);

} // namespace iian::cli
