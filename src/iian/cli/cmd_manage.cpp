// iian ps / stop / restart / logs / status
#include "args.h"
#include "client.h"
#include "commands.h"
#include "paths.h"
#include "state.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <thread>
#include <unistd.h>

namespace iian::cli {

int cmd_ps(const Args & args) {
    ArgParser p("iian ps", "iian ps [options]", "List servers started with `iian serve`.");
    p.add("--prune", "Remove records of servers that are no longer running");
    p.add("--json", "Machine-readable output");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;

    auto recs = list_records();
    if (p.get_bool("json")) {
        nlohmann::json a = nlohmann::json::array();
        for (auto & r : recs) { auto j = r.to_json(); j["alive"] = pid_alive(r.pid); j.erase("api_key"); a.push_back(j); }
        printf("%s\n", a.dump(2).c_str());
        return 0;
    }
    if (recs.empty()) { printf("no servers running (start one with `iian serve <model> -d`)\n"); return 0; }
    printf("%-18s %-8s %-30s %-28s %-9s %-10s %s\n", "NAME", "PID", "MODEL", "ENDPOINT", "STATUS", "UPTIME", "REQS (run/wait)");
    const int64_t now = (int64_t) std::time(nullptr);
    for (auto & r : recs) {
        const bool alive = pid_alive(r.pid);
        std::string status = "dead", reqs = "-";
        if (alive) {
            HealthInfo h = probe_health(r);
            status = h.reachable ? h.state : "starting";
            if (h.reachable && r.admin) {
                try {
                    ServerClient c(r, std::chrono::milliseconds(800));
                    auto res = c.get("/v1/engine/stats");
                    if (res.ok()) { auto j = res.json(); if (j.contains("stats")) reqs = std::to_string(j["stats"]["num_running"].get<int>()) + "/" + std::to_string(j["stats"]["num_waiting"].get<int>()); }
                } catch (...) {}
            }
        }
        std::string st = status == "ok" ? color("running", 32) : status == "dead" ? color("dead", 31) : color(status, 33);
        printf("%-18s %-8lld %-30s %-28s %-9s %-10s %s\n", r.name.c_str(), (long long) r.pid, pad(r.model_name, 30).c_str(), r.endpoint().c_str(),
               (st + std::string(9 > status.size() ? 9 - status.size() : 0, ' ')).c_str(), alive ? human_duration((double) (now - r.started_at)).c_str() : "-", reqs.c_str());
        if (!alive && p.get_bool("prune")) remove_record(r.name);
    }
    return 0;
}

static int stop_records(const std::vector<RunRecord> & recs, int timeout_s, bool quiet) {
    int failures = 0;
    for (auto & r : recs) {
        if (!pid_alive(r.pid)) { if (!quiet) printf("%s: not running (record removed)\n", r.name.c_str()); remove_record(r.name); continue; }
        if (kill((pid_t) r.pid, SIGTERM) != 0) { failures++; fprintf(stderr, "%s: kill failed\n", r.name.c_str()); continue; }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        while (pid_alive(r.pid) && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (pid_alive(r.pid)) { kill((pid_t) r.pid, SIGKILL); std::this_thread::sleep_for(std::chrono::milliseconds(200)); if (!quiet) printf("%s: killed (did not exit within %ds)\n", r.name.c_str(), timeout_s); }
        else if (!quiet) printf("%s: stopped\n", r.name.c_str());
        remove_record(r.name);
    }
    return failures ? 1 : 0;
}

int cmd_stop(const Args & args) {
    ArgParser p("iian stop", "iian stop <name|pid|all> [options]", "Stop one or all running servers (SIGTERM, then SIGKILL after --timeout).");
    p.positional("target", "Server name, pid, or 'all'");
    p.add("--timeout", "Seconds to wait for a graceful exit", "S", "10");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    auto recs = find_records(p.positional(0));
    if (recs.empty()) return fail("no server matches '" + p.positional(0) + "' (see `iian ps`)");
    return stop_records(recs, (int) p.get_int("timeout"), false);
}

int cmd_restart(const Args & args) {
    ArgParser p("iian restart", "iian restart <name>", "Stop a server and start it again with the arguments it was started with (as a daemon).");
    p.positional("name", "Server name");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    auto rec = load_record(p.positional(0));
    if (!rec) return fail("no server named '" + p.positional(0) + "'");
    RunRecord copy = *rec;
    stop_records({*rec}, 10, false);
    Args a = copy.args;
    bool has_detach = false;
    for (auto & x : a) if (x == "-d" || x == "--detach") has_detach = true;
    if (!has_detach) a.push_back("--detach");
    bool has_name = false;
    for (auto & x : a) if (x == "--name") has_name = true;
    if (!has_name) { a.push_back("--name"); a.push_back(copy.name); }
    return cmd_serve(a);
}

int cmd_logs(const Args & args) {
    ArgParser p("iian logs", "iian logs <name> [options]", "Print a server's log file.");
    p.positional("name", "Server name (see `iian ps`)");
    p.add("-f,--follow", "Keep printing new lines");
    p.add("-n,--lines", "Number of trailing lines to show", "N", "100");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    auto rec = load_record(p.positional(0));
    std::string path = rec ? rec->log_file : logs_dir() + "/" + p.positional(0) + ".log";
    std::ifstream in(path);
    if (!in) return fail("no log file for '" + p.positional(0) + "' (" + path + ")");
    std::vector<std::string> tail;
    const size_t n = (size_t) p.get_int("lines");
    std::string line;
    while (std::getline(in, line)) { tail.push_back(line); if (tail.size() > n) tail.erase(tail.begin()); }
    for (auto & l : tail) printf("%s\n", l.c_str());
    if (!p.get_bool("follow")) return 0;
    in.clear();
    while (true) {
        if (std::getline(in, line)) { printf("%s\n", line.c_str()); fflush(stdout); continue; }
        in.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if (rec && !pid_alive(rec->pid)) { printf("%s\n", dim("[server exited]").c_str()); return 0; }
    }
}

int cmd_status(const Args & args) {
    ArgParser p("iian status", "iian status <name|endpoint>", "Show health, model and engine statistics of a server.");
    p.positional("target", "Server name from `iian ps`, or an http://host:port endpoint");
    p.add("--api-key", "API key for a remote endpoint", "KEY");
    p.add("--json", "Print the raw JSON from the server");
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;
    std::string target = p.positional(0);
    std::unique_ptr<ServerClient> c;
    std::optional<RunRecord> rec = load_record(target);
    if (rec) c = std::make_unique<ServerClient>(*rec, std::chrono::seconds(3));
    else if (target.rfind("http", 0) == 0) c = std::make_unique<ServerClient>(target, p.has("api-key") ? p.get("api-key") : "", std::chrono::seconds(3));
    else return fail("unknown server '" + target + "' (see `iian ps`, or pass http://host:port)");

    auto health = c->get("/health");
    if (health.status == 0) return fail("cannot reach " + c->endpoint() + ": " + health.error);
    auto models = c->get("/v1/models");
    auto stats = c->get("/v1/engine/stats");
    if (p.get_bool("json")) {
        nlohmann::json j = {{"health", health.json()}, {"models", models.json()}};
        if (stats.ok()) j["engine"] = stats.json();
        printf("%s\n", j.dump(2).c_str());
        return 0;
    }
    printf("%s %s\n", bold("endpoint:").c_str(), c->endpoint().c_str());
    printf("%s %s\n", bold("health:  ").c_str(), health.json().is_object() && health.json().contains("status") ? health.json()["status"].get<std::string>().c_str() : std::to_string(health.status).c_str());
    if (rec) printf("%s %s (pid %lld, %s)\n", bold("process: ").c_str(), rec->name.c_str(), (long long) rec->pid, pid_alive(rec->pid) ? "alive" : "dead");
    if (models.ok()) {
        for (auto & m : models.json().value("data", nlohmann::json::array())) {
            printf("%s %s", bold("model:   ").c_str(), m.value("id", "?").c_str());
            if (m.contains("max_model_len")) printf("  (ctx %lld", (long long) m["max_model_len"].get<int64_t>());
            if (m.contains("ftype")) printf(", %s", m["ftype"].get<std::string>().c_str());
            if (m.contains("max_model_len")) printf(")");
            printf("\n");
        }
    }
    if (stats.ok()) {
        auto j = stats.json();
        if (j.contains("stats")) {
            auto & s = j["stats"];
            printf("%s running %d, waiting %d, kv cache %.1f%%, preemptions %lld\n", bold("engine:  ").c_str(), s.value("num_running", 0), s.value("num_waiting", 0),
                   100.0 * s.value("kv_usage", 0.0), (long long) s.value("preemptions", 0));
            printf("%s requests %lld finished / %lld total, prompt tokens %lld, generated %lld (%lld from prefix cache)\n", bold("totals:  ").c_str(),
                   (long long) s.value("requests_finished", 0), (long long) s.value("requests_total", 0), (long long) s.value("prompt_tokens", 0),
                   (long long) s.value("generation_tokens", 0), (long long) s.value("cached_prompt_tokens", 0));
            double q = s.value("prefix_cache_queries", 0.0), h = s.value("prefix_cache_hits", 0.0);
            printf("%s prompt %.1f tok/s, gen %.1f tok/s (last window), TTFT avg %.0f ms, prefix-cache hit %.1f%%\n", bold("perf:    ").c_str(),
                   s.value("prompt_tps", 0.0), s.value("generation_tps", 0.0), s.value("ttft_ms_avg", 0.0), q > 0 ? 100.0 * h / q : 0.0);
        }
        if (j.contains("uptime_s")) printf("%s %s\n", bold("uptime:  ").c_str(), human_duration(j["uptime_s"].get<double>()).c_str());
    } else if (stats.status == 403) {
        printf("%s\n", dim("engine stats need the server to run with --enable-admin").c_str());
    }
    return 0;
}

} // namespace iian::cli
