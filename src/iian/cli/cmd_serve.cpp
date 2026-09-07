// iian serve: load a model and expose the OpenAI-compatible API, in the foreground or as a daemon.
#include "args.h"
#include "commands.h"
#include "engine_options.h"
#include "model_resolve.h"
#include "paths.h"
#include "state.h"

#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/server/http_server.h"
#include "iian/server/model_registry.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace iian::cli {

namespace {

std::atomic<int> g_signal{0};
void on_signal(int sig) { g_signal.store(sig); }

bool daemonize(const std::string & log_file, std::string & err) {
    pid_t pid = fork();
    if (pid < 0) { err = std::string("fork failed: ") + strerror(errno); return false; }
    if (pid > 0) _exit(0);              // parent: the caller printed the summary already
    if (setsid() < 0) { err = "setsid failed"; return false; }
    pid = fork();                      // second fork: never re-acquire a controlling terminal
    if (pid < 0) { err = "fork failed"; return false; }
    if (pid > 0) _exit(0);
    umask(022);
    int fd = ::open(log_file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    int null = ::open("/dev/null", O_RDONLY);
    if (null >= 0) { dup2(null, STDIN_FILENO); ::close(null); }
    if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); ::close(fd); }
    return true;
}

} // namespace

int cmd_serve(const Args & args) {
    ArgParser p("iian serve", "iian serve <model> [options]",
                "Serve a GGUF model over an OpenAI-compatible HTTP API.\n"
                "<model> is a .gguf path, hf:org/repo[:quant] (downloaded to $IIAN_HOME/models), or a name from `iian ls`.");
    p.rest("model", "One or more models: GGUF path, hf:org/repo[:quant], or cached model name (first = default)");
    p.group("Server");
    p.add("--host", "Bind address", "HOST", "127.0.0.1");
    p.add("-p,--port", "Port", "PORT", "9931");
    p.add("--unix-socket", "Listen on a unix socket path instead of host:port", "PATH");
    p.add("--api-key", "Require 'Authorization: Bearer <key>' (or X-Api-Key) on API endpoints", "KEY");
    p.add("--served-model-name", "Model name reported by /v1/models (default: file name without .gguf)", "NAME");
    p.add("--chat-template", "Jinja chat template file (or literal) overriding the one in the GGUF", "FILE");
    p.add("--allow-request-chat-template", "Let requests override the chat template (chat_template field)");
    p.add("--enable-admin", "Enable /v1/engine/* control endpoints (stats, abort)");
    p.add("--threads-http", "HTTP worker threads (default: auto)", "N");
    p.add("-d,--detach", "Run in the background (daemon); logs go to $IIAN_HOME/logs/<name>.log");
    p.add("--name", "Server name for ps/stop/logs (default: derived from the model file)", "NAME");
    p.add("--config", "JSON file of flag -> value (CLI > env > config > defaults)", "FILE");
    p.group("Multi-model");
    p.add("--models-dir", "Serve every .gguf in this directory (loaded on first request)", "DIR");
    p.add("--models-max", "Keep at most N models loaded; least recently used ones are unloaded (0 = unlimited)", "N", "0");
    p.add("--lazy", "Do not preload the listed models; load each on its first request");
    add_model_flags(p);
    add_engine_flags(p);
    add_logging_flags(p);
    p.epilog("Examples:\n  iian serve models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8000\n"
             "  iian serve hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m -d --name qwen\n"
             "  iian serve model.gguf -c 8192 --max-num-seqs 64 --kv-dtype q8_0 -ngl 99");
    // config file must be loaded before parsing so precedence works
    for (size_t i = 0; i + 1 < args.size(); i++) if (args[i] == "--config") { std::string err; if (!load_config_file(p, args[i + 1], err)) return fail(err); }
    int rc = p.parse(args);
    if (rc) return rc == 2 ? 0 : rc;

    std::string err;
    if (!setup_logging(p, err)) return fail(err);

    // resolve every model spec (pulls hf: specs); names default to the file name without .gguf
    std::vector<std::pair<std::string, std::string>> models;   // (path, served name)
    try {
        for (const auto & spec : p.positionals()) {
            std::string path = resolve_model(spec, /*allow_server*/ false, /*pull*/ true).path;
            models.push_back({path, strip_gguf_ext(basename_of(path))});
        }
        if (p.has("models-dir")) {
            std::string dir = expand_user(p.get("models-dir"));
            if (!is_directory(dir)) return fail("--models-dir " + dir + " is not a directory");
            for (const auto & f : list_dir(dir)) {
                if (f.size() < 6 || f.compare(f.size() - 5, 5, ".gguf") != 0) continue;
                std::string path = dir + "/" + f;
                bool dup = false;
                for (auto & m : models) if (m.first == path) dup = true;
                if (!dup) models.push_back({path, strip_gguf_ext(f)});
            }
        }
    } catch (const std::exception & e) { return fail(e.what()); }
    if (models.empty()) return fail("no model given (pass a .gguf path, hf:org/repo[:quant], a cached name, or --models-dir)");
    if (p.has("served-model-name")) {
        if (models.size() != 1) return fail("--served-model-name applies to a single model; with several models the file names are used");
        models[0].second = p.get("served-model-name");
    }
    for (size_t i = 0; i < models.size(); i++) for (size_t j = i + 1; j < models.size(); j++)
        if (models[i].second == models[j].second) return fail("two models would be served under the same name '" + models[i].second + "'; rename one of the files");
    const std::string model_path = models[0].first;

    const std::string name = p.has("name") ? p.get("name") : default_server_name(model_path);
    if (auto existing = load_record(name); existing && pid_alive(existing->pid)) {
        return fail("a server named '" + name + "' is already running (pid " + std::to_string(existing->pid) + "); stop it with `iian stop " + name + "` or pick --name");
    }

    server::ServerConfig sc;
    sc.host = p.get("host");
    sc.port = (int) p.get_int("port");
    if (p.has("unix-socket")) sc.unix_socket = expand_user(p.get("unix-socket"));
    if (p.has("api-key")) sc.api_key = p.get("api-key");
    sc.model_name = models[0].second;
    sc.model_path = model_path;
    sc.models = models;
    sc.models_max = (int) p.get_int("models-max");
    sc.lazy_load = p.get_bool("lazy") || (p.has("models-dir") && p.positionals().empty());
    sc.allow_request_chat_template = p.get_bool("allow-request-chat-template");
    sc.enable_admin = p.get_bool("enable-admin");
    if (p.has("threads-http")) sc.n_threads_http = (int) p.get_int("threads-http");
    sc.build_info = build_info_string();
    if (p.has("chat-template")) {
        std::string t = p.get("chat-template");
        if (auto f = read_file(expand_user(t))) sc.chat_template = *f; else sc.chat_template = t;
    }

    const std::string log_file = logs_dir() + "/" + name + ".log";
    const bool detach = p.get_bool("detach");
    RunRecord rec;
    rec.name = name; rec.host = sc.host; rec.port = sc.port; rec.unix_socket = sc.unix_socket; rec.model_path = model_path;
    rec.model_name = models[0].second;
    for (size_t i = 1; i < models.size(); i++) rec.model_name += "," + models[i].second;
    rec.started_at = (int64_t) std::time(nullptr); rec.log_file = log_file; rec.args = args; rec.admin = sc.enable_admin;
    rec.detached = detach; rec.api_key = sc.api_key;

    if (detach) {
        printf("%s %s (model %s)\n  endpoint: %s\n  logs:     %s\n  use: iian ps | iian logs -f %s | iian stop %s\n",
               color("starting", 32).c_str(), bold(name).c_str(), model_path.c_str(),
               sc.unix_socket.empty() ? ("http://" + sc.host + ":" + std::to_string(sc.port)).c_str() : ("unix://" + sc.unix_socket).c_str(),
               log_file.c_str(), name.c_str(), name.c_str());
        fflush(stdout);
        if (!daemonize(log_file, err)) return fail(err);
        Logger::instance().set_color(false);
    } else if (!p.has("log-file")) {
        Logger::instance().set_file(log_file);   // foreground servers are still visible to `iian logs`
    }
    rec.pid = (int64_t) getpid();
    save_record(rec);

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    LOG_INF("cli", "%s starting server '%s' (pid %d), %zu model(s)%s", build_info_string().c_str(), name.c_str(), (int) getpid(), models.size(), sc.lazy_load ? " (lazy)" : "");
    server::HttpServer http(sc);
    const DeviceConfig dc = device_config_from_args(p);
    const EngineConfig ec = engine_config_from_args(p);
    http.set_loader([dc, ec](server::ModelSlot & slot) {
        int last_pct = -1;
        slot.model = ModelLoader::load(slot.path, dc, [&](float f) {
            int pct = (int) (f * 100);
            if (pct / 25 != last_pct / 25) { LOG_INF("cli", "[%s] loading weights... %d%%", slot.name.c_str(), pct); last_pct = pct; }
            return g_signal.load() == 0;
        });
        slot.engine = std::make_unique<Engine>(slot.model, ec);
        slot.engine->start();
        slot.ctx.attach(slot.engine.get());
        LOG_INF("cli", "[%s] %s | backends: %s", slot.name.c_str(), model_banner(*slot.model).c_str(), slot.engine->backend_summary().c_str());
    });
    if (!http.start(err)) { remove_record(name); return fail("cannot start HTTP server: " + err); }
    LOG_INF("cli", "listening on %s%s", http.endpoint().c_str(), sc.lazy_load ? "" : " (models loading, /health reports 'loading' until the first one is ready)");
    if (!sc.lazy_load) {
        auto t0 = std::chrono::steady_clock::now();
        http.wait_preload();
        size_t ok = http.registry().n_loaded();
        if (ok == 0) { LOG_ERR("cli", "no model could be loaded"); http.stop(); remove_record(name); return 1; }
        LOG_INF("cli", "ready in %.1fs: %s  (%zu/%zu models loaded)", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), http.endpoint().c_str(), ok, models.size());
    } else {
        LOG_INF("cli", "ready: %s  (%zu models, loaded on first use)", http.endpoint().c_str(), models.size());
    }

    while (g_signal.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    LOG_INF("cli", "received %s, shutting down", g_signal.load() == SIGINT ? "SIGINT" : "SIGTERM");
    http.stop();
    remove_record(name);
    LOG_INF("cli", "bye");
    return 0;
}

} // namespace iian::cli
