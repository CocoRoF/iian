#include "model_resolve.h"

#include "args.h"
#include "paths.h"
#include "state.h"

#include "cpp-httplib/httplib.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace iian::cli {

using json = nlohmann::json;

std::optional<HfSpec> parse_hf_spec(const std::string & spec_in) {
    std::string spec = spec_in;
    bool prefixed = false;
    if (spec.rfind("hf:", 0) == 0) { spec = spec.substr(3); prefixed = true; }
    else if (spec.rfind("hf://", 0) == 0) { spec = spec.substr(5); prefixed = true; }
    if (spec.rfind("https://huggingface.co/", 0) == 0) { spec = spec.substr(23); prefixed = true; }
    if (!prefixed && (file_exists(spec_in) || spec.find('/') == std::string::npos)) return std::nullopt;
    HfSpec s;
    size_t slash = spec.find('/');
    if (slash == std::string::npos) return std::nullopt;
    s.org = spec.substr(0, slash);
    std::string rest = spec.substr(slash + 1);
    size_t colon = rest.find(':');
    if (colon != std::string::npos) { s.quant = rest.substr(colon + 1); rest = rest.substr(0, colon); }
    // allow "org/repo/file.gguf"
    size_t slash2 = rest.find('/');
    if (slash2 != std::string::npos) { s.quant = rest.substr(slash2 + 1); rest = rest.substr(0, slash2); }
    s.repo = rest;
    if (s.org.empty() || s.repo.empty()) return std::nullopt;
    if (!prefixed && s.org.find('.') != std::string::npos) return std::nullopt;   // looks like a path
    return s;
}

std::string hf_cache_filename(const HfSpec & s, const std::string & file) {
    return s.org + "__" + s.repo + "__" + basename_of(file);
}

static std::string lower(std::string s) { std::transform(s.begin(), s.end(), s.begin(), ::tolower); return s; }

// ---- progress bar ---------------------------------------------------------------------------

struct Progress {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(), t_last = t0;
    uint64_t last_bytes = 0;
    bool quiet = false;
    void update(uint64_t cur, uint64_t total, bool final = false) {
        if (quiet) return;
        auto now = std::chrono::steady_clock::now();
        if (!final && std::chrono::duration<double>(now - t_last).count() < 0.1) return;
        double dt = std::chrono::duration<double>(now - t0).count();
        double rate = dt > 0 ? cur / dt / 1048576.0 : 0.0;
        t_last = now; last_bytes = cur;
        if (total > 0) {
            double pct = 100.0 * cur / total;
            int width = 30, filled = (int) (width * cur / total);
            std::string bar(filled, '#'); bar += std::string(width - filled, '-');
            fprintf(stderr, "\r  [%s] %5.1f%%  %s / %s  %.1f MB/s   ", bar.c_str(), pct, human_bytes(cur).c_str(), human_bytes(total).c_str(), rate);
        } else {
            fprintf(stderr, "\r  %s  %.1f MB/s   ", human_bytes(cur).c_str(), rate);
        }
        if (final) fprintf(stderr, "\n");
        fflush(stderr);
    }
};

// ---- HTTP(S) download -----------------------------------------------------------------------

static bool have_curl() { return system("command -v curl >/dev/null 2>&1") == 0; }

static std::string shell_quote(const std::string & s) {
    std::string o = "'";
    for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; }
    return o + "'";
}

static void download_with_curl(const std::string & url, const std::string & dst, bool quiet) {
    std::string cmd = "curl -L --fail --retry 3 -C - " + std::string(quiet ? "-sS " : "--progress-bar ") +
                      "-o " + shell_quote(dst) + " " + shell_quote(url);
    const char * tok = getenv("HF_TOKEN");
    if (tok && *tok) cmd += " -H " + shell_quote("Authorization: Bearer " + std::string(tok));
    int rc = system(cmd.c_str());
    if (rc != 0) throw std::runtime_error("curl failed (exit " + std::to_string(rc) + ") downloading " + url);
}

// Downloads url -> dst (resumable via Range on a ".part" file). Uses httplib+OpenSSL when built with TLS.
static void download(const std::string & url, const std::string & dst, bool quiet) {
    std::string part = dst + ".part";
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (url.rfind("https://", 0) == 0) {
        if (!have_curl()) throw std::runtime_error("this build has no TLS support and `curl` was not found; install curl or download the file manually to " + dst);
        download_with_curl(url, part, quiet);
        if (::rename(part.c_str(), dst.c_str()) != 0) throw std::runtime_error("failed to move " + part + " into place");
        return;
    }
#endif
    // split URL into scheme://host and path
    size_t p = url.find("://");
    if (p == std::string::npos) throw std::runtime_error("bad URL: " + url);
    size_t path_start = url.find('/', p + 3);
    std::string base = path_start == std::string::npos ? url : url.substr(0, path_start);
    std::string path = path_start == std::string::npos ? "/" : url.substr(path_start);

    std::string current_url = url;
    for (int redirects = 0; redirects < 10; redirects++) {
        p = current_url.find("://");
        path_start = current_url.find('/', p + 3);
        base = path_start == std::string::npos ? current_url : current_url.substr(0, path_start);
        path = path_start == std::string::npos ? "/" : current_url.substr(path_start);

        httplib::Client cli(base);
        cli.set_connection_timeout(std::chrono::seconds(20));
        cli.set_read_timeout(std::chrono::seconds(60));
        cli.set_follow_location(false);   // handled manually so Range + auth survive cross-host redirects
        httplib::Headers headers{{"User-Agent", "iian/0.1"}};
        const char * tok = getenv("HF_TOKEN");
        if (tok && *tok && base.find("huggingface.co") != std::string::npos) headers.emplace("Authorization", std::string("Bearer ") + tok);
        uint64_t have = file_size(part).value_or(0);
        if (have > 0) headers.emplace("Range", "bytes=" + std::to_string(have) + "-");

        std::ofstream out;
        uint64_t total = 0, cur = have;
        Progress prog; prog.quiet = quiet;
        int status = 0;
        std::string location;
        bool restarted = false;
        auto res = cli.Get(path, headers,
            [&](const httplib::Response & r) {
                status = r.status;
                if (status >= 300 && status < 400) { location = r.get_header_value("Location"); return false; }
                if (status == 200 && have > 0) { restarted = true; cur = 0; }     // server ignored Range
                if (status == 416) return false;
                if (status >= 400) return false;
                out.open(part, std::ios::binary | (restarted || have == 0 ? std::ios::trunc : std::ios::app));
                if (!out) throw std::runtime_error("cannot write " + part);
                std::string cl = r.get_header_value("Content-Length");
                total = cl.empty() ? 0 : cur + std::stoull(cl);
                return true;
            },
            [&](const char * data, size_t len) {
                out.write(data, (std::streamsize) len);
                cur += len;
                prog.update(cur, total);
                return true;
            });
        if (status >= 300 && status < 400 && !location.empty()) {
            if (location[0] == '/') location = base + location;
            current_url = location;
            continue;
        }
        if (status == 416) { if (::rename(part.c_str(), dst.c_str()) != 0) throw std::runtime_error("failed to move " + part); return; }   // already complete
        if (status == 401 || status == 403) throw std::runtime_error("HTTP " + std::to_string(status) + " for " + current_url + " (gated/private repo? set HF_TOKEN=<your token>)");
        if (status == 404) throw std::runtime_error("HTTP 404: " + current_url + " does not exist");
        if (status >= 400) throw std::runtime_error("HTTP " + std::to_string(status) + " downloading " + current_url);
        if (!res && status == 0) throw std::runtime_error("download failed: " + httplib::to_string(res.error()) + " (" + current_url + ")");
        if (!res) throw std::runtime_error("download interrupted: " + httplib::to_string(res.error()) + "; re-run to resume");
        out.close();
        prog.update(cur, total, true);
        if (total > 0 && cur != total) throw std::runtime_error("short download (" + std::to_string(cur) + " of " + std::to_string(total) + " bytes); re-run to resume");
        if (::rename(part.c_str(), dst.c_str()) != 0) throw std::runtime_error("failed to move " + part + " into place");
        return;
    }
    throw std::runtime_error("too many redirects for " + url);
}

static json hf_api_get(const std::string & path) {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (!have_curl()) throw std::runtime_error("this build has no TLS support and `curl` was not found; cannot query huggingface.co");
    std::string tmp = iian_home() + "/.hf_api.json";
    std::string cmd = "curl -sSL --fail -o " + shell_quote(tmp) + " " + shell_quote("https://huggingface.co" + path);
    const char * tok = getenv("HF_TOKEN");
    if (tok && *tok) cmd += " -H " + shell_quote("Authorization: Bearer " + std::string(tok));
    if (system(cmd.c_str()) != 0) throw std::runtime_error("failed to query https://huggingface.co" + path);
    auto s = read_file(tmp);
    ::unlink(tmp.c_str());
    return json::parse(s.value_or("null"), nullptr, false);
#else
    httplib::Client cli("https://huggingface.co");
    cli.set_connection_timeout(std::chrono::seconds(20));
    cli.set_read_timeout(std::chrono::seconds(30));
    cli.set_follow_location(true);
    httplib::Headers headers{{"User-Agent", "iian/0.1"}};
    const char * tok = getenv("HF_TOKEN");
    if (tok && *tok) headers.emplace("Authorization", std::string("Bearer ") + tok);
    auto r = cli.Get(path, headers);
    if (!r) throw std::runtime_error("cannot reach huggingface.co: " + httplib::to_string(r.error()) + " (check your network / proxy)");
    if (r->status == 401 || r->status == 403) throw std::runtime_error("huggingface.co returned " + std::to_string(r->status) + " for " + path + " (gated/private repo? set HF_TOKEN=<your token>)");
    if (r->status == 404) throw std::runtime_error("repository not found on huggingface.co: " + path.substr(path.rfind("/models/") == std::string::npos ? 0 : 12));
    if (r->status != 200) throw std::runtime_error("huggingface.co returned HTTP " + std::to_string(r->status) + " for " + path);
    return json::parse(r->body, nullptr, false);
#endif
}

static std::vector<std::string> hf_list_ggufs(const HfSpec & spec) {
    json j = hf_api_get("/api/models/" + spec.repo_id());
    std::vector<std::string> files;
    if (j.is_object() && j.contains("siblings") && j["siblings"].is_array()) {
        for (const auto & s : j["siblings"]) {
            std::string f = s.value("rfilename", "");
            if (f.size() > 5 && lower(f).compare(f.size() - 5, 5, ".gguf") == 0) files.push_back(f);
        }
    }
    if (files.empty()) throw std::runtime_error("no .gguf files found in huggingface.co/" + spec.repo_id() + " (is it a GGUF repo? try e.g. hf:bartowski/<model>-GGUF)");
    std::sort(files.begin(), files.end());
    return files;
}

static std::string choose_gguf(const HfSpec & spec, const std::vector<std::string> & files) {
    auto is_shard = [](const std::string & f) { return lower(f).find("-of-") != std::string::npos && lower(f).find("00001-of") == std::string::npos; };
    auto match_tag = [&](const std::string & tag) -> std::string {
        std::string t = lower(tag);
        for (const auto & f : files) if (lower(f) == t || lower(basename_of(f)) == t) return f;
        for (const auto & f : files) if (!is_shard(f) && lower(f).find(t) != std::string::npos) return f;
        return "";
    };
    if (!spec.quant.empty()) {
        std::string f = match_tag(spec.quant);
        if (f.empty()) {
            std::string avail;
            for (const auto & x : files) avail += "\n    " + x;
            throw std::runtime_error("no file matching quant '" + spec.quant + "' in " + spec.repo_id() + ". Available:" + avail);
        }
        return f;
    }
    for (const char * pref : {"q4_k_m", "q8_0", "f16", "bf16", "q6_k", "q5_k_m", "q4_0", "f32"}) {
        std::string f = match_tag(pref);
        if (!f.empty()) return f;
    }
    for (const auto & f : files) if (!is_shard(f)) return f;
    std::string avail;
    for (const auto & x : files) avail += "\n    " + x;
    throw std::runtime_error("could not pick a default quant in " + spec.repo_id() + "; specify one with hf:" + spec.repo_id() + ":<quant>. Available:" + avail);
}

std::string hf_pull(const HfSpec & spec, const PullOptions & opts) {
    std::string dir = opts.dir.empty() ? models_dir() : expand_user(opts.dir);
    ensure_dir(dir);
    // fast path: a cached file for this repo+quant already exists
    if (!opts.force && !spec.quant.empty()) {
        std::string prefix = lower(spec.org + "__" + spec.repo + "__");
        for (const auto & f : list_dir(dir)) {
            std::string lf = lower(f);
            if (lf.rfind(prefix, 0) == 0 && lf.find(lower(spec.quant)) != std::string::npos && lf.size() > 5 && lf.compare(lf.size() - 5, 5, ".gguf") == 0)
                return dir + "/" + f;
        }
    }
    if (!opts.quiet) fprintf(stderr, "Resolving hf:%s%s ...\n", spec.repo_id().c_str(), spec.quant.empty() ? "" : (":" + spec.quant).c_str());
    std::vector<std::string> files = hf_list_ggufs(spec);
    std::string file = choose_gguf(spec, files);
    if (lower(file).find("00001-of-") != std::string::npos) throw std::runtime_error("'" + file + "' is a multi-part GGUF; sharded models are not supported yet. Pick a different quant.");
    std::string dst = dir + "/" + hf_cache_filename(spec, file);
    if (!opts.force && file_exists(dst)) {
        if (!opts.quiet) fprintf(stderr, "Already cached: %s\n", dst.c_str());
        return dst;
    }
    std::string url = "https://huggingface.co/" + spec.repo_id() + "/resolve/main/" + file;
    if (!opts.quiet) fprintf(stderr, "Downloading %s\n  -> %s\n", url.c_str(), dst.c_str());
    download(url, dst, opts.quiet);
    if (!opts.quiet) fprintf(stderr, "Saved %s (%s)\n", dst.c_str(), human_bytes(file_size(dst).value_or(0)).c_str());
    return dst;
}

std::string url_pull(const std::string & url, const PullOptions & opts) {
    std::string dir = opts.dir.empty() ? models_dir() : expand_user(opts.dir);
    ensure_dir(dir);
    std::string name = basename_of(url);
    size_t q = name.find('?');
    if (q != std::string::npos) name = name.substr(0, q);
    if (name.empty()) throw std::runtime_error("cannot derive a file name from " + url);
    std::string dst = dir + "/" + name;
    if (!opts.force && file_exists(dst)) { if (!opts.quiet) fprintf(stderr, "Already cached: %s\n", dst.c_str()); return dst; }
    if (!opts.quiet) fprintf(stderr, "Downloading %s\n  -> %s\n", url.c_str(), dst.c_str());
    download(url, dst, opts.quiet);
    return dst;
}

ResolvedModel resolve_model(const std::string & spec, bool allow_server, bool pull_if_missing, const std::string & extra_dir) {
    ResolvedModel r;
    if (allow_server) {
        if (auto rec = load_record(spec); rec && pid_alive(rec->pid)) { r.server_name = spec; return r; }
    }
    if (file_exists(spec) && !is_directory(spec)) { r.path = spec; return r; }
    if (spec.rfind("http://", 0) == 0 || spec.rfind("https://", 0) == 0) {
        if (auto hs = parse_hf_spec(spec)) { if (!pull_if_missing) throw std::runtime_error("model not cached: " + spec); r.path = hf_pull(*hs, {}); return r; }
        if (!pull_if_missing) throw std::runtime_error("model not cached: " + spec);
        r.path = url_pull(spec, {});
        return r;
    }
    if (auto hs = parse_hf_spec(spec)) {
        if (!pull_if_missing) throw std::runtime_error("model not cached: " + spec + " (run `iian pull " + spec + "`)");
        r.path = hf_pull(*hs, {});
        return r;
    }
    // cached model by name (with or without .gguf)
    std::vector<std::string> dirs{models_dir()};
    if (!extra_dir.empty()) dirs.push_back(expand_user(extra_dir));
    for (const auto & d : dirs) {
        for (const auto & cand : {d + "/" + spec, d + "/" + spec + ".gguf"}) if (file_exists(cand)) { r.path = cand; return r; }
        for (const auto & f : list_dir(d)) if (lower(strip_gguf_ext(f)) == lower(spec) || lower(f).find(lower(spec)) != std::string::npos) {
            if (f.size() > 5 && lower(f).compare(f.size() - 5, 5, ".gguf") == 0) { r.path = d + "/" + f; return r; }
        }
    }
    std::string msg = "model not found: '" + spec + "'. Expected a .gguf path, hf:org/repo[:quant], or a name from `iian ls`";
    if (allow_server) msg += " / a running server from `iian ps`";
    throw std::runtime_error(msg + ".");
}

std::vector<LocalModel> list_local_models(const std::vector<std::string> & dirs) {
    std::vector<LocalModel> out;
    for (const auto & d0 : dirs) {
        std::string d = expand_user(d0);
        for (const auto & f : list_dir(d)) {
            if (f.size() < 6 || lower(f).compare(f.size() - 5, 5, ".gguf") != 0) continue;
            LocalModel m;
            m.path = d + "/" + f;
            m.name = strip_gguf_ext(f);
            m.bytes = file_size(m.path).value_or(0);
            out.push_back(m);
        }
    }
    return out;
}

} // namespace iian::cli
