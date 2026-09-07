#pragma once
// Model specs: a GGUF path, "hf:org/repo[:quant]" (downloaded into $IIAN_HOME/models), or a cached name.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace iian::cli {

struct HfSpec {
    std::string org, repo, quant;     // quant may be empty; may also be an explicit filename ending in .gguf
    std::string repo_id() const { return org + "/" + repo; }
};
std::optional<HfSpec> parse_hf_spec(const std::string & spec);   // "hf:org/repo[:quant]" or "org/repo[:quant]"
std::string hf_cache_filename(const HfSpec & s, const std::string & file);   // org__repo__file

struct PullOptions {
    std::string dir;              // destination dir (default models_dir())
    bool quiet = false;
    bool force = false;           // re-download even if present
};
// Resolves the file in the repo, downloads it (with progress on stderr) and returns the local path.
// Throws std::runtime_error with an actionable message.
std::string hf_pull(const HfSpec & spec, const PullOptions & opts);
// Direct URL download (http/https) into dir; returns the local path.
std::string url_pull(const std::string & url, const PullOptions & opts);

struct ResolvedModel {
    std::string path;             // local GGUF path ("" if the spec names a running server)
    std::string server_name;      // set if the spec matched a run record
};
// Resolve a model spec: server name (if allow_server) > existing path > hf: spec (pull) > cached model name.
ResolvedModel resolve_model(const std::string & spec, bool allow_server, bool pull_if_missing, const std::string & extra_dir = "");

struct LocalModel { std::string path, name; uint64_t bytes = 0; };
std::vector<LocalModel> list_local_models(const std::vector<std::string> & dirs);

} // namespace iian::cli
