#include "paths.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace iian::cli {

std::string expand_user(const std::string & path) {
    if (path.rfind("~/", 0) == 0 || path == "~") {
        const char * home = getenv("HOME");
        return std::string(home ? home : "") + path.substr(1);
    }
    return path;
}

std::string iian_home() {
    static std::string home = [] {
        const char * env = getenv("IIAN_HOME");
        std::string h = env && *env ? expand_user(env) : expand_user("~/.iian");
        ensure_dir(h);
        return h;
    }();
    return home;
}

std::string run_dir()    { std::string d = iian_home() + "/run";    ensure_dir(d); return d; }
std::string logs_dir()   { std::string d = iian_home() + "/logs";   ensure_dir(d); return d; }
std::string models_dir() { std::string d = iian_home() + "/models"; ensure_dir(d); return d; }

bool ensure_dir(const std::string & path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) return true;
    return fs::create_directories(path, ec);
}
bool file_exists(const std::string & path) { struct stat st{}; return ::stat(path.c_str(), &st) == 0; }
bool is_directory(const std::string & path) { std::error_code ec; return fs::is_directory(path, ec); }
std::optional<uint64_t> file_size(const std::string & path) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return std::nullopt;
    return (uint64_t) st.st_size;
}

std::optional<std::string> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool write_file_atomic(const std::string & path, const std::string & content, int mode) {
    std::string tmp = path + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << content;
        if (!f) return false;
    }
    ::chmod(tmp.c_str(), (mode_t) mode);
    if (::rename(tmp.c_str(), path.c_str()) != 0) { ::unlink(tmp.c_str()); return false; }
    return true;
}

std::string basename_of(const std::string & path) { return fs::path(path).filename().string(); }
std::string strip_gguf_ext(const std::string & name) {
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) return name.substr(0, name.size() - 5);
    return name;
}

std::vector<std::string> list_dir(const std::string & path) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(path, ec)) out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace iian::cli
