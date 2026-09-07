#pragma once
// Filesystem layout: $IIAN_HOME (default ~/.iian) / {run, logs, models, config.json}
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace iian::cli {

std::string iian_home();      // creates it if missing
std::string run_dir();
std::string logs_dir();
std::string models_dir();
std::string expand_user(const std::string & path);   // "~/x" -> "$HOME/x"
bool ensure_dir(const std::string & path);
bool file_exists(const std::string & path);
bool is_directory(const std::string & path);
std::optional<uint64_t> file_size(const std::string & path);
std::optional<std::string> read_file(const std::string & path);
bool write_file_atomic(const std::string & path, const std::string & content, int mode = 0644);
std::string basename_of(const std::string & path);
std::string strip_gguf_ext(const std::string & name);
std::vector<std::string> list_dir(const std::string & path);   // sorted entry names

} // namespace iian::cli
