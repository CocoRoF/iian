#pragma once
// Tiny dependency-free argument parser with grouped, sectioned help, "did you mean" suggestions,
// and layered values (CLI > environment IIAN_<FLAG> > config file > default).
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace iian::cli {

struct Flag {
    std::string name;          // long name without dashes, e.g. "max-model-len"
    std::string short_name;    // e.g. "p" (may be empty)
    std::string value_name;    // e.g. "N"; empty => boolean switch
    std::string help;
    std::string group;
    std::string default_value; // shown in help; also the fallback
    bool        repeatable = false;
    bool        is_bool() const { return value_name.empty(); }
};

struct Positional {
    std::string name;
    std::string help;
    bool        required = true;
    bool        rest = false;   // absorbs all remaining positionals
};

class ArgParser {
public:
    ArgParser(std::string prog, std::string usage, std::string description);

    // Subsequent add() calls land in this help section.
    ArgParser & group(const std::string & name);
    // spec: "--name", "-s,--name". A non-empty value_name makes it a valued flag, otherwise a switch.
    ArgParser & add(const std::string & spec, const std::string & help, const std::string & value_name = "", const std::string & default_value = "");
    ArgParser & add_repeatable(const std::string & spec, const std::string & help, const std::string & value_name);
    ArgParser & positional(const std::string & name, const std::string & help, bool required = true);
    ArgParser & rest(const std::string & name, const std::string & help);
    ArgParser & epilog(const std::string & text);
    // Environment prefix (default "IIAN_"): --max-model-len <- IIAN_MAX_MODEL_LEN.
    ArgParser & env_prefix(const std::string & p) { env_prefix_ = p; return *this; }
    // Flat JSON object of flag-name -> value (values may be strings, numbers or booleans).
    void set_config(const nlohmann::json & cfg);

    // Parses argv[1..]; on --help prints help and returns 2; on error prints the message and returns 1; else 0.
    int parse(int argc, char ** argv);
    int parse(const std::vector<std::string> & args);

    bool has(const std::string & name) const;          // set explicitly (CLI / env / config)
    bool given_on_cli(const std::string & name) const;
    std::string get(const std::string & name) const;   // resolved value or default
    std::optional<std::string> get_opt(const std::string & name) const;   // nullopt if unset and no default
    bool get_bool(const std::string & name) const;
    int64_t get_int(const std::string & name) const;
    double get_double(const std::string & name) const;
    std::vector<std::string> get_all(const std::string & name) const;      // repeatable flags
    const std::vector<std::string> & positionals() const { return positionals_; }
    std::string positional(size_t i, const std::string & def = "") const { return i < positionals_.size() ? positionals_[i] : def; }
    const std::vector<std::string> & raw_args() const { return raw_; }

    void print_help(FILE * f = stdout) const;
    std::string usage_line() const;
    const Flag * find(const std::string & name) const;
    std::string suggest(const std::string & bad) const;   // nearest flag by edit distance or ""

private:
    std::string prog_, usage_, description_, epilog_, current_group_ = "Options", env_prefix_ = "IIAN_";
    std::vector<Flag> flags_;
    std::vector<std::string> group_order_;
    std::vector<Positional> pos_spec_;
    std::map<std::string, std::vector<std::string>> cli_values_;
    nlohmann::json config_ = nlohmann::json::object();
    std::vector<std::string> positionals_;
    std::vector<std::string> raw_;
    std::optional<std::string> env_value(const std::string & name) const;
    std::optional<std::string> config_value(const std::string & name) const;
};

size_t levenshtein(const std::string & a, const std::string & b);

// Formatting helpers shared by the commands.
std::string human_bytes(uint64_t bytes);
std::string human_duration(double seconds);
std::string pad(const std::string & s, size_t w, bool left = true);
bool is_tty(FILE * f);
// ANSI helpers (no-ops when stdout is not a TTY or NO_COLOR is set)
std::string bold(const std::string & s);
std::string dim(const std::string & s);
std::string color(const std::string & s, int code);   // 31 red, 32 green, 33 yellow, 36 cyan

} // namespace iian::cli
