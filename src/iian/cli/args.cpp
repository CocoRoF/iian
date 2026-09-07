#include "args.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace iian::cli {

ArgParser::ArgParser(std::string prog, std::string usage, std::string description)
    : prog_(std::move(prog)), usage_(std::move(usage)), description_(std::move(description)) {
    add("-h,--help", "Show this help and exit");
}

ArgParser & ArgParser::group(const std::string & name) {
    current_group_ = name;
    if (std::find(group_order_.begin(), group_order_.end(), name) == group_order_.end()) group_order_.push_back(name);
    return *this;
}

ArgParser & ArgParser::add(const std::string & spec, const std::string & help, const std::string & value_name, const std::string & default_value) {
    Flag f;
    f.help = help; f.value_name = value_name; f.default_value = default_value; f.group = current_group_;
    if (std::find(group_order_.begin(), group_order_.end(), current_group_) == group_order_.end()) group_order_.push_back(current_group_);
    size_t start = 0;
    while (start <= spec.size()) {
        size_t comma = spec.find(',', start);
        std::string part = spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (part.rfind("--", 0) == 0) f.name = part.substr(2);
        else if (part.rfind("-", 0) == 0) f.short_name = part.substr(1);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    flags_.push_back(std::move(f));
    return *this;
}

ArgParser & ArgParser::add_repeatable(const std::string & spec, const std::string & help, const std::string & value_name) {
    add(spec, help, value_name);
    flags_.back().repeatable = true;
    return *this;
}

ArgParser & ArgParser::positional(const std::string & name, const std::string & help, bool required) {
    pos_spec_.push_back({name, help, required, false});
    return *this;
}
ArgParser & ArgParser::rest(const std::string & name, const std::string & help) {
    pos_spec_.push_back({name, help, false, true});
    return *this;
}
ArgParser & ArgParser::epilog(const std::string & text) { epilog_ = text; return *this; }
void ArgParser::set_config(const nlohmann::json & cfg) { if (cfg.is_object()) config_ = cfg; }

const Flag * ArgParser::find(const std::string & name) const {
    for (const auto & f : flags_) if (f.name == name) return &f;
    return nullptr;
}

size_t levenshtein(const std::string & a, const std::string & b) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); j++) prev[j] = j;
    for (size_t i = 1; i <= a.size(); i++) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); j++)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

std::string ArgParser::suggest(const std::string & bad) const {
    std::string best; size_t best_d = 1000;
    for (const auto & f : flags_) {
        size_t d = levenshtein(bad, f.name);
        if (d < best_d) { best_d = d; best = f.name; }
    }
    return best_d <= std::max<size_t>(2, bad.size() / 3) ? best : "";
}

int ArgParser::parse(int argc, char ** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) args.emplace_back(argv[i]);
    return parse(args);
}

int ArgParser::parse(const std::vector<std::string> & args) {
    raw_ = args;
    bool only_positionals = false;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string & a = args[i];
        if (only_positionals || a == "-" || a.empty() || a[0] != '-' || (a.size() > 1 && isdigit((unsigned char) a[1]) && !find(a.substr(1)))) {
            positionals_.push_back(a);
            continue;
        }
        if (a == "--") { only_positionals = true; continue; }
        std::string name, value; bool has_value = false;
        const Flag * f = nullptr;
        if (a.rfind("--", 0) == 0) {
            name = a.substr(2);
            size_t eq = name.find('=');
            if (eq != std::string::npos) { value = name.substr(eq + 1); name = name.substr(0, eq); has_value = true; }
            // --no-<flag> for boolean flags defined as --<flag>, and --<flag> for flags defined as --no-<flag>
            f = find(name);
            bool negate = false;
            if (!f && name.rfind("no-", 0) == 0) { f = find(name.substr(3)); if (f && f->is_bool()) { negate = true; name = f->name; } else f = nullptr; }
            if (!f) {
                std::string s = suggest(name);
                fprintf(stderr, "%s: unknown option '--%s'%s\n", prog_.c_str(), name.c_str(), s.empty() ? "" : (" (did you mean '--" + s + "'?)").c_str());
                fprintf(stderr, "Run '%s --help' for the list of options.\n", prog_.c_str());
                return 1;
            }
            if (negate) { cli_values_[name] = {"false"}; continue; }
        } else {
            std::string sn = a.substr(1);
            // -n5 / -n 5
            for (const auto & fl : flags_) if (!fl.short_name.empty() && sn.rfind(fl.short_name, 0) == 0 && (sn.size() == fl.short_name.size() || !fl.is_bool())) { f = &fl; break; }
            if (!f) {
                fprintf(stderr, "%s: unknown option '%s'\nRun '%s --help' for the list of options.\n", prog_.c_str(), a.c_str(), prog_.c_str());
                return 1;
            }
            name = f->name;
            if (sn.size() > f->short_name.size()) { value = sn.substr(f->short_name.size()); has_value = true; if (!value.empty() && value[0] == '=') value.erase(0, 1); }
        }
        if (name == "help") { print_help(stdout); return 2; }
        if (f->is_bool()) {
            if (has_value) cli_values_[name] = {value};
            else cli_values_[name] = {"true"};
            continue;
        }
        if (!has_value) {
            if (i + 1 >= args.size()) { fprintf(stderr, "%s: option '--%s' requires a value <%s>\n", prog_.c_str(), name.c_str(), f->value_name.c_str()); return 1; }
            value = args[++i];
        }
        if (f->repeatable) cli_values_[name].push_back(value);
        else cli_values_[name] = {value};
    }
    size_t required = 0;
    for (const auto & p : pos_spec_) if (p.required) required++;
    if (positionals_.size() < required) {
        fprintf(stderr, "%s: missing required argument <%s>\n\nUsage: %s\n", prog_.c_str(), pos_spec_[positionals_.size()].name.c_str(), usage_line().c_str());
        return 1;
    }
    bool has_rest = !pos_spec_.empty() && pos_spec_.back().rest;
    if (!has_rest && positionals_.size() > pos_spec_.size()) {
        fprintf(stderr, "%s: unexpected argument '%s'\n\nUsage: %s\n", prog_.c_str(), positionals_[pos_spec_.size()].c_str(), usage_line().c_str());
        return 1;
    }
    return 0;
}

std::optional<std::string> ArgParser::env_value(const std::string & name) const {
    std::string key = env_prefix_;
    for (char c : name) key += c == '-' ? '_' : (char) toupper((unsigned char) c);
    const char * v = getenv(key.c_str());
    if (!v) return std::nullopt;
    return std::string(v);
}

std::optional<std::string> ArgParser::config_value(const std::string & name) const {
    auto it = config_.find(name);
    if (it == config_.end()) { std::string alt = name; std::replace(alt.begin(), alt.end(), '-', '_'); it = config_.find(alt); }
    if (it == config_.end() || it->is_null()) return std::nullopt;
    if (it->is_string()) return it->get<std::string>();
    if (it->is_boolean()) return std::string(it->get<bool>() ? "true" : "false");
    if (it->is_number_integer()) return std::to_string(it->get<int64_t>());
    if (it->is_number_float()) { char b[64]; snprintf(b, sizeof b, "%g", it->get<double>()); return std::string(b); }
    return it->dump();
}

bool ArgParser::given_on_cli(const std::string & name) const { return cli_values_.count(name) > 0; }
bool ArgParser::has(const std::string & name) const { return given_on_cli(name) || env_value(name) || config_value(name); }

std::optional<std::string> ArgParser::get_opt(const std::string & name) const {
    auto it = cli_values_.find(name);
    if (it != cli_values_.end() && !it->second.empty()) return it->second.back();
    if (auto e = env_value(name)) return e;
    if (auto c = config_value(name)) return c;
    const Flag * f = find(name);
    if (f && !f->default_value.empty()) return f->default_value;
    if (f && f->is_bool()) return std::string("false");
    return std::nullopt;
}
std::string ArgParser::get(const std::string & name) const { return get_opt(name).value_or(""); }

bool ArgParser::get_bool(const std::string & name) const {
    std::string v = get(name);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return v == "1" || v == "true" || v == "yes" || v == "on";
}
int64_t ArgParser::get_int(const std::string & name) const {
    std::string v = get(name);
    if (v.empty()) return 0;
    char * end = nullptr;
    long long r = strtoll(v.c_str(), &end, 0);
    if (end == v.c_str() || *end != '\0') {
        fprintf(stderr, "%s: option '--%s' expects an integer, got '%s'\n", prog_.c_str(), name.c_str(), v.c_str());
        exit(1);
    }
    return r;
}
double ArgParser::get_double(const std::string & name) const {
    std::string v = get(name);
    if (v.empty()) return 0.0;
    char * end = nullptr;
    double r = strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') {
        fprintf(stderr, "%s: option '--%s' expects a number, got '%s'\n", prog_.c_str(), name.c_str(), v.c_str());
        exit(1);
    }
    return r;
}
std::vector<std::string> ArgParser::get_all(const std::string & name) const {
    auto it = cli_values_.find(name);
    if (it != cli_values_.end()) return it->second;
    if (auto e = env_value(name)) return {*e};
    if (auto c = config_value(name)) return {*c};
    return {};
}

std::string ArgParser::usage_line() const {
    std::string u = prog_ + " " + usage_;
    return u;
}

static std::string wrap_text(const std::string & text, size_t indent, size_t width) {
    std::string out, line;
    size_t col = 0;
    std::string word;
    auto flush_word = [&] {
        if (word.empty()) return;
        if (col + word.size() + (col ? 1 : 0) > width) { out += line + "\n" + std::string(indent, ' '); line.clear(); col = 0; }
        if (col) { line += ' '; col++; }
        line += word; col += word.size(); word.clear();
    };
    for (char c : text) {
        if (c == ' ') flush_word();
        else if (c == '\n') { flush_word(); out += line + "\n" + std::string(indent, ' '); line.clear(); col = 0; }
        else word += c;
    }
    flush_word();
    return out + line;
}

void ArgParser::print_help(FILE * f) const {
    fprintf(f, "%s\n\n", wrap_text(description_, 0, 100).c_str());
    fprintf(f, "%s %s\n\n", bold("Usage:").c_str(), usage_line().c_str());
    if (!pos_spec_.empty()) {
        fprintf(f, "%s\n", bold("Arguments:").c_str());
        for (const auto & p : pos_spec_) fprintf(f, "  %-24s %s\n", (p.rest ? "[" + p.name + "...]" : p.required ? "<" + p.name + ">" : "[" + p.name + "]").c_str(), p.help.c_str());
        fprintf(f, "\n");
    }
    for (const auto & g : group_order_) {
        fprintf(f, "%s\n", bold(g + ":").c_str());
        for (const auto & fl : flags_) {
            if (fl.group != g) continue;
            std::string left = "  ";
            if (!fl.short_name.empty()) left += "-" + fl.short_name + ", ";
            else left += "    ";
            left += "--" + fl.name;
            if (!fl.is_bool()) left += " <" + fl.value_name + ">";
            std::string help = fl.help;
            if (!fl.default_value.empty()) help += " " + dim("[default: " + fl.default_value + "]");
            if (left.size() > 30) fprintf(f, "%s\n%30s %s\n", left.c_str(), "", wrap_text(help, 31, 100).c_str());
            else fprintf(f, "%-30s %s\n", left.c_str(), wrap_text(help, 31, 100).c_str());
        }
        fprintf(f, "\n");
    }
    if (!epilog_.empty()) fprintf(f, "%s\n", epilog_.c_str());
}

// ---- formatting helpers ----

std::string human_bytes(uint64_t b) {
    char buf[64];
    if (b >= (1ull << 30)) snprintf(buf, sizeof buf, "%.2f GiB", b / (double) (1ull << 30));
    else if (b >= (1ull << 20)) snprintf(buf, sizeof buf, "%.1f MiB", b / (double) (1ull << 20));
    else if (b >= 1024) snprintf(buf, sizeof buf, "%.1f KiB", b / 1024.0);
    else snprintf(buf, sizeof buf, "%llu B", (unsigned long long) b);
    return buf;
}

std::string human_duration(double s) {
    char buf[64];
    if (s < 60) snprintf(buf, sizeof buf, "%ds", (int) s);
    else if (s < 3600) snprintf(buf, sizeof buf, "%dm%02ds", (int) s / 60, (int) s % 60);
    else if (s < 86400) snprintf(buf, sizeof buf, "%dh%02dm", (int) s / 3600, ((int) s % 3600) / 60);
    else snprintf(buf, sizeof buf, "%dd%02dh", (int) s / 86400, ((int) s % 86400) / 3600);
    return buf;
}

std::string pad(const std::string & s, size_t w, bool left) {
    if (s.size() >= w) return s;
    return left ? s + std::string(w - s.size(), ' ') : std::string(w - s.size(), ' ') + s;
}

bool is_tty(FILE * f) { return isatty(fileno(f)) != 0; }

static bool use_color() {
    static const bool on = is_tty(stdout) && getenv("NO_COLOR") == nullptr && (getenv("TERM") == nullptr || strcmp(getenv("TERM"), "dumb") != 0);
    return on;
}
std::string bold(const std::string & s) { return use_color() ? "\033[1m" + s + "\033[0m" : s; }
std::string dim(const std::string & s) { return use_color() ? "\033[2m" + s + "\033[0m" : s; }
std::string color(const std::string & s, int code) { return use_color() ? "\033[" + std::to_string(code) + "m" + s + "\033[0m" : s; }

} // namespace iian::cli
