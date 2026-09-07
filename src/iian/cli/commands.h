#pragma once
// Subcommand registry for the `iian` binary.
#include <string>
#include <vector>

namespace iian::cli {

using Args = std::vector<std::string>;   // argv without the program name and subcommand

struct Command {
    const char * name;
    const char * summary;
    int (*run)(const Args & args);
    const char * section;   // "Serve" | "Manage" | "Use" | "Models" | "Other"
};

const std::vector<Command> & commands();
const Command * find_command(const std::string & name);
int print_global_help();
std::string build_info_string();

// Each command lives in its own file.
int cmd_serve(const Args &);
int cmd_ps(const Args &);
int cmd_stop(const Args &);
int cmd_restart(const Args &);
int cmd_logs(const Args &);
int cmd_status(const Args &);
int cmd_run(const Args &);
int cmd_complete(const Args &);
int cmd_embed(const Args &);
int cmd_pull(const Args &);
int cmd_ls(const Args &);
int cmd_rm(const Args &);
int cmd_info(const Args &);
int cmd_bench(const Args &);
int cmd_version(const Args &);

// Shared: print "error: ..." to stderr and return 1.
int fail(const std::string & msg);

} // namespace iian::cli
