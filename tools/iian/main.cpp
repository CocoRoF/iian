// The `iian` command-line binary: dispatches to subcommands in src/iian/cli.
#include "iian/cli/args.h"
#include "iian/cli/commands.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    using namespace iian::cli;
    if (argc < 2) return print_global_help();
    std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        if (argc > 2) {
            if (const Command * c = find_command(argv[2])) return c->run({"--help"});
            return fail(std::string("unknown command '") + argv[2] + "'");
        }
        return print_global_help();
    }
    if (cmd == "--version" || cmd == "-V") return cmd_version({});
    const Command * c = find_command(cmd);
    if (!c) {
        std::string best; size_t bd = 100;
        for (const auto & k : commands()) { size_t d = levenshtein(cmd, k.name); if (d < bd) { bd = d; best = k.name; } }
        fprintf(stderr, "iian: unknown command '%s'%s\nRun 'iian --help' for the list of commands.\n", cmd.c_str(), bd <= 2 ? (" (did you mean '" + best + "'?)").c_str() : "");
        return 1;
    }
    Args args(argv + 2, argv + argc);
    return c->run(args);
}
