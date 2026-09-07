#pragma once
// Flags shared by serve / run / bench: model placement, engine + KV cache, scheduler, logging.
#include "args.h"

#include "iian/engine.h"
#include "iian/log.h"
#include "iian/model.h"

#include <string>

namespace iian::cli {

void add_model_flags(ArgParser & p);      // Model group: --threads, --n-gpu-layers, --device, --no-mmap, --mlock
void add_engine_flags(ArgParser & p);     // Engine/KV cache + Scheduler groups
void add_logging_flags(ArgParser & p);    // Logging group
void add_sampling_flags(ArgParser & p);   // Sampling group (run/complete/bench)

DeviceConfig device_config_from_args(const ArgParser & p);
EngineConfig engine_config_from_args(const ArgParser & p);   // includes DeviceConfig
// Applies --log-level/--log-format/--log-file/--no-color. Returns false on a bad level name.
bool setup_logging(const ArgParser & p, std::string & err);
// Loads a JSON --config file into the parser (flat object of flag -> value). Returns false + err if unreadable.
bool load_config_file(ArgParser & p, const std::string & path, std::string & err);

// Human-readable model summary lines for banners.
std::string model_banner(const Model & m);

} // namespace iian::cli
