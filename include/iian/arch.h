#pragma once
// Architecture plug-in interface.
//
// Adding a model architecture = one file in src/iian/arch/ that:
//   1. subclasses ArchDef (load_hparams / create_tensors / build_graph)
//   2. calls IIAN_REGISTER_ARCH(<gguf arch name>, <class>)
// Nothing else needs to be touched (no enums, no switch statements, no CMake edits: arch/*.cpp is globbed).
#include "iian/hparams.h"
#include "iian/model.h"
#include "iian/types.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace iian {

class GraphContext;

class ArchDef {
public:
    virtual ~ArchDef() = default;
    virtual const char * name() const = 0;
    // Called after the generic keys are parsed; read arch-specific keys / override defaults.
    virtual void load_hparams(const GgufFile & f, HParams & hp) const = 0;
    // Declare the tensors. Use tc.layer_tensor / tc.global_tensor; store into model.layers[il] / model.*
    virtual void create_tensors(Model & model, TensorCreator & tc) const = 0;
    // Build the forward graph for one micro-batch. Must set gc.t_logits (and gc.t_embd).
    virtual void build_graph(GraphContext & gc) const = 0;
    // RoPE variant used by this architecture.
    virtual RopeType rope_type(const HParams & hp) const { (void) hp; return RopeType::NORMAL; }
};

class ArchRegistry {
public:
    using Factory = std::function<std::unique_ptr<ArchDef>()>;
    static ArchRegistry & instance();
    void add(const std::string & name, Factory f);
    const ArchDef * get(const std::string & name) const;   // nullptr if unsupported
    std::vector<std::string> names() const;
private:
    std::map<std::string, std::unique_ptr<ArchDef>> archs_;
};

struct ArchRegistrar {
    ArchRegistrar(const char * name, ArchRegistry::Factory f) { ArchRegistry::instance().add(name, std::move(f)); }
};

#define IIAN_REGISTER_ARCH(NAME, CLASS) \
    static ::iian::ArchRegistrar iian_arch_registrar_##CLASS(NAME, []() { return std::unique_ptr<::iian::ArchDef>(new CLASS()); })

// Force-link all built-in architectures (called once by the loader).
void register_builtin_archs();

} // namespace iian
