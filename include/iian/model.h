#pragma once
// Model: loaded weights on backend buffers + hparams + architecture definition.
#include "iian/gguf.h"
#include "iian/hparams.h"
#include "iian/types.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_backend_buffer;
struct ggml_backend_device;
struct ggml_backend;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_device * ggml_backend_dev_t;
typedef struct ggml_backend * ggml_backend_t;

namespace iian {

class ArchDef;
class Tokenizer;

// Generic per-layer weight slots. Architectures use the subset they need; unused slots stay null.
// This mirrors llama.cpp's llama_layer so that graph-builder code can be ported almost verbatim.
struct LayerWeights {
    ggml_tensor * attn_norm = nullptr, * attn_norm_b = nullptr;
    ggml_tensor * attn_norm_2 = nullptr, * attn_norm_2_b = nullptr;
    ggml_tensor * attn_q_norm = nullptr, * attn_k_norm = nullptr;
    ggml_tensor * attn_post_norm = nullptr;
    ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr, * wqkv = nullptr;
    ggml_tensor * bq = nullptr, * bk = nullptr, * bv = nullptr, * bo = nullptr, * bqkv = nullptr;
    ggml_tensor * ffn_norm = nullptr, * ffn_norm_b = nullptr, * ffn_post_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr, * ffn_down = nullptr, * ffn_up = nullptr;
    ggml_tensor * ffn_gate_b = nullptr, * ffn_down_b = nullptr, * ffn_up_b = nullptr;
    // MoE
    ggml_tensor * ffn_gate_inp = nullptr, * ffn_gate_inp_b = nullptr;
    ggml_tensor * ffn_gate_exps = nullptr, * ffn_down_exps = nullptr, * ffn_up_exps = nullptr;
    ggml_tensor * ffn_gate_shexp = nullptr, * ffn_down_shexp = nullptr, * ffn_up_shexp = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    // rope
    ggml_tensor * rope_freqs = nullptr, * rope_long = nullptr, * rope_short = nullptr;
    // free-form extras for exotic archs: name -> tensor
    std::map<std::string, ggml_tensor *> extra;
};

struct DeviceConfig {
    // Devices to place layers on, in order (empty -> auto: best GPU if any, else CPU).
    std::vector<std::string> devices;
    int      n_gpu_layers = -1;   // -1 = all layers if a GPU exists
    // Fraction of the GPU layers each device gets, in device order (like llama.cpp's --tensor-split); empty ->
    // proportional to each GPU's free memory at load time.
    std::vector<float> tensor_split;
    bool     use_mmap     = true;
    bool     use_mlock    = false;
    bool     use_extra_bufts = true;   // CPU weight repacking (ggml "extra" buffer types: Q4_0/Q4_K/Q8_0/IQ4_NL/Q2_K)
    int      n_threads    = -1;   // CPU threads for compute (-1 = auto)
    int      n_threads_batch = -1;
};

class Model {
public:
    ~Model();

    const HParams & hparams() const { return hparams_; }
    const ArchDef & arch() const { return *arch_; }
    const std::string & path() const { return gguf_->path(); }
    const GgufFile & gguf() const { return *gguf_; }
    const Tokenizer & tokenizer() const { return *tokenizer_; }
    Tokenizer & tokenizer() { return *tokenizer_; }

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * tok_norm = nullptr, * tok_norm_b = nullptr;
    ggml_tensor * output_norm = nullptr, * output_norm_b = nullptr;
    ggml_tensor * output = nullptr, * output_b = nullptr;
    std::vector<LayerWeights> layers;
    std::map<std::string, ggml_tensor *> extra;   // non-layer extras

    // Device placement
    ggml_backend_dev_t dev_input() const { return dev_input_; }
    ggml_backend_dev_t dev_output() const { return dev_output_; }
    ggml_backend_dev_t dev_layer(uint32_t il) const { return dev_layer_.at(il); }
    const std::vector<ggml_backend_dev_t> & devices() const { return devices_; }
    size_t total_bytes() const { return total_bytes_; }
    std::string ftype_name() const { return ftype_; }
    std::string size_label() const;   // e.g. "135M", "7.2B"
    uint64_t n_params() const { return n_params_; }

    // Lookup tensor by GGUF name (nullptr if absent)
    ggml_tensor * get(const std::string & name) const;

private:
    friend class ModelLoader;
    friend class TensorCreatorImpl;
    Model() = default;
    HParams hparams_;
    const ArchDef * arch_ = nullptr;
    std::unique_ptr<GgufFile> gguf_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::vector<ggml_context *> ctxs_;
    std::vector<ggml_backend_buffer_t> bufs_;
    std::map<std::string, ggml_tensor *> tensors_;
    std::vector<ggml_backend_dev_t> devices_;
    ggml_backend_dev_t dev_input_ = nullptr, dev_output_ = nullptr;
    std::vector<ggml_backend_dev_t> dev_layer_;
    size_t total_bytes_ = 0;
    uint64_t n_params_ = 0;
    std::string ftype_;
};

using ProgressCallback = std::function<bool(float progress)>;

class ModelLoader {
public:
    static std::unique_ptr<Model> load(const std::string & path, const DeviceConfig & cfg, ProgressCallback progress = nullptr);
    // Parse only metadata + tokenizer (no weights): used by `iian info` and for quick validation.
    static std::unique_ptr<Model> load_metadata(const std::string & path);
};

// TensorCreator: given to ArchDef::create_tensors so architectures declare the weights they expect.
class TensorCreator {
public:
    enum Flags : uint32_t { REQUIRED = 0, NOT_REQUIRED = 1 << 0, DUPLICATED = 1 << 1 };
    // name: GGUF tensor name (e.g. "blk.3.attn_q.weight"); ne: expected dims (validated against the file).
    virtual ggml_tensor * create(const std::string & name, std::initializer_list<int64_t> ne, int layer, uint32_t flags = REQUIRED) = 0;
    ggml_tensor * layer_tensor(const char * base, int il, const char * suffix, std::initializer_list<int64_t> ne, uint32_t flags = REQUIRED);
    ggml_tensor * global_tensor(const char * base, const char * suffix, std::initializer_list<int64_t> ne, uint32_t flags = REQUIRED);
    virtual ~TensorCreator() = default;
};

} // namespace iian
