#pragma once
// Thin, typed wrapper around ggml's GGUF reader with mmap support.
#include "iian/types.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ggml.h"

struct gguf_context;

namespace iian {

struct GgufTensorInfo {
    std::string name;
    ggml_type   type;
    int64_t     ne[4];
    int         n_dims;
    size_t      offset;   // relative to data section
    size_t      nbytes;
};

class GgufFile {
public:
    // Opens the file, parses metadata and (optionally) mmaps the whole file for zero-copy weight access.
    static std::unique_ptr<GgufFile> open(const std::string & path, bool use_mmap = true);
    ~GgufFile();

    const std::string & path() const { return path_; }
    const gguf_context * ctx() const { return ctx_; }
    std::string architecture() const;   // "general.architecture"

    // ---- typed metadata access (key may contain "%s" which is replaced with the architecture) ----
    bool has(const std::string & key) const;
    std::optional<std::string> get_str(const std::string & key) const;
    std::optional<uint32_t>    get_u32(const std::string & key) const;   // any integer type is converted
    std::optional<int32_t>     get_i32(const std::string & key) const;
    std::optional<float>       get_f32(const std::string & key) const;
    std::optional<bool>        get_bool(const std::string & key) const;
    // Arrays of numbers (converted to the requested type); empty if the key is missing.
    std::vector<uint32_t>      get_arr_u32(const std::string & key) const;
    std::vector<float>         get_arr_f32(const std::string & key) const;
    std::vector<std::string>   get_arr_str(const std::string & key) const;
    size_t                     get_arr_n(const std::string & key) const;
    // Per-layer value: key may be a scalar (broadcast) or an array of n_layer entries.
    std::vector<uint32_t>      get_u32_per_layer(const std::string & key, uint32_t n_layer, std::optional<uint32_t> fallback) const;

    // Returns the key with "%s" expanded (e.g. "%s.block_count" -> "llama.block_count")
    std::string expand(const std::string & key) const;

    // ---- tensors ----
    const std::vector<GgufTensorInfo> & tensors() const { return tensors_; }
    const GgufTensorInfo * find_tensor(const std::string & name) const;
    size_t data_offset() const { return data_offset_; }
    size_t total_tensor_bytes() const { return total_bytes_; }

    // Pointer to a tensor's bytes inside the mapping (nullptr if not mmapped).
    const void * tensor_data(const GgufTensorInfo & t) const;
    // Read tensor bytes from the file into dst (works with or without mmap).
    void read_tensor(const GgufTensorInfo & t, void * dst) const;
    bool mmapped() const { return map_addr_ != nullptr; }
    const void * map_addr() const { return map_addr_; }
    size_t map_size() const { return map_size_; }
    // Dump all metadata (key = value) for `iian info`.
    std::vector<std::pair<std::string, std::string>> metadata_dump() const;

private:
    GgufFile() = default;
    std::string   path_;
    gguf_context * ctx_ = nullptr;
    std::string   arch_;
    std::vector<GgufTensorInfo> tensors_;
    size_t        data_offset_ = 0;
    size_t        total_bytes_ = 0;
    void *        map_addr_ = nullptr;
    size_t        map_size_ = 0;
    int           fd_ = -1;
};

} // namespace iian
