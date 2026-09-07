#include "iian/gguf.h"
#include "iian/log.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace iian {

std::unique_ptr<GgufFile> GgufFile::open(const std::string & path, bool use_mmap) {
    std::unique_ptr<GgufFile> f(new GgufFile());
    f->path_ = path;

    gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ nullptr };
    f->ctx_ = gguf_init_from_file(path.c_str(), p);
    if (!f->ctx_) {
        throw std::runtime_error("failed to open GGUF file: " + path);
    }
    if (int64_t k = gguf_find_key(f->ctx_, "general.architecture"); k >= 0) {
        f->arch_ = gguf_get_val_str(f->ctx_, k);
    }
    f->data_offset_ = gguf_get_data_offset(f->ctx_);

    const int64_t nt = gguf_get_n_tensors(f->ctx_);
    f->tensors_.reserve(nt);
    for (int64_t i = 0; i < nt; i++) {
        GgufTensorInfo t;
        t.name   = gguf_get_tensor_name(f->ctx_, i);
        t.type   = gguf_get_tensor_type(f->ctx_, i);
        t.offset = gguf_get_tensor_offset(f->ctx_, i);
        t.nbytes = gguf_get_tensor_size(f->ctx_, i);
        // reconstruct dims: gguf stores ne[] via the tensor-info accessor
        // gguf.h does not expose ne directly; we rebuild a temporary ggml tensor description
        // through gguf's metadata: use gguf_get_tensor_shape? Not available -> parse from a
        // ggml context created with no_alloc.
        t.n_dims = 0;
        t.ne[0] = t.ne[1] = t.ne[2] = t.ne[3] = 1;
        f->tensors_.push_back(std::move(t));
        f->total_bytes_ += f->tensors_.back().nbytes;
    }
    // Recover shapes: gguf only exposes them via a ggml context. Re-parse with a metadata ctx.
    {
        ggml_context * meta = nullptr;
        gguf_init_params p2 = { /*no_alloc*/ true, &meta };
        gguf_context * c2 = gguf_init_from_file(path.c_str(), p2);
        if (!c2 || !meta) throw std::runtime_error("failed to parse tensor shapes: " + path);
        for (auto & t : f->tensors_) {
            ggml_tensor * gt = ggml_get_tensor(meta, t.name.c_str());
            if (!gt) throw std::runtime_error("tensor missing in meta ctx: " + t.name);
            t.n_dims = ggml_n_dims(gt);
            for (int d = 0; d < 4; d++) t.ne[d] = gt->ne[d];
        }
        gguf_free(c2);
        ggml_free(meta);
    }

    if (use_mmap) {
        f->fd_ = ::open(path.c_str(), O_RDONLY);
        if (f->fd_ >= 0) {
            struct stat st{};
            if (fstat(f->fd_, &st) == 0) {
                f->map_size_ = (size_t) st.st_size;
                void * addr = mmap(nullptr, f->map_size_, PROT_READ, MAP_PRIVATE, f->fd_, 0);
                if (addr != MAP_FAILED) {
                    f->map_addr_ = addr;
#ifdef MADV_WILLNEED
                    madvise(addr, f->map_size_, MADV_WILLNEED);
#endif
                } else {
                    LOG_WRN("gguf", "mmap failed for %s, falling back to read()", path.c_str());
                }
            }
        }
    }
    LOG_DBG("gguf", "opened %s: arch=%s kv=%lld tensors=%lld data_off=%zu mmap=%d",
            path.c_str(), f->arch_.c_str(), (long long) gguf_get_n_kv(f->ctx_), (long long) nt, f->data_offset_, f->map_addr_ != nullptr);
    return f;
}

GgufFile::~GgufFile() {
    if (map_addr_) munmap(map_addr_, map_size_);
    if (fd_ >= 0) ::close(fd_);
    if (ctx_) gguf_free(ctx_);
}

std::string GgufFile::architecture() const { return arch_; }

std::string GgufFile::expand(const std::string & key) const {
    size_t pos = key.find("%s");
    if (pos == std::string::npos) return key;
    return key.substr(0, pos) + arch_ + key.substr(pos + 2);
}

bool GgufFile::has(const std::string & key) const { return gguf_find_key(ctx_, expand(key).c_str()) >= 0; }

std::optional<std::string> GgufFile::get_str(const std::string & key) const {
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0 || gguf_get_kv_type(ctx_, k) != GGUF_TYPE_STRING) return std::nullopt;
    return std::string(gguf_get_val_str(ctx_, k));
}

static std::optional<double> gguf_get_num(const gguf_context * ctx, int64_t k) {
    switch (gguf_get_kv_type(ctx, k)) {
        case GGUF_TYPE_UINT8:   return (double) gguf_get_val_u8(ctx, k);
        case GGUF_TYPE_INT8:    return (double) gguf_get_val_i8(ctx, k);
        case GGUF_TYPE_UINT16:  return (double) gguf_get_val_u16(ctx, k);
        case GGUF_TYPE_INT16:   return (double) gguf_get_val_i16(ctx, k);
        case GGUF_TYPE_UINT32:  return (double) gguf_get_val_u32(ctx, k);
        case GGUF_TYPE_INT32:   return (double) gguf_get_val_i32(ctx, k);
        case GGUF_TYPE_UINT64:  return (double) gguf_get_val_u64(ctx, k);
        case GGUF_TYPE_INT64:   return (double) gguf_get_val_i64(ctx, k);
        case GGUF_TYPE_FLOAT32: return (double) gguf_get_val_f32(ctx, k);
        case GGUF_TYPE_FLOAT64: return (double) gguf_get_val_f64(ctx, k);
        case GGUF_TYPE_BOOL:    return (double) gguf_get_val_bool(ctx, k);
        default: return std::nullopt;
    }
}

std::optional<uint32_t> GgufFile::get_u32(const std::string & key) const {
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0) return std::nullopt;
    auto v = gguf_get_num(ctx_, k);
    if (!v) return std::nullopt;
    return (uint32_t) *v;
}
std::optional<int32_t> GgufFile::get_i32(const std::string & key) const {
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0) return std::nullopt;
    auto v = gguf_get_num(ctx_, k);
    if (!v) return std::nullopt;
    return (int32_t) *v;
}
std::optional<float> GgufFile::get_f32(const std::string & key) const {
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0) return std::nullopt;
    auto v = gguf_get_num(ctx_, k);
    if (!v) return std::nullopt;
    return (float) *v;
}
std::optional<bool> GgufFile::get_bool(const std::string & key) const {
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0) return std::nullopt;
    auto v = gguf_get_num(ctx_, k);
    if (!v) return std::nullopt;
    return *v != 0.0;
}

static double gguf_arr_num(const gguf_context * ctx, int64_t k, size_t i) {
    const void * data = gguf_get_arr_data(ctx, k);
    switch (gguf_get_arr_type(ctx, k)) {
        case GGUF_TYPE_UINT8:   return ((const uint8_t  *) data)[i];
        case GGUF_TYPE_INT8:    return ((const int8_t   *) data)[i];
        case GGUF_TYPE_UINT16:  return ((const uint16_t *) data)[i];
        case GGUF_TYPE_INT16:   return ((const int16_t  *) data)[i];
        case GGUF_TYPE_UINT32:  return ((const uint32_t *) data)[i];
        case GGUF_TYPE_INT32:   return ((const int32_t  *) data)[i];
        case GGUF_TYPE_UINT64:  return (double) ((const uint64_t *) data)[i];
        case GGUF_TYPE_INT64:   return (double) ((const int64_t  *) data)[i];
        case GGUF_TYPE_FLOAT32: return ((const float    *) data)[i];
        case GGUF_TYPE_FLOAT64: return ((const double   *) data)[i];
        case GGUF_TYPE_BOOL:    return ((const int8_t   *) data)[i] != 0;
        default: return 0.0;
    }
}

std::vector<uint32_t> GgufFile::get_arr_u32(const std::string & key) const {
    std::vector<uint32_t> out;
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0 || gguf_get_kv_type(ctx_, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_, k) == GGUF_TYPE_STRING) return out;
    size_t n = gguf_get_arr_n(ctx_, k);
    out.resize(n);
    for (size_t i = 0; i < n; i++) out[i] = (uint32_t) gguf_arr_num(ctx_, k, i);
    return out;
}
std::vector<float> GgufFile::get_arr_f32(const std::string & key) const {
    std::vector<float> out;
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0 || gguf_get_kv_type(ctx_, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_, k) == GGUF_TYPE_STRING) return out;
    size_t n = gguf_get_arr_n(ctx_, k);
    out.resize(n);
    for (size_t i = 0; i < n; i++) out[i] = (float) gguf_arr_num(ctx_, k, i);
    return out;
}
std::vector<std::string> GgufFile::get_arr_str(const std::string & key) const {
    std::vector<std::string> out;
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0 || gguf_get_kv_type(ctx_, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_, k) != GGUF_TYPE_STRING) return out;
    size_t n = gguf_get_arr_n(ctx_, k);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(ctx_, k, i));
    return out;
}
size_t GgufFile::get_arr_n(const std::string & key) const {
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0 || gguf_get_kv_type(ctx_, k) != GGUF_TYPE_ARRAY) return 0;
    return gguf_get_arr_n(ctx_, k);
}

std::vector<uint32_t> GgufFile::get_u32_per_layer(const std::string & key, uint32_t n_layer, std::optional<uint32_t> fallback) const {
    std::vector<uint32_t> out(n_layer, fallback.value_or(0));
    int64_t k = gguf_find_key(ctx_, expand(key).c_str());
    if (k < 0) {
        if (!fallback) throw std::runtime_error("missing required key: " + expand(key));
        return out;
    }
    if (gguf_get_kv_type(ctx_, k) == GGUF_TYPE_ARRAY) {
        auto arr = get_arr_u32(key);
        if (arr.size() != n_layer) throw std::runtime_error("key " + expand(key) + " has " + std::to_string(arr.size()) + " entries, expected " + std::to_string(n_layer));
        return arr;
    }
    auto v = get_u32(key);
    std::fill(out.begin(), out.end(), *v);
    return out;
}

const GgufTensorInfo * GgufFile::find_tensor(const std::string & name) const {
    for (const auto & t : tensors_) if (t.name == name) return &t;
    return nullptr;
}

const void * GgufFile::tensor_data(const GgufTensorInfo & t) const {
    if (!map_addr_) return nullptr;
    return (const char *) map_addr_ + data_offset_ + t.offset;
}

void GgufFile::read_tensor(const GgufTensorInfo & t, void * dst) const {
    if (map_addr_) { memcpy(dst, tensor_data(t), t.nbytes); return; }
    int fd = fd_ >= 0 ? fd_ : ::open(path_.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path_);
    size_t off = data_offset_ + t.offset, done = 0;
    while (done < t.nbytes) {
        ssize_t r = pread(fd, (char *) dst + done, t.nbytes - done, (off_t) (off + done));
        if (r <= 0) throw std::runtime_error("short read on " + path_);
        done += (size_t) r;
    }
    if (fd != fd_) ::close(fd);
}

std::vector<std::pair<std::string, std::string>> GgufFile::metadata_dump() const {
    std::vector<std::pair<std::string, std::string>> out;
    const int64_t n = gguf_get_n_kv(ctx_);
    for (int64_t i = 0; i < n; i++) {
        const char * key = gguf_get_key(ctx_, i);
        std::string val;
        gguf_type t = gguf_get_kv_type(ctx_, i);
        if (t == GGUF_TYPE_STRING) {
            val = gguf_get_val_str(ctx_, i);
            if (val.size() > 120) val = val.substr(0, 117) + "...";
        } else if (t == GGUF_TYPE_ARRAY) {
            size_t an = gguf_get_arr_n(ctx_, i);
            val = "[array of " + std::to_string(an) + "]";
            if (an <= 8 && gguf_get_arr_type(ctx_, i) != GGUF_TYPE_STRING) {
                val = "[";
                for (size_t j = 0; j < an; j++) { if (j) val += ", "; val += std::to_string(gguf_arr_num(ctx_, i, j)); }
                val += "]";
            }
        } else if (auto v = gguf_get_num(ctx_, i)) {
            if (t == GGUF_TYPE_FLOAT32 || t == GGUF_TYPE_FLOAT64) { char b[64]; snprintf(b, sizeof b, "%g", *v); val = b; }
            else val = std::to_string((long long) *v);
        }
        out.emplace_back(key, val);
    }
    return out;
}

} // namespace iian
