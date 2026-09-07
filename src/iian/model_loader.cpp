#include "iian/arch.h"
#include "iian/gguf.h"
#include "iian/log.h"
#include "iian/model.h"
#include "iian/tokenizer.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

namespace iian {

// ---------------------------------------------------------------------------------------------
// HParams
// ---------------------------------------------------------------------------------------------
uint32_t HParams::n_head_max() const { uint32_t m = 0; for (auto v : n_head) m = std::max(m, v); return m; }
uint32_t HParams::n_head_kv_max() const { uint32_t m = 0; for (auto v : n_head_kv) m = std::max(m, v); return m; }

std::string HParams::summary() const {
    char buf[512];
    snprintf(buf, sizeof buf,
             "arch=%s n_layer=%u n_embd=%u n_head=%u n_head_kv=%u head_dim=%u n_ff=%u n_vocab=%u n_ctx_train=%u rope=%s base=%.0f scale=%.3f",
             arch.c_str(), n_layer, n_embd, n_head.empty() ? 0 : n_head[0], n_head_kv.empty() ? 0 : n_head_kv[0],
             n_embd_head_k, n_ff.empty() ? 0 : n_ff[0], n_vocab, n_ctx_train,
             rope_type == RopeType::NEOX ? "neox" : rope_type == RopeType::NORMAL ? "normal" : "none",
             rope_freq_base, rope_freq_scale);
    return buf;
}

static void load_generic_hparams(const GgufFile & f, HParams & hp) {
    hp.arch = f.architecture();
    hp.name = f.get_str("general.name").value_or("");

    auto req_u32 = [&](const char * key) {
        auto v = f.get_u32(key);
        if (!v) throw std::runtime_error(std::string("GGUF is missing required key: ") + f.expand(key));
        return *v;
    };

    hp.n_ctx_train = req_u32("%s.context_length");
    hp.n_embd      = req_u32("%s.embedding_length");
    hp.n_layer     = req_u32("%s.block_count");
    hp.n_expert    = f.get_u32("%s.expert_count").value_or(0);
    hp.n_expert_used = f.get_u32("%s.expert_used_count").value_or(0);
    hp.causal_attn = f.get_bool("%s.attention.causal").value_or(true);

    hp.n_ff      = f.get_u32_per_layer("%s.feed_forward_length", hp.n_layer, 0u);
    hp.n_head    = f.get_u32_per_layer("%s.attention.head_count", hp.n_layer, 0u);
    hp.n_head_kv = f.has("%s.attention.head_count_kv") ? f.get_u32_per_layer("%s.attention.head_count_kv", hp.n_layer, 0u) : hp.n_head;
    hp.n_ff_exp  = f.get_u32_per_layer("%s.expert_feed_forward_length", hp.n_layer, 0u);

    // vocab size: prefer explicit key, else tokenizer list length
    if (auto v = f.get_u32("%s.vocab_size")) hp.n_vocab = *v;
    else hp.n_vocab = (uint32_t) f.get_arr_n("tokenizer.ggml.tokens");

    const uint32_t n_head0 = hp.n_head_max();
    if (n_head0 > 0) {
        hp.n_embd_head_k = f.get_u32("%s.attention.key_length").value_or(hp.n_embd / n_head0);
        hp.n_embd_head_v = f.get_u32("%s.attention.value_length").value_or(hp.n_embd / n_head0);
        hp.n_rot         = f.get_u32("%s.rope.dimension_count").value_or(hp.n_embd_head_k);
    }

    hp.f_norm_eps     = f.get_f32("%s.attention.layer_norm_epsilon").value_or(1e-5f);
    hp.f_norm_rms_eps = f.get_f32("%s.attention.layer_norm_rms_epsilon").value_or(1e-5f);
    hp.f_max_alibi_bias = f.get_f32("%s.attention.max_alibi_bias").value_or(0.0f);
    hp.f_clamp_kqv    = f.get_f32("%s.attention.clamp_kqv").value_or(0.0f);
    hp.f_logit_scale  = f.get_f32("%s.logit_scale").value_or(0.0f);
    hp.f_attention_scale = f.get_f32("%s.attention.scale").value_or(0.0f);
    hp.f_attn_logit_softcapping  = f.get_f32("%s.attn_logit_softcapping").value_or(0.0f);
    hp.f_final_logit_softcapping = f.get_f32("%s.final_logit_softcapping").value_or(0.0f);

    // rope
    hp.rope_freq_base = f.get_f32("%s.rope.freq_base").value_or(10000.0f);
    hp.n_ctx_orig_yarn = f.get_u32("%s.rope.scaling.original_context_length").value_or(hp.n_ctx_train);
    hp.rope_finetuned = f.get_bool("%s.rope.scaling.finetuned").value_or(false);
    std::string scaling = f.get_str("%s.rope.scaling.type").value_or("linear");
    if (scaling == "none") hp.rope_scaling = RopeScaling::NONE;
    else if (scaling == "linear") hp.rope_scaling = RopeScaling::LINEAR;
    else if (scaling == "yarn") hp.rope_scaling = RopeScaling::YARN;
    else if (scaling == "longrope") hp.rope_scaling = RopeScaling::LONGROPE;
    else throw std::runtime_error("unknown rope scaling type: " + scaling);
    float ropescale = f.get_f32("%s.rope.scaling.factor").value_or(f.get_f32("%s.rope.scale_linear").value_or(0.0f));
    hp.rope_freq_scale = ropescale == 0.0f ? 1.0f : 1.0f / ropescale;
    if (hp.rope_scaling == RopeScaling::NONE) hp.rope_freq_scale = 1.0f;
    hp.yarn_attn_factor  = f.get_f32("%s.rope.scaling.attn_factor").value_or(1.0f);
    hp.yarn_ext_factor   = f.get_f32("%s.rope.scaling.yarn_ext_factor").value_or(-1.0f);
    hp.yarn_beta_fast    = f.get_f32("%s.rope.scaling.yarn_beta_fast").value_or(32.0f);
    hp.yarn_beta_slow    = f.get_f32("%s.rope.scaling.yarn_beta_slow").value_or(1.0f);
    hp.rope_yarn_log_mul = f.get_f32("%s.rope.scaling.yarn_log_multiplier").value_or(0.0f);
    if (hp.yarn_ext_factor < 0.0f) hp.yarn_ext_factor = hp.rope_scaling == RopeScaling::YARN ? 1.0f : 0.0f;
    if (hp.yarn_ext_factor != 0.0f && hp.rope_yarn_log_mul != 0.0f) {
        // ref: transformers modeling_rope_utils (assumes mscale == 1)
        auto get_mscale = [](float scale, float mscale) { return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f); };
        const float factor = 1.0f / hp.rope_freq_scale;
        hp.yarn_attn_factor = get_mscale(factor, 1.0f) / get_mscale(factor, hp.rope_yarn_log_mul);
    }

    // SWA
    hp.n_swa = f.get_u32("%s.attention.sliding_window").value_or(0);
    hp.is_swa.assign(hp.n_layer, 0);

    hp.expert_weights_scale = f.get_f32("%s.expert_weights_scale").value_or(0.0f);
    hp.expert_weights_norm  = f.get_bool("%s.expert_weights_norm").value_or(false);
}

// ---------------------------------------------------------------------------------------------
// TensorCreator helpers
// ---------------------------------------------------------------------------------------------
ggml_tensor * TensorCreator::layer_tensor(const char * base, int il, const char * suffix, std::initializer_list<int64_t> ne, uint32_t flags) {
    char name[256];
    snprintf(name, sizeof name, "blk.%d.%s.%s", il, base, suffix);
    return create(name, ne, il, flags);
}
ggml_tensor * TensorCreator::global_tensor(const char * base, const char * suffix, std::initializer_list<int64_t> ne, uint32_t flags) {
    std::string name = std::string(base) + "." + suffix;
    return create(name, ne, -1, flags);
}

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------
Model::~Model() {
    for (auto * b : bufs_) ggml_backend_buffer_free(b);
    for (auto * c : ctxs_) ggml_free(c);
}

ggml_tensor * Model::get(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

std::string Model::size_label() const {
    double n = (double) n_params_;
    char buf[32];
    if (n >= 1e12) snprintf(buf, sizeof buf, "%.1fT", n / 1e12);
    else if (n >= 1e9) snprintf(buf, sizeof buf, "%.1fB", n / 1e9);
    else if (n >= 1e6) snprintf(buf, sizeof buf, "%.0fM", n / 1e6);
    else snprintf(buf, sizeof buf, "%.0fK", n / 1e3);
    return buf;
}

// ---------------------------------------------------------------------------------------------
// ModelLoader
// ---------------------------------------------------------------------------------------------
namespace {

struct ArchNames {
    static std::string ftype_name(uint32_t ftype) {
        switch (ftype) {
            case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 3: return "Q4_1";
            case 7: return "Q8_0"; case 8: return "Q5_0"; case 9: return "Q5_1"; case 10: return "Q2_K";
            case 11: return "Q3_K_S"; case 12: return "Q3_K_M"; case 13: return "Q3_K_L"; case 14: return "Q4_K_S";
            case 15: return "Q4_K_M"; case 16: return "Q5_K_S"; case 17: return "Q5_K_M"; case 18: return "Q6_K";
            case 19: return "IQ2_XXS"; case 20: return "IQ2_XS"; case 21: return "Q2_K_S"; case 22: return "IQ3_XS";
            case 23: return "IQ3_XXS"; case 24: return "IQ1_S"; case 25: return "IQ4_NL"; case 26: return "IQ3_S";
            case 27: return "IQ3_M"; case 28: return "IQ2_S"; case 29: return "IQ2_M"; case 30: return "IQ4_XS";
            case 31: return "IQ1_M"; case 32: return "BF16"; case 36: return "TQ1_0"; case 37: return "TQ2_0";
            case 38: return "MXFP4"; default: return "unknown(" + std::to_string(ftype) + ")";
        }
    }
};

} // namespace

class TensorCreatorImpl : public TensorCreator {
public:
    TensorCreatorImpl(Model & m, const GgufFile & f) : model_(m), file_(f) {}

    // buffer type for a given tensor: layer tensors go to their layer's device, globals per role
    std::function<ggml_backend_buffer_type_t(int layer, const std::string & name, const GgufTensorInfo & info)> select_buft;

    ggml_tensor * create(const std::string & name, std::initializer_list<int64_t> ne, int layer, uint32_t flags) override {
        const GgufTensorInfo * info = file_.find_tensor(name);
        if (!info) {
            if (flags & NOT_REQUIRED) return nullptr;
            throw std::runtime_error("model is missing required tensor: " + name);
        }
        if (flags & DUPLICATED) {
            auto it = model_.tensors_.find(name);
            if (it != model_.tensors_.end()) return it->second;
        }
        if (model_.tensors_.count(name)) throw std::runtime_error("tensor created twice: " + name);

        // validate dims (allow trailing 1s)
        std::vector<int64_t> want(ne);
        for (size_t i = 0; i < 4; i++) {
            int64_t w = i < want.size() ? want[i] : 1;
            if (info->ne[i] != w) {
                char b[256];
                snprintf(b, sizeof b, "tensor '%s' has wrong shape: expected [%lld,%lld,%lld,%lld], got [%lld,%lld,%lld,%lld]",
                         name.c_str(), (long long)(want.size() > 0 ? want[0] : 1), (long long)(want.size() > 1 ? want[1] : 1),
                         (long long)(want.size() > 2 ? want[2] : 1), (long long)(want.size() > 3 ? want[3] : 1),
                         (long long) info->ne[0], (long long) info->ne[1], (long long) info->ne[2], (long long) info->ne[3]);
                throw std::runtime_error(b);
            }
        }

        ggml_backend_buffer_type_t buft = select_buft(layer, name, *info);
        ggml_context * ctx = ctx_for(buft);
        ggml_tensor * t = ggml_new_tensor(ctx, info->type, info->n_dims, info->ne);
        ggml_set_name(t, name.c_str());
        model_.tensors_[name] = t;
        tensor_info_[t] = info;
        model_.n_params_ += (uint64_t) ggml_nelements(t);
        model_.total_bytes_ += info->nbytes;
        return t;
    }

    ggml_context * ctx_for(ggml_backend_buffer_type_t buft) {
        auto it = ctx_map_.find(buft);
        if (it != ctx_map_.end()) return it->second;
        const size_t n_tensors_max = file_.tensors().size() + 8;
        ggml_init_params p = { ggml_tensor_overhead() * n_tensors_max, nullptr, /*no_alloc*/ true };
        ggml_context * ctx = ggml_init(p);
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ctx_map_[buft] = ctx;
        ctx_order_.push_back(buft);
        model_.ctxs_.push_back(ctx);
        return ctx;
    }

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map_;
    std::vector<ggml_backend_buffer_type_t> ctx_order_;
    std::map<ggml_tensor *, const GgufTensorInfo *> tensor_info_;

private:
    Model & model_;
    const GgufFile & file_;
};

namespace {
ggml_backend_dev_t find_device(const std::string & name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * d = ggml_backend_dev_get(i);
        if (name == ggml_backend_dev_name(d)) return d;
    }
    return nullptr;
}

} // namespace

std::unique_ptr<Model> ModelLoader::load_metadata(const std::string & path) {
    register_builtin_archs();
    std::unique_ptr<Model> m(new Model());
    m->gguf_ = GgufFile::open(path, /*mmap*/ false);
    load_generic_hparams(*m->gguf_, m->hparams_);
    m->arch_ = ArchRegistry::instance().get(m->hparams_.arch);
    if (m->arch_) {
        m->arch_->load_hparams(*m->gguf_, m->hparams_);
        m->hparams_.rope_type = m->arch_->rope_type(m->hparams_);
    }
    m->ftype_ = ArchNames::ftype_name(m->gguf_->get_u32("general.file_type").value_or(9999));
    m->total_bytes_ = m->gguf_->total_tensor_bytes();
    for (const auto & t : m->gguf_->tensors()) { uint64_t n = 1; for (int d = 0; d < 4; d++) n *= t.ne[d]; m->n_params_ += n; }
    m->tokenizer_ = Tokenizer::from_gguf(m->gguf_->ctx());
    return m;
}

std::unique_ptr<Model> ModelLoader::load(const std::string & path, const DeviceConfig & cfg, ProgressCallback progress) {
    register_builtin_archs();
    ggml_backend_load_all();

    const int64_t t0 = ggml_time_us();
    std::unique_ptr<Model> m(new Model());
    m->gguf_ = GgufFile::open(path, cfg.use_mmap);
    GgufFile & f = *m->gguf_;
    HParams & hp = m->hparams_;
    load_generic_hparams(f, hp);

    m->arch_ = ArchRegistry::instance().get(hp.arch);
    if (!m->arch_) {
        std::string supported;
        for (auto & n : ArchRegistry::instance().names()) supported += (supported.empty() ? "" : ", ") + n;
        throw std::runtime_error("unsupported model architecture '" + hp.arch + "' (supported: " + supported + ")");
    }
    m->arch_->load_hparams(f, hp);
    hp.rope_type = m->arch_->rope_type(hp);
    m->ftype_ = ArchNames::ftype_name(f.get_u32("general.file_type").value_or(9999));
    LOG_INF("model", "%s", hp.summary().c_str());

    // ---- tokenizer (metadata only) ----
    m->tokenizer_ = Tokenizer::from_gguf(f.ctx());
    LOG_INF("model", "tokenizer: %s/%s, %d tokens, bos=%d eos=%d add_bos=%d",
            m->tokenizer_->type_name().c_str(), m->tokenizer_->pre_type_name().c_str(), m->tokenizer_->n_tokens(),
            m->tokenizer_->special().bos, m->tokenizer_->special().eos, (int) m->tokenizer_->special().add_bos);

    // ---- device selection ----
    std::vector<ggml_backend_dev_t> gpus;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) throw std::runtime_error("no CPU backend device found (ggml CPU backend not registered?)");
    if (!cfg.devices.empty()) {
        for (auto & name : cfg.devices) {
            auto * d = find_device(name);
            if (!d) throw std::runtime_error("unknown device: " + name);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) gpus.push_back(d);
        }
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            auto * d = ggml_backend_dev_get(i);
            auto t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) gpus.push_back(d);
        }
    }
    int n_gpu_layers = cfg.n_gpu_layers;
    if (gpus.empty()) n_gpu_layers = 0;
    else if (n_gpu_layers < 0) n_gpu_layers = (int) hp.n_layer + 1;   // all + output
    n_gpu_layers = std::min(n_gpu_layers, (int) hp.n_layer + 1);

    m->devices_.push_back(cpu);
    for (auto * g : gpus) m->devices_.push_back(g);

    // layer -> device (single-GPU for now; multi-GPU split is a follow-up: even split by layer)
    m->dev_layer_.assign(hp.n_layer, cpu);
    const int first_gpu_layer = (int) hp.n_layer - n_gpu_layers;
    for (int il = 0; il < (int) hp.n_layer; il++) {
        if (il >= first_gpu_layer && !gpus.empty()) {
            // spread across GPUs proportionally to free memory (simple: round-robin contiguous chunks)
            size_t idx = gpus.size() == 1 ? 0 : (size_t) ((il - first_gpu_layer) * gpus.size() / std::max(1, n_gpu_layers));
            m->dev_layer_[il] = gpus[std::min(idx, gpus.size() - 1)];
        }
    }
    m->dev_input_  = cpu;
    m->dev_output_ = (n_gpu_layers > (int) hp.n_layer && !gpus.empty()) ? gpus.back() : cpu;

    // ---- create tensors ----
    m->layers.resize(hp.n_layer);
    TensorCreatorImpl tc(*m, f);
    const bool mmap_ok = f.mmapped();
    // CPU "extra" buffer types (weight repacking for faster matmul; same numerics as llama.cpp's default)
    std::vector<ggml_backend_buffer_type_t> cpu_extra_bufts;
    if (cfg.use_extra_bufts) {
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra) for (ggml_backend_buffer_type_t * e = get_extra(cpu); e && *e; ++e) cpu_extra_bufts.push_back(*e);
        LOG_DBG("model", "cpu extra buffer types: %zu (proc=%p)", cpu_extra_bufts.size(), (void *) get_extra);
        for (auto * b : cpu_extra_bufts) LOG_DBG("model", "  extra buft: %s", ggml_backend_buft_name(b));
    }
    size_t n_repacked = 0;
    tc.select_buft = [&](int layer, const std::string & name, const GgufTensorInfo & info) -> ggml_backend_buffer_type_t {
        ggml_backend_dev_t dev;
        if (layer >= 0) dev = m->dev_layer_[layer];
        else if (name.rfind("token_embd", 0) == 0) dev = m->dev_input_;
        else dev = m->dev_output_;
        ggml_backend_buffer_type_t def = ggml_backend_dev_buffer_type(dev);
        if (dev != cpu || cpu_extra_bufts.empty()) return def;
        // only matmul weights qualify: 2-D (mul_mat) or 3-D expert stacks (mul_mat_id); embeddings use get_rows
        const bool is_weight = name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0;
        if (!is_weight || name.rfind("token_embd", 0) == 0 || name.find("norm") != std::string::npos || info.n_dims < 2) return def;
        for (auto * buft : cpu_extra_bufts) {
            ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, 0);
            ggml_tensor * w = ggml_new_tensor(ctx, info.type, info.n_dims, info.ne);
            ggml_tensor * op;
            if (info.n_dims == 3) {
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.ne[0], hp.n_expert_used ? hp.n_expert_used : 1, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, hp.n_expert_used ? hp.n_expert_used : 1, 512);
                op = ggml_mul_mat_id(ctx, w, b, ids);
            } else {
                op = ggml_mul_mat(ctx, w, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, info.ne[0], 512));
            }
            for (int i = 0; i < GGML_MAX_SRC; i++) if (op->src[i]) op->src[i]->buffer = buf;
            const bool ok = ggml_backend_dev_supports_op(dev, op);
            LOG_TRC("model", "repack probe %s type=%s buft=%s -> %d", name.c_str(), ggml_type_name(info.type), ggml_backend_buft_name(buft), (int) ok);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            if (ok) { n_repacked++; return buft; }
        }
        return def;
    };
    m->arch_->create_tensors(*m, tc);
    if (n_repacked) LOG_INF("model", "%zu weight tensors use CPU repacked buffers (disable with --no-repack)", n_repacked);

    // warn about tensors in the file that we did not use
    for (const auto & t : f.tensors()) {
        if (!m->tensors_.count(t.name)) LOG_WRN("model", "unused tensor in file: %s", t.name.c_str());
    }

    // ---- allocate buffers & upload ----
    size_t n_done = 0;
    const size_t n_total = m->tensors_.size();
    for (auto * buft : tc.ctx_order_) {
        ggml_context * ctx = tc.ctx_map_[buft];
        ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
        const bool is_cpu_buft = buft_dev && ggml_backend_dev_type(buft_dev) == GGML_BACKEND_DEVICE_TYPE_CPU
                                 && buft == ggml_backend_dev_buffer_type(buft_dev);
        bool from_host_ptr = false;
        if (mmap_ok && is_cpu_buft && buft_dev) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(buft_dev, &props);
            from_host_ptr = props.caps.buffer_from_host_ptr;
        }
        if (from_host_ptr) {
            // zero-copy: tensors point straight into the mmap'd file
            size_t first = SIZE_MAX, last = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                const auto * info = tc.tensor_info_[t];
                first = std::min(first, f.data_offset() + info->offset);
                last  = std::max(last, f.data_offset() + info->offset + info->nbytes);
            }
            if (first == SIZE_MAX) continue;
            size_t max_size = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) max_size = std::max(max_size, ggml_nbytes(t));
            ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(buft_dev, (char *) f.map_addr() + first, last - first, max_size);
            if (!buf) throw std::runtime_error("failed to create host-ptr buffer");
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            m->bufs_.push_back(buf);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                const auto * info = tc.tensor_info_[t];
                ggml_backend_tensor_alloc(buf, t, (char *) f.map_addr() + f.data_offset() + info->offset);
                if (progress && !progress((float) ++n_done / n_total)) throw std::runtime_error("model load cancelled");
            }
            LOG_DBG("model", "buffer %s: %.2f MiB (mmap, zero-copy)", ggml_backend_buft_name(buft), (last - first) / 1048576.0);
        } else {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) throw std::runtime_error(std::string("failed to allocate buffer on ") + ggml_backend_buft_name(buft));
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            m->bufs_.push_back(buf);
            std::vector<uint8_t> tmp;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                const auto * info = tc.tensor_info_[t];
                if (const void * src = f.tensor_data(*info)) {
                    ggml_backend_tensor_set(t, src, 0, info->nbytes);
                } else {
                    tmp.resize(info->nbytes);
                    f.read_tensor(*info, tmp.data());
                    ggml_backend_tensor_set(t, tmp.data(), 0, info->nbytes);
                }
                if (progress && !progress((float) ++n_done / n_total)) throw std::runtime_error("model load cancelled");
            }
            LOG_DBG("model", "buffer %s: %.2f MiB", ggml_backend_buft_name(buft), ggml_backend_buffer_get_size(buf) / 1048576.0);
        }
    }

    LOG_INF("model", "loaded %s (%s, %s, %.2f GiB) in %.2fs; layers on GPU: %d/%u",
            hp.name.empty() ? path.c_str() : hp.name.c_str(), m->size_label().c_str(), m->ftype_.c_str(),
            m->total_bytes_ / 1073741824.0, (ggml_time_us() - t0) / 1e6, std::min(n_gpu_layers, (int) hp.n_layer), hp.n_layer);
    return m;
}

// ---------------------------------------------------------------------------------------------
// ArchRegistry
// ---------------------------------------------------------------------------------------------
ArchRegistry & ArchRegistry::instance() { static ArchRegistry r; return r; }
void ArchRegistry::add(const std::string & name, Factory f) { archs_[name] = f(); }
const ArchDef * ArchRegistry::get(const std::string & name) const {
    auto it = archs_.find(name);
    return it == archs_.end() ? nullptr : it->second.get();
}
std::vector<std::string> ArchRegistry::names() const {
    std::vector<std::string> out;
    for (auto & kv : archs_) out.push_back(kv.first);
    return out;
}

} // namespace iian
