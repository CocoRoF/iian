#pragma once
// Model hyper-parameters. Generic fields are loaded by the loader from the standard GGUF keys;
// architectures may add arch-specific values via HParams::extra.
#include "iian/types.h"

#include <map>
#include <string>
#include <vector>

namespace iian {

enum class RopeType { NONE, NORMAL, NEOX, MROPE, VISION };
enum class RopeScaling { NONE, LINEAR, YARN, LONGROPE };
enum class NormType { RMS, LAYER };
enum class FfnAct { SILU, GELU, RELU, RELU_SQR, SWIGLU, GEGLU };

struct HParams {
    std::string arch;
    std::string name;

    uint32_t n_vocab      = 0;
    uint32_t n_ctx_train  = 0;
    uint32_t n_embd       = 0;
    uint32_t n_layer      = 0;
    uint32_t n_expert     = 0;
    uint32_t n_expert_used = 0;

    // per-layer (may vary for heterogeneous models)
    std::vector<uint32_t> n_head;
    std::vector<uint32_t> n_head_kv;
    std::vector<uint32_t> n_ff;
    std::vector<uint32_t> n_ff_exp;

    uint32_t n_embd_head_k = 0;   // head dim for K/Q
    uint32_t n_embd_head_v = 0;   // head dim for V
    uint32_t n_rot         = 0;   // rotary dims

    float f_norm_eps       = 1e-5f;
    float f_norm_rms_eps   = 1e-5f;
    float f_attention_scale = 0.0f;   // 0 -> 1/sqrt(head_dim)
    float f_logit_scale    = 0.0f;
    float f_attn_logit_softcapping  = 0.0f;
    float f_final_logit_softcapping = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_clamp_kqv      = 0.0f;
    bool  use_kq_norm      = false;
    bool  causal_attn      = true;

    // rope
    RopeType    rope_type          = RopeType::NORMAL;
    RopeScaling rope_scaling       = RopeScaling::NONE;
    float       rope_freq_base     = 10000.0f;
    float       rope_freq_scale    = 1.0f;
    uint32_t    n_ctx_orig_yarn    = 0;
    float       yarn_ext_factor    = -1.0f;   // <0 -> auto (1.0 if YARN else 0)
    float       yarn_attn_factor   = 1.0f;
    float       yarn_beta_fast     = 32.0f;
    float       yarn_beta_slow     = 1.0f;
    float       rope_yarn_log_mul  = 0.0f;
    bool        rope_finetuned     = false;

    // sliding window attention (0 = none); per-layer flag
    uint32_t n_swa = 0;
    std::vector<uint8_t> is_swa;
    // rope parameters used by SWA layers of hybrid SWA/global models (e.g. gemma3); generic layers use rope_freq_base/scale
    float rope_freq_base_swa  = 10000.0f;
    float rope_freq_scale_swa = 1.0f;

    // MoE
    float expert_weights_scale = 0.0f;
    bool  expert_weights_norm  = false;

    // arch-specific extras (string -> number) for things not covered above
    std::map<std::string, float> extra;

    // derived helpers
    uint32_t n_head_max() const;
    uint32_t n_head_kv_max() const;
    uint32_t n_embd_k_gqa(uint32_t il) const { return n_embd_head_k * n_head_kv.at(il); }
    uint32_t n_embd_v_gqa(uint32_t il) const { return n_embd_head_v * n_head_kv.at(il); }
    uint32_t n_gqa(uint32_t il) const { return n_head_kv.at(il) ? n_head.at(il) / n_head_kv.at(il) : 0; }
    bool has_kv(uint32_t il) const { return n_head_kv.at(il) > 0; }
    std::string summary() const;
};

} // namespace iian
