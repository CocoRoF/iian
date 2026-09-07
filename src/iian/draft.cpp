#include "draft.h"

#include "iian/arch.h"
#include "iian/log.h"
#include "iian/tokenizer.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace iian {

struct DraftModel::GraphState {
    std::vector<uint8_t> meta;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::unique_ptr<GraphContext> gc;
    UBatch ub;
    struct Key { uint32_t n_tokens = 0, n_outputs = 0, n_kv = 0, kv_start = 0; bool valid = false; } key;
    ~GraphState() { if (ctx) ggml_free(ctx); }
};

DraftModel::DraftModel(const Model & target, const DraftConfig & cfg, uint32_t max_cells, uint32_t target_max_len) : cfg_(cfg) {
    model_ = ModelLoader::load(cfg.model_path, cfg.device);
    const HParams & th = target.hparams(), & dh = model_->hparams();
    if (th.n_vocab != dh.n_vocab)
        throw std::runtime_error("draft model vocabulary (" + std::to_string(dh.n_vocab) + ") differs from the target's (" + std::to_string(th.n_vocab) + "); speculative decoding needs a draft with the same tokenizer");
    for (auto * dev : model_->devices()) {
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) throw std::runtime_error("draft: failed to init backend");
        backends_.push_back(b);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            int nt = cfg.n_threads > 0 ? cfg.n_threads : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
            auto * reg = ggml_backend_dev_backend_reg(dev);
            auto set_nt = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_n_threads");
            if (set_nt) set_nt(b, nt);
        }
    }
    std::stable_partition(backends_.begin(), backends_.end(), [](ggml_backend_t b) { return ggml_backend_dev_type(ggml_backend_get_device(b)) != GGML_BACKEND_DEVICE_TYPE_CPU; });

    KVCacheConfig kc;
    kc.block_size = 16;
    kc.type_k = kc.type_v = GGML_TYPE_F16;
    kc.enable_prefix_caching = false;
    kc.max_cells = cfg.max_cells ? cfg.max_cells : max_cells;
    kv_ = std::make_unique<PagedKVCache>(*model_, kc);
    kv_->set_max_seq_len(target_max_len);

    max_nodes_ = std::max<size_t>(8192, 8 * (model_->layers.size() * 24 + 16));
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (auto * b : backends_) bufts.push_back(ggml_backend_get_default_buffer_type(b));
    sched_ = ggml_backend_sched_new(backends_.data(), bufts.data(), (int) backends_.size(), max_nodes_, false, true);
    graph_ = std::make_unique<GraphState>();
    graph_->meta.resize(ggml_tensor_overhead() * max_nodes_ + ggml_graph_overhead_custom(max_nodes_, false));
    LOG_INF("spec", "draft model %s (%s, %s) loaded: %u drafts/step, kv %.0f MiB", dh.name.c_str(), model_->size_label().c_str(), model_->ftype_name().c_str(),
            cfg.n_draft, kv_->bytes() / 1048576.0);
}

DraftModel::~DraftModel() {
    if (sched_) ggml_backend_sched_free(sched_);
    graph_.reset();
    kv_.reset();
    for (auto * b : backends_) ggml_backend_free(b);
}

void DraftModel::release(Request & r) {
    kv_->free(r.draft_kv);
    r.draft_computed = 0;
}

bool DraftModel::ensure_cells(Request & r, uint32_t n_total) {
    return kv_->allocate_slots(r.draft_kv, n_total, {});
}

void DraftModel::run(UBatch & ub, std::vector<float> & logits) {
    GraphState & g = *graph_;
    const bool reuse = g.key.valid && g.key.n_tokens == ub.n_tokens && g.key.n_outputs == ub.n_outputs && g.key.n_kv == ub.n_kv && g.key.kv_start == ub.kv_start;
    g.ub = std::move(ub);
    if (!reuse) {
        if (g.ctx) { ggml_free(g.ctx); g.ctx = nullptr; }
        ggml_init_params ip = { g.meta.size(), g.meta.data(), true };
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, max_nodes_, false);
        g.gc = std::make_unique<GraphContext>(g.ctx, g.gf, *model_, g.ub, *kv_, /*flash*/ true, /*paged*/ false);
        model_->arch().build_graph(*g.gc);
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) throw std::runtime_error("draft: failed to allocate graph");
        g.key = { g.ub.n_tokens, g.ub.n_outputs, g.ub.n_kv, g.ub.kv_start, true };
    }
    g.gc->set_inputs();
    if (ggml_backend_sched_graph_compute(sched_, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("draft: compute failed");
    logits.resize((size_t) g.ub.n_outputs * model_->hparams().n_vocab);
    ggml_backend_tensor_get(g.gc->t_logits, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_sched_synchronize(sched_);
}

void DraftModel::draft(const std::vector<Request *> & reqs_in) {
    const uint32_t n_vocab = model_->hparams().n_vocab;
    const Tokenizer & tok = model_->tokenizer();
    const uint32_t k = cfg_.n_draft;
    if (k == 0 || reqs_in.empty()) return;

    // requests that can be drafted for: decode phase, draft KV allocatable
    std::vector<Request *> reqs;
    for (Request * r : reqs_in) {
        if (r->is_finished() || r->is_embedding || r->grammar) continue;
        if (r->num_computed_tokens != r->n_tokens() - 1) continue;
        if (r->draft_epoch != r->num_preemptions) { release(*r); r->draft_epoch = r->num_preemptions; }
        if (r->draft_computed > r->n_tokens()) { release(*r); }   // should not happen; resync from scratch
        if (!ensure_cells(*r, r->n_tokens() + k)) { r->spec_tokens.clear(); continue; }
        reqs.push_back(r);
    }
    if (reqs.empty()) return;

    // ---- phase 1: catch up on real tokens (chunked), keeping logits of each request's last token ----
    std::vector<float> logits;
    std::vector<int32_t> last_row(reqs.size(), -1);
    std::vector<std::vector<float>> last_logits(reqs.size());
    const uint32_t budget = 512;
    while (true) {
        UBatch ub;
        std::vector<size_t> in_batch;
        int64_t cmin = INT64_MAX, cmax = -1;
        uint32_t used = 0;
        for (size_t i = 0; i < reqs.size() && used < budget; i++) {
            Request * r = reqs[i];
            if (r->draft_computed >= r->n_tokens()) continue;
            const uint32_t n_new = std::min<uint32_t>(r->n_tokens() - r->draft_computed, budget - used);
            UBatch::SeqInfo seq;
            const uint32_t n_ctx = r->draft_computed + n_new;
            for (uint32_t t = 0; t < n_ctx; t++) { int64_t c = kv_->cell_of(r->draft_kv, t); seq.cells.push_back(c); seq.cell_pos.push_back((pos_t) t); cmin = std::min(cmin, c); cmax = std::max(cmax, c); }
            for (uint32_t t = r->draft_computed; t < n_ctx; t++) { ub.tokens.push_back(r->tokens[t]); ub.pos.push_back((pos_t) t); ub.seq_idx.push_back((int32_t) ub.seqs.size()); ub.slots.push_back(kv_->cell_of(r->draft_kv, t)); }
            if (n_ctx == r->n_tokens()) { last_row[i] = (int32_t) ub.out_ids.size(); ub.out_ids.push_back((int32_t) ub.tokens.size() - 1); }
            ub.seqs.push_back(std::move(seq));
            in_batch.push_back(i);
            r->draft_computed = n_ctx;
            used += n_new;
        }
        if (in_batch.empty()) break;
        ub.n_tokens = (uint32_t) ub.tokens.size();
        ub.n_outputs = (uint32_t) ub.out_ids.size();
        if (ub.n_outputs == 0) { ub.out_ids.push_back(0); ub.n_outputs = 1; }
        ub.kv_start = (uint32_t) ((cmin / 256) * 256);
        ub.n_kv = std::min<uint32_t>(kv_->num_cells() - ub.kv_start, (uint32_t) (((cmax + 1 - ub.kv_start + 255) / 256) * 256));
        run(ub, logits);
        for (size_t i : in_batch) if (last_row[i] >= 0 && last_logits[i].empty()) last_logits[i].assign(logits.begin() + (size_t) last_row[i] * n_vocab, logits.begin() + (size_t) (last_row[i] + 1) * n_vocab);
    }

    // ---- phase 2: k greedy draft rounds (batched decode) ----
    std::vector<bool> active(reqs.size(), true);
    for (size_t i = 0; i < reqs.size(); i++) { reqs[i]->spec_tokens.clear(); if (last_logits[i].empty()) active[i] = false; }
    auto argmax = [&](const float * l) { uint32_t b = 0; for (uint32_t t = 1; t < n_vocab; t++) if (l[t] > l[b]) b = t; return (token_t) b; };
    for (size_t i = 0; i < reqs.size(); i++) if (active[i]) {
        token_t d = argmax(last_logits[i].data());
        reqs[i]->spec_tokens.push_back(d);
        if (tok.is_eog(d)) active[i] = false;
    }
    for (uint32_t round = 1; round < k; round++) {
        UBatch ub;
        std::vector<size_t> in_batch;
        int64_t cmin = INT64_MAX, cmax = -1;
        for (size_t i = 0; i < reqs.size(); i++) {
            if (!active[i]) continue;
            Request * r = reqs[i];
            const uint32_t pos = r->n_tokens() + round - 1;          // position of the draft token being fed
            UBatch::SeqInfo seq;
            for (uint32_t t = 0; t <= pos; t++) { int64_t c = kv_->cell_of(r->draft_kv, t); seq.cells.push_back(c); seq.cell_pos.push_back((pos_t) t); cmin = std::min(cmin, c); cmax = std::max(cmax, c); }
            ub.tokens.push_back(r->spec_tokens.back()); ub.pos.push_back((pos_t) pos); ub.seq_idx.push_back((int32_t) ub.seqs.size()); ub.slots.push_back(kv_->cell_of(r->draft_kv, pos));
            ub.out_ids.push_back((int32_t) ub.tokens.size() - 1);
            ub.seqs.push_back(std::move(seq));
            in_batch.push_back(i);
        }
        if (in_batch.empty()) break;
        ub.n_tokens = (uint32_t) ub.tokens.size();
        ub.n_outputs = (uint32_t) ub.out_ids.size();
        ub.kv_start = (uint32_t) ((cmin / 256) * 256);
        ub.n_kv = std::min<uint32_t>(kv_->num_cells() - ub.kv_start, (uint32_t) (((cmax + 1 - ub.kv_start + 255) / 256) * 256));
        run(ub, logits);
        for (size_t j = 0; j < in_batch.size(); j++) {
            const size_t i = in_batch[j];
            token_t d = argmax(logits.data() + j * n_vocab);
            reqs[i]->spec_tokens.push_back(d);
            if (tok.is_eog(d)) active[i] = false;
        }
    }
}

} // namespace iian
