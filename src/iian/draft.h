#pragma once
// Draft-model speculative decoding: a small model proposes tokens that the target verifies.
// The draft model keeps its own paged KV cache per request, kept in sync with the accepted tokens.
#include "iian/graph.h"
#include "iian/kv_cache.h"
#include "iian/model.h"
#include "iian/request.h"

#include "ggml-backend.h"

#include <memory>
#include <string>
#include <vector>

namespace iian {

struct DraftConfig {
    std::string model_path;
    uint32_t n_draft = 5;          // tokens proposed per step
    uint32_t max_cells = 0;        // draft KV cells (default: same count as the target cache)
    int n_threads = -1;
    DeviceConfig device;
};

class DraftModel {
public:
    DraftModel(const Model & target, const DraftConfig & cfg, uint32_t max_cells, uint32_t target_max_len);
    ~DraftModel();

    // Sync the draft KV with each request's accepted tokens and fill r->spec_tokens (greedy drafts).
    // Requests must be in the decode phase (num_computed_tokens == n_tokens - 1). Batched across requests.
    void draft(const std::vector<Request *> & reqs);
    // Release the request's draft KV (finished / aborted).
    void release(Request & r);

    const Model & model() const { return *model_; }
    uint32_t n_draft() const { return cfg_.n_draft; }

private:
    struct GraphState;
    // Runs one micro-batch through the draft model; returns logits rows for `out_ids` order.
    void run(UBatch & ub, std::vector<float> & logits);
    bool ensure_cells(Request & r, uint32_t n_total);

    DraftConfig cfg_;
    std::shared_ptr<Model> model_;
    std::unique_ptr<PagedKVCache> kv_;
    std::vector<ggml_backend_t> backends_;
    ggml_backend_sched_t sched_ = nullptr;
    std::unique_ptr<GraphState> graph_;
    size_t max_nodes_ = 0;
};

} // namespace iian
