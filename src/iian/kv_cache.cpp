#include "iian/kv_cache.h"
#include "iian/hparams.h"
#include "iian/log.h"
#include "iian/model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace iian {

// ---------------------------------------------------------------------------------------------
// hashing (xxh3-like mixing; speed matters, security does not — tokens are verified on hit)
// ---------------------------------------------------------------------------------------------
static inline uint64_t mix64(uint64_t h) {
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    return h;
}
uint64_t hash_block(uint64_t parent_hash, const token_t * tokens, uint32_t n, uint64_t extra) {
    uint64_t h = mix64(parent_hash ^ 0x9E3779B97F4A7C15ULL) ^ mix64(extra + 0x632BE59BD9B4E019ULL);
    for (uint32_t i = 0; i < n; i++) h = mix64(h ^ ((uint64_t) (uint32_t) tokens[i] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2)));
    if (h == HASH_NONE) h = 1;
    return h;
}

// ---------------------------------------------------------------------------------------------
// BlockPool
// ---------------------------------------------------------------------------------------------
BlockPool::BlockPool(uint32_t num_blocks, uint32_t block_size, bool enable_caching)
    : blocks_(num_blocks), block_tokens_((size_t) num_blocks * block_size, TOKEN_NULL), block_size_(block_size), caching_(enable_caching) {
    for (uint32_t i = 0; i < num_blocks; i++) list_push_back((int32_t) i);
}

void BlockPool::list_remove(int32_t b) {
    Block & x = blocks_[b];
    if (!x.in_free) return;
    if (x.prev >= 0) blocks_[x.prev].next = x.next; else head_ = x.next;
    if (x.next >= 0) blocks_[x.next].prev = x.prev; else tail_ = x.prev;
    x.prev = x.next = -1;
    x.in_free = false;
    n_free_--;
}
void BlockPool::list_push_front(int32_t b) {
    Block & x = blocks_[b];
    x.prev = -1; x.next = head_;
    if (head_ >= 0) blocks_[head_].prev = b; else tail_ = b;
    head_ = b; x.in_free = true; n_free_++;
}
void BlockPool::list_push_back(int32_t b) {
    Block & x = blocks_[b];
    x.next = -1; x.prev = tail_;
    if (tail_ >= 0) blocks_[tail_].next = b; else head_ = b;
    tail_ = b; x.in_free = true; n_free_++;
}
int32_t BlockPool::list_pop_front() {
    int32_t b = head_;
    if (b < 0) return -1;
    list_remove(b);
    return b;
}
void BlockPool::evict_hash(int32_t b) {
    Block & x = blocks_[b];
    if (x.hash == HASH_NONE) return;
    auto range = hash_to_block_.equal_range(x.hash);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == b) { hash_to_block_.erase(it); break; }
    }
    x.hash = HASH_NONE;
    stats_.evictions++;
}

int32_t BlockPool::get_cached_block(uint64_t hash, const token_t * tokens) const {
    if (!caching_) return -1;
    auto range = hash_to_block_.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
        const int32_t b = it->second;
        if (memcmp(&block_tokens_[(size_t) b * block_size_], tokens, block_size_ * sizeof(token_t)) == 0) return b;
    }
    return -1;
}

void BlockPool::touch(int32_t b) {
    Block & x = blocks_[b];
    if (x.ref_cnt == 0) list_remove(b);
    x.ref_cnt++;
}

void BlockPool::get_new_blocks(uint32_t n, std::vector<int32_t> & out) {
    if (n > n_free_) throw std::runtime_error("BlockPool: out of blocks");
    for (uint32_t i = 0; i < n; i++) {
        int32_t b = list_pop_front();
        Block & x = blocks_[b];
        if (x.hash != HASH_NONE) evict_hash(b);
        x.ref_cnt = 1;
        out.push_back(b);
    }
}

void BlockPool::free_blocks(const std::vector<int32_t> & blocks) {
    std::vector<int32_t> uncached, cached;
    for (int32_t b : blocks) {
        Block & x = blocks_[b];
        if (x.ref_cnt <= 0) throw std::runtime_error("BlockPool: double free of block " + std::to_string(b));
        if (--x.ref_cnt == 0) {
            if (x.hash == HASH_NONE || !caching_) uncached.push_back(b); else cached.push_back(b);
        }
    }
    // uncached blocks: reuse first (LIFO for locality); cached: keep for prefix reuse (LRU at the back)
    for (auto it = uncached.rbegin(); it != uncached.rend(); ++it) list_push_front(*it);
    for (int32_t b : cached) list_push_back(b);
}

void BlockPool::cache_block(int32_t b, uint64_t hash, const token_t * tokens) {
    if (!caching_) return;
    Block & x = blocks_[b];
    if (x.hash != HASH_NONE) return;   // already cached
    x.hash = hash;
    memcpy(&block_tokens_[(size_t) b * block_size_], tokens, block_size_ * sizeof(token_t));
    hash_to_block_.emplace(hash, b);
}

bool BlockPool::reset_prefix_cache() {
    if (n_free_ != blocks_.size()) return false;
    for (size_t b = 0; b < blocks_.size(); b++) blocks_[b].hash = HASH_NONE;
    hash_to_block_.clear();
    return true;
}

// ---------------------------------------------------------------------------------------------
// PagedKVCache
// ---------------------------------------------------------------------------------------------
PagedKVCache::PagedKVCache(const Model & model, const KVCacheConfig & cfg)
    : model_(model), cfg_(cfg), block_size_(cfg.block_size),
      num_cells_(cfg.num_blocks ? cfg.num_blocks * cfg.block_size : ((cfg.max_cells + cfg.block_size - 1) / cfg.block_size) * cfg.block_size),
      pool_(num_cells_ / cfg.block_size, cfg.block_size, cfg.enable_prefix_caching) {
    const HParams & hp = model.hparams();
    if (num_cells_ == 0) throw std::runtime_error("KV cache size is zero");

    struct BuftCmp { bool operator()(ggml_backend_buffer_type_t a, ggml_backend_buffer_type_t b) const { return strcmp(ggml_backend_buft_name(a), ggml_backend_buft_name(b)) < 0; } };
    std::map<ggml_backend_buffer_type_t, ggml_context *, BuftCmp> ctx_map;
    auto ctx_for = [&](ggml_backend_buffer_type_t buft) {
        auto it = ctx_map.find(buft);
        if (it != ctx_map.end()) return it->second;
        ggml_init_params p = { 2 * hp.n_layer * ggml_tensor_overhead() + 1024, nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ctx_map[buft] = ctx;
        ctxs_.push_back(ctx);
        return ctx;
    };

    k_l_.assign(hp.n_layer, nullptr);
    v_l_.assign(hp.n_layer, nullptr);
    for (uint32_t il = 0; il < hp.n_layer; il++) {
        if (!hp.has_kv(il)) continue;
        ggml_backend_dev_t dev = cfg.offload ? model.dev_layer(il) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        ggml_context * ctx = ctx_for(buft);
        ggml_tensor * k = ggml_new_tensor_2d(ctx, cfg.type_k, hp.n_embd_k_gqa(il), num_cells_);
        ggml_tensor * v = ggml_new_tensor_2d(ctx, cfg.type_v, hp.n_embd_v_gqa(il), num_cells_);
        ggml_format_name(k, "cache_k_l%u", il);
        ggml_format_name(v, "cache_v_l%u", il);
        k_l_[il] = k;
        v_l_[il] = v;
    }
    for (auto & kv : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(kv.second, kv.first);
        if (!buf) throw std::runtime_error(std::string("failed to allocate KV cache on ") + ggml_backend_buft_name(kv.first));
        ggml_backend_buffer_clear(buf, 0);
        bufs_.push_back(buf);
        bytes_ += ggml_backend_buffer_get_size(buf);
        LOG_INF("kv", "%s KV buffer: %.2f MiB (%u cells, block_size=%u, %u blocks, K=%s V=%s)",
                ggml_backend_buft_name(kv.first), ggml_backend_buffer_get_size(buf) / 1048576.0, num_cells_, block_size_,
                pool_.num_blocks(), ggml_type_name(cfg.type_k), ggml_type_name(cfg.type_v));
    }
}

PagedKVCache::~PagedKVCache() {
    for (auto * b : bufs_) ggml_backend_buffer_free(b);
    for (auto * c : ctxs_) ggml_free(c);
}

ggml_type PagedKVCache::type_k() const { return cfg_.type_k; }
ggml_type PagedKVCache::type_v() const { return cfg_.type_v; }

void PagedKVCache::update_hashes(std::vector<uint64_t> & hashes, const std::vector<token_t> & tokens, uint64_t extra) const {
    const uint32_t n_full = (uint32_t) tokens.size() / block_size_;
    while (hashes.size() < n_full) {
        const uint32_t i = (uint32_t) hashes.size();
        const uint64_t parent = i == 0 ? HASH_NONE : hashes[i - 1];
        hashes.push_back(hash_block(parent, tokens.data() + (size_t) i * block_size_, block_size_, extra));
    }
}

uint32_t PagedKVCache::get_computed_blocks(const std::vector<token_t> & tokens, const std::vector<uint64_t> & hashes,
                                           std::vector<int32_t> & blocks) const {
    if (!cfg_.enable_prefix_caching || tokens.size() < 2) return 0;
    const uint32_t max_hit = (uint32_t) tokens.size() - 1;   // always recompute the last token
    const uint32_t n_blocks_max = max_hit / block_size_;
    uint32_t n = 0;
    for (uint32_t i = 0; i < n_blocks_max && i < hashes.size(); i++) {
        int32_t b = pool_.get_cached_block(hashes[i], tokens.data() + (size_t) i * block_size_);
        if (b < 0) break;
        blocks.push_back(b);
        n++;
    }
    return n * block_size_;
}

bool PagedKVCache::allocate_slots(KVRequestState & st, uint32_t n_total_tokens, const std::vector<int32_t> & computed_blocks) {
    const uint32_t n_blocks_needed = (n_total_tokens + block_size_ - 1) / block_size_;
    const uint32_t n_have = (uint32_t) st.block_ids.size() + (st.block_ids.empty() ? (uint32_t) computed_blocks.size() : 0);
    const uint32_t n_new = n_blocks_needed > n_have ? n_blocks_needed - n_have : 0;
    // cached-but-free blocks we are about to touch also come out of the free list
    uint32_t n_touch_free = 0;
    if (st.block_ids.empty()) for (int32_t b : computed_blocks) if (pool_.ref_count(b) == 0) n_touch_free++;
    if (n_new + n_touch_free > pool_.num_free()) return false;

    if (st.block_ids.empty() && !computed_blocks.empty()) {
        for (int32_t b : computed_blocks) { pool_.touch(b); st.block_ids.push_back(b); }
        st.num_cached_blocks = (uint32_t) computed_blocks.size();
    }
    if (n_new) pool_.get_new_blocks(n_new, st.block_ids);
    return true;
}

void PagedKVCache::cache_blocks(KVRequestState & st, const std::vector<token_t> & tokens, uint32_t n_tokens_to_cache, const std::vector<uint64_t> & hashes) {
    if (!cfg_.enable_prefix_caching) return;
    const uint32_t n_full = std::min<uint32_t>(n_tokens_to_cache / block_size_, (uint32_t) hashes.size());
    for (uint32_t i = st.num_cached_blocks; i < n_full && i < st.block_ids.size(); i++) {
        pool_.cache_block(st.block_ids[i], hashes[i], tokens.data() + (size_t) i * block_size_);
    }
    st.num_cached_blocks = std::max(st.num_cached_blocks, std::min<uint32_t>(n_full, (uint32_t) st.block_ids.size()));
}

void PagedKVCache::free(KVRequestState & st) {
    if (st.block_ids.empty()) return;
    std::vector<int32_t> rev(st.block_ids.rbegin(), st.block_ids.rend());
    pool_.free_blocks(rev);
    st.block_ids.clear();
    st.num_cached_blocks = 0;
}

ggml_tensor * PagedKVCache::k_view(ggml_context * ctx, int il, uint32_t start, uint32_t n) const {
    const HParams & hp = model_.hparams();
    ggml_tensor * k = k_l_[il];
    return ggml_view_3d(ctx, k, hp.n_embd_head_k, hp.n_head_kv[il], n,
                        ggml_row_size(k->type, hp.n_embd_head_k), ggml_row_size(k->type, hp.n_embd_k_gqa(il)),
                        (size_t) start * ggml_row_size(k->type, hp.n_embd_k_gqa(il)));
}
ggml_tensor * PagedKVCache::v_view(ggml_context * ctx, int il, uint32_t start, uint32_t n) const {
    const HParams & hp = model_.hparams();
    ggml_tensor * v = v_l_[il];
    return ggml_view_3d(ctx, v, hp.n_embd_head_v, hp.n_head_kv[il], n,
                        ggml_row_size(v->type, hp.n_embd_head_v), ggml_row_size(v->type, hp.n_embd_v_gqa(il)),
                        (size_t) start * ggml_row_size(v->type, hp.n_embd_v_gqa(il)));
}

} // namespace iian
