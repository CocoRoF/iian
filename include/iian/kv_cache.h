#pragma once
// Paged KV cache (vLLM-style) on top of ggml tensors.
//
//   * The cache is a flat pool of `num_blocks * block_size` cells per layer (K and V tensors of shape
//     [n_embd_gqa, n_cells]). A request owns a *block table*: an ordered list of block ids.
//   * Token i of a request lives in cell  block_table[i / block_size] * block_size + i % block_size.
//   * Automatic prefix caching: full blocks are hashed (chained hash of parent + token ids); freed blocks
//     stay in an LRU free list *with their hash* until reused, so identical prefixes are re-hit for free.
//     On a hit the stored token ids are verified, so a hash collision can never return wrong KV.
//   * Ref-counted: shared prefix blocks are used by many requests simultaneously (zero copy).
#include "iian/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace iian {

class Model;

struct KVCacheConfig {
    uint32_t block_size = 16;
    uint32_t num_blocks = 0;          // 0 -> derived from max_cells
    uint32_t max_cells  = 0;          // total cells (e.g. n_ctx * n_seq); ignored if num_blocks != 0
    ggml_type type_k;
    ggml_type type_v;
    bool enable_prefix_caching = true;
    bool offload = true;              // place cache on the layer's device
};

// Chained block hash. 64-bit; correctness is guaranteed by token verification on hit.
uint64_t hash_block(uint64_t parent_hash, const token_t * tokens, uint32_t n, uint64_t extra = 0);
constexpr uint64_t HASH_NONE = 0;

class BlockPool {
public:
    explicit BlockPool(uint32_t num_blocks, uint32_t block_size, bool enable_caching);

    uint32_t num_blocks() const { return (uint32_t) blocks_.size(); }
    uint32_t num_free() const { return n_free_; }
    uint32_t num_cached() const { return (uint32_t) hash_to_block_.size(); }
    uint32_t block_size() const { return block_size_; }

    // Prefix cache lookup: block with this hash whose stored tokens equal `tokens` (block_size of them).
    int32_t get_cached_block(uint64_t hash, const token_t * tokens) const;
    // Take a reference; a block sitting in the free list (ref==0) is removed from it.
    void touch(int32_t block);
    // Pop n fresh blocks (ref=1, hash cleared). Caller must check num_free() first.
    void get_new_blocks(uint32_t n, std::vector<int32_t> & out);
    // Release references. Blocks reaching ref==0 go back to the free list: uncached ones to the
    // front (reused first), cached ones to the back (kept for prefix reuse, LRU).
    void free_blocks(const std::vector<int32_t> & blocks);
    // Register a full block as cached (hash + tokens).
    void cache_block(int32_t block, uint64_t hash, const token_t * tokens);
    bool is_cached(int32_t block) const { return blocks_[block].hash != HASH_NONE; }
    int32_t ref_count(int32_t block) const { return blocks_[block].ref_cnt; }
    // Drop all cached entries (only succeeds when nothing is in use).
    bool reset_prefix_cache();

    struct Stats { uint64_t cache_queries = 0, cache_hits = 0, evictions = 0; };
    const Stats & stats() const { return stats_; }
    void stat_query(uint32_t queries, uint32_t hits) { stats_.cache_queries += queries; stats_.cache_hits += hits; }

private:
    struct Block {
        int32_t  ref_cnt = 0;
        uint64_t hash    = HASH_NONE;
        int32_t  prev = -1, next = -1;   // free-list links
        bool     in_free = false;
    };
    void list_remove(int32_t b);
    void list_push_front(int32_t b);
    void list_push_back(int32_t b);
    int32_t list_pop_front();
    void evict_hash(int32_t b);

    std::vector<Block> blocks_;
    std::vector<token_t> block_tokens_;   // [num_blocks * block_size] tokens of cached blocks
    std::unordered_multimap<uint64_t, int32_t> hash_to_block_;
    int32_t head_ = -1, tail_ = -1;
    uint32_t n_free_ = 0;
    uint32_t block_size_;
    bool caching_;
    Stats stats_;
};

// Per-request KV state, owned by the scheduler's request bookkeeping.
struct KVRequestState {
    std::vector<int32_t>  block_ids;      // block table
    std::vector<uint64_t> block_hashes;   // chained hashes of full blocks, computed incrementally
    uint32_t num_cached_blocks = 0;       // how many leading blocks were registered in the pool
};

class PagedKVCache {
public:
    PagedKVCache(const Model & model, const KVCacheConfig & cfg);
    ~PagedKVCache();

    uint32_t block_size() const { return block_size_; }
    uint32_t num_blocks() const { return pool_.num_blocks(); }
    uint32_t num_cells() const { return num_cells_; }
    // per-sequence context length the engine enforces (max_model_len); used for LongRoPE factor selection
    void set_max_seq_len(uint32_t n) { max_seq_len_ = n; }
    uint32_t max_seq_len() const { return max_seq_len_ ? max_seq_len_ : num_cells_; }
    size_t   bytes() const { return bytes_; }
    BlockPool & pool() { return pool_; }
    const BlockPool & pool() const { return pool_; }
    bool prefix_caching() const { return cfg_.enable_prefix_caching; }
    ggml_type type_k() const;
    ggml_type type_v() const;
    float usage() const { return 1.0f - (float) pool_.num_free() / pool_.num_blocks(); }

    // ---- scheduler-facing (vLLM KVCacheManager semantics) ----
    // Longest cached prefix of `tokens` (full blocks only, at most n_tokens-1 tokens so that the
    // last token is always recomputed to produce logits). Blocks are NOT referenced yet.
    // Returns number of matched tokens; matched block ids are appended to `blocks`.
    uint32_t get_computed_blocks(const std::vector<token_t> & tokens, const std::vector<uint64_t> & hashes,
                                 std::vector<int32_t> & blocks) const;
    // Ensure the request has cells for `n_total_tokens` (= computed + new). `computed_blocks`
    // (from get_computed_blocks) are touched and prepended if the request has no blocks yet.
    // Returns false (allocating nothing) if the pool cannot satisfy it.
    bool allocate_slots(KVRequestState & st, uint32_t n_total_tokens, const std::vector<int32_t> & computed_blocks);
    // Register full blocks covering tokens[0, n_tokens_to_cache) in the prefix cache.
    void cache_blocks(KVRequestState & st, const std::vector<token_t> & tokens, uint32_t n_tokens_to_cache, const std::vector<uint64_t> & hashes);
    // Return all blocks (in reverse order so the tail is evicted first).
    void free(KVRequestState & st);
    // Update chained hashes for full blocks of `tokens` (idempotent, incremental).
    void update_hashes(std::vector<uint64_t> & hashes, const std::vector<token_t> & tokens, uint64_t extra = 0) const;

    int64_t cell_of(const KVRequestState & st, uint32_t token_idx) const {
        return (int64_t) st.block_ids[token_idx / block_size_] * block_size_ + token_idx % block_size_;
    }

    // ---- graph-facing ----
    ggml_tensor * k_full(int il) const { return k_l_.at(il); }   // [n_embd_k_gqa, n_cells]
    ggml_tensor * v_full(int il) const { return v_l_.at(il); }   // [n_embd_v_gqa, n_cells]
    // Views over cells [start, start+n) shaped [head_dim, n_head_kv, n]
    ggml_tensor * k_view(ggml_context * ctx, int il, uint32_t start, uint32_t n) const;
    ggml_tensor * v_view(ggml_context * ctx, int il, uint32_t start, uint32_t n) const;

private:
    const Model & model_;
    KVCacheConfig cfg_;
    uint32_t block_size_;
    uint32_t num_cells_;
    uint32_t max_seq_len_ = 0;
    size_t   bytes_ = 0;
    BlockPool pool_;
    std::vector<ggml_tensor *> k_l_, v_l_;
    std::vector<ggml_context *> ctxs_;
    std::vector<ggml_backend_buffer_t> bufs_;
};

} // namespace iian
