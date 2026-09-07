#include "iian/kv_cache.h"
#include <cstdio>
#include <vector>
using namespace iian;
#define CHECK(x) do { if (!(x)) { printf("CHECK failed: %s (line %d)\n", #x, __LINE__); return 1; } } while (0)

int main() {
    BlockPool pool(8, 4, true);
    CHECK(pool.num_free() == 8);
    std::vector<int32_t> a, b;
    pool.get_new_blocks(3, a);
    CHECK(a.size() == 3 && pool.num_free() == 5);
    token_t toks[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    uint64_t h0 = hash_block(HASH_NONE, toks, 4), h1 = hash_block(h0, toks + 4, 4), h2 = hash_block(h1, toks + 8, 4);
    CHECK(h0 != h1 && h1 != h2 && h0 != HASH_NONE);
    pool.cache_block(a[0], h0, toks); pool.cache_block(a[1], h1, toks + 4); pool.cache_block(a[2], h2, toks + 8);
    CHECK(pool.get_cached_block(h0, toks) == a[0]);
    token_t wrong[4] = {1,2,3,5};
    CHECK(pool.get_cached_block(h0, wrong) == -1);   // token verification
    // free in reverse order -> tail evicted first
    pool.free_blocks({a[2], a[1], a[0]});
    CHECK(pool.num_free() == 8);
    CHECK(pool.get_cached_block(h1, toks + 4) == a[1]);   // still cached after free
    // touch a cached free block -> leaves free list
    pool.touch(a[1]);
    CHECK(pool.num_free() == 7 && pool.ref_count(a[1]) == 1);
    pool.free_blocks({a[1]});
    // allocate 8 fresh blocks: uncached ones first, then cached (LRU order: a2, a1, a0 at the back)
    std::vector<int32_t> c;
    pool.get_new_blocks(5, c);
    CHECK(pool.get_cached_block(h0, toks) == a[0]);   // a0 still cached (evicted last)
    pool.get_new_blocks(3, c);
    CHECK(pool.num_free() == 0);
    CHECK(pool.get_cached_block(h0, toks) == -1);     // now evicted
    CHECK(pool.stats().evictions == 3);
    bool threw = false;
    try { pool.get_new_blocks(1, c); } catch (...) { threw = true; }
    CHECK(threw);
    printf("test-block-pool: ok\n");
    return 0;
}
