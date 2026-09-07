#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "iian/version.h"
#include <cstdio>
int main(int argc, char ** argv) {
    ggml_backend_load_all();
    printf("iian %s, ggml devices: %zu\n", iian::version(), ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev = ggml_backend_dev_get(i);
        size_t f, t; ggml_backend_dev_memory(dev, &f, &t);
        printf("  [%zu] %s (%s) mem %zu/%zu MB\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), f>>20, t>>20);
    }
    if (argc > 1) {
        struct gguf_init_params p = { true, nullptr };
        auto * ctx = gguf_init_from_file(argv[1], p);
        if (!ctx) { printf("bad gguf\n"); return 1; }
        printf("gguf: %lld kv, %lld tensors, arch=%s\n", (long long)gguf_get_n_kv(ctx), (long long)gguf_get_n_tensors(ctx),
               gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture")));
        gguf_free(ctx);
    }
    return 0;
}
