/*
 * libgemma_hw — stable C ABI for Python (ctypes) and native benchmarks.
 *
 * IOSurface handles are IOSurfaceRef cast to void*; release with gemma_iosurface_release.
 */

#ifndef GEMMA_HW_H
#define GEMMA_HW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEMMA_HW_VERSION_STRING "1.0.0"

typedef struct GemmaHWCapabilities {
    int is_darwin;
    int is_apple_silicon;
    int has_accelerate_sgemm;
    int iosurface_create_ok;
    int metal_device_present;
    int sysctl_sme;
    int sysctl_sme2;
    int ane_named_class_hits;
    int ane_known_private_loaded;
    char cpu_brand[128];
} GemmaHWCapabilities;

/* Fills caps; safe to call from any thread. On non-Darwin, zeros the struct. */
void gemma_hw_capabilities_fill(GemmaHWCapabilities *caps);

/*
 * Row-major SGEMM: C = alpha * A*B + beta * C
 * A is M×K, B is K×N, C is M×N. Returns 0 on success, -1 on unsupported platform.
 */
int gemma_hw_sgemm_row_major(int m, int n, int k,
                              float alpha, const float *a, int lda,
                              const float *b, int ldb,
                              float beta, float *c, int ldc);

/* IOSurface tensor slab (same layout as iosurface_bridge.m). Returns 0 on success. */
int gemma_iosurface_create_packed(int width, int height, int bytes_per_element,
                                  void **out_surface);
void gemma_iosurface_release(void *surface);
int gemma_iosurface_lock(void *surface, uint32_t options);
int gemma_iosurface_unlock(void *surface, uint32_t options);
void *gemma_iosurface_get_base_address(void *surface);
size_t gemma_iosurface_get_alloc_size(void *surface);
int gemma_iosurface_get_bytes_per_row(void *surface);

/*
 * Runs one Metal compute dispatch on IOSurface-backed buffers (newBufferWithBytesNoCopy).
 * Returns 0 on success; negative on failure (no GPU, shader compile, wrong result).
 */
int gemma_hw_iosurface_metal_vec_mul_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA_HW_H */
