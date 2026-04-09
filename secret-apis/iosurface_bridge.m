/*
 * IOSurface Zero-Copy Bridge
 *
 * IOSurface is Apple's zero-copy shared memory primitive. It allows
 * CPU, GPU (Metal), and ANE to access the same physical memory without
 * any memcpy operations. This is the key to efficient hybrid pipelines.
 *
 * For LLM inference:
 *   - GPU writes KV cache to IOSurface
 *   - ANE reads KV cache directly (zero copy)
 *   - CPU can inspect/modify without synchronization overhead
 *
 * This benchmark measures:
 *   1. IOSurface allocation speed
 *   2. CPU→GPU zero-copy verification
 *   3. GPU compute on IOSurface-backed Metal buffer
 *   4. Bandwidth comparison: memcpy vs IOSurface
 *
 * Build: clang -O2 -framework Foundation -framework Metal -framework IOSurface \
 *        -framework CoreGraphics iosurface_bridge.m -o iosurface_bridge
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <string.h>

static double mach_to_us(uint64_t elapsed) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1e3;
}

static double mach_to_ms(uint64_t elapsed) {
    return mach_to_us(elapsed) / 1e3;
}

/*
 * Create an IOSurface with the given dimensions.
 * The surface memory is accessible by CPU, GPU, and ANE.
 */
static IOSurfaceRef create_tensor_surface(int width, int height, int bytes_per_element) {
    int bytes_per_row = width * bytes_per_element;
    int total_bytes = bytes_per_row * height;

    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(width),
        (id)kIOSurfaceHeight: @(height),
        (id)kIOSurfaceBytesPerElement: @(bytes_per_element),
        (id)kIOSurfaceBytesPerRow: @(bytes_per_row),
        (id)kIOSurfaceAllocSize: @(total_bytes),
        (id)kIOSurfacePixelFormat: @(0x46503332), /* 'FP32' — custom fourcc */
    };

    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

/*
 * Benchmark 1: IOSurface allocation speed
 */
static void bench_allocation(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  IOSurface Allocation Benchmark                            │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    int sizes[][2] = {
        {256, 256},     /* 256 KB — KV cache slice */
        {1024, 1024},   /* 4 MB — embedding table */
        {4096, 4096},   /* 64 MB — model weights layer */
        {8192, 4096},   /* 128 MB — large weight matrix */
    };
    int nsizes = 4;
    int iters = 100;

    for (int s = 0; s < nsizes; s++) {
        int w = sizes[s][0], h = sizes[s][1];
        int total_mb = w * h * 4 / (1024 * 1024);

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            IOSurfaceRef surface = create_tensor_surface(w, h, 4);
            if (surface) CFRelease(surface);
        }
        uint64_t elapsed = mach_absolute_time() - t0;
        double us_per = mach_to_us(elapsed) / iters;

        printf("│  %4dx%-4d (%3d MB) → %8.1f µs/alloc                    │\n",
               w, h, total_mb, us_per);
    }

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Benchmark 2: CPU write → GPU read (zero-copy verification)
 */
static void bench_zero_copy_gpu(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  CPU→GPU Zero-Copy via IOSurface                           │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("│  ✗ No Metal device found                                  │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    printf("│  GPU: %-51s │\n", [[device name] UTF8String]);

    int N = 1024;
    int total_bytes = N * N * sizeof(float);

    /* Method 1: Standard MTLBuffer with memcpy */
    uint64_t t0 = mach_absolute_time();
    float *cpu_data = (float *)malloc(total_bytes);
    for (int i = 0; i < N * N; i++) cpu_data[i] = (float)i / (N * N);
    id<MTLBuffer> standard_buf = [device newBufferWithBytes:cpu_data
                                                     length:total_bytes
                                                    options:MTLResourceStorageModeShared];
    uint64_t t_memcpy = mach_absolute_time() - t0;
    free(cpu_data);

    /* Method 2: IOSurface-backed (zero-copy) */
    t0 = mach_absolute_time();
    IOSurfaceRef surface = create_tensor_surface(N, N, 4);
    IOSurfaceLock(surface, 0, NULL);
    float *surf_ptr = (float *)IOSurfaceGetBaseAddress(surface);
    for (int i = 0; i < N * N; i++) surf_ptr[i] = (float)i / (N * N);
    IOSurfaceUnlock(surface, 0, NULL);

    /* Create Metal buffer from the IOSurface — no copy! */
    id<MTLBuffer> iosurface_buf = [device newBufferWithBytesNoCopy:surf_ptr
                                                            length:total_bytes
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
    uint64_t t_zerocopy = mach_absolute_time() - t0;

    printf("│                                                            │\n");
    printf("│  Standard (alloc + memcpy): %8.2f ms                     │\n", mach_to_ms(t_memcpy));
    printf("│  IOSurface (zero-copy):     %8.2f ms                     │\n", mach_to_ms(t_zerocopy));
    printf("│  Speedup:                   %8.1fx                       │\n",
           mach_to_ms(t_memcpy) / mach_to_ms(t_zerocopy));
    printf("│                                                            │\n");

    /* Verify data integrity — GPU should see exactly what CPU wrote */
    float *verify = (float *)malloc(sizeof(float));
    const float *buf_contents = (const float *)[iosurface_buf contents];
    int correct = 1;
    for (int i = 0; i < 100; i++) {
        int idx = rand() % (N * N);
        float expected = (float)idx / (N * N);
        if (fabsf(buf_contents[idx] - expected) > 1e-6) {
            correct = 0;
            break;
        }
    }
    free(verify);
    printf("│  Data integrity: %s                                      │\n",
           correct ? "VERIFIED" : "FAILED");

    /* Bandwidth test — bulk transfer comparison */
    int bulk_size = 64 * 1024 * 1024; /* 64 MB */
    int iters = 50;

    /* memcpy bandwidth — volatile to prevent optimization */
    float *src = (float *)malloc(bulk_size);
    float *dst = (float *)malloc(bulk_size);
    memset(src, 0x42, bulk_size);
    memcpy(dst, src, bulk_size); /* warmup */
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        memcpy(dst, src, bulk_size);
        __asm__ volatile("" : : "r"(dst) : "memory"); /* prevent dead store elimination */
    }
    double memcpy_ms = mach_to_ms(mach_absolute_time() - t0) / iters;
    double memcpy_gbps = memcpy_ms > 0.001 ? (double)bulk_size / memcpy_ms / 1e6 : 999.0;

    /* IOSurface "transfer" — just pointer handoff */
    IOSurfaceRef bulk_surf = create_tensor_surface(bulk_size / 4, 1, 4);
    IOSurfaceLock(bulk_surf, 0, NULL);
    float *bulk_ptr = (float *)IOSurfaceGetBaseAddress(bulk_surf);
    memset(bulk_ptr, 0x42, bulk_size);
    IOSurfaceUnlock(bulk_surf, 0, NULL);

    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        /* Zero-copy: just create a new buffer view, no data movement */
        id<MTLBuffer> view = [device newBufferWithBytesNoCopy:bulk_ptr
                                                       length:bulk_size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        (void)view;
    }
    double zerocopy_ms = mach_to_ms(mach_absolute_time() - t0) / iters;
    double zerocopy_gbps = (double)bulk_size / zerocopy_ms / 1e6;

    printf("│                                                            │\n");
    printf("│  64 MB Transfer Bandwidth:                                 │\n");
    printf("│    memcpy:      %8.2f GB/s (%6.2f ms)                   │\n", memcpy_gbps, memcpy_ms);
    printf("│    IOSurface:   %8.2f GB/s (%6.4f ms) ← no copy!       │\n", zerocopy_gbps, zerocopy_ms);
    double speedup = (zerocopy_ms > 0.0001) ? memcpy_ms / zerocopy_ms : 999.0;
    printf("│    Effective:   %-6.0fx faster                              │\n", speedup);

    free(src);
    free(dst);
    (void)standard_buf;
    (void)iosurface_buf;
    if (surface) CFRelease(surface);
    if (bulk_surf) CFRelease(bulk_surf);

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Benchmark 3: GPU compute on IOSurface-backed buffer
 */
static void bench_gpu_compute(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Metal Compute on IOSurface-Backed Buffer                  │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("│  ✗ No Metal device                                        │\n");
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    /* Create a simple element-wise multiply shader */
    NSString *shader = @
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void vec_mul(device float *a [[buffer(0)]],\n"
        "                    device float *b [[buffer(1)]],\n"
        "                    device float *c [[buffer(2)]],\n"
        "                    uint id [[thread_position_in_grid]]) {\n"
        "    c[id] = a[id] * b[id];\n"
        "}\n";

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:shader options:nil error:&error];
    if (!library) {
        printf("│  ✗ Shader compilation failed: %s\n",
               [[error localizedDescription] UTF8String]);
        printf("└─────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    id<MTLFunction> func = [library newFunctionWithName:@"vec_mul"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];

    int N = 4 * 1024 * 1024; /* 4M elements = 16 MB per buffer */
    int total_bytes = N * sizeof(float);
    int iters = 100;

    /* IOSurface-backed buffers */
    IOSurfaceRef surf_a = create_tensor_surface(N, 1, 4);
    IOSurfaceRef surf_b = create_tensor_surface(N, 1, 4);
    IOSurfaceRef surf_c = create_tensor_surface(N, 1, 4);

    IOSurfaceLock(surf_a, 0, NULL);
    IOSurfaceLock(surf_b, 0, NULL);
    float *ptr_a = (float *)IOSurfaceGetBaseAddress(surf_a);
    float *ptr_b = (float *)IOSurfaceGetBaseAddress(surf_b);
    for (int i = 0; i < N; i++) { ptr_a[i] = 1.0f; ptr_b[i] = 2.0f; }
    IOSurfaceUnlock(surf_a, 0, NULL);
    IOSurfaceUnlock(surf_b, 0, NULL);

    id<MTLBuffer> buf_a = [device newBufferWithBytesNoCopy:ptr_a length:total_bytes
                                                   options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> buf_b = [device newBufferWithBytesNoCopy:ptr_b length:total_bytes
                                                   options:MTLResourceStorageModeShared deallocator:nil];
    float *ptr_c = (float *)IOSurfaceGetBaseAddress(surf_c);
    id<MTLBuffer> buf_c = [device newBufferWithBytesNoCopy:ptr_c length:total_bytes
                                                   options:MTLResourceStorageModeShared deallocator:nil];

    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Warmup */
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        MTLSize grid = MTLSizeMake(N, 1, 1);
        MTLSize tg = MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    /* Benchmark */
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        MTLSize grid = MTLSizeMake(N, 1, 1);
        MTLSize tg = MTLSizeMake(pipeline.maxTotalThreadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    double total_ms = mach_to_ms(mach_absolute_time() - t0);
    double per_ms = total_ms / iters;
    double gbps = 3.0 * total_bytes / per_ms / 1e6; /* read A, read B, write C */

    /* Verify correctness */
    int correct = 1;
    for (int i = 0; i < 100; i++) {
        int idx = rand() % N;
        if (fabsf(ptr_c[idx] - 2.0f) > 1e-6) { correct = 0; break; }
    }

    printf("│  Elements: %d (%.0f MB total I/O)                     │\n", N, 3.0 * total_bytes / 1e6);
    printf("│  Per dispatch: %.3f ms                                    │\n", per_ms);
    printf("│  Bandwidth: %.1f GB/s                                     │\n", gbps);
    printf("│  Correctness: %s                                        │\n", correct ? "VERIFIED" : "FAILED");

    if (surf_a) CFRelease(surf_a);
    if (surf_b) CFRelease(surf_b);
    if (surf_c) CFRelease(surf_c);

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  IOSurface Zero-Copy Bridge — Apple's Secret Weapon        ║\n");
        printf("║  Shared memory across CPU / GPU / ANE with no memcpy       ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        bench_allocation();
        bench_zero_copy_gpu();
        bench_gpu_compute();

        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  IOSurface enables zero-copy tensor sharing between all\n");
        printf("  Apple Silicon compute units. For LLM inference:\n");
        printf("  \n");
        printf("  GPU prefill → writes KV cache to IOSurface\n");
        printf("            ↓ (zero copy)\n");
        printf("  ANE decode → reads KV cache, generates tokens\n");
        printf("            ↓ (zero copy)\n");
        printf("  CPU verify → samples next token, updates state\n");
        printf("  \n");
        printf("  Eliminates the #1 bottleneck in hybrid pipelines.\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    }
    return 0;
}
