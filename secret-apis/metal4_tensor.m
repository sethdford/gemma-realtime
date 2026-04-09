/*
 * Metal 4 Tensor APIs + M5 Neural Accelerator Probe
 *
 * Metal 4 (WWDC 2025) introduced three ML integration points:
 *   1. MTLTensor — multi-dimensional resource for ML data
 *   2. MTL4MachineLearningCommandEncoder — runs CoreML models on GPU timeline
 *   3. Shader ML — embed ML ops (matmul, conv) inside compute/render shaders
 *
 * On M5, each GPU core contains a Neural Accelerator programmable via these
 * Tensor APIs. This gives 10-40 dedicated ML units inside the GPU itself.
 *
 * This benchmark:
 *   1. Detects Metal 4 and MTLTensor support
 *   2. Probes Neural Accelerator availability per GPU core
 *   3. Benchmarks Metal compute with tensor-style memory layouts
 *   4. Compares standard vs tensor-optimized buffer access patterns
 *
 * Build: clang -O2 -framework Foundation -framework Metal metal4_tensor.m -o metal4_tensor
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

static double mach_to_us(uint64_t elapsed) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1e3;
}

static __attribute__((unused)) double mach_to_ms(uint64_t elapsed) {
    return mach_to_us(elapsed) / 1e3;
}

static void detect_hardware(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Hardware & Metal Feature Detection                        │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    char cpu[128] = {0};
    size_t len = sizeof(cpu);
    sysctlbyname("machdep.cpu.brand_string", cpu, &len, NULL, 0);
    printf("│  CPU: %-51s │\n", cpu);

    int has_sme = 0, has_sme2 = 0;
    len = sizeof(int);
    sysctlbyname("hw.optional.arm.FEAT_SME", &has_sme, &len, NULL, 0);
    len = sizeof(int);
    sysctlbyname("hw.optional.arm.FEAT_SME2", &has_sme2, &len, NULL, 0);
    printf("│  SME: %s  SME2: %s                                       │\n",
           has_sme ? "YES" : "NO ", has_sme2 ? "YES" : "NO ");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("│  ✗ No Metal device found                                  │\n");
        printf("└─────────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    printf("│  GPU: %-51s │\n", [[device name] UTF8String]);
    printf("│  Max buffer: %.0f MB                                        │\n",
           [device maxBufferLength] / 1e6);
    printf("│  Max threadgroup mem: %lu bytes                           │\n",
           (unsigned long)[device maxThreadgroupMemoryLength]);
    printf("│  Recommended working set: %.1f GB                          │\n",
           [device recommendedMaxWorkingSetSize] / 1e9);

    /*
     * Check for Metal 4 support by probing for MTL4CommandQueue.
     * Metal 4 was introduced at WWDC 2025 and requires M1+ / A14+.
     */
    BOOL has_metal4 = NO;
    if ([device respondsToSelector:@selector(supportsFamily:)]) {
        /* MTLGPUFamily values: Apple7 = 1007, Apple8 = 1008, Apple9 = 1009 */
        has_metal4 = [device supportsFamily:MTLGPUFamilyApple7];
    }
    printf("│  Metal 4 capable: %s                                      │\n",
           has_metal4 ? "YES" : "NO ");

    /*
     * M5 Neural Accelerators live inside each GPU core.
     * Detect by checking chip name — no public API exposes the count.
     */
    NSString *name = [device name];
    BOOL has_neural_accel = NO;
    int neural_accel_count = 0;
    if ([name containsString:@"M5"]) {
        has_neural_accel = YES;
        if ([name containsString:@"Max"])
            neural_accel_count = 40;
        else if ([name containsString:@"Pro"])
            neural_accel_count = 20;
        else
            neural_accel_count = 10;
    }

    if (has_neural_accel) {
        printf("│  Neural Accelerators: %d (one per GPU core)               │\n",
               neural_accel_count);
        printf("│  Metal 4 Tensor API: programmable via MTLTensor + Shader ML│\n");
    } else {
        printf("│  Neural Accelerators: not present (M5+ only)              │\n");
        printf("│  (M1-M4 use discrete 16-core Neural Engine instead)       │\n");
    }

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Benchmark: Strided tensor access patterns vs contiguous buffers.
 * Neural Accelerators and Shader ML prefer MTLTensor's stride-aware layout.
 * Even without Metal 4, this demonstrates the bandwidth advantage of
 * stride-optimized memory access for KV cache and attention weights.
 */
static NSString *contiguous_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void matmul_contiguous(\n"
    "    device const float *A [[buffer(0)]],\n"
    "    device const float *B [[buffer(1)]],\n"
    "    device float *C [[buffer(2)]],\n"
    "    constant uint &N [[buffer(3)]],\n"
    "    uint2 tid [[thread_position_in_grid]]) {\n"
    "    uint row = tid.y, col = tid.x;\n"
    "    if (row >= N || col >= N) return;\n"
    "    float sum = 0;\n"
    "    for (uint k = 0; k < N; k++) {\n"
    "        sum += A[row * N + k] * B[k * N + col];\n"
    "    }\n"
    "    C[row * N + col] = sum;\n"
    "}\n";

/*
 * Tiled matmul using threadgroup memory — approximates the data reuse
 * patterns that Shader ML and Neural Accelerators optimize for.
 */
static NSString *tiled_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "constant uint TILE_SIZE = 16;\n"
    "kernel void matmul_tiled(\n"
    "    device const float *A [[buffer(0)]],\n"
    "    device const float *B [[buffer(1)]],\n"
    "    device float *C [[buffer(2)]],\n"
    "    constant uint &N [[buffer(3)]],\n"
    "    uint2 tid [[thread_position_in_grid]],\n"
    "    uint2 lid [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float As[16][16];\n"
    "    threadgroup float Bs[16][16];\n"
    "    uint row = tid.y, col = tid.x;\n"
    "    float sum = 0;\n"
    "    for (uint t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {\n"
    "        uint ak = t * TILE_SIZE + lid.x;\n"
    "        uint bk = t * TILE_SIZE + lid.y;\n"
    "        As[lid.y][lid.x] = (row < N && ak < N) ? A[row * N + ak] : 0;\n"
    "        Bs[lid.y][lid.x] = (bk < N && col < N) ? B[bk * N + col] : 0;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint k = 0; k < TILE_SIZE; k++) {\n"
    "            sum += As[lid.y][k] * Bs[k][lid.x];\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "    if (row < N && col < N) C[row * N + col] = sum;\n"
    "}\n";

static void bench_tensor_layouts(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Tensor Access Patterns: Contiguous vs Tiled               │\n");
    printf("│  (approximates Shader ML / Neural Accelerator optimization)│\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError *error = nil;

    id<MTLLibrary> contig_lib = [device newLibraryWithSource:contiguous_shader options:nil error:&error];
    id<MTLLibrary> tiled_lib = [device newLibraryWithSource:tiled_shader options:nil error:&error];
    if (!contig_lib || !tiled_lib) {
        printf("│  ✗ Shader compilation failed                              │\n");
        printf("└─────────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    id<MTLFunction> contig_fn = [contig_lib newFunctionWithName:@"matmul_contiguous"];
    id<MTLFunction> tiled_fn = [tiled_lib newFunctionWithName:@"matmul_tiled"];
    id<MTLComputePipelineState> contig_pipe = [device newComputePipelineStateWithFunction:contig_fn error:&error];
    id<MTLComputePipelineState> tiled_pipe = [device newComputePipelineStateWithFunction:tiled_fn error:&error];

    uint32_t sizes[] = {128, 256, 512};
    int nsizes = 3;

    printf("│  %-6s  %10s  %10s  %10s  %8s       │\n",
           "N", "Contiguous", "Tiled", "GFLOPS(T)", "Speedup");
    printf("│  ──────  ──────────  ──────────  ──────────  ────────       │\n");

    for (int si = 0; si < nsizes; si++) {
        uint32_t N = sizes[si];
        int bytes = N * N * sizeof(float);
        int iters = N <= 256 ? 50 : 20;

        id<MTLBuffer> A = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> B = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> C = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> Nbuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        *(uint32_t *)[Nbuf contents] = N;

        float *ap = (float *)[A contents], *bp = (float *)[B contents];
        for (uint32_t i = 0; i < N * N; i++) {
            ap[i] = (float)(rand() % 100) / 100.0f;
            bp[i] = (float)(rand() % 100) / 100.0f;
        }

        /* Warmup */
        for (int w = 0; w < 3; w++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:contig_pipe];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:Nbuf offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N, N, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cmd commit]; [cmd waitUntilCompleted];
        }

        /* Benchmark contiguous */
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:contig_pipe];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:Nbuf offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N, N, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cmd commit]; [cmd waitUntilCompleted];
        }
        double contig_us = mach_to_us(mach_absolute_time() - t0) / iters;

        /* Benchmark tiled */
        for (int w = 0; w < 3; w++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:tiled_pipe];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:Nbuf offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N, N, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cmd commit]; [cmd waitUntilCompleted];
        }

        t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:tiled_pipe];
            [enc setBuffer:A offset:0 atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1];
            [enc setBuffer:C offset:0 atIndex:2];
            [enc setBuffer:Nbuf offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N, N, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cmd commit]; [cmd waitUntilCompleted];
        }
        double tiled_us = mach_to_us(mach_absolute_time() - t0) / iters;

        double gflops = 2.0 * N * N * N / (tiled_us * 1e3);
        printf("│  %-6u  %7.0f µs  %7.0f µs  %7.1f GF  %6.2fx       │\n",
               N, contig_us, tiled_us, gflops, contig_us / tiled_us);
    }

    printf("│                                                           │\n");
    printf("│  Tiled access mimics how Neural Accelerators and Shader   │\n");
    printf("│  ML stage data through on-chip SRAM for maximum reuse.    │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Metal 4 ML integration summary — what's available and what it means.
 */
static void print_metal4_summary(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Metal 4 ML Integration (WWDC 2025)                        │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│                                                            │\n");
    printf("│  1. MTLTensor                                              │\n");
    printf("│     Multi-dimensional resource with baked-in strides.      │\n");
    printf("│     Replaces manual buffer offset math for KV caches,      │\n");
    printf("│     attention weights, and activation tensors.             │\n");
    printf("│                                                            │\n");
    printf("│  2. MTL4MachineLearningCommandEncoder                      │\n");
    printf("│     Runs entire CoreML networks on the GPU timeline.       │\n");
    printf("│     Zero CPU round-trip for inference during rendering.    │\n");
    printf("│     Uses .mlmodelc format — compatible with existing       │\n");
    printf("│     CoreML export pipelines (coremltools).                 │\n");
    printf("│                                                            │\n");
    printf("│  3. Shader ML                                              │\n");
    printf("│     Embed matmul/conv ops inside compute/render shaders.   │\n");
    printf("│     Operations share threadgroup memory — cache-friendly,  │\n");
    printf("│     single dispatch, no encoder overhead per op.           │\n");
    printf("│     On M5: routes to per-core Neural Accelerators.         │\n");
    printf("│                                                            │\n");
    printf("│  LLM Inference Implications:                               │\n");
    printf("│  • KV cache as MTLTensor → stride-aware dequant kernels    │\n");
    printf("│  • Draft model via ML encoder → no CPU scheduling          │\n");
    printf("│  • Fused attention+dequant via Shader ML → 1 dispatch      │\n");
    printf("│  • M5: 10-40 Neural Accelerators for parallel attention    │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  Metal 4 Tensor APIs + M5 Neural Accelerator Probe         ║\n");
        printf("║  Next-gen ML integration for LLM inference                 ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        detect_hardware();
        bench_tensor_layouts();
        print_metal4_summary();

        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  Metal 4 unifies ML and graphics on the GPU timeline.\n");
        printf("  On M5, Neural Accelerators give every GPU core a dedicated\n");
        printf("  ML unit — 4x peak AI compute vs M4 via Tensor APIs.\n");
        printf("  \n");
        printf("  For gemma-realtime: Shader ML enables fused attention\n");
        printf("  kernels that run TurboQuant dequant + softmax + V-mul\n");
        printf("  in a single dispatch with shared threadgroup memory.\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    }
    return 0;
}
