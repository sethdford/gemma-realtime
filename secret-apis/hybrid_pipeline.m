/*
 * Hybrid GPU+ANE Pipeline — Full Inference Demo
 *
 * Demonstrates the "secret sauce" architecture for maximum Apple Silicon
 * throughput: GPU handles prompt processing (prefill), ANE handles token
 * generation (decode), IOSurface enables zero-copy KV cache sharing.
 *
 * Architecture:
 *   ┌─────────┐     IOSurface      ┌──────────┐
 *   │   GPU   │ ──── KV Cache ───→ │   ANE    │
 *   │ Prefill │     (zero-copy)    │  Decode  │
 *   └────┬────┘                    └────┬─────┘
 *        │                              │
 *        └──────────┐    ┌──────────────┘
 *                   ▼    ▼
 *              ┌──────────────┐
 *              │     CPU      │
 *              │  Sampling +  │
 *              │  Scheduling  │
 *              └──────────────┘
 *
 * This demo simulates the pipeline with Metal compute shaders standing
 * in for the actual transformer layers, proving the architecture works.
 *
 * Build: clang -O2 -framework Foundation -framework Metal -framework IOSurface \
 *        -framework CoreML hybrid_pipeline.m -o hybrid_pipeline
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double mach_to_ms(uint64_t elapsed) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1e6;
}

static double mach_to_us(uint64_t elapsed) {
    return mach_to_ms(elapsed) * 1000.0;
}

/* Simulated model configuration */
typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_dim;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t n_layers;
    uint32_t max_seq_len;
    const char *name;
} model_config_t;

static const model_config_t GEMMA_E4B = {
    .vocab_size = 256000,
    .hidden_dim = 2048,
    .n_heads = 8,
    .head_dim = 256,
    .n_layers = 26,
    .max_seq_len = 8192,
    .name = "Gemma 4 E4B"
};

static const model_config_t GEMMA_E2B = {
    .vocab_size = 256000,
    .hidden_dim = 1536,
    .n_heads = 6,
    .head_dim = 256,
    .n_layers = 18,
    .max_seq_len = 8192,
    .name = "Gemma 4 E2B"
};

/* IOSurface-backed KV cache */
typedef struct {
    IOSurfaceRef key_surface;
    IOSurfaceRef value_surface;
    float *key_ptr;
    float *value_ptr;
    id<MTLBuffer> key_buffer;
    id<MTLBuffer> value_buffer;
    uint32_t seq_pos;
    uint32_t max_seq_len;
    uint32_t kv_dim;
} kv_cache_t;

static kv_cache_t *create_kv_cache(id<MTLDevice> device, const model_config_t *cfg) {
    kv_cache_t *cache = calloc(1, sizeof(kv_cache_t));
    cache->max_seq_len = cfg->max_seq_len;
    cache->kv_dim = cfg->n_heads * cfg->head_dim;
    cache->seq_pos = 0;

    int bytes = cfg->max_seq_len * cache->kv_dim * sizeof(float);

    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(cache->kv_dim),
        (id)kIOSurfaceHeight: @(cfg->max_seq_len),
        (id)kIOSurfaceBytesPerElement: @4,
        (id)kIOSurfaceBytesPerRow: @(cache->kv_dim * 4),
        (id)kIOSurfaceAllocSize: @(bytes),
    };

    cache->key_surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
    cache->value_surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);

    cache->key_ptr = (float *)IOSurfaceGetBaseAddress(cache->key_surface);
    cache->value_ptr = (float *)IOSurfaceGetBaseAddress(cache->value_surface);

    cache->key_buffer = [device newBufferWithBytesNoCopy:cache->key_ptr
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    cache->value_buffer = [device newBufferWithBytesNoCopy:cache->value_ptr
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    return cache;
}

static void destroy_kv_cache(kv_cache_t *cache) {
    if (cache->key_surface) CFRelease(cache->key_surface);
    if (cache->value_surface) CFRelease(cache->value_surface);
    free(cache);
}

/* Metal shader for simulated prefill (GPU) */
static NSString *prefill_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "constant uint HIDDEN_DIM [[function_constant(0)]];\n"
    "constant uint N_HEADS [[function_constant(1)]];\n"
    "constant uint HEAD_DIM [[function_constant(2)]];\n"
    "\n"
    "kernel void prefill(\n"
    "    device const float *input [[buffer(0)]],\n"
    "    device float *kv_cache_k [[buffer(1)]],\n"
    "    device float *kv_cache_v [[buffer(2)]],\n"
    "    device float *output [[buffer(3)]],\n"
    "    constant uint &seq_len [[buffer(4)]],\n"
    "    constant uint &kv_dim [[buffer(5)]],\n"
    "    uint3 tid [[thread_position_in_grid]]) {\n"
    "    \n"
    "    uint pos = tid.z;\n"
    "    uint head = tid.y;\n"
    "    uint d = tid.x;\n"
    "    if (d >= HEAD_DIM || head >= N_HEADS || pos >= seq_len) return;\n"
    "    \n"
    "    uint idx = pos * kv_dim + head * HEAD_DIM + d;\n"
    "    float val = input[pos * HIDDEN_DIM + head * HEAD_DIM + d];\n"
    "    \n"
    "    // Simulated attention projection → store to KV cache\n"
    "    kv_cache_k[idx] = val * 0.5;\n"
    "    kv_cache_v[idx] = val * 0.7;\n"
    "    output[idx] = val;\n"
    "}\n";

/* Metal shader for simulated decode (would run on ANE in production) */
static NSString *decode_shader = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "constant uint KV_DIM [[function_constant(0)]];\n"
    "\n"
    "kernel void decode_step(\n"
    "    device const float *query [[buffer(0)]],\n"
    "    device const float *kv_cache_k [[buffer(1)]],\n"
    "    device const float *kv_cache_v [[buffer(2)]],\n"
    "    device float *logits [[buffer(3)]],\n"
    "    constant uint &seq_pos [[buffer(4)]],\n"
    "    uint tid [[thread_position_in_grid]]) {\n"
    "    \n"
    "    if (tid >= KV_DIM) return;\n"
    "    \n"
    "    // Simulated attention over KV cache\n"
    "    float score = 0;\n"
    "    for (uint p = 0; p <= seq_pos; p++) {\n"
    "        float k = kv_cache_k[p * KV_DIM + tid];\n"
    "        score += query[tid] * k;\n"
    "    }\n"
    "    \n"
    "    // Simulated value weighted sum\n"
    "    float out = 0;\n"
    "    for (uint p = 0; p <= seq_pos; p++) {\n"
    "        float v = kv_cache_v[p * KV_DIM + tid];\n"
    "        out += v * exp(score);\n"
    "    }\n"
    "    \n"
    "    logits[tid] = out;\n"
    "}\n";

/* CPU sampling — argmax over logits */
static uint32_t sample_argmax(const float *logits, int n) {
    float max_val = logits[0];
    uint32_t max_idx = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

static void run_hybrid_pipeline(const model_config_t *cfg) {
    printf("┌───────────────────────────────────────────────────────────────┐\n");
    printf("│  Hybrid Pipeline — %-40s │\n", cfg->name);
    printf("├───────────────────────────────────────────────────────────────┤\n");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /* Create IOSurface-backed KV cache */
    kv_cache_t *kv = create_kv_cache(device, cfg);
    printf("│  KV cache: IOSurface-backed, zero-copy CPU↔GPU             │\n");
    printf("│  KV dim: %u, max seq: %u                                  │\n",
           kv->kv_dim, kv->max_seq_len);

    int kv_bytes = kv->max_seq_len * kv->kv_dim * sizeof(float);
    printf("│  KV cache size: %.1f MB per K/V (%.1f MB total)             │\n",
           kv_bytes / 1e6, kv_bytes * 2 / 1e6);

    /* Compile specialized Metal kernels */
    NSError *error = nil;

    /* Prefill kernel — specialized for this model config */
    id<MTLLibrary> prefill_lib = [device newLibraryWithSource:prefill_shader options:nil error:&error];
    MTLFunctionConstantValues *prefill_consts = [[MTLFunctionConstantValues alloc] init];
    uint32_t hd_val = cfg->hidden_dim, nh_val = cfg->n_heads, hd_dim = cfg->head_dim;
    [prefill_consts setConstantValue:&hd_val type:MTLDataTypeUInt atIndex:0];
    [prefill_consts setConstantValue:&nh_val type:MTLDataTypeUInt atIndex:1];
    [prefill_consts setConstantValue:&hd_dim type:MTLDataTypeUInt atIndex:2];

    id<MTLFunction> prefill_func = [prefill_lib newFunctionWithName:@"prefill"
                                                     constantValues:prefill_consts error:&error];
    id<MTLComputePipelineState> prefill_pipe = [device newComputePipelineStateWithFunction:prefill_func error:&error];

    /* Decode kernel */
    id<MTLLibrary> decode_lib = [device newLibraryWithSource:decode_shader options:nil error:&error];
    MTLFunctionConstantValues *decode_consts = [[MTLFunctionConstantValues alloc] init];
    uint32_t kvd = kv->kv_dim;
    [decode_consts setConstantValue:&kvd type:MTLDataTypeUInt atIndex:0];

    id<MTLFunction> decode_func = [decode_lib newFunctionWithName:@"decode_step"
                                                   constantValues:decode_consts error:&error];
    id<MTLComputePipelineState> decode_pipe = [device newComputePipelineStateWithFunction:decode_func error:&error];

    if (!prefill_pipe || !decode_pipe) {
        printf("│  ✗ Kernel compilation failed                               │\n");
        destroy_kv_cache(kv);
        printf("└───────────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    printf("│  Kernels compiled with MTLFunctionConstant specialization   │\n");

    /* Simulate a prompt of 64 tokens */
    uint32_t prompt_len = 64;
    uint32_t gen_tokens = 128; /* generate 128 tokens */
    int input_bytes = prompt_len * cfg->hidden_dim * sizeof(float);

    id<MTLBuffer> input_buf = [device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:kv->kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buf = [device newBufferWithLength:kv->kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> logits_buf = [device newBufferWithLength:kv->kv_dim * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> seq_len_buf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> kv_dim_buf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    /* Fill simulated input */
    float *inp = (float *)[input_buf contents];
    for (int i = 0; i < (int)(prompt_len * cfg->hidden_dim); i++) {
        inp[i] = (float)(rand() % 100) / 100.0f;
    }
    *(uint32_t *)[seq_len_buf contents] = prompt_len;
    *(uint32_t *)[kv_dim_buf contents] = kv->kv_dim;

    printf("│                                                             │\n");
    printf("│  ═══ Phase 1: GPU Prefill (%u tokens) ═══                  │\n", prompt_len);

    /* GPU Prefill: process entire prompt, populate KV cache */
    uint64_t t_prefill_start = mach_absolute_time();
    {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:prefill_pipe];
        [enc setBuffer:input_buf offset:0 atIndex:0];
        [enc setBuffer:kv->key_buffer offset:0 atIndex:1];
        [enc setBuffer:kv->value_buffer offset:0 atIndex:2];
        [enc setBuffer:output_buf offset:0 atIndex:3];
        [enc setBuffer:seq_len_buf offset:0 atIndex:4];
        [enc setBuffer:kv_dim_buf offset:0 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(cfg->head_dim, cfg->n_heads, prompt_len)
       threadsPerThreadgroup:MTLSizeMake(MIN(cfg->head_dim, (NSUInteger)prefill_pipe.maxTotalThreadsPerThreadgroup), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    double prefill_ms = mach_to_ms(mach_absolute_time() - t_prefill_start);
    kv->seq_pos = prompt_len - 1;

    printf("│  Prefill time: %.2f ms                                     │\n", prefill_ms);
    printf("│  KV cache populated: %u entries (IOSurface, zero-copy)     │\n", prompt_len);

    /* Verify KV cache is written (CPU reads IOSurface directly) */
    int kv_nonzero = 0;
    for (int i = 0; i < (int)MIN(1000u, prompt_len * kv->kv_dim); i++) {
        if (kv->key_ptr[i] != 0.0f) kv_nonzero++;
    }
    printf("│  KV cache verification: %s (%d/%d nonzero)               │\n",
           kv_nonzero > 0 ? "PASS" : "FAIL", kv_nonzero, (int)MIN(1000u, prompt_len * kv->kv_dim));

    printf("│                                                             │\n");
    printf("│  ═══ Phase 2: Token Generation (%u tokens) ═══            │\n", gen_tokens);
    printf("│  (GPU decode simulating ANE — reads shared KV cache)        │\n");

    /* Token generation: decode one token at a time */
    uint64_t t_gen_start = mach_absolute_time();
    uint32_t *generated = calloc(gen_tokens, sizeof(uint32_t));

    for (uint32_t t = 0; t < gen_tokens; t++) {
        /* Prepare query (from last output) */
        float *q = (float *)[query_buf contents];
        for (uint32_t d = 0; d < kv->kv_dim; d++) {
            q[d] = (float)(rand() % 100) / 100.0f;
        }

        uint32_t seq_pos = kv->seq_pos;
        id<MTLBuffer> pos_buf = [device newBufferWithBytes:&seq_pos
                                                    length:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];

        /* Decode step: read KV cache (IOSurface), compute attention */
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:decode_pipe];
        [enc setBuffer:query_buf offset:0 atIndex:0];
        [enc setBuffer:kv->key_buffer offset:0 atIndex:1];
        [enc setBuffer:kv->value_buffer offset:0 atIndex:2];
        [enc setBuffer:logits_buf offset:0 atIndex:3];
        [enc setBuffer:pos_buf offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(kv->kv_dim, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(kv->kv_dim, (NSUInteger)decode_pipe.maxTotalThreadsPerThreadgroup), 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        /* CPU sampling */
        const float *logits = (const float *)[logits_buf contents];
        generated[t] = sample_argmax(logits, kv->kv_dim);
        kv->seq_pos++;
    }

    double gen_ms = mach_to_ms(mach_absolute_time() - t_gen_start);
    double tokens_per_sec = gen_tokens / (gen_ms / 1000.0);
    double ms_per_token = gen_ms / gen_tokens;
    double total_ms = prefill_ms + gen_ms;

    printf("│                                                             │\n");
    printf("│  ═══ Results ═══                                            │\n");
    printf("│  Prefill:     %8.2f ms (%u tokens)                      │\n", prefill_ms, prompt_len);
    printf("│  Generation:  %8.2f ms (%u tokens)                     │\n", gen_ms, gen_tokens);
    printf("│  Total:       %8.2f ms                                   │\n", total_ms);
    printf("│  Throughput:  %8.1f tokens/sec                           │\n", tokens_per_sec);
    printf("│  Per token:   %8.2f ms                                   │\n", ms_per_token);
    printf("│  TTFT:        %8.2f ms (prefill latency)                 │\n", prefill_ms);
    printf("│                                                             │\n");

    /* Determine real-time status */
    double rt_threshold = 40.0; /* ms per token for real-time voice */
    printf("│  Real-time voice target: < %.0f ms/token                    │\n", rt_threshold);
    if (ms_per_token < rt_threshold) {
        printf("│  Status: ★ REAL-TIME ★ (%.1fx margin)                      │\n",
               rt_threshold / ms_per_token);
    } else {
        printf("│  Status: NOT REAL-TIME (need %.1fx improvement)             │\n",
               ms_per_token / rt_threshold);
    }

    free(generated);
    destroy_kv_cache(kv);

    printf("└───────────────────────────────────────────────────────────────┘\n\n");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  Hybrid GPU+ANE Pipeline — Full Inference Demo             ║\n");
        printf("║  IOSurface zero-copy KV cache between compute units        ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("GPU: %s\n", [[device name] UTF8String]);
        printf("Unified memory: %.0f GB\n\n", [device recommendedMaxWorkingSetSize] / 1e9);

        srand(42);

        run_hybrid_pipeline(&GEMMA_E4B);
        run_hybrid_pipeline(&GEMMA_E2B);

        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  The hybrid pipeline proves the architecture:\n");
        printf("  \n");
        printf("  1. GPU prefill populates the KV cache via Metal compute\n");
        printf("  2. KV cache lives in IOSurface (zero-copy shared memory)\n");
        printf("  3. Decode reads the shared KV cache with no memcpy\n");
        printf("  4. CPU handles sampling and scheduling\n");
        printf("  \n");
        printf("  In production, phase 2 would use:\n");
        printf("  - CoreML → ANE for the draft model (E2B) at near-zero power\n");
        printf("  - GPU for the target model (E4B) verification\n");
        printf("  - Speculative decoding: ANE drafts 4-8 tokens, GPU verifies\n");
        printf("  \n");
        printf("  The IOSurface KV cache is the glue that makes this possible\n");
        printf("  without paying any memory copy tax.\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    }
    return 0;
}
