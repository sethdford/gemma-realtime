/*
 * libgemma_hw — Darwin: capabilities, IOSurface helpers, ANE runtime scan.
 */

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <sys/sysctl.h>

#include "gemma_hw.h"
#include <math.h>
#include <string.h>

static void sysctl_int(const char *name, int *out) {
    int v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, NULL, 0) == 0 && out)
        *out = v;
    else if (out)
        *out = 0;
}

/* Same property bundle as iosurface_bridge.m `create_tensor_surface`. */
static IOSurfaceRef create_tensor_surface(int width, int height, int bytes_per_element) {
    if (width <= 0 || height <= 0 || bytes_per_element <= 0)
        return NULL;
    if (bytes_per_element > 0 && width > 0x7fffffff / bytes_per_element)
        return NULL;
    int bytes_per_row = width * bytes_per_element;
    if (bytes_per_row > 0 && height > 0x7fffffff / bytes_per_row)
        return NULL;
    int total_bytes = bytes_per_row * height;

    NSDictionary *props = @{
        (id)kIOSurfaceWidth : @(width),
        (id)kIOSurfaceHeight : @(height),
        (id)kIOSurfaceBytesPerElement : @(bytes_per_element),
        (id)kIOSurfaceBytesPerRow : @(bytes_per_row),
        (id)kIOSurfaceAllocSize : @(total_bytes),
        (id)kIOSurfacePixelFormat : @(0x46503332), /* 'FP32' */
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static int apple_silicon_from_brand(const char *brand) {
    if (!brand)
        return 0;
    return strstr(brand, "Apple") != NULL;
}

void gemma_hw_capabilities_fill(GemmaHWCapabilities *caps) {
    if (!caps)
        return;
    memset(caps, 0, sizeof(*caps));
    caps->is_darwin = 1;
    caps->has_accelerate_sgemm = 1;

    size_t len = sizeof(caps->cpu_brand);
    if (sysctlbyname("machdep.cpu.brand_string", caps->cpu_brand, &len, NULL, 0) != 0)
        caps->cpu_brand[0] = '\0';
    caps->is_apple_silicon = apple_silicon_from_brand(caps->cpu_brand);

    sysctl_int("hw.optional.arm.FEAT_SME", &caps->sysctl_sme);
    sysctl_int("hw.optional.arm.FEAT_SME2", &caps->sysctl_sme2);

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev) {
        caps->metal_device_present = 1;
    }

    IOSurfaceRef probe = create_tensor_surface(64, 64, 4);
    if (probe) {
        caps->iosurface_create_ok = 1;
        CFRelease(probe);
    }

    @autoreleasepool {
        MLModelConfiguration *mlcfg = [[MLModelConfiguration alloc] init];
        (void)mlcfg;
    }

    /* ANE-related ObjC names (CoreML nudge pulls more stacks into the runtime). */
    unsigned int class_count = 0;
    Class *classes = objc_copyClassList(&class_count);
    int ane_hits = 0;
    if (classes) {
        for (unsigned i = 0; i < class_count; i++) {
            const char *name = class_getName(classes[i]);
            if (name && strstr(name, "ANE") != NULL)
                ane_hits++;
        }
        free(classes);
    }
    caps->ane_named_class_hits = ane_hits;

    const char *known[] = {
        "_ANEClient", "_ANECompiler", "_ANEModel", "_ANEInMemoryModel",
        "_ANEInMemoryModelDescriptor", "_ANEIOSurfaceObject", NULL,
    };
    int loaded = 0;
    for (int k = 0; known[k]; k++) {
        if (objc_getClass(known[k]))
            loaded++;
    }
    caps->ane_known_private_loaded = loaded;
}

int gemma_iosurface_create_packed(int width, int height, int bytes_per_element, void **out_surface) {
    if (!out_surface)
        return -1;
    *out_surface = NULL;
    IOSurfaceRef s = create_tensor_surface(width, height, bytes_per_element);
    if (!s)
        return -2;
    *out_surface = (void *)s;
    return 0;
}

void gemma_iosurface_release(void *surface) {
    if (surface)
        CFRelease((IOSurfaceRef)surface);
}

int gemma_iosurface_lock(void *surface, uint32_t options) {
    if (!surface)
        return -1;
    return IOSurfaceLock((IOSurfaceRef)surface, options, NULL) == 0 ? 0 : -2;
}

int gemma_iosurface_unlock(void *surface, uint32_t options) {
    if (!surface)
        return -1;
    return IOSurfaceUnlock((IOSurfaceRef)surface, options, NULL) == 0 ? 0 : -2;
}

void *gemma_iosurface_get_base_address(void *surface) {
    if (!surface)
        return NULL;
    return IOSurfaceGetBaseAddress((IOSurfaceRef)surface);
}

size_t gemma_iosurface_get_alloc_size(void *surface) {
    if (!surface)
        return 0;
    return IOSurfaceGetAllocSize((IOSurfaceRef)surface);
}

int gemma_iosurface_get_bytes_per_row(void *surface) {
    if (!surface)
        return 0;
    return (int)IOSurfaceGetBytesPerRow((IOSurfaceRef)surface);
}

int gemma_hw_iosurface_metal_vec_mul_selftest(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return -1;

        NSString *src =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void vec_mul(device float *a [[buffer(0)]],\n"
             "                    device float *b [[buffer(1)]],\n"
             "                    device float *c [[buffer(2)]],\n"
             "                    uint id [[thread_position_in_grid]]) {\n"
             "  c[id] = a[id] * b[id];\n"
             "}\n";

        NSError *err = nil;
        id<MTLLibrary> mlib = [device newLibraryWithSource:src options:nil error:&err];
        if (!mlib)
            return -2;
        id<MTLFunction> fn = [mlib newFunctionWithName:@"vec_mul"];
        if (!fn)
            return -2;
        id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe)
            return -3;

        const int N = 2048;
        const int total_bytes = N * (int)sizeof(float);
        IOSurfaceRef sa = create_tensor_surface(N, 1, 4);
        IOSurfaceRef sb = create_tensor_surface(N, 1, 4);
        IOSurfaceRef sc = create_tensor_surface(N, 1, 4);
        if (!sa || !sb || !sc) {
            if (sa)
                CFRelease(sa);
            if (sb)
                CFRelease(sb);
            if (sc)
                CFRelease(sc);
            return -4;
        }

        IOSurfaceLock(sa, 0, NULL);
        IOSurfaceLock(sb, 0, NULL);
        float *pa = (float *)IOSurfaceGetBaseAddress(sa);
        float *pb = (float *)IOSurfaceGetBaseAddress(sb);
        for (int i = 0; i < N; i++) {
            pa[i] = 1.0f;
            pb[i] = 2.0f;
        }
        IOSurfaceUnlock(sa, 0, NULL);
        IOSurfaceUnlock(sb, 0, NULL);

        id<MTLBuffer> buf_a = [device newBufferWithBytesNoCopy:pa
                                                        length:(NSUInteger)total_bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        id<MTLBuffer> buf_b = [device newBufferWithBytesNoCopy:pb
                                                        length:(NSUInteger)total_bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];
        float *pc_base = (float *)IOSurfaceGetBaseAddress(sc);
        id<MTLBuffer> buf_c = [device newBufferWithBytesNoCopy:pc_base
                                                        length:(NSUInteger)total_bytes
                                                       options:MTLResourceStorageModeShared
                                                   deallocator:nil];

        id<MTLCommandQueue> q = [device newCommandQueue];
        id<MTLCommandBuffer> cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        MTLSize grid = MTLSizeMake((NSUInteger)N, 1, 1);
        MTLSize tg = MTLSizeMake(pipe.maxTotalThreadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        IOSurfaceLock(sc, 0, NULL);
        float *pc = (float *)IOSurfaceGetBaseAddress(sc);
        int ok = (fabsf(pc[0] - 2.0f) < 1e-4f && fabsf(pc[N / 2] - 2.0f) < 1e-4f);
        IOSurfaceUnlock(sc, 0, NULL);

        CFRelease(sa);
        CFRelease(sb);
        CFRelease(sc);

        return ok ? 0 : -5;
    }
}
