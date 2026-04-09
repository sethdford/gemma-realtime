/*
 * Direct ANE Access — Bypassing CoreML via Private APIs
 *
 * Based on discoveries by maderix/ANE (MIT, 2026): the M4 Neural Engine
 * can be accessed directly through reverse-engineered private APIs in
 * AppleNeuralEngine.framework, bypassing CoreML's scheduling and overhead.
 *
 * Key private classes:
 *   _ANEClient              — hardware gateway (sharedConnection singleton)
 *   _ANEInMemoryModelDescriptor — in-memory MIL compilation (no disk mlmodelc)
 *   _ANEInMemoryModel        — compile, load, assess, unload cycle
 *   _ANERequest              — execution request with IOSurface I/O bindings
 *   _ANEIOSurfaceObject      — zero-copy tensor I/O wrapper
 *   _ANECompiler             — direct ANE program compilation
 *   _ANEChainingRequest      — chain multiple models in single dispatch
 *   _ANESharedEvents         — Metal-style fence/signal for GPU↔ANE sync
 *   _ANEPerformanceStats     — hardware performance counters (unexplored)
 *
 * This probe:
 *   1. Loads AppleNeuralEngine.framework at runtime via dlopen
 *   2. Discovers all ANE-related private classes and methods
 *   3. Attempts _ANEClient connection (works without entitlements)
 *   4. Probes for newly discovered classes from maderix research
 *   5. Maps the full ANE software stack
 *
 * Build: clang -O2 -fobjc-arc -framework Foundation -framework CoreML \
 *        -framework IOSurface -ldl ane_direct.m -o ane_direct
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static __attribute__((unused)) double mach_to_ms(uint64_t elapsed) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1e6;
}

/*
 * Phase 1: Load AppleNeuralEngine.framework at runtime.
 * This framework isn't linked publicly — we dlopen it.
 */
static BOOL load_ane_framework(void) {
    const char *paths[] = {
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        "/System/Library/Frameworks/CoreML.framework/CoreML",
        NULL
    };

    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Phase 1: Loading Private Frameworks                       │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    for (int i = 0; paths[i]; i++) {
        void *handle = dlopen(paths[i], RTLD_NOW);
        if (handle) {
            const char *name = strrchr(paths[i], '/');
            printf("│  ✓ Loaded: %-46s │\n", name ? name + 1 : paths[i]);
        } else {
            const char *name = strrchr(paths[i], '/');
            printf("│  ✗ Failed: %-46s │\n", name ? name + 1 : paths[i]);
        }
    }

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
    return YES;
}

/*
 * Phase 2: Deep class discovery — enumerate all ANE-related classes
 * with method counts, categorized by function.
 */
static void discover_ane_classes(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Phase 2: ANE Class Discovery (ObjC Runtime Scan)          │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    unsigned int class_count = 0;
    Class *classes = objc_copyClassList(&class_count);

    typedef struct { const char *name; const char *category; int methods; } ane_class_t;
    ane_class_t found[128];
    int nfound = 0;

    int ane_total = 0, espresso_total = 0, neural_total = 0, mil_total = 0;

    for (unsigned int i = 0; i < class_count && nfound < 128; i++) {
        const char *name = class_getName(classes[i]);
        if (!name) continue;

        const char *cat = NULL;
        if (strstr(name, "_ANE")) { cat = "ANE Private"; ane_total++; }
        else if (strstr(name, "ANECompiler") || strstr(name, "ANEC")) { cat = "ANE Compiler"; ane_total++; }
        else if (strstr(name, "Espresso")) { cat = "Espresso"; espresso_total++; continue; }
        else if (strstr(name, "NeuralEngine") || strstr(name, "NeuralNetwork")) { cat = "Neural"; neural_total++; }
        else if (strstr(name, "MIL") && !strstr(name, "Family")) { cat = "MIL"; mil_total++; }
        else continue;

        if (cat && nfound < 128) {
            unsigned int mc = 0;
            Method *methods = class_copyMethodList(classes[i], &mc);
            free(methods);
            found[nfound++] = (ane_class_t){name, cat, mc};
        }
    }
    free(classes);

    /* Print top classes by method count */
    for (int i = 0; i < nfound - 1; i++)
        for (int j = i + 1; j < nfound; j++)
            if (found[j].methods > found[i].methods) {
                ane_class_t tmp = found[i]; found[i] = found[j]; found[j] = tmp;
            }

    int shown = 0;
    for (int i = 0; i < nfound && shown < 20; i++) {
        printf("│  %-38s %3d methods  %-10s│\n",
               found[i].name, found[i].methods, found[i].category);
        shown++;
    }

    printf("│                                                           │\n");
    printf("│  Totals: ANE=%d  Espresso=%d  Neural=%d  MIL=%d          │\n",
           ane_total, espresso_total, neural_total, mil_total);
    printf("│  Total ObjC classes scanned: %u                          │\n", class_count);
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Phase 3: Probe specific private APIs discovered by maderix/ANE.
 * These are the key classes for direct ANE access.
 */
static void probe_private_apis(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Phase 3: Private API Deep Probe (maderix/ANE findings)    │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    typedef struct {
        const char *name;
        const char *purpose;
        const char *key_methods[4];
    } probe_t;

    probe_t probes[] = {
        {"_ANEClient", "Hardware gateway (sharedConnection)",
         {"sharedConnection", "compileModel:options:qos:error:", "loadModel:", "evaluateWithRequest:"}},
        {"_ANEInMemoryModelDescriptor", "In-memory MIL compilation",
         {"initWithMILText:weightsBuffer:", "setInputFeatures:", "compile", NULL}},
        {"_ANEInMemoryModel", "Compiled model lifecycle",
         {"compileFromDescriptor:error:", "loadAndReturnError:", "assessAndReturnError:", NULL}},
        {"_ANERequest", "Execution request + IOSurface I/O",
         {"initWithModel:", "setInputAtIndex:ioSurface:", "evaluate", NULL}},
        {"_ANEIOSurfaceObject", "Zero-copy tensor wrapper",
         {"initWithIOSurface:", "ioSurface", NULL, NULL}},
        {"_ANECompiler", "Direct ANE program compiler",
         {"compileModel:", "compiledModelExistsFor:", NULL, NULL}},
        {"_ANEChainingRequest", "Multi-model dispatch chain",
         {NULL, NULL, NULL, NULL}},
        {"_ANESharedEvents", "GPU↔ANE fence/signal sync",
         {NULL, NULL, NULL, NULL}},
        {"_ANEPerformanceStats", "Hardware perf counters",
         {NULL, NULL, NULL, NULL}},
        {"_ANEDeviceController", "Hardware control",
         {NULL, NULL, NULL, NULL}},
        {NULL, NULL, {NULL, NULL, NULL, NULL}}
    };

    for (int i = 0; probes[i].name; i++) {
        Class cls = objc_getClass(probes[i].name);
        if (cls) {
            unsigned int mc = 0;
            Method *methods = class_copyMethodList(cls, &mc);

            unsigned int cmc = 0;
            Class metacls = objc_getMetaClass(probes[i].name);
            Method *class_methods = metacls ? class_copyMethodList(metacls, &cmc) : NULL;

            printf("│  ✓ %-36s %3u+%u methods │\n", probes[i].name, mc, cmc);
            printf("│    Purpose: %-45s │\n", probes[i].purpose);

            /* Check for specific key methods (instance + class) */
            for (int m = 0; m < 4 && probes[i].key_methods[m]; m++) {
                BOOL found = NO;
                for (unsigned int j = 0; j < mc && !found; j++) {
                    const char *sel = sel_getName(method_getName(methods[j]));
                    if (strstr(sel, probes[i].key_methods[m]))
                        found = YES;
                }
                for (unsigned int j = 0; j < cmc && !found; j++) {
                    const char *sel = sel_getName(method_getName(class_methods[j]));
                    if (strstr(sel, probes[i].key_methods[m]))
                        found = YES;
                }
                printf("│      %s %-45s │\n",
                       found ? "→" : "✗", probes[i].key_methods[m]);
            }

            free(methods);
            free(class_methods);
        } else {
            printf("│  ✗ %-36s not loaded     │\n", probes[i].name);
            printf("│    Purpose: %-45s │\n", probes[i].purpose);
        }
        printf("│                                                           │\n");
    }

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Phase 4: Attempt _ANEClient connection.
 * The object can be created without entitlements, but compile/load/eval
 * require com.apple.ane.iokit-user-access (Apple-signed only).
 */
static void attempt_client_connection(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Phase 4: _ANEClient Connection Attempt                    │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");

    Class ANEClient = objc_getClass("_ANEClient");
    if (!ANEClient) {
        printf("│  ✗ _ANEClient class not available                         │\n");
        printf("└─────────────────────────────────────────────────────────────┘\n\n");
        return;
    }

    SEL sharedSel = sel_registerName("sharedConnection");
    if ([ANEClient respondsToSelector:sharedSel]) {
        printf("│  ✓ _ANEClient responds to sharedConnection               │\n");

        id client = ((id (*)(id, SEL))objc_msgSend)(ANEClient, sharedSel);
        if (client) {
            printf("│  ✓ Client object obtained: %s                            │\n",
                   class_getName([client class]));

            /* Check capabilities */
            SEL numSel = sel_registerName("numANEs");
            if ([client respondsToSelector:numSel]) {
                NSInteger n = ((NSInteger (*)(id, SEL))objc_msgSend)(client, numSel);
                printf("│  ✓ Number of ANE devices: %ld                            │\n", (long)n);
            }

            printf("│                                                           │\n");
            printf("│  ⚠ Note: compile/load/eval require entitlements:         │\n");
            printf("│    com.apple.ane.iokit-user-access (Apple-signed only)    │\n");
            printf("│    com.apple.aned.private.allow (primary ANE access)      │\n");
            printf("│                                                           │\n");
            printf("│  Workarounds:                                             │\n");
            printf("│  1. CoreML with MLComputeUnitsAll (public, works now)     │\n");
            printf("│  2. maderix/ANE runtime approach (private, may break)     │\n");
            printf("│  3. Metal 4 Tensor APIs on M5 (public, routes to         │\n");
            printf("│     per-core Neural Accelerators)                         │\n");
        } else {
            printf("│  ✗ sharedConnection returned nil                          │\n");
        }
    } else {
        printf("│  ✗ sharedConnection selector not found                    │\n");
    }

    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/*
 * Phase 5: Map the complete ANE software stack.
 */
static void print_ane_stack(void) {
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  Apple Neural Engine — Complete Software Stack             │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│                                                           │\n");
    printf("│  ┌─────────────────────────────────────────────────────┐  │\n");
    printf("│  │ PUBLIC: CoreML / coremltools / MLModel              │  │\n");
    printf("│  │ MLComputeUnitsAll → automatic ANE routing           │  │\n");
    printf("│  └────────────────────────┬────────────────────────────┘  │\n");
    printf("│                           │                               │\n");
    printf("│  ┌────────────────────────▼────────────────────────────┐  │\n");
    printf("│  │ SEMI-PUBLIC: Espresso.framework (40+ pass classes)  │  │\n");
    printf("│  │ EspressoContext → EspressoNetwork → CPU/GPU/ANE     │  │\n");
    printf("│  └────────────────────────┬────────────────────────────┘  │\n");
    printf("│                           │                               │\n");
    printf("│  ┌────────────────────────▼────────────────────────────┐  │\n");
    printf("│  │ PRIVATE: AppleNeuralEngine.framework (67+ classes)  │  │\n");
    printf("│  │ _ANEClient → _ANECompiler → _ANEInMemoryModel      │  │\n");
    printf("│  │ _ANERequest → _ANEIOSurfaceObject (zero-copy I/O)  │  │\n");
    printf("│  │ NEW: _ANEChainingRequest, _ANESharedEvents,        │  │\n");
    printf("│  │      _ANEPerformanceStats (unexplored)              │  │\n");
    printf("│  └────────────────────────┬────────────────────────────┘  │\n");
    printf("│                           │                               │\n");
    printf("│  ┌────────────────────────▼────────────────────────────┐  │\n");
    printf("│  │ KERNEL: IOKit driver (com.apple.driver.AppleH16ANE) │  │\n");
    printf("│  │ XPC daemon: /usr/libexec/aned                       │  │\n");
    printf("│  │ E5 binary format → 16-core ANE hardware             │  │\n");
    printf("│  └─────────────────────────────────────────────────────┘  │\n");
    printf("│                                                           │\n");
    printf("│  M4 ANE: 15.8 TFLOPS FP16, 38 TOPS INT8                 │\n");
    printf("│  ~32 MB on-chip SRAM, ~119 compile limit per process     │\n");
    printf("│  Weight baking: weights frozen at compile time            │\n");
    printf("│  1×1 conv gives 3x throughput vs matmul (conv engine)    │\n");
    printf("│                                                           │\n");
    printf("│  M5 shift: Neural Accelerators in GPU cores (Metal 4)    │\n");
    printf("│  Public API, no entitlements needed, 4x AI compute       │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  Direct ANE Access — Private API Deep Probe                ║\n");
        printf("║  Based on maderix/ANE reverse engineering (2026)           ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        load_ane_framework();
        discover_ane_classes();
        probe_private_apis();
        attempt_client_connection();
        print_ane_stack();

        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  The M4 ANE is a 15.8 TFLOPS accelerator sitting idle in\n");
        printf("  every Mac. maderix proved direct access is possible:\n");
        printf("  - 67 private classes in AppleNeuralEngine.framework\n");
        printf("  - In-memory MIL compilation via _ANEInMemoryModelDescriptor\n");
        printf("  - Full forward+backward pass (training!) on ANE hardware\n");
        printf("  - 6.6 TFLOPS/W — 80x more efficient than A100\n");
        printf("  \n");
        printf("  For gemma-realtime, the practical path is:\n");
        printf("  M4: CoreML + MLComputeUnitsAll for ANE draft model (E2B)\n");
        printf("  M5: Metal 4 Tensor APIs for per-core Neural Accelerators\n");
        printf("  Both: IOSurface zero-copy KV cache between GPU and ANE\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    }
    return 0;
}
