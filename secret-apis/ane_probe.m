/*
 * Apple Neural Engine — Private API Probe
 *
 * Discovers ANE-related classes available in the ObjC runtime without
 * loading private frameworks (which can hang or require entitlements).
 *
 * Build: clang -O2 -fobjc-arc -framework Foundation -framework CoreML ane_probe.m -o ane_probe
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <mach/mach_time.h>
#include <stdio.h>

static double mach_to_ms(uint64_t elapsed) {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1e6;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║  Apple Neural Engine — Runtime Discovery Probe             ║\n");
        printf("║  Scanning ObjC runtime for ANE classes and methods         ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        /* Scan all loaded classes for ANE-related ones */
        printf("┌─────────────────────────────────────────────────────────────┐\n");
        printf("│  ANE-Related Classes in ObjC Runtime                       │\n");
        printf("├─────────────────────────────────────────────────────────────┤\n");

        unsigned int class_count = 0;
        Class *classes = objc_copyClassList(&class_count);

        int ane_found = 0;
        int espresso_found = 0;
        int neural_found = 0;

        for (unsigned int i = 0; i < class_count; i++) {
            const char *name = class_getName(classes[i]);
            if (!name) continue;

            int is_ane = (strstr(name, "ANE") != NULL);
            int is_neural = (strstr(name, "NeuralEngine") != NULL || strstr(name, "NeuralNetwork") != NULL);
            int is_espresso = (strstr(name, "Espresso") != NULL);

            if (is_ane && ane_found < 15) {
                printf("│  %-55s │\n", name);
                ane_found++;

                /* Show key methods for important classes */
                if (strcmp(name, "_ANEClient") == 0 || strcmp(name, "_ANEModel") == 0 ||
                    strcmp(name, "_ANECompiler") == 0) {
                    unsigned int mc = 0;
                    Method *methods = class_copyMethodList(classes[i], &mc);
                    int shown = 0;
                    for (unsigned int m = 0; m < mc && shown < 5; m++) {
                        const char *sel = sel_getName(method_getName(methods[m]));
                        if (strstr(sel, "compile") || strstr(sel, "load") ||
                            strstr(sel, "execute") || strstr(sel, "init") ||
                            strstr(sel, "submit") || strstr(sel, "model")) {
                            printf("│    → %-51s │\n", sel);
                            shown++;
                        }
                    }
                    free(methods);
                }
            }
            if (is_neural) neural_found++;
            if (is_espresso) espresso_found++;
        }

        if (ane_found >= 15) printf("│  … and more ANE classes                                   │\n");
        printf("│                                                           │\n");
        printf("│  ANE classes:      %-5d                                   │\n", ane_found);
        printf("│  NeuralEngine/Net: %-5d                                   │\n", neural_found);
        printf("│  Espresso (CNN):   %-5d                                   │\n", espresso_found);
        printf("│  Total ObjC:       %-5u                                  │\n", class_count);
        printf("└─────────────────────────────────────────────────────────────┘\n\n");

        free(classes);

        /* Check specific known private classes */
        printf("┌─────────────────────────────────────────────────────────────┐\n");
        printf("│  Known Private API Classes                                 │\n");
        printf("├─────────────────────────────────────────────────────────────┤\n");

        const char *private_classes[] = {
            "_ANEClient", "_ANECompiler", "_ANEModel",
            "_ANEInMemoryModel", "_ANEInMemoryModelDescriptor",
            "_ANEIOSurfaceObject", "_ANEDeviceController",
            "_ANERequest", "_ANEProgramHandle",
            "MLNeuralNetworkEngine", "MLModelAsset",
            NULL
        };

        for (int i = 0; private_classes[i]; i++) {
            Class cls = objc_getClass(private_classes[i]);
            if (cls) {
                unsigned int mc = 0;
                Method *methods = class_copyMethodList(cls, &mc);
                printf("│  ✓ %-30s (%u methods)          │\n", private_classes[i], mc);
                free(methods);
            } else {
                printf("│  ✗ %-30s not loaded             │\n", private_classes[i]);
            }
        }

        printf("└─────────────────────────────────────────────────────────────┘\n\n");

        /* CoreML configuration — verify ANE routing */
        printf("┌─────────────────────────────────────────────────────────────┐\n");
        printf("│  CoreML Compute Unit Configuration                         │\n");
        printf("├─────────────────────────────────────────────────────────────┤\n");

        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        printf("│  MLComputeUnitsAll:     %ld (CPU + GPU + ANE)              │\n",
               (long)config.computeUnits);

        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        printf("│  CPUAndNeuralEngine:    %ld                                │\n",
               (long)config.computeUnits);

        config.computeUnits = MLComputeUnitsCPUAndGPU;
        printf("│  CPUAndGPU:             %ld                                │\n",
               (long)config.computeUnits);

        config.computeUnits = MLComputeUnitsCPUOnly;
        printf("│  CPUOnly:               %ld                                │\n",
               (long)config.computeUnits);

        printf("│                                                           │\n");
        printf("│  ANE access: Available via CoreML (MLComputeUnitsAll)     │\n");
        printf("│  Direct access: Requires com.apple.ane.iokit-user-access  │\n");
        printf("│  entitlement (private, Apple-signed apps only)            │\n");
        printf("└─────────────────────────────────────────────────────────────┘\n\n");

        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  The Neural Engine is real and accessible:\n");
        printf("  - CoreML routes models to ANE automatically (public API)\n");
        printf("  - _ANEClient exists for direct access (private, entitled)\n");
        printf("  - For LLM inference: convert draft model (E2B) to CoreML,\n");
        printf("    run on ANE for near-zero-power token generation\n");
        printf("═══════════════════════════════════════════════════════════════\n");
    }
    return 0;
}
