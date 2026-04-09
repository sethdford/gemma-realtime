# Secret APIs — Build System
# Builds all Apple Silicon secret API benchmarks
#
# Usage:
#   make all        Build everything
#   make amx        AMX coprocessor benchmark
#   make sme2       ARM SME2 benchmark
#   make ane        ANE private API probe
#   make iosurface  IOSurface zero-copy bridge
#   make metal      Dynamic Metal kernels
#   make hybrid     Hybrid GPU+ANE pipeline
#   make clean      Remove build artifacts
#   make bench      Build and run all benchmarks

CC = clang
OBJC = clang

CFLAGS = -O3 -Wall -Wextra
OBJCFLAGS = -O2 -Wall -fobjc-arc

# Frameworks
FW_ACCEL = -framework Accelerate
FW_METAL = -framework Foundation -framework Metal
FW_COREML = -framework Foundation -framework CoreML
FW_IOSURFACE = -framework Foundation -framework Metal -framework IOSurface -framework CoreGraphics

BUILDDIR = build

.PHONY: all clean bench amx sme2 ane iosurface metal hybrid metal4 ane_direct

all: $(BUILDDIR)/amx_matmul $(BUILDDIR)/sme2_matmul $(BUILDDIR)/ane_probe \
     $(BUILDDIR)/iosurface_bridge $(BUILDDIR)/metal_dynamic $(BUILDDIR)/hybrid_pipeline \
     $(BUILDDIR)/metal4_tensor $(BUILDDIR)/ane_direct

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# AMX Coprocessor — undocumented matrix multiply
amx: $(BUILDDIR)/amx_matmul
$(BUILDDIR)/amx_matmul: amx_matmul.c amx.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(FW_ACCEL) $< -o $@

# SME2 — ARM Scalable Matrix Extension v2
# Note: -march=armv9-a+sme2 enables the SME2 intrinsics
# Falls back gracefully if not compiled with SME2 support
sme2: $(BUILDDIR)/sme2_matmul
# SME2 intrinsics compile on Apple Clang but crash at runtime (SIGILL) because
# macOS doesn't enable SME streaming mode in userspace. Apple uses AMX instead.
# We compile without -march=armv9-a+sme2 so the Accelerate baseline still runs.
$(BUILDDIR)/sme2_matmul: sme2_matmul.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(FW_ACCEL) $< -o $@

# ANE Private API Probe
ane: $(BUILDDIR)/ane_probe
$(BUILDDIR)/ane_probe: ane_probe.m | $(BUILDDIR)
	$(OBJC) $(OBJCFLAGS) $(FW_COREML) -ldl $< -o $@

# IOSurface Zero-Copy Bridge
iosurface: $(BUILDDIR)/iosurface_bridge
$(BUILDDIR)/iosurface_bridge: iosurface_bridge.m | $(BUILDDIR)
	$(OBJC) $(OBJCFLAGS) $(FW_IOSURFACE) $< -o $@

# Dynamic Metal Kernel Compilation
metal: $(BUILDDIR)/metal_dynamic
$(BUILDDIR)/metal_dynamic: metal_dynamic.m | $(BUILDDIR)
	$(OBJC) $(OBJCFLAGS) $(FW_METAL) $< -o $@

# Hybrid GPU+ANE Pipeline
hybrid: $(BUILDDIR)/hybrid_pipeline
$(BUILDDIR)/hybrid_pipeline: hybrid_pipeline.m | $(BUILDDIR)
	$(OBJC) $(OBJCFLAGS) $(FW_IOSURFACE) $(FW_COREML) $< -o $@

# Metal 4 Tensor APIs + M5 Neural Accelerator Probe
metal4: $(BUILDDIR)/metal4_tensor
$(BUILDDIR)/metal4_tensor: metal4_tensor.m | $(BUILDDIR)
	$(OBJC) $(OBJCFLAGS) $(FW_METAL) $< -o $@

# Direct ANE Access — Private API Deep Probe
ane_direct: $(BUILDDIR)/ane_direct
$(BUILDDIR)/ane_direct: ane_direct.m | $(BUILDDIR)
	$(OBJC) $(OBJCFLAGS) $(FW_IOSURFACE) $(FW_COREML) -ldl $< -o $@

# Build and run all benchmarks
bench: all
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║  Running all Secret API benchmarks...                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@for bin in amx_matmul sme2_matmul ane_probe ane_direct iosurface_bridge metal_dynamic metal4_tensor hybrid_pipeline; do \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		$(BUILDDIR)/$$bin 2>&1; \
		echo ""; \
	done

clean:
	rm -rf $(BUILDDIR)
