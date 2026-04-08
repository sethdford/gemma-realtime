#!/bin/bash
# bench_all_secrets.sh — Build and run all secret API benchmarks
#
# This script:
#   1. Builds all native benchmarks (C, Objective-C, Metal)
#   2. Runs each one and captures output
#   3. Generates a summary report
#
# Usage: ./bench_all_secrets.sh [--report]
#
# Requirements: Xcode Command Line Tools (clang, Metal compiler)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
REPORT_FILE="$SCRIPT_DIR/build/benchmark-report.txt"
GENERATE_REPORT=false

if [ "${1:-}" = "--report" ]; then
    GENERATE_REPORT=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ${CYAN}Apple Silicon Secret API Suite${NC}${BOLD}                                       ║${NC}"
echo -e "${BOLD}║  ${NC}AMX · SME2 · ANE · IOSurface · Metal Dynamic · Hybrid Pipeline${BOLD}     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Hardware info
echo -e "${CYAN}Hardware:${NC}"
echo "  Chip: $(sysctl -n machdep.cpu.brand_string)"
echo "  RAM:  $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
echo "  SME:  $(sysctl -n hw.optional.arm.FEAT_SME 2>/dev/null || echo 'N/A')"
echo "  SME2: $(sysctl -n hw.optional.arm.FEAT_SME2 2>/dev/null || echo 'N/A')"
echo "  macOS: $(sw_vers -productVersion)"
echo "  Xcode: $(xcodebuild -version 2>/dev/null | head -1 || echo 'N/A')"
echo ""

# Build phase
echo -e "${YELLOW}Building all benchmarks...${NC}"
mkdir -p "$BUILD_DIR"

BENCHMARKS=(
    "amx_matmul:AMX Coprocessor:clang -O3 -framework Accelerate amx_matmul.c -o build/amx_matmul"
    "sme2_matmul:SME2 Matrix Ext:clang -O3 -framework Accelerate sme2_matmul.c -o build/sme2_matmul"
    "ane_probe:ANE Private API:clang -O2 -fobjc-arc -framework Foundation -framework CoreML -ldl ane_probe.m -o build/ane_probe"
    "iosurface_bridge:IOSurface Bridge:clang -O2 -fobjc-arc -framework Foundation -framework Metal -framework IOSurface -framework CoreGraphics iosurface_bridge.m -o build/iosurface_bridge"
    "metal_dynamic:Metal Dynamic:clang -O2 -fobjc-arc -framework Foundation -framework Metal metal_dynamic.m -o build/metal_dynamic"
    "hybrid_pipeline:Hybrid Pipeline:clang -O2 -fobjc-arc -framework Foundation -framework Metal -framework IOSurface -framework CoreGraphics -framework CoreML hybrid_pipeline.m -o build/hybrid_pipeline"
)

BUILD_SUCCESS=0
BUILD_FAIL=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r name label cmd <<< "$entry"
    printf "  %-22s " "$label"
    if (cd "$SCRIPT_DIR" && eval "$cmd" 2>/dev/null); then
        echo -e "${GREEN}✓ built${NC}"
        BUILD_SUCCESS=$((BUILD_SUCCESS + 1))
    else
        echo -e "${RED}✗ failed${NC}"
        BUILD_FAIL=$((BUILD_FAIL + 1))
    fi
done

echo ""
echo -e "  Built: ${GREEN}${BUILD_SUCCESS}${NC} / $((BUILD_SUCCESS + BUILD_FAIL))"
echo ""

if [ "$BUILD_SUCCESS" -eq 0 ]; then
    echo -e "${RED}No benchmarks built successfully. Check Xcode CLI tools.${NC}"
    exit 1
fi

# Run phase
echo -e "${YELLOW}Running benchmarks...${NC}"
echo ""

if $GENERATE_REPORT; then
    exec > >(tee "$REPORT_FILE")
    echo "# Secret API Benchmark Report"
    echo "# Date: $(date)"
    echo "# Chip: $(sysctl -n machdep.cpu.brand_string)"
    echo "# RAM:  $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
    echo ""
fi

PASS=0
FAIL=0

for entry in "${BENCHMARKS[@]}"; do
    IFS=':' read -r name label cmd <<< "$entry"

    if [ ! -f "$BUILD_DIR/$name" ]; then
        echo -e "${RED}  [$label] SKIPPED (build failed)${NC}"
        FAIL=$((FAIL + 1))
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "  ${BOLD}Running: $label${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if "$BUILD_DIR/$name" 2>&1; then
        PASS=$((PASS + 1))
    else
        echo -e "${RED}  [$label] exited with error${NC}"
        FAIL=$((FAIL + 1))
    fi

    echo ""
done

# Summary
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY                                                           ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║                                                                    ║"
printf "║  Benchmarks passed: %-3d / %-3d                                     ║\n" "$PASS" "$((PASS + FAIL))"
echo "║                                                                    ║"
echo "║  Layers of the Secret Performance Stack:                           ║"
echo "║    1. AMX  — Undocumented CPU matrix coprocessor (M1-M3)           ║"
echo "║    2. SME2 — Official ARM matrix extension (M4+)                   ║"
echo "║    3. ANE  — Neural Engine direct access via private API           ║"
echo "║    4. IOSurface — Zero-copy shared memory (CPU/GPU/ANE)            ║"
echo "║    5. Metal Dynamic — MTLFunctionConstant kernel specialization    ║"
echo "║    6. Hybrid Pipeline — GPU prefill + ANE decode + zero-copy KV    ║"
echo "║                                                                    ║"
echo "║  These are the building blocks that make real-time LLM inference   ║"
echo "║  possible on Apple Silicon. Each layer removes a bottleneck:       ║"
echo "║    - AMX/SME2: CPU-side matrix ops at GPU-like throughput          ║"
echo "║    - ANE: Dedicated neural accelerator for efficient decode        ║"
echo "║    - IOSurface: Eliminates memory copy between compute units       ║"
echo "║    - Metal Dynamic: Shape-specialized GPU kernels (no branching)   ║"
echo "║    - Hybrid: Orchestrates all units for maximum utilization        ║"
echo "║                                                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if $GENERATE_REPORT; then
    echo ""
    echo -e "${GREEN}Report saved to: $REPORT_FILE${NC}"
fi
