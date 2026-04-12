#!/usr/bin/env python3
"""
PROVE NATIVE HW: libgemma_hw.dylib + default IOSurfaceKVManager path.

On macOS, requires ``secret-apis/build/libgemma_hw.dylib`` (build with
``make -C secret-apis libgemma_hw``). Exits non-zero if the dylib is missing,
fails to load, or IOSurface / SGEMM checks fail.

On other platforms, exits 0 (nothing to prove).

Usage:
    python3 scripts/prove-native-hw.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    print("═" * 62)
    print("  PROVE NATIVE HW — libgemma_hw + IOSurface defaults")
    print("═" * 62)

    if platform.system() != "Darwin":
        print("  Skip: not macOS")
        return 0

    import native_hw

    dylib_path = native_hw.libgemma_hw_dylib_path()
    if not native_hw.libgemma_hw_present():
        print("  FAIL: libgemma_hw.dylib not found.")
        print(f"         Expected: {dylib_path}")
        print("         Run: make -C secret-apis libgemma_hw")
        return 1

    cap = native_hw.capabilities()
    print(
        f"  capabilities: IOSurfaceOK={cap.iosurface_create_ok} "
        f"Metal={cap.metal_device_present} ANE~{cap.ane_named_class_hits}"
    )
    if not cap.iosurface_create_ok:
        print("  FAIL: gemma_hw_capabilities_fill reports IOSurface create not OK")
        return 1

    from hw_accel import IOSurfaceKVManager

    mgr = IOSurfaceKVManager()
    if not mgr.native_zero_copy:
        err = native_hw.library_load_error()
        print(f"  FAIL: IOSurfaceKVManager native_zero_copy=False ({err})")
        return 1

    if not mgr.allocate_kv_surface("prove_kv", 16 * 1024):
        print("  FAIL: allocate_kv_surface did not use native IOSurface backend")
        return 1
    buf = mgr.get_surface("prove_kv")
    if buf is None or len(buf) < 16 * 1024:
        print("  FAIL: KV buffer missing or too small")
        return 1
    buf.fill(0)
    buf[0] = 0xC3
    buf[-1] = 0x3C
    if int(mgr.get_surface("prove_kv")[0]) != 0xC3 or int(mgr.get_surface("prove_kv")[-1]) != 0x3C:
        print("  FAIL: IOSurface CPU read-back mismatch")
        return 1
    mgr.release_all()

    import numpy as np

    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    c = native_hw.sgemm(a, b)
    err = float(np.max(np.abs(c - a @ b)))
    if err > 1e-3:
        print(f"  FAIL: SGEMM vs numpy max_err={err}")
        return 1

    if not native_hw.metal_iosurface_selftest():
        print("  FAIL: Metal IOSurface vec_mul selftest")
        return 1
    print("  Metal IOSurface GPU selftest: ok")

    print("  PASS — libgemma_hw + IOSurface + SGEMM + Metal zero-copy dispatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
