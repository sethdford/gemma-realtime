#!/usr/bin/env python3
"""
Bindings for secret-apis/libgemma_hw.dylib (ctypes).

Build the library:
    make -C secret-apis libgemma_hw

Provides IOSurface-backed CPU buffers, Accelerate SGEMM, and hardware capability
introspection (sysctl, Metal, IOSurface probe, lightweight CoreML init for ANE class visibility).
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import c_float, c_int, c_size_t, c_uint32, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


class GemmaHWCapabilitiesC(ctypes.Structure):
    _fields_ = [
        ("is_darwin", c_int),
        ("is_apple_silicon", c_int),
        ("has_accelerate_sgemm", c_int),
        ("iosurface_create_ok", c_int),
        ("metal_device_present", c_int),
        ("sysctl_sme", c_int),
        ("sysctl_sme2", c_int),
        ("ane_named_class_hits", c_int),
        ("ane_known_private_loaded", c_int),
        ("cpu_brand", ctypes.c_char * 128),
    ]


@dataclass(frozen=True)
class Capabilities:
    is_darwin: bool
    is_apple_silicon: bool
    has_accelerate_sgemm: bool
    iosurface_create_ok: bool
    metal_device_present: bool
    sysctl_sme: bool
    sysctl_sme2: bool
    ane_named_class_hits: int
    ane_known_private_loaded: int
    cpu_brand: str


_lib: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def libgemma_hw_dylib_path() -> Path:
    """Preferred build output (file may not exist until ``make -C secret-apis libgemma_hw``)."""
    return _repo_root() / "secret-apis" / "build" / "libgemma_hw.dylib"


def find_libgemma_hw() -> Optional[Path]:
    root = _repo_root()
    for rel in ("secret-apis/build/libgemma_hw.dylib", "secret-apis/libgemma_hw.dylib"):
        p = root / rel
        if p.is_file():
            return p
    return None


def libgemma_hw_present() -> bool:
    """True if a ``libgemma_hw.dylib`` is present at a known search path."""
    return find_libgemma_hw() is not None


def load_library() -> Optional[ctypes.CDLL]:
    """Load libgemma_hw.dylib once; returns None on failure."""
    global _lib, _load_error
    if _lib is not None:
        return _lib
    if _load_error is not None:
        return None
    if platform.system() != "Darwin":
        _load_error = "libgemma_hw is only built for macOS"
        return None
    path = find_libgemma_hw()
    if path is None:
        _load_error = "libgemma_hw.dylib not found (run: make -C secret-apis libgemma_hw)"
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        _load_error = str(e)
        return None

    lib.gemma_hw_capabilities_fill.argtypes = [ctypes.POINTER(GemmaHWCapabilitiesC)]
    lib.gemma_hw_capabilities_fill.restype = None

    lib.gemma_hw_sgemm_row_major.argtypes = [
        c_int,
        c_int,
        c_int,
        c_float,
        ctypes.c_void_p,
        c_int,
        ctypes.c_void_p,
        c_int,
        c_float,
        ctypes.c_void_p,
        c_int,
    ]
    lib.gemma_hw_sgemm_row_major.restype = c_int

    lib.gemma_iosurface_create_packed.argtypes = [c_int, c_int, c_int, ctypes.POINTER(c_void_p)]
    lib.gemma_iosurface_create_packed.restype = c_int

    lib.gemma_iosurface_release.argtypes = [c_void_p]
    lib.gemma_iosurface_release.restype = None

    lib.gemma_iosurface_lock.argtypes = [c_void_p, c_uint32]
    lib.gemma_iosurface_lock.restype = c_int

    lib.gemma_iosurface_unlock.argtypes = [c_void_p, c_uint32]
    lib.gemma_iosurface_unlock.restype = c_int

    lib.gemma_iosurface_get_base_address.argtypes = [c_void_p]
    lib.gemma_iosurface_get_base_address.restype = ctypes.c_void_p

    lib.gemma_iosurface_get_alloc_size.argtypes = [c_void_p]
    lib.gemma_iosurface_get_alloc_size.restype = c_size_t

    lib.gemma_iosurface_get_bytes_per_row.argtypes = [c_void_p]
    lib.gemma_iosurface_get_bytes_per_row.restype = c_int

    lib.gemma_hw_iosurface_metal_vec_mul_selftest.argtypes = []
    lib.gemma_hw_iosurface_metal_vec_mul_selftest.restype = c_int

    _lib = lib
    return _lib


def library_load_error() -> Optional[str]:
    if _lib is not None:
        return None
    load_library()
    return _load_error


def capabilities() -> Capabilities:
    lib = load_library()
    c = GemmaHWCapabilitiesC()
    if lib is None:
        return Capabilities(
            is_darwin=platform.system() == "Darwin",
            is_apple_silicon=False,
            has_accelerate_sgemm=False,
            iosurface_create_ok=False,
            metal_device_present=False,
            sysctl_sme=False,
            sysctl_sme2=False,
            ane_named_class_hits=0,
            ane_known_private_loaded=0,
            cpu_brand="",
        )
    lib.gemma_hw_capabilities_fill(ctypes.byref(c))
    return Capabilities(
        is_darwin=bool(c.is_darwin),
        is_apple_silicon=bool(c.is_apple_silicon),
        has_accelerate_sgemm=bool(c.has_accelerate_sgemm),
        iosurface_create_ok=bool(c.iosurface_create_ok),
        metal_device_present=bool(c.metal_device_present),
        sysctl_sme=bool(c.sysctl_sme),
        sysctl_sme2=bool(c.sysctl_sme2),
        ane_named_class_hits=int(c.ane_named_class_hits),
        ane_known_private_loaded=int(c.ane_known_private_loaded),
        cpu_brand=c.cpu_brand.decode("utf-8", "replace").rstrip("\0"),
    )


def metal_iosurface_selftest() -> bool:
    """True if Metal vec_mul on IOSurface-backed MTLBuffers succeeds (requires macOS + dylib)."""
    lib = load_library()
    if lib is None:
        return False
    return int(lib.gemma_hw_iosurface_metal_vec_mul_selftest()) == 0


def health_payload_for_http(kv_manager: Optional[Any] = None) -> dict:
    """JSON-friendly fragment for HTTP ``/health`` (empty dict off macOS)."""
    out: dict = {}
    if platform.system() != "Darwin":
        return out
    try:
        p = find_libgemma_hw()
        out["iosurface_dylib_path"] = str(p) if p else ""
        lib = load_library()
        cap = capabilities()
        out["iosurface_lib_loaded"] = lib is not None
        out["iosurface_create_ok"] = cap.iosurface_create_ok
        if lib is None:
            out["iosurface_lib_error"] = library_load_error()
        if kv_manager is not None:
            out["iosurface_kv_available"] = kv_manager.available
            out["iosurface_kv_native"] = kv_manager.native_zero_copy
            out["iosurface_kv_surface_count"] = kv_manager.surface_count
    except Exception as e:
        out["iosurface_health_error"] = str(e)
    return out


def iosurface_dims_for_bytes(size_bytes: int, max_width: int = 16384) -> tuple[int, int, int, int]:
    """
    Return (width, height, bytes_per_element, padded_size) for IOSurfaceCreate.
    Pads to 4-byte alignment and uses FP32-style packing (same as iosurface_bridge.m).
    """
    if size_bytes <= 0:
        raise ValueError("size_bytes must be positive")
    pad = (4 - (size_bytes % 4)) % 4
    padded = size_bytes + pad
    num_floats = padded // 4
    w = min(num_floats, max_width)
    h = (num_floats + w - 1) // w
    return w, h, 4, padded


class NativeIOSurface:
    """IOSurfaceRef wrapper; CPU memory is valid only while locked."""

    __slots__ = ("_ptr", "_locked")

    def __init__(self, ptr: int):
        self._ptr = ptr
        self._locked = False

    @property
    def ptr(self) -> int:
        return self._ptr

    @classmethod
    def create_packed(cls, width: int, height: int, bytes_per_element: int) -> NativeIOSurface:
        lib = load_library()
        if lib is None:
            raise RuntimeError(library_load_error() or "libgemma_hw not loaded")
        out = c_void_p()
        r = lib.gemma_iosurface_create_packed(width, height, bytes_per_element, ctypes.byref(out))
        if r != 0 or not out.value:
            raise RuntimeError(f"gemma_iosurface_create_packed failed (code {r})")
        return cls(out.value or 0)

    def lock(self, options: int = 0) -> None:
        lib = load_library()
        if lib is None or not self._ptr:
            raise RuntimeError("invalid surface")
        r = lib.gemma_iosurface_lock(c_void_p(self._ptr), c_uint32(options))
        if r != 0:
            raise RuntimeError("IOSurfaceLock failed")
        self._locked = True

    def unlock(self, options: int = 0) -> None:
        lib = load_library()
        if lib is None or not self._ptr:
            return
        lib.gemma_iosurface_unlock(c_void_p(self._ptr), c_uint32(options))
        self._locked = False

    def release(self) -> None:
        lib = load_library()
        if lib is None or not self._ptr:
            return
        if self._locked:
            lib.gemma_iosurface_unlock(c_void_p(self._ptr), c_uint32(0))
            self._locked = False
        lib.gemma_iosurface_release(c_void_p(self._ptr))
        self._ptr = 0

    def __enter__(self) -> NativeIOSurface:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def numpy_uint8_view(self) -> np.ndarray:
        """Writable view over allocated bytes; call lock() first."""
        lib = load_library()
        if lib is None or not self._ptr:
            raise RuntimeError("invalid surface")
        n = int(lib.gemma_iosurface_get_alloc_size(c_void_p(self._ptr)))
        base = lib.gemma_iosurface_get_base_address(c_void_p(self._ptr))
        if not base:
            raise RuntimeError("IOSurface base address is NULL")
        addr = ctypes.cast(base, ctypes.c_void_p).value or 0
        buf = (ctypes.c_uint8 * n).from_address(addr)
        return np.frombuffer(buf, dtype=np.uint8, count=n)

    def logical_view(self, logical_bytes: int) -> np.ndarray:
        """First logical_bytes of the IOSurface (after lock)."""
        v = self.numpy_uint8_view()
        return v[:logical_bytes]


def sgemm(
    a: np.ndarray,
    b: np.ndarray,
    *,
    c: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Row-major C = alpha * A @ B + beta * C via Accelerate (AMX/SME internally).
    Shapes: A (m, k), B (k, n), C (m, n).
    """
    lib = load_library()
    if lib is None:
        raise RuntimeError(library_load_error() or "libgemma_hw not loaded")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2-D")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError("inner dimensions must match")
    a32 = np.ascontiguousarray(a, dtype=np.float32)
    b32 = np.ascontiguousarray(b, dtype=np.float32)
    if c is None:
        c32 = np.zeros((m, n), dtype=np.float32)
    else:
        c32 = np.ascontiguousarray(c, dtype=np.float32)
        if c32.shape != (m, n):
            raise ValueError("C shape mismatch")
    pa = a32.ctypes.data_as(ctypes.c_void_p)
    pb = b32.ctypes.data_as(ctypes.c_void_p)
    pc = c32.ctypes.data_as(ctypes.c_void_p)
    r = lib.gemma_hw_sgemm_row_major(
        c_int(m),
        c_int(n),
        c_int(k),
        c_float(alpha),
        pa,
        c_int(k),
        pb,
        c_int(n),
        c_float(beta),
        pc,
        c_int(n),
    )
    if r != 0:
        raise RuntimeError("gemma_hw_sgemm_row_major failed")
    return c32


def main() -> None:
    print("libgemma_hw:", find_libgemma_hw() or "(not built)")
    print("load:", library_load_error() or "ok")
    cap = capabilities()
    print(
        "capabilities:",
        f"AppleSilicon={cap.is_apple_silicon}",
        f"IOSurfaceOK={cap.iosurface_create_ok}",
        f"Metal={cap.metal_device_present}",
        f"SME2={cap.sysctl_sme2}",
        f"ANE_classes={cap.ane_named_class_hits}",
        f"ANE_private_loaded={cap.ane_known_private_loaded}",
    )
    print(f"cpu: {cap.cpu_brand!r}")
    if load_library() is None:
        return
    w, h, bpe, pad = iosurface_dims_for_bytes(1024)
    with NativeIOSurface.create_packed(w, h, bpe) as surf:
        surf.lock(0)
        v = surf.logical_view(1024)
        v[0] = 0xAB
        surf.unlock(0)
        surf.lock(0)
        assert int(v[0]) == 0xAB
        surf.unlock(0)
    print("IOSurface read/write: ok")
    a = np.random.randn(64, 64).astype(np.float32)
    b = np.random.randn(64, 64).astype(np.float32)
    c = sgemm(a, b)
    ref = a @ b
    err = float(np.max(np.abs(c - ref)))
    print(f"SGEMM max abs err vs numpy: {err:.6e}")
    assert err < 1e-4, err
    assert metal_iosurface_selftest(), "gemma_hw_iosurface_metal_vec_mul_selftest failed"
    print("Metal IOSurface vec_mul selftest: ok")


if __name__ == "__main__":
    main()
