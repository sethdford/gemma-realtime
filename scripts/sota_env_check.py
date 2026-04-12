#!/usr/bin/env python3
"""Print SOTA-relevant env, imports, and native_hw status (no server required)."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
import os
import platform
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> int:
    print("sota_env_check")
    print(f"  platform: {platform.system()} {platform.machine()} python={sys.version.split()[0]}")

    for mod in ("mlx", "mlx_lm", "mlx_vlm"):
        print(f"  import {mod}: {'yes' if _has(mod) else 'no'}")

    tq = False
    try:
        from mlx.nn.layers import turbo_kv_cache  # noqa: F401

        tq = True
    except Exception:
        pass
    print(f"  mlx.nn.layers.turbo_kv_cache: {'yes' if tq else 'no'}")

    try:
        import native_hw as nh

        present = nh.libgemma_hw_present()
        print(f"  libgemma_hw.dylib: {'present' if present else 'absent'}")
        if present:
            print(f"  native_hw capabilities: {asdict(nh.capabilities())}")
    except Exception as e:
        print(f"  native_hw: error ({e})")

    keys = (
        "MLX_MODEL",
        "MLX_PORT",
        "MLX_SPECULATIVE_TOKENS",
        "GEMMA_SPECULATIVE_TOKENS",
        "GEMMA_KV_ASYMMETRIC",
        "GEMMA_IOSURFACE_KV",
        "GEMMA_IOSURFACE_KV_BYTES",
        "GEMMA_MIN_FLUSH_CHARS",
        "GEMMA_MAX_BUFFER_CHARS",
    )
    print("  environment (set only):")
    for k in keys:
        v = os.environ.get(k)
        if v is not None and v != "":
            print(f"    {k}={v!r}")

    print("  guide: guides/08-inference-sota-roadmap.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
