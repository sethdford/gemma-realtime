#!/usr/bin/env python3
"""Locate Voxtral speculative draft-head weights (stdlib only — no MLX import)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_draft_heads_path(cli_path: str | None) -> str | None:
    """Resolve path to draft heads for Voxtral speculative decoding.

    Precedence:
      1. Non-empty ``cli_path`` (``--draft-heads``).
      2. Environment variable ``VOXTRAL_DRAFT_HEADS`` — path to a .safetensors file,
         or ``0`` / ``false`` / ``off`` / ``none`` (case-insensitive) to disable.
      3. If unset, prefer ``adapters/draft-heads/heads-libri.safetensors``, then
         ``adapters/draft-heads/heads.safetensors`` when the file exists under the
         repo root (parent of ``scripts/``).

    Returns:
        Absolute or repo-relative path string, or ``None`` if disabled / not found.
    """
    if cli_path:
        return cli_path

    env = os.environ.get("VOXTRAL_DRAFT_HEADS")
    if env is not None:
        e = env.strip()
        if not e or e.lower() in ("0", "false", "off", "none"):
            return None
        return e

    for rel in (
        "adapters/draft-heads/heads-libri.safetensors",
        "adapters/draft-heads/heads.safetensors",
    ):
        p = _REPO_ROOT / rel
        if p.is_file():
            return str(p)

    return None
