"""Phase-A spec (specs/phase-a-speech-lane) Task 9 / AC-8: the fish (true-S2S) and
cascade speech hot paths must run fully on-device — no cloud SDK on the hot path,
so no audio leaves the machine and there is no per-minute external API.

Static contract (no model needed). Scans *import lines only* of the hot-path
compute modules — robust against vendor names appearing in docstrings/comments
(e.g. "follows the OpenAI Realtime API shape"), which are not egress.

The vendor baseline (scripts/vendor_s2s_baseline.py, Task 5) is intentionally
EXEMPT: it is comparison-only and never on the product hot path.
"""
from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# True-S2S + codec hot path (fish lane) plus the cascade orchestrator.
HOT_PATH_FILES = [
    "fish_sts.py",
    "fish_dac_loader.py",
    "codec.py",
    "speech_model.py",
    "speech_encoder.py",
    "speech_decoder.py",
    "sota_pipeline.py",
]

# Cloud SDK import roots that would imply off-device egress.
BANNED_IMPORT_ROOTS = {
    "openai", "boto3", "botocore", "anthropic", "elevenlabs", "cartesia",
    "hume", "cohere", "replicate", "azure",
    "google",  # google.generativeai / google.cloud / vertexai paths
    "vertexai", "deepgram", "assemblyai",
}

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)


def _imported_roots(text: str) -> set[str]:
    roots: set[str] = set()
    for m in _IMPORT_RE.finditer(text):
        roots.add(m.group(1).split(".")[0].lower())
    return roots


def test_hot_path_files_exist():
    present = [f for f in HOT_PATH_FILES if (SCRIPTS / f).exists()]
    # At least the core S2S + codec modules must exist for this contract to mean anything.
    assert "fish_sts.py" in present and "codec.py" in present, f"missing core hot-path files; found {present}"


def test_no_cloud_sdk_imports_on_speech_hot_path():
    offenders: dict[str, set[str]] = {}
    for fname in HOT_PATH_FILES:
        p = SCRIPTS / fname
        if not p.exists():
            continue
        roots = _imported_roots(p.read_text(encoding="utf-8"))
        bad = roots & BANNED_IMPORT_ROOTS
        if bad:
            offenders[fname] = bad
    assert not offenders, f"cloud-SDK imports on the on-device speech hot path (AC-8 violation): {offenders}"


def test_no_external_wss_endpoint_literals_on_hot_path():
    # A non-localhost wss:// or external vendor host literal in code (not comment)
    # would imply a live cloud audio stream on the hot path.
    bad_hosts = ("api.openai.com", "generativelanguage.googleapis", "bedrock", "api.hume.ai", "api.cartesia.ai")
    offenders: dict[str, list[str]] = {}
    for fname in HOT_PATH_FILES:
        p = SCRIPTS / fname
        if not p.exists():
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]  # strip trailing comment
            hits = [h for h in bad_hosts if h in code]
            if hits:
                offenders.setdefault(fname, []).append(f"L{lineno}: {hits}")
    assert not offenders, f"external vendor endpoint literals on hot path (AC-8 violation): {offenders}"
