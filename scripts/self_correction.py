#!/usr/bin/env python3
"""Loader for the Phase-A self-correction scenario set (AC-6, Task 7).

specs/phase-a-speech-lane. Each scenario is a mid-utterance intent change
(FDB-v3 self-correction structure). The audio is rendered downstream by the
chosen lane's TTS; here we just load and validate the intent-pairs that the
self-correction Pass@1 metric scores against.
"""
from __future__ import annotations

import json
from pathlib import Path

# Authored source-of-truth fixture (tracked with the spec, as .json because the
# repo .gitignore blocks *.jsonl for privacy). The rendered AUDIO for these
# scenarios lives under data/ (gitignored, built by Task 1).
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "specs" / "phase-a-speech-lane" / "self_correction_scenarios.json"

REQUIRED_FIELDS = ("id", "domain", "correction_type", "utterance", "original_intent", "corrected_intent", "expected_final")


def load_self_correction_scenarios(path: str | Path | None = None) -> list[dict]:
    """Load + validate the self-correction scenarios. Raises ValueError on a
    malformed set so a bad fixture fails loud rather than silently scoring 0."""
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"self-correction scenarios not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a JSON array of scenarios")
    else:  # .jsonl — one object per line
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    scenarios: list[dict] = []
    ids: set[str] = set()
    for i, obj in enumerate(rows):
        where = f"{p}[{i}]"
        for f in REQUIRED_FIELDS:
            if f not in obj:
                raise ValueError(f"{where} missing required field {f!r}")
        if obj["id"] in ids:
            raise ValueError(f"{where} duplicate scenario id {obj['id']!r}")
        ids.add(obj["id"])
        # A self-correction must actually change the intent.
        if obj["original_intent"] == obj["corrected_intent"]:
            raise ValueError(f"{where} ({obj['id']}) original_intent == corrected_intent — not a correction")
        scenarios.append(obj)
    return scenarios


if __name__ == "__main__":
    s = load_self_correction_scenarios()
    print(f"loaded {len(s)} self-correction scenarios from {DEFAULT_PATH}")
    from collections import Counter
    print("by correction_type:", dict(Counter(x["correction_type"] for x in s)))
    print("by domain:", dict(Counter(x["domain"] for x in s)))
