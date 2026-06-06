"""Phase-A spec (specs/phase-a-speech-lane) Task 1 / step 1: the audio-render
manifest maps every self-correction scenario to a planned wav with its utterance
text. Pure (no TTS, no disk).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from render_scenario_audio import build_render_manifest  # noqa: E402
from self_correction import load_self_correction_scenarios  # noqa: E402


def test_manifest_covers_all_scenarios_with_text_and_paths():
    scenarios = load_self_correction_scenarios()
    manifest = build_render_manifest(scenarios, "/tmp/render-test")
    assert len(manifest) == len(scenarios) == 21
    by_id = {m["id"]: m for m in manifest}
    for s in scenarios:
        m = by_id[s["id"]]
        assert m["text"] == s["utterance"]                 # renders the actual utterance
        assert m["wav"].endswith(f"{s['id']}.wav")
        assert m["wav"].startswith("/tmp/render-test")
        assert m["rendered"] is False and m["duration_s"] is None  # not yet rendered


def test_manifest_wav_paths_unique():
    manifest = build_render_manifest(load_self_correction_scenarios(), "/tmp/x")
    wavs = [m["wav"] for m in manifest]
    assert len(set(wavs)) == len(wavs)
