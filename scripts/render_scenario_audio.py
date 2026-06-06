#!/usr/bin/env python3
"""Render audio fixtures for the Phase-A conversational gates (Task 1 / step 1).

TTS-renders the self-correction scenario utterances to .wav via on-device Kokoro
(24 kHz), so the streaming runner (conversational_runner.run_lane_conversational)
has real audio to drive fish/cascade inference against. Audio lands under data/
(gitignored); a manifest.json records id -> wav + duration.

  python3 scripts/render_scenario_audio.py            # render the 21 scenarios
  python3 scripts/render_scenario_audio.py --voice af_bella --speed 1.0

The manifest builder is pure/testable; only render() touches the TTS + disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_correction import load_self_correction_scenarios

SAMPLE_RATE = 24000  # Kokoro output rate (matches speech-server.SAMPLE_RATE)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "realistic-audio" / "self_correction"


def build_render_manifest(scenarios: list[dict], out_dir: str | Path) -> list[dict]:
    """Pure: map each scenario to its planned render record (no TTS, no disk)."""
    out_dir = Path(out_dir)
    manifest: list[dict] = []
    for s in scenarios:
        manifest.append({
            "id": s["id"],
            "text": s["utterance"],
            "wav": str(out_dir / f"{s['id']}.wav"),
            "domain": s.get("domain"),
            "correction_type": s.get("correction_type"),
            "rendered": False,
            "duration_s": None,
        })
    return manifest


def render(scenarios: list[dict], out_dir: str | Path = DEFAULT_OUT,
           voice: str = "af_bella", speed: float = 1.0) -> list[dict]:
    """Render each scenario utterance to a wav. Returns the populated manifest."""
    import soundfile as sf
    from kokoro import KPipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")
    manifest = build_render_manifest(scenarios, out_dir)

    for item in manifest:
        samples = []
        for r in pipeline(item["text"], voice=voice, speed=speed):
            if r.audio is not None:
                samples.append(r.audio.numpy() if hasattr(r.audio, "numpy") else np.array(r.audio))
        if not samples:
            print(f"  [{item['id']}] no audio produced", flush=True)
            continue
        audio = np.concatenate(samples).astype(np.float32)
        sf.write(item["wav"], audio, SAMPLE_RATE)
        item["rendered"] = True
        item["duration_s"] = round(len(audio) / SAMPLE_RATE, 3)
        print(f"  [{item['id']}] {item['duration_s']:.2f}s -> {Path(item['wav']).name}", flush=True)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Render self-correction scenario audio fixtures")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--voice", default="af_bella")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    scenarios = load_self_correction_scenarios()
    manifest = render(scenarios, out_dir=args.out, voice=args.voice, speed=args.speed)
    manifest_path = Path(args.out) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_ok = sum(1 for m in manifest if m["rendered"])
    print(f"\nrendered {n_ok}/{len(manifest)} scenarios -> {args.out}\nmanifest: {manifest_path}")
    return 0 if n_ok == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
