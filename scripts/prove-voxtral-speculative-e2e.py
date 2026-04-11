#!/usr/bin/env python3
"""
E2E proof: Voxtral 6bit + on-disk draft heads -> measurable audio output.

Loads the same resolution rules as speech-server / realtime-ws (see draft_heads_resolve).
Exits 0 on success, 1 on failure. Requires Apple Silicon + mlx-audio + model cache.

Usage:
    python3 scripts/prove-voxtral-speculative-e2e.py
    python3 scripts/prove-voxtral-speculative-e2e.py --draft-heads adapters/draft-heads/heads-libri.safetensors
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from draft_heads_resolve import resolve_draft_heads_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove Voxtral speculative decode end-to-end")
    parser.add_argument("--draft-heads", default=None, help="Override draft heads path")
    parser.add_argument("--voice", default="cheerful_male", help="Voxtral voice (Libri heads trained on cheerful_male)")
    parser.add_argument("--precision", default="6bit", choices=["4bit", "6bit", "bf16"])
    parser.add_argument("--denoise-steps", type=int, default=4, choices=[2, 4, 6, 8])
    args = parser.parse_args()

    heads_path = resolve_draft_heads_path(args.draft_heads)
    if not heads_path:
        print(
            "[FAIL] No draft heads found. Train or place weights at "
            "adapters/draft-heads/heads-libri.safetensors or set VOXTRAL_DRAFT_HEADS.",
            file=sys.stderr,
        )
        return 1

    proof_dir = _SCRIPTS.parent / "proof-artifacts"
    proof_dir.mkdir(exist_ok=True)
    out_wav = proof_dir / "voxtral_speculative_e2e.wav"

    from mlx_audio.tts.utils import load as mlx_audio_load

    from voxtral_speculative import DraftHeadSet, speculative_generate

    models = {
        "4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "6bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
        "bf16": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
    }

    text = "This is an end to end check of speculative decoding with trained draft heads."

    print(f"Loading Voxtral {args.precision}...", flush=True)
    model = mlx_audio_load(models[args.precision])
    if args.denoise_steps != 8:
        model.acoustic_transformer.args.n_denoising_steps = args.denoise_steps

    print(f"Loading draft heads: {heads_path}", flush=True)
    head_set = DraftHeadSet.load(heads_path, n_heads=3)

    t0 = time.time()
    audio, stats = speculative_generate(
        model,
        text,
        args.voice,
        head_set,
        verbose=False,
    )
    elapsed = time.time() - t0

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    dur_s = len(audio) / 24000.0

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    ok = True
    if len(audio) < 2000:
        print(f"[FAIL] Audio too short: {len(audio)} samples", file=sys.stderr)
        ok = False
    # RMS varies with utterance/gain; gate on non-silence (not digital silence)
    if rms < 0.005 and peak < 0.02:
        print(f"[FAIL] Audio too quiet (RMS={rms:.6f}, peak={peak:.4f})", file=sys.stderr)
        ok = False

    import soundfile as sf

    sf.write(str(out_wav), audio, 24000)

    acc = stats.get("acceptance_rate", 0.0)
    print(f"  Wrote {out_wav} ({dur_s:.2f}s audio, {elapsed:.2f}s wall, RTF~{elapsed/dur_s:.3f}x, accept={acc:.1%})")
    print(f"  RMS={rms:.4f}")

    if ok:
        print("[PASS] Voxtral speculative E2E")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
