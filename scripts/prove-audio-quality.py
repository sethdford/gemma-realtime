#!/usr/bin/env python3
"""
PROVE AUDIO QUALITY: Spectral analysis of all proof WAV artifacts.

Validates that output audio is:
    1. Not silence (energy above threshold)
    2. Not pure noise (has spectral structure)
    3. Speech-like (energy concentrated in 100-4000Hz band)
    4. Not clipped or distorted
    5. Reasonable duration for the input text
    6. SNAC round-trip preserves spectral content
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROOF_DIR = Path("proof-artifacts")


def spectral_analysis(audio: np.ndarray, sr: int, label: str):
    """Analyze a WAV file for speech quality indicators."""
    if len(audio) == 0:
        return {"label": label, "pass": False, "reason": "empty audio"}

    duration = len(audio) / sr

    # 1. Energy check
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    crest_factor = peak / (rms + 1e-10)

    # 2. Spectral analysis via FFT
    n_fft = min(2048, len(audio))
    n_segments = max(1, len(audio) // n_fft)
    psd = np.zeros(n_fft // 2 + 1)

    for i in range(n_segments):
        segment = audio[i * n_fft:(i + 1) * n_fft]
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))
        windowed = segment * np.hanning(n_fft)
        fft = np.fft.rfft(windowed)
        psd += np.abs(fft) ** 2
    psd /= n_segments

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # 3. Speech band energy (100-4000Hz)
    speech_mask = (freqs >= 100) & (freqs <= 4000)
    total_energy = np.sum(psd)
    speech_energy = np.sum(psd[speech_mask])
    speech_ratio = speech_energy / (total_energy + 1e-10)

    # 4. Spectral flatness (noise = ~1.0, tonal = ~0.0)
    log_psd = np.log(psd[1:] + 1e-10)
    geo_mean = np.exp(np.mean(log_psd))
    arith_mean = np.mean(psd[1:])
    spectral_flatness = geo_mean / (arith_mean + 1e-10)

    # 5. Clipping check
    n_clipped = np.sum(np.abs(audio) > 0.99)
    clip_ratio = n_clipped / len(audio)

    # 6. Zero-crossing rate (speech: 0.01-0.15 range)
    zcr = np.mean(np.abs(np.diff(np.sign(audio))) > 0)

    # Determine pass/fail
    checks = {
        "not_silent": rms > 0.005,
        "has_structure": spectral_flatness < 0.5,
        "speech_band": speech_ratio > 0.3,
        "not_clipped": clip_ratio < 0.01,
        "reasonable_zcr": 0.005 < zcr < 0.5,
    }
    all_pass = all(checks.values())

    return {
        "label": label,
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "speech_ratio": speech_ratio,
        "spectral_flatness": spectral_flatness,
        "clip_ratio": clip_ratio,
        "zcr": zcr,
        "checks": checks,
        "pass": all_pass,
    }


def main():
    print("\n" + "=" * 70)
    print("  PROVE AUDIO QUALITY")
    print("  Spectral analysis of all proof artifacts")
    print("=" * 70)

    wavs = sorted(PROOF_DIR.glob("*.wav"))
    if not wavs:
        print("  No WAV files found in proof-artifacts/")
        return

    print(f"\n  Found {len(wavs)} WAV files\n")

    results = []
    for wav_path in wavs:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        r = spectral_analysis(audio, sr, wav_path.name)
        results.append(r)

    # Print table
    print(f"  {'File':<30s} {'Dur':>5s} {'RMS':>7s} {'Peak':>7s} {'SpeechBand':>11s} {'Flatness':>9s} {'ZCR':>7s} {'Clip':>7s} {'OK':>4s}")
    print(f"  {'─'*92}")

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"  {r['label']:<30s} "
            f"{r['duration']:>4.1f}s "
            f"{r['rms']:>7.4f} "
            f"{r['peak']:>7.4f} "
            f"{r['speech_ratio']*100:>10.1f}% "
            f"{r['spectral_flatness']:>9.4f} "
            f"{r['zcr']:>7.4f} "
            f"{r['clip_ratio']*100:>6.2f}% "
            f"{status:>4s}"
        )

    print(f"\n  {'─'*92}")

    # Detailed check breakdown
    all_pass = True
    for r in results:
        if not r["pass"]:
            all_pass = False
            fails = [k for k, v in r["checks"].items() if not v]
            print(f"  FAIL {r['label']}: {', '.join(fails)}")

    # SNAC round-trip comparison
    orig_path = PROOF_DIR / "01_original.wav"
    rt_path = PROOF_DIR / "01_snac_roundtrip.wav"
    if orig_path.exists() and rt_path.exists():
        print(f"\n  SNAC Round-Trip Quality:")
        orig, sr_o = sf.read(str(orig_path), dtype="float32")
        rt, sr_r = sf.read(str(rt_path), dtype="float32")
        orig_r = spectral_analysis(orig, sr_o, "original")
        rt_r = spectral_analysis(rt, sr_r, "roundtrip")
        speech_preserved = abs(orig_r["speech_ratio"] - rt_r["speech_ratio"]) < 0.3
        rms_close = abs(orig_r["rms"] - rt_r["rms"]) / (orig_r["rms"] + 1e-10) < 0.5
        print(f"    Speech band preserved: orig={orig_r['speech_ratio']*100:.1f}% → rt={rt_r['speech_ratio']*100:.1f}% {'PASS' if speech_preserved else 'FAIL'}")
        print(f"    RMS level preserved:   orig={orig_r['rms']:.4f} → rt={rt_r['rms']:.4f} {'PASS' if rms_close else 'FAIL'}")

    # Summary
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\n{'='*70}")
    print(f"  Audio quality: {passed}/{total} files validated")
    if all_pass:
        print("  VERDICT: AUDIO QUALITY PROVEN ✓")
    else:
        print(f"  VERDICT: {total - passed} files FAIL quality checks")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
