#!/usr/bin/env python3
"""Benchmark Voxtral TTS across all modes: standard, reduced denoising, speculative.

Runs a diverse sentence set through each configuration and reports RTF,
audio duration, generation time, and acceptance rates (for speculative mode).

Usage:
    python3 scripts/benchmark-tts.py
    python3 scripts/benchmark-tts.py --precision 6bit --draft-heads adapters/draft-heads/heads.safetensors
    python3 scripts/benchmark-tts.py --output proof-artifacts/benchmark.json
    # With weights under adapters/draft-heads/, speculative benchmark runs automatically (see draft_heads_resolve.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BENCHMARK_SENTENCES = [
    "Hello there.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today? I hope everything is going well for you.",
    "Machine learning models can now generate human quality speech in real time on consumer hardware.",
    "The weather today is sunny with a high of seventy two degrees Fahrenheit and low winds from the southwest.",
    "To be or not to be, that is the question. Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
    "The acoustic transformer uses Euler integration with classifier free guidance, running multiple denoising steps per audio frame.",
    "One, two, three, four, five. Testing numerical pronunciation and short pauses between items in a list.",
    "Can you believe it? This is amazing! I never thought we'd get here so quickly.",
]


def estimate_max_frames(text, headroom=2.0):
    n_words = len(text.split())
    est_seconds = n_words / 2.5
    max_frames = int(est_seconds * 12.5 * headroom)
    return max(50, min(max_frames, 500))


def benchmark_standard(model, sentences, voice, denoise_steps, n_runs=1):
    """Benchmark standard generation."""
    results = []
    for text in sentences:
        max_tok = estimate_max_frames(text)
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            chunks = []
            for r in model.generate(text=text, voice=voice, max_tokens=max_tok):
                if r.audio is not None:
                    chunks.append(np.array(r.audio))
            elapsed = time.time() - t0
            audio = np.concatenate(chunks) if chunks else np.zeros(0)
            times.append(elapsed)
            dur = len(audio) / 24000

        avg_time = np.mean(times)
        results.append({
            "text": text[:80],
            "n_words": len(text.split()),
            "audio_s": round(dur, 2),
            "gen_s": round(avg_time, 3),
            "rtf": round(avg_time / dur, 4) if dur > 0 else 0,
            "max_tokens": max_tok,
        })
    return results


def benchmark_speculative(model, sentences, voice, head_set, n_runs=1,
                          acoustic_tolerance=2, acoustic_threshold=0.7):
    """Benchmark speculative generation."""
    sys.path.insert(0, os.path.dirname(__file__))
    from voxtral_speculative import speculative_generate

    results = []
    for text in sentences:
        times = []
        last_stats = {}
        for _ in range(n_runs):
            audio, stats = speculative_generate(
                model, text, voice, head_set,
                acoustic_tolerance=acoustic_tolerance,
                acoustic_threshold=acoustic_threshold,
            )
            times.append(stats["elapsed_s"])
            last_stats = stats

        avg_time = np.mean(times)
        dur = last_stats.get("audio_duration_s", 0)
        results.append({
            "text": text[:80],
            "n_words": len(text.split()),
            "audio_s": dur,
            "gen_s": round(avg_time, 3),
            "rtf": round(avg_time / dur, 4) if dur > 0 else 0,
            "frames": last_stats.get("total_frames", 0),
            "lm_steps": last_stats.get("lm_steps", 0),
            "accept_rate": last_stats.get("acceptance_rate", 0),
            "acoustic_sim": last_stats.get("acoustic_accept_avg", 0),
        })
    return results


def print_results(label, results):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'Sentence':<45} {'Words':>5} {'Audio':>6} {'Gen':>6} {'RTF':>7}")
    print(f"  {'-'*45} {'-'*5} {'-'*6} {'-'*6} {'-'*7}")

    for r in results:
        txt = r["text"][:43] + ".." if len(r["text"]) > 43 else r["text"]
        print(f"  {txt:<45} {r['n_words']:>5} {r['audio_s']:>5.1f}s {r['gen_s']:>5.2f}s {r['rtf']:>6.3f}x")

    rtfs = [r["rtf"] for r in results if r["rtf"] > 0]
    gen_times = [r["gen_s"] for r in results]
    audio_durs = [r["audio_s"] for r in results]

    print(f"\n  Summary:")
    print(f"    Mean RTF:    {np.mean(rtfs):.4f}x ({1/np.mean(rtfs):.1f}x real-time)")
    print(f"    Median RTF:  {np.median(rtfs):.4f}x")
    print(f"    Total audio: {sum(audio_durs):.1f}s generated in {sum(gen_times):.1f}s")

    if "accept_rate" in results[0]:
        accept_rates = [r["accept_rate"] for r in results]
        ac_sims = [r.get("acoustic_sim", 0) for r in results]
        print(f"    Mean acceptance: {np.mean(accept_rates):.1%}")
        print(f"    Mean acoustic similarity: {np.mean(ac_sims):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Voxtral TTS modes")
    parser.add_argument("--precision", default="6bit")
    parser.add_argument("--draft-heads", default=None)
    parser.add_argument("--n-heads", type=int, default=3)
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per sentence for averaging")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from draft_heads_resolve import resolve_draft_heads_path

    draft_path = resolve_draft_heads_path(args.draft_heads)

    from mlx_audio.tts.utils import load as mlx_audio_load

    models_map = {
        "4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "6bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
        "bf16": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
    }

    print(f"Loading Voxtral {args.precision}...", flush=True)
    model = mlx_audio_load(models_map[args.precision])
    voice = "cheerful_male"

    all_results = {}

    print("\n[1/3] Benchmarking: Standard 8-step denoising")
    model.acoustic_transformer.args.n_denoising_steps = 8
    results_8 = benchmark_standard(model, BENCHMARK_SENTENCES, voice, 8, n_runs=args.runs)
    print_results(f"Standard (8-step, {args.precision})", results_8)
    all_results["standard_8step"] = results_8

    print("\n[2/3] Benchmarking: Reduced 4-step denoising")
    model.acoustic_transformer.args.n_denoising_steps = 4
    results_4 = benchmark_standard(model, BENCHMARK_SENTENCES, voice, 4, n_runs=args.runs)
    print_results(f"Reduced denoising (4-step, {args.precision})", results_4)
    all_results["reduced_4step"] = results_4

    if draft_path:
        print("\n[3/3] Benchmarking: Speculative decoding (4-step + draft heads)")
        from voxtral_speculative import DraftHeadSet

        head_set = DraftHeadSet.load(draft_path, n_heads=args.n_heads)
        results_spec = benchmark_speculative(
            model, BENCHMARK_SENTENCES, voice, head_set, n_runs=args.runs,
        )
        print_results(f"Speculative (4-step + draft heads, {args.precision})", results_spec)
        all_results["speculative"] = results_spec
    else:
        print("\n[3/3] Skipping speculative benchmark (no --draft-heads provided)")

    # Comparison summary
    print(f"\n{'='*80}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*80}")
    for mode, results in all_results.items():
        rtfs = [r["rtf"] for r in results if r["rtf"] > 0]
        mean_rtf = np.mean(rtfs)
        print(f"  {mode:<25} mean RTF={mean_rtf:.4f}x  ({1/mean_rtf:.1f}x real-time)")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
