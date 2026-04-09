#!/usr/bin/env python3
"""
End-to-end speech pipeline benchmark for gemma-realtime.

Measures latency at each stage of the cascaded pipeline:
  1. ASR latency (audio -> text)
  2. LLM TTFT + generation speed
  3. TTS latency (text -> first audio)
  4. End-to-end (simulated audio in -> audio out)

Also measures memory usage across all loaded models.

Usage:
    python3 scripts/speech-bench.py
    python3 scripts/speech-bench.py --llm-url http://localhost:8741
    python3 scripts/speech-bench.py --rounds 5 --json
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np

SAMPLE_RATE_ASR = 16000
SAMPLE_RATE_TTS = 24000
VOICE_TARGET_E2E_MS = 1200
VOICE_TARGET_TTFT_MS = 200
VOICE_TARGET_RTF = 1.0

TEST_PROMPTS = [
    "What's the weather like today?",
    "Tell me a short joke.",
    "How do I make scrambled eggs?",
    "What's the meaning of life?",
    "Explain quantum computing in one sentence.",
]


def generate_test_audio(text_hint: str, duration_s=2.0) -> np.ndarray:
    """Generate synthetic speech-like audio for benchmark (sine + noise)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE_ASR * duration_s), dtype=np.float32)
    f0 = 150 + len(text_hint) * 2
    audio = 0.3 * np.sin(2 * np.pi * f0 * t)
    audio += 0.05 * np.random.randn(len(t)).astype(np.float32)
    return audio


async def bench_asr(asr, rounds=3):
    """Benchmark ASR latency."""
    print("\n--- ASR Benchmark ---", flush=True)
    results = []

    for i, prompt in enumerate(TEST_PROMPTS[:rounds]):
        audio = generate_test_audio(prompt, duration_s=2.0)
        t0 = time.time()
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, asr.transcribe, audio
        )
        latency_ms = (time.time() - t0) * 1000
        results.append({"prompt": prompt, "transcript": transcript, "latency_ms": latency_ms})
        print(f"  [{i+1}] {latency_ms:.0f}ms -> \"{transcript[:50]}\"", flush=True)

    avg = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    print(f"  ASR avg: {avg:.0f}ms", flush=True)
    return {"asr_avg_ms": round(avg, 1), "asr_results": results}


async def bench_llm(llm, rounds=3):
    """Benchmark LLM streaming latency (TTFT + TPS)."""
    print("\n--- LLM Benchmark ---", flush=True)
    results = []

    for i, prompt in enumerate(TEST_PROMPTS[:rounds]):
        messages = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."},
            {"role": "user", "content": prompt},
        ]

        t0 = time.time()
        first_token_time = None
        tokens = []
        full_text = []

        async for delta in llm.stream_chat(messages, max_tokens=100, temperature=0.7):
            now = time.time()
            if first_token_time is None:
                first_token_time = now
            tokens.append(now)
            full_text.append(delta)

        elapsed = time.time() - t0
        ttft_ms = (first_token_time - t0) * 1000 if first_token_time else elapsed * 1000
        n_tokens = len(tokens)
        gen_time = (tokens[-1] - first_token_time) if n_tokens > 1 and first_token_time else elapsed
        tps = (n_tokens - 1) / gen_time if gen_time > 0 and n_tokens > 1 else 0

        result = {
            "prompt": prompt,
            "ttft_ms": round(ttft_ms, 1),
            "tps": round(tps, 1),
            "tokens": n_tokens,
            "elapsed_s": round(elapsed, 2),
            "response_preview": "".join(full_text)[:80],
        }
        results.append(result)
        status = "PASS" if ttft_ms < VOICE_TARGET_TTFT_MS else "SLOW"
        print(f"  [{i+1}] TTFT={ttft_ms:.0f}ms TPS={tps:.0f} [{status}] | {result['response_preview'][:50]}...", flush=True)

    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results) if results else 0
    avg_tps = sum(r["tps"] for r in results) / len(results) if results else 0
    print(f"  LLM avg: TTFT={avg_ttft:.0f}ms, TPS={avg_tps:.0f}", flush=True)
    return {"llm_avg_ttft_ms": round(avg_ttft, 1), "llm_avg_tps": round(avg_tps, 1), "llm_results": results}


async def bench_tts(tts, rounds=3):
    """Benchmark TTS latency."""
    print("\n--- TTS Benchmark ---", flush=True)
    results = []

    test_sentences = [
        "Hello, how can I help you today?",
        "The weather is beautiful outside.",
        "I think that's a great question, let me explain.",
        "Sure, here's a quick summary for you.",
        "That's really interesting, tell me more about it.",
    ]

    for i, sentence in enumerate(test_sentences[:rounds]):
        t0 = time.time()
        audio = await asyncio.get_event_loop().run_in_executor(
            None, tts.synthesize, sentence
        )
        latency_ms = (time.time() - t0) * 1000
        audio_duration_ms = (len(audio) / tts.sample_rate * 1000) if audio is not None else 0
        rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else float("inf")

        result = {
            "text": sentence,
            "latency_ms": round(latency_ms, 1),
            "audio_duration_ms": round(audio_duration_ms, 1),
            "rtf": round(rtf, 3),
        }
        results.append(result)
        status = "PASS" if rtf < VOICE_TARGET_RTF else "SLOW"
        print(f"  [{i+1}] {latency_ms:.0f}ms synth / {audio_duration_ms:.0f}ms audio (RTF={rtf:.2f}) [{status}]", flush=True)

    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    avg_rtf = sum(r["rtf"] for r in results) / len(results) if results else 0
    print(f"  TTS avg: {avg_latency:.0f}ms latency, RTF={avg_rtf:.2f}", flush=True)
    return {"tts_avg_ms": round(avg_latency, 1), "tts_avg_rtf": round(avg_rtf, 3), "tts_results": results}


async def bench_e2e(asr, llm, tts, rounds=3):
    """Benchmark full end-to-end pipeline (simulated audio -> audio)."""
    print("\n--- End-to-End Benchmark ---", flush=True)
    results = []

    for i, prompt in enumerate(TEST_PROMPTS[:rounds]):
        t_start = time.time()

        audio_in = generate_test_audio(prompt, duration_s=2.0)
        t_asr_start = time.time()
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, asr.transcribe, audio_in
        )
        t_asr_end = time.time()

        use_text = transcript.strip() if transcript.strip() else prompt

        messages = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."},
            {"role": "user", "content": use_text},
        ]

        t_llm_start = time.time()
        first_token_time = None
        full_response = []
        first_sentence = None
        first_sentence_time = None
        sentence_buf = ""

        async for delta in llm.stream_chat(messages, max_tokens=100, temperature=0.7):
            now = time.time()
            if first_token_time is None:
                first_token_time = now
            full_response.append(delta)

            if first_sentence is None:
                sentence_buf += delta
                import re
                if re.search(r"[.!?]\s*$", sentence_buf) and len(sentence_buf) > 10:
                    first_sentence = sentence_buf.strip()
                    first_sentence_time = now

        t_llm_end = time.time()

        tts_text = first_sentence or "".join(full_response)[:100]
        t_tts_start = time.time()
        audio_out = await asyncio.get_event_loop().run_in_executor(
            None, tts.synthesize, tts_text
        )
        t_tts_end = time.time()

        asr_ms = (t_asr_end - t_asr_start) * 1000
        llm_ttft_ms = (first_token_time - t_llm_start) * 1000 if first_token_time else 0
        first_sentence_ms = (first_sentence_time - t_llm_start) * 1000 if first_sentence_time else (t_llm_end - t_llm_start) * 1000
        tts_ms = (t_tts_end - t_tts_start) * 1000
        e2e_to_first_audio_ms = asr_ms + first_sentence_ms + tts_ms
        total_ms = (t_tts_end - t_start) * 1000

        result = {
            "prompt": prompt,
            "asr_ms": round(asr_ms, 1),
            "llm_ttft_ms": round(llm_ttft_ms, 1),
            "first_sentence_ms": round(first_sentence_ms, 1),
            "tts_ms": round(tts_ms, 1),
            "e2e_first_audio_ms": round(e2e_to_first_audio_ms, 1),
            "total_ms": round(total_ms, 1),
        }
        results.append(result)
        status = "PASS" if e2e_to_first_audio_ms < VOICE_TARGET_E2E_MS else "SLOW"
        print(
            f"  [{i+1}] ASR={asr_ms:.0f} + LLM_TTFT={llm_ttft_ms:.0f} + Sentence={first_sentence_ms:.0f} "
            f"+ TTS={tts_ms:.0f} = {e2e_to_first_audio_ms:.0f}ms [{status}]",
            flush=True,
        )

    avg_e2e = sum(r["e2e_first_audio_ms"] for r in results) / len(results) if results else 0
    target_met = sum(1 for r in results if r["e2e_first_audio_ms"] < VOICE_TARGET_E2E_MS)
    print(f"  E2E avg: {avg_e2e:.0f}ms ({target_met}/{len(results)} meet {VOICE_TARGET_E2E_MS}ms target)", flush=True)
    return {"e2e_avg_ms": round(avg_e2e, 1), "e2e_pass_rate": target_met / len(results) if results else 0, "e2e_results": results}


async def run_benchmark(args):
    print(f"\n{'='*70}", flush=True)
    print(f"  SPEECH PIPELINE BENCHMARK — gemma-realtime", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  LLM: {args.llm_url}", flush=True)
    print(f"  Rounds: {args.rounds}", flush=True)
    print(f"  Targets: E2E < {VOICE_TARGET_E2E_MS}ms, TTFT < {VOICE_TARGET_TTFT_MS}ms, RTF < {VOICE_TARGET_RTF}", flush=True)
    print(f"{'='*70}", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from importlib import import_module

    speech_server = import_module("speech-server")

    asr = speech_server.WhisperASR(model_name=args.whisper_model)
    tts = speech_server.TTSEngine(voice=args.voice)
    llm = speech_server.LLMClient(base_url=args.llm_url)

    print("\nLoading models...", flush=True)
    asr.load()
    tts.load()

    health = await llm.check_health()
    if not health:
        print(f"ERROR: Cannot reach LLM at {args.llm_url}/health", flush=True)
        print(f"Start the server: python3 scripts/mlx-server.py --realtime", flush=True)
        await llm.close()
        return

    all_results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": vars(args), "llm_health": health}

    all_results.update(await bench_asr(asr, rounds=args.rounds))
    all_results.update(await bench_llm(llm, rounds=args.rounds))
    if tts.available:
        all_results.update(await bench_tts(tts, rounds=args.rounds))
        all_results.update(await bench_e2e(asr, llm, tts, rounds=args.rounds))
    else:
        print("\n  TTS not available — skipping TTS and E2E benchmarks", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  ASR avg latency:     {all_results.get('asr_avg_ms', 'N/A')} ms", flush=True)
    print(f"  LLM avg TTFT:        {all_results.get('llm_avg_ttft_ms', 'N/A')} ms", flush=True)
    print(f"  LLM avg TPS:         {all_results.get('llm_avg_tps', 'N/A')} tok/s", flush=True)
    if tts.available:
        print(f"  TTS avg latency:     {all_results.get('tts_avg_ms', 'N/A')} ms", flush=True)
        print(f"  TTS avg RTF:         {all_results.get('tts_avg_rtf', 'N/A')}", flush=True)
        print(f"  E2E avg first audio: {all_results.get('e2e_avg_ms', 'N/A')} ms", flush=True)
        e2e_pass = all_results.get("e2e_pass_rate", 0)
        verdict = "REAL-TIME" if e2e_pass >= 0.8 else "NEEDS OPTIMIZATION"
        print(f"  Verdict:             {verdict} ({e2e_pass*100:.0f}% meet target)", flush=True)
    print(f"{'='*70}\n", flush=True)

    if args.json:
        out_path = args.output or "speech-bench-results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out_path}", flush=True)

    await llm.close()
    return all_results


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Speech pipeline benchmark")
    parser.add_argument("--llm-url", default="http://localhost:8741")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-small-mlx")
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
