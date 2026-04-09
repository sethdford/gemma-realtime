#!/usr/bin/env python3
"""
Comprehensive red team for the full cascaded speech pipeline:

    Audio → Whisper ASR → text → Gemma LLM → response → Speech Decoder → SNAC → audio

Tests every stage individually and end-to-end with latency benchmarks.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Stage 1: Whisper ASR ──────────────────────────────────────

def test_whisper_asr(valid_items, n=10):
    """Test Whisper ASR quality and latency on validation data."""
    import torch
    import whisper
    import soundfile as sf

    print(f"\n{'─'*60}")
    print("  TEST 1: Whisper ASR (Audio → Text)")
    print(f"{'─'*60}")

    model = whisper.load_model("small", device="cpu")
    if torch.backends.mps.is_available():
        model = model.to("mps")
    print(f"  Whisper small loaded on {'MPS' if torch.backends.mps.is_available() else 'CPU'}")

    wer_scores = []
    latencies = []
    results_items = []

    for i, item in enumerate(valid_items[:n]):
        audio_path = item.get("audio_path", "")
        if not audio_path or not Path(audio_path).exists():
            continue
        true_text = item["text"].strip().lower()

        t0 = time.time()
        result = whisper.transcribe(model, audio_path, language="en", fp16=False)
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)

        pred_text = result["text"].strip().lower()
        true_words = true_text.split()
        pred_words = pred_text.split()
        overlap = len(set(true_words) & set(pred_words))
        word_acc = overlap / max(len(true_words), 1)
        wer_scores.append(word_acc)

        results_items.append({
            "true": true_text,
            "pred": pred_text,
            "text_for_llm": result["text"].strip(),
        })

        if i < 5:
            print(f"  [{i+1}] TRUE: {true_text[:70]}")
            print(f"       PRED: {pred_text[:70]}")
            print(f"       Word recall: {word_acc*100:.0f}% | Latency: {elapsed:.0f}ms")
            print()

    avg_recall = np.mean(wer_scores) * 100 if wer_scores else 0
    avg_lat = np.mean(latencies) if latencies else 0
    p50_lat = np.percentile(latencies, 50) if latencies else 0
    p95_lat = np.percentile(latencies, 95) if latencies else 0

    recall_pass = avg_recall > 70
    lat_pass = p50_lat < 2000

    print(f"  Results ({len(wer_scores)} samples):")
    print(f"    Word recall: {avg_recall:.1f}% {'PASS' if recall_pass else 'FAIL'} (threshold: >70%)")
    print(f"    Latency P50: {p50_lat:.0f}ms, P95: {p95_lat:.0f}ms {'PASS' if lat_pass else 'FAIL'} (<2000ms)")

    del model
    return {
        "asr_recall": avg_recall,
        "asr_latency_p50": p50_lat,
        "asr_latency_p95": p95_lat,
        "asr_pass": recall_pass and lat_pass,
        "transcriptions": results_items,
    }


# ── Stage 2: Gemma LLM Response ──────────────────────────────

def test_gemma_llm(transcriptions, n=5):
    """Test Gemma's response quality and latency for transcribed speech."""
    import mlx.core as mx
    from mlx_lm import load as lm_load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"\n{'─'*60}")
    print("  TEST 2: Gemma LLM (Text → Response)")
    print(f"{'─'*60}")

    model, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    print("  Gemma loaded", flush=True)

    ttfts = []
    gen_speeds = []
    responses = []

    for i, item in enumerate(transcriptions[:n]):
        user_text = item["text_for_llm"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Respond briefly to: {user_text}"}],
            tokenize=False, add_generation_prompt=True,
        )

        t0 = time.time()
        first_token_time = None
        text_parts = []
        n_tokens = 0

        sampler = make_sampler(temp=0.7)
        for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=100, sampler=sampler):
            if first_token_time is None:
                first_token_time = time.time()
            text = resp.text or ""
            if "<end_of_turn>" in text:
                text_parts.append(text.split("<end_of_turn>")[0])
                break
            text_parts.append(text)
            n_tokens += 1

        elapsed = time.time() - t0
        ttft = (first_token_time - t0) * 1000 if first_token_time else elapsed * 1000
        tps = n_tokens / elapsed if elapsed > 0 else 0
        ttfts.append(ttft)
        gen_speeds.append(tps)

        response = "".join(text_parts).strip()
        responses.append(response)
        is_coherent = len(response) > 5 and not response.startswith("<") and response[0].isalpha()

        if i < 5:
            print(f"  [{i+1}] USER: {user_text[:60]}")
            print(f"       RESP: {response[:80]}")
            print(f"       TTFT: {ttft:.0f}ms | {tps:.0f} tok/s | Coherent: {is_coherent}")
            print()

    avg_ttft = np.mean(ttfts) if ttfts else 0
    avg_tps = np.mean(gen_speeds) if gen_speeds else 0
    coherent_count = sum(1 for r in responses if len(r) > 5 and not r.startswith("<"))

    ttft_pass = avg_ttft < 1500
    coherent_pass = coherent_count >= len(responses) * 0.8

    print(f"  Results:")
    print(f"    TTFT: {avg_ttft:.0f}ms {'PASS' if ttft_pass else 'FAIL'} (<1500ms)")
    print(f"    Gen speed: {avg_tps:.0f} tok/s")
    print(f"    Coherent: {coherent_count}/{len(responses)} {'PASS' if coherent_pass else 'FAIL'}")

    del model
    return {
        "llm_ttft": avg_ttft,
        "llm_tps": avg_tps,
        "llm_coherent": coherent_count,
        "llm_total": len(responses),
        "llm_pass": ttft_pass and coherent_pass,
        "responses": responses,
    }


# ── Stage 3: Speech Decoder ──────────────────────────────────

def test_speech_decoder(responses, tokenizer_name="mlx-community/gemma-4-26b-a4b-it-4bit"):
    """Test speech decoder: text response → SNAC codec tokens."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils

    print(f"\n{'─'*60}")
    print("  TEST 3: Speech Decoder (Response → SNAC tokens)")
    print(f"{'─'*60}")

    decoder_path = Path("adapters/speech-decoder/speech_decoder.safetensors")
    if not decoder_path.exists():
        print("  No decoder checkpoint — SKIP")
        return {"decoder_pass": None}

    sys.path.insert(0, "scripts")
    from speech_decoder import SpeechDecoder

    from mlx_lm import load as lm_load
    model, tokenizer = lm_load(tokenizer_name)
    model.freeze()
    if hasattr(model, "language_model"):
        inner = model.language_model.model
    else:
        inner = model.model

    decoder = SpeechDecoder(llm_dim=2816)
    dec_weights = mx.load(str(decoder_path))
    decoder.load_weights(list(dec_weights.items()))
    print("  Decoder loaded", flush=True)

    eos_count = 0
    valid_range_count = 0
    latencies = []
    total = 0

    for i, response in enumerate(responses[:5]):
        if not response or len(response) < 3:
            continue
        total += 1

        # Sentence-level: decode first sentence only (matching training distribution)
        first_sentence = response.split('.')[0] + '.'
        ids = tokenizer.encode(first_sentence[:100], add_special_tokens=False)
        if not ids:
            continue
        emb = inner.embed_tokens(mx.array([ids]))

        t0 = time.time()
        pred_tokens_mx = decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(pred_tokens_mx)
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)

        pred_tokens = pred_tokens_mx[0].tolist() if pred_tokens_mx.ndim > 1 else pred_tokens_mx.tolist()
        # generate() returns tokens without EOS; EOS = generated fewer than max_tokens
        has_eos = len(pred_tokens) < decoder.max_tokens
        valid_tokens = pred_tokens
        in_range = all(0 <= t < 4096 for t in valid_tokens) if valid_tokens else True

        if has_eos:
            eos_count += 1
        if in_range:
            valid_range_count += 1

        if i < 3:
            print(f"  [{i+1}] Response: {first_sentence[:50]}")
            print(f"       SNAC tokens: {len(valid_tokens)} (EOS: {has_eos}, range: {in_range})")
            print(f"       Latency: {elapsed:.1f}ms")
            print()

    avg_lat = np.mean(latencies) if latencies else 0
    eos_rate = eos_count / max(total, 1) * 100
    range_rate = valid_range_count / max(total, 1) * 100

    eos_pass = eos_rate >= 80
    range_pass = range_rate >= 90
    lat_pass = avg_lat < 50

    print(f"  Results ({total} samples):")
    print(f"    EOS rate: {eos_rate:.0f}% {'PASS' if eos_pass else 'FAIL'} (>=80%)")
    print(f"    Valid range: {range_rate:.0f}% {'PASS' if range_pass else 'FAIL'} (>=90%)")
    print(f"    Latency: {avg_lat:.1f}ms {'PASS' if lat_pass else 'FAIL'} (<50ms)")

    del model, decoder
    return {
        "decoder_eos_rate": eos_rate,
        "decoder_range_rate": range_rate,
        "decoder_latency": avg_lat,
        "decoder_pass": eos_pass and range_pass,
    }


# ── Stage 4: End-to-End Pipeline ─────────────────────────────

def test_end_to_end(valid_items, n=5):
    """Full pipeline: Audio → Whisper → Gemma → Decoder → tokens, with total latency."""
    import torch
    import whisper
    import soundfile as sf
    import mlx.core as mx
    from mlx_lm import load as lm_load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"\n{'─'*60}")
    print("  TEST 4: End-to-End Pipeline")
    print("  Audio → Whisper → Gemma → Speech Decoder → SNAC")
    print(f"{'─'*60}")

    # Load all components
    print("  Loading components...", flush=True)
    whisper_model = whisper.load_model("small", device="cpu")
    if torch.backends.mps.is_available():
        whisper_model = whisper_model.to("mps")

    gemma_model, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    if hasattr(gemma_model, "language_model"):
        inner = gemma_model.language_model.model
    else:
        inner = gemma_model.model

    # Try loading decoder
    decoder = None
    decoder_path = Path("adapters/speech-decoder/speech_decoder.safetensors")
    if decoder_path.exists():
        sys.path.insert(0, "scripts")
        from speech_decoder import SpeechDecoder
        decoder = SpeechDecoder(llm_dim=2816)
        dec_weights = mx.load(str(decoder_path))
        decoder.load_weights(list(dec_weights.items()))

    print("  All components loaded", flush=True)

    total_latencies = []
    stage_latencies = {"asr": [], "llm": [], "decoder": []}

    for i, item in enumerate(valid_items[:n]):
        audio_path = item.get("audio_path", "")
        if not audio_path or not Path(audio_path).exists():
            continue
        true_text = item["text"].strip()

        total_t0 = time.time()

        # Stage 1: Whisper ASR
        asr_t0 = time.time()
        result = whisper.transcribe(whisper_model, audio_path, language="en", fp16=False)
        asr_time = (time.time() - asr_t0) * 1000
        stage_latencies["asr"].append(asr_time)
        asr_text = result["text"].strip()

        # Stage 2: Gemma LLM
        llm_t0 = time.time()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Respond briefly to: {asr_text}"}],
            tokenize=False, add_generation_prompt=True,
        )
        text_parts = []
        sampler = make_sampler(temp=0.7)
        for resp in stream_generate(gemma_model, tokenizer, prompt=prompt, max_tokens=60, sampler=sampler):
            text = resp.text or ""
            if "<end_of_turn>" in text:
                text_parts.append(text.split("<end_of_turn>")[0])
                break
            text_parts.append(text)
        llm_response = "".join(text_parts).strip()
        llm_time = (time.time() - llm_t0) * 1000
        stage_latencies["llm"].append(llm_time)

        # Stage 3: Speech decoder
        dec_time = 0
        snac_count = 0
        if decoder and llm_response:
            dec_t0 = time.time()
            ids = tokenizer.encode(llm_response, add_special_tokens=False)
            if ids:
                # Sentence-level decoding: take first sentence only
                first_sentence = llm_response.split('.')[0] + '.'
                sent_ids = tokenizer.encode(first_sentence[:100], add_special_tokens=False)
                emb = inner.embed_tokens(mx.array([sent_ids]))
                pred_tokens_mx = decoder.generate(emb, temperature=0.0, top_k=0)
                mx.eval(pred_tokens_mx)
                pred_tokens = pred_tokens_mx[0].tolist() if pred_tokens_mx.ndim > 1 else pred_tokens_mx.tolist()
                snac_count = len(pred_tokens)
            dec_time = (time.time() - dec_t0) * 1000
        stage_latencies["decoder"].append(dec_time)

        total_time = (time.time() - total_t0) * 1000
        total_latencies.append(total_time)

        print(f"  [{i+1}] Input: {true_text[:60]}")
        print(f"       ASR:  {asr_text[:60]} ({asr_time:.0f}ms)")
        print(f"       LLM:  {llm_response[:60]} ({llm_time:.0f}ms)")
        print(f"       SNAC: {snac_count} tokens ({dec_time:.0f}ms)")
        print(f"       TOTAL: {total_time:.0f}ms")
        print()

    avg_total = np.mean(total_latencies) if total_latencies else 0
    avg_asr = np.mean(stage_latencies["asr"]) if stage_latencies["asr"] else 0
    avg_llm = np.mean(stage_latencies["llm"]) if stage_latencies["llm"] else 0
    avg_dec = np.mean(stage_latencies["decoder"]) if stage_latencies["decoder"] else 0

    total_pass = avg_total < 5000  # 5s total budget

    print(f"  Pipeline Latency Breakdown:")
    print(f"    Whisper ASR:    {avg_asr:.0f}ms")
    print(f"    Gemma LLM:      {avg_llm:.0f}ms")
    print(f"    Speech Decoder: {avg_dec:.0f}ms")
    print(f"    Total:          {avg_total:.0f}ms {'PASS' if total_pass else 'FAIL'} (<5000ms)")

    return {
        "e2e_total": avg_total,
        "e2e_asr": avg_asr,
        "e2e_llm": avg_llm,
        "e2e_decoder": avg_dec,
        "e2e_pass": total_pass,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  DEEP RED TEAM: Full Cascaded Speech Pipeline")
    print("  Audio → Whisper ASR → Gemma LLM → Speech Decoder → SNAC")
    print("=" * 70)

    # Load validation data
    valid_data = []
    with open("data/libritts-valid.jsonl") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("audio_path") and item.get("text"):
                if Path(item["audio_path"]).exists():
                    valid_data.append(item)
    print(f"\n  Validation samples with audio: {len(valid_data)}")

    # Run tests
    results = {}

    # Test 1: ASR
    asr = test_whisper_asr(valid_data, n=10)
    results.update(asr)

    # Test 2: LLM
    llm = test_gemma_llm(asr["transcriptions"][:5])
    results.update(llm)

    # Test 3: Decoder
    dec = test_speech_decoder(llm["responses"])
    results.update(dec)

    # Test 4: End-to-end
    e2e = test_end_to_end(valid_data[:5])
    results.update(e2e)

    # ── Scorecard ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SCORECARD: Full Cascaded Pipeline")
    print(f"{'='*70}")

    checks = [
        ("Whisper ASR word recall > 70%",     results.get("asr_pass", False),       f"{results.get('asr_recall', 0):.1f}%"),
        ("ASR latency P50 < 2000ms",          results.get("asr_latency_p50", 9999) < 2000, f"{results.get('asr_latency_p50', 0):.0f}ms"),
        ("Gemma TTFT < 1500ms",               results.get("llm_ttft", 9999) < 1500, f"{results.get('llm_ttft', 0):.0f}ms"),
        ("Gemma responses coherent > 80%",    results.get("llm_pass", False),       f"{results.get('llm_coherent', 0)}/{results.get('llm_total', 0)}"),
        ("Decoder EOS rate >= 80%",            results.get("decoder_eos_rate", 0) >= 80, f"{results.get('decoder_eos_rate', 0):.0f}%"),
        ("Decoder tokens in range >= 90%",     results.get("decoder_range_rate", 0) >= 90, f"{results.get('decoder_range_rate', 0):.0f}%"),
        ("End-to-end total < 5000ms",          results.get("e2e_pass", False),       f"{results.get('e2e_total', 0):.0f}ms"),
    ]

    passed = 0
    for name, p, val in checks:
        if val is None:
            status = "SKIP"
        elif p:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
        print(f"  [{status:4s}] {name:40s} = {val}")

    total_checks = sum(1 for _, _, v in checks if v is not None)
    print(f"\n  Score: {passed}/{total_checks} checks passed")

    print(f"\n  Latency Breakdown:")
    print(f"    Whisper ASR (P50): {results.get('asr_latency_p50', 0):.0f}ms")
    print(f"    Gemma TTFT:        {results.get('llm_ttft', 0):.0f}ms")
    print(f"    Gemma gen speed:   {results.get('llm_tps', 0):.0f} tok/s")
    print(f"    Speech decoder:    {results.get('decoder_latency', 0):.1f}ms")
    print(f"    E2E total:         {results.get('e2e_total', 0):.0f}ms")

    print(f"\n  Pipeline Quality:")
    print(f"    ASR word recall:   {results.get('asr_recall', 0):.1f}%")
    print(f"    LLM coherent:      {results.get('llm_coherent', 0)}/{results.get('llm_total', 0)}")
    print(f"    Decoder EOS:       {results.get('decoder_eos_rate', 0):.0f}%")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
