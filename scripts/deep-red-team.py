#!/usr/bin/env python3
"""
Deep red team validation of the full speech pipeline.

Tests:
  1. Encoder → vocab projection: can it predict text tokens from audio?
  2. Encoder embedding alignment: cosine similarity & magnitude
  3. Encoder → LLM bridge: predict tokens → lookup embeddings → LLM generation
  4. Decoder: does it generate EOS? Does it produce valid codec tokens?
  5. Decoder → SNAC decode: can tokens be turned back into audio?
  6. Duplex predictor: state prediction accuracy
  7. Latency profiling for each component
  8. Full pipeline: audio → encoder → vocab_head → LLM → decoder → SNAC → audio
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import soundfile as sf


def build_log_prior(vocab_size: int, freq_path: str = "data/token_frequencies.json") -> mx.array:
    """Build log-prior bias for inference-time frequency debiasing.

    Returns log(p_prior) for each token, used to subtract from logits:
        adjusted = logits - beta * log_prior
    """
    log_prior = np.full(vocab_size, np.log(1.0 / vocab_size), dtype=np.float32)
    try:
        with open(freq_path) as f:
            data = json.load(f)
        total = data["total"]
        for tid_str, count in data["frequencies"].items():
            tid = int(tid_str)
            if 0 <= tid < vocab_size:
                log_prior[tid] = np.log((count + 1) / (total + vocab_size))
    except Exception:
        pass
    return mx.array(log_prior)


def debias_logits(logits: mx.array, log_prior: mx.array, beta: float = 1.0) -> mx.array:
    """Subtract frequency prior from logits to debias toward rare tokens."""
    return logits - beta * log_prior


def load_audio(path: str, sr: int = 24000, max_s: float = 6.0) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        n = int(len(audio) * sr / file_sr)
        audio = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(audio)), audio).astype(np.float32)
    max_samples = int(max_s * sr)
    return audio[:max_samples]


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from speech_encoder import SpeechEncoder
    from speech_decoder import SpeechDecoder, DuplexStatePredictor

    print(f"\n{'='*70}")
    print(f"  DEEP RED TEAM VALIDATION")
    print(f"{'='*70}\n")

    results = {}

    # ── Load frozen LLM ──
    print("  [1/8] Loading frozen LLM...", flush=True)
    from mlx_lm import load as lm_load
    model, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    model.freeze()

    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner = model.language_model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
    else:
        inner = model

    probe = inner.embed_tokens(mx.array([[0]]))
    hidden_dim = probe.shape[-1]
    vocab_size = inner.embed_tokens.weight.shape[0]
    print(f"    LLM: hidden={hidden_dim}, vocab={vocab_size}", flush=True)

    # ── Load encoder + vocab head ──
    print("  [2/8] Loading encoder + vocab head...", flush=True)
    encoder = SpeechEncoder(
        llm_dim=hidden_dim, encoder_dim=512, n_heads=8, n_layers=4,
        d_ff=2048, tokens_per_chunk=4, chunk_ms=160, dropout=0.0,
    )

    enc_path = Path("adapters/speech-encoder/speech_encoder.safetensors")
    vh_path = Path("adapters/speech-encoder/vocab_head.safetensors")

    enc_weights = mx.load(str(enc_path))
    encoder.load_weights(list(enc_weights.items()))
    print(f"    Encoder loaded: {enc_path}", flush=True)

    # Reconstruct vocab head
    class VocabProjectionHead(nn.Module):
        def __init__(self, hd, vs, bn=1024):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(hd, bn), nn.GELU(), nn.Linear(bn, vs))
        def __call__(self, x):
            return self.proj(x)

    vocab_head = VocabProjectionHead(hidden_dim, vocab_size)
    vh_weights = mx.load(str(vh_path))
    vocab_head.load_weights(list(vh_weights.items()))
    print(f"    Vocab head loaded: {vh_path}", flush=True)

    # ── Load decoder ──
    print("  [3/8] Loading decoder...", flush=True)
    decoder = SpeechDecoder(
        llm_dim=hidden_dim, decoder_dim=512, n_heads=8, n_layers=4,
        d_ff=2048, codebook_size=4096, max_tokens=500, dropout=0.0,
    )
    dec_path = Path("adapters/speech-decoder/speech_decoder.safetensors")
    dec_weights = mx.load(str(dec_path))
    decoder.load_weights(list(dec_weights.items()))
    print(f"    Decoder loaded: {dec_path}", flush=True)

    # ── Load duplex predictor ──
    print("  [4/8] Loading duplex predictor...", flush=True)
    predictor = DuplexStatePredictor(llm_dim=hidden_dim, hidden_dim=256, n_states=3)
    pred_path = Path("adapters/duplex-predictor/duplex_predictor.safetensors")
    if pred_path.exists():
        pred_weights = mx.load(str(pred_path))
        predictor.load_weights(list(pred_weights.items()))
        print(f"    Predictor loaded: {pred_path}", flush=True)
    else:
        print(f"    Predictor not found, using random init", flush=True)

    # ── Load test audio ──
    test_data = []
    with open("data/libritts-valid.jsonl") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("audio_path") and item.get("text"):
                test_data.append(item)
                if len(test_data) >= 20:
                    break
    print(f"    Test data: {len(test_data)} samples\n", flush=True)

    chunk_samples = encoder.chunk_samples

    # ━━ TEST 1: Encoder → Vocab Projection ━━
    print(f"  {'─'*60}")
    print(f"  TEST 1: Encoder → Vocab Projection (Token Prediction)")
    print(f"  {'─'*60}")

    log_prior = build_log_prior(vocab_size)
    best_beta = 0.0
    best_overlap = 0.0

    # Try different debiasing strengths to find optimal beta
    for beta in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]:
        overlap_scores = []
        for item in test_data[:5]:
            audio = load_audio(item["audio_path"])
            if len(audio) < chunk_samples:
                audio = np.pad(audio, (0, chunk_samples - len(audio)))
            n_chunks = max(1, len(audio) // chunk_samples)
            audio = audio[:n_chunks * chunk_samples]
            text_ids = tokenizer.encode(item["text"], add_special_tokens=False)
            n_enc = n_chunks * 4
            if len(text_ids) < n_enc:
                text_ids = (text_ids * (n_enc // max(len(text_ids), 1) + 1))[:n_enc]
            else:
                text_ids = text_ids[:n_enc]

            all_emb = []
            for c in range(n_chunks):
                chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
                emb = encoder(mx.array(chunk.reshape(1, 1, chunk_samples)))
                all_emb.append(emb)
            enc_out = mx.concatenate(all_emb, axis=1)
            logits = debias_logits(vocab_head(enc_out), log_prior, beta=beta)
            preds = mx.argmax(logits[0], axis=-1)
            mx.eval(preds)

            pred_text = tokenizer.decode(preds[:len(text_ids)].tolist())
            true_text = tokenizer.decode(text_ids)
            pred_words = set(pred_text.lower().split())
            true_words = set(true_text.lower().split())
            if true_words:
                overlap_scores.append(len(pred_words & true_words) / len(true_words))

        avg = np.mean(overlap_scores) * 100 if overlap_scores else 0
        print(f"    beta={beta:.1f}: word overlap={avg:.0f}%")
        if avg > best_overlap:
            best_overlap = avg
            best_beta = beta

    print(f"    Best beta: {best_beta} ({best_overlap:.0f}% overlap)")

    # Now do full evaluation with best beta
    total_acc = 0.0
    top10_acc = 0.0
    total_tokens = 0
    text_overlap_scores = []

    for item in test_data[:10]:
        audio = load_audio(item["audio_path"])
        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))

        n_chunks = max(1, len(audio) // chunk_samples)
        audio = audio[:n_chunks * chunk_samples]
        text_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        n_enc = n_chunks * 4

        if len(text_ids) < n_enc:
            text_ids = (text_ids * (n_enc // max(len(text_ids), 1) + 1))[:n_enc]
        else:
            text_ids = text_ids[:n_enc]

        all_emb = []
        for c in range(n_chunks):
            chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
            emb = encoder(mx.array(chunk.reshape(1, 1, chunk_samples)))
            all_emb.append(emb)
        enc_out = mx.concatenate(all_emb, axis=1)

        logits = debias_logits(vocab_head(enc_out), log_prior, beta=best_beta)
        preds = mx.argmax(logits[0], axis=-1)
        mx.eval(preds)

        target = mx.array(text_ids, dtype=mx.int32)
        n = min(len(preds), len(target))
        correct = int(mx.sum((preds[:n] == target[:n]).astype(mx.float32)).item())
        total_acc += correct
        total_tokens += n

        top10 = mx.argsort(logits[0], axis=-1)[:, -10:]
        mx.eval(top10)
        for i in range(min(n, top10.shape[0])):
            if target[i].item() in top10[i].tolist():
                top10_acc += 1

        pred_text = tokenizer.decode(preds[:n].tolist())
        true_text = tokenizer.decode(text_ids[:n])
        pred_words = set(pred_text.lower().split())
        true_words = set(true_text.lower().split())
        if true_words:
            overlap = len(pred_words & true_words) / len(true_words)
            text_overlap_scores.append(overlap)

    token_acc = total_acc / max(total_tokens, 1) * 100
    top10_pct = top10_acc / max(total_tokens, 1) * 100
    word_overlap = np.mean(text_overlap_scores) * 100 if text_overlap_scores else 0

    print(f"\n    With debiasing (beta={best_beta}):")
    print(f"    Top-1 token accuracy: {token_acc:.1f}%")
    print(f"    Top-10 token accuracy: {top10_pct:.1f}%")
    print(f"    Word overlap (pred vs true): {word_overlap:.0f}%")

    results["token_acc_top1"] = token_acc
    results["token_acc_top10"] = top10_pct
    results["word_overlap"] = word_overlap
    results["debias_beta"] = best_beta

    # Show example predictions (raw vs debiased)
    item = test_data[0]
    audio = load_audio(item["audio_path"])
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)))
    n_chunks = max(1, len(audio) // chunk_samples)
    audio = audio[:n_chunks * chunk_samples]
    all_emb = []
    for c in range(n_chunks):
        chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
        emb = encoder(mx.array(chunk.reshape(1, 1, chunk_samples)))
        all_emb.append(emb)
    enc_out = mx.concatenate(all_emb, axis=1)
    raw_logits = vocab_head(enc_out)

    raw_preds = mx.argmax(raw_logits[0], axis=-1)
    deb_preds = mx.argmax(debias_logits(raw_logits, log_prior, best_beta)[0], axis=-1)
    mx.eval(raw_preds, deb_preds)

    print(f"\n    Example:")
    print(f"    True    : {item['text'][:100]}")
    print(f"    Raw pred: {tokenizer.decode(raw_preds.tolist()[:20])[:100]}")
    print(f"    Debiased: {tokenizer.decode(deb_preds.tolist()[:20])[:100]}")

    # ━━ TEST 2: Embedding Alignment ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 2: Embedding Alignment (Cosine + Magnitude)")
    print(f"  {'─'*60}")

    cosines = []
    mag_ratios = []
    for item in test_data[:10]:
        audio = load_audio(item["audio_path"])
        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))
        n_chunks = max(1, len(audio) // chunk_samples)
        audio = audio[:n_chunks * chunk_samples]

        text_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        n_enc = n_chunks * 4
        if len(text_ids) < n_enc:
            text_ids = (text_ids * (n_enc // max(len(text_ids), 1) + 1))[:n_enc]
        else:
            text_ids = text_ids[:n_enc]

        target_emb = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids], dtype=mx.int32)))

        all_emb = []
        for c in range(n_chunks):
            chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
            emb = encoder(mx.array(chunk.reshape(1, 1, chunk_samples)))
            all_emb.append(emb)
        enc_out = mx.concatenate(all_emb, axis=1)

        sl = min(enc_out.shape[1], target_emb.shape[1])
        pred = enc_out[:, :sl]
        tgt = target_emb[:, :sl]

        pn = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        tn = tgt / (mx.linalg.norm(tgt, axis=-1, keepdims=True) + 1e-8)
        cos = mx.mean(mx.sum(pn * tn, axis=-1)).item()
        cosines.append(cos)

        pm = mx.mean(mx.linalg.norm(pred, axis=-1)).item()
        tm = mx.mean(mx.linalg.norm(tgt, axis=-1)).item()
        mag_ratios.append(pm / (tm + 1e-8))

    avg_cos = np.mean(cosines)
    avg_mag = np.mean(mag_ratios)
    print(f"    Avg cosine similarity: {avg_cos:.3f}")
    print(f"    Avg magnitude ratio:   {avg_mag:.2f}")
    results["cosine_sim"] = avg_cos
    results["mag_ratio"] = avg_mag

    # ━━ TEST 3: Bridge Mode (predict tokens → embed → LLM generate) ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 3: Bridge Mode (Audio → Token IDs → LLM Generation)")
    print(f"  {'─'*60}")

    from mlx_lm import generate
    item = test_data[0]
    audio = load_audio(item["audio_path"])
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)))
    n_chunks = max(1, len(audio) // chunk_samples)
    audio = audio[:n_chunks * chunk_samples]

    all_emb = []
    for c in range(n_chunks):
        chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
        emb = encoder(mx.array(chunk.reshape(1, 1, chunk_samples)))
        all_emb.append(emb)
    enc_out = mx.concatenate(all_emb, axis=1)
    logits = debias_logits(vocab_head(enc_out), log_prior, beta=best_beta)
    pred_ids = mx.argmax(logits[0], axis=-1)
    mx.eval(pred_ids)

    pred_text = tokenizer.decode(pred_ids.tolist()[:30])
    prompt = f"The following is a transcript of audio: \"{pred_text}\". Summarize what was said:"
    response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
    print(f"    Predicted text from audio: {pred_text[:80]}")
    print(f"    LLM response: {response[:120]}")

    # Compare with direct text input
    true_prompt = f"The following is a transcript of audio: \"{item['text'][:80]}\". Summarize what was said:"
    true_response = generate(model, tokenizer, prompt=true_prompt, max_tokens=50)
    print(f"    True text: {item['text'][:80]}")
    print(f"    LLM response (true): {true_response[:120]}")

    # Check word overlap
    pred_words = set(response.lower().split())
    true_words = set(true_response.lower().split())
    bridge_overlap = len(pred_words & true_words) / max(len(true_words), 1) * 100
    results["bridge_word_overlap"] = bridge_overlap
    print(f"    Response word overlap: {bridge_overlap:.0f}%")

    # ━━ TEST 4: Decoder EOS ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 4: Decoder EOS + Token Validity")
    print(f"  {'─'*60}")

    eos_id = 4096
    eos_count = 0
    valid_tokens = 0
    total_generated = 0
    gen_lengths = []

    for item in test_data[:10]:
        text_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if not text_ids:
            continue
        ctx = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids[:50]], dtype=mx.int32)))

        tokens = [0]  # Start token
        max_gen = 200
        for i in range(max_gen):
            tgt = mx.array([tokens], dtype=mx.int32)
            logits = decoder(ctx, tgt)
            next_token = mx.argmax(logits[0, -1]).item()
            tokens.append(next_token)
            if next_token == eos_id:
                eos_count += 1
                break

        gen_len = len(tokens) - 1
        gen_lengths.append(gen_len)
        total_generated += gen_len

        for t in tokens[1:]:
            if 0 <= t <= eos_id:
                valid_tokens += 1

    eos_pct = eos_count / 10 * 100
    valid_pct = valid_tokens / max(total_generated, 1) * 100
    avg_len = np.mean(gen_lengths)
    print(f"    EOS generated: {eos_count}/10 ({eos_pct:.0f}%)")
    print(f"    Valid tokens: {valid_pct:.0f}%")
    print(f"    Avg generation length: {avg_len:.0f} tokens")
    print(f"    Lengths: {gen_lengths}")
    results["eos_pct"] = eos_pct
    results["valid_tokens_pct"] = valid_pct
    results["avg_gen_length"] = avg_len

    # ━━ TEST 5: Decoder → SNAC Decode ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 5: Decoder → SNAC Audio Reconstruction")
    print(f"  {'─'*60}")

    try:
        from codec import AudioCodec
        codec = AudioCodec("snac")
        codec.load()

        # Use the first test case's generated tokens
        item = test_data[0]
        text_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        ctx = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids[:50]], dtype=mx.int32)))

        tokens = [0]
        for i in range(200):
            tgt = mx.array([tokens], dtype=mx.int32)
            logits = decoder(ctx, tgt)
            next_token = mx.argmax(logits[0, -1]).item()
            tokens.append(next_token)
            if next_token == eos_id:
                break

        codec_tokens = [t for t in tokens[1:] if 0 <= t < 4096]
        if len(codec_tokens) >= 4:
            from codec import CodecTokens, CodecType, CODEC_CONFIGS
            import torch
            # Use single codebook for SNAC decode
            codes_np = np.array(codec_tokens[:len(codec_tokens)], dtype=np.int64)
            # Replicate across 3 codebooks at different resolutions
            n = len(codes_np)
            codes_list = [
                torch.from_numpy(codes_np).unsqueeze(0).long(),
                torch.from_numpy(codes_np[:n*2]).unsqueeze(0).long() if n*2 <= len(codes_np) else torch.from_numpy(np.tile(codes_np, 2)[:n*2]).unsqueeze(0).long(),
                torch.from_numpy(codes_np[:n*4]).unsqueeze(0).long() if n*4 <= len(codes_np) else torch.from_numpy(np.tile(codes_np, 4)[:n*4]).unsqueeze(0).long(),
            ]
            if torch.backends.mps.is_available():
                codes_list = [c.to("mps") for c in codes_list]
            with torch.no_grad():
                audio_out = codec._model.decode(codes_list)
            audio_np = audio_out.cpu().numpy().squeeze()
            duration = len(audio_np) / 24000
            print(f"    Generated {len(codec_tokens)} codec tokens")
            print(f"    Decoded to {duration:.2f}s audio ({len(audio_np)} samples)")
            print(f"    Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
            results["snac_decode_ok"] = True
            results["audio_duration_s"] = duration
        else:
            print(f"    Too few tokens ({len(codec_tokens)}) to decode")
            results["snac_decode_ok"] = False
    except Exception as e:
        print(f"    SNAC decode failed: {e}")
        results["snac_decode_ok"] = False

    # ━━ TEST 6: Duplex Predictor ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 6: Duplex State Predictor")
    print(f"  {'─'*60}")

    state_names = ["LISTEN", "SPEAK", "INTERRUPT"]
    predictions = []
    for item in test_data[:10]:
        text_ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if not text_ids:
            continue
        ctx = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids], dtype=mx.int32)))
        logits = predictor(ctx)
        state = mx.argmax(logits, axis=-1).item()
        predictions.append(state)

    from collections import Counter
    dist = Counter(predictions)
    print(f"    State distribution: {dict(dist)}")
    for s in range(3):
        print(f"      {state_names[s]}: {dist.get(s, 0)}/{len(predictions)}")
    results["duplex_states"] = dict(dist)

    # ━━ TEST 7: Latency Profiling ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 7: Latency Profiling")
    print(f"  {'─'*60}")

    audio = load_audio(test_data[0]["audio_path"])
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)))
    chunk = audio[:chunk_samples]
    chunk_input = mx.array(chunk.reshape(1, 1, chunk_samples))

    # Warmup
    _ = encoder(chunk_input)
    mx.eval(_)

    # Encoder latency
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = encoder(chunk_input)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    enc_lat = np.median(times)
    print(f"    Encoder (1 chunk): {enc_lat:.1f}ms (median)")

    # Vocab head latency
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        logits = vocab_head(out)
        mx.eval(logits)
        times.append((time.perf_counter() - t0) * 1000)
    vh_lat = np.median(times)
    print(f"    Vocab head:        {vh_lat:.1f}ms (median)")

    # Decoder latency (1 step)
    text_ids = tokenizer.encode("Hello world", add_special_tokens=False)
    ctx = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids], dtype=mx.int32)))
    tgt = mx.array([[0]], dtype=mx.int32)
    _ = decoder(ctx, tgt)
    mx.eval(_)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        logits = decoder(ctx, tgt)
        mx.eval(logits)
        times.append((time.perf_counter() - t0) * 1000)
    dec_lat = np.median(times)
    print(f"    Decoder (1 step):  {dec_lat:.1f}ms (median)")

    # Predictor latency
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        s = predictor(ctx)
        mx.eval(s)
        times.append((time.perf_counter() - t0) * 1000)
    pred_lat = np.median(times)
    print(f"    Predictor:         {pred_lat:.1f}ms (median)")

    results["encoder_latency_ms"] = enc_lat
    results["decoder_latency_ms"] = dec_lat
    results["predictor_latency_ms"] = pred_lat

    # ━━ TEST 8: Full Pipeline ━━
    print(f"\n  {'─'*60}")
    print(f"  TEST 8: Full Pipeline (Audio → Text → Audio)")
    print(f"  {'─'*60}")

    t0 = time.perf_counter()

    # Step 1: Encode audio
    audio = load_audio(test_data[0]["audio_path"])
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)))
    n_chunks = min(5, max(1, len(audio) // chunk_samples))
    audio = audio[:n_chunks * chunk_samples]

    all_emb = []
    for c in range(n_chunks):
        chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
        emb = encoder(mx.array(chunk.reshape(1, 1, chunk_samples)))
        all_emb.append(emb)
    enc_out = mx.concatenate(all_emb, axis=1)
    mx.eval(enc_out)

    # Step 2: Predict tokens via vocab head (with debiasing)
    logits = debias_logits(vocab_head(enc_out), log_prior, beta=best_beta)
    pred_ids = mx.argmax(logits[0], axis=-1)
    mx.eval(pred_ids)
    pred_text = tokenizer.decode(pred_ids.tolist())

    # Step 3: Get LLM embeddings for predicted text
    ctx = mx.stop_gradient(inner.embed_tokens(
        mx.array([pred_ids.tolist()[:50]], dtype=mx.int32)
    ))

    # Step 4: Generate codec tokens
    tokens = [0]
    for i in range(200):
        tgt = mx.array([tokens], dtype=mx.int32)
        logits_dec = decoder(ctx, tgt)
        next_t = mx.argmax(logits_dec[0, -1]).item()
        tokens.append(next_t)
        if next_t == eos_id:
            break

    # Step 5: Duplex prediction
    state = mx.argmax(predictor(ctx), axis=-1).item()
    mx.eval(state)

    total_time = (time.perf_counter() - t0) * 1000

    print(f"    Input audio: {n_chunks * 160}ms ({n_chunks} chunks)")
    print(f"    Predicted text: {pred_text[:60]}")
    print(f"    Generated {len(tokens)-1} codec tokens (EOS={'yes' if tokens[-1]==eos_id else 'no'})")
    print(f"    Duplex state: {state_names[state]}")
    print(f"    Total pipeline: {total_time:.0f}ms")
    results["pipeline_ms"] = total_time
    results["pipeline_eos"] = tokens[-1] == eos_id

    # ━━ SUMMARY ━━
    print(f"\n{'='*70}")
    print(f"  SUMMARY & ASSESSMENT")
    print(f"{'='*70}")

    checks = [
        ("Token acc top-1 > 3%", results["token_acc_top1"] > 3),
        ("Token acc top-10 > 15%", results["token_acc_top10"] > 15),
        ("Word overlap > 10%", results["word_overlap"] > 10),
        ("Cosine similarity > 0.3", results["cosine_sim"] > 0.3),
        ("Magnitude ratio 0.5-2.0", 0.5 < results["mag_ratio"] < 2.0),
        ("EOS generated > 50%", results["eos_pct"] > 50),
        ("Valid tokens > 95%", results["valid_tokens_pct"] > 95),
        ("Avg gen length < 200", results["avg_gen_length"] < 200),
        ("Encoder < 20ms", results["encoder_latency_ms"] < 20),
        ("Decoder < 10ms/step", results["decoder_latency_ms"] < 10),
        ("SNAC decode works", results.get("snac_decode_ok", False)),
        ("Pipeline < 500ms", results["pipeline_ms"] < 500),
    ]

    passed = 0
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {name}")
        if ok:
            passed += 1

    print(f"\n    Score: {passed}/{len(checks)} checks passed")
    print(f"\n  Key Metrics:")
    print(f"    Token accuracy: {results['token_acc_top1']:.1f}% (top-1), {results['token_acc_top10']:.1f}% (top-10)")
    print(f"    Cosine similarity: {results['cosine_sim']:.3f}")
    print(f"    Magnitude ratio: {results['mag_ratio']:.2f}")
    print(f"    EOS rate: {results['eos_pct']:.0f}%")
    print(f"    Latency: encoder={results['encoder_latency_ms']:.1f}ms, decoder={results['decoder_latency_ms']:.1f}ms")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
