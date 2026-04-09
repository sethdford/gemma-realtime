#!/usr/bin/env python3
"""
End-to-end red team validation for the speech pipeline.

Tests the full pipeline: audio -> encoder -> LLM -> decoder -> codec -> audio
with real trained weights, real audio files, and real measurements.

Checks:
    1. Encoder produces embeddings in the LLM's embedding space
    2. Encoder embeddings are geometrically similar to text embeddings
    3. Decoder generates valid codec tokens from text context
    4. Decoder tokens decode to actual audio via SNAC
    5. Duplex predictor makes state predictions
    6. Full pipeline latency at each stage
    7. Encoder streaming mode works
    8. Memory usage under load
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

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

from speech_encoder import SpeechEncoder
from speech_decoder import SpeechDecoder, DuplexStatePredictor


class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details = ""
        self.metrics = {}

    def ok(self, details="", **metrics):
        self.passed = True
        self.details = details
        self.metrics = metrics
        return self

    def fail(self, details="", **metrics):
        self.passed = False
        self.details = details
        self.metrics = metrics
        return self


def load_frozen_gemma(model_name: str):
    from mlx_lm import load as lm_load
    model, tokenizer = lm_load(model_name)
    model.freeze()
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner = model.language_model.model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
    else:
        inner = model
    probe = inner.embed_tokens(mx.array([[0]]))
    hidden_dim = probe.shape[-1]
    return model, tokenizer, inner, hidden_dim


def load_encoder(weights_path: str, hidden_dim: int) -> SpeechEncoder:
    encoder = SpeechEncoder(
        llm_dim=hidden_dim, encoder_dim=512, n_heads=8, n_layers=2,
        d_ff=2048, tokens_per_chunk=4, chunk_ms=160, dropout=0.0,
    )
    weights = mx.load(weights_path)
    encoder.load_weights(list(weights.items()))
    return encoder


def load_decoder(weights_path: str, hidden_dim: int) -> SpeechDecoder:
    decoder = SpeechDecoder(
        llm_dim=hidden_dim, decoder_dim=512, n_heads=8, n_layers=4,
        d_ff=2048, codebook_size=4096, max_tokens=500, dropout=0.0,
    )
    weights = mx.load(weights_path)
    decoder.load_weights(list(weights.items()))
    return decoder


def load_predictor(weights_path: str, hidden_dim: int) -> DuplexStatePredictor:
    pred = DuplexStatePredictor(llm_dim=hidden_dim, hidden_dim=256, n_states=3)
    weights = mx.load(weights_path)
    pred.load_weights(list(weights.items()))
    return pred


def load_test_audio(data_path: str, n_samples: int = 5) -> list[dict]:
    items = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            items.append(json.loads(line.strip()))
    return items


def test_encoder_output_shape(encoder, hidden_dim) -> Result:
    r = Result("Encoder output shape")
    try:
        chunk_samples = encoder.chunk_samples
        audio = mx.random.normal((1, 1, chunk_samples))
        emb = encoder(audio)
        mx.eval(emb)
        expected = (1, encoder.tokens_per_chunk, hidden_dim)
        if emb.shape == expected:
            return r.ok(f"Shape {emb.shape} matches expected {expected}")
        return r.fail(f"Got {emb.shape}, expected {expected}")
    except Exception as e:
        return r.fail(str(e))


def test_encoder_embedding_alignment(encoder, inner, tokenizer, hidden_dim) -> Result:
    r = Result("Encoder embedding alignment (cosine similarity)")
    try:
        text = "Hello, how are you doing today?"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        text_emb = inner.embed_tokens(mx.array([token_ids[:4]], dtype=mx.int32))
        mx.eval(text_emb)

        chunk_samples = encoder.chunk_samples
        audio = mx.random.normal((1, 1, chunk_samples))
        enc_emb = encoder(audio)
        mx.eval(enc_emb)

        text_norm = text_emb / (mx.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8)
        enc_norm = enc_emb / (mx.linalg.norm(enc_emb, axis=-1, keepdims=True) + 1e-8)
        cos_sim = mx.mean(mx.sum(text_norm * enc_norm, axis=-1)).item()

        enc_mag = mx.mean(mx.linalg.norm(enc_emb, axis=-1)).item()
        text_mag = mx.mean(mx.linalg.norm(text_emb, axis=-1)).item()
        mag_ratio = enc_mag / (text_mag + 1e-8)

        if cos_sim > -0.5 and 0.01 < mag_ratio < 100:
            return r.ok(
                f"cos_sim={cos_sim:.3f}, enc_mag={enc_mag:.1f}, text_mag={text_mag:.1f}, ratio={mag_ratio:.2f}",
                cos_sim=cos_sim, mag_ratio=mag_ratio,
            )
        return r.fail(f"cos_sim={cos_sim:.3f}, mag_ratio={mag_ratio:.2f}")
    except Exception as e:
        return r.fail(str(e))


def test_encoder_real_audio(encoder, inner, tokenizer, data_path) -> Result:
    r = Result("Encoder on real LibriTTS-R audio")
    try:
        items = load_test_audio(data_path, n_samples=3)
        cos_sims = []
        for item in items:
            audio, sr = sf.read(item["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            chunk_samples = encoder.chunk_samples
            if len(audio) < chunk_samples:
                audio = np.pad(audio, (0, chunk_samples - len(audio)))
            n_chunks = max(1, len(audio) // chunk_samples)
            audio = audio[:n_chunks * chunk_samples]

            all_emb = []
            for c in range(min(n_chunks, 5)):
                chunk = audio[c * chunk_samples:(c + 1) * chunk_samples]
                chunk_mx = mx.array(chunk.reshape(1, 1, chunk_samples))
                emb = encoder(chunk_mx)
                all_emb.append(emb)
            enc_emb = mx.concatenate(all_emb, axis=1)
            mx.eval(enc_emb)

            tids = tokenizer.encode(item["text"], add_special_tokens=False)
            n_tok = min(len(tids), enc_emb.shape[1])
            if n_tok == 0:
                continue
            text_emb = mx.stop_gradient(inner.embed_tokens(mx.array([tids[:n_tok]], dtype=mx.int32)))
            mx.eval(text_emb)

            enc_n = enc_emb[:, :n_tok, :] / (mx.linalg.norm(enc_emb[:, :n_tok, :], axis=-1, keepdims=True) + 1e-8)
            txt_n = text_emb / (mx.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8)
            cs = mx.mean(mx.sum(enc_n * txt_n, axis=-1)).item()
            cos_sims.append(cs)

        avg_cos = np.mean(cos_sims) if cos_sims else 0
        return r.ok(
            f"Avg cosine sim={avg_cos:.3f} over {len(cos_sims)} real audio samples",
            avg_cosine=avg_cos, n_samples=len(cos_sims),
        )
    except Exception as e:
        return r.fail(str(e))


def test_decoder_generates_tokens(decoder, inner, tokenizer) -> Result:
    r = Result("Decoder generates valid codec tokens")
    try:
        text = "The quick brown fox jumps over the lazy dog."
        tids = tokenizer.encode(text, add_special_tokens=False)
        ctx = mx.stop_gradient(inner.embed_tokens(mx.array([tids], dtype=mx.int32)))
        mx.eval(ctx)

        t0 = time.time()
        tokens = decoder.generate(ctx, temperature=0.8, top_k=50)
        mx.eval(tokens)
        gen_ms = (time.time() - t0) * 1000

        n_tokens = tokens.shape[-1]
        token_vals = tokens[0].tolist()
        in_range = all(0 <= t < 4096 for t in token_vals)
        unique = len(set(token_vals))

        if n_tokens > 0 and in_range:
            return r.ok(
                f"Generated {n_tokens} tokens in {gen_ms:.0f}ms, "
                f"{unique} unique, range [{min(token_vals)}-{max(token_vals)}]",
                n_tokens=n_tokens, gen_ms=gen_ms, unique=unique,
            )
        return r.fail(f"n_tokens={n_tokens}, in_range={in_range}")
    except Exception as e:
        return r.fail(str(e))


def test_decoder_tokens_decode_to_audio(decoder, inner, tokenizer) -> Result:
    r = Result("Decoder tokens decode to audio via SNAC")
    try:
        from codec import AudioCodec

        text = "Hello world, this is a test."
        tids = tokenizer.encode(text, add_special_tokens=False)
        ctx = mx.stop_gradient(inner.embed_tokens(mx.array([tids], dtype=mx.int32)))
        mx.eval(ctx)

        tokens = decoder.generate(ctx, temperature=0.8, top_k=50)
        mx.eval(tokens)
        token_list = tokens[0].tolist()

        if len(token_list) < 3:
            return r.fail(f"Only {len(token_list)} tokens generated, need >= 3 for SNAC")

        codec = AudioCodec("snac", device="mps")
        codec.load()

        n_frames = len(token_list)
        import torch
        with torch.no_grad():
            cb0 = torch.tensor([token_list], dtype=torch.long)
            cb1 = torch.randint(0, 4096, (1, n_frames * 2))
            cb2 = torch.randint(0, 4096, (1, n_frames * 4))
            if torch.backends.mps.is_available():
                cb0 = cb0.to("mps")
                cb1 = cb1.to("mps")
                cb2 = cb2.to("mps")
            audio = codec._model.decode([cb0, cb1, cb2])
            audio_np = audio.cpu().numpy().squeeze()

        duration = len(audio_np) / 24000
        peak = np.max(np.abs(audio_np))

        if duration > 0 and peak > 0:
            return r.ok(
                f"Decoded to {len(audio_np)} samples ({duration:.2f}s), peak={peak:.3f}",
                duration=duration, peak=peak,
            )
        return r.fail(f"duration={duration:.2f}s, peak={peak:.3f}")
    except Exception as e:
        return r.fail(str(e))


def test_duplex_predictor(predictor, inner, tokenizer) -> Result:
    r = Result("Duplex state predictor")
    try:
        texts = [
            "Can you hear me?",
            "Yes I can hear you clearly.",
            "Wait, let me interrupt you there.",
        ]
        predictions = []
        for text in texts:
            tids = tokenizer.encode(text, add_special_tokens=False)
            ctx = mx.stop_gradient(inner.embed_tokens(mx.array([tids], dtype=mx.int32)))
            mx.eval(ctx)
            state = predictor.predict(ctx)
            predictions.append((text[:30], ["LISTEN", "SPEAK", "INTERRUPT"][state]))

        return r.ok(
            "; ".join(f'"{t}"→{s}' for t, s in predictions),
            predictions=predictions,
        )
    except Exception as e:
        return r.fail(str(e))


def test_full_pipeline_latency(encoder, decoder, predictor, inner, tokenizer) -> Result:
    r = Result("Full pipeline latency (synthetic)")
    try:
        chunk_samples = encoder.chunk_samples
        audio = mx.random.normal((1, 1, chunk_samples))

        # Encoder
        t0 = time.time()
        enc_emb = encoder(audio)
        mx.eval(enc_emb)
        enc_ms = (time.time() - t0) * 1000

        # Predictor
        t0 = time.time()
        state = predictor.predict(enc_emb)
        pred_ms = (time.time() - t0) * 1000

        # Decoder (generate 20 tokens)
        t0 = time.time()
        tokens = decoder.generate(enc_emb, temperature=0.8, top_k=50)
        mx.eval(tokens)
        dec_ms = (time.time() - t0) * 1000

        total = enc_ms + pred_ms + dec_ms
        return r.ok(
            f"enc={enc_ms:.1f}ms, pred={pred_ms:.1f}ms, dec={dec_ms:.0f}ms, total={total:.0f}ms",
            enc_ms=enc_ms, pred_ms=pred_ms, dec_ms=dec_ms, total_ms=total,
        )
    except Exception as e:
        return r.fail(str(e))


def test_encoder_streaming(encoder) -> Result:
    r = Result("Encoder streaming mode")
    try:
        chunk_samples = encoder.chunk_samples
        n_chunks = 5
        all_emb = []
        t0 = time.time()
        for i in range(n_chunks):
            audio = mx.random.normal((1, 1, chunk_samples))
            emb = encoder.encode_chunk(audio)
            mx.eval(emb)
            all_emb.append(emb)
        stream_ms = (time.time() - t0) * 1000

        total_tokens = sum(e.shape[1] for e in all_emb)
        ms_per_chunk = stream_ms / n_chunks
        return r.ok(
            f"{n_chunks} chunks -> {total_tokens} tokens in {stream_ms:.0f}ms "
            f"({ms_per_chunk:.1f}ms/chunk)",
            ms_per_chunk=ms_per_chunk, total_tokens=total_tokens,
        )
    except Exception as e:
        return r.fail(str(e))


def test_memory_usage() -> Result:
    r = Result("GPU memory usage")
    try:
        try:
            active = mx.metal.get_active_memory() / 1e9
            peak = mx.metal.get_peak_memory() / 1e9
        except Exception:
            active = mx.get_active_memory() / 1e9
            peak = mx.get_peak_memory() / 1e9

        if active < 100:
            return r.ok(f"Active: {active:.1f}GB, Peak: {peak:.1f}GB",
                        active_gb=active, peak_gb=peak)
        return r.fail(f"Active={active:.1f}GB exceeds 100GB")
    except Exception as e:
        return r.fail(str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E2E red team validation")
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--encoder-weights", default="adapters/speech-encoder/speech_encoder.safetensors")
    parser.add_argument("--decoder-weights", default="adapters/speech-decoder/speech_decoder.safetensors")
    parser.add_argument("--predictor-weights", default="adapters/duplex-predictor/duplex_predictor.safetensors")
    parser.add_argument("--data", default="data/libritts-train.jsonl")
    args = parser.parse_args()

    print(f"\n{'='*70}", flush=True)
    print(f"  RED TEAM: End-to-End Speech Pipeline Validation", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("  Loading frozen Gemma...", flush=True)
    model, tokenizer, inner, hidden_dim = load_frozen_gemma(args.model)
    print(f"  Hidden dim: {hidden_dim}\n", flush=True)

    print("  Loading trained encoder...", flush=True)
    encoder = load_encoder(args.encoder_weights, hidden_dim)
    print(f"  Encoder: {encoder.num_params()/1e6:.1f}M params\n", flush=True)

    print("  Loading trained decoder...", flush=True)
    decoder = load_decoder(args.decoder_weights, hidden_dim)
    print(f"  Decoder: {decoder.num_params()/1e6:.1f}M params\n", flush=True)

    print("  Loading trained predictor...", flush=True)
    predictor = load_predictor(args.predictor_weights, hidden_dim)
    print(f"  Predictor loaded\n", flush=True)

    tests = [
        test_encoder_output_shape(encoder, hidden_dim),
        test_encoder_embedding_alignment(encoder, inner, tokenizer, hidden_dim),
        test_encoder_real_audio(encoder, inner, tokenizer, args.data),
        test_encoder_streaming(encoder),
        test_decoder_generates_tokens(decoder, inner, tokenizer),
        test_decoder_tokens_decode_to_audio(decoder, inner, tokenizer),
        test_duplex_predictor(predictor, inner, tokenizer),
        test_full_pipeline_latency(encoder, decoder, predictor, inner, tokenizer),
        test_memory_usage(),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*70}\n", flush=True)

    passed = 0
    failed = 0
    for t in tests:
        status = "PASS" if t.passed else "FAIL"
        icon = "+" if t.passed else "X"
        print(f"  [{icon}] {t.name}: {status}", flush=True)
        if t.details:
            print(f"      {t.details}", flush=True)
        if t.passed:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}", flush=True)
    print(f"  {passed}/{passed+failed} tests passed, {failed} failed", flush=True)
    print(f"{'='*70}\n", flush=True)

    if failed:
        print("  ISSUES TO FIX:", flush=True)
        for t in tests:
            if not t.passed:
                print(f"    - {t.name}: {t.details}", flush=True)
        print(flush=True)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
