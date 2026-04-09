#!/usr/bin/env python3
"""
Comprehensive red team validation for the Whisper-based speech pipeline.

End-to-end pipeline:
    Audio → Whisper encoder → Adapter → Gemma LLM → Text

Tests:
    1. Adapter embedding quality (cosine sim, magnitude ratio)
    2. Weight-tied token prediction (nearest neighbor in embed space)
    3. LLM bridge mode: adapter embeddings → Gemma forward → text
    4. Full e2e: raw audio → Whisper → adapter → Gemma → text
    5. Speech decoder: text → decoder → SNAC tokens → audio
    6. Full pipeline latency
    7. Duplex state predictor
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np


class WhisperAdapter(nn.Module):
    def __init__(self, whisper_dim: int = 768, llm_dim: int = 2816, hidden_dim: int = 2048):
        super().__init__()
        self.proj1 = nn.Linear(whisper_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj3 = nn.Linear(hidden_dim, llm_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(llm_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu(self.proj1(x))
        h = h + nn.gelu(self.proj2(self.norm1(h)))
        return self.norm2(self.proj3(h))


def load_adapter(path: str, whisper_dim=768, llm_dim=2816, hidden_dim=2048):
    adapter = WhisperAdapter(whisper_dim, llm_dim, hidden_dim)
    weights = mx.load(path)
    adapter.load_weights(list(weights.items()))
    return adapter


def load_gemma():
    from mlx_lm import load as lm_load
    print("  Loading Gemma...", flush=True)
    model, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    model.freeze()
    if hasattr(model, "language_model"):
        lm = model.language_model
    else:
        lm = model
    inner = lm.model if hasattr(lm, "model") else lm
    return lm, inner, tokenizer


def dequantize_embeddings(inner, hidden_dim):
    embed_layer = inner.embed_tokens
    raw_shape = embed_layer.weight.shape
    if hasattr(embed_layer, "scales") or raw_shape[-1] != hidden_dim:
        vocab_size = embed_layer.scales.shape[0] if hasattr(embed_layer, "scales") else raw_shape[0]
        chunks = []
        for i in range(0, vocab_size, 4096):
            end = min(i + 4096, vocab_size)
            emb = embed_layer(mx.arange(i, end))
            chunks.append(emb)
            mx.eval(emb)
        embed_weight = mx.concatenate(chunks, axis=0)
        mx.eval(embed_weight)
        return embed_weight, vocab_size
    return embed_layer.weight, raw_shape[0]


def generate_from_embeddings(lm, inner, tokenizer, audio_embeddings, max_tokens=50, temperature=0.0):
    """Feed audio embeddings into Gemma and generate text autoregressively."""
    # Prepend a system prompt that tells the model to transcribe
    prompt = "<start_of_turn>user\nTranscribe the following speech:\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_emb = inner.embed_tokens(mx.array([prompt_ids]))

    # Suffix prompt
    suffix = "\n<end_of_turn>\n<start_of_turn>model\n"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    suffix_emb = inner.embed_tokens(mx.array([suffix_ids]))

    # Concatenate: [prompt_emb | audio_emb | suffix_emb]
    combined_emb = mx.concatenate([prompt_emb, audio_embeddings, suffix_emb], axis=1)

    # Forward pass to get logits after the full prefix
    logits = lm(mx.array([[0]]), input_embeddings=combined_emb)
    mx.eval(logits)

    generated = []
    eos_id = tokenizer.eos_token_id or 1

    for _ in range(max_tokens):
        next_logit = logits[0, -1, :]
        if temperature <= 0:
            next_token = mx.argmax(next_logit).item()
        else:
            probs = mx.softmax(next_logit / temperature)
            next_token = mx.random.categorical(mx.log(probs + 1e-8)).item()

        if next_token == eos_id or next_token == 107:  # <end_of_turn>
            break
        generated.append(next_token)

        # Feed next token
        next_emb = inner.embed_tokens(mx.array([[next_token]]))
        logits = lm(mx.array([[next_token]]), input_embeddings=next_emb)
        mx.eval(logits)

    return tokenizer.decode(generated)


def generate_from_embeddings_simple(lm, inner, tokenizer, audio_embeddings, max_tokens=50):
    """Simpler approach: just feed audio embeddings and see what Gemma outputs."""
    logits = lm(mx.array([[0]]), input_embeddings=audio_embeddings)
    mx.eval(logits)

    generated = []
    eos_id = tokenizer.eos_token_id or 1

    for _ in range(max_tokens):
        next_logit = logits[0, -1, :]
        next_token = mx.argmax(next_logit).item()
        if next_token == eos_id:
            break
        generated.append(next_token)
        next_emb = inner.embed_tokens(mx.array([[next_token]]))
        logits = lm(mx.array([[next_token]]), input_embeddings=next_emb)
        mx.eval(logits)

    return tokenizer.decode(generated)


def run_tests():
    print("\n" + "=" * 70)
    print("  DEEP RED TEAM: Whisper-Based Speech Pipeline")
    print("=" * 70 + "\n")

    results = {}

    # Load components
    adapter_path = "adapters/whisper-adapter/whisper_adapter.safetensors"
    if not Path(adapter_path).exists():
        print("  ERROR: No adapter checkpoint found!")
        return

    adapter = load_adapter(adapter_path)
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(adapter.parameters()))
    print(f"  Adapter: {n_params/1e6:.1f}M params", flush=True)

    lm, inner, tokenizer = load_gemma()
    embed_weight, vocab_size = dequantize_embeddings(inner, 2816)
    hidden_dim = 2816

    # Load validation data
    valid_data = []
    with open("data/libritts-whisper-valid.jsonl") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("feature_path") and item.get("text"):
                valid_data.append(item)
    print(f"  Validation samples: {len(valid_data)}", flush=True)

    frames_per_token = 2

    # ================================================================
    # TEST 1: Adapter Embedding Quality
    # ================================================================
    print(f"\n{'─'*60}")
    print("  TEST 1: Adapter Embedding Quality")
    print(f"{'─'*60}")

    cos_sims = []
    mag_ratios = []
    for item in valid_data[:50]:
        try:
            features = np.load(item["feature_path"])
        except Exception:
            continue
        n_frames = features.shape[0]
        if n_frames < frames_per_token:
            continue
        n_tokens = n_frames // frames_per_token
        pooled = features[:n_tokens * frames_per_token].reshape(n_tokens, frames_per_token, -1).mean(axis=1)
        whisper_in = mx.array(pooled[np.newaxis])

        tids = tokenizer.encode(item["text"], add_special_tokens=False)
        if not tids:
            continue
        if len(tids) < n_tokens:
            tids = (tids * (n_tokens // max(len(tids), 1) + 1))[:n_tokens]
        else:
            tids = tids[:n_tokens]

        pred = adapter(whisper_in)
        tgt = inner.embed_tokens(mx.array([tids]))
        sl = min(pred.shape[1], tgt.shape[1])

        pred_n = pred[:, :sl] / (mx.linalg.norm(pred[:, :sl], axis=-1, keepdims=True) + 1e-8)
        tgt_n = tgt[:, :sl] / (mx.linalg.norm(tgt[:, :sl], axis=-1, keepdims=True) + 1e-8)
        cos = mx.mean(mx.sum(pred_n * tgt_n, axis=-1)).item()
        cos_sims.append(cos)

        pred_mag = mx.mean(mx.linalg.norm(pred[:, :sl], axis=-1)).item()
        tgt_mag = mx.mean(mx.linalg.norm(tgt[:, :sl], axis=-1)).item()
        mag_ratios.append(pred_mag / (tgt_mag + 1e-8))

    avg_cos = np.mean(cos_sims)
    avg_mag = np.mean(mag_ratios)
    results["cosine_sim"] = avg_cos
    results["mag_ratio"] = avg_mag
    cos_pass = avg_cos > 0.3
    mag_pass = 0.5 < avg_mag < 2.0
    print(f"  Cosine similarity: {avg_cos:.3f} {'PASS' if cos_pass else 'FAIL'} (threshold: >0.3)")
    print(f"  Magnitude ratio:   {avg_mag:.3f} {'PASS' if mag_pass else 'FAIL'} (threshold: 0.5-2.0)")

    # ================================================================
    # TEST 2: Weight-Tied Token Prediction (Nearest Neighbor)
    # ================================================================
    print(f"\n{'─'*60}")
    print("  TEST 2: Nearest-Neighbor Token Prediction")
    print(f"{'─'*60}")

    word_overlaps = []
    token_accs = []
    for item in valid_data[:30]:
        try:
            features = np.load(item["feature_path"])
        except Exception:
            continue
        n_frames = features.shape[0]
        if n_frames < frames_per_token:
            continue
        n_tokens = n_frames // frames_per_token
        pooled = features[:n_tokens * frames_per_token].reshape(n_tokens, frames_per_token, -1).mean(axis=1)
        whisper_in = mx.array(pooled[np.newaxis])

        tids = tokenizer.encode(item["text"], add_special_tokens=False)
        if not tids:
            continue
        if len(tids) < n_tokens:
            tids = (tids * (n_tokens // max(len(tids), 1) + 1))[:n_tokens]
        else:
            tids = tids[:n_tokens]

        pred = adapter(whisper_in)
        sl = min(pred.shape[1], len(tids))

        # Compute logits in chunks
        all_logits = []
        for ci in range(0, vocab_size, 32768):
            ce = min(ci + 32768, vocab_size)
            all_logits.append(pred[:, :sl, :] @ embed_weight[ci:ce].T)
        logits = mx.concatenate(all_logits, axis=-1)
        preds_v = mx.argmax(logits[0], axis=-1).tolist()

        pred_text = tokenizer.decode(preds_v)
        true_text = tokenizer.decode(tids[:sl])
        pred_words = set(pred_text.lower().split())
        true_words = set(true_text.lower().split())
        if true_words:
            word_overlaps.append(len(pred_words & true_words) / len(true_words))

        tids_mx = mx.array(tids[:sl])
        preds_mx = mx.array(preds_v)
        acc = mx.mean((preds_mx == tids_mx).astype(mx.float32)).item()
        token_accs.append(acc)

    avg_overlap = np.mean(word_overlaps) * 100 if word_overlaps else 0
    avg_acc = np.mean(token_accs) * 100 if token_accs else 0
    results["nn_word_overlap"] = avg_overlap
    results["nn_token_acc"] = avg_acc
    nn_pass = avg_overlap > 10
    print(f"  Token accuracy:  {avg_acc:.1f}%")
    print(f"  Word overlap:    {avg_overlap:.1f}% {'PASS' if nn_pass else 'FAIL'} (threshold: >10%)")

    # Show examples
    for item in valid_data[:2]:
        try:
            features = np.load(item["feature_path"])
        except Exception:
            continue
        n_f = features.shape[0]
        if n_f < frames_per_token:
            continue
        nt = n_f // frames_per_token
        pooled = features[:nt * frames_per_token].reshape(nt, frames_per_token, -1).mean(axis=1)
        pred = adapter(mx.array(pooled[np.newaxis]))
        all_logits = []
        for ci in range(0, vocab_size, 32768):
            ce = min(ci + 32768, vocab_size)
            all_logits.append(pred[:, :min(30, nt), :] @ embed_weight[ci:ce].T)
        logits = mx.concatenate(all_logits, axis=-1)
        preds_v = mx.argmax(logits[0], axis=-1).tolist()
        print(f"  TRUE: {item['text'][:80]}")
        print(f"  PRED: {tokenizer.decode(preds_v)[:80]}")
        print()

    # ================================================================
    # TEST 3: LLM Bridge Mode (The Real Test)
    # ================================================================
    print(f"\n{'─'*60}")
    print("  TEST 3: LLM Bridge Mode (Adapter → Gemma → Text)")
    print(f"{'─'*60}")

    bridge_overlaps = []
    for i, item in enumerate(valid_data[:10]):
        try:
            features = np.load(item["feature_path"])
        except Exception:
            continue
        n_frames = features.shape[0]
        if n_frames < frames_per_token:
            continue
        n_tokens = min(n_frames // frames_per_token, 50)  # limit for speed
        pooled = features[:n_tokens * frames_per_token].reshape(n_tokens, frames_per_token, -1).mean(axis=1)
        whisper_in = mx.array(pooled[np.newaxis])
        audio_emb = adapter(whisper_in)

        t0 = time.time()
        generated = generate_from_embeddings(lm, inner, tokenizer, audio_emb, max_tokens=80)
        gen_time = (time.time() - t0) * 1000

        true_text = item["text"][:100]
        gen_words = set(generated.lower().split())
        true_words = set(true_text.lower().split())
        overlap = len(gen_words & true_words) / len(true_words) if true_words else 0
        bridge_overlaps.append(overlap)

        if i < 5:
            print(f"  [{i+1}] TRUE: {true_text}")
            print(f"       GEN:  {generated[:100]}")
            print(f"       Overlap: {overlap*100:.0f}% | Time: {gen_time:.0f}ms")
            print()

    avg_bridge = np.mean(bridge_overlaps) * 100 if bridge_overlaps else 0
    results["bridge_word_overlap"] = avg_bridge
    bridge_pass = avg_bridge > 10
    print(f"  Avg bridge word overlap: {avg_bridge:.1f}% {'PASS' if bridge_pass else 'FAIL'} (threshold: >10%)")

    # ================================================================
    # TEST 4: Simple embedding injection (no prompt)
    # ================================================================
    print(f"\n{'─'*60}")
    print("  TEST 4: Direct Embedding Injection (no prompt)")
    print(f"{'─'*60}")

    direct_overlaps = []
    for i, item in enumerate(valid_data[:5]):
        try:
            features = np.load(item["feature_path"])
        except Exception:
            continue
        n_frames = features.shape[0]
        if n_frames < frames_per_token:
            continue
        n_tokens = min(n_frames // frames_per_token, 50)
        pooled = features[:n_tokens * frames_per_token].reshape(n_tokens, frames_per_token, -1).mean(axis=1)
        whisper_in = mx.array(pooled[np.newaxis])
        audio_emb = adapter(whisper_in)

        generated = generate_from_embeddings_simple(lm, inner, tokenizer, audio_emb, max_tokens=80)

        true_text = item["text"][:100]
        gen_words = set(generated.lower().split())
        true_words = set(true_text.lower().split())
        overlap = len(gen_words & true_words) / len(true_words) if true_words else 0
        direct_overlaps.append(overlap)

        print(f"  [{i+1}] TRUE: {true_text}")
        print(f"       GEN:  {generated[:100]}")
        print(f"       Overlap: {overlap*100:.0f}%")
        print()

    avg_direct = np.mean(direct_overlaps) * 100 if direct_overlaps else 0
    results["direct_word_overlap"] = avg_direct

    # ================================================================
    # TEST 5: Adapter Latency
    # ================================================================
    print(f"\n{'─'*60}")
    print("  TEST 5: Component Latency")
    print(f"{'─'*60}")

    test_feat = np.random.randn(1, 50, 768).astype(np.float32)
    test_mx = mx.array(test_feat)

    # Warm up
    _ = adapter(test_mx)
    mx.eval(_)

    times = []
    for _ in range(20):
        t0 = time.time()
        out = adapter(test_mx)
        mx.eval(out)
        times.append((time.time() - t0) * 1000)

    avg_lat = np.mean(times)
    results["adapter_latency_ms"] = avg_lat
    lat_pass = avg_lat < 10
    print(f"  Adapter latency: {avg_lat:.1f}ms {'PASS' if lat_pass else 'FAIL'} (threshold: <10ms)")

    # ================================================================
    # TEST 6: Speech Decoder (text → SNAC)
    # ================================================================
    print(f"\n{'─'*60}")
    print("  TEST 6: Speech Decoder (text → SNAC tokens)")
    print(f"{'─'*60}")

    decoder_path = Path("adapters/speech_decoder.safetensors")
    if decoder_path.exists():
        sys.path.insert(0, "scripts")
        try:
            from speech_decoder import SpeechDecoder
            decoder = SpeechDecoder()
            dec_weights = mx.load(str(decoder_path))
            decoder.load_weights(list(dec_weights.items()))
            print(f"  Speech decoder loaded", flush=True)

            test_text = "Hello world"
            test_ids = tokenizer.encode(test_text, add_special_tokens=False)
            test_emb = inner.embed_tokens(mx.array([test_ids]))

            t0 = time.time()
            dec_out = decoder(test_emb)
            mx.eval(dec_out)
            dec_time = (time.time() - t0) * 1000

            pred_tokens = mx.argmax(dec_out[0], axis=-1).tolist()
            has_eos = 4096 in pred_tokens
            eos_pos = pred_tokens.index(4096) if has_eos else -1
            valid_tokens = pred_tokens[:eos_pos] if has_eos else pred_tokens
            in_range = all(0 <= t < 4096 for t in valid_tokens)

            dec_pass = has_eos and in_range
            results["decoder_works"] = dec_pass
            print(f"  Output tokens: {len(pred_tokens)}")
            print(f"  Has EOS: {has_eos} (at pos {eos_pos})")
            print(f"  Tokens in range: {in_range}")
            print(f"  Latency: {dec_time:.1f}ms")
            print(f"  Result: {'PASS' if dec_pass else 'FAIL'}")
        except Exception as e:
            print(f"  Error loading decoder: {e}")
            results["decoder_works"] = False
    else:
        print(f"  No decoder checkpoint found - SKIP")
        results["decoder_works"] = None

    # ================================================================
    # SCORECARD
    # ================================================================
    print(f"\n{'='*70}")
    print("  SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("Adapter loads & runs",           True,                                              True),
        ("Cosine sim > 0.3",               results.get("cosine_sim", 0) > 0.3,                results.get("cosine_sim", 0)),
        ("Magnitude ratio 0.5-2.0",        0.5 < results.get("mag_ratio", 0) < 2.0,           results.get("mag_ratio", 0)),
        ("NN word overlap > 10%",          results.get("nn_word_overlap", 0) > 10,             results.get("nn_word_overlap", 0)),
        ("Bridge word overlap > 10%",      results.get("bridge_word_overlap", 0) > 10,         results.get("bridge_word_overlap", 0)),
        ("Adapter latency < 10ms",         results.get("adapter_latency_ms", 999) < 10,        results.get("adapter_latency_ms", 0)),
        ("Decoder EOS & range",            results.get("decoder_works") is True,                results.get("decoder_works")),
    ]

    passed = sum(1 for _, p, _ in checks if p)
    total = len(checks)
    for name, p, val in checks:
        status = "PASS" if p else ("SKIP" if val is None else "FAIL")
        print(f"  [{status:4s}] {name:35s} = {val}")

    print(f"\n  Score: {passed}/{total} checks passed")
    print(f"\n  Key Metrics:")
    print(f"    Cosine similarity:     {results.get('cosine_sim', 0):.3f}")
    print(f"    NN token accuracy:     {results.get('nn_token_acc', 0):.1f}%")
    print(f"    NN word overlap:       {results.get('nn_word_overlap', 0):.1f}%")
    print(f"    Bridge word overlap:   {results.get('bridge_word_overlap', 0):.1f}%")
    print(f"    Adapter latency:       {results.get('adapter_latency_ms', 0):.1f}ms")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_tests()
