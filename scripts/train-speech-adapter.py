#!/usr/bin/env python3
"""
Speech adapter training for Freeze-Omni architecture on MLX.

Stage 1: Embedding Alignment -- train the speech encoder so that its output
embeddings match the LLM's text token embeddings for the corresponding
transcript. The LLM backbone stays frozen (used only to produce target
embeddings, no gradients flow through it).

The loss is a combination of:
    - MSE between encoder output and target text embeddings
    - Cosine similarity (to align direction, not just magnitude)

This approach:
    - Works with quantized (4-bit) models (no backprop through LLM needed)
    - Is well-established (used by LLaVA, speech adapters, etc.)
    - Produces embeddings the LLM can directly consume for generation

Usage:
    python3 scripts/train-speech-adapter.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --data data/libritts-train.jsonl \\
        --valid-data data/libritts-valid.jsonl \\
        --iters 2000 --lr 3e-4

    # Quick validation run
    python3 scripts/train-speech-adapter.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --data data/libritts-train.jsonl \\
        --iters 10
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000


def load_audio(path: str, max_duration_s: float = 10.0) -> np.ndarray:
    """Load audio file as float32, resample to 24kHz mono if needed."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / sr)
        x_old = np.linspace(0, 1, len(audio))
        x_new = np.linspace(0, 1, n_out)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    max_samples = int(max_duration_s * SAMPLE_RATE)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset with audio_path + text fields."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("audio_path") and item.get("text"):
                    data.append(item)
    return data


class FrozenGemmaEmbeddings:
    """Loads a frozen Gemma model and extracts target text embeddings.

    Used to produce the target embeddings that the speech encoder
    must learn to match. No gradients flow through this model.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._inner = None
        self.tokenizer = None
        self.hidden_dim = 0
        self.vocab_size = 0

    def load(self):
        from mlx_lm import load as lm_load
        print(f"  Loading frozen LLM: {self.model_name}", flush=True)
        t0 = time.time()
        model, self.tokenizer = lm_load(self.model_name)
        model.freeze()
        elapsed = time.time() - t0

        self._inner = self._find_text_model(model)
        probe = self._inner.embed_tokens(mx.array([[0]]))
        self.hidden_dim = probe.shape[-1]
        self.vocab_size = self._inner.embed_tokens.weight.shape[0]

        n_layers = len(self._inner.layers)
        try:
            mem_gb = mx.metal.get_active_memory() / 1e9
        except Exception:
            mem_gb = 0

        print(f"  Loaded in {elapsed:.1f}s: {n_layers} layers, "
              f"hidden={self.hidden_dim}, vocab={self.vocab_size}, {mem_gb:.1f}GB",
              flush=True)
        return self.hidden_dim

    @staticmethod
    def _find_text_model(model):
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            return model.language_model.model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        if hasattr(model, "layers"):
            return model
        raise AttributeError("Cannot find text model with embed_tokens + layers")

    def embed(self, token_ids: mx.array) -> mx.array:
        """Get text embeddings for token IDs (detached from gradient graph)."""
        emb = self._inner.embed_tokens(token_ids)
        return mx.stop_gradient(emb)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def compute_alignment_loss(predicted: mx.array, target: mx.array, cosine_weight: float = 0.5):
    """Combined MSE + cosine similarity loss for embedding alignment."""
    mse = mx.mean((predicted - target) ** 2)

    pred_norm = predicted / (mx.linalg.norm(predicted, axis=-1, keepdims=True) + 1e-8)
    tgt_norm = target / (mx.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    cosine = 1.0 - mx.mean(mx.sum(pred_norm * tgt_norm, axis=-1))

    return (1.0 - cosine_weight) * mse + cosine_weight * cosine


def encode_audio_chunks(encoder, audio_np: np.ndarray, chunk_samples: int) -> mx.array:
    """Encode audio into LLM-space embeddings, one chunk at a time."""
    n_chunks = max(1, len(audio_np) // chunk_samples)
    audio_np = audio_np[:n_chunks * chunk_samples]

    all_emb = []
    for c in range(n_chunks):
        chunk = audio_np[c * chunk_samples:(c + 1) * chunk_samples]
        chunk_input = mx.array(chunk.reshape(1, 1, chunk_samples))
        emb = encoder(chunk_input)
        all_emb.append(emb)
    return mx.concatenate(all_emb, axis=1)


def train_stage1(args):
    """Stage 1: Embedding alignment training.

    The encoder learns to produce embeddings that match the frozen LLM's
    text token embeddings for the corresponding transcript.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from speech_encoder import SpeechEncoder

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 1: Speech Encoder Embedding Alignment", flush=True)
    print(f"{'='*60}", flush=True)

    gemma = FrozenGemmaEmbeddings(args.model)
    hidden_dim = gemma.load()

    encoder = SpeechEncoder(
        llm_dim=hidden_dim,
        encoder_dim=args.encoder_dim,
        n_heads=args.encoder_heads,
        n_layers=args.encoder_layers,
        d_ff=args.encoder_dim * 4,
        tokens_per_chunk=args.tokens_per_chunk,
        chunk_ms=160,
        dropout=0.1,
    )
    n_params = encoder.num_params()
    print(f"  Encoder: {n_params/1e6:.1f}M params "
          f"(dim={args.encoder_dim}, heads={args.encoder_heads}, "
          f"layers={args.encoder_layers}, tokens/chunk={args.tokens_per_chunk})", flush=True)

    print(f"  Loading data: {args.data}", flush=True)
    train_data = load_dataset(args.data)
    valid_data = load_dataset(args.valid_data) if args.valid_data else train_data[:100]
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}", flush=True)

    output_dir = Path(args.output_dir) / "speech-encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = max(1, min(args.warmup_steps, args.iters // 5))
    lr_schedule = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps), end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, warmup_steps)
    schedule = optim.join_schedules([warmup, lr_schedule], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    chunk_samples = encoder.chunk_samples
    cosine_weight = args.cosine_weight

    print(f"\n  Config:", flush=True)
    print(f"    Iters: {args.iters}, LR: {args.lr}", flush=True)
    print(f"    Chunk: {chunk_samples} samples ({encoder.chunk_ms}ms @ 24kHz)", flush=True)
    print(f"    Loss: MSE ({1-cosine_weight:.0%}) + Cosine ({cosine_weight:.0%})", flush=True)
    print(f"    Output: {output_dir}", flush=True)
    print(f"\n  Training...\n", flush=True)

    total_loss = 0.0
    report_count = 0
    t0 = time.time()

    for step in range(1, args.iters + 1):
        idx = np.random.randint(0, len(train_data))
        item = train_data[idx]

        try:
            audio = load_audio(item["audio_path"], max_duration_s=6.0)
        except Exception as e:
            print(f"  [skip] {item['audio_path']}: {e}", flush=True)
            continue

        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))

        n_chunks = max(1, len(audio) // chunk_samples)
        audio = audio[:n_chunks * chunk_samples]

        target_ids = gemma.tokenize(item["text"])
        if not target_ids:
            continue

        n_encoder_tokens = n_chunks * args.tokens_per_chunk
        if len(target_ids) < n_encoder_tokens:
            target_ids = (target_ids * (n_encoder_tokens // max(len(target_ids), 1) + 1))[:n_encoder_tokens]
        else:
            target_ids = target_ids[:n_encoder_tokens]

        target_emb = gemma.embed(mx.array([target_ids], dtype=mx.int32))

        def loss_step(encoder):
            predicted = encode_audio_chunks(encoder, audio, chunk_samples)
            seq_len = min(predicted.shape[1], target_emb.shape[1])
            return compute_alignment_loss(
                predicted[:, :seq_len, :],
                target_emb[:, :seq_len, :],
                cosine_weight=cosine_weight,
            )

        loss, grads = mx.value_and_grad(loss_step)(encoder)
        optimizer.update(encoder, grads)
        mx.eval(encoder.parameters(), optimizer.state)

        total_loss += loss.item()
        report_count += 1

        if step % args.report_every == 0:
            avg = total_loss / max(report_count, 1)
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            try:
                mem_gb = mx.metal.get_active_memory() / 1e9
            except Exception:
                mem_gb = 0
            lr_now = schedule(step).item() if callable(schedule) else args.lr
            print(
                f"  Step {step:>5}/{args.iters}: loss={avg:.6f} | "
                f"lr={lr_now:.2e} | {sps:.1f} step/s | {mem_gb:.1f}GB",
                flush=True,
            )
            total_loss = 0.0
            report_count = 0

        if step % args.save_every == 0:
            ckpt = output_dir / f"encoder_step{step}.safetensors"
            weights = dict(mlx.utils.tree_flatten(encoder.parameters()))
            mx.save_safetensors(str(ckpt), weights)
            print(f"  Checkpoint: {ckpt}", flush=True)

    final = output_dir / "speech_encoder.safetensors"
    weights = dict(mlx.utils.tree_flatten(encoder.parameters()))
    mx.save_safetensors(str(final), weights)
    print(f"\n  Encoder saved: {final}", flush=True)

    if args.valid_data:
        print(f"  Validating...", flush=True)
        val_loss = 0.0
        val_n = 0
        for item in valid_data[:50]:
            try:
                aud = load_audio(item["audio_path"], max_duration_s=6.0)
            except Exception:
                continue
            if len(aud) < chunk_samples:
                aud = np.pad(aud, (0, chunk_samples - len(aud)))
            nc = max(1, len(aud) // chunk_samples)
            aud = aud[:nc * chunk_samples]
            tids = gemma.tokenize(item["text"])
            if not tids:
                continue
            nt = nc * args.tokens_per_chunk
            if len(tids) < nt:
                tids = (tids * (nt // max(len(tids), 1) + 1))[:nt]
            else:
                tids = tids[:nt]
            tgt = gemma.embed(mx.array([tids], dtype=mx.int32))
            pred = encode_audio_chunks(encoder, aud, chunk_samples)
            sl = min(pred.shape[1], tgt.shape[1])
            loss = compute_alignment_loss(pred[:, :sl, :], tgt[:, :sl, :],
                                          cosine_weight=cosine_weight)
            mx.eval(loss)
            val_loss += loss.item()
            val_n += 1
        if val_n:
            print(f"  Val loss: {val_loss/val_n:.6f} ({val_n} samples)", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Done in {total_elapsed/60:.1f} min. Output: {output_dir}/", flush=True)
    print(f"{'='*60}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train Freeze-Omni speech encoder for Gemma on MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit",
                        help="Frozen Gemma model (default: cached 26b-a4b)")
    parser.add_argument("--adapter-path", default=None,
                        help="LoRA adapter to load into frozen LLM")
    parser.add_argument("--data", required=True,
                        help="Training data JSONL (audio_path + text pairs)")
    parser.add_argument("--valid-data", default=None,
                        help="Validation data JSONL")
    parser.add_argument("--output-dir", default="adapters",
                        help="Output directory for encoder weights")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tokens-per-chunk", type=int, default=4)
    parser.add_argument("--cosine-weight", type=float, default=0.5,
                        help="Weight for cosine loss vs MSE (0=pure MSE, 1=pure cosine)")
    args = parser.parse_args()

    train_stage1(args)


if __name__ == "__main__":
    main()
