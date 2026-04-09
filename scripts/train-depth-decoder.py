#!/usr/bin/env python3
"""
Depth Decoder Training: cb0 → cb1, cb2 for full SNAC multi-codebook audio.

Inspired by Sesame CSM and Moshi's depth transformer:
  - Small transformer that takes cb0 tokens and generates cb1/cb2 tokens
  - SNAC has 3 codebooks at 12/24/47 Hz (ratios 1:2:4)
  - cb0 = semantic (12 Hz), cb1 = mid-detail (24 Hz), cb2 = fine (47 Hz)
  - Each codebook has 4096 entries

Architecture:
    cb0 tokens → Embedding → Transformer layers → cb1_head + cb2_head
    The depth decoder models inter-codebook dependencies per time step.
    For each cb0 frame, it predicts 2 cb1 tokens and 4 cb2 tokens.

Phase 1: Extract multi-codebook tokens from training audio
Phase 2: Train depth decoder on (cb0 → cb1, cb2) pairs
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np


# ══════════════════════════════════════════════════════════════
# Phase 1: Extract multi-codebook data
# ══════════════════════════════════════════════════════════════

def extract_codebooks(args):
    """Re-encode training audio with SNAC to get all 3 codebooks."""
    import soundfile as sf
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from codec import AudioCodec

    codec = AudioCodec("snac")
    codec.load()

    input_path = Path(args.input_data)
    output_path = Path(args.output_data)

    print(f"\n{'='*60}")
    print(f"  Phase 1: Multi-Codebook Extraction")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    items = []
    with open(input_path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("text") and item.get("audio_path"):
                items.append(item)

    print(f"  Samples: {len(items)}")

    extracted = 0
    errors = 0
    with open(output_path, "w") as out_f:
        for i, item in enumerate(items):
            audio_path = item["audio_path"]
            if not Path(audio_path).exists():
                errors += 1
                continue

            try:
                audio, sr = sf.read(audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                if sr != 24000:
                    ratio = 24000 / sr
                    n_out = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, 1, n_out),
                        np.linspace(0, 1, len(audio)),
                        audio,
                    ).astype(np.float32)

                tokens = codec.encode(audio)

                if tokens.codes.shape[1] == 0:
                    continue

                # SNAC returns 3 codebooks with different lengths
                # cb0: N frames, cb1: 2N frames, cb2: 4N frames
                raw = codec._snac_codes_raw if hasattr(codec, '_snac_codes_raw') else None
                if raw is None or len(raw) < 3:
                    continue

                cb0 = raw[0].tolist() if hasattr(raw[0], 'tolist') else list(raw[0])
                cb1 = raw[1].tolist() if hasattr(raw[1], 'tolist') else list(raw[1])
                cb2 = raw[2].tolist() if hasattr(raw[2], 'tolist') else list(raw[2])

                record = {
                    "text": item["text"],
                    "audio_path": item["audio_path"],
                    "cb0": cb0,
                    "cb1": cb1,
                    "cb2": cb2,
                    "cb0_len": len(cb0),
                    "cb1_len": len(cb1),
                    "cb2_len": len(cb2),
                }
                out_f.write(json.dumps(record) + "\n")
                extracted += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error on {audio_path}: {e}")

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(items)}: extracted={extracted}, errors={errors}", flush=True)

    print(f"\n  Done: {extracted} extracted, {errors} errors")
    print(f"  Output: {output_path}")

    # Verify
    with open(output_path) as f:
        first = json.loads(f.readline())
        print(f"  Sample: cb0={first['cb0_len']}, cb1={first['cb1_len']}, cb2={first['cb2_len']}")
        print(f"  Ratios: cb1/cb0={first['cb1_len']/max(first['cb0_len'],1):.1f}x, cb2/cb0={first['cb2_len']/max(first['cb0_len'],1):.1f}x")


# ══════════════════════════════════════════════════════════════
# Depth Decoder Architecture
# ══════════════════════════════════════════════════════════════

class DepthDecoder(nn.Module):
    """Small transformer that upsamples cb0 tokens to cb1 + cb2.

    For each cb0 frame:
      - Predicts 2 cb1 tokens (24 Hz / 12 Hz = 2x)
      - Predicts 4 cb2 tokens (47 Hz / 12 Hz ≈ 4x)

    Architecture: lightweight transformer (CSM/Moshi style depth decoder).
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 1024,
        cb1_ratio: int = 2,
        cb2_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.cb1_ratio = cb1_ratio
        self.cb2_ratio = cb2_ratio

        self.cb0_embed = nn.Embedding(codebook_size, d_model)

        self.layers = [
            _DepthBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]

        self.cb1_upsample = nn.Linear(d_model, d_model * cb1_ratio)
        self.cb2_upsample = nn.Linear(d_model, d_model * cb2_ratio)

        self.cb1_head = nn.Linear(d_model, codebook_size)
        self.cb2_head = nn.Linear(d_model, codebook_size)

        self.norm = nn.LayerNorm(d_model)

    def __call__(self, cb0_tokens: mx.array):
        """Forward pass.

        Args:
            cb0_tokens: (batch, N) cb0 token IDs

        Returns:
            cb1_logits: (batch, N*cb1_ratio, codebook_size)
            cb2_logits: (batch, N*cb2_ratio, codebook_size)
        """
        B, N = cb0_tokens.shape
        x = self.cb0_embed(cb0_tokens)  # (B, N, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Upsample for cb1: (B, N, d_model) -> (B, N, d_model*2) -> (B, N*2, d_model)
        cb1_up = self.cb1_upsample(x).reshape(B, N * self.cb1_ratio, self.d_model)
        cb1_logits = self.cb1_head(cb1_up)

        # Upsample for cb2: (B, N, d_model) -> (B, N, d_model*4) -> (B, N*4, d_model)
        cb2_up = self.cb2_upsample(x).reshape(B, N * self.cb2_ratio, self.d_model)
        cb2_logits = self.cb2_head(cb2_up)

        return cb1_logits, cb2_logits

    def generate(self, cb0_tokens: mx.array) -> tuple[mx.array, mx.array]:
        """Inference: predict cb1 and cb2 from cb0.

        Args:
            cb0_tokens: (1, N) cb0 token IDs

        Returns:
            cb1_tokens: (1, N*cb1_ratio)
            cb2_tokens: (1, N*cb2_ratio)
        """
        cb1_logits, cb2_logits = self.__call__(cb0_tokens)
        cb1 = mx.argmax(cb1_logits, axis=-1)
        cb2 = mx.argmax(cb2_logits, axis=-1)
        return cb1, cb2

    def num_params(self) -> int:
        import mlx.utils
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))


class _DepthBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x)
        x = self.drop(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + residual
        return x


# ══════════════════════════════════════════════════════════════
# Phase 2: Training
# ══════════════════════════════════════════════════════════════

def load_depth_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("cb0") and item.get("cb1") and item.get("cb2"):
                data.append(item)
    return data


def train_depth(args):
    print(f"\n{'='*60}")
    print(f"  Phase 2: Depth Decoder Training")
    print(f"{'='*60}")

    data = load_depth_data(args.data)
    np.random.shuffle(data)
    n_valid = min(200, len(data) // 10)
    valid_data = data[:n_valid]
    train_data = data[n_valid:]
    print(f"  Train: {len(train_data)}, Valid: {n_valid}")

    decoder = DepthDecoder(
        codebook_size=4096,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        cb1_ratio=2,
        cb2_ratio=4,
        dropout=0.1,
    )
    n_params = decoder.num_params()
    print(f"  Depth decoder: {n_params/1e6:.2f}M params")

    output_dir = Path(args.output_dir) / "depth-decoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = min(500, args.iters // 5)
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps), end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, max(1, warmup_steps))
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    loss_fn = nn.losses.cross_entropy
    total_loss = 0
    report_count = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.iters + 1):
        item = train_data[np.random.randint(0, len(train_data))]
        cb0 = item["cb0"]
        cb1 = item["cb1"]
        cb2 = item["cb2"]

        # Truncate to manageable size
        max_cb0 = min(len(cb0), 100)
        cb0_arr = mx.array([cb0[:max_cb0]], dtype=mx.int32)
        cb1_arr = mx.array([cb1[:max_cb0 * 2]], dtype=mx.int32)
        cb2_arr = mx.array([cb2[:max_cb0 * 4]], dtype=mx.int32)

        # Ensure exact ratios — use IGNORE_IDX (-1) for padding so loss skips them
        IGNORE_IDX = -1
        target_cb1_len = max_cb0 * 2
        target_cb2_len = max_cb0 * 4

        cb1_list = cb1[:target_cb1_len]
        cb2_list = cb2[:target_cb2_len]
        cb1_list += [IGNORE_IDX] * max(0, target_cb1_len - len(cb1_list))
        cb2_list += [IGNORE_IDX] * max(0, target_cb2_len - len(cb2_list))
        cb1_arr = mx.array([cb1_list[:target_cb1_len]], dtype=mx.int32)
        cb2_arr = mx.array([cb2_list[:target_cb2_len]], dtype=mx.int32)

        def loss_step(model):
            cb1_logits, cb2_logits = model(cb0_arr)
            cb1_flat = cb1_logits.reshape(-1, 4096)
            cb2_flat = cb2_logits.reshape(-1, 4096)
            cb1_tgt = cb1_arr.reshape(-1)
            cb2_tgt = cb2_arr.reshape(-1)

            # Mask out padded positions (IGNORE_IDX = -1)
            cb1_mask = (cb1_tgt >= 0).astype(mx.float32)
            cb2_mask = (cb2_tgt >= 0).astype(mx.float32)

            # Clamp targets to valid range for CE (masked positions don't matter)
            cb1_tgt_safe = mx.maximum(cb1_tgt, 0)
            cb2_tgt_safe = mx.maximum(cb2_tgt, 0)

            l1 = loss_fn(cb1_flat, cb1_tgt_safe, reduction="none")
            l2 = loss_fn(cb2_flat, cb2_tgt_safe, reduction="none")

            l1 = mx.sum(l1 * cb1_mask) / mx.maximum(mx.sum(cb1_mask), 1.0)
            l2 = mx.sum(l2 * cb2_mask) / mx.maximum(mx.sum(cb2_mask), 1.0)
            return l1 + l2

        loss, grads = mx.value_and_grad(loss_step)(decoder)
        optimizer.update(decoder, grads)
        mx.eval(decoder.parameters(), optimizer.state)

        total_loss += loss.item()
        report_count += 1

        if step % args.report_every == 0:
            avg = total_loss / max(report_count, 1)
            elapsed = time.time() - t0
            sps = step / elapsed
            try:
                mem_gb = mx.get_active_memory() / 1e9
            except Exception:
                mem_gb = 0
            print(f"  Step {step:>5}/{args.iters}: loss={avg:.4f} | {sps:.1f} step/s | {mem_gb:.1f}GB", flush=True)
            total_loss = 0
            report_count = 0

        if step % args.save_every == 0 or step == args.iters:
            # Validation
            val_losses = []
            for vi in range(min(50, len(valid_data))):
                v = valid_data[vi]
                vc0 = v["cb0"][:50]
                vc1 = v["cb1"]
                vc2 = v["cb2"]
                if not vc0:
                    continue
                vc0_arr = mx.array([vc0], dtype=mx.int32)
                t1 = len(vc0) * 2
                t2 = len(vc0) * 4
                # Proper padding with ignore index
                vc1_list = vc1[:t1] + [-1] * max(0, t1 - len(vc1[:t1]))
                vc2_list = vc2[:t2] + [-1] * max(0, t2 - len(vc2[:t2]))
                vc1_arr = mx.array([vc1_list[:t1]], dtype=mx.int32)
                vc2_arr = mx.array([vc2_list[:t2]], dtype=mx.int32)
                cb1_l, cb2_l = decoder(vc0_arr)
                cb1_tgt = vc1_arr.reshape(-1)
                cb2_tgt = vc2_arr.reshape(-1)
                cb1_mask = (cb1_tgt >= 0).astype(mx.float32)
                cb2_mask = (cb2_tgt >= 0).astype(mx.float32)
                l1_raw = loss_fn(cb1_l.reshape(-1, 4096), mx.maximum(cb1_tgt, 0), reduction="none")
                l2_raw = loss_fn(cb2_l.reshape(-1, 4096), mx.maximum(cb2_tgt, 0), reduction="none")
                l1 = mx.sum(l1_raw * cb1_mask) / mx.maximum(mx.sum(cb1_mask), 1.0)
                l2 = mx.sum(l2_raw * cb2_mask) / mx.maximum(mx.sum(cb2_mask), 1.0)
                mx.eval(l1, l2)
                val_losses.append(l1.item() + l2.item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            print(f"  Validation loss: {val_loss:.4f} {'(best)' if val_loss < best_val_loss else ''}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            weights = dict(mlx.utils.tree_flatten(decoder.trainable_parameters()))
            mx.savez(str(output_dir / "depth_decoder.npz"), **weights)
            flat = dict(mlx.utils.tree_flatten(decoder.parameters()))
            mx.save_safetensors(str(output_dir / "depth_decoder.safetensors"), flat)
            print(f"  Saved to {output_dir}", flush=True)

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Depth decoder: cb0 → cb1, cb2")
    sub = parser.add_subparsers(dest="command")

    # Phase 1: extract
    p1 = sub.add_parser("extract", help="Extract multi-codebook tokens from audio")
    p1.add_argument("--input-data", default="data/libritts-codec-train-full-eos.jsonl")
    p1.add_argument("--output-data", default="data/libritts-multicodebook.jsonl")

    # Phase 2: train
    p2 = sub.add_parser("train", help="Train depth decoder")
    p2.add_argument("--data", default="data/libritts-multicodebook.jsonl")
    p2.add_argument("--output-dir", default="adapters")
    p2.add_argument("--d-model", type=int, default=256)
    p2.add_argument("--n-heads", type=int, default=4)
    p2.add_argument("--n-layers", type=int, default=3)
    p2.add_argument("--lr", type=float, default=3e-4)
    p2.add_argument("--iters", type=int, default=8000)
    p2.add_argument("--report-every", type=int, default=100)
    p2.add_argument("--save-every", type=int, default=2000)

    args = parser.parse_args()

    if args.command == "extract":
        extract_codebooks(args)
    elif args.command == "train":
        train_depth(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
