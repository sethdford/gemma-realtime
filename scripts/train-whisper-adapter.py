#!/usr/bin/env python3
"""
Train a lightweight adapter from Whisper encoder features to Gemma embeddings.

Whisper's pre-trained encoder (680K hours) provides rich speech representations.
We only need to learn a simple projection: whisper_features(768d) -> gemma_embeddings(2816d).

Key insight: Use WEIGHT-TIED LOGITS for the CE loss. Instead of a trainable vocab
head (271M params that OOMs), we compute logits = adapter_output @ embed_weight.T
using Gemma's frozen embedding matrix. This gives us:
  - Zero extra trainable params for token prediction
  - The adapter is directly forced to produce embeddings in Gemma's exact space
  - Massive memory savings (only 3.7M trainable params total)

Multi-task loss:
    1. Weight-tied CE: adapter_output @ embed.T -> token_logits
    2. Embedding alignment: adapter_output ≈ embed_tokens(text)

Usage:
    python3 scripts/train-whisper-adapter.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --data data/libritts-whisper-train.jsonl \\
        --valid-data data/libritts-whisper-valid.jsonl \\
        --iters 20000 --lr 3e-4
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np


class WhisperAdapter(nn.Module):
    """Projects Whisper encoder features to Gemma's embedding space."""

    def __init__(self, whisper_dim: int = 768, llm_dim: int = 2816, hidden_dim: int = 2048):
        super().__init__()
        self.proj1 = nn.Linear(whisper_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj3 = nn.Linear(hidden_dim, llm_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(llm_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu(self.proj1(x))
        h = h + nn.gelu(self.proj2(self.norm1(h)))  # residual
        return self.norm2(self.proj3(h))


class FrozenGemmaEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._inner = None
        self.tokenizer = None
        self.hidden_dim = 0
        self.vocab_size = 0
        self.embed_weight = None  # frozen embedding matrix for weight-tied logits

    def load(self):
        from mlx_lm import load as lm_load
        print(f"  Loading frozen LLM: {self.model_name}", flush=True)
        t0 = time.time()
        model, self.tokenizer = lm_load(self.model_name)
        model.freeze()

        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            self._inner = model.language_model.model
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self._inner = model.model
        else:
            self._inner = model

        probe = self._inner.embed_tokens(mx.array([[0]]))
        self.hidden_dim = probe.shape[-1]

        # Dequantize embedding: the 4-bit packed weight can't be used for matmul,
        # so we reconstruct the full float matrix via lookup in chunks
        embed_layer = self._inner.embed_tokens
        raw_shape = embed_layer.weight.shape
        if hasattr(embed_layer, "scales") or raw_shape[-1] != self.hidden_dim:
            # Quantized embedding — dequantize via lookup
            self.vocab_size = embed_layer.num_embeddings if hasattr(embed_layer, "num_embeddings") else raw_shape[0]
            # Infer vocab_size from scales if available
            if hasattr(embed_layer, "scales"):
                self.vocab_size = embed_layer.scales.shape[0]
            print(f"  Dequantizing embedding ({self.vocab_size} x {self.hidden_dim})...", flush=True)
            chunks = []
            chunk_sz = 4096
            for i in range(0, self.vocab_size, chunk_sz):
                end = min(i + chunk_sz, self.vocab_size)
                ids = mx.arange(i, end)
                emb = embed_layer(ids)
                chunks.append(emb)
                mx.eval(emb)
            self.embed_weight = mx.stop_gradient(mx.concatenate(chunks, axis=0))
            mx.eval(self.embed_weight)
        else:
            self.vocab_size = raw_shape[0]
            self.embed_weight = mx.stop_gradient(embed_layer.weight)

        print(f"  Loaded in {time.time()-t0:.1f}s: hidden={self.hidden_dim}, vocab={self.vocab_size}", flush=True)
        print(f"  Embed weight: {self.embed_weight.shape}, {self.embed_weight.dtype}", flush=True)
        return self.hidden_dim, self.vocab_size

    def embed(self, token_ids: mx.array) -> mx.array:
        return mx.stop_gradient(self._inner.embed_tokens(token_ids))

    def tied_logits(self, adapter_output: mx.array) -> mx.array:
        """Compute logits via dot product with frozen embedding matrix."""
        return adapter_output @ self.embed_weight.T

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def load_whisper_dataset(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("feature_path") and item.get("text"):
                    data.append(item)
    return data


def train(args):
    print(f"\n{'='*60}", flush=True)
    print(f"  Whisper Adapter Training (Weight-Tied)", flush=True)
    print(f"  (Pre-trained features → Gemma embeddings)", flush=True)
    print(f"{'='*60}", flush=True)

    gemma = FrozenGemmaEmbeddings(args.model)
    hidden_dim, vocab_size = gemma.load()

    adapter = WhisperAdapter(
        whisper_dim=args.whisper_dim,
        llm_dim=hidden_dim,
        hidden_dim=args.adapter_hidden,
    )

    n_adapter = sum(v.size for _, v in mlx.utils.tree_flatten(adapter.parameters()))
    print(f"  Adapter: {n_adapter/1e6:.1f}M trainable params (weight-tied, no vocab head)", flush=True)

    train_data = load_whisper_dataset(args.data)
    valid_data = load_whisper_dataset(args.valid_data) if args.valid_data else train_data[:200]
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}", flush=True)

    output_dir = Path(args.output_dir) / "whisper-adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = max(1, min(args.warmup_steps, args.iters // 5))
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps), end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, warmup_steps)
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    ce_weight = args.ce_weight
    align_weight = 1.0 - ce_weight
    frames_per_token = 2

    print(f"\n  Config:", flush=True)
    print(f"    Iters: {args.iters}, LR: {args.lr}", flush=True)
    print(f"    Loss: CE ({ce_weight:.0%}) + Align ({align_weight:.0%})", flush=True)
    print(f"    Frames/token: {frames_per_token}", flush=True)
    print(f"    Output: {output_dir}\n", flush=True)

    total_loss = 0.0
    total_ce = 0.0
    total_align = 0.0
    total_acc = 0.0
    report_count = 0
    t0 = time.time()

    for step in range(1, args.iters + 1):
        idx = np.random.randint(0, len(train_data))
        item = train_data[idx]

        try:
            features = np.load(item["feature_path"])
        except Exception:
            continue

        n_frames = features.shape[0]
        if n_frames < frames_per_token:
            continue

        n_tokens = n_frames // frames_per_token
        pooled = features[:n_tokens * frames_per_token].reshape(n_tokens, frames_per_token, -1).mean(axis=1)
        whisper_input = mx.array(pooled[np.newaxis])

        target_ids = gemma.tokenize(item["text"])
        if not target_ids:
            continue

        if len(target_ids) < n_tokens:
            target_ids = (target_ids * (n_tokens // max(len(target_ids), 1) + 1))[:n_tokens]
        else:
            target_ids = target_ids[:n_tokens]

        target_ids_mx = mx.array([target_ids], dtype=mx.int32)
        target_emb = gemma.embed(target_ids_mx)
        embed_weight = gemma.embed_weight

        # Sampled softmax: target tokens + K random negatives
        n_neg = 8192
        unique_tids = list(set(target_ids))
        neg_ids = np.random.randint(0, vocab_size, size=n_neg).tolist()
        sampled_ids = unique_tids + [n for n in neg_ids if n not in set(unique_tids)]
        sampled_ids_np = np.array(sampled_ids, dtype=np.int32)
        sampled_embeds = embed_weight[mx.array(sampled_ids_np)]
        tid_to_sampled = {t: i for i, t in enumerate(sampled_ids)}
        remapped_ids = [tid_to_sampled[t] for t in target_ids]
        remapped_mx = mx.array([remapped_ids], dtype=mx.int32)

        temperature = 0.05  # sharp softmax forces fine-grained discrimination

        def loss_step(m):
            pred = m(whisper_input)
            seq_len = min(pred.shape[1], target_emb.shape[1], remapped_mx.shape[1])
            pred = pred[:, :seq_len, :]
            tgt_emb = target_emb[:, :seq_len, :]
            remap = remapped_mx[:, :seq_len]

            # Normalized similarity logits (contrastive style)
            pred_n = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
            sampled_n = sampled_embeds / (mx.linalg.norm(sampled_embeds, axis=-1, keepdims=True) + 1e-8)
            logits = (pred_n @ sampled_n.T) / temperature
            logits_flat = logits.reshape(-1, logits.shape[-1])
            tgt_flat = remap.reshape(-1)
            ce = mx.mean(nn.losses.cross_entropy(logits_flat, tgt_flat))

            # Embedding alignment
            mse = mx.mean((pred - tgt_emb) ** 2)
            pred_n = pred / (mx.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
            tgt_n = tgt_emb / (mx.linalg.norm(tgt_emb, axis=-1, keepdims=True) + 1e-8)
            cosine = 1.0 - mx.mean(mx.sum(pred_n * tgt_n, axis=-1))
            pred_mag = mx.mean(mx.linalg.norm(pred, axis=-1))
            tgt_mag = mx.mean(mx.linalg.norm(tgt_emb, axis=-1))
            mag = (pred_mag - tgt_mag) ** 2 / (tgt_mag ** 2 + 1e-8)
            align = 0.5 * mse + 0.3 * cosine + 0.2 * mag

            total = ce_weight * ce + align_weight * align
            return total, (ce, align, logits, remap)

        (loss, (ce_val, align_val, logits, remap)), grads = \
            mx.value_and_grad(loss_step)(adapter)

        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state)

        preds = mx.argmax(logits[0], axis=-1)
        acc = mx.mean((preds == remap[0]).astype(mx.float32)).item()

        total_loss += loss.item()
        total_ce += ce_val.item()
        total_align += align_val.item()
        total_acc += acc
        report_count += 1

        if step % args.report_every == 0:
            n = max(report_count, 1)
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            try:
                mem_gb = mx.get_active_memory() / 1e9
            except Exception:
                try:
                    mem_gb = mx.metal.get_active_memory() / 1e9
                except Exception:
                    mem_gb = 0
            lr_now = schedule(step).item() if callable(schedule) else args.lr
            print(
                f"  Step {step:>5}/{args.iters}: "
                f"loss={total_loss/n:.4f} ce={total_ce/n:.2f} "
                f"align={total_align/n:.4f} acc={total_acc/n*100:.1f}% | "
                f"lr={lr_now:.2e} | {sps:.0f} step/s | {mem_gb:.1f}GB",
                flush=True,
            )
            total_loss = total_ce = total_align = total_acc = 0.0
            report_count = 0

        if step % args.save_every == 0:
            mx.save_safetensors(
                str(output_dir / f"adapter_step{step}.safetensors"),
                dict(mlx.utils.tree_flatten(adapter.parameters())),
            )
            print(f"  Checkpoint: step {step}", flush=True)

    # Save final
    mx.save_safetensors(
        str(output_dir / "whisper_adapter.safetensors"),
        dict(mlx.utils.tree_flatten(adapter.parameters())),
    )
    print(f"\n  Saved to {output_dir}/", flush=True)

    # Validation
    if args.valid_data:
        print(f"  Validating on {min(100, len(valid_data))} samples...", flush=True)
        val_ce = val_cos = val_acc = 0.0
        val_n = 0
        word_overlaps = []

        for item in valid_data[:100]:
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

            tids = gemma.tokenize(item["text"])
            if not tids:
                continue
            if len(tids) < n_tokens:
                tids = (tids * (n_tokens // max(len(tids), 1) + 1))[:n_tokens]
            else:
                tids = tids[:n_tokens]

            tgt = gemma.embed(mx.array([tids], dtype=mx.int32))
            pred = adapter(whisper_in)
            sl = min(pred.shape[1], tgt.shape[1])

            # Full vocab logits in chunks to avoid OOM
            pred_sl = pred[:, :sl, :]
            tgt_ids = mx.array([tids[:sl]], dtype=mx.int32)
            chunk_size = 32768
            all_logits = []
            for ci in range(0, vocab_size, chunk_size):
                ce_end = min(ci + chunk_size, vocab_size)
                chunk_emb = gemma.embed_weight[ci:ce_end]
                all_logits.append(pred_sl @ chunk_emb.T)
            logits = mx.concatenate(all_logits, axis=-1)
            ce = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, vocab_size), tgt_ids.reshape(-1)))

            pred_n = pred[:, :sl] / (mx.linalg.norm(pred[:, :sl], axis=-1, keepdims=True) + 1e-8)
            tgt_n = tgt[:, :sl] / (mx.linalg.norm(tgt[:, :sl], axis=-1, keepdims=True) + 1e-8)
            cos = mx.mean(mx.sum(pred_n * tgt_n, axis=-1)).item()

            preds_v = mx.argmax(logits[0], axis=-1)
            acc = mx.mean((preds_v == tgt_ids[0]).astype(mx.float32)).item()

            pred_text = gemma.tokenizer.decode(preds_v.tolist())
            true_text = gemma.tokenizer.decode(tids[:sl])
            pred_words = set(pred_text.lower().split())
            true_words = set(true_text.lower().split())
            if true_words:
                word_overlaps.append(len(pred_words & true_words) / len(true_words))

            mx.eval(ce)
            val_ce += ce.item()
            val_cos += cos
            val_acc += acc
            val_n += 1

        if val_n:
            avg_overlap = np.mean(word_overlaps) * 100 if word_overlaps else 0
            print(f"\n  Validation Results ({val_n} samples):", flush=True)
            print(f"    CE Loss:      {val_ce/val_n:.2f}", flush=True)
            print(f"    Cosine Sim:   {val_cos/val_n:.3f}", flush=True)
            print(f"    Token Acc:    {val_acc/val_n*100:.1f}%", flush=True)
            print(f"    Word Overlap: {avg_overlap:.1f}%", flush=True)

            # Show example predictions
            print(f"\n  Example predictions:", flush=True)
            for item in valid_data[:3]:
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
                pred = adapter(whisper_in)
                sl = min(pred.shape[1], 30)
                pred_sl = pred[:, :sl, :]
                all_logits = []
                for ci in range(0, vocab_size, 32768):
                    ce_end = min(ci + 32768, vocab_size)
                    all_logits.append(pred_sl @ gemma.embed_weight[ci:ce_end].T)
                logits = mx.concatenate(all_logits, axis=-1)
                preds_v = mx.argmax(logits[0], axis=-1)
                pred_text = gemma.tokenizer.decode(preds_v.tolist())
                true_text = item["text"][:100]
                print(f"    TRUE: {true_text}", flush=True)
                print(f"    PRED: {pred_text[:100]}", flush=True)
                print(flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  Done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"{'='*60}\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--valid-data", default=None)
    parser.add_argument("--output-dir", default="adapters")
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--whisper-dim", type=int, default=768)
    parser.add_argument("--adapter-hidden", type=int, default=1024)
    parser.add_argument("--ce-weight", type=float, default=0.8)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
