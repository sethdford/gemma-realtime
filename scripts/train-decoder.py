#!/usr/bin/env python3
"""
Stage 2: Speech decoder training for Freeze-Omni architecture on MLX.
Stage 3: Duplex state predictor training.

Key improvements for EOS learning:
    - 50x EOS weight in main training
    - Dedicated EOS fine-tuning phase at end
    - Short sequence curriculum (EOS appears more frequently)
    - EOS accuracy tracking

Usage:
    python3 scripts/train-decoder.py stage2 \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --data data/libritts-codec-train-eos.jsonl \\
        --valid-data data/libritts-codec-valid-eos.jsonl \\
        --iters 10000 --lr 3e-4

    python3 scripts/train-decoder.py stage3 \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --data data/libritts-codec-train-eos.jsonl \\
        --iters 2000 --lr 1e-3
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


class FrozenGemmaEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._inner = None
        self.tokenizer = None
        self.hidden_dim = 0

    def load(self):
        from mlx_lm import load as lm_load
        print(f"  Loading frozen LLM: {self.model_name}", flush=True)
        t0 = time.time()
        model, self.tokenizer = lm_load(self.model_name)
        model.freeze()
        elapsed = time.time() - t0

        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            self._inner = model.language_model.model
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self._inner = model.model
        else:
            self._inner = model

        probe = self._inner.embed_tokens(mx.array([[0]]))
        self.hidden_dim = probe.shape[-1]
        print(f"  Loaded in {elapsed:.1f}s, hidden={self.hidden_dim}", flush=True)
        return self.hidden_dim

    def embed(self, token_ids: mx.array) -> mx.array:
        return mx.stop_gradient(self._inner.embed_tokens(token_ids))

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def load_codec_dataset(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("text") and item.get("codec_tokens"):
                    data.append(item)
    return data


def train_stage2(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from speech_decoder import SpeechDecoder

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 2: Speech Decoder Training (EOS-aware)", flush=True)
    print(f"{'='*60}", flush=True)

    gemma = FrozenGemmaEmbeddings(args.model)
    hidden_dim = gemma.load()

    eos_id = args.codebook_size  # 4096

    decoder = SpeechDecoder(
        llm_dim=hidden_dim,
        decoder_dim=args.decoder_dim,
        n_heads=args.decoder_heads,
        n_layers=args.decoder_layers,
        d_ff=args.decoder_dim * 4,
        codebook_size=args.codebook_size,
        max_tokens=500,
        dropout=0.1,
    )
    n_params = decoder.num_params()
    print(f"  Decoder: {n_params/1e6:.1f}M params", flush=True)

    train_data = load_codec_dataset(args.data)
    valid_data = load_codec_dataset(args.valid_data) if args.valid_data else train_data[:100]
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}", flush=True)

    # Bucket data by length for curriculum learning
    short_data = [d for d in train_data if len(d["codec_tokens"]) <= 30]
    med_data = [d for d in train_data if 30 < len(d["codec_tokens"]) <= 100]
    long_data = [d for d in train_data if len(d["codec_tokens"]) > 100]
    print(f"  Buckets: short={len(short_data)}, med={len(med_data)}, long={len(long_data)}", flush=True)

    output_dir = Path(args.output_dir) / "speech-decoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = max(1, min(args.warmup_steps, args.iters // 5))
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps), end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, warmup_steps)
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    loss_fn = nn.losses.cross_entropy
    eos_weight = args.eos_weight

    print(f"\n  Config:", flush=True)
    print(f"    Iters: {args.iters} + {args.eos_finetune_iters} EOS finetune", flush=True)
    print(f"    EOS weight: {eos_weight}x, EOS ID: {eos_id}", flush=True)
    print(f"    Output: {output_dir}\n", flush=True)

    total_loss = 0.0
    total_eos_correct = 0
    total_eos_count = 0
    report_count = 0
    t0 = time.time()

    for step in range(1, args.iters + 1):
        # Curriculum: start with short sequences (EOS appears proportionally more)
        progress = step / args.iters
        if progress < 0.3 and short_data:
            pool = short_data
        elif progress < 0.6 and (short_data or med_data):
            pool = med_data if med_data else short_data
        else:
            pool = train_data

        idx = np.random.randint(0, len(pool))
        item = pool[idx]

        text_ids = gemma.tokenize(item["text"])
        if not text_ids or len(text_ids) < 2:
            continue

        codec_tokens = item["codec_tokens"]
        if not codec_tokens or len(codec_tokens) < 2:
            continue

        max_len = min(len(codec_tokens), 200)
        codec_tokens = codec_tokens[:max_len]

        context_emb = gemma.embed(mx.array([text_ids], dtype=mx.int32))
        target = mx.array([codec_tokens], dtype=mx.int32)

        def loss_step(decoder):
            logits = decoder(context_emb, target)
            logits_flat = logits.reshape(-1, args.codebook_size + 1)
            target_flat = target.reshape(-1)
            per_token_loss = loss_fn(logits_flat, target_flat)

            eos_mask = (target_flat == eos_id).astype(mx.float32)
            non_eos_mask = 1.0 - eos_mask
            weights = non_eos_mask + eos_mask * eos_weight
            return mx.sum(per_token_loss * weights) / mx.sum(weights)

        loss, grads = mx.value_and_grad(loss_step)(decoder)
        optimizer.update(decoder, grads)
        mx.eval(decoder.parameters(), optimizer.state)

        total_loss += loss.item()
        report_count += 1

        # Track EOS accuracy (every 10 steps to avoid overhead)
        if step % 10 == 0:
            target_np = np.array(codec_tokens)
            eos_positions = np.where(target_np == eos_id)[0]
            if len(eos_positions) > 0:
                logits = decoder(context_emb, target)
                logits_flat = logits.reshape(-1, args.codebook_size + 1)
                preds = mx.argmax(logits_flat, axis=-1)
                mx.eval(preds)
                for p in eos_positions:
                    total_eos_count += 1
                    if preds[p].item() == eos_id:
                        total_eos_correct += 1

        if step % args.report_every == 0:
            avg = total_loss / max(report_count, 1)
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            eos_acc = total_eos_correct / max(total_eos_count, 1) * 100
            try:
                mem_gb = mx.metal.get_active_memory() / 1e9
            except Exception:
                mem_gb = 0
            lr_now = schedule(step).item() if callable(schedule) else args.lr
            print(
                f"  Step {step:>5}/{args.iters}: loss={avg:.4f} | "
                f"eos_acc={eos_acc:.0f}% ({total_eos_correct}/{total_eos_count}) | "
                f"lr={lr_now:.2e} | {sps:.1f} step/s | {mem_gb:.1f}GB",
                flush=True,
            )
            total_loss = 0.0
            total_eos_correct = 0
            total_eos_count = 0
            report_count = 0

        if step % args.save_every == 0:
            ckpt = output_dir / f"decoder_step{step}.safetensors"
            mx.save_safetensors(str(ckpt), dict(mlx.utils.tree_flatten(decoder.parameters())))
            print(f"  Checkpoint: {ckpt}", flush=True)

    # EOS fine-tuning phase: only short sequences, very high EOS weight
    if args.eos_finetune_iters > 0:
        print(f"\n  --- EOS Fine-tuning Phase ({args.eos_finetune_iters} iters) ---", flush=True)
        eos_ft_pool = short_data if short_data else train_data
        eos_ft_optimizer = optim.Adam(learning_rate=args.lr * 0.1)
        eos_ft_weight = eos_weight * 5  # 250x

        eos_loss_total = 0.0
        eos_correct = 0
        eos_total = 0
        for ft_step in range(1, args.eos_finetune_iters + 1):
            idx = np.random.randint(0, len(eos_ft_pool))
            item = eos_ft_pool[idx]
            text_ids = gemma.tokenize(item["text"])
            codec_tokens = item["codec_tokens"]
            if not text_ids or not codec_tokens or len(codec_tokens) < 2:
                continue

            max_len = min(len(codec_tokens), 50)
            codec_tokens = codec_tokens[:max_len]
            if codec_tokens[-1] != eos_id:
                codec_tokens = codec_tokens[:-1] + [eos_id]

            context_emb = gemma.embed(mx.array([text_ids], dtype=mx.int32))
            target = mx.array([codec_tokens], dtype=mx.int32)

            def eos_loss_step(decoder):
                logits = decoder(context_emb, target)
                logits_flat = logits.reshape(-1, args.codebook_size + 1)
                target_flat = target.reshape(-1)
                per_token_loss = loss_fn(logits_flat, target_flat)
                eos_mask = (target_flat == eos_id).astype(mx.float32)
                weights = (1.0 - eos_mask) + eos_mask * eos_ft_weight
                return mx.sum(per_token_loss * weights) / mx.sum(weights)

            loss, grads = mx.value_and_grad(eos_loss_step)(decoder)
            eos_ft_optimizer.update(decoder, grads)
            mx.eval(decoder.parameters(), eos_ft_optimizer.state)
            eos_loss_total += loss.item()

            logits = decoder(context_emb, target)
            preds = mx.argmax(logits[0], axis=-1)
            mx.eval(preds)
            if preds[-1].item() == eos_id:
                eos_correct += 1
            eos_total += 1

            if ft_step % 100 == 0:
                avg = eos_loss_total / ft_step
                acc = eos_correct / max(eos_total, 1) * 100
                print(f"    EOS-FT {ft_step}/{args.eos_finetune_iters}: "
                      f"loss={avg:.4f} eos_acc={acc:.0f}%", flush=True)

    # Save final
    final = output_dir / "speech_decoder.safetensors"
    mx.save_safetensors(str(final), dict(mlx.utils.tree_flatten(decoder.parameters())))
    print(f"\n  Decoder saved: {final}", flush=True)

    # Validation with EOS check
    if args.valid_data:
        print(f"  Validating...", flush=True)
        val_loss = 0.0
        val_eos_correct = 0
        val_n = 0
        for item in valid_data[:50]:
            tids = gemma.tokenize(item["text"])
            ct = item.get("codec_tokens", [])
            if not tids or not ct or len(ct) < 2:
                continue
            ct = ct[:200]
            ctx = gemma.embed(mx.array([tids], dtype=mx.int32))
            tgt = mx.array([ct], dtype=mx.int32)
            logits = decoder(ctx, tgt)
            loss = mx.mean(loss_fn(
                logits.reshape(-1, args.codebook_size + 1),
                tgt.reshape(-1)
            ))
            mx.eval(loss)
            val_loss += loss.item()

            preds = mx.argmax(logits[0], axis=-1)
            if ct[-1] == eos_id and preds[-1].item() == eos_id:
                val_eos_correct += 1
            val_n += 1

        if val_n:
            print(f"  Val loss: {val_loss/val_n:.4f} | "
                  f"EOS acc: {val_eos_correct/val_n*100:.0f}% ({val_n} samples)", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 2 done in {total_elapsed/60:.1f} min", flush=True)
    print(f"{'='*60}\n", flush=True)


def train_stage3(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from speech_decoder import DuplexStatePredictor

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 3: Duplex State Predictor Training", flush=True)
    print(f"{'='*60}", flush=True)

    gemma = FrozenGemmaEmbeddings(args.model)
    hidden_dim = gemma.load()

    predictor = DuplexStatePredictor(llm_dim=hidden_dim, hidden_dim=256, n_states=3)
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(predictor.parameters()))
    print(f"  Predictor: {n_params/1e3:.0f}K params", flush=True)

    train_data = load_codec_dataset(args.data)
    print(f"  Train: {len(train_data)} samples", flush=True)

    output_dir = Path(args.output_dir) / "duplex-predictor"
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_fn = nn.losses.cross_entropy

    print(f"\n  Training...\n", flush=True)

    total_loss = 0.0
    total_acc = 0.0
    report_count = 0
    t0 = time.time()

    for step in range(1, args.iters + 1):
        idx = np.random.randint(0, len(train_data))
        item = train_data[idx]

        text_ids = gemma.tokenize(item["text"])
        if not text_ids:
            continue

        ctx = gemma.embed(mx.array([text_ids], dtype=mx.int32))

        has_audio = len(item.get("codec_tokens", [])) > 5
        r = np.random.random()
        if r < 0.1:
            label = 2
        elif has_audio and r < 0.6:
            label = 1
        else:
            label = 0

        target = mx.array([label], dtype=mx.int32)

        def loss_step(predictor):
            logits = predictor(ctx)
            return mx.mean(loss_fn(logits, target))

        loss, grads = mx.value_and_grad(loss_step)(predictor)
        optimizer.update(predictor, grads)
        mx.eval(predictor.parameters(), optimizer.state)

        total_loss += loss.item()
        pred_state = mx.argmax(predictor(ctx), axis=-1).item()
        total_acc += float(pred_state == label)
        report_count += 1

        if step % args.report_every == 0:
            avg_loss = total_loss / max(report_count, 1)
            avg_acc = total_acc / max(report_count, 1) * 100
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            print(
                f"  Step {step:>5}/{args.iters}: loss={avg_loss:.4f} | "
                f"acc={avg_acc:.0f}% | {sps:.0f} step/s",
                flush=True,
            )
            total_loss = total_acc = 0.0
            report_count = 0

    final = output_dir / "duplex_predictor.safetensors"
    mx.save_safetensors(str(final), dict(mlx.utils.tree_flatten(predictor.parameters())))

    total_elapsed = time.time() - t0
    print(f"\n  Predictor saved: {final}", flush=True)
    print(f"  Done in {total_elapsed/60:.1f} min\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train speech decoder and duplex predictor")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    p2 = subparsers.add_parser("stage2", help="Train speech decoder")
    p2.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    p2.add_argument("--data", required=True)
    p2.add_argument("--valid-data", default=None)
    p2.add_argument("--output-dir", default="adapters")
    p2.add_argument("--iters", type=int, default=10000)
    p2.add_argument("--lr", type=float, default=3e-4)
    p2.add_argument("--warmup-steps", type=int, default=200)
    p2.add_argument("--report-every", type=int, default=100)
    p2.add_argument("--save-every", type=int, default=2500)
    p2.add_argument("--decoder-dim", type=int, default=512)
    p2.add_argument("--decoder-heads", type=int, default=8)
    p2.add_argument("--decoder-layers", type=int, default=4)
    p2.add_argument("--codebook-size", type=int, default=4096)
    p2.add_argument("--eos-weight", type=float, default=50.0)
    p2.add_argument("--eos-finetune-iters", type=int, default=2000,
                    help="Dedicated EOS fine-tuning steps after main training")

    p3 = subparsers.add_parser("stage3", help="Train duplex state predictor")
    p3.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    p3.add_argument("--data", required=True)
    p3.add_argument("--output-dir", default="adapters")
    p3.add_argument("--iters", type=int, default=2000)
    p3.add_argument("--lr", type=float, default=1e-3)
    p3.add_argument("--report-every", type=int, default=100)

    args = parser.parse_args()
    if args.stage == "stage2":
        train_stage2(args)
    elif args.stage == "stage3":
        train_stage3(args)


if __name__ == "__main__":
    main()
