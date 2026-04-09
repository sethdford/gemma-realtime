#!/usr/bin/env python3
"""
Speech adapter training for Freeze-Omni architecture on MLX.

Multi-task training with two complementary objectives:
    1. Vocabulary projection (CE loss): encoder -> vocab_head -> token_id_logits
       Strong discrete signal -- predict the exact text token for each position.
       Gradients flow cleanly through non-quantized classification head.
    2. Embedding alignment (MSE + cosine): encoder_output ≈ embed_tokens(text)
       Soft continuous signal -- match the embedding geometry.

At inference time the encoder can either:
    - Output embeddings directly (fast, uses alignment)
    - Predict token IDs -> look up perfect embeddings (bridge mode, exact)

Usage:
    python3 scripts/train-speech-adapter.py \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --data data/libritts-train.jsonl \\
        --valid-data data/libritts-valid.jsonl \\
        --iters 10000 --lr 5e-4
"""

import argparse
import json
import math
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
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("audio_path") and item.get("text"):
                    data.append(item)
    return data


class VocabProjectionHead(nn.Module):
    """Projects encoder hidden states to vocabulary logits.

    A lightweight 2-layer MLP that maps from encoder_dim to vocab_size.
    This provides the strong CE training signal that drives real alignment.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, bottleneck: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, vocab_size),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class FrozenGemmaEmbeddings:
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

        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            self._inner = model.language_model.model
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self._inner = model.model
        else:
            self._inner = model

        probe = self._inner.embed_tokens(mx.array([[0]]))
        self.hidden_dim = probe.shape[-1]
        self.vocab_size = self._inner.embed_tokens.weight.shape[0]

        try:
            mem_gb = mx.metal.get_active_memory() / 1e9
        except Exception:
            mem_gb = 0
        print(f"  Loaded in {elapsed:.1f}s: hidden={self.hidden_dim}, "
              f"vocab={self.vocab_size}, {mem_gb:.1f}GB", flush=True)
        return self.hidden_dim, self.vocab_size

    def embed(self, token_ids: mx.array) -> mx.array:
        return mx.stop_gradient(self._inner.embed_tokens(token_ids))

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


def encode_audio_chunks(encoder, audio_np: np.ndarray, chunk_samples: int) -> mx.array:
    n_chunks = max(1, len(audio_np) // chunk_samples)
    audio_np = audio_np[:n_chunks * chunk_samples]
    all_emb = []
    for c in range(n_chunks):
        chunk = audio_np[c * chunk_samples:(c + 1) * chunk_samples]
        chunk_input = mx.array(chunk.reshape(1, 1, chunk_samples))
        emb = encoder(chunk_input)
        all_emb.append(emb)
    return mx.concatenate(all_emb, axis=1)


def build_class_weights(vocab_size: int, freq_path: str = "data/token_frequencies.json",
                         alpha: float = 0.3) -> mx.array:
    """Build inverse-frequency class weights for the CE loss.

    Weight = (1 / (freq + 1))^alpha, normalized so mean = 1.0.
    Alpha controls how aggressively we upweight rare tokens.
    """
    weights = np.ones(vocab_size, dtype=np.float32)
    try:
        with open(freq_path) as f:
            data = json.load(f)
        total = data["total"]
        for tid_str, count in data["frequencies"].items():
            tid = int(tid_str)
            if 0 <= tid < vocab_size:
                weights[tid] = (total / (count + 1)) ** alpha
        weights /= weights.mean()
        print(f"  Class weights: min={weights.min():.2f}, max={weights.max():.2f}, "
              f"mean={weights.mean():.2f}", flush=True)
    except Exception as e:
        print(f"  No frequency file, using uniform weights: {e}", flush=True)
    return mx.array(weights)


def train_stage1(args):
    sys.path.insert(0, str(Path(__file__).parent))
    from speech_encoder import SpeechEncoder

    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 1: Multi-Task Encoder Training", flush=True)
    print(f"  (Focal Loss + Freq Weights + Embedding Alignment)", flush=True)
    print(f"{'='*60}", flush=True)

    gemma = FrozenGemmaEmbeddings(args.model)
    hidden_dim, vocab_size = gemma.load()

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

    vocab_head = VocabProjectionHead(hidden_dim, vocab_size, bottleneck=1024)

    # Resume from checkpoint if available
    if args.resume:
        enc_ckpt = Path(args.resume) / "speech_encoder.safetensors"
        vh_ckpt = Path(args.resume) / "vocab_head.safetensors"
        if enc_ckpt.exists():
            encoder.load_weights(list(mx.load(str(enc_ckpt)).items()))
            print(f"  Resumed encoder: {enc_ckpt}", flush=True)
        if vh_ckpt.exists():
            vocab_head.load_weights(list(mx.load(str(vh_ckpt)).items()))
            print(f"  Resumed vocab head: {vh_ckpt}", flush=True)

    class TrainableModel(nn.Module):
        def __init__(self, enc, vh):
            super().__init__()
            self.encoder = enc
            self.vocab_head = vh
    model = TrainableModel(encoder, vocab_head)

    enc_params = encoder.num_params()
    vh_params = sum(v.size for _, v in mlx.utils.tree_flatten(vocab_head.parameters()))
    print(f"  Encoder: {enc_params/1e6:.1f}M params", flush=True)
    print(f"  Vocab head: {vh_params/1e6:.1f}M params", flush=True)

    train_data = load_dataset(args.data)
    valid_data = load_dataset(args.valid_data) if args.valid_data else train_data[:100]
    print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}", flush=True)

    output_dir = Path(args.output_dir) / "speech-encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = max(1, min(args.warmup_steps, args.iters // 5))
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps), end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, warmup_steps)
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    chunk_samples = encoder.chunk_samples
    ce_weight = args.ce_weight
    align_weight = 1.0 - ce_weight

    class_weights = build_class_weights(vocab_size, alpha=args.freq_alpha)
    focal_gamma = args.focal_gamma

    print(f"\n  Config:", flush=True)
    print(f"    Iters: {args.iters}, LR: {args.lr}", flush=True)
    print(f"    Loss: FocalCE ({ce_weight:.0%}, gamma={focal_gamma}) + Align ({align_weight:.0%})", flush=True)
    print(f"    Freq alpha: {args.freq_alpha}", flush=True)
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
            audio = load_audio(item["audio_path"], max_duration_s=6.0)
        except Exception:
            continue

        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))

        n_chunks = max(1, len(audio) // chunk_samples)
        audio = audio[:n_chunks * chunk_samples]

        target_ids = gemma.tokenize(item["text"])
        if not target_ids:
            continue

        n_enc_tokens = n_chunks * args.tokens_per_chunk
        if len(target_ids) < n_enc_tokens:
            target_ids = (target_ids * (n_enc_tokens // max(len(target_ids), 1) + 1))[:n_enc_tokens]
        else:
            target_ids = target_ids[:n_enc_tokens]

        target_ids_mx = mx.array([target_ids], dtype=mx.int32)
        target_emb = gemma.embed(target_ids_mx)

        def loss_step(m):
            predicted = encode_audio_chunks(m.encoder, audio, chunk_samples)
            seq_len = min(predicted.shape[1], target_emb.shape[1])
            pred = predicted[:, :seq_len, :]
            tgt_emb = target_emb[:, :seq_len, :]
            tgt_ids = target_ids_mx[:, :seq_len]

            logits = m.vocab_head(pred)
            logits_flat = logits.reshape(-1, vocab_size)
            tgt_flat = tgt_ids.reshape(-1)

            # Focal loss with per-class frequency weights
            per_token_ce = nn.losses.cross_entropy(logits_flat, tgt_flat)

            # Per-token class weights from frequency
            w = class_weights[tgt_flat]

            # Focal modulation: downweight easy (high-prob) predictions
            log_probs = mx.softmax(logits_flat, axis=-1)
            p_t = mx.take_along_axis(log_probs, tgt_flat.reshape(-1, 1), axis=-1).squeeze(-1)
            focal_mod = (1.0 - p_t) ** focal_gamma

            ce = mx.mean(per_token_ce * w * focal_mod)

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
            return total, (ce, align, logits, tgt_ids)

        (loss, (ce_val, align_val, logits, tgt_ids)), grads = \
            mx.value_and_grad(loss_step)(model)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Track top-1 accuracy
        preds = mx.argmax(logits[0], axis=-1)
        acc = mx.mean((preds == tgt_ids[0]).astype(mx.float32)).item()

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
                mem_gb = mx.metal.get_active_memory() / 1e9
            except Exception:
                mem_gb = 0
            lr_now = schedule(step).item() if callable(schedule) else args.lr
            print(
                f"  Step {step:>5}/{args.iters}: "
                f"loss={total_loss/n:.4f} ce={total_ce/n:.2f} "
                f"align={total_align/n:.4f} acc={total_acc/n*100:.1f}% | "
                f"lr={lr_now:.2e} | {sps:.1f} step/s | {mem_gb:.1f}GB",
                flush=True,
            )
            total_loss = total_ce = total_align = total_acc = 0.0
            report_count = 0

        if step % args.save_every == 0:
            ckpt_enc = output_dir / f"encoder_step{step}.safetensors"
            ckpt_vh = output_dir / f"vocab_head_step{step}.safetensors"
            mx.save_safetensors(str(ckpt_enc), dict(mlx.utils.tree_flatten(model.encoder.parameters())))
            mx.save_safetensors(str(ckpt_vh), dict(mlx.utils.tree_flatten(model.vocab_head.parameters())))
            print(f"  Checkpoint: {ckpt_enc}", flush=True)

    mx.save_safetensors(
        str(output_dir / "speech_encoder.safetensors"),
        dict(mlx.utils.tree_flatten(model.encoder.parameters())),
    )
    mx.save_safetensors(
        str(output_dir / "vocab_head.safetensors"),
        dict(mlx.utils.tree_flatten(model.vocab_head.parameters())),
    )
    print(f"\n  Saved: encoder + vocab_head to {output_dir}/", flush=True)

    # Validation
    if args.valid_data:
        print(f"  Validating...", flush=True)
        val_ce = val_align = val_acc = 0.0
        val_cos = 0.0
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
            pred = encode_audio_chunks(model.encoder, aud, chunk_samples)
            sl = min(pred.shape[1], tgt.shape[1])

            logits = model.vocab_head(pred[:, :sl, :])
            tgt_ids = mx.array([tids[:sl]], dtype=mx.int32)
            ce = mx.mean(nn.losses.cross_entropy(logits.reshape(-1, vocab_size), tgt_ids.reshape(-1)))

            pred_n = pred[:, :sl] / (mx.linalg.norm(pred[:, :sl], axis=-1, keepdims=True) + 1e-8)
            tgt_n = tgt[:, :sl] / (mx.linalg.norm(tgt[:, :sl], axis=-1, keepdims=True) + 1e-8)
            cos = mx.mean(mx.sum(pred_n * tgt_n, axis=-1)).item()

            preds_v = mx.argmax(logits[0], axis=-1)
            acc = mx.mean((preds_v == tgt_ids[0]).astype(mx.float32)).item()

            mx.eval(ce)
            val_ce += ce.item()
            val_cos += cos
            val_acc += acc
            val_n += 1

        if val_n:
            print(f"  Val CE: {val_ce/val_n:.2f} | "
                  f"Cosine: {val_cos/val_n:.3f} | "
                  f"Token acc: {val_acc/val_n*100:.1f}%", flush=True)

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"  Done in {total_elapsed/60:.1f} min", flush=True)
    print(f"{'='*60}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train speech encoder for Gemma")
    parser.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    parser.add_argument("--data", required=True)
    parser.add_argument("--valid-data", default=None)
    parser.add_argument("--output-dir", default="adapters")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--tokens-per-chunk", type=int, default=4)
    parser.add_argument("--ce-weight", type=float, default=0.7,
                        help="Weight for CE loss (rest goes to alignment)")
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0=standard CE, 2=strong focal)")
    parser.add_argument("--freq-alpha", type=float, default=0.3,
                        help="Inverse frequency weight exponent (0=uniform, 1=full inverse)")
    args = parser.parse_args()
    train_stage1(args)


if __name__ == "__main__":
    main()
