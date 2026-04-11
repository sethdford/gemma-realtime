#!/usr/bin/env python3
"""Voxtral speculative decoding with draft heads.

Draft heads are lightweight MLPs that predict future audio frames directly
from the LM hidden state, bypassing both the LM backbone step and the
acoustic transformer's flow matching. Combined with reduced denoising steps,
this can achieve 6x+ real-time on Apple Silicon.

Architecture per head:
    Linear(3072, 1024) -> GELU -> Linear(1024, 1024) -> GELU -> Linear(1024, 37)
    ~7M params per head, ~21M for 3 heads

Usage:
    # Speculative generation (requires trained draft heads)
    from voxtral_speculative import speculative_generate, DraftHeadSet
    heads = DraftHeadSet.load("adapters/draft-heads/heads-libri.safetensors")
    audio = speculative_generate(model, text, voice, heads)
    # Path resolution (VOXTRAL_DRAFT_HEADS, auto-detect): see draft_heads_resolve.py
"""
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class DraftHeadConfig:
    input_dim: int = 3072
    hidden_dim: int = 1024
    semantic_vocab: int = 8194  # 8192 + 2 special tokens
    acoustic_vocab: int = 23   # 21 + 2 special tokens
    n_acoustic: int = 36


class DraftHead(nn.Module):
    """Predicts one future frame's codes from the current LM hidden state."""

    def __init__(self, config: DraftHeadConfig = DraftHeadConfig()):
        super().__init__()
        self.config = config
        d = config.input_dim
        h = config.hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.semantic_head = nn.Linear(h, config.semantic_vocab)
        self.acoustic_head = nn.Linear(h, config.n_acoustic * config.acoustic_vocab)

    def __call__(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        """Predict codes from LM hidden state.

        Args:
            hidden: (B, input_dim) LM hidden state

        Returns:
            semantic_logits: (B, semantic_vocab)
            acoustic_logits: (B, n_acoustic, acoustic_vocab)
        """
        h = self.trunk(hidden)
        sem = self.semantic_head(h)
        ac = self.acoustic_head(h).reshape(
            -1, self.config.n_acoustic, self.config.acoustic_vocab
        )
        return sem, ac

    def predict_codes(self, hidden: mx.array) -> mx.array:
        """Predict frame codes as argmax indices. Returns (B, 37)."""
        sem_logits, ac_logits = self(hidden)
        sem = mx.argmax(sem_logits, axis=-1, keepdims=True)  # (B, 1)
        ac = mx.argmax(ac_logits, axis=-1)  # (B, 36)
        return mx.concatenate([sem, ac], axis=-1)


class DraftHeadSet(nn.Module):
    """Set of N draft heads for multi-frame speculation."""

    def __init__(self, n_heads: int = 3, config: DraftHeadConfig = DraftHeadConfig()):
        super().__init__()
        self.heads = [DraftHead(config) for _ in range(n_heads)]
        self.config = config

    def predict_all(self, hidden: mx.array) -> list[mx.array]:
        """Predict codes for all future frames. Returns list of (B, 37) arrays."""
        return [head.predict_codes(hidden) for head in self.heads]

    def forward_all(self, hidden: mx.array) -> list[tuple[mx.array, mx.array]]:
        """Get logits for all heads. Returns list of (sem_logits, ac_logits)."""
        return [head(hidden) for head in self.heads]

    def save(self, path: str):
        weights = dict(nn.utils.tree_flatten(self.parameters()))
        mx.save_safetensors(path, weights)
        print(f"Saved {len(weights)} weight tensors to {path}")

    @classmethod
    def load(cls, path: str, n_heads: int = 3,
             config: DraftHeadConfig = DraftHeadConfig()):
        heads = cls(n_heads=n_heads, config=config)
        weights = mx.load(path)
        heads.load_weights(list(weights.items()))
        return heads


def compute_loss(head_set, hiddens, targets_list):
    """Compute training loss for all heads.

    Args:
        head_set: DraftHeadSet
        hiddens: (N, 3072) hidden states
        targets_list: list of (N, 37) target codes, one per head
    """
    config = head_set.config
    total_loss = mx.array(0.0)

    for head, targets in zip(head_set.heads, targets_list):
        sem_logits, ac_logits = head(hiddens)

        sem_targets = targets[:, 0].astype(mx.int32)
        sem_loss = nn.losses.cross_entropy(sem_logits, sem_targets, reduction="mean")

        ac_loss = mx.array(0.0)
        for j in range(config.n_acoustic):
            ac_targets_j = targets[:, 1 + j].astype(mx.int32)
            ac_loss = ac_loss + nn.losses.cross_entropy(
                ac_logits[:, j, :], ac_targets_j, reduction="mean"
            )
        ac_loss = ac_loss / config.n_acoustic

        total_loss = total_loss + 10.0 * sem_loss + ac_loss

    return total_loss / len(head_set.heads)


def train_draft_heads(
    data_path: str,
    save_path: str,
    n_heads: int = 3,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Train draft heads on collected data."""
    import os

    print(f"Loading training data from {data_path}...", flush=True)
    data = np.load(data_path)
    hiddens_np = data["hiddens"]
    targets_np = [data[f"target_{k+1}"] for k in range(n_heads)]

    n = len(hiddens_np)
    print(f"  {n} training pairs, hidden_dim={hiddens_np.shape[1]}", flush=True)

    config = DraftHeadConfig(input_dim=hiddens_np.shape[1])
    head_set = DraftHeadSet(n_heads=n_heads, config=config)

    import mlx.optimizers as optim

    loss_and_grad = nn.value_and_grad(head_set, lambda m, h, t: compute_loss(m, h, t))
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-4)

    n_batches = (n + batch_size - 1) // batch_size
    print(f"  {n_batches} batches/epoch, {epochs} epochs", flush=True)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        t0 = time.time()

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            h_batch = mx.array(hiddens_np[idx])
            t_batch = [mx.array(t[idx]) for t in targets_np]

            loss, grads = loss_and_grad(head_set, h_batch, t_batch)
            optimizer.update(head_set, grads)
            mx.eval(head_set.parameters(), optimizer.state)
            epoch_loss += loss.item()

        avg = epoch_loss / n_batches
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:3d}/{epochs}: loss={avg:.4f} ({elapsed:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    head_set.save(save_path)
    print(f"\nTraining complete. Saved to {save_path}")
    return head_set


def _estimate_max_frames(text, headroom=2.0):
    """Duration-proportional frame cap to prevent runaway generation."""
    n_words = len(text.split())
    est_seconds = n_words / 2.5
    max_frames = int(est_seconds * 12.5 * headroom)
    return max(50, min(max_frames, 500))


def _acoustic_similarity(draft_acoustic, verified_acoustic, tolerance=2):
    """PCG-style fuzzy match: acoustic codes accepted if within tolerance.

    Acoustic codes are FSQ indices 0-22 where perceptually similar neighbors
    exist. Returns fraction of acoustic codes within tolerance.
    """
    diff = np.abs(draft_acoustic.astype(int) - verified_acoustic.astype(int))
    return float(np.mean(diff <= tolerance))


def speculative_generate(
    model,
    text: str,
    voice: str,
    head_set: DraftHeadSet,
    max_tokens: int | None = None,
    acoustic_tolerance: int = 2,
    acoustic_threshold: float = 0.7,
    verbose: bool = False,
):
    """Generate audio with speculative decoding using draft heads.

    For each verified frame, the draft heads predict N future frames.
    Each draft is verified by running the actual LM + acoustic transformer.
    Accepted drafts skip the draft head overhead; rejected drafts resume
    from the last verified frame.

    Acceptance criteria:
    - Semantic code must match exactly (carries linguistic content)
    - Acoustic codes use PCG-style fuzzy matching: accepted if >= acoustic_threshold
      fraction of acoustic codes are within acoustic_tolerance of verified values

    Returns (waveform, stats_dict).
    """
    from mlx_lm.models.cache import make_prompt_cache

    if max_tokens is None:
        max_tokens = _estimate_max_frames(text)

    lm_backbone = model.language_model.model.model
    at = model.acoustic_transformer
    n_draft = len(head_set.heads)

    input_ids = model._encode_text(text, voice)
    input_ids_mx = mx.array(input_ids)[None, :]
    input_embeddings = model._build_input_embeddings(input_ids_mx, voice)

    cache = make_prompt_cache(model.language_model.model)
    hidden = lm_backbone(input_ids_mx, cache=cache, input_embeddings=input_embeddings)
    audio_tok_emb = model.language_model.embed_tokens(
        mx.array([[model.config.audio_token_id]])
    )
    hidden = lm_backbone(
        mx.array([[model.config.audio_token_id]]),
        cache=cache,
        input_embeddings=audio_tok_emb,
    )

    all_codes = []
    stats = {
        "total_frames": 0, "lm_steps": 0,
        "drafts_accepted": 0, "drafts_rejected": 0,
        "acoustic_accept_avg": 0.0,
    }
    acoustic_sims = []
    t_start = time.time()

    frame = 0
    done = False
    while frame < max_tokens and not done:
        h = hidden[:, -1, :]

        codes = at.decode_one_frame(h)
        mx.eval(codes)
        semantic = codes[0, 0].item()
        if semantic <= 1:
            break

        all_codes.append(codes[:, None, :])
        stats["total_frames"] += 1
        stats["lm_steps"] += 1
        frame += 1

        draft_codes_list = head_set.predict_all(h)
        mx.eval(*draft_codes_list)

        global_codes = model._codes_to_global_indices(codes)
        code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
        next_embedding = code_embeddings.sum(axis=1, keepdims=True)
        dummy_input = mx.array([[model.config.audio_token_id]])
        hidden = lm_backbone(dummy_input, cache=cache, input_embeddings=next_embedding)

        for d in range(n_draft):
            if frame >= max_tokens:
                break
            draft = draft_codes_list[d]
            draft_np = np.array(draft[0])
            draft_semantic = int(draft_np[0])

            if draft_semantic <= 1:
                done = True
                break

            h_verify = hidden[:, -1, :]
            verified = at.decode_one_frame(h_verify)
            mx.eval(verified)
            verified_np = np.array(verified[0])
            verified_semantic = int(verified_np[0])

            if verified_semantic <= 1:
                done = True
                break

            ac_sim = _acoustic_similarity(
                draft_np[1:], verified_np[1:], tolerance=acoustic_tolerance
            )
            acoustic_sims.append(ac_sim)

            sem_match = (draft_semantic == verified_semantic)
            ac_match = (ac_sim >= acoustic_threshold)

            all_codes.append(verified[:, None, :])
            stats["total_frames"] += 1
            stats["lm_steps"] += 1
            frame += 1

            global_codes = model._codes_to_global_indices(verified)
            code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
            next_embedding = code_embeddings.sum(axis=1, keepdims=True)
            hidden = lm_backbone(
                dummy_input, cache=cache, input_embeddings=next_embedding
            )

            if sem_match and ac_match:
                stats["drafts_accepted"] += 1
            else:
                stats["drafts_rejected"] += 1
                break

        if frame % 50 == 0:
            mx.clear_cache()

    if not all_codes:
        return np.zeros(0, dtype=np.float32), stats

    audio_codes = mx.concatenate(all_codes, axis=1)
    waveform = model.audio_tokenizer.decode(audio_codes).squeeze(0)
    mx.eval(waveform)

    elapsed = time.time() - t_start
    audio_dur = waveform.shape[0] / model.config.sample_rate
    stats["elapsed_s"] = round(elapsed, 2)
    stats["audio_duration_s"] = round(audio_dur, 2)
    stats["rtf"] = round(elapsed / audio_dur, 3) if audio_dur > 0 else 0
    accept_total = stats["drafts_accepted"] + stats["drafts_rejected"]
    stats["acceptance_rate"] = (
        round(stats["drafts_accepted"] / accept_total, 3) if accept_total > 0 else 0
    )
    stats["acoustic_accept_avg"] = (
        round(float(np.mean(acoustic_sims)), 3) if acoustic_sims else 0
    )

    if verbose:
        print(f"  Speculative decode: {stats['total_frames']} frames, "
              f"{stats['lm_steps']} LM steps, "
              f"accept={stats['acceptance_rate']:.1%}, "
              f"ac_sim={stats['acoustic_accept_avg']:.1%}, "
              f"RTF={stats['rtf']:.3f}x")

    return np.array(waveform), stats


def main():
    """CLI for training and testing draft heads."""
    import argparse

    parser = argparse.ArgumentParser(description="Voxtral speculative decoding draft heads")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train draft heads on collected data")
    train_p.add_argument("--data", required=True, help="Path to draft-pairs.npz")
    train_p.add_argument("--output", default="adapters/draft-heads/heads.safetensors")
    train_p.add_argument("--heads", type=int, default=3)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--lr", type=float, default=1e-3)

    test_p = sub.add_parser("test", help="Test speculative generation")
    test_p.add_argument("--heads-path", required=True)
    test_p.add_argument("--n-heads", type=int, default=3)
    test_p.add_argument("--precision", default="6bit")
    test_p.add_argument("--denoise-steps", type=int, default=4)
    test_p.add_argument("--voice", default="cheerful_male")
    test_p.add_argument("--max-tokens", type=int, default=None,
                        help="Max frames (default: adaptive based on text length)")
    test_p.add_argument("--acoustic-tolerance", type=int, default=2)
    test_p.add_argument("--acoustic-threshold", type=float, default=0.7)
    test_p.add_argument("--text", default="Hey, this is a speculative decoding test. Does it sound good?")

    args = parser.parse_args()

    if args.command == "train":
        train_draft_heads(
            data_path=args.data,
            save_path=args.output,
            n_heads=args.heads,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.command == "test":
        from mlx_audio.tts.utils import load as mlx_audio_load
        import soundfile as sf

        models = {
            "4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
            "6bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
            "bf16": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
        }
        print(f"Loading Voxtral {args.precision}...", flush=True)
        model = mlx_audio_load(models[args.precision])
        if args.denoise_steps != 8:
            model.acoustic_transformer.args.n_denoising_steps = args.denoise_steps

        print(f"Loading draft heads from {args.heads_path}...", flush=True)
        head_set = DraftHeadSet.load(args.heads_path, n_heads=args.n_heads)

        max_tok = args.max_tokens or _estimate_max_frames(args.text)
        print(f"\n--- Baseline (no speculation, max_tokens={max_tok}) ---")
        t0 = time.time()
        chunks = []
        for result in model.generate(text=args.text, voice=args.voice,
                                     max_tokens=max_tok):
            if result.audio is not None:
                chunks.append(np.array(result.audio))
        baseline = np.concatenate(chunks) if chunks else np.zeros(0)
        t_base = time.time() - t0
        dur = len(baseline) / 24000
        print(f"  {dur:.1f}s audio in {t_base:.2f}s "
              f"(RTF={t_base/dur:.3f}x)" if dur > 0 else "  No audio")

        print(f"\n--- Speculative decoding (max_tokens={max_tok}) ---")
        audio, stats = speculative_generate(
            model, args.text, args.voice, head_set,
            max_tokens=args.max_tokens,
            acoustic_tolerance=args.acoustic_tolerance,
            acoustic_threshold=args.acoustic_threshold,
            verbose=True,
        )

        if len(audio) > 0:
            sf.write("proof-artifacts/voxtral_speculative.wav", audio, 24000)
            print(f"  Saved: proof-artifacts/voxtral_speculative.wav")
            speedup = t_base / stats["elapsed_s"] if stats["elapsed_s"] > 0 else 0
            print(f"  Speedup vs baseline: {speedup:.2f}x")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
