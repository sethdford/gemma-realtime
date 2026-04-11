#!/usr/bin/env python3
"""
Fish STS Training Pipeline: True Speech-to-Speech on Fish Audio's Codec.

Three-phase training to build a text-free STS system:

    Phase A — Projection warm-up (~1 hour, ~2K iters)
        Freeze everything. Train AudioInputProjection + AudioOutputHead only.
        Aligns Fish cb0 token space ↔ Gemma embedding space.
        Data: paired (audio, text) from LibriTTS — we have cb0 tokens from SNAC
              multicodebook extraction, and can also do Fish re-encoding.
        Loss: CTC-style alignment + embedding cosine similarity.

    Phase B — Speech layer pre-training (~8 hours, ~20K iters)
        Freeze Gemma text weights + Fish codec. Train speech branch layers.
        Data: LibriTTS full training set (31K utterances).
        Loss: next-token prediction on cb0 + inner monologue CE.

    Phase C — Joint STS fine-tuning (~24 hours, ~50K iters)
        Freeze Gemma text weights. Fine-tune all speech layers jointly.
        Data: spoken QA pairs, multi-turn conversations.
        Loss: cb0 generation + inner monologue + turn-state prediction.

Usage:
    # SNAC proxy training (existing pipeline)
    python3 scripts/train-fish-sts.py all --codec snac

    # Fish DAC training (requires extract step first)
    python3 scripts/train-fish-sts.py extract \\
        --input data/libritts-codec-train-full-eos.jsonl \\
        --output data/libritts-fish-dac-tokens.jsonl
    python3 scripts/train-fish-sts.py all --codec fish

    # Individual phases with Fish DAC
    python3 scripts/train-fish-sts.py phase-a --codec fish --iters 2000 --lr 3e-4
    python3 scripts/train-fish-sts.py phase-b --codec fish --iters 20000 --lr 1e-4
    python3 scripts/train-fish-sts.py phase-c --codec fish --iters 50000 --lr 5e-5
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

sys.path.insert(0, str(Path(__file__).parent))

FISH_CODEBOOK_SIZE = 1024
FISH_N_CODEBOOKS = 8
SNAC_CODEBOOK_SIZE = 4096
SNAC_N_CODEBOOKS = 3
OUTPUT_DIR = Path("adapters/fish-sts")

# Default data paths per codec
DEFAULT_DATA = {
    "snac": "data/libritts-multicodebook.jsonl",
    "fish": "data/libritts-fish-dac-tokens.jsonl",
}


def resolve_codec_config(args):
    """Resolve data path and codec parameters from --codec flag."""
    codec = getattr(args, "codec", "snac")
    if args.data is None:
        args.data = DEFAULT_DATA[codec]
    if codec == "fish":
        return FISH_CODEBOOK_SIZE, FISH_N_CODEBOOKS, "fish_cb0"
    else:
        return SNAC_CODEBOOK_SIZE, SNAC_N_CODEBOOKS, "cb0"


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_multicodebook_data(path: str) -> list[dict]:
    """Load data with cb0/cb1/cb2 SNAC tokens and text."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("cb0") and item.get("text"):
                data.append(item)
    return data


def load_fish_token_data(path: str) -> list[dict]:
    """Load data with Fish DAC codec tokens."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("fish_cb0") and item.get("text"):
                data.append(item)
    return data


def load_audio_text_data(path: str) -> list[dict]:
    """Load basic audio_path + text data."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("audio_path") and item.get("text"):
                data.append(item)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Phase: Extract Fish Codec Tokens
# ══════════════════════════════════════════════════════════════════════════════

def extract_fish_tokens(args):
    """Re-encode training audio with Fish DAC to get 10-codebook tokens."""
    import soundfile as sf
    from codec import AudioCodec

    print(f"\n{'='*60}")
    print(f"  Fish Codec Token Extraction")
    print(f"{'='*60}")

    codec = AudioCodec("fish")
    try:
        codec.load()
        use_fish = True
    except Exception as e:
        print(f"  Fish DAC not available ({e}), using SNAC cb0 as proxy", flush=True)
        print(f"  (SNAC cb0 is semantic — good enough for projection training)", flush=True)
        use_fish = False

    input_path = Path(args.input)
    output_path = Path(args.output)

    items = load_audio_text_data(str(input_path))
    print(f"  Input:  {input_path} ({len(items)} samples)")
    print(f"  Output: {output_path}")

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

                target_sr = 44100 if use_fish else 24000
                if sr != target_sr:
                    n_out = int(len(audio) * target_sr / sr)
                    audio = np.interp(
                        np.linspace(0, 1, n_out),
                        np.linspace(0, 1, len(audio)),
                        audio,
                    ).astype(np.float32)

                tokens = codec.encode(audio)
                if tokens.codes.shape[1] == 0:
                    continue

                if use_fish:
                    record = {
                        "text": item["text"],
                        "audio_path": audio_path,
                        "fish_cb0": tokens.codes[0].tolist(),
                        "fish_all_cbs": [tokens.codes[i].tolist()
                                         for i in range(tokens.n_codebooks)],
                        "n_codebooks": tokens.n_codebooks,
                        "n_frames": tokens.n_frames,
                    }
                else:
                    raw = getattr(codec, '_snac_codes_raw', None)
                    if raw is None or len(raw) < 1:
                        continue
                    record = {
                        "text": item["text"],
                        "audio_path": audio_path,
                        "fish_cb0": raw[0].tolist() if hasattr(raw[0], 'tolist') else list(raw[0]),
                        "n_codebooks": 1,
                        "n_frames": len(raw[0]),
                        "proxy": "snac_cb0",
                    }

                out_f.write(json.dumps(record) + "\n")
                extracted += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error on {audio_path}: {e}")

            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(items)}: extracted={extracted}, errors={errors}",
                      flush=True)

    print(f"\n  Done: {extracted} extracted, {errors} errors")
    print(f"  Output: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Projection Warm-up
# ══════════════════════════════════════════════════════════════════════════════

def train_phase_a(args):
    """Train AudioInputProjection + AudioOutputHead.

    Aligns Fish/SNAC cb0 tokens ↔ Gemma embedding space.
    Uses existing SNAC cb0 data as proxy (both are semantic codebooks).

    Two loss terms:
        1. Embedding alignment: projected cb0 embeddings ≈ embed_tokens(text)
        2. Reconstruction: output head should recover cb0 from Gemma embeddings
    """
    from fish_sts import FishSpeechToSpeech, FishSTSConfig, PRESET_CONFIGS

    cb_size, n_cbs, cb0_key = resolve_codec_config(args)

    print(f"\n{'='*60}")
    print(f"  Phase A: Projection Warm-up")
    print(f"  Codec: {'Fish DAC' if cb0_key == 'fish_cb0' else 'SNAC'} "
          f"({cb_size} vocab, {n_cbs} codebooks)")
    print(f"  Data: {args.data}")
    print(f"{'='*60}")

    # Load data
    if cb0_key == "fish_cb0":
        data = load_fish_token_data(args.data)
    else:
        data = load_multicodebook_data(args.data)
    np.random.shuffle(data)
    n_valid = min(200, len(data) // 10)
    valid_data = data[:n_valid]
    train_data = data[n_valid:]
    print(f"  Train: {len(train_data)}, Valid: {n_valid}")

    # Load Gemma for embeddings
    print("  Loading Gemma for embeddings...", flush=True)
    from mlx_lm import load as lm_load
    gemma, tokenizer = lm_load(args.model)
    if hasattr(gemma, "language_model"):
        inner = gemma.language_model.model
    else:
        inner = gemma.model

    probe = inner.embed_tokens(mx.array([[0]]))
    llm_dim = probe.shape[-1]
    print(f"  Gemma embed_dim: {llm_dim}", flush=True)

    # Build STS model (only projections will be trained)
    config = PRESET_CONFIGS.get(args.target, PRESET_CONFIGS["e4b"])
    config = FishSTSConfig(
        llm_dim=llm_dim,
        fish_codebook_size=cb_size,
        fish_n_codebooks=n_cbs,
        speech_adapter_dim=512,
        speech_adapter_heads=8,
        speech_adapter_layers=4,
        speech_adapter_ff=2048,
        inner_monologue=False,
    )
    model = FishSpeechToSpeech(config)
    n_params = model.num_params()
    print(f"  STS model: {n_params/1e6:.1f}M total params", flush=True)

    # Only train projections (audio_input + layer_split.speech_output)
    model.freeze()
    model.audio_input.unfreeze()
    model.layer_split.speech_output.unfreeze()

    trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable (projections only): {trainable/1e6:.2f}M", flush=True)

    # Optimizer
    output_dir = OUTPUT_DIR / "phase-a"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = min(200, args.iters // 5)
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps),
                                  end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, max(1, warmup_steps))
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    total_loss = 0
    report_count = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.iters + 1):
        item = train_data[np.random.randint(0, len(train_data))]
        cb0 = item.get(cb0_key) or item.get("cb0")
        text = item["text"]

        max_cb0 = min(len(cb0), 100)
        cb0_arr = mx.array([cb0[:max_cb0]], dtype=mx.int32)

        # Text -> Gemma embeddings (frozen, target for alignment)
        text_ids = tokenizer.encode(text[:200], add_special_tokens=False)
        if not text_ids:
            continue
        text_ids_arr = mx.array([text_ids], dtype=mx.int32)
        target_emb = mx.stop_gradient(inner.embed_tokens(text_ids_arr))

        def loss_step(model):
            # Forward: cb0 → projected embeddings
            projected = model.audio_input(cb0_arr)  # (1, T_audio, llm_dim)

            # Loss 1: Embedding alignment (mean pool both, cosine similarity)
            audio_mean = mx.mean(projected, axis=1)     # (1, llm_dim)
            text_mean = mx.mean(target_emb, axis=1)     # (1, llm_dim)

            # Cosine distance
            cos_sim = mx.sum(audio_mean * text_mean, axis=-1) / (
                mx.sqrt(mx.sum(audio_mean ** 2, axis=-1)) *
                mx.sqrt(mx.sum(text_mean ** 2, axis=-1)) + 1e-8
            )
            align_loss = 1.0 - mx.mean(cos_sim)

            # MSE component
            min_len = min(projected.shape[1], target_emb.shape[1])
            mse_loss = mx.mean((projected[:, :min_len] - target_emb[:, :min_len]) ** 2)

            # Loss 2: Reconstruction (output head should recover cb0 from embeddings)
            recon_logits = model.layer_split.speech_output(projected)
            cb0_targets = cb0_arr[:, :projected.shape[1]]
            if cb0_targets.shape[1] < recon_logits.shape[1]:
                recon_logits = recon_logits[:, :cb0_targets.shape[1]]
            elif cb0_targets.shape[1] > recon_logits.shape[1]:
                cb0_targets = cb0_targets[:, :recon_logits.shape[1]]

            recon_flat = recon_logits.reshape(-1, recon_logits.shape[-1])
            target_flat = cb0_targets.reshape(-1)
            recon_loss = mx.mean(nn.losses.cross_entropy(recon_flat, target_flat))

            return 0.4 * align_loss + 0.2 * mse_loss + 0.4 * recon_loss

        loss, grads = mx.value_and_grad(loss_step)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

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
            print(f"  Step {step:>5}/{args.iters}: loss={avg:.4f} | "
                  f"{sps:.1f} step/s | {mem_gb:.1f}GB", flush=True)
            total_loss = 0
            report_count = 0

        if step % args.save_every == 0 or step == args.iters:
            # Validation
            val_losses = []
            for vi in range(min(50, len(valid_data))):
                v = valid_data[vi]
                vcb0 = (v.get(cb0_key) or v.get("cb0"))[:50]
                vtext = v["text"]
                if not vcb0 or not vtext:
                    continue
                vcb0_arr = mx.array([vcb0], dtype=mx.int32)
                vids = tokenizer.encode(vtext[:200], add_special_tokens=False)
                if not vids:
                    continue
                vtarget = mx.stop_gradient(inner.embed_tokens(mx.array([vids])))
                vproj = model.audio_input(vcb0_arr)

                # Cosine distance
                a_mean = mx.mean(vproj, axis=1)
                t_mean = mx.mean(vtarget, axis=1)
                cs = mx.sum(a_mean * t_mean, axis=-1) / (
                    mx.sqrt(mx.sum(a_mean ** 2, axis=-1)) *
                    mx.sqrt(mx.sum(t_mean ** 2, axis=-1)) + 1e-8
                )
                v_align = 1.0 - mx.mean(cs)
                mx.eval(v_align)
                val_losses.append(v_align.item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            marker = " (best)" if val_loss < best_val_loss else ""
            print(f"  Validation: align_loss={val_loss:.4f}{marker}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            weights = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(output_dir / "phase_a.safetensors"), weights)
            print(f"  Saved to {output_dir}", flush=True)

    print(f"\n  Phase A complete. Best val alignment loss: {best_val_loss:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Phase B: Speech Layer Pre-training
# ══════════════════════════════════════════════════════════════════════════════

def train_phase_b(args):
    """Train speech branch layers for next-token prediction on cb0.

    The speech layers learn to generate audio token sequences given
    input audio tokens — pure autoregressive modeling on cb0.
    """
    from fish_sts import FishSpeechToSpeech, FishSTSConfig, PRESET_CONFIGS

    cb_size, n_cbs, cb0_key = resolve_codec_config(args)

    print(f"\n{'='*60}")
    print(f"  Phase B: Speech Layer Pre-training")
    print(f"  Codec: {'Fish DAC' if cb0_key == 'fish_cb0' else 'SNAC'} "
          f"({cb_size} vocab, {n_cbs} codebooks)")
    print(f"{'='*60}")

    if cb0_key == "fish_cb0":
        data = load_fish_token_data(args.data)
    else:
        data = load_multicodebook_data(args.data)
    np.random.shuffle(data)
    n_valid = min(200, len(data) // 10)
    valid_data = data[:n_valid]
    train_data = data[n_valid:]
    print(f"  Train: {len(train_data)}, Valid: {n_valid}")

    # Load Gemma
    print("  Loading Gemma...", flush=True)
    from mlx_lm import load as lm_load
    gemma, tokenizer = lm_load(args.model)
    if hasattr(gemma, "language_model"):
        inner = gemma.language_model.model
    else:
        inner = gemma.model
    probe = inner.embed_tokens(mx.array([[0]]))
    llm_dim = probe.shape[-1]

    config = FishSTSConfig(
        llm_dim=llm_dim,
        fish_codebook_size=cb_size,
        fish_n_codebooks=n_cbs,
        speech_adapter_dim=512,
        speech_adapter_heads=8,
        speech_adapter_layers=4,
        speech_adapter_ff=2048,
        inner_monologue=True,
    )
    model = FishSpeechToSpeech(config)

    # Load Phase A weights if available
    phase_a_path = OUTPUT_DIR / "phase-a" / "phase_a.safetensors"
    if phase_a_path.exists():
        w = mx.load(str(phase_a_path))
        model.load_weights(list(w.items()), strict=False)
        print(f"  Loaded Phase A weights from {phase_a_path}", flush=True)
    else:
        print("  No Phase A weights found, training from scratch", flush=True)

    # Freeze everything except speech layers + fast_ar + state_head
    model.freeze()
    model.layer_split.unfreeze()
    model.fast_ar.unfreeze()
    model.state_head.unfreeze()
    model.audio_input.unfreeze()
    model.audio_embed.unfreeze()

    trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable: {trainable/1e6:.2f}M", flush=True)

    output_dir = OUTPUT_DIR / "phase-b"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = min(500, args.iters // 5)
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps),
                                  end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, max(1, warmup_steps))
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    total_loss = 0
    report_count = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.iters + 1):
        item = train_data[np.random.randint(0, len(train_data))]
        cb0 = item.get(cb0_key) or item.get("cb0")
        text = item["text"]

        max_len = min(len(cb0), 80)
        if max_len < 5:
            continue

        # Input: cb0[:-1], Target: cb0[1:]
        cb0_input = mx.array([cb0[:max_len - 1]], dtype=mx.int32)
        cb0_target = mx.array([cb0[1:max_len]], dtype=mx.int32)

        # Also get text embeddings for inner monologue supervision
        text_ids = tokenizer.encode(text[:200], add_special_tokens=False)
        text_emb = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids]))) if text_ids else None

        def loss_step(model):
            # Project input cb0 to Gemma space
            audio_emb = model.audio_input(cb0_input)

            # Run through speech branch
            T = audio_emb.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            speech_logits = model.layer_split.forward_speech_branch(audio_emb, mask=mask)

            # CE loss on next-token prediction
            logits_flat = speech_logits.reshape(-1, speech_logits.shape[-1])
            targets_flat = cb0_target.reshape(-1)

            # Clamp targets to valid range
            valid_mask = (targets_flat >= 0) & (targets_flat < speech_logits.shape[-1])
            targets_safe = mx.where(valid_mask, targets_flat,
                                    mx.zeros_like(targets_flat))
            mask_float = valid_mask.astype(mx.float32)

            ce = nn.losses.cross_entropy(logits_flat, targets_safe, reduction="none")
            ce_loss = mx.sum(ce * mask_float) / mx.maximum(mx.sum(mask_float), 1.0)

            # Inner monologue alignment (if text available)
            mono_loss = mx.array(0.0)
            if text_emb is not None and model.config.inner_monologue:
                audio_mean = mx.mean(audio_emb, axis=1)
                text_mean = mx.mean(text_emb, axis=1)
                cos_sim = mx.sum(audio_mean * text_mean, axis=-1) / (
                    mx.sqrt(mx.sum(audio_mean ** 2, axis=-1)) *
                    mx.sqrt(mx.sum(text_mean ** 2, axis=-1)) + 1e-8
                )
                mono_loss = 1.0 - mx.mean(cos_sim)

            return 0.8 * ce_loss + 0.2 * mono_loss

        loss, grads = mx.value_and_grad(loss_step)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        report_count += 1

        if step % args.report_every == 0:
            avg = total_loss / max(report_count, 1)
            elapsed = time.time() - t0
            sps = step / elapsed
            nats_above_random = avg - np.log(SNAC_CODEBOOK_SIZE)
            try:
                mem_gb = mx.get_active_memory() / 1e9
            except Exception:
                mem_gb = 0
            print(f"  Step {step:>5}/{args.iters}: loss={avg:.4f} "
                  f"(Δrandom={nats_above_random:+.2f}) | "
                  f"{sps:.1f} step/s | {mem_gb:.1f}GB", flush=True)
            total_loss = 0
            report_count = 0

        if step % args.save_every == 0 or step == args.iters:
            val_losses = []
            for vi in range(min(50, len(valid_data))):
                v = valid_data[vi]
                vcb0 = (v.get(cb0_key) or v.get("cb0"))[:50]
                if len(vcb0) < 5:
                    continue
                vin = mx.array([vcb0[:-1]], dtype=mx.int32)
                vtgt = mx.array([vcb0[1:]], dtype=mx.int32)
                vemb = model.audio_input(vin)
                T = vemb.shape[1]
                vmask = nn.MultiHeadAttention.create_additive_causal_mask(T)
                vlogits = model.layer_split.forward_speech_branch(vemb, mask=vmask)
                vflat = vlogits.reshape(-1, vlogits.shape[-1])
                vtgt_flat = vtgt.reshape(-1)
                vce = mx.mean(nn.losses.cross_entropy(vflat, vtgt_flat))
                mx.eval(vce)
                val_losses.append(vce.item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            marker = " (best)" if val_loss < best_val_loss else ""
            print(f"  Validation: loss={val_loss:.4f}{marker}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            weights = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(output_dir / "phase_b.safetensors"), weights)
            print(f"  Saved to {output_dir}", flush=True)

    print(f"\n  Phase B complete. Best val loss: {best_val_loss:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Phase C: Joint STS Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

def train_phase_c(args):
    """Full STS fine-tuning: audio input → generate audio response.

    Trains the complete pipeline: encode user audio → speech branch reasoning
    → generate response cb0 tokens. Uses text as supervision signal only
    (inner monologue), never as the primary pathway.
    """
    from fish_sts import FishSpeechToSpeech, FishSTSConfig

    cb_size, n_cbs, cb0_key = resolve_codec_config(args)

    print(f"\n{'='*60}")
    print(f"  Phase C: Joint STS Fine-tuning")
    print(f"  Codec: {'Fish DAC' if cb0_key == 'fish_cb0' else 'SNAC'} "
          f"({cb_size} vocab, {n_cbs} codebooks)")
    print(f"{'='*60}")

    if cb0_key == "fish_cb0":
        data = load_fish_token_data(args.data)
    else:
        data = load_multicodebook_data(args.data)
    np.random.shuffle(data)
    n_valid = min(200, len(data) // 10)
    valid_data = data[:n_valid]
    train_data = data[n_valid:]
    print(f"  Train: {len(train_data)}, Valid: {n_valid}")

    # Load Gemma
    print("  Loading Gemma...", flush=True)
    from mlx_lm import load as lm_load
    gemma, tokenizer = lm_load(args.model)
    if hasattr(gemma, "language_model"):
        inner = gemma.language_model.model
    else:
        inner = gemma.model
    probe = inner.embed_tokens(mx.array([[0]]))
    llm_dim = probe.shape[-1]

    config = FishSTSConfig(
        llm_dim=llm_dim,
        fish_codebook_size=cb_size,
        fish_n_codebooks=n_cbs,
        speech_adapter_dim=512,
        speech_adapter_heads=8,
        speech_adapter_layers=4,
        speech_adapter_ff=2048,
        inner_monologue=True,
    )
    model = FishSpeechToSpeech(config)

    # Load Phase B weights (or Phase A as fallback)
    phase_b_path = OUTPUT_DIR / "phase-b" / "phase_b.safetensors"
    phase_a_path = OUTPUT_DIR / "phase-a" / "phase_a.safetensors"
    if phase_b_path.exists():
        w = mx.load(str(phase_b_path))
        model.load_weights(list(w.items()), strict=False)
        print(f"  Loaded Phase B weights from {phase_b_path}", flush=True)
        # Free disk: Phase C will save its own weights
        if args.cleanup_prior:
            phase_b_path.unlink()
            print(f"  Deleted Phase B weights to free disk", flush=True)
    elif phase_a_path.exists():
        w = mx.load(str(phase_a_path))
        model.load_weights(list(w.items()), strict=False)
        print(f"  Loaded Phase A weights (no Phase B found)", flush=True)
    else:
        print("  No prior weights found, training from scratch", flush=True)

    # Unfreeze all speech components
    model.freeze()
    model.audio_input.unfreeze()
    model.audio_embed.unfreeze()
    model.layer_split.unfreeze()
    model.fast_ar.unfreeze()
    model.state_head.unfreeze()

    trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"  Trainable: {trainable/1e6:.2f}M", flush=True)

    output_dir = OUTPUT_DIR / "phase-c"
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = min(1000, args.iters // 5)
    lr_sched = optim.cosine_decay(args.lr, max(1, args.iters - warmup_steps),
                                  end=args.lr * 0.01)
    warmup = optim.linear_schedule(0, args.lr, max(1, warmup_steps))
    schedule = optim.join_schedules([warmup, lr_sched], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

    total_loss = 0
    report_count = 0
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.iters + 1):
        # STS training: simulate user audio → response audio
        # Use two random utterances: first = "user audio", second = "response audio"
        idx1 = np.random.randint(0, len(train_data))
        idx2 = np.random.randint(0, len(train_data))
        user_item = train_data[idx1]
        resp_item = train_data[idx2]

        user_cb0 = (user_item.get(cb0_key) or user_item.get("cb0"))[:40]
        resp_cb0 = (resp_item.get(cb0_key) or resp_item.get("cb0"))[:60]
        resp_text = resp_item["text"]

        if len(user_cb0) < 3 or len(resp_cb0) < 3:
            continue

        user_arr = mx.array([user_cb0], dtype=mx.int32)
        resp_input = mx.array([resp_cb0[:-1]], dtype=mx.int32)
        resp_target = mx.array([resp_cb0[1:]], dtype=mx.int32)

        text_ids = tokenizer.encode(resp_text[:200], add_special_tokens=False)
        text_emb = mx.stop_gradient(inner.embed_tokens(mx.array([text_ids]))) if text_ids else None

        def loss_step(model):
            # Encode user audio
            user_emb = model.audio_input(user_arr)

            # Encode response prefix (teacher forcing)
            resp_emb = model.audio_input(resp_input)

            # Concatenate: [user_audio, response_prefix]
            combined = mx.concatenate([user_emb, resp_emb], axis=1)

            T = combined.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            speech_logits = model.layer_split.forward_speech_branch(combined, mask=mask)

            # Only compute loss on response portion
            resp_logits = speech_logits[:, user_emb.shape[1]:, :]
            logits_flat = resp_logits.reshape(-1, resp_logits.shape[-1])
            targets_flat = resp_target.reshape(-1)

            valid_mask = (targets_flat >= 0) & (targets_flat < resp_logits.shape[-1])
            targets_safe = mx.where(valid_mask, targets_flat,
                                    mx.zeros_like(targets_flat))
            mask_float = valid_mask.astype(mx.float32)

            ce = nn.losses.cross_entropy(logits_flat, targets_safe, reduction="none")
            ce_loss = mx.sum(ce * mask_float) / mx.maximum(mx.sum(mask_float), 1.0)

            # Inner monologue supervision
            mono_loss = mx.array(0.0)
            if text_emb is not None:
                resp_mean = mx.mean(resp_emb, axis=1)
                text_mean = mx.mean(text_emb, axis=1)
                cos_sim = mx.sum(resp_mean * text_mean, axis=-1) / (
                    mx.sqrt(mx.sum(resp_mean ** 2, axis=-1)) *
                    mx.sqrt(mx.sum(text_mean ** 2, axis=-1)) + 1e-8
                )
                mono_loss = 1.0 - mx.mean(cos_sim)

            # State prediction: after user audio → should be SPEAK(1)
            state_logits = model.state_head(user_emb[:, -1:, :]).squeeze(1)
            state_target = mx.array([1], dtype=mx.int32)  # SPEAK
            state_loss = mx.mean(nn.losses.cross_entropy(state_logits, state_target))

            return 0.7 * ce_loss + 0.2 * mono_loss + 0.1 * state_loss

        loss, grads = mx.value_and_grad(loss_step)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        report_count += 1

        if step % args.report_every == 0:
            avg = total_loss / max(report_count, 1)
            elapsed = time.time() - t0
            sps = step / elapsed
            remaining = (args.iters - step) / max(sps, 0.01)
            eta_h, eta_m = divmod(int(remaining), 3600)
            eta_m = eta_m // 60
            try:
                mem_gb = mx.get_active_memory() / 1e9
            except Exception:
                mem_gb = 0
            print(f"  Step {step:>5}/{args.iters}: loss={avg:.4f} | "
                  f"{sps:.1f} step/s | {mem_gb:.1f}GB | "
                  f"ETA {eta_h}h{eta_m:02d}m", flush=True)
            total_loss = 0
            report_count = 0

        if step % args.save_every == 0 or step == args.iters:
            val_losses = []
            for vi in range(min(30, len(valid_data))):
                v = valid_data[vi]
                vcb0 = (v.get(cb0_key) or v.get("cb0"))[:40]
                if len(vcb0) < 5:
                    continue
                vin = mx.array([vcb0[:-1]], dtype=mx.int32)
                vtgt = mx.array([vcb0[1:]], dtype=mx.int32)
                vemb = model.audio_input(vin)
                T = vemb.shape[1]
                vmask = nn.MultiHeadAttention.create_additive_causal_mask(T)
                vlogits = model.layer_split.forward_speech_branch(vemb, mask=vmask)
                vflat = vlogits.reshape(-1, vlogits.shape[-1])
                vtgt_flat = vtgt.reshape(-1)
                vce = mx.mean(nn.losses.cross_entropy(vflat, vtgt_flat))
                mx.eval(vce)
                val_losses.append(vce.item())

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            marker = " (best)" if val_loss < best_val_loss else ""
            print(f"  Validation: loss={val_loss:.4f}{marker}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            weights = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
            save_path = output_dir / "fish_sts_final.safetensors"
            tmp_path = output_dir / "fish_sts_final.tmp.safetensors"
            try:
                mx.save_safetensors(str(tmp_path), weights)
                tmp_path.replace(save_path)
                print(f"  Saved to {save_path} ({save_path.stat().st_size / 1e9:.2f} GB)", flush=True)
            except RuntimeError as e:
                print(f"  WARNING: Save failed ({e}), continuing training...", flush=True)
                tmp_path.unlink(missing_ok=True)

    print(f"\n  Phase C complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Final weights: {output_dir / 'fish_sts_final.safetensors'}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fish STS Training: True Speech-to-Speech Pipeline"
    )
    sub = parser.add_subparsers(dest="command")

    # Extract
    p_ext = sub.add_parser("extract", help="Extract Fish codec tokens from audio")
    p_ext.add_argument("--input", default="data/libritts-codec-train-full-eos.jsonl")
    p_ext.add_argument("--output", default="data/libritts-fish-tokens.jsonl")

    # Shared args factory for all training phases
    def _add_train_args(p, lr, iters, report=50, save=500):
        p.add_argument("--data", default=None,
                       help="Training data JSONL (auto-detected from --codec if omitted)")
        p.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
        p.add_argument("--target", default="e4b")
        p.add_argument("--codec", choices=["snac", "fish"], default="snac",
                       help="Codec: snac (3 CB, 4096 vocab) or fish (8 CB, 1024 vocab)")
        p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--iters", type=int, default=iters)
        p.add_argument("--report-every", type=int, default=report)
        p.add_argument("--save-every", type=int, default=save)

    # Phase A
    p_a = sub.add_parser("phase-a", help="Phase A: projection warm-up")
    _add_train_args(p_a, lr=3e-4, iters=2000, report=50, save=500)

    # Phase B
    p_b = sub.add_parser("phase-b", help="Phase B: speech layer pre-training")
    _add_train_args(p_b, lr=1e-4, iters=20000, report=100, save=2000)

    # Phase C
    p_c = sub.add_parser("phase-c", help="Phase C: joint STS fine-tuning")
    _add_train_args(p_c, lr=5e-5, iters=50000, report=200, save=5000)
    p_c.add_argument("--cleanup-prior", action="store_true",
                     help="Delete Phase B weights after loading to free disk")

    # All phases
    p_all = sub.add_parser("all", help="Run all phases sequentially")
    p_all.add_argument("--data", default=None)
    p_all.add_argument("--model", default="mlx-community/gemma-4-26b-a4b-it-4bit")
    p_all.add_argument("--target", default="e4b")
    p_all.add_argument("--codec", choices=["snac", "fish"], default="snac")

    args = parser.parse_args()

    if args.command == "extract":
        extract_fish_tokens(args)
    elif args.command == "phase-a":
        train_phase_a(args)
    elif args.command == "phase-b":
        train_phase_b(args)
    elif args.command == "phase-c":
        train_phase_c(args)
    elif args.command == "all":
        # Phase A
        args.lr = 3e-4
        args.iters = 2000
        args.report_every = 50
        args.save_every = 500
        train_phase_a(args)

        # Phase B
        args.lr = 1e-4
        args.iters = 20000
        args.report_every = 100
        args.save_every = 2000
        train_phase_b(args)

        # Phase C
        args.lr = 5e-5
        args.iters = 50000
        args.report_every = 200
        args.save_every = 5000
        train_phase_c(args)

        print(f"\n{'='*60}")
        print(f"  All phases complete!")
        print(f"  Final weights: {OUTPUT_DIR / 'phase-c' / 'fish_sts_final.safetensors'}")
        print(f"{'='*60}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
