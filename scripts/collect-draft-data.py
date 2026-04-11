#!/usr/bin/env python3
"""Collect (hidden_state, future_codes) training pairs from Voxtral.

Runs Voxtral on text inputs, intercepts the LM hidden states and acoustic
transformer output codes at each frame, and saves them as training data
for the speculative decoding draft heads.

Usage:
    # From LibriSpeech test-clean (downloads transcripts via HuggingFace)
    python3 scripts/collect-draft-data.py --librispeech --output data/draft-pairs-libri.npz

    # From a text file (one sentence per line)
    python3 scripts/collect-draft-data.py --input data/sentences.txt --output data/draft-pairs.npz

    # Quick test with inline sentences
    python3 scripts/collect-draft-data.py --output data/draft-pairs.npz

    # Append to existing data
    python3 scripts/collect-draft-data.py --librispeech --append data/draft-pairs.npz --output data/draft-pairs.npz
"""
import argparse
import os
import time

import mlx.core as mx
import numpy as np

DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, this is a test of the Voxtral text to speech system.",
    "Machine learning on Apple Silicon is incredibly fast these days.",
    "Can you believe how natural this voice sounds? It's all running locally.",
    "The weather today is sunny with a high of seventy two degrees.",
    "I think we should consider the implications of this decision carefully.",
    "Speculative decoding uses draft models to predict future tokens in parallel.",
    "Real time voice assistants need to generate speech faster than playback speed.",
    "The six bit quantized model provides the best balance of quality and speed.",
    "Every frame of audio requires a forward pass through the language model backbone.",
    "Flow matching with classifier free guidance doubles the compute per frame.",
    "Reducing denoising steps from eight to four nearly halves generation time.",
    "Training lightweight draft heads can amortize the cost of sequential decoding.",
    "Apple's unified memory architecture means zero copy data sharing between CPU and GPU.",
    "The acoustic transformer uses Euler integration with configurable step counts.",
    "Twenty preset voices across nine languages are available out of the box.",
    "Conversational AI requires low latency from speech recognition through synthesis.",
    "This sentence contains a variety of phonemes for robust training data coverage.",
    "Numbers like one thousand two hundred and thirty four test numeral handling.",
    "Questions are important too, right? And exclamations! Don't forget those.",
]


def estimate_max_frames(text, words_per_second=2.5, frame_rate=12.5, headroom=2.0):
    """Estimate a sensible frame cap from text length.

    English speech averages ~150 WPM = 2.5 WPS. We allow `headroom`x the
    estimate to accommodate pauses and slow voices, with a floor of 50 frames
    (~4s) and ceiling of 500 frames (~40s).
    """
    n_words = len(text.split())
    est_seconds = n_words / words_per_second
    max_frames = int(est_seconds * frame_rate * headroom)
    return max(50, min(max_frames, 500))


def load_librispeech_transcripts(data_dir="data/librispeech"):
    """Load LibriSpeech test-clean transcripts from extracted .trans.txt files.

    Falls back to pre-parsed data/librispeech-sentences.txt if available.
    """
    txt_path = os.path.join(os.path.dirname(data_dir), "librispeech-sentences.txt")
    if os.path.exists(txt_path):
        print(f"  Loading transcripts from {txt_path}...", flush=True)
        with open(txt_path) as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(sentences)} sentences", flush=True)
        return sentences

    import glob
    pattern = os.path.join(data_dir, "LibriSpeech/test-clean/*/*/*.trans.txt")
    trans_files = glob.glob(pattern)
    if not trans_files:
        raise FileNotFoundError(
            f"No .trans.txt files found at {pattern}. "
            "Download: curl -L https://www.openslr.org/resources/12/test-clean.tar.gz | "
            "tar xz -C data/librispeech --include='*.trans.txt'"
        )

    print(f"  Parsing {len(trans_files)} transcript files...", flush=True)
    sentences = []
    for f in trans_files:
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    text = parts[1].strip().capitalize()
                    if text[-1].isalnum():
                        text += "."
                    sentences.append(text)

    print(f"  Loaded {len(sentences)} LibriSpeech transcripts", flush=True)
    return sentences


def collect_pairs(model, sentences, voice, lookahead, denoise_steps,
                   max_frames_per_sentence=None, use_adaptive_cap=True):
    """Run Voxtral on sentences, capture hidden states and codes.

    If use_adaptive_cap is True, max_frames_per_sentence is computed per
    sentence from text length (prevents runaway while allowing long utterances).
    """
    from mlx_lm.models.cache import make_prompt_cache

    lm_backbone = model.language_model.model.model
    at = model.acoustic_transformer

    if denoise_steps != 8:
        at.args.n_denoising_steps = denoise_steps

    all_hiddens = []
    all_codes = []
    sent_boundaries = []

    for i, text in enumerate(sentences):
        if not text.strip():
            continue
        if use_adaptive_cap:
            frame_cap = estimate_max_frames(text)
        else:
            frame_cap = max_frames_per_sentence or 250

        t0 = time.time()
        input_ids = model._encode_text(text, voice)
        input_ids_mx = mx.array(input_ids)[None, :]
        input_embeddings = model._build_input_embeddings(input_ids_mx, voice)

        cache = make_prompt_cache(model.language_model.model)
        hidden = lm_backbone(
            input_ids_mx, cache=cache, input_embeddings=input_embeddings
        )
        audio_tok_emb = model.language_model.embed_tokens(
            mx.array([[model.config.audio_token_id]])
        )
        hidden = lm_backbone(
            mx.array([[model.config.audio_token_id]]),
            cache=cache,
            input_embeddings=audio_tok_emb,
        )

        frame_hiddens = []
        frame_codes = []

        for frame_idx in range(frame_cap):
            h = hidden[:, -1, :]
            h_f32 = h.astype(mx.float32)
            mx.eval(h_f32)
            frame_hiddens.append(np.array(h_f32[0]))

            codes = at.decode_one_frame(h)
            codes = codes.astype(mx.int32)
            mx.eval(codes)

            semantic_code = codes[0, 0].item()
            if semantic_code <= 1:
                break

            frame_codes.append(np.array(codes[0]))

            global_codes = model._codes_to_global_indices(codes)
            code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
            next_embedding = code_embeddings.sum(axis=1, keepdims=True)
            dummy_input = mx.array([[model.config.audio_token_id]])
            hidden = lm_backbone(
                dummy_input, cache=cache, input_embeddings=next_embedding
            )

            if frame_idx % 50 == 0:
                mx.clear_cache()

        n_frames = len(frame_codes)
        capped = (frame_idx == frame_cap - 1) and (n_frames == frame_cap)
        elapsed = time.time() - t0
        tag = f" [CAPPED@{frame_cap}]" if capped else ""
        print(
            f"  [{i+1}/{len(sentences)}] {n_frames} frames "
            f"({n_frames * 0.08:.1f}s audio) in {elapsed:.1f}s{tag}: "
            f"{text[:60]}{'...' if len(text) > 60 else ''}",
            flush=True,
        )

        start_idx = len(all_codes)
        all_hiddens.extend(frame_hiddens[:n_frames])
        all_codes.extend(frame_codes)
        sent_boundaries.append((start_idx, len(all_codes)))

        mx.clear_cache()
        del cache

    if not all_hiddens:
        raise RuntimeError("No frames collected from any sentence")

    hiddens = np.stack(all_hiddens)
    codes = np.stack(all_codes)

    pair_hiddens_list = []
    targets_list = {f"target_{k}": [] for k in range(1, lookahead + 1)}

    for start, end in sent_boundaries:
        n = end - start - lookahead
        if n <= 0:
            continue
        pair_hiddens_list.append(hiddens[start : start + n])
        for k in range(1, lookahead + 1):
            targets_list[f"target_{k}"].append(codes[start + k : start + n + k])

    pair_hiddens = np.concatenate(pair_hiddens_list)
    targets = {k: np.concatenate(v) for k, v in targets_list.items()}

    return pair_hiddens, targets


def main():
    parser = argparse.ArgumentParser(description="Collect draft head training data from Voxtral")
    parser.add_argument("--output", default="data/draft-pairs.npz")
    parser.add_argument("--input", default=None, help="Text file with one sentence per line")
    parser.add_argument("--librispeech", action="store_true",
                        help="Use LibriSpeech test-clean transcripts (downloads via HuggingFace)")
    parser.add_argument("--append", default=None,
                        help="Path to existing .npz to append new data to")
    parser.add_argument("--voice", default="cheerful_male")
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--precision", default="6bit")
    parser.add_argument("--denoise-steps", type=int, default=8,
                        help="Denoising steps for data collection (default: 8 for reliable generation)")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Override adaptive frame cap with fixed value")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from .checkpoint.npz (same --output path). Requires "
        "checkpoint_batch_end in the file or --resume-after-sentences.",
    )
    parser.add_argument(
        "--resume-after-sentences",
        type=int,
        default=None,
        help="When --resume and the checkpoint lacks checkpoint_batch_end, "
        "start from this sentence index (multiple of 200 recommended).",
    )
    args = parser.parse_args()

    from mlx_audio.tts.utils import load as mlx_audio_load

    print(f"Loading Voxtral {args.precision}...", flush=True)
    model = mlx_audio_load(
        {"4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
         "6bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
         "bf16": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"}[args.precision]
    )

    if args.librispeech:
        sentences = load_librispeech_transcripts()
    elif args.input:
        with open(args.input) as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    if args.max_sentences:
        sentences = sentences[: args.max_sentences]

    use_adaptive = args.max_frames is None
    checkpoint_every = 200
    n_total = len(sentences)
    print(f"Collecting data from {n_total} sentences "
          f"(lookahead={args.lookahead}, denoise_steps={args.denoise_steps}, "
          f"{'adaptive' if use_adaptive else f'fixed@{args.max_frames}'} cap, "
          f"checkpoint every {checkpoint_every})...",
          flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    checkpoint_path = args.output.replace(".npz", ".checkpoint.npz")
    t0 = time.time()
    merged_h = None
    merged_t = None
    batch_start0 = 0

    if args.resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"--resume requires {checkpoint_path} (missing). "
                "Remove --resume for a fresh run."
            )
        ckpt = np.load(checkpoint_path)
        merged_h = np.asarray(ckpt["hiddens"])
        merged_t = {k: np.asarray(ckpt[k]) for k in ckpt.files if k.startswith("target_")}
        if "checkpoint_batch_end" in ckpt.files:
            batch_start0 = int(np.asarray(ckpt["checkpoint_batch_end"]).reshape(-1)[0])
        elif args.resume_after_sentences is not None:
            batch_start0 = args.resume_after_sentences
        else:
            raise ValueError(
                f"{checkpoint_path} has no checkpoint_batch_end array. "
                "Re-save with a current collector, or pass --resume-after-sentences N "
                "(sentence index after last fully saved batch)."
            )
        if batch_start0 < 0 or batch_start0 > n_total:
            raise ValueError(f"Invalid resume start {batch_start0} (n_sentences={n_total})")
        if batch_start0 >= n_total:
            print(
                f"Checkpoint already covers all {n_total} sentences; "
                f"writing final {args.output} from checkpoint.",
                flush=True,
            )
        else:
            print(
                f"Resuming from sentence {batch_start0} ({len(merged_h)} pairs in checkpoint)",
                flush=True,
            )

    for batch_start in range(batch_start0, n_total, checkpoint_every):
        batch_end = min(batch_start + checkpoint_every, n_total)
        batch_sents = sentences[batch_start:batch_end]

        print(f"\n--- Batch {batch_start}-{batch_end} of {n_total} ---", flush=True)
        hiddens, targets = collect_pairs(
            model, batch_sents, args.voice, args.lookahead, args.denoise_steps,
            max_frames_per_sentence=args.max_frames,
            use_adaptive_cap=use_adaptive,
        )
        if merged_h is None:
            merged_h, merged_t = hiddens, targets
        else:
            merged_h = np.concatenate([merged_h, hiddens])
            merged_t = {
                k: np.concatenate([merged_t[k], targets[k]]) for k in targets
            }

        np.savez_compressed(
            checkpoint_path,
            hiddens=merged_h,
            checkpoint_batch_end=np.array([batch_end], dtype=np.int32),
            **merged_t,
        )
        elapsed_so_far = time.time() - t0
        rate = batch_end / elapsed_so_far
        eta = (n_total - batch_end) / rate if rate > 0 else 0
        print(f"  Checkpoint: {len(merged_h)} pairs saved to {checkpoint_path} "
              f"({elapsed_so_far/60:.0f}min elapsed, ~{eta/60:.0f}min remaining)",
              flush=True)

    elapsed = time.time() - t0

    if merged_h is None:
        raise RuntimeError("No data collected (empty sentence list?)")

    final_h = merged_h
    final_t = merged_t

    if args.append and os.path.exists(args.append):
        print(f"\nAppending to existing data at {args.append}...", flush=True)
        existing = np.load(args.append)
        final_h = np.concatenate([existing["hiddens"], final_h])
        for k in final_t:
            final_t[k] = np.concatenate([existing[k], final_t[k]])

    np.savez_compressed(args.output, hiddens=final_h, **final_t)

    checkpoint_path = args.output.replace(".npz", ".checkpoint.npz")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nSaved {len(final_h)} pairs to {args.output}")
    print(f"  hiddens: {final_h.shape} ({final_h.dtype})")
    for k, v in final_t.items():
        print(f"  {k}: {v.shape}")
    print(f"  Total time: {elapsed/60:.1f}min ({elapsed:.0f}s)")
    print(f"  File size: {os.path.getsize(args.output) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
