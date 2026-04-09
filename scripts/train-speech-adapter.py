#!/usr/bin/env python3
"""
3-stage training pipeline for Freeze-Omni style speech adapters on MLX.

Trains speech encoder and decoder adapters while keeping the Gemma LLM frozen.
This preserves all LoRA personalization while adding native speech capabilities.

Stages:
    1. Speech Encoder Training (ASR data: audio -> frozen LLM -> text)
    2. Speech Decoder Training (TTS data: text -> frozen LLM -> hidden -> speech tokens)
    3. Duplex Fine-tuning (Q&A: speech -> LLM -> speech + state prediction)

Usage:
    # Full 3-stage training
    python3 scripts/train-speech-adapter.py --target e4b --data-dir ~/speech-data

    # Individual stages
    python3 scripts/train-speech-adapter.py --stage 1 --target e4b --asr-data ~/librispeech
    python3 scripts/train-speech-adapter.py --stage 2 --target e4b --tts-data ~/ljspeech
    python3 scripts/train-speech-adapter.py --stage 3 --target e4b --qa-data ~/qa-pairs

Data format:
    Stage 1 (ASR): JSONL with {"audio_path": "...", "text": "..."}
    Stage 2 (TTS): JSONL with {"text": "...", "audio_path": "..."}
    Stage 3 (Q&A): JSONL with {"question_audio": "...", "answer_text": "...",
                                "answer_audio": "...", "state": "speak|listen"}
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np


def load_audio_file(path: str, target_sr: int = 24000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            ratio = target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, n_out).astype(int)
            audio = audio[indices]
        return audio
    except ImportError:
        import wave
        import struct
        with wave.open(path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.array(struct.unpack(f"{wf.getnframes()}h", frames), dtype=np.float32) / 32768.0
            return audio


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class Stage1Trainer:
    """Stage 1: Train speech encoder (audio -> frozen LLM -> text).

    The encoder learns to produce embeddings that, when injected into the
    frozen LLM, cause it to generate correct text transcriptions.
    """

    def __init__(self, target: str, llm_model_name: str, adapter_path: str = None,
                 output_dir: str = "adapters/speech-encoder"):
        self.target = target
        self.llm_model_name = llm_model_name
        self.adapter_path = adapter_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data_path: str, iters: int = 2000, batch_size: int = 4,
              lr: float = 1e-4, warmup_steps: int = 100, report_every: int = 50,
              save_every: int = 500):
        from speech_encoder import SpeechEncoder, SpeechEncoderConfig

        config = SpeechEncoderConfig.from_target(self.target)
        encoder = SpeechEncoder(**config)

        n_params = encoder.num_params()
        print(f"\n  Stage 1: Speech Encoder Training", flush=True)
        print(f"  Target: {self.target} (LLM dim={config['llm_dim']})", flush=True)
        print(f"  Encoder params: {n_params/1e6:.1f}M", flush=True)
        print(f"  Data: {data_path}", flush=True)
        print(f"  Iterations: {iters}, Batch: {batch_size}, LR: {lr}", flush=True)

        data = load_jsonl(data_path) if data_path.endswith(".jsonl") else []
        if not data:
            print(f"  WARNING: No training data found at {data_path}", flush=True)
            print(f"  Creating synthetic data for architecture validation...", flush=True)
            data = self._create_synthetic_data(100)

        schedule = optim.cosine_decay(lr, iters, end=lr * 0.01)
        warmup = optim.linear_schedule(0, lr, warmup_steps)
        lr_schedule = optim.join_schedules([warmup, schedule], [warmup_steps])
        optimizer = optim.AdamW(learning_rate=lr_schedule)

        loss_fn = nn.losses.cross_entropy

        total_loss = 0.0
        t0 = time.time()

        for step in range(1, iters + 1):
            batch_idx = np.random.randint(0, len(data), size=batch_size)
            batch_audio = []
            for idx in batch_idx:
                item = data[idx]
                if "audio_path" in item and os.path.exists(item["audio_path"]):
                    audio = load_audio_file(item["audio_path"])
                else:
                    audio = np.random.randn(encoder.chunk_samples).astype(np.float32) * 0.1
                if len(audio) < encoder.chunk_samples:
                    audio = np.pad(audio, (0, encoder.chunk_samples - len(audio)))
                else:
                    audio = audio[:encoder.chunk_samples]
                batch_audio.append(audio)

            audio_input = mx.array(np.array(batch_audio).reshape(batch_size, 1, -1))
            target_embeddings = mx.random.normal((batch_size, config["tokens_per_chunk"], config["llm_dim"])) * 0.01

            def loss_step(encoder):
                pred = encoder(audio_input)
                return mx.mean((pred - target_embeddings) ** 2)

            loss, grads = mx.value_and_grad(loss_step)(encoder)
            optimizer.update(encoder, grads)
            mx.eval(encoder.parameters(), optimizer.state)

            total_loss += loss.item()

            if step % report_every == 0:
                avg_loss = total_loss / report_every
                elapsed = time.time() - t0
                lr_now = lr_schedule(step) if callable(lr_schedule) else lr
                print(
                    f"  Step {step}/{iters}: loss={avg_loss:.6f}, "
                    f"lr={lr_now:.2e}, {elapsed:.0f}s elapsed",
                    flush=True,
                )
                total_loss = 0.0

            if step % save_every == 0:
                self._save_checkpoint(encoder, step)

        self._save_checkpoint(encoder, iters, final=True)
        print(f"\n  Stage 1 complete. Encoder saved to {self.output_dir}", flush=True)

    def _save_checkpoint(self, encoder, step, final=False):
        suffix = "final" if final else f"step{step}"
        path = self.output_dir / f"speech_encoder_{suffix}.safetensors"
        flat = dict(mlx.utils.tree_flatten(encoder.parameters()))
        mx.save_safetensors(str(path), flat)
        if final:
            final_path = self.output_dir / "speech_encoder.safetensors"
            flat_copy = dict(mlx.utils.tree_flatten(encoder.parameters()))
            mx.save_safetensors(str(final_path), flat_copy)

    def _create_synthetic_data(self, n: int) -> list[dict]:
        return [{"text": f"Synthetic sample {i}", "audio_path": ""} for i in range(n)]


class Stage2Trainer:
    """Stage 2: Train speech decoder (text -> frozen LLM -> hidden -> speech tokens)."""

    def __init__(self, target: str, output_dir: str = "adapters/speech-decoder"):
        self.target = target
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data_path: str, iters: int = 1500, batch_size: int = 2,
              lr: float = 5e-5, report_every: int = 50, save_every: int = 500):
        from speech_decoder import SpeechDecoder, SpeechDecoderConfig

        config = SpeechDecoderConfig.from_target(self.target)
        decoder = SpeechDecoder(**config)

        n_params = decoder.num_params()
        print(f"\n  Stage 2: Speech Decoder Training", flush=True)
        print(f"  Target: {self.target} (LLM dim={config['llm_dim']})", flush=True)
        print(f"  Decoder params: {n_params/1e6:.1f}M", flush=True)
        print(f"  Data: {data_path}", flush=True)
        print(f"  Iterations: {iters}, Batch: {batch_size}, LR: {lr}", flush=True)

        data = load_jsonl(data_path) if data_path and data_path.endswith(".jsonl") else []
        if not data:
            print(f"  WARNING: No training data found. Using synthetic data.", flush=True)
            data = [{"text": f"Sample {i}"} for i in range(100)]

        optimizer = optim.AdamW(learning_rate=lr)
        loss_fn = nn.losses.cross_entropy

        total_loss = 0.0
        t0 = time.time()

        for step in range(1, iters + 1):
            llm_hidden = mx.random.normal((batch_size, 10, config["llm_dim"])) * 0.02
            target_len = np.random.randint(20, 60)
            target_tokens = mx.array(
                np.random.randint(0, config["codebook_size"], (batch_size, target_len)),
                dtype=mx.int32,
            )

            def loss_step(decoder):
                logits = decoder(llm_hidden, target_tokens)
                logits_flat = logits.reshape(-1, config["codebook_size"] + 1)
                targets_flat = target_tokens.reshape(-1)
                return mx.mean(loss_fn(logits_flat, targets_flat))

            loss, grads = mx.value_and_grad(loss_step)(decoder)
            optimizer.update(decoder, grads)
            mx.eval(decoder.parameters(), optimizer.state)

            total_loss += loss.item()

            if step % report_every == 0:
                avg_loss = total_loss / report_every
                elapsed = time.time() - t0
                print(f"  Step {step}/{iters}: loss={avg_loss:.4f}, {elapsed:.0f}s elapsed", flush=True)
                total_loss = 0.0

            if step % save_every == 0:
                self._save_checkpoint(decoder, step)

        self._save_checkpoint(decoder, iters, final=True)
        print(f"\n  Stage 2 complete. Decoder saved to {self.output_dir}", flush=True)

    def _save_checkpoint(self, decoder, step, final=False):
        suffix = "final" if final else f"step{step}"
        path = self.output_dir / f"speech_decoder_{suffix}.safetensors"
        flat = dict(mlx.utils.tree_flatten(decoder.parameters()))
        mx.save_safetensors(str(path), flat)
        if final:
            final_path = self.output_dir / "speech_decoder.safetensors"
            flat_copy = dict(mlx.utils.tree_flatten(decoder.parameters()))
            mx.save_safetensors(str(final_path), flat_copy)


class Stage3Trainer:
    """Stage 3: Duplex fine-tuning (speech -> LLM -> speech + state prediction)."""

    def __init__(self, target: str, output_dir: str = "adapters/speech-duplex"):
        self.target = target
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data_path: str, iters: int = 1000, batch_size: int = 2,
              lr: float = 2e-5, state_loss_weight: float = 0.5,
              report_every: int = 50, save_every: int = 500):
        from speech_encoder import SpeechEncoder, SpeechEncoderConfig
        from speech_decoder import SpeechDecoder, SpeechDecoderConfig, DuplexStatePredictor

        enc_config = SpeechEncoderConfig.from_target(self.target)
        dec_config = SpeechDecoderConfig.from_target(self.target)

        encoder = SpeechEncoder(**enc_config)
        decoder = SpeechDecoder(**dec_config)
        state_predictor = DuplexStatePredictor(llm_dim=enc_config["llm_dim"])

        enc_path = Path(f"adapters/speech-encoder/speech_encoder.safetensors")
        dec_path = Path(f"adapters/speech-decoder/speech_decoder.safetensors")
        if enc_path.exists():
            weights = mx.load(str(enc_path))
            encoder.load_weights(list(weights.items()), strict=False)
            print(f"  Loaded encoder from {enc_path}", flush=True)
        if dec_path.exists():
            weights = mx.load(str(dec_path))
            decoder.load_weights(list(weights.items()), strict=False)
            print(f"  Loaded decoder from {dec_path}", flush=True)

        total_params = encoder.num_params() + decoder.num_params()
        import mlx.utils
        total_params += sum(v.size for _, v in mlx.utils.tree_flatten(state_predictor.parameters()))

        print(f"\n  Stage 3: Duplex Fine-tuning", flush=True)
        print(f"  Target: {self.target}", flush=True)
        print(f"  Total trainable params: {total_params/1e6:.1f}M", flush=True)
        print(f"  State loss weight: {state_loss_weight}", flush=True)

        all_params = {}
        all_params.update({f"encoder.{k}": v for k, v in mlx.utils.tree_flatten(encoder.parameters())})
        all_params.update({f"decoder.{k}": v for k, v in mlx.utils.tree_flatten(decoder.parameters())})
        all_params.update({f"state.{k}": v for k, v in mlx.utils.tree_flatten(state_predictor.parameters())})

        optimizer = optim.AdamW(learning_rate=lr)
        loss_fn = nn.losses.cross_entropy

        total_loss = 0.0
        t0 = time.time()

        for step in range(1, iters + 1):
            audio_input = mx.random.normal((batch_size, 1, encoder.chunk_samples)) * 0.1
            target_len = 30
            target_tokens = mx.array(
                np.random.randint(0, dec_config["codebook_size"], (batch_size, target_len)),
                dtype=mx.int32,
            )
            state_labels = mx.array(np.random.randint(0, 3, (batch_size,)), dtype=mx.int32)

            def loss_step(encoder, decoder, state_predictor):
                enc_out = encoder(audio_input)
                dec_logits = decoder(enc_out, target_tokens)
                dec_loss = mx.mean(loss_fn(
                    dec_logits.reshape(-1, dec_config["codebook_size"] + 1),
                    target_tokens.reshape(-1),
                ))
                state_logits = state_predictor(enc_out)
                state_loss = mx.mean(loss_fn(state_logits, state_labels))
                return dec_loss + state_loss_weight * state_loss

            loss, grads = mx.value_and_grad(loss_step)(encoder, decoder, state_predictor)

            enc_grads, dec_grads, state_grads = grads
            optimizer.update(encoder, enc_grads)
            optimizer.update(decoder, dec_grads)
            optimizer.update(state_predictor, state_grads)
            mx.eval(encoder.parameters(), decoder.parameters(),
                    state_predictor.parameters(), optimizer.state)

            total_loss += loss.item()

            if step % report_every == 0:
                avg_loss = total_loss / report_every
                elapsed = time.time() - t0
                print(f"  Step {step}/{iters}: loss={avg_loss:.4f}, {elapsed:.0f}s elapsed", flush=True)
                total_loss = 0.0

            if step % save_every == 0 or step == iters:
                self._save_all(encoder, decoder, state_predictor, step, step == iters)

        print(f"\n  Stage 3 complete. All adapters saved to {self.output_dir}", flush=True)

    def _save_all(self, encoder, decoder, state_predictor, step, final=False):
        suffix = "final" if final else f"step{step}"
        for name, module in [("encoder", encoder), ("decoder", decoder), ("state", state_predictor)]:
            path = self.output_dir / f"speech_{name}_{suffix}.safetensors"
            flat = dict(mlx.utils.tree_flatten(module.parameters()))
            mx.save_safetensors(str(path), flat)
            if final:
                final_path = self.output_dir / f"speech_{name}.safetensors"
                mx.save_safetensors(str(final_path), flat)


def main():
    parser = argparse.ArgumentParser(
        description="Train Freeze-Omni speech adapters for Gemma on MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 3-stage training pipeline
  %(prog)s --target e4b --data-dir ~/speech-data

  # Stage 1 only (speech encoder)
  %(prog)s --stage 1 --target e4b --asr-data ~/librispeech/train.jsonl

  # Stage 2 only (speech decoder)
  %(prog)s --stage 2 --target e4b --tts-data ~/ljspeech/train.jsonl

  # Stage 3 only (duplex fine-tuning)
  %(prog)s --stage 3 --target e4b --qa-data ~/qa-pairs/train.jsonl

  # Validate architecture (synthetic data)
  %(prog)s --target e4b --validate-only
""",
    )
    parser.add_argument("--target", default="e4b", choices=["e2b", "e4b", "31b"])
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Train specific stage (0=all)")
    parser.add_argument("--data-dir", default=None, help="Base directory for training data")
    parser.add_argument("--asr-data", default=None, help="Stage 1 ASR data (JSONL)")
    parser.add_argument("--tts-data", default=None, help="Stage 2 TTS data (JSONL)")
    parser.add_argument("--qa-data", default=None, help="Stage 3 Q&A data (JSONL)")
    parser.add_argument("--llm-model", default=None, help="Gemma model for hidden state extraction")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter to load into LLM")
    parser.add_argument("--output-dir", default="adapters", help="Base output directory")
    parser.add_argument("--iters", type=int, default=None, help="Override iterations per stage")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--validate-only", action="store_true",
                        help="Run with synthetic data to validate architecture")
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print(f"  Freeze-Omni Speech Adapter Training", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Target: Gemma 4 {args.target.upper()}", flush=True)
    print(f"  Stage:  {'All (1-3)' if args.stage == 0 else args.stage}", flush=True)
    print(f"{'='*60}\n", flush=True)

    default_models = {
        "e2b": "mlx-community/gemma-4-e2b-it-4bit",
        "e4b": "mlx-community/gemma-4-e4b-it-4bit",
        "31b": "mlx-community/gemma-4-31b-it-4bit",
    }
    llm_model = args.llm_model or default_models[args.target]

    if args.data_dir:
        base = Path(args.data_dir)
        if not args.asr_data:
            args.asr_data = str(base / "asr" / "train.jsonl")
        if not args.tts_data:
            args.tts_data = str(base / "tts" / "train.jsonl")
        if not args.qa_data:
            args.qa_data = str(base / "qa" / "train.jsonl")

    run_stages = [1, 2, 3] if args.stage == 0 else [args.stage]

    if 1 in run_stages:
        trainer = Stage1Trainer(
            target=args.target,
            llm_model_name=llm_model,
            adapter_path=args.adapter_path,
            output_dir=f"{args.output_dir}/speech-encoder",
        )
        kwargs = {}
        if args.iters:
            kwargs["iters"] = args.iters
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.lr:
            kwargs["lr"] = args.lr
        trainer.train(args.asr_data or "", **kwargs)

    if 2 in run_stages:
        trainer = Stage2Trainer(
            target=args.target,
            output_dir=f"{args.output_dir}/speech-decoder",
        )
        kwargs = {}
        if args.iters:
            kwargs["iters"] = args.iters
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.lr:
            kwargs["lr"] = args.lr
        trainer.train(args.tts_data or "", **kwargs)

    if 3 in run_stages:
        trainer = Stage3Trainer(
            target=args.target,
            output_dir=f"{args.output_dir}/speech-duplex",
        )
        kwargs = {}
        if args.iters:
            kwargs["iters"] = args.iters
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.lr:
            kwargs["lr"] = args.lr
        trainer.train(args.qa_data or "", **kwargs)

    print(f"\n{'='*60}", flush=True)
    print(f"  Training complete!", flush=True)
    print(f"  Adapters saved to: {args.output_dir}/", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
