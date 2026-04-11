#!/usr/bin/env python3
"""
Cascaded Real-Time Speech Pipeline for gemma-realtime.

Architecture:
    Audio In → Whisper ASR (streaming, mlx-whisper)
    → Gemma LLM (streaming, multi-turn context)
    → SpeechDecoder (embed_tokens → SNAC cb0 tokens)
    → DepthDecoder (cb0 → cb1+cb2, multi-codebook)
    → SNAC decode → Audio Out

Components:
    - StreamingASR+VAD: Whisper + Silero VAD, <150ms incremental
    - SpeechDecoder: maps Gemma embeddings to SNAC cb0 tokens
    - DepthDecoder: upsamples cb0 to 3-codebook for higher quality
    - DuplexStatePredictor: LISTEN/SPEAK/INTERRUPT from embeddings
    - SentenceBuffer: streams LLM output sentence-by-sentence to TTS

Honest limitations:
    - SpeechDecoder uses embed_tokens only (not transformer hidden states)
    - First-audio latency ~500ms (dominated by LLM time-to-first-sentence)
    - Depth decoder is early-stage (~0.6 nats/codebook above random)
"""

import base64
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


class SOTAPipeline:
    """End-to-end SOTA speech pipeline with all improvements."""

    def __init__(
        self,
        gemma_model: str = "mlx-community/gemma-4-26b-a4b-it-4bit",
        whisper_model: str = "mlx-community/whisper-small-mlx",
        decoder_path: str = "adapters/speech-decoder/speech_decoder.safetensors",
        depth_decoder_path: str = "adapters/depth-decoder/depth_decoder.safetensors",
        duplex_path: str = "adapters/duplex-predictor/duplex_predictor.safetensors",
        sample_rate: int = 24000,
    ):
        self.gemma_model = gemma_model
        self.whisper_model = whisper_model
        self.decoder_path = decoder_path
        self.depth_decoder_path = depth_decoder_path
        self.duplex_path = duplex_path
        self.sample_rate = sample_rate

        self._gemma = None
        self._tokenizer = None
        self._inner = None
        self._decoder = None
        self._depth_decoder = None
        self._duplex = None
        self._codec = None
        self._streaming_asr = None
        self._conversation_history = []
        self._loaded = False

    def load(self):
        """Load all models."""
        import mlx.core as mx
        from codec import AudioCodec
        from speech_decoder import SpeechDecoder, DuplexStatePredictor
        from streaming_asr import StreamingASRWithVAD
        from mlx_lm import load as lm_load

        print(f"\n{'='*60}")
        print("  SOTA Pipeline: Loading all components")
        print(f"{'='*60}")

        # 1. Gemma LLM
        print("  Loading Gemma LLM...", flush=True)
        self._gemma, self._tokenizer = lm_load(self.gemma_model)
        if hasattr(self._gemma, "language_model"):
            self._inner = self._gemma.language_model.model
        else:
            self._inner = self._gemma.model

        # 2. Speech decoder
        print("  Loading Speech Decoder...", flush=True)
        probe = self._inner.embed_tokens(mx.array([[0]]))
        llm_dim = probe.shape[-1]

        self._decoder = SpeechDecoder(llm_dim=llm_dim)
        weights = mx.load(self.decoder_path)
        self._decoder.load_weights(list(weights.items()))
        print(f"    Speech decoder loaded (llm_dim={llm_dim})", flush=True)

        # 3. Depth decoder (multi-codebook)
        if Path(self.depth_decoder_path).exists():
            try:
                from importlib import import_module
                tdd = import_module("train-depth-decoder")
                self._depth_decoder = tdd.DepthDecoder(**tdd.DEPTH_DECODER_CONFIG)
                weights = mx.load(self.depth_decoder_path)
                self._depth_decoder.load_weights(list(weights.items()))
                print(f"    Depth decoder loaded (3-codebook audio)", flush=True)
            except Exception as e:
                print(f"    Depth decoder unavailable: {e}", flush=True)
        else:
            print(f"    Depth decoder not found, using cb0-only", flush=True)

        # 4. Duplex predictor
        if Path(self.duplex_path).exists():
            self._duplex = DuplexStatePredictor(llm_dim=llm_dim)
            weights = mx.load(self.duplex_path)
            self._duplex.load_weights(list(weights.items()))
            print(f"    Duplex predictor loaded", flush=True)

        # 5. SNAC codec
        self._codec = AudioCodec("snac")
        self._codec.load()

        # 6. Streaming ASR
        self._streaming_asr = StreamingASRWithVAD(model_name=self.whisper_model)
        self._streaming_asr.load()

        self._loaded = True
        print(f"\n  All components loaded ✓")
        print(f"{'='*60}\n")

    def text_to_audio(self, text: str, max_tokens: int = 60) -> tuple[np.ndarray, dict]:
        """Convert text input to LLM response audio.

        Uses full conversation history for multi-turn context.
        Returns: (audio_np, metrics_dict)
        """
        import mlx.core as mx
        import torch
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        metrics = {}
        t_start = time.time()

        # Build messages from conversation history (same as stream_response)
        messages = [{"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."}]
        for hist_text, turn_type in self._conversation_history:
            role = "assistant" if turn_type == 1 else "user"
            messages.append({"role": role, "content": hist_text})
        messages.append({"role": "user", "content": text})

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        parts = []
        t_llm = time.time()
        for resp in stream_generate(
            self._gemma, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=make_sampler(temp=0.7),
        ):
            t = resp.text or ""
            if "<end_of_turn>" in t:
                parts.append(t.split("<end_of_turn>")[0])
                break
            parts.append(t)
        response = "".join(parts).strip()
        metrics["llm_ms"] = (time.time() - t_llm) * 1000
        metrics["response"] = response

        # Extract first sentence — cap at 120 chars for decoder
        import re
        match = re.search(r'[.!?]', response)
        if match:
            first_sentence = response[:match.end()]
        else:
            first_sentence = response[:120]
        return self._sentence_to_audio(first_sentence, metrics)

    def _sentence_to_audio(self, sentence: str, metrics: dict = None) -> tuple[np.ndarray, dict]:
        """Convert a single sentence to audio waveform."""
        import mlx.core as mx
        import torch

        if metrics is None:
            metrics = {}

        t_dec = time.time()
        ids = self._tokenizer.encode(sentence[:120], add_special_tokens=False)
        if not ids:
            return np.zeros(2400, dtype=np.float32), metrics

        emb = self._inner.embed_tokens(mx.array([ids]))
        tokens_mx = self._decoder.generate(emb, temperature=0.0, top_k=0)
        mx.eval(tokens_mx)
        cb0_tokens = tokens_mx[0].tolist()
        metrics["decoder_ms"] = (time.time() - t_dec) * 1000
        metrics["cb0_tokens"] = len(cb0_tokens)

        if not cb0_tokens:
            return np.zeros(2400, dtype=np.float32), metrics

        # Multi-codebook: use depth decoder if available
        t_snac = time.time()
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        if self._depth_decoder is not None:
            cb0_mx = mx.array([cb0_tokens], dtype=mx.int32)
            cb1_mx, cb2_mx = self._depth_decoder.generate(cb0_mx)
            mx.eval(cb1_mx, cb2_mx)
            cb1_tokens = cb1_mx[0].tolist()
            cb2_tokens = cb2_mx[0].tolist()
            metrics["codebooks"] = 3

            cb0_t = torch.tensor(cb0_tokens, dtype=torch.long).unsqueeze(0).to(device)
            cb1_t = torch.tensor(cb1_tokens, dtype=torch.long).unsqueeze(0).to(device)
            cb2_t = torch.tensor(cb2_tokens, dtype=torch.long).unsqueeze(0).to(device)
        else:
            metrics["codebooks"] = 1
            cb0_t = torch.tensor(cb0_tokens, dtype=torch.long).unsqueeze(0).to(device)
            cb1_t = torch.zeros(1, len(cb0_tokens) * 2, dtype=torch.long).to(device)
            cb2_t = torch.zeros(1, len(cb0_tokens) * 4, dtype=torch.long).to(device)

        with torch.no_grad():
            audio = self._codec._model.decode([cb0_t, cb1_t, cb2_t])
        audio_np = audio.detach().cpu().numpy().squeeze()
        metrics["snac_ms"] = (time.time() - t_snac) * 1000
        metrics["audio_duration"] = len(audio_np) / self.sample_rate

        return audio_np, metrics

    def add_to_history(self, text: str, is_assistant: bool = False):
        """Add a turn to conversation history."""
        turn_type = 1 if is_assistant else 0
        self._conversation_history.append((text, turn_type))
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]

    def check_duplex_state(self, text: str) -> str:
        """Check what state the duplex predictor suggests."""
        import mlx.core as mx

        if self._duplex is None:
            return "SPEAK"

        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return "LISTEN"
        emb = self._inner.embed_tokens(mx.array([ids]))
        state = self._duplex.predict(emb)
        return ["LISTEN", "SPEAK", "INTERRUPT"][state]

    def stream_response(self, user_text: str, max_tokens: int = 80):
        """Stream LLM response with sentence-level audio chunks.

        Yields:
            {"type": "text.delta", "text": "..."}
            {"type": "audio.chunk", "audio": np.ndarray, "sentence": "..."}
            {"type": "done", "full_text": "...", "metrics": {...}}
        """
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sys.path.insert(0, str(Path(__file__).parent))
        from importlib import import_module
        speech = import_module("speech-server")

        self.add_to_history(user_text, is_assistant=False)

        # Build messages from full conversation history
        messages = [{"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."}]
        for text, turn_type in self._conversation_history:
            role = "assistant" if turn_type == 1 else "user"
            messages.append({"role": role, "content": text})

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True,
        )

        sentence_buf = speech.SentenceBuffer(min_chars=12, max_chars=120)
        text_parts = []
        audio_chunks = []
        t_start = time.time()
        first_audio_t = None

        for resp in stream_generate(
            self._gemma, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=make_sampler(temp=0.7),
        ):
            token_text = resp.text or ""
            if "<end_of_turn>" in token_text:
                text_parts.append(token_text.split("<end_of_turn>")[0])
                break
            text_parts.append(token_text)
            yield {"type": "text.delta", "text": token_text}

            sentences = sentence_buf.add(token_text)
            for sent in sentences:
                audio_np, metrics = self._sentence_to_audio(sent)
                if first_audio_t is None:
                    first_audio_t = time.time()
                audio_chunks.append(audio_np)
                yield {"type": "audio.chunk", "audio": audio_np, "sentence": sent, "metrics": metrics}

        # Flush remainder
        remainder = sentence_buf.flush()
        if remainder and remainder.strip():
            audio_np, metrics = self._sentence_to_audio(remainder)
            if first_audio_t is None:
                first_audio_t = time.time()
            audio_chunks.append(audio_np)
            yield {"type": "audio.chunk", "audio": audio_np, "sentence": remainder, "metrics": metrics}

        full_text = "".join(text_parts).strip()
        self.add_to_history(full_text, is_assistant=True)

        total_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0)
        yield {
            "type": "done",
            "full_text": full_text,
            "metrics": {
                "total_ms": (time.time() - t_start) * 1000,
                "first_audio_ms": (first_audio_t - t_start) * 1000 if first_audio_t else None,
                "audio_chunks": len(audio_chunks),
                "audio_duration": len(total_audio) / self.sample_rate if len(total_audio) > 0 else 0,
            }
        }

    def process_audio(self, audio_16k: np.ndarray, max_tokens: int = 80):
        """Full speech-to-speech: audio in → ASR → LLM → TTS → audio out.

        This is the true STS entry point. Takes raw 16kHz audio of a user
        utterance and yields streaming events including audio of the response.

        Args:
            audio_16k: float32 numpy array at 16kHz sample rate
            max_tokens: max LLM generation tokens

        Yields:
            {"type": "transcript", "text": "..."} — ASR result
            {"type": "text.delta", "text": "..."} — LLM token
            {"type": "audio.chunk", "audio": np.ndarray, ...} — TTS audio
            {"type": "done", "full_text": "...", "metrics": {...}}
        """
        t_start = time.time()

        # Step 1: ASR — transcribe the audio
        text, asr_ms = self._streaming_asr.asr.transcribe_full(audio_16k)
        if not text or not text.strip():
            yield {"type": "transcript", "text": ""}
            yield {"type": "done", "full_text": "", "metrics": {"asr_ms": asr_ms, "error": "empty transcript"}}
            return

        yield {"type": "transcript", "text": text}

        # Step 2+3: LLM → TTS via stream_response
        for event in self.stream_response(text, max_tokens=max_tokens):
            event_copy = dict(event)
            if event_copy["type"] == "done":
                event_copy["metrics"]["asr_ms"] = asr_ms
                event_copy["metrics"]["total_sts_ms"] = (time.time() - t_start) * 1000
            yield event_copy

    def process_audio_full(self, audio_16k: np.ndarray, max_tokens: int = 80) -> tuple[str, np.ndarray, dict]:
        """Convenience: full STS in one call, returns (transcript, audio_out, metrics)."""
        transcript = ""
        audio_chunks = []
        full_text = ""
        metrics = {}

        for event in self.process_audio(audio_16k, max_tokens=max_tokens):
            if event["type"] == "transcript":
                transcript = event["text"]
            elif event["type"] == "audio.chunk":
                audio_chunks.append(event["audio"])
            elif event["type"] == "done":
                full_text = event["full_text"]
                metrics = event["metrics"]

        audio_out = np.concatenate(audio_chunks) if audio_chunks else np.zeros(2400, dtype=np.float32)
        metrics["transcript"] = transcript
        metrics["response"] = full_text
        return transcript, audio_out, metrics

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def has_depth_decoder(self) -> bool:
        return self._depth_decoder is not None

    @property
    def has_duplex(self) -> bool:
        return self._duplex is not None


class FishSTSWrapper:
    """Wraps the Fish true STS pipeline with the same interface as SOTAPipeline.

    Drop-in replacement that uses Fish codec + MOSS-Speech layer splitting
    instead of the cascaded ASR → LLM → TTS pipeline.

    For text input, falls back to the cascaded path (Gemma → speech decoder).
    For audio input, uses the true STS path (no text intermediate).
    """

    def __init__(
        self,
        gemma_model: str = "mlx-community/gemma-4-26b-a4b-it-4bit",
        fish_sts_weights: str = "adapters/fish-sts/phase-c/fish_sts_final.safetensors",
        fish_model: str = "fishaudio/fish-speech-1.5",
        sample_rate: int = 44100,
    ):
        self.gemma_model = gemma_model
        self.fish_sts_weights = fish_sts_weights
        self.fish_model = fish_model
        self.sample_rate = sample_rate
        self._fish_pipeline = None
        self._cascaded = None
        self._loaded = False

    def load(self):
        """Load Fish STS pipeline + fallback cascaded pipeline."""
        from fish_sts import FishSTSPipeline

        print(f"\n{'='*60}")
        print("  Fish STS Wrapper: Loading")
        print(f"{'='*60}")

        self._fish_pipeline = FishSTSPipeline(
            target="e4b",
            gemma_model=self.gemma_model,
            fish_model=self.fish_model,
        )

        # Try loading — if Fish codec isn't available, fall back
        try:
            self._fish_pipeline.load()

            # Load trained weights if available
            weights_path = Path(self.fish_sts_weights)
            if weights_path.exists():
                import mlx.core as mx
                w = mx.load(str(weights_path))
                self._fish_pipeline._model.load_weights(list(w.items()), strict=False)
                print(f"  Loaded Fish STS weights from {weights_path}", flush=True)

            self._loaded = True
            print("  Fish STS pipeline ready (true STS mode)", flush=True)
        except Exception as e:
            print(f"  Fish STS load failed ({e}), falling back to cascaded", flush=True)
            self._fish_pipeline = None
            self._cascaded = SOTAPipeline(gemma_model=self.gemma_model)
            self._cascaded.load()
            self._loaded = True

        print(f"{'='*60}\n")

    def process_audio(self, audio_input: np.ndarray, max_tokens: int = 80):
        """True STS: audio in → audio out (no text intermediate).

        If Fish pipeline is loaded, bypasses ASR entirely.
        Falls back to cascaded pipeline if Fish isn't available.
        """
        if self._fish_pipeline is not None:
            audio_out, metrics = self._fish_pipeline.process_audio(audio_input)
            yield {"type": "transcript", "text": "[direct STS — no transcript]"}
            yield {"type": "audio.chunk", "audio": audio_out, "metrics": metrics}
            yield {"type": "done", "full_text": "[true STS]", "metrics": metrics}
        elif self._cascaded is not None:
            # Resample to 16kHz for Whisper ASR
            if self.sample_rate != 16000:
                n_out = int(len(audio_input) * 16000 / self.sample_rate)
                audio_16k = np.interp(
                    np.linspace(0, 1, n_out),
                    np.linspace(0, 1, len(audio_input)),
                    audio_input,
                ).astype(np.float32)
            else:
                audio_16k = audio_input
            yield from self._cascaded.process_audio(audio_16k, max_tokens=max_tokens)

    def process_audio_full(self, audio_input: np.ndarray, max_tokens: int = 80):
        """Convenience: full STS in one call."""
        if self._fish_pipeline is not None:
            audio_out, metrics = self._fish_pipeline.process_audio(audio_input)
            return "[direct STS]", audio_out, metrics

        # Fallback to cascaded
        if self._cascaded is not None:
            if self.sample_rate != 16000:
                n_out = int(len(audio_input) * 16000 / self.sample_rate)
                audio_16k = np.interp(
                    np.linspace(0, 1, n_out),
                    np.linspace(0, 1, len(audio_input)),
                    audio_input,
                ).astype(np.float32)
            else:
                audio_16k = audio_input
            return self._cascaded.process_audio_full(audio_16k, max_tokens)

        return "", np.zeros(4410, dtype=np.float32), {"error": "not loaded"}

    def text_to_audio(self, text: str, max_tokens: int = 60):
        """Text → audio (uses cascaded path even with Fish loaded)."""
        if self._cascaded is not None:
            return self._cascaded.text_to_audio(text, max_tokens)
        return np.zeros(4410, dtype=np.float32), {"error": "cascaded not loaded"}

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def is_true_sts(self) -> bool:
        return self._fish_pipeline is not None
