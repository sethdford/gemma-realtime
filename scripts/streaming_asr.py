#!/usr/bin/env python3
"""
Streaming ASR: Chunked Whisper processing for <150ms incremental transcription.

Instead of waiting for full utterance, processes audio in overlapping chunks
and emits partial transcriptions with very low latency.

Approaches (in priority order):
    1. mlx-whisper with chunked decoding (native Apple Silicon)
    2. faster-whisper with VAD-based chunking (CPU fallback)
    3. OpenAI whisper with manual chunking (baseline)

Key technique: sliding window with overlap. Each chunk overlaps the previous
by 50% to avoid cutting words at boundaries. Partial results are emitted
immediately; final result after silence detection.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np


class StreamingASR:
    """Streaming speech-to-text with incremental output.

    Processes audio in chunks and emits partial transcriptions as they
    become available. Targets <150ms latency per chunk.
    """

    SAMPLE_RATE = 16000
    CHUNK_DURATION_S = 1.0
    OVERLAP_S = 0.5
    MIN_AUDIO_S = 0.3

    def __init__(self, model_name: str = "mlx-community/whisper-small-mlx"):
        self.model_name = model_name
        self._transcribe = None
        self._buffer = np.array([], dtype=np.float32)
        self._partial_text = ""
        self._final_text = ""
        self._backend = None

    def load(self):
        """Load the best available Whisper backend."""
        # Try mlx-whisper first (fastest on Apple Silicon)
        try:
            import mlx_whisper
            self._transcribe = self._transcribe_mlx
            self._backend = "mlx-whisper"
            # Warm up
            dummy = np.zeros(self.SAMPLE_RATE, dtype=np.float32)
            mlx_whisper.transcribe(dummy, path_or_hf_repo=self.model_name, language="en")
            print(f"  StreamingASR: mlx-whisper loaded ({self.model_name})", flush=True)
            return
        except ImportError:
            pass

        # Fall back to openai-whisper
        try:
            import whisper
            self._whisper_model = whisper.load_model("small")
            self._transcribe = self._transcribe_whisper
            self._backend = "openai-whisper"
            print("  StreamingASR: OpenAI Whisper loaded (small)", flush=True)
            return
        except ImportError:
            pass

        raise RuntimeError("No Whisper backend available. Install: pip install mlx-whisper")

    def _transcribe_mlx(self, audio: np.ndarray) -> str:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.model_name,
            language="en", fp16=True,
            condition_on_previous_text=False,
        )
        return result.get("text", "").strip()

    def _transcribe_whisper(self, audio: np.ndarray) -> str:
        result = self._whisper_model.transcribe(
            audio, language="en", fp16=False,
            condition_on_previous_text=False,
        )
        return result.get("text", "").strip()

    def feed_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Feed an audio chunk and get partial transcription if available.

        Args:
            audio_chunk: float32 audio at 16kHz

        Returns:
            Partial transcription string, or None if not enough audio yet
        """
        self._buffer = np.concatenate([self._buffer, audio_chunk])

        chunk_samples = int(self.CHUNK_DURATION_S * self.SAMPLE_RATE)
        if len(self._buffer) < chunk_samples:
            return None

        # Transcribe the current buffer
        audio_to_transcribe = self._buffer.copy()
        text = self._transcribe(audio_to_transcribe)

        # Keep overlap for next chunk
        overlap_samples = int(self.OVERLAP_S * self.SAMPLE_RATE)
        if len(self._buffer) > overlap_samples:
            self._buffer = self._buffer[-overlap_samples:]
        else:
            self._buffer = np.array([], dtype=np.float32)

        if text and text != self._partial_text:
            self._partial_text = text
            return text

        return None

    def finalize(self) -> str:
        """Transcribe any remaining audio and return final result."""
        if len(self._buffer) > int(self.MIN_AUDIO_S * self.SAMPLE_RATE):
            text = self._transcribe(self._buffer)
            if text:
                self._final_text = text
        elif self._partial_text:
            self._final_text = self._partial_text

        result = self._final_text
        self.reset()
        return result

    def reset(self):
        """Reset state for next utterance."""
        self._buffer = np.array([], dtype=np.float32)
        self._partial_text = ""
        self._final_text = ""

    @property
    def backend(self) -> str:
        return self._backend or "none"

    def transcribe_full(self, audio: np.ndarray) -> tuple[str, float]:
        """Transcribe a complete audio buffer. Returns (text, latency_ms)."""
        t0 = time.time()
        text = self._transcribe(audio)
        latency = (time.time() - t0) * 1000
        return text, latency


class StreamingASRWithVAD:
    """Streaming ASR with integrated Voice Activity Detection.

    Automatically detects speech start/end and transcribes utterances.
    Emits partial results during speech and final result after silence.
    """

    def __init__(self, model_name: str = "mlx-community/whisper-small-mlx",
                 vad_threshold: float = 0.4, silence_ms: int = 800):
        self.asr = StreamingASR(model_name)
        self.vad_threshold = vad_threshold
        self.silence_ms = silence_ms
        self._vad = None
        self._is_speaking = False
        self._silence_start = None
        self._speech_buffer = np.array([], dtype=np.float32)

    def load(self):
        self.asr.load()

        try:
            import torch
            model, utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                force_reload=False, onnx=True,
            )
            self._vad = model
            print("  StreamingASR+VAD: Silero VAD loaded", flush=True)
        except Exception as e:
            print(f"  StreamingASR+VAD: VAD unavailable ({e}), using energy-based", flush=True)

    def is_speech(self, audio_16k: np.ndarray) -> bool:
        if self._vad is not None:
            import torch
            tensor = torch.from_numpy(audio_16k.astype(np.float32))
            if tensor.abs().max() > 1.0:
                tensor = tensor / 32768.0
            conf = self._vad(tensor, 16000).item()
            return conf > self.vad_threshold

        rms = np.sqrt(np.mean(audio_16k.astype(np.float32) ** 2))
        return rms > 0.01

    def feed(self, audio_16k: np.ndarray) -> dict:
        """Feed audio and return event dict.

        Returns:
            {"type": "partial", "text": "..."} — during speech
            {"type": "final", "text": "..."} — after speech ends
            {"type": "silence"} — no speech detected
        """
        speech = self.is_speech(audio_16k)

        if speech:
            self._is_speaking = True
            self._silence_start = None
            self._speech_buffer = np.concatenate([self._speech_buffer, audio_16k])

            partial = self.asr.feed_chunk(audio_16k)
            if partial:
                return {"type": "partial", "text": partial}
            return {"type": "listening"}

        elif self._is_speaking:
            self._speech_buffer = np.concatenate([self._speech_buffer, audio_16k])

            if self._silence_start is None:
                self._silence_start = time.time()
            elif (time.time() - self._silence_start) * 1000 > self.silence_ms:
                self._is_speaking = False
                self._silence_start = None

                final_text = self.asr.finalize()
                self._speech_buffer = np.array([], dtype=np.float32)

                if final_text.strip():
                    return {"type": "final", "text": final_text}
                return {"type": "silence"}

            return {"type": "listening"}

        return {"type": "silence"}

    def reset(self):
        self._is_speaking = False
        self._silence_start = None
        self._speech_buffer = np.array([], dtype=np.float32)
        self.asr.reset()
        if self._vad and hasattr(self._vad, "reset_states"):
            self._vad.reset_states()
