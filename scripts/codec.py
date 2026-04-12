#!/usr/bin/env python3
"""
Neural audio codec abstraction for gemma-realtime.

Provides a unified interface for encoding/decoding audio using neural codecs
(SNAC, Mimi, EnCodec, or Fish DAC). These codecs convert raw audio waveforms
to discrete token sequences for language model speech generation.

Supported codecs:
    - SNAC:     Multi-scale RVQ (12/23/47 Hz), MIT license, 3 codebooks
    - Mimi:     Kyutai's codec from Moshi (12.5 Hz, 1.1 kbps, 80ms latency)
    - EnCodec:  Meta's baseline codec (75 Hz, higher bitrate)
    - Fish DAC: Fish Audio's S2 codec (8 FSQ groups × 1000, ~21 Hz, 44.1kHz, SOTA quality)
                Standing on Fish Audio's shoulders for true speech-to-speech.

Usage:
    from codec import AudioCodec

    codec = AudioCodec("snac")    # or "fish" for Fish DAC
    codec.load()

    tokens = codec.encode(audio_np)
    audio_out = codec.decode(tokens)

    # Fish DAC: 8 FSQ groups, cb0 = semantic, cb1-7 = acoustic detail
    # cb0 feeds Gemma for reasoning, Fish's Fast AR fills cb1-7
    fish = AudioCodec("fish")
    fish.load()
    tokens = fish.encode(audio_44k)  # -> (8, T) codes at ~21 Hz
    cb0_semantic = tokens.codes[0]    # primary semantic codebook

    # Streaming
    for chunk in audio_chunks:
        tokens = codec.encode_chunk(chunk)
        process(tokens)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class CodecType(Enum):
    SNAC = "snac"
    MIMI = "mimi"
    ENCODEC = "encodec"
    FISH_DAC = "fish"


@dataclass
class CodecConfig:
    """Configuration for a neural audio codec."""
    codec_type: CodecType
    sample_rate: int
    frame_rate: float
    n_codebooks: int
    codebook_size: int
    bandwidth_kbps: float
    latency_ms: float
    streaming: bool


CODEC_CONFIGS = {
    CodecType.SNAC: CodecConfig(
        codec_type=CodecType.SNAC,
        sample_rate=24000,
        frame_rate=12.0,
        n_codebooks=3,
        codebook_size=4096,
        bandwidth_kbps=2.4,
        latency_ms=42.0,
        streaming=True,
    ),
    CodecType.MIMI: CodecConfig(
        codec_type=CodecType.MIMI,
        sample_rate=24000,
        frame_rate=12.5,
        n_codebooks=8,
        codebook_size=2048,
        bandwidth_kbps=1.1,
        latency_ms=80.0,
        streaming=True,
    ),
    CodecType.ENCODEC: CodecConfig(
        codec_type=CodecType.ENCODEC,
        sample_rate=24000,
        frame_rate=75.0,
        n_codebooks=8,
        codebook_size=1024,
        bandwidth_kbps=6.0,
        latency_ms=13.3,
        streaming=False,
    ),
    CodecType.FISH_DAC: CodecConfig(
        codec_type=CodecType.FISH_DAC,
        sample_rate=44100,
        frame_rate=21.53,       # 44100 / 2048 downsampling ratio
        n_codebooks=8,          # 8 FSQ groups
        codebook_size=1000,     # prod([8,5,5,5]) per FSQ group
        bandwidth_kbps=2.5,
        latency_ms=46.4,        # 2048 / 44100 * 1000
        streaming=True,
    ),
}


@dataclass
class CodecTokens:
    """Discrete token representation from a neural audio codec."""
    codes: np.ndarray
    n_codebooks: int
    frame_rate: float
    codec_type: CodecType

    @property
    def n_frames(self) -> int:
        return self.codes.shape[-1] if self.codes.ndim > 1 else len(self.codes)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.frame_rate

    @property
    def flat_tokens(self) -> np.ndarray:
        """Flatten multi-codebook tokens into a single sequence with offsets."""
        if self.codes.ndim == 1:
            return self.codes
        flat = []
        codebook_size = CODEC_CONFIGS[self.codec_type].codebook_size
        for frame_idx in range(self.n_frames):
            for cb_idx in range(self.n_codebooks):
                token = self.codes[cb_idx, frame_idx] + cb_idx * codebook_size
                flat.append(token)
        return np.array(flat, dtype=np.int64)

    @staticmethod
    def from_flat(flat_tokens: np.ndarray, n_codebooks: int,
                  codec_type: CodecType) -> "CodecTokens":
        """Reconstruct CodecTokens from flattened sequence."""
        codebook_size = CODEC_CONFIGS[codec_type].codebook_size
        n_frames = len(flat_tokens) // n_codebooks
        codes = np.zeros((n_codebooks, n_frames), dtype=np.int64)
        for i, token in enumerate(flat_tokens[:n_frames * n_codebooks]):
            frame_idx = i // n_codebooks
            cb_idx = i % n_codebooks
            codes[cb_idx, frame_idx] = token - cb_idx * codebook_size
        return CodecTokens(
            codes=codes,
            n_codebooks=n_codebooks,
            frame_rate=CODEC_CONFIGS[codec_type].frame_rate,
            codec_type=codec_type,
        )


class AudioCodec:
    """Unified neural audio codec interface."""

    def __init__(self, codec_type: str = "snac", device: str = "mps",
                 fish_model: str = "fishaudio/fish-speech-1.5"):
        self.codec_type = CodecType(codec_type)
        self.device = device
        self.config = CODEC_CONFIGS[self.codec_type]
        self._model = None
        self._loaded = False
        self._encode_buffer = np.array([], dtype=np.float32)
        self._chunk_samples = int(self.config.sample_rate / self.config.frame_rate)
        self._fish_model_id = fish_model

    def load(self):
        """Load the codec model."""
        if self.codec_type == CodecType.SNAC:
            self._load_snac()
        elif self.codec_type == CodecType.MIMI:
            self._load_mimi()
        elif self.codec_type == CodecType.ENCODEC:
            self._load_encodec()
        elif self.codec_type == CodecType.FISH_DAC:
            self._load_fish_dac()

    def _load_snac(self):
        try:
            import torch
            from snac import SNAC
            self._model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            if self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._loaded = True
            print(f"  Codec: SNAC loaded ({self.config.n_codebooks} codebooks, {self.config.bandwidth_kbps} kbps)", flush=True)
        except ImportError:
            print("  Codec: SNAC not available — install: pip install snac", flush=True)
            raise

    def _load_mimi(self):
        try:
            from transformers import MimiModel, AutoFeatureExtractor
            self._model = MimiModel.from_pretrained("kyutai/mimi")
            self._feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
            self._loaded = True
            print(f"  Codec: Mimi loaded ({self.config.bandwidth_kbps} kbps, {self.config.latency_ms}ms latency)", flush=True)
        except ImportError:
            print("  Codec: Mimi not available — install: pip install transformers", flush=True)
            raise

    def _load_encodec(self):
        try:
            from transformers import EncodecModel, AutoProcessor
            self._model = EncodecModel.from_pretrained("facebook/encodec_24khz")
            self._processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
            self._loaded = True
            print(f"  Codec: EnCodec loaded ({self.config.bandwidth_kbps} kbps)", flush=True)
        except ImportError:
            print("  Codec: EnCodec not available — install: pip install transformers", flush=True)
            raise

    def _load_fish_dac(self):
        """Load Fish Audio's DAC-based codec (8 FSQ groups, 44.1kHz, ~21 Hz).

        Tries two loading paths:
          1. fish_speech package (pip install fish-speech)
          2. descript-audio-codec with Fish's pretrained weights from HuggingFace
        """
        # Path 1: fish_speech native package
        try:
            from fish_speech.models.dac.modded_dac import DACModel
            self._model = DACModel.from_pretrained(self._fish_model_id)
            self._model.eval()
            self._fish_backend = "fish_speech"
            self._loaded = True
            print(f"  Codec: Fish DAC loaded via fish_speech ({self.config.n_codebooks} codebooks, "
                  f"{self.config.sample_rate}Hz, ~{self.config.frame_rate:.0f} Hz frame rate)", flush=True)
            return
        except ImportError:
            pass

        # Path 2: HuggingFace + descript-audio-codec
        try:
            import torch
            import dac
            from huggingface_hub import hf_hub_download

            generator_path = hf_hub_download(
                repo_id=self._fish_model_id,
                filename="firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            )
            model = dac.DAC.load(generator_path)
            model.eval()
            if self.device == "mps" and torch.backends.mps.is_available():
                model = model.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")

            self._model = model
            self._fish_backend = "dac"
            n_cb = getattr(model, "n_codebooks", self.config.n_codebooks)
            cb_size = getattr(model, "codebook_size", self.config.codebook_size)
            self.config = CodecConfig(
                codec_type=CodecType.FISH_DAC,
                sample_rate=self.config.sample_rate,
                frame_rate=self.config.frame_rate,
                n_codebooks=n_cb,
                codebook_size=cb_size,
                bandwidth_kbps=self.config.bandwidth_kbps,
                latency_ms=self.config.latency_ms,
                streaming=True,
            )
            self._loaded = True
            print(f"  Codec: Fish DAC loaded via descript-audio-codec ({n_cb} codebooks, "
                  f"{self.config.sample_rate}Hz)", flush=True)
            return
        except (ImportError, Exception) as e:
            pass

        # Path 3: Standalone Fish DAC loader (reconstructs FireflyArchitecture)
        try:
            from fish_dac_loader import load_fish_dac
            from huggingface_hub import hf_hub_download

            generator_path = hf_hub_download(
                repo_id=self._fish_model_id,
                filename="firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            )
            self._model = load_fish_dac(generator_path, device=self.device)
            self._fish_backend = "firefly"
            self.config = CodecConfig(
                codec_type=CodecType.FISH_DAC,
                sample_rate=44100,
                frame_rate=21.5,
                n_codebooks=8,
                codebook_size=1024,
                bandwidth_kbps=self.config.bandwidth_kbps,
                latency_ms=self.config.latency_ms,
                streaming=True,
            )
            self._loaded = True
            print(f"  Codec: Fish DAC loaded via FireflyArchitecture "
                  f"({self.config.n_codebooks} codebooks, {self.config.sample_rate}Hz, "
                  f"~{self.config.frame_rate:.0f} Hz)", flush=True)
            return
        except (ImportError, Exception) as e:
            pass

        # Path 4: Raw state dict fallback (no encode/decode — will trigger SNAC fallback upstream)
        try:
            import torch
            from huggingface_hub import hf_hub_download

            generator_path = hf_hub_download(
                repo_id=self._fish_model_id,
                filename="firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            )
            self._model = torch.load(generator_path, map_location="cpu", weights_only=False)
            if hasattr(self._model, "eval"):
                self._model.eval()
            self._fish_backend = "raw_torch"
            self._loaded = True
            print(f"  Codec: Fish DAC loaded from raw weights ({self.config.n_codebooks} codebooks, no encode/decode)", flush=True)
            return
        except Exception as e:
            print(f"  Codec: Fish DAC failed to load: {e}", flush=True)
            print("  Install: pip install vector_quantize_pytorch einops torchaudio", flush=True)
            raise RuntimeError(f"Fish DAC codec not available: {e}")

    def encode(self, audio: np.ndarray) -> CodecTokens:
        """Encode a complete audio waveform to discrete tokens."""
        if not self._loaded:
            raise RuntimeError("Codec not loaded — call .load() first")

        if audio.size == 0:
            return CodecTokens(
                codes=np.zeros((self.config.n_codebooks, 0), dtype=np.int64),
                n_codebooks=self.config.n_codebooks,
                frame_rate=self.config.frame_rate,
                codec_type=self.codec_type,
            )

        min_samples = int(self.config.sample_rate * 0.01)  # 10ms minimum
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        audio = self._normalize(audio)

        if self.codec_type == CodecType.SNAC:
            return self._encode_snac(audio)
        elif self.codec_type == CodecType.MIMI:
            return self._encode_mimi(audio)
        elif self.codec_type == CodecType.ENCODEC:
            return self._encode_encodec(audio)
        elif self.codec_type == CodecType.FISH_DAC:
            return self._encode_fish_dac(audio)

    def decode(self, tokens: CodecTokens) -> np.ndarray:
        """Decode discrete tokens back to audio waveform."""
        if not self._loaded:
            raise RuntimeError("Codec not loaded — call .load() first")

        if self.codec_type == CodecType.SNAC:
            return self._decode_snac(tokens)
        elif self.codec_type == CodecType.MIMI:
            return self._decode_mimi(tokens)
        elif self.codec_type == CodecType.ENCODEC:
            return self._decode_encodec(tokens)
        elif self.codec_type == CodecType.FISH_DAC:
            return self._decode_fish_dac(tokens)

    def encode_chunk(self, audio_chunk: np.ndarray) -> Optional[CodecTokens]:
        """Streaming encode: buffer audio and encode when enough samples collected."""
        audio_chunk = self._normalize(audio_chunk)
        self._encode_buffer = np.concatenate([self._encode_buffer, audio_chunk])

        min_samples = self._chunk_samples * 4
        if len(self._encode_buffer) < min_samples:
            return None

        n_frames = len(self._encode_buffer) // self._chunk_samples
        n_samples = n_frames * self._chunk_samples
        to_encode = self._encode_buffer[:n_samples]
        self._encode_buffer = self._encode_buffer[n_samples:]

        return self.encode(to_encode)

    def flush_encode(self) -> Optional[CodecTokens]:
        """Flush remaining buffered audio."""
        if len(self._encode_buffer) < self._chunk_samples:
            self._encode_buffer = np.array([], dtype=np.float32)
            return None

        padded = np.pad(
            self._encode_buffer,
            (0, self._chunk_samples - len(self._encode_buffer) % self._chunk_samples),
        )
        self._encode_buffer = np.array([], dtype=np.float32)
        return self.encode(padded)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        return audio

    def _encode_snac(self, audio: np.ndarray) -> CodecTokens:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
            if self.device == "mps" and torch.backends.mps.is_available():
                x = x.to("mps")
            codes = self._model.encode(x)
            if isinstance(codes, (list, tuple)):
                codes_np = [np.atleast_1d(c.cpu().numpy().squeeze()) for c in codes]
                self._snac_codes_raw = codes_np
                max_len = max(len(c) for c in codes_np)
                aligned = np.zeros((len(codes_np), max_len), dtype=np.int64)
                for i, c in enumerate(codes_np):
                    aligned[i, :len(c)] = c
                return CodecTokens(
                    codes=aligned,
                    n_codebooks=len(codes_np),
                    frame_rate=self.config.frame_rate,
                    codec_type=self.codec_type,
                )
            else:
                codes_np = codes.cpu().numpy().squeeze()
                if codes_np.ndim == 1:
                    codes_np = codes_np.reshape(1, -1)
                return CodecTokens(
                    codes=codes_np,
                    n_codebooks=codes_np.shape[0],
                    frame_rate=self.config.frame_rate,
                    codec_type=self.codec_type,
                )

    def _decode_snac(self, tokens: CodecTokens) -> np.ndarray:
        import torch
        with torch.no_grad():
            if hasattr(self, '_snac_codes_raw') and self._snac_codes_raw is not None:
                codes_list = [torch.from_numpy(c).unsqueeze(0).long() for c in self._snac_codes_raw]
            elif tokens.codes.ndim == 1:
                codes_list = [torch.from_numpy(tokens.codes).unsqueeze(0).long()]
            else:
                codes_list = [torch.from_numpy(tokens.codes[i]).unsqueeze(0).long()
                              for i in range(tokens.n_codebooks)]

            # SNAC 24kHz expects hierarchical temporal resolution (1:2:4).
            # If all codebooks are the same length (e.g. from depth decoder),
            # upsample cb1 by 2x and cb2 by 4x via repeat_interleave.
            if (len(codes_list) == 3
                    and codes_list[0].shape[-1] == codes_list[1].shape[-1]
                    and codes_list[0].shape[-1] == codes_list[2].shape[-1]):
                T = codes_list[0].shape[-1]
                codes_list[1] = codes_list[1].repeat_interleave(2, dim=-1)
                codes_list[2] = codes_list[2].repeat_interleave(4, dim=-1)

            if self.device == "mps" and torch.backends.mps.is_available():
                codes_list = [c.to("mps") for c in codes_list]

            audio = self._model.decode(codes_list)
            return audio.cpu().numpy().squeeze()

    def _encode_fish_dac(self, audio: np.ndarray) -> CodecTokens:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()
            if self.device == "mps" and torch.backends.mps.is_available():
                x = x.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                x = x.to("cuda")

            if self._fish_backend == "fish_speech":
                codes = self._model.encode(x)
                if isinstance(codes, tuple):
                    codes = codes[0]
                codes_np = codes.cpu().numpy().squeeze()
            elif self._fish_backend == "dac":
                x_processed = self._model.preprocess(x, self.config.sample_rate)
                _, codes, _, _, _ = self._model.encode(x_processed)
                codes_np = codes.cpu().numpy().squeeze()
            elif self._fish_backend == "firefly":
                # FireflyArchitecture.encode expects (B, T) audio
                if x.ndim == 3:
                    x = x.squeeze(1)
                codes = self._model.encode(x)
                codes_np = codes.cpu().numpy().squeeze()
            else:
                if hasattr(self._model, "encode"):
                    codes = self._model.encode(x)
                    if isinstance(codes, tuple):
                        codes = codes[0]
                    codes_np = codes.cpu().numpy().squeeze()
                else:
                    raise RuntimeError("Fish DAC model has no encode method")

            if codes_np.ndim == 1:
                codes_np = codes_np.reshape(1, -1)

            self._fish_codes_raw = codes_np

            return CodecTokens(
                codes=codes_np.astype(np.int64),
                n_codebooks=codes_np.shape[0],
                frame_rate=self.config.frame_rate,
                codec_type=self.codec_type,
            )

    def _decode_fish_dac(self, tokens: CodecTokens) -> np.ndarray:
        import torch
        with torch.no_grad():
            codes = torch.from_numpy(tokens.codes).long().unsqueeze(0)
            if self.device == "mps" and torch.backends.mps.is_available():
                codes = codes.to("mps")
            elif self.device == "cuda" and torch.cuda.is_available():
                codes = codes.to("cuda")

            if self._fish_backend == "fish_speech":
                audio = self._model.decode(codes)
            elif self._fish_backend == "dac":
                z = self._model.quantizer.from_codes(codes)[0]
                audio = self._model.decode(z)
            elif self._fish_backend == "firefly":
                fsq_max = 999  # prod([8,5,5,5]) - 1
                codes = codes.clamp(0, fsq_max)
                audio = self._model.decode(codes)
            else:
                if hasattr(self._model, "decode"):
                    audio = self._model.decode(codes)
                else:
                    raise RuntimeError("Fish DAC model has no decode method")

            return audio.cpu().numpy().squeeze()

    def _encode_mimi(self, audio: np.ndarray) -> CodecTokens:
        import torch
        inputs = self._feature_extractor(
            raw_audio=audio, sampling_rate=self.config.sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            encoded = self._model.encode(**inputs)
            codes = encoded.audio_codes.cpu().numpy().squeeze()
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
        return CodecTokens(
            codes=codes,
            n_codebooks=codes.shape[0],
            frame_rate=self.config.frame_rate,
            codec_type=self.codec_type,
        )

    def _decode_mimi(self, tokens: CodecTokens) -> np.ndarray:
        import torch
        codes_tensor = torch.from_numpy(tokens.codes).unsqueeze(0)
        with torch.no_grad():
            decoded = self._model.decode(codes_tensor)
            return decoded.audio_values.cpu().numpy().squeeze()

    def _encode_encodec(self, audio: np.ndarray) -> CodecTokens:
        import torch
        inputs = self._processor(
            raw_audio=audio, sampling_rate=self.config.sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            encoded = self._model.encode(**inputs)
            codes = encoded.audio_codes.cpu().numpy().squeeze()
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)
        return CodecTokens(
            codes=codes,
            n_codebooks=codes.shape[0],
            frame_rate=self.config.frame_rate,
            codec_type=self.codec_type,
        )

    def _decode_encodec(self, tokens: CodecTokens) -> np.ndarray:
        import torch
        codes_tensor = torch.from_numpy(tokens.codes).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            decoded = self._model.decode(codes_tensor, [None])
            return decoded.audio_values.cpu().numpy().squeeze()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def vocab_size(self) -> int:
        return self.config.codebook_size * self.config.n_codebooks

    @property
    def tokens_per_second(self) -> float:
        return self.config.frame_rate * self.config.n_codebooks

    @property
    def loaded(self) -> bool:
        return self._loaded


def main():
    """Quick codec test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test neural audio codec")
    parser.add_argument("--codec", default="snac", choices=["snac", "mimi", "encodec", "fish"])
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()

    import time

    codec = AudioCodec(args.codec)
    codec.load()

    sr = codec.sample_rate
    t = np.linspace(0, args.duration, int(sr * args.duration), dtype=np.float32)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    print(f"\nCodec: {args.codec}", flush=True)
    print(f"  Sample rate: {codec.sample_rate} Hz", flush=True)
    print(f"  Frame rate:  {codec.frame_rate} Hz", flush=True)
    print(f"  Vocab size:  {codec.vocab_size} ({codec.config.n_codebooks} x {codec.config.codebook_size})", flush=True)
    print(f"  Tokens/sec:  {codec.tokens_per_second}", flush=True)
    print(f"  Bandwidth:   {codec.config.bandwidth_kbps} kbps", flush=True)
    print(f"  Latency:     {codec.config.latency_ms} ms", flush=True)

    t0 = time.time()
    tokens = codec.encode(test_audio)
    encode_ms = (time.time() - t0) * 1000
    print(f"\nEncode: {encode_ms:.1f}ms for {args.duration}s audio -> {tokens.n_frames} frames", flush=True)
    print(f"  Codes shape: {tokens.codes.shape}", flush=True)
    print(f"  Flat tokens: {len(tokens.flat_tokens)}", flush=True)

    t0 = time.time()
    reconstructed = codec.decode(tokens)
    decode_ms = (time.time() - t0) * 1000
    print(f"Decode: {decode_ms:.1f}ms -> {len(reconstructed)} samples ({len(reconstructed)/sr:.2f}s)", flush=True)

    if len(reconstructed) >= len(test_audio):
        reconstructed = reconstructed[:len(test_audio)]
    else:
        test_audio = test_audio[:len(reconstructed)]

    mse = np.mean((test_audio - reconstructed) ** 2)
    snr = 10 * np.log10(np.mean(test_audio ** 2) / (mse + 1e-10))
    print(f"Quality: MSE={mse:.6f}, SNR={snr:.1f} dB", flush=True)

    print(f"\nStreaming encode test:", flush=True)
    chunk_size = int(sr * 0.08)
    stream_tokens = []
    t0 = time.time()
    for start in range(0, len(test_audio), chunk_size):
        chunk = test_audio[start:start + chunk_size]
        result = codec.encode_chunk(chunk)
        if result is not None:
            stream_tokens.append(result)
    final = codec.flush_encode()
    if final is not None:
        stream_tokens.append(final)
    stream_ms = (time.time() - t0) * 1000
    total_frames = sum(t.n_frames for t in stream_tokens)
    print(f"  {len(stream_tokens)} chunks, {total_frames} total frames, {stream_ms:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
