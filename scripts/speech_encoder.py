#!/usr/bin/env python3
"""
Speech encoder adapter for Freeze-Omni style architecture on MLX.

Converts streaming audio chunks into embeddings compatible with Gemma's
embedding space, allowing the frozen LLM to process speech input directly.

Architecture:
    Audio (24kHz, 160ms chunks)
    -> Log-mel spectrogram (80 bins)
    -> Linear projection to encoder_dim
    -> Positional encoding
    -> Transformer encoder (4 layers)
    -> Chunk pooling -> Linear projection to LLM dim
    -> [Injected into Gemma's embedding sequence]

The encoder is fully trainable while the LLM backbone stays frozen.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def mel_filterbank(n_fft: int, n_mels: int, sr: int) -> mx.array:
    """Build mel filterbank matrix (compatible with Whisper/librosa)."""
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            if k < fb.shape[0]:
                fb[k, m] = (k - bins[m]) / max(bins[m + 1] - bins[m], 1)
        for k in range(bins[m + 1], bins[m + 2]):
            if k < fb.shape[0]:
                fb[k, m] = (bins[m + 2] - k) / max(bins[m + 2] - bins[m + 1], 1)
    return mx.array(fb)


class MelSpectrogram(nn.Module):
    """Compute log-mel spectrogram from raw audio using MLX operations.

    This produces a compact, informative representation that is much easier
    for a small transformer to learn from than raw waveforms.
    """

    def __init__(self, n_fft: int = 400, hop_length: int = 160,
                 n_mels: int = 80, sample_rate: int = 24000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        self._window = mx.array(window)
        self._mel_fb = mel_filterbank(n_fft, n_mels, sample_rate)

        # Precompute DFT basis matrices for efficient spectrogram
        k = np.arange(n_fft // 2 + 1).reshape(1, -1).astype(np.float32)
        t = np.arange(n_fft).reshape(-1, 1).astype(np.float32)
        angles = -2.0 * np.pi * k * t / n_fft
        self._dft_cos = mx.array(np.cos(angles).astype(np.float32))  # (n_fft, n_fft//2+1)
        self._dft_sin = mx.array(np.sin(angles).astype(np.float32))

    def __call__(self, audio: mx.array) -> mx.array:
        """audio: (batch, samples) -> (batch, n_frames, n_mels)"""
        if audio.ndim == 3:
            audio = audio.squeeze(1)

        batch, n_samples = audio.shape
        n_frames = (n_samples - self.n_fft) // self.hop_length + 1
        if n_frames <= 0:
            return mx.zeros((batch, 1, self.n_mels))

        # Extract overlapping frames via indexing
        indices = mx.arange(self.n_fft)[None, :] + mx.arange(n_frames)[:, None] * self.hop_length
        frames_list = []
        for b in range(batch):
            f = audio[b][indices.reshape(-1)].reshape(n_frames, self.n_fft)
            frames_list.append(f)
        frames = mx.stack(frames_list)  # (batch, n_frames, n_fft)

        windowed = frames * self._window[None, None, :]

        # Power spectrum via matmul with precomputed DFT basis
        # windowed: (batch, n_frames, n_fft)
        # _dft_cos/_dft_sin: (n_fft, n_fft//2+1)
        real = windowed @ self._dft_cos  # (batch, n_frames, n_fft//2+1)
        imag = windowed @ self._dft_sin
        power = real ** 2 + imag ** 2

        mel = power @ self._mel_fb  # (batch, n_frames, n_mels)
        mel = mx.log(mx.maximum(mel, 1e-10))

        mel = (mel - mx.mean(mel, axis=(1, 2), keepdims=True)) / (mx.std(mel, axis=(1, 2), keepdims=True) + 1e-8)
        return mel


class ChunkPooling(nn.Module):
    """Pool frame-level features into chunk-level features for the LLM."""

    def __init__(self, tokens_per_chunk=4, feature_dim=512):
        super().__init__()
        self.tokens_per_chunk = tokens_per_chunk
        self.pool_proj = nn.Linear(feature_dim, feature_dim * tokens_per_chunk)
        self.reshape_dim = feature_dim

    def __call__(self, x: mx.array) -> mx.array:
        pooled = mx.mean(x, axis=1, keepdims=True)
        projected = self.pool_proj(pooled)
        batch = projected.shape[0]
        return mx.reshape(projected, (batch, self.tokens_per_chunk, self.reshape_dim))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len).reshape(-1, 1).astype(mx.float32)
        div_term = mx.exp(mx.arange(0, d_model, 2).astype(mx.float32) * (-math.log(10000.0) / d_model))
        pe_sin = mx.sin(position * div_term)
        pe_cos = mx.cos(position * div_term)
        pe = mx.concatenate([pe_sin, pe_cos], axis=-1)[:, :d_model]
        self._pe = pe

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = x.shape[1]
        return x + self._pe[:seq_len]


class SpeechEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout(x) + residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x) + residual
        return x


class SpeechEncoder(nn.Module):
    """Full speech encoder: audio -> LLM-compatible embeddings.

    Uses log-mel spectrogram features instead of raw audio for much better
    speech representation learning with limited model capacity.
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        encoder_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        tokens_per_chunk: int = 4,
        chunk_ms: int = 160,
        dropout: float = 0.1,
        n_mels: int = 80,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.encoder_dim = encoder_dim
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(24000 * chunk_ms / 1000)
        self.n_mels = n_mels

        self.mel_spec = MelSpectrogram(n_fft=400, hop_length=160, n_mels=n_mels, sample_rate=24000)
        self.input_proj = nn.Linear(n_mels, encoder_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(encoder_dim)
        self.encoder_layers = [
            SpeechEncoderLayer(encoder_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        self.chunk_pool = ChunkPooling(tokens_per_chunk, encoder_dim)
        self.output_proj = nn.Linear(encoder_dim, llm_dim)
        self.output_norm = nn.LayerNorm(llm_dim)

    def __call__(self, audio: mx.array) -> mx.array:
        """Encode audio to LLM-compatible embeddings.

        Args:
            audio: (batch, 1, samples) raw audio waveform
        Returns:
            (batch, tokens_per_chunk, llm_dim)
        """
        # audio: (batch, 1, samples) -> (batch, samples)
        if audio.ndim == 3:
            x = audio.squeeze(1)
        else:
            x = audio

        mel = self.mel_spec(x)  # (batch, n_frames, n_mels)
        x = self.input_proj(mel)  # (batch, n_frames, encoder_dim)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        pooled = self.chunk_pool(x)
        projected = self.output_proj(pooled)
        return self.output_norm(projected)

    def encode_chunk(self, audio_chunk: mx.array) -> mx.array:
        return self.__call__(audio_chunk)

    def num_params(self) -> int:
        import mlx.utils
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))


class SpeechEncoderConfig:
    PRESETS = {
        "e2b": {"llm_dim": 2304, "encoder_dim": 384, "n_heads": 6, "n_layers": 3, "d_ff": 1536, "tokens_per_chunk": 4},
        "e4b": {"llm_dim": 2560, "encoder_dim": 512, "n_heads": 8, "n_layers": 4, "d_ff": 2048, "tokens_per_chunk": 4},
        "31b": {"llm_dim": 4096, "encoder_dim": 768, "n_heads": 12, "n_layers": 4, "d_ff": 3072, "tokens_per_chunk": 4},
    }

    @classmethod
    def from_target(cls, target: str, **overrides) -> dict:
        if target not in cls.PRESETS:
            raise ValueError(f"Unknown target: {target}")
        config = dict(cls.PRESETS[target])
        config.update(overrides)
        return config


def main():
    import time
    for target in ["e2b", "e4b", "31b"]:
        config = SpeechEncoderConfig.from_target(target)
        encoder = SpeechEncoder(**config)
        n_params = encoder.num_params()

        chunk_samples = encoder.chunk_samples
        audio = mx.random.normal((1, 1, chunk_samples))

        t0 = time.time()
        for _ in range(10):
            emb = encoder(audio)
            mx.eval(emb)
        elapsed = (time.time() - t0) / 10 * 1000

        print(f"  {target}: {n_params/1e6:.1f}M params, output={emb.shape}, {elapsed:.1f}ms/chunk")


if __name__ == "__main__":
    main()
