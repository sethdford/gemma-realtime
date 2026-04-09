#!/usr/bin/env python3
"""
Speech encoder adapter for Freeze-Omni style architecture on MLX.

Converts streaming audio chunks into embeddings compatible with Gemma's
embedding space, allowing the frozen LLM to process speech input directly.

Architecture (from Freeze-Omni paper, adapted for MLX):
    Audio (24kHz, 160ms chunks)
    -> Conv1D feature extractor (downsample to frame-level features)
    -> Positional encoding
    -> Transformer encoder (lightweight, 2-4 layers)
    -> Linear projection to LLM embedding dimension
    -> [Injected into Gemma's embedding sequence]

The encoder is fully trainable while the LLM backbone stays frozen.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class ConvFeatureExtractor(nn.Module):
    """Multi-scale 1D convolution stack that downsamples raw audio to frame features.

    Inspired by Whisper/wav2vec2 feature extractors but smaller for low latency.
    Downsamples 24kHz audio by ~320x to 75 Hz frame rate, then pools to ~12.5 Hz.
    MLX Conv1d uses NLC layout: (batch, length, channels).
    """

    def __init__(self, in_channels=1, hidden_dim=512, out_dim=512):
        super().__init__()
        self.conv_layers = [
            nn.Conv1d(in_channels, 64, kernel_size=10, stride=5, padding=0),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=0),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.Conv1d(256, hidden_dim, kernel_size=4, stride=2, padding=0),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=4, stride=2, padding=0),
        ]
        self.norms = [nn.LayerNorm(d) for d in [64, 128, 256, hidden_dim, out_dim]]

    def __call__(self, x: mx.array) -> mx.array:
        """x: (batch, time, channels) NLC format -> (batch, time', out_dim)"""
        for conv, norm in zip(self.conv_layers, self.norms):
            x = conv(x)
            x = norm(x)
            x = nn.gelu(x)
        return x


class ChunkPooling(nn.Module):
    """Pool frame-level features into chunk-level features for the LLM.

    Each 160ms audio chunk produces a fixed number of embedding tokens
    that get injected into the LLM's sequence.
    """

    def __init__(self, tokens_per_chunk=4, feature_dim=512):
        super().__init__()
        self.tokens_per_chunk = tokens_per_chunk
        self.pool_proj = nn.Linear(feature_dim, feature_dim * tokens_per_chunk)
        self.reshape_dim = feature_dim

    def __call__(self, x: mx.array) -> mx.array:
        """x: (batch, frames, dim) -> (batch, tokens_per_chunk, dim)"""
        pooled = mx.mean(x, axis=1, keepdims=True)
        projected = self.pool_proj(pooled)
        batch = projected.shape[0]
        return mx.reshape(projected, (batch, self.tokens_per_chunk, self.reshape_dim))


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

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
        """x: (batch, seq_len, dim)"""
        seq_len = x.shape[1]
        return x + self._pe[:seq_len]


class SpeechEncoderLayer(nn.Module):
    """Single transformer encoder layer for speech features."""

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

    Designed to be trained while the LLM backbone stays frozen.
    Outputs embeddings in the same space as Gemma's token embeddings.

    Args:
        llm_dim: Gemma's hidden dimension (2048 for E4B, 2560 for E2B, 4096 for 31B)
        encoder_dim: Internal encoder dimension
        n_heads: Attention heads in encoder transformer
        n_layers: Number of encoder transformer layers
        tokens_per_chunk: Number of embedding tokens per 160ms audio chunk
        chunk_ms: Audio chunk duration in milliseconds
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        encoder_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        tokens_per_chunk: int = 4,
        chunk_ms: int = 160,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.encoder_dim = encoder_dim
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(24000 * chunk_ms / 1000)

        self.feature_extractor = ConvFeatureExtractor(
            in_channels=1, hidden_dim=encoder_dim, out_dim=encoder_dim
        )
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
            audio: (batch, 1, samples) raw audio waveform [NCL]
                   Internally transposed to (batch, samples, 1) for MLX Conv1d [NLC]

        Returns:
            (batch, tokens_per_chunk, llm_dim) embeddings for injection into LLM
        """
        x = mx.transpose(audio, (0, 2, 1))
        features = self.feature_extractor(x)
        features = self.pos_encoding(features)

        x = features
        for layer in self.encoder_layers:
            x = layer(x)

        pooled = self.chunk_pool(x)
        projected = self.output_proj(pooled)
        return self.output_norm(projected)

    def encode_chunk(self, audio_chunk: mx.array) -> mx.array:
        """Encode a single 160ms audio chunk for streaming.

        Args:
            audio_chunk: (1, 1, chunk_samples) single audio chunk

        Returns:
            (1, tokens_per_chunk, llm_dim)
        """
        return self.__call__(audio_chunk)

    def num_params(self) -> int:
        """Count trainable parameters."""
        import mlx.utils
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))


class SpeechEncoderConfig:
    """Configuration presets for different Gemma model sizes."""

    PRESETS = {
        "e2b": {
            "llm_dim": 2304,
            "encoder_dim": 384,
            "n_heads": 6,
            "n_layers": 2,
            "d_ff": 1536,
            "tokens_per_chunk": 4,
        },
        "e4b": {
            "llm_dim": 2560,
            "encoder_dim": 512,
            "n_heads": 8,
            "n_layers": 2,
            "d_ff": 2048,
            "tokens_per_chunk": 4,
        },
        "31b": {
            "llm_dim": 4096,
            "encoder_dim": 768,
            "n_heads": 12,
            "n_layers": 3,
            "d_ff": 3072,
            "tokens_per_chunk": 4,
        },
    }

    @classmethod
    def from_target(cls, target: str, **overrides) -> dict:
        if target not in cls.PRESETS:
            raise ValueError(f"Unknown target: {target}. Choose from: {list(cls.PRESETS.keys())}")
        config = dict(cls.PRESETS[target])
        config.update(overrides)
        return config


def main():
    """Quick test of the speech encoder."""
    import time

    for target in ["e2b", "e4b", "31b"]:
        config = SpeechEncoderConfig.from_target(target)
        encoder = SpeechEncoder(**config)
        n_params = encoder.num_params()

        batch_size = 1
        chunk_samples = encoder.chunk_samples
        audio = mx.random.normal((batch_size, 1, chunk_samples))

        t0 = time.time()
        for _ in range(10):
            embeddings = encoder(audio)
            mx.eval(embeddings)
        elapsed = (time.time() - t0) / 10 * 1000

        print(
            f"  {target}: {n_params/1e6:.1f}M params, "
            f"input=({batch_size}, 1, {chunk_samples}), "
            f"output={embeddings.shape}, "
            f"{elapsed:.1f}ms/chunk",
            flush=True,
        )


if __name__ == "__main__":
    main()
