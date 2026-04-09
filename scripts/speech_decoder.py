#!/usr/bin/env python3
"""
Speech decoder adapter for Freeze-Omni style architecture on MLX.

Converts Gemma's hidden states into discrete audio codec tokens using an
autoregressive decoder. Produces streaming audio output via neural codec.

Architecture (from Freeze-Omni, adapted for MLX):
    Gemma hidden states (llm_dim)
    -> Linear adapter (llm_dim -> decoder_dim)
    -> AR Transformer decoder (causal, generates codec tokens one at a time)
    -> Codebook projection head (decoder_dim -> codebook_size)
    -> [Feed tokens to SNAC/Mimi decoder for audio waveform]

The decoder uses a single codebook for lowest latency (Freeze-Omni finding).
Multi-codebook depth decoding is available for higher quality (Phase 5).

Duplex state predictor is integrated: after each LLM hidden state,
predict whether the model should speak, listen, or handle an interruption.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention for autoregressive decoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, kv_cache: Optional[tuple] = None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)
        new_cache = (k, v)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale

        T_k = k.shape[2]
        causal_mask = mx.triu(mx.full((T, T_k), -1e9), k=T_k - T + 1)
        attn = attn + causal_mask

        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out), new_cache


class CrossAttention(nn.Module):
    """Cross-attention from decoder to LLM hidden states."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        B, T, C = x.shape
        T_ctx = context.shape[1]

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, T_ctx, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, T_ctx, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        attn = mx.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class SpeechDecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention + cross-attention to LLM."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, context: mx.array,
                 kv_cache: Optional[tuple] = None):
        residual = x
        x = self.norm1(x)
        x, new_cache = self.self_attn(x, kv_cache=kv_cache)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, context)
        x = self.dropout(x) + residual

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = self.dropout(x) + residual

        return x, new_cache


class SpeechDecoder(nn.Module):
    """Autoregressive speech token decoder.

    Takes Gemma's hidden states and generates codec tokens autoregressively.
    Uses a single codebook for minimum latency (Freeze-Omni approach).

    Args:
        llm_dim: Gemma hidden dimension
        decoder_dim: Internal decoder dimension
        n_heads: Attention heads
        n_layers: Decoder transformer layers
        codebook_size: Size of the codec vocabulary (e.g. 4096 for SNAC)
        max_tokens: Maximum audio tokens to generate
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        decoder_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        codebook_size: int = 4096,
        max_tokens: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.decoder_dim = decoder_dim
        self.codebook_size = codebook_size
        self.max_tokens = max_tokens

        self.input_adapter = nn.Sequential(
            nn.Linear(llm_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU(),
        )

        self.token_embedding = nn.Embedding(codebook_size + 2, decoder_dim)
        self.BOS_TOKEN = codebook_size
        self.EOS_TOKEN = codebook_size + 1

        self.pos_encoding = _SinusoidalPE(decoder_dim)

        self.layers = [
            SpeechDecoderLayer(decoder_dim, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]

        self.output_norm = nn.LayerNorm(decoder_dim)
        self.output_head = nn.Linear(decoder_dim, codebook_size + 1)

    def __call__(self, llm_hidden: mx.array, target_tokens: mx.array = None):
        """Forward pass for training (teacher-forced).

        Args:
            llm_hidden: (batch, seq, llm_dim) hidden states from frozen Gemma
            target_tokens: (batch, audio_len) ground-truth codec tokens

        Returns:
            logits: (batch, audio_len, codebook_size+1) next-token predictions
        """
        context = self.input_adapter(llm_hidden)

        bos = mx.full((target_tokens.shape[0], 1), self.BOS_TOKEN, dtype=mx.int32)
        input_tokens = mx.concatenate([bos, target_tokens[:, :-1]], axis=1)

        x = self.token_embedding(input_tokens)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x, _ = layer(x, context)

        x = self.output_norm(x)
        return self.output_head(x)

    def generate(self, llm_hidden: mx.array, temperature: float = 0.8,
                 top_k: int = 50) -> mx.array:
        """Autoregressive generation of codec tokens.

        Args:
            llm_hidden: (1, seq, llm_dim) hidden states from frozen Gemma

        Returns:
            tokens: (1, generated_len) codec token IDs
        """
        context = self.input_adapter(llm_hidden)

        tokens = [self.BOS_TOKEN]
        kv_caches = [None] * len(self.layers)

        for step in range(self.max_tokens):
            token_id = mx.array([[tokens[-1]]], dtype=mx.int32)
            x = self.token_embedding(token_id)
            x = self.pos_encoding(x, offset=step)

            new_caches = []
            for i, layer in enumerate(self.layers):
                x, cache = layer(x, context, kv_cache=kv_caches[i])
                new_caches.append(cache)
            kv_caches = new_caches

            x = self.output_norm(x)
            logits = self.output_head(x[:, -1, :])

            if top_k > 0:
                top_vals = mx.sort(logits, axis=-1)[:, -top_k]
                logits = mx.where(logits < top_vals, -1e9, logits)

            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs).item()
            else:
                next_token = mx.argmax(logits, axis=-1).item()

            if next_token == self.EOS_TOKEN or next_token >= self.codebook_size:
                break

            tokens.append(next_token)

        return mx.array([tokens[1:]], dtype=mx.int32)

    def generate_streaming(self, llm_hidden: mx.array, chunk_size: int = 12,
                           temperature: float = 0.8, top_k: int = 50):
        """Streaming generation: yield chunks of codec tokens.

        Yields every `chunk_size` tokens for incremental decoding.
        At 12.5 Hz frame rate, chunk_size=12 = ~960ms of audio per yield.
        """
        context = self.input_adapter(llm_hidden)
        tokens = [self.BOS_TOKEN]
        kv_caches = [None] * len(self.layers)
        chunk_buffer = []

        for step in range(self.max_tokens):
            token_id = mx.array([[tokens[-1]]], dtype=mx.int32)
            x = self.token_embedding(token_id)
            x = self.pos_encoding(x, offset=step)

            new_caches = []
            for i, layer in enumerate(self.layers):
                x, cache = layer(x, context, kv_cache=kv_caches[i])
                new_caches.append(cache)
            kv_caches = new_caches

            x = self.output_norm(x)
            logits = self.output_head(x[:, -1, :])

            if top_k > 0:
                top_vals = mx.sort(logits, axis=-1)[:, -top_k]
                logits = mx.where(logits < top_vals, -1e9, logits)

            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs).item()
            else:
                next_token = mx.argmax(logits, axis=-1).item()

            if next_token == self.EOS_TOKEN or next_token >= self.codebook_size:
                if chunk_buffer:
                    yield mx.array([chunk_buffer], dtype=mx.int32)
                return

            tokens.append(next_token)
            chunk_buffer.append(next_token)

            if len(chunk_buffer) >= chunk_size:
                yield mx.array([chunk_buffer], dtype=mx.int32)
                chunk_buffer = []

    def num_params(self) -> int:
        import mlx.utils
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))


class DuplexStatePredictor(nn.Module):
    """Predicts conversation state from LLM hidden states.

    At each chunk boundary, predicts:
        0 = LISTEN (user is speaking, agent should be quiet)
        1 = SPEAK  (agent should generate speech)
        2 = INTERRUPT (user interrupted, stop generating)

    Attached after the last LLM layer, trained in Stage 3.
    """

    LISTEN = 0
    SPEAK = 1
    INTERRUPT = 2

    def __init__(self, llm_dim: int = 2048, hidden_dim: int = 256, n_states: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_states),
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Predict state from last hidden state.

        Args:
            hidden_states: (batch, seq, llm_dim)

        Returns:
            logits: (batch, n_states) state prediction logits
        """
        last_hidden = hidden_states[:, -1, :]
        return self.classifier(last_hidden)

    def predict(self, hidden_states: mx.array) -> int:
        """Get predicted state as integer."""
        logits = self.__call__(hidden_states)
        return mx.argmax(logits, axis=-1).item()


class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding with offset support for streaming."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len).reshape(-1, 1).astype(mx.float32)
        div_term = mx.exp(
            mx.arange(0, d_model, 2).astype(mx.float32) * (-math.log(10000.0) / d_model)
        )
        pe_sin = mx.sin(position * div_term)
        pe_cos = mx.cos(position * div_term)
        pe = mx.concatenate([pe_sin, pe_cos], axis=-1)[:, :d_model]
        self._pe = pe

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        seq_len = x.shape[1]
        return x + self._pe[offset : offset + seq_len]


class SpeechDecoderConfig:
    """Configuration presets for different Gemma model sizes."""

    PRESETS = {
        "e2b": {
            "llm_dim": 2304,
            "decoder_dim": 384,
            "n_heads": 6,
            "n_layers": 3,
            "d_ff": 1536,
            "codebook_size": 4096,
            "max_tokens": 500,
        },
        "e4b": {
            "llm_dim": 2560,
            "decoder_dim": 512,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff": 2048,
            "codebook_size": 4096,
            "max_tokens": 500,
        },
        "31b": {
            "llm_dim": 4096,
            "decoder_dim": 768,
            "n_heads": 12,
            "n_layers": 4,
            "d_ff": 3072,
            "codebook_size": 4096,
            "max_tokens": 500,
        },
    }

    @classmethod
    def from_target(cls, target: str, **overrides) -> dict:
        if target not in cls.PRESETS:
            raise ValueError(f"Unknown target: {target}")
        config = dict(cls.PRESETS[target])
        config.update(overrides)
        return config


def main():
    """Quick test of speech decoder and state predictor."""
    import time

    for target in ["e2b", "e4b", "31b"]:
        dec_config = SpeechDecoderConfig.from_target(target)
        decoder = SpeechDecoder(**dec_config)
        predictor = DuplexStatePredictor(llm_dim=dec_config["llm_dim"])

        dec_params = decoder.num_params()
        import mlx.utils
        pred_params = sum(v.size for _, v in mlx.utils.tree_flatten(predictor.parameters()))

        llm_hidden = mx.random.normal((1, 10, dec_config["llm_dim"]))

        t0 = time.time()
        tokens = decoder.generate(llm_hidden, temperature=0.8, top_k=50)
        mx.eval(tokens)
        gen_ms = (time.time() - t0) * 1000

        state = predictor.predict(llm_hidden)

        print(
            f"  {target}: decoder={dec_params/1e6:.1f}M params + predictor={pred_params/1e3:.0f}K params, "
            f"generated {tokens.shape[-1]} tokens in {gen_ms:.0f}ms, "
            f"state={['LISTEN', 'SPEAK', 'INTERRUPT'][state]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
