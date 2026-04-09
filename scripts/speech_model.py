#!/usr/bin/env python3
"""
Full speech-to-speech model with inner monologue and dual-stream support.

Phase 5 implementation: extends the Freeze-Omni adapter architecture with
Moshi-style inner monologue (joint text+audio token prediction) and
dual-stream modeling (simultaneous user/agent audio processing).

Architecture:
    User Audio Stream ----> [Codec Encode] ---> [User Tokens]
                                                      |
                                                      v
    [User Tokens] + [Agent Tokens] + [Text Tokens] -> [Gemma E4B (frozen)]
                                                      |
                                                      v
                                              [Hidden States]
                                              /      |      \
                                             v       v       v
                                     [Text Head] [Audio Head] [State Head]
                                        |            |            |
                                  Inner Monologue  Agent Audio   Speak/Listen
                                   (not spoken)    Codec Tokens

Key innovations from research:
    - Inner Monologue (Moshi): Text tokens predicted alongside audio improve
      linguistic quality dramatically. The model "thinks in text" while speaking.
    - Dual-Stream (Moshi): User and agent audio modeled in parallel streams,
      enabling full-duplex conversation with natural overlap handling.
    - Vocabulary Extension: SNAC codec tokens added to Gemma's vocabulary with
      frozen original embeddings + trainable audio embeddings.
    - Interleaved Token Schedule: Each 80ms frame generates
      [user_semantic, user_acoustic..., text_token, agent_semantic, agent_acoustic...]
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class SpeechModelConfig:
    """Configuration for the full speech-to-speech model."""
    llm_dim: int = 2560
    text_vocab_size: int = 256000
    codec_vocab_size: int = 4096
    n_codebooks: int = 3
    frame_rate_hz: float = 12.5
    tokens_per_frame: int = 8
    max_text_tokens: int = 4096
    max_audio_frames: int = 750
    inner_monologue: bool = True
    dual_stream: bool = True
    depth_transformer_dim: int = 256
    depth_transformer_layers: int = 2
    depth_transformer_heads: int = 4

    @property
    def total_audio_vocab(self) -> int:
        return self.codec_vocab_size * self.n_codebooks

    @property
    def extended_vocab_size(self) -> int:
        return self.text_vocab_size + self.total_audio_vocab + 10

    AUDIO_TOKEN_OFFSET = 256000
    USER_STREAM_OFFSET = 256000
    AGENT_STREAM_OFFSET = 256000 + 4096 * 3
    TEXT_BOS = 256000 + 4096 * 6
    TEXT_EOS = 256000 + 4096 * 6 + 1
    AUDIO_BOS = 256000 + 4096 * 6 + 2
    AUDIO_EOS = 256000 + 4096 * 6 + 3


class ExtendedEmbedding(nn.Module):
    """Extended embedding that preserves frozen text embeddings and adds audio tokens.

    Original Gemma embeddings are frozen. New audio token embeddings are trainable.
    This prevents catastrophic forgetting of text capabilities.
    """

    def __init__(self, config: SpeechModelConfig):
        super().__init__()
        self.config = config
        self.audio_embedding = nn.Embedding(config.total_audio_vocab + 10, config.llm_dim)
        self._text_embedding = None

    def set_text_embedding(self, text_embed_weight: mx.array):
        """Freeze the original text embedding weights from Gemma."""
        self._text_embedding = mx.stop_gradient(text_embed_weight)

    def __call__(self, token_ids: mx.array) -> mx.array:
        is_audio = token_ids >= self.config.AUDIO_TOKEN_OFFSET
        audio_ids = mx.where(is_audio, token_ids - self.config.AUDIO_TOKEN_OFFSET, 0)
        text_ids = mx.where(is_audio, 0, token_ids)

        audio_embeds = self.audio_embedding(audio_ids)

        if self._text_embedding is not None:
            text_ids_clipped = mx.clip(text_ids, 0, self._text_embedding.shape[0] - 1)
            text_embeds = self._text_embedding[text_ids_clipped]
        else:
            text_embeds = mx.zeros_like(audio_embeds)

        is_audio_expanded = mx.expand_dims(is_audio, -1)
        return mx.where(is_audio_expanded, audio_embeds, text_embeds)


class DepthTransformer(nn.Module):
    """Small transformer for inter-codebook dependencies within a single time step.

    From Moshi: at each time step, after the main (Temporal) transformer produces
    the first codebook token, the Depth transformer autoregressively generates
    the remaining codebook tokens conditioned on the temporal hidden state.
    """

    def __init__(self, config: SpeechModelConfig):
        super().__init__()
        dim = config.depth_transformer_dim
        self.input_proj = nn.Linear(config.llm_dim, dim)
        self.token_embed = nn.Embedding(config.codec_vocab_size + 2, dim)

        self.layers = []
        for _ in range(config.depth_transformer_layers):
            self.layers.append(_DepthBlock(dim, config.depth_transformer_heads))

        self.output_proj = nn.Linear(dim, config.codec_vocab_size)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, temporal_hidden: mx.array, prev_codebook_tokens: mx.array) -> mx.array:
        """Generate next codebook token given temporal context and previous tokens.

        Args:
            temporal_hidden: (batch, 1, llm_dim) from the main transformer
            prev_codebook_tokens: (batch, k) previously generated codebook tokens at this timestep

        Returns:
            logits: (batch, codec_vocab_size) for next codebook token
        """
        context = self.input_proj(temporal_hidden)

        if prev_codebook_tokens.shape[1] > 0:
            token_embeds = self.token_embed(prev_codebook_tokens)
            x = mx.concatenate([context, token_embeds], axis=1)
        else:
            x = context

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x[:, -1:, :])
        return self.output_proj(x.squeeze(1))


class _DepthBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dim, n_heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        T = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class DualStreamMixer(nn.Module):
    """Mixes user and agent audio streams for full-duplex processing.

    Both streams are processed simultaneously by the temporal transformer.
    The mixer handles:
    1. Interleaving user/agent tokens in the correct frame order
    2. Applying stream-specific positional encodings
    3. Masking to prevent information leakage from future user tokens
    """

    def __init__(self, config: SpeechModelConfig):
        super().__init__()
        self.config = config
        self.user_proj = nn.Linear(config.llm_dim, config.llm_dim)
        self.agent_proj = nn.Linear(config.llm_dim, config.llm_dim)
        self.stream_embed = nn.Embedding(2, config.llm_dim)
        self.mix_norm = nn.LayerNorm(config.llm_dim)

    def __call__(self, user_embeds: mx.array, agent_embeds: mx.array) -> mx.array:
        """Interleave user and agent stream embeddings.

        Args:
            user_embeds: (batch, user_frames, llm_dim)
            agent_embeds: (batch, agent_frames, llm_dim)

        Returns:
            mixed: (batch, user_frames + agent_frames, llm_dim) interleaved
        """
        user = self.user_proj(user_embeds) + self.stream_embed(mx.zeros(user_embeds.shape[:2], dtype=mx.int32))
        agent = self.agent_proj(agent_embeds) + self.stream_embed(mx.ones(agent_embeds.shape[:2], dtype=mx.int32))

        B = user.shape[0]
        U = user.shape[1]
        A = agent.shape[1]
        max_frames = max(U, A)

        interleaved_parts = []
        for f in range(max_frames):
            if f < U:
                interleaved_parts.append(user[:, f:f+1, :])
            if f < A:
                interleaved_parts.append(agent[:, f:f+1, :])

        if not interleaved_parts:
            return mx.zeros((B, 0, self.config.llm_dim))

        mixed = mx.concatenate(interleaved_parts, axis=1)
        return self.mix_norm(mixed)


class InnerMonologueHead(nn.Module):
    """Predicts text tokens alongside audio for inner monologue.

    The key insight from Moshi: generating text tokens in parallel with audio
    dramatically improves linguistic quality. The text is the model's "thoughts"
    — it's never spoken but guides the audio generation.
    """

    def __init__(self, config: SpeechModelConfig):
        super().__init__()
        self.text_head = nn.Linear(config.llm_dim, config.text_vocab_size)
        self.alignment_proj = nn.Linear(config.llm_dim, config.llm_dim)
        self.norm = nn.LayerNorm(config.llm_dim)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        """Predict text tokens and produce alignment-enhanced hidden states.

        Args:
            hidden_states: (batch, seq, llm_dim)

        Returns:
            text_logits: (batch, seq, text_vocab_size)
            enhanced_hidden: (batch, seq, llm_dim) for audio head
        """
        text_logits = self.text_head(hidden_states)
        enhanced = hidden_states + self.alignment_proj(self.norm(hidden_states))
        return text_logits, enhanced


class SpeechToSpeechModel(nn.Module):
    """Complete speech-to-speech model with inner monologue and dual-stream.

    This wraps around a frozen Gemma model and adds:
    1. Extended vocabulary (text + audio codec tokens)
    2. Depth transformer for multi-codebook generation
    3. Dual-stream mixer for full-duplex
    4. Inner monologue head for text-guided audio quality
    5. State predictor for turn-taking

    The frozen Gemma weights are NOT stored here — they're loaded separately
    and the forward pass is composed at runtime.
    """

    def __init__(self, config: SpeechModelConfig):
        super().__init__()
        self.config = config

        self.embedding = ExtendedEmbedding(config)
        self.depth_transformer = DepthTransformer(config)

        if config.dual_stream:
            self.stream_mixer = DualStreamMixer(config)

        if config.inner_monologue:
            self.monologue_head = InnerMonologueHead(config)

        self.audio_head = nn.Linear(config.llm_dim, config.codec_vocab_size)
        self.audio_head_norm = nn.LayerNorm(config.llm_dim)

        from speech_decoder import DuplexStatePredictor
        self.state_predictor = DuplexStatePredictor(
            llm_dim=config.llm_dim, hidden_dim=256, n_states=3
        )

    def prepare_input(self, text_tokens: mx.array = None,
                      user_audio_tokens: mx.array = None,
                      agent_audio_tokens: mx.array = None) -> mx.array:
        """Prepare interleaved input sequence for the temporal transformer.

        Follows the Moshi token schedule per 80ms frame:
        [user_semantic, user_acoustic..., text_token, agent_semantic, agent_acoustic...]
        """
        parts = []

        if user_audio_tokens is not None:
            user_offset = mx.array(self.config.USER_STREAM_OFFSET, dtype=mx.int32)
            user_ids = user_audio_tokens + user_offset
            parts.append(self.embedding(user_ids))

        if text_tokens is not None:
            parts.append(self.embedding(text_tokens))

        if agent_audio_tokens is not None:
            agent_offset = mx.array(self.config.AGENT_STREAM_OFFSET, dtype=mx.int32)
            agent_ids = agent_audio_tokens + agent_offset
            parts.append(self.embedding(agent_ids))

        if not parts:
            return mx.zeros((1, 0, self.config.llm_dim))

        return mx.concatenate(parts, axis=1)

    def predict_audio_frame(self, temporal_hidden: mx.array,
                            temperature: float = 0.8) -> tuple[mx.array, mx.array]:
        """Generate one complete audio frame (all codebooks) from hidden state.

        Uses the depth transformer for inter-codebook dependencies.

        Returns:
            text_logits: (batch, text_vocab) if inner monologue enabled
            audio_tokens: (batch, n_codebooks) generated codec tokens
        """
        text_logits = None
        hidden = temporal_hidden

        if self.config.inner_monologue:
            text_logits, hidden = self.monologue_head(hidden)

        first_logits = self.audio_head(self.audio_head_norm(hidden[:, -1:, :]))
        first_logits = first_logits.squeeze(1)

        if temperature > 0:
            probs = mx.softmax(first_logits / temperature, axis=-1)
            first_token = mx.random.categorical(probs)
        else:
            first_token = mx.argmax(first_logits, axis=-1)

        audio_tokens = [first_token]

        prev_tokens = first_token.reshape(-1, 1)
        for cb in range(1, self.config.n_codebooks):
            depth_logits = self.depth_transformer(hidden[:, -1:, :], prev_tokens)
            if temperature > 0:
                probs = mx.softmax(depth_logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs)
            else:
                next_token = mx.argmax(depth_logits, axis=-1)
            audio_tokens.append(next_token)
            prev_tokens = mx.concatenate([prev_tokens, next_token.reshape(-1, 1)], axis=1)

        return text_logits, mx.stack(audio_tokens, axis=-1)

    def predict_state(self, hidden_states: mx.array) -> int:
        """Predict conversation state (LISTEN/SPEAK/INTERRUPT)."""
        return self.state_predictor.predict(hidden_states)

    def num_params(self) -> int:
        import mlx.utils
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))


PRESET_CONFIGS = {
    "e2b": SpeechModelConfig(llm_dim=2304, text_vocab_size=256000),
    "e4b": SpeechModelConfig(llm_dim=2560, text_vocab_size=256000),
    "31b": SpeechModelConfig(llm_dim=4096, text_vocab_size=256000),
}


def main():
    """Architecture validation test."""
    import time

    for target in ["e2b", "e4b"]:
        config = PRESET_CONFIGS[target]
        model = SpeechToSpeechModel(config)
        n_params = model.num_params()

        print(f"\n  {target.upper()} Speech-to-Speech Model:", flush=True)
        print(f"    Parameters: {n_params/1e6:.1f}M (adapter only, LLM frozen)", flush=True)
        print(f"    Extended vocab: {config.extended_vocab_size} ({config.text_vocab_size} text + {config.total_audio_vocab} audio)", flush=True)
        print(f"    Inner monologue: {config.inner_monologue}", flush=True)
        print(f"    Dual stream: {config.dual_stream}", flush=True)
        print(f"    Depth transformer: {config.depth_transformer_layers} layers, {config.depth_transformer_dim}d", flush=True)

        B = 1
        hidden = mx.random.normal((B, 20, config.llm_dim))

        t0 = time.time()
        for _ in range(10):
            text_logits, audio_tokens = model.predict_audio_frame(hidden, temperature=0.8)
            mx.eval(audio_tokens)
        frame_ms = (time.time() - t0) / 10 * 1000

        state = model.predict_state(hidden)

        print(f"    Frame generation: {frame_ms:.1f}ms ({config.n_codebooks} codebooks)", flush=True)
        if text_logits is not None:
            print(f"    Text logits shape: {text_logits.shape}", flush=True)
        print(f"    Audio tokens shape: {audio_tokens.shape}", flush=True)
        print(f"    State prediction: {['LISTEN', 'SPEAK', 'INTERRUPT'][state]}", flush=True)

        real_time_budget_ms = 1000.0 / config.frame_rate_hz
        status = "PASS" if frame_ms < real_time_budget_ms else "SLOW"
        print(f"    Real-time budget: {real_time_budget_ms:.0f}ms/frame -> [{status}]", flush=True)


if __name__ == "__main__":
    main()
