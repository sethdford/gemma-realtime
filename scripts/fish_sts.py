#!/usr/bin/env python3
"""
True Speech-to-Speech on Fish Audio's Shoulders.

No text bottleneck. Audio tokens in, reasoning happens, audio tokens out.
Uses Fish Audio's pre-trained DAC codec (10-codebook RVQ, 44.1kHz, ~21 Hz)
and Fast AR depth model as the audio backbone, with Gemma providing reasoning
via MOSS-Speech-style modality-based layer splitting.

Architecture:
    User Audio (44.1kHz)
        → Fish DAC Encode → [cb0..cb9] tokens (~21 Hz, 10 codebooks)
        → Extract cb0 (semantic) → AudioInputProjection → Gemma embedding space
        → Gemma (frozen text weights, layer-split speech layers):
            Lower N layers: shared (text + speech)
            Upper M layers: split into text branch + speech branch
            Text branch → inner monologue (never spoken, grounds reasoning)
            Speech branch → agent cb0 tokens (semantic content)
        → Fish Fast AR: agent cb0 → cb1..cb9 (acoustic reconstruction)
        → Fish DAC Decode → Agent Audio (44.1kHz)

Why this works:
    1. Fish's codec is SOTA (10 codebooks, 10M+ hours training, GRPO-aligned)
    2. Fish's Fast AR (400M) replaces our hand-trained depth decoder
    3. MOSS-Speech layer splitting preserves Gemma's text reasoning intact
    4. Frozen pre-training: Gemma's text weights never change → no forgetting
    5. cb0 carries semantic meaning → Gemma can reason over speech directly

Training strategy (3 phases):
    Phase A: Projection warm-up (~1 hour on 1 GPU)
        Freeze everything. Train AudioInputProjection + AudioOutputHead only.
        Data: paired (audio, text) from LibriSpeech/LibriTTS.
        Loss: CTC alignment between projected cb0 and text embeddings.

    Phase B: Layer-split speech pre-training (~8 hours)
        Freeze Gemma text weights + Fish codec. Train speech-specific layers.
        Data: large-scale speech (Libri-Light 60k hours, GigaSpeech).
        Loss: next-token prediction on cb0 + inner monologue CE.

    Phase C: Joint STS fine-tuning (~24 hours)
        Freeze Gemma text weights + Fish codec. Fine-tune all speech layers.
        Data: spoken QA pairs (Spoken-SQuAD, conversational speech).
        Loss: cb0 generation + inner monologue + turn-state prediction.

Install:
    pip install fish-speech        # Fish Audio's codec + models
    # OR
    pip install descript-audio-codec huggingface_hub  # codec only

Usage:
    # Architecture validation
    python3 scripts/fish_sts.py

    # Integration with realtime-ws.py
    python3 scripts/realtime-ws.py --tts fish
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FishSTSConfig:
    """Configuration for the true STS model on Fish's codec."""

    # Gemma backbone
    llm_dim: int = 2560                 # Gemma E4B embedding dim
    text_vocab_size: int = 256000       # Gemma tokenizer vocab
    n_llm_layers: int = 42             # total Gemma transformer layers
    split_layer: int = 28              # shared layers before split (MOSS-Speech)

    # Fish DAC codec
    fish_codebook_size: int = 1024     # FSQ entries per codebook
    fish_n_codebooks: int = 10         # total RVQ codebooks
    fish_frame_rate: float = 21.53     # ~21 Hz (44100 / 2048)
    fish_sample_rate: int = 44100

    # Fast AR depth model (Fish's cb0 → cb1..cb9)
    fast_ar_dim: int = 512             # Fish Fast AR hidden dim
    fast_ar_layers: int = 4            # depth transformer layers
    fast_ar_heads: int = 8

    # Speech adapter dimensions
    speech_adapter_dim: int = 512
    speech_adapter_heads: int = 8
    speech_adapter_layers: int = 4
    speech_adapter_ff: int = 2048

    # Inner monologue
    inner_monologue: bool = True

    # Generation
    max_audio_frames: int = 750        # ~35s at 21 Hz
    max_text_tokens: int = 256

    @property
    def n_shared_layers(self) -> int:
        return self.split_layer

    @property
    def n_split_layers(self) -> int:
        return self.n_llm_layers - self.split_layer

    @property
    def audio_vocab_size(self) -> int:
        """cb0 vocabulary for Gemma's extended embedding."""
        return self.fish_codebook_size + 2  # +BOS, +EOS

    @property
    def extended_vocab_size(self) -> int:
        return self.text_vocab_size + self.audio_vocab_size

    AUDIO_TOKEN_OFFSET = 256000
    AUDIO_BOS = 256000 + 1024
    AUDIO_EOS = 256000 + 1025


PRESET_CONFIGS = {
    "e2b": FishSTSConfig(llm_dim=2304, n_llm_layers=26, split_layer=18),
    "e4b": FishSTSConfig(llm_dim=2560, n_llm_layers=42, split_layer=28),
    "27b": FishSTSConfig(llm_dim=4608, n_llm_layers=62, split_layer=42),
}


# ══════════════════════════════════════════════════════════════════════════════
# Audio ↔ Embedding Projections
# ══════════════════════════════════════════════════════════════════════════════

class AudioInputProjection(nn.Module):
    """Projects Fish cb0 tokens into Gemma's embedding space.

    Fish's cb0 (primary codebook) carries semantic content — this projection
    maps those discrete tokens into continuous vectors that Gemma's transformer
    can attend to alongside text embeddings.
    """

    def __init__(self, config: FishSTSConfig):
        super().__init__()
        self.cb0_embed = nn.Embedding(config.fish_codebook_size, config.speech_adapter_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.speech_adapter_dim, config.llm_dim),
            nn.LayerNorm(config.llm_dim),
            nn.GELU(),
            nn.Linear(config.llm_dim, config.llm_dim),
        )
        self.frame_pos = _SinusoidalPE(config.llm_dim, max_len=2048)

    def __call__(self, cb0_tokens: mx.array) -> mx.array:
        """Map cb0 token IDs to Gemma-compatible embeddings.

        Args:
            cb0_tokens: (batch, T) Fish cb0 token IDs

        Returns:
            embeddings: (batch, T, llm_dim) ready for Gemma's transformer
        """
        x = self.cb0_embed(cb0_tokens)
        x = self.proj(x)
        x = self.frame_pos(x)
        return x


class AudioOutputHead(nn.Module):
    """Predicts Fish cb0 tokens from Gemma's hidden states.

    The speech branch of the layer-split transformer outputs hidden states;
    this head converts them to cb0 logits for autoregressive generation.
    """

    def __init__(self, config: FishSTSConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.llm_dim)
        self.head = nn.Linear(config.llm_dim, config.fish_codebook_size + 1)

    def __call__(self, hidden: mx.array) -> mx.array:
        """Predict cb0 token logits.

        Args:
            hidden: (batch, T, llm_dim) from speech branch

        Returns:
            logits: (batch, T, codebook_size+1) including EOS
        """
        return self.head(self.norm(hidden))


# ══════════════════════════════════════════════════════════════════════════════
# MOSS-Speech Layer Splitting
# ══════════════════════════════════════════════════════════════════════════════

class SpeechBranchLayer(nn.Module):
    """One transformer layer for the speech-specific branch.

    After the shared lower layers, the speech branch processes hidden states
    with speech-specific parameters while the text branch uses the original
    Gemma layers (frozen). This is the core MOSS-Speech insight.
    """

    def __init__(self, config: FishSTSConfig):
        super().__init__()
        dim = config.llm_dim
        self.self_attn = nn.MultiHeadAttention(dim, config.speech_adapter_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, config.speech_adapter_ff),
            nn.GELU(),
            nn.Linear(config.speech_adapter_ff, dim),
        )
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        residual = x
        h = self.norm1(x)
        h = self.self_attn(h, h, h, mask=mask)
        x = residual + h

        residual = x
        h = self.norm2(x)
        x = residual + self.ff(h)
        return x


class LayerSplitAdapter(nn.Module):
    """MOSS-Speech modality-based layer splitting for Gemma.

    Architecture:
        Input tokens (text + audio)
            → Gemma layers 0..split_layer (shared, frozen)
            → Split:
                Text branch:   Gemma layers split_layer..N (frozen) → text logits
                Speech branch: SpeechBranchLayer x M (trainable) → audio logits

    The text branch is the frozen Gemma upper layers — its output becomes
    the "inner monologue" that guides reasoning without being spoken.
    The speech branch is fresh trainable layers that learn to produce audio.

    During frozen pre-training, only the speech branch + projections train.
    Gemma's text weights are never modified → no catastrophic forgetting.
    """

    def __init__(self, config: FishSTSConfig):
        super().__init__()
        self.config = config
        self.speech_layers = [
            SpeechBranchLayer(config) for _ in range(config.n_split_layers)
        ]
        self.speech_output = AudioOutputHead(config)

        if config.inner_monologue:
            self.monologue_alignment = nn.Linear(config.llm_dim, config.llm_dim)
            self.monologue_norm = nn.LayerNorm(config.llm_dim)

    def forward_speech_branch(self, shared_hidden: mx.array,
                              mask: Optional[mx.array] = None) -> mx.array:
        """Run the speech-specific upper layers.

        Args:
            shared_hidden: (batch, seq, llm_dim) output from shared Gemma layers

        Returns:
            cb0_logits: (batch, seq, codebook_size+1)
        """
        x = shared_hidden
        for layer in self.speech_layers:
            x = layer(x, mask=mask)
        return self.speech_output(x)

    def align_monologue(self, text_hidden: mx.array, speech_hidden: mx.array) -> mx.array:
        """Enhance speech hidden states with inner monologue signal.

        The text branch's hidden states carry linguistic reasoning that
        improves speech generation quality (Moshi's key finding).
        """
        if not self.config.inner_monologue:
            return speech_hidden
        text_signal = self.monologue_alignment(self.monologue_norm(text_hidden))
        return speech_hidden + 0.1 * text_signal


# ══════════════════════════════════════════════════════════════════════════════
# Fish Fast AR Depth Decoder (cb0 → cb1..cb9)
# ══════════════════════════════════════════════════════════════════════════════

class FishFastAR(nn.Module):
    """Depth transformer generating cb1..cb9 from cb0 at each timestep.

    This is the MLX equivalent of Fish's 400M Fast AR model. It can be:
      1. Initialized randomly and trained on Fish's codec data
      2. Loaded from Fish's pre-trained Fast AR weights (recommended)

    For each time frame, given cb0 and the temporal hidden state, it
    autoregressively generates cb1, cb2, ..., cb9 (9 residual codebooks).
    """

    def __init__(self, config: FishSTSConfig):
        super().__init__()
        dim = config.fast_ar_dim
        self.n_depth_codebooks = config.fish_n_codebooks - 1  # 9 residual

        self.context_proj = nn.Linear(config.llm_dim, dim)
        self.cb_embed = nn.Embedding(config.fish_codebook_size + 2, dim)
        self.cb_pos = nn.Embedding(config.fish_n_codebooks, dim)

        self.layers = [
            _DepthBlock(dim, config.fast_ar_heads)
            for _ in range(config.fast_ar_layers)
        ]
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, config.fish_codebook_size)

    def __call__(self, temporal_hidden: mx.array, cb0_token: mx.array) -> mx.array:
        """Generate all residual codebook tokens for one time frame.

        Args:
            temporal_hidden: (batch, 1, llm_dim) from speech branch
            cb0_token: (batch,) cb0 token for this frame

        Returns:
            depth_tokens: (batch, 9) cb1..cb9 tokens
        """
        context = self.context_proj(temporal_hidden)     # (B, 1, dim)
        cb0_emb = self.cb_embed(cb0_token)[:, None, :]  # (B, 1, dim)
        cb0_emb = cb0_emb + self.cb_pos(mx.zeros_like(cb0_token))[:, None, :]

        x = mx.concatenate([context, cb0_emb], axis=1)   # (B, 2, dim)
        tokens = []

        for cb_idx in range(self.n_depth_codebooks):
            for layer in self.layers:
                x = layer(x)

            logits = self.head(self.norm(x[:, -1:, :]))  # (B, 1, codebook_size)
            next_token = mx.argmax(logits, axis=-1).squeeze(-1)  # (B,)
            tokens.append(next_token)

            if cb_idx < self.n_depth_codebooks - 1:
                next_emb = self.cb_embed(next_token)[:, None, :]
                pos = self.cb_pos(mx.full(next_token.shape, cb_idx + 1, dtype=mx.int32))
                next_emb = next_emb + pos[:, None, :]
                x = mx.concatenate([x, next_emb], axis=1)

        return mx.stack(tokens, axis=-1)  # (B, 9)

    def generate_frame(self, temporal_hidden: mx.array, cb0_token: mx.array,
                       temperature: float = 0.6) -> mx.array:
        """Generate with temperature sampling for better diversity."""
        context = self.context_proj(temporal_hidden)
        cb0_emb = self.cb_embed(cb0_token)[:, None, :]
        cb0_emb = cb0_emb + self.cb_pos(mx.zeros_like(cb0_token))[:, None, :]

        x = mx.concatenate([context, cb0_emb], axis=1)
        tokens = []

        for cb_idx in range(self.n_depth_codebooks):
            for layer in self.layers:
                x = layer(x)

            logits = self.head(self.norm(x[:, -1:, :]).squeeze(1))
            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs)
            else:
                next_token = mx.argmax(logits, axis=-1)
            tokens.append(next_token)

            if cb_idx < self.n_depth_codebooks - 1:
                next_emb = self.cb_embed(next_token)[:, None, :]
                pos = self.cb_pos(mx.full(next_token.shape, cb_idx + 1, dtype=mx.int32))
                next_emb = next_emb + pos[:, None, :]
                x = mx.concatenate([x, next_emb], axis=1)

        return mx.stack(tokens, axis=-1)


class _DepthBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dim, n_heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        T = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=mask)
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Complete True STS Model
# ══════════════════════════════════════════════════════════════════════════════

class FishSpeechToSpeech(nn.Module):
    """True speech-to-speech model: audio in → reasoning → audio out.

    No text bottleneck. Stands on Fish Audio's shoulders for the codec
    and depth model, uses MOSS-Speech layer splitting for Gemma integration.

    Components:
        audio_input:    Fish cb0 tokens → Gemma embedding space
        layer_split:    Speech-specific upper layers (trainable)
        fast_ar:        Fish Fast AR for cb0 → cb1..cb9 depth decoding
        state_head:     LISTEN/SPEAK/INTERRUPT turn-taking prediction

    The frozen Gemma backbone is NOT stored here — it's loaded separately
    and the forward pass is composed at runtime (same pattern as speech_model.py).
    """

    def __init__(self, config: FishSTSConfig):
        super().__init__()
        self.config = config

        # Audio ↔ Gemma projections
        self.audio_input = AudioInputProjection(config)
        self.audio_embed = nn.Embedding(config.audio_vocab_size, config.llm_dim)

        # MOSS-Speech layer splitting
        self.layer_split = LayerSplitAdapter(config)

        # Fish Fast AR depth decoder
        self.fast_ar = FishFastAR(config)

        # Turn-taking state prediction
        self.state_head = nn.Sequential(
            nn.Linear(config.llm_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def encode_user_audio(self, cb0_tokens: mx.array) -> mx.array:
        """Encode user's speech (Fish cb0) into Gemma's embedding space.

        This replaces Whisper ASR — no text intermediate needed.
        """
        return self.audio_input(cb0_tokens)

    def forward_shared_layers(self, embeddings: mx.array,
                              gemma_shared_fn) -> mx.array:
        """Run input through Gemma's shared (frozen) lower layers.

        Args:
            embeddings: (batch, seq, llm_dim) — may mix text + audio embeddings
            gemma_shared_fn: callable that runs Gemma layers 0..split_layer

        Returns:
            shared_hidden: (batch, seq, llm_dim)
        """
        return gemma_shared_fn(embeddings)

    def forward_split(self, shared_hidden: mx.array,
                      gemma_text_fn=None) -> tuple[mx.array, Optional[mx.array]]:
        """Run the split: speech branch + optional text branch.

        Args:
            shared_hidden: output from shared layers
            gemma_text_fn: if provided, runs frozen Gemma upper layers for monologue

        Returns:
            speech_logits: (batch, seq, codebook_size+1)
            text_logits: (batch, seq, text_vocab) or None
        """
        # Speech branch (trainable)
        speech_hidden = shared_hidden
        T = shared_hidden.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        speech_logits = self.layer_split.forward_speech_branch(speech_hidden, mask=mask)

        # Text branch (frozen Gemma) for inner monologue
        text_logits = None
        if gemma_text_fn is not None and self.config.inner_monologue:
            text_hidden = mx.stop_gradient(gemma_text_fn(shared_hidden))
            speech_hidden_enhanced = self.layer_split.align_monologue(
                text_hidden, speech_hidden
            )
            speech_logits = self.layer_split.speech_output(speech_hidden_enhanced)
            text_logits = text_hidden  # raw hidden states for text loss

        return speech_logits, text_logits

    def decode_depth(self, cb0_tokens: mx.array, speech_hidden: mx.array,
                     temperature: float = 0.6) -> mx.array:
        """Run Fish Fast AR to get full 10-codebook frame from cb0.

        Args:
            cb0_tokens: (batch, T) generated cb0 tokens
            speech_hidden: (batch, T, llm_dim) from speech branch

        Returns:
            all_codes: (batch, 10, T) full codebook tokens for Fish DAC decode
        """
        B, T = cb0_tokens.shape
        depth_tokens = []

        for t in range(T):
            frame_hidden = speech_hidden[:, t:t+1, :]
            frame_cb0 = cb0_tokens[:, t]
            depth = self.fast_ar.generate_frame(frame_hidden, frame_cb0, temperature)
            depth_tokens.append(depth)

        depth_all = mx.stack(depth_tokens, axis=1)  # (B, T, 9)

        # Combine: cb0 (B, T, 1) + cb1..cb9 (B, T, 9) → (B, 10, T)
        cb0_expanded = cb0_tokens.reshape(B, T, 1)
        all_codes = mx.concatenate([cb0_expanded, depth_all], axis=-1)  # (B, T, 10)
        return all_codes.transpose(0, 2, 1)  # (B, 10, T)

    def predict_state(self, hidden: mx.array) -> int:
        """Predict LISTEN(0)/SPEAK(1)/INTERRUPT(2) from hidden states."""
        logits = self.state_head(hidden[:, -1, :])
        return mx.argmax(logits, axis=-1).item()

    def generate_cb0(self, shared_hidden: mx.array,
                     temperature: float = 0.8, top_k: int = 50,
                     max_frames: int = 500) -> tuple[mx.array, mx.array]:
        """Autoregressive cb0 generation from shared hidden states.

        This is the main generation loop for the speech branch.
        Produces semantic audio tokens frame-by-frame.

        Returns:
            cb0_tokens: (1, T) generated cb0 tokens
            speech_hidden: (1, T, llm_dim) hidden states for depth decoding
        """
        B = shared_hidden.shape[0]
        tokens = []
        hiddens = []

        bos = mx.full((B, 1), self.config.AUDIO_BOS - self.config.AUDIO_TOKEN_OFFSET,
                       dtype=mx.int32)
        prev_emb = self.audio_embed(bos)

        x = mx.concatenate([shared_hidden, prev_emb], axis=1)

        for step in range(max_frames):
            T = x.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            for layer in self.layer_split.speech_layers:
                x = layer(x, mask=mask)

            last_hidden = x[:, -1:, :]
            hiddens.append(last_hidden)

            logits = self.layer_split.speech_output(last_hidden).squeeze(1)

            if top_k > 0:
                top_vals = mx.sort(logits, axis=-1)[:, -top_k]
                logits = mx.where(logits < top_vals, -1e9, logits)

            if temperature > 0:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_token = mx.random.categorical(probs)
            else:
                next_token = mx.argmax(logits, axis=-1)

            eos_idx = self.config.fish_codebook_size
            if next_token.item() >= eos_idx:
                break

            tokens.append(next_token)
            next_emb = self.audio_embed(
                mx.clip(next_token, 0, self.config.audio_vocab_size - 1).reshape(B, 1)
            )
            x = mx.concatenate([x, next_emb], axis=1)

        if not tokens:
            return mx.zeros((B, 0), dtype=mx.int32), mx.zeros((B, 0, self.config.llm_dim))

        cb0 = mx.concatenate(tokens, axis=0).reshape(1, -1)
        hidden_out = mx.concatenate(hiddens, axis=1)
        return cb0, hidden_out

    def num_params(self) -> int:
        import mlx.utils
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class FishSTSPipeline:
    """Complete true STS pipeline: Fish codec + Gemma + Fish Fast AR.

    Usage:
        pipeline = FishSTSPipeline(target="e4b")
        pipeline.load()

        # True STS: audio in → audio out (no text intermediate)
        audio_out, metrics = pipeline.process_audio(audio_in_44k)

        # With inner monologue visibility
        audio_out, text_thought, metrics = pipeline.process_audio(
            audio_in_44k, return_monologue=True
        )
    """

    def __init__(self, target: str = "e4b",
                 gemma_model: str = "mlx-community/gemma-4-26b-a4b-it-4bit",
                 fish_model: str = "fishaudio/fish-speech-1.5"):
        self.config = PRESET_CONFIGS[target]
        self.gemma_model_id = gemma_model
        self.fish_model_id = fish_model
        self._model = None
        self._gemma = None
        self._tokenizer = None
        self._codec = None
        self._loaded = False

    def load(self, weights_path: Optional[str] = None):
        """Load all components.

        Args:
            weights_path: path to trained .safetensors.  Auto-searches
                ``adapters/fish-sts/phase-c/fish_sts_final.safetensors``,
                then phase-b, phase-a when *None*.
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from codec import AudioCodec

        print(f"\n{'='*60}")
        print(f"  Fish True STS Pipeline")
        print(f"{'='*60}")

        # 1. Fish DAC codec (with encode/decode sanity check)
        print("  Loading Fish DAC codec...", flush=True)
        self._use_snac_fallback = False
        try:
            _trial = AudioCodec("fish", fish_model=self.fish_model_id)
            _trial.load()
            _probe = np.zeros(int(_trial.sample_rate * 0.05), dtype=np.float32)
            _trial.encode(_probe)
            self._codec = _trial
        except Exception as e:
            del _trial  # free Fish DAC weights before loading fallback
            import gc; gc.collect()
            print(f"    Fish DAC unusable ({e}), using SNAC fallback", flush=True)
            self._codec = AudioCodec("snac")
            self._codec.load()
            self._use_snac_fallback = True

        # 2. STS model (speech adapter layers)
        print("  Loading STS model...", flush=True)
        w_path = self._resolve_weights(weights_path)
        if w_path is not None:
            self.config = self.config_from_weights(str(w_path))
            print(f"    Config inferred from {w_path.name}: "
                  f"llm_dim={self.config.llm_dim}, "
                  f"codebook={self.config.fish_codebook_size}, "
                  f"n_cb={self.config.fish_n_codebooks}", flush=True)

        self._model = FishSpeechToSpeech(self.config)
        n_params = self._model.num_params()
        print(f"    STS adapter: {n_params/1e6:.1f}M params", flush=True)
        print(f"    Layer split: {self.config.n_shared_layers} shared + "
              f"{self.config.n_split_layers} speech-specific", flush=True)
        print(f"    Fish codec: {self.config.fish_n_codebooks} codebooks, "
              f"~{self.config.fish_frame_rate:.0f} Hz", flush=True)
        print(f"    Inner monologue: {self.config.inner_monologue}", flush=True)

        if w_path is not None:
            w = mx.load(str(w_path))
            self._model.load_weights(list(w.items()), strict=False)
            print(f"    Loaded weights from {w_path}", flush=True)
        else:
            print("    WARNING: No trained weights — running with random init", flush=True)

        # 3. Gemma backbone (frozen)
        print("  Loading Gemma backbone...", flush=True)
        from mlx_lm import load as lm_load
        self._gemma, self._tokenizer = lm_load(self.gemma_model_id)
        if hasattr(self._gemma, "language_model"):
            self._inner = self._gemma.language_model.model
        else:
            self._inner = self._gemma.model

        self._loaded = True
        print(f"\n  True STS pipeline ready")
        print(f"  Audio in → Fish encode → Gemma reason → Fish decode → Audio out")
        print(f"{'='*60}\n")

    @staticmethod
    def _resolve_weights(explicit: Optional[str]) -> Optional[Path]:
        from pathlib import Path
        if explicit:
            p = Path(explicit)
            return p if p.is_file() else None
        repo = Path(__file__).resolve().parent.parent
        for rel in (
            "adapters/fish-sts/phase-c/fish_sts_final.safetensors",
            "adapters/fish-sts/phase-b/phase_b.safetensors",
            "adapters/fish-sts/phase-a/phase_a.safetensors",
        ):
            p = repo / rel
            if p.is_file():
                return p
        return None

    @staticmethod
    def config_from_weights(weights_path: str) -> "FishSTSConfig":
        """Infer FishSTSConfig from saved weight shapes (avoids mismatched dims)."""
        w = mx.load(weights_path)
        llm_dim = int(w["audio_input.proj.layers.0.weight"].shape[0])
        fish_cb = int(w["audio_input.cb0_embed.weight"].shape[0])
        n_cb = int(w["fast_ar.cb_pos.weight"].shape[0])
        n_speech_layers = sum(1 for k in w if k.startswith("layer_split.speech_layers.") and k.endswith(".norm1.weight"))
        has_monologue = "layer_split.monologue_alignment.weight" in w
        return FishSTSConfig(
            llm_dim=llm_dim,
            fish_codebook_size=fish_cb,
            fish_n_codebooks=n_cb,
            speech_adapter_dim=512,
            speech_adapter_heads=8,
            speech_adapter_layers=max(n_speech_layers, 4),
            speech_adapter_ff=2048,
            inner_monologue=has_monologue,
        )

    def _make_shared_fn(self):
        """Build a callable that runs Gemma's first ``split_layer`` layers."""
        layers = self._inner.layers[: self.config.split_layer]

        def shared_fn(embeddings: mx.array) -> mx.array:
            h = embeddings
            for layer in layers:
                h = layer(h, mask=None)
            return h

        return shared_fn

    def _make_text_fn(self):
        """Build a callable that runs Gemma's upper (frozen) layers."""
        layers = self._inner.layers[self.config.split_layer :]

        def text_fn(hidden: mx.array) -> mx.array:
            h = hidden
            for layer in layers:
                h = layer(h, mask=None)
            return h

        return text_fn

    def process_audio(self, audio: np.ndarray,
                      return_monologue: bool = False,
                      use_gemma_backbone: bool = True) -> tuple:
        """True speech-to-speech: audio in → audio out.

        Args:
            audio: float32 numpy array at the codec's native sample rate
                   (44.1 kHz for Fish DAC, 24 kHz for SNAC fallback).
            return_monologue: if True, also return inner monologue text.
            use_gemma_backbone: if True (default), route through Gemma's
                frozen shared layers before the speech branch.  Set False
                for a lightweight "adapters-only" path (faster, weaker).

        Returns:
            audio_out: float32 numpy at codec sample rate
            metrics: dict with timing info
            (text_thought: str — only if return_monologue=True)
        """
        import time
        from codec import CodecTokens, CodecType
        metrics = {}
        t_start = time.time()

        # Step 1: Encode user audio → discrete tokens
        t_enc = time.time()
        tokens = self._codec.encode(audio)
        cb0_input = mx.array(tokens.codes[0:1, :], dtype=mx.int32)
        metrics["encode_ms"] = (time.time() - t_enc) * 1000
        metrics["input_frames"] = tokens.n_frames

        # Step 2: Project cb0 into Gemma embedding space
        t_proj = time.time()
        audio_embeddings = self._model.encode_user_audio(cb0_input)
        mx.eval(audio_embeddings)
        metrics["project_ms"] = (time.time() - t_proj) * 1000

        # Step 3 (optional): Run through Gemma's frozen shared layers
        if use_gemma_backbone and hasattr(self, "_inner"):
            t_shared = time.time()
            shared_fn = self._make_shared_fn()
            shared_hidden = mx.stop_gradient(shared_fn(audio_embeddings))
            mx.eval(shared_hidden)
            metrics["shared_layers_ms"] = (time.time() - t_shared) * 1000
        else:
            shared_hidden = audio_embeddings

        # Step 4: Duplex state prediction
        state = self._model.predict_state(shared_hidden)
        metrics["duplex_state"] = ["LISTEN", "SPEAK", "INTERRUPT"][state]

        # Step 5: Generate response cb0 via speech branch
        t_gen = time.time()
        cb0_out, speech_hidden = self._model.generate_cb0(
            shared_hidden, temperature=0.8, top_k=50
        )
        mx.eval(cb0_out, speech_hidden)
        metrics["generate_ms"] = (time.time() - t_gen) * 1000
        metrics["output_frames"] = cb0_out.shape[-1] if cb0_out.size > 0 else 0

        if cb0_out.size == 0:
            sr = self._codec.sample_rate
            empty = np.zeros(int(sr * 0.1), dtype=np.float32)
            metrics["total_ms"] = (time.time() - t_start) * 1000
            return (empty, "", metrics) if return_monologue else (empty, metrics)

        # Step 6: Inner monologue via Gemma's frozen upper layers
        monologue_text = ""
        if return_monologue and self.config.inner_monologue and hasattr(self, "_inner"):
            try:
                text_fn = self._make_text_fn()
                text_hidden = mx.stop_gradient(text_fn(shared_hidden))
                text_logits = self._inner.embed_tokens.as_linear(text_hidden)
                text_ids = mx.argmax(text_logits[:, :20, :], axis=-1)[0].tolist()
                monologue_text = self._tokenizer.decode(text_ids)
            except Exception:
                monologue_text = "[monologue decode error]"

        # Step 7: Fish Fast AR depth decode → full codebook tokens
        t_depth = time.time()
        all_codes = self._model.decode_depth(cb0_out, speech_hidden)
        mx.eval(all_codes)
        metrics["depth_ms"] = (time.time() - t_depth) * 1000

        # Step 8: Codec decode → audio waveform
        t_dec = time.time()
        codec_type = (CodecType.SNAC if getattr(self, "_use_snac_fallback", False)
                      else CodecType.FISH_DAC)
        out_tokens = CodecTokens(
            codes=np.array(all_codes[0].tolist(), dtype=np.int64),
            n_codebooks=self.config.fish_n_codebooks,
            frame_rate=self.config.fish_frame_rate,
            codec_type=codec_type,
        )
        audio_out = self._codec.decode(out_tokens)
        metrics["decode_ms"] = (time.time() - t_dec) * 1000
        metrics["audio_duration_s"] = len(audio_out) / self._codec.sample_rate
        metrics["total_ms"] = (time.time() - t_start) * 1000

        if return_monologue:
            return audio_out, monologue_text, metrics
        return audio_out, metrics

    @property
    def loaded(self) -> bool:
        return self._loaded


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

class _SinusoidalPE(nn.Module):
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
        return x + self._pe[offset:offset + seq_len]


# ══════════════════════════════════════════════════════════════════════════════
# Architecture Validation
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Validate the Fish STS architecture — shapes, timing, memory."""
    import time

    for target in ["e2b", "e4b"]:
        config = PRESET_CONFIGS[target]
        model = FishSpeechToSpeech(config)
        n_params = model.num_params()

        print(f"\n  {target.upper()} Fish True STS Model:", flush=True)
        print(f"    Adapter params:    {n_params/1e6:.1f}M (Gemma frozen, Fish frozen)", flush=True)
        print(f"    Layer split:       {config.n_shared_layers} shared + "
              f"{config.n_split_layers} speech-specific", flush=True)
        print(f"    Fish codec:        {config.fish_n_codebooks} codebooks × "
              f"{config.fish_codebook_size} entries, ~{config.fish_frame_rate:.0f} Hz", flush=True)
        print(f"    Extended vocab:    {config.extended_vocab_size} "
              f"({config.text_vocab_size} text + {config.audio_vocab_size} audio)", flush=True)
        print(f"    Inner monologue:   {config.inner_monologue}", flush=True)
        print(f"    Max audio:         {config.max_audio_frames} frames "
              f"(~{config.max_audio_frames/config.fish_frame_rate:.0f}s)", flush=True)

        # Test audio input projection
        B = 1
        T_in = 50  # ~2.3s of audio
        cb0_in = mx.random.randint(0, config.fish_codebook_size, (B, T_in))
        t0 = time.time()
        audio_emb = model.encode_user_audio(cb0_in)
        mx.eval(audio_emb)
        proj_ms = (time.time() - t0) * 1000
        print(f"    Input projection:  {cb0_in.shape} → {audio_emb.shape} ({proj_ms:.1f}ms)", flush=True)

        # Test speech branch
        shared_hidden = mx.random.normal((B, T_in, config.llm_dim))
        t0 = time.time()
        T = shared_hidden.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        speech_logits = model.layer_split.forward_speech_branch(shared_hidden, mask=mask)
        mx.eval(speech_logits)
        branch_ms = (time.time() - t0) * 1000
        print(f"    Speech branch:     {shared_hidden.shape} → {speech_logits.shape} ({branch_ms:.1f}ms)", flush=True)

        # Test Fast AR depth decode
        cb0_out = mx.random.randint(0, config.fish_codebook_size, (B, 10))
        speech_h = mx.random.normal((B, 10, config.llm_dim))
        t0 = time.time()
        all_codes = model.decode_depth(cb0_out, speech_h)
        mx.eval(all_codes)
        depth_ms = (time.time() - t0) * 1000
        print(f"    Depth decode:      cb0(1,10) → {all_codes.shape} ({depth_ms:.1f}ms)", flush=True)

        # Test state prediction
        state = model.predict_state(shared_hidden)
        print(f"    State prediction:  {['LISTEN', 'SPEAK', 'INTERRUPT'][state]}", flush=True)

        # Real-time check
        frame_budget_ms = 1000.0 / config.fish_frame_rate
        total_per_frame = proj_ms / T_in + branch_ms / T_in + depth_ms / 10
        status = "PASS" if total_per_frame < frame_budget_ms else "SLOW"
        print(f"    Per-frame budget:  {frame_budget_ms:.0f}ms, actual ~{total_per_frame:.1f}ms [{status}]",
              flush=True)


if __name__ == "__main__":
    main()
