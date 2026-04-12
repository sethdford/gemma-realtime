# Guide 07: Real-Time Speech Pipeline

> From text LLM to full-duplex speech-to-speech on Apple Silicon.

## Overview

The speech pipeline transforms gemma-realtime from a text-only inference server into a bidirectional streaming speech system. It progresses through 6 phases, each building on the last:

| Phase | What | Latency Target | Key Script |
|-------|------|----------------|------------|
| 0 | Fix server gaps | N/A | `mlx-server.py` |
| 1 | Cascaded ASR + LLM + TTS | < 1.2s E2E | `speech-server.py` |
| 2 | WebSocket bidirectional API | < 1.2s E2E | `realtime-ws.py` |
| 3 | Neural audio codec | N/A (infra) | `codec.py` |
| 4 | Freeze-Omni speech adapters | < 500ms E2E | `speech_encoder.py`, `speech_decoder.py` |
| 5 | Inner monologue + dual-stream | < 300ms E2E | `speech_model.py` |
| 6 | Hardware acceleration | < 200ms E2E | `hw_accel.py` |
| 7 | Fish codec + true STS | < 250ms E2E | `fish_sts.py`, `codec.py` |

## Quick Start

### Phase 1: Cascaded Pipeline (fastest to try)

```bash
# Terminal 1: Start the MLX text server
python3 scripts/mlx-server.py --model mlx-community/gemma-4-e4b-it-4bit --realtime

# Terminal 2: Start the speech server
pip install mlx-whisper kokoro sounddevice aiohttp
python3 scripts/speech-server.py

# Or text-only mode (no microphone needed)
python3 scripts/speech-server.py --text-only
```

### Phase 2: WebSocket API

```bash
# Start the WebSocket server
pip install websockets
python3 scripts/realtime-ws.py --port 8742

# Connect with any WebSocket client
# Protocol: ws://localhost:8742/v1/realtime
```

### Benchmark

```bash
python3 scripts/speech-bench.py --rounds 5
```

### Voxtral speculative draft heads (MLP)

Lightweight heads in `scripts/voxtral_speculative.py` predict the next few audio frame codes from the LM hidden state so synthesis can run with reduced denoising steps and partial speculation. This path is **only for Voxtral TTS**, and is **not** the same as Gemma token-level speculative decoding in `mlx-server.py` (E2B draft LM) or the research EAGLE-style block in `hw_accel.py` (not wired into production Voxtral).

| Step | Command / artifact |
|------|-------------------|
| Collect | `python3 scripts/collect-draft-data.py --librispeech --output data/draft-pairs-libri.npz` |
| Train | `python3 scripts/voxtral_speculative.py train --data data/draft-pairs-libri.npz --output adapters/draft-heads/heads-libri.safetensors` |
| Benchmark | `python3 scripts/benchmark-tts.py --output proof-artifacts/benchmark-libri.json` (auto-finds draft heads) |
| E2E proof | `python3 scripts/prove-voxtral-speculative-e2e.py` |

**Auto-loading:** `speech-server.py` and `realtime-ws.py` with `--tts voxtral` resolve weights via `scripts/draft_heads_resolve.py`: optional `--draft-heads PATH`, or environment variable `VOXTRAL_DRAFT_HEADS` (set to `0` / `false` / `none` to disable), otherwise `adapters/draft-heads/heads-libri.safetensors` then `heads.safetensors` if present. **`--no-draft-heads`** forces baseline Voxtral without speculation.

**Voice match:** Weights from the Libri overnight pipeline are trained with `cheerful_male`; other voices may lower draft acceptance until you re-collect.

## Architecture

### Phase 1-2: Cascaded Pipeline

```
Microphone -> Silero VAD -> Whisper ASR -> text
text -> Gemma E4B (mlx-server.py) -> text deltas (SSE stream)
text deltas -> Sentence Buffer (env: GEMMA_MIN_FLUSH_CHARS, GEMMA_MAX_BUFFER_CHARS) -> Kokoro TTS -> Speaker
```

Components:
- **Silero VAD**: Voice activity detection, <1ms per frame, ONNX on CPU
- **Whisper ASR**: `mlx-whisper` on Apple Silicon, ~300ms for 2s audio
- **Sentence Buffer**: Flushes at `.!?` boundaries or every 120 chars
- **Kokoro TTS**: 82M params, StyleTTS 2 architecture, <300ms first audio

### Phase 3: Neural Audio Codec

Replaces raw PCM with discrete tokens via SNAC (Multi-Scale Neural Audio Codec):

```
Audio (24kHz) -> SNAC Encoder -> Discrete Tokens (3 codebooks, 12/23/47 Hz)
Discrete Tokens -> SNAC Decoder -> Audio (24kHz)
```

Supported codecs:

| Codec | Frame Rate | Bitrate | Codebooks | Streaming |
|-------|-----------|---------|-----------|-----------|
| SNAC | 12 Hz | 2.4 kbps | 3 | Yes |
| Mimi | 12.5 Hz | 1.1 kbps | 8 | Yes |
| EnCodec | 75 Hz | 6.0 kbps | 8 | No |

### Phase 4: Freeze-Omni Speech Adapters

The key architectural change. Gemma stays frozen (preserving LoRA personalization), and trainable speech modules are added:

```
Audio Chunks (160ms)
  -> Conv Feature Extractor (downsample)
  -> Transformer Encoder (2 layers)
  -> Linear Projection -> [Gemma embedding space]

Gemma E4B (FROZEN + LoRA)
  -> Hidden States

Hidden States
  -> Linear Adapter
  -> AR Transformer Decoder (4 layers, causal)
  -> Codebook Projection -> SNAC tokens
  -> SNAC Decode -> Audio
```

#### Training (3 stages)

```bash
# Full pipeline
python3 scripts/train-speech-adapter.py --target e4b --data-dir ~/speech-data

# Individual stages
python3 scripts/train-speech-adapter.py --stage 1 --asr-data ~/librispeech/train.jsonl
python3 scripts/train-speech-adapter.py --stage 2 --tts-data ~/ljspeech/train.jsonl
python3 scripts/train-speech-adapter.py --stage 3 --qa-data ~/qa-pairs/train.jsonl

# Architecture validation (synthetic data)
python3 scripts/train-speech-adapter.py --target e4b --validate-only
```

**Stage 1** trains the speech encoder on ASR data (LibriSpeech recommended).
**Stage 2** trains the speech decoder on TTS data (LJSpeech + VCTK).
**Stage 3** fine-tunes everything jointly with Q&A data + state prediction.

#### Model Sizes

| Target | Encoder | Decoder | State | Total Adapter |
|--------|---------|---------|-------|--------------|
| E2B | 3.2M | 8.5M | 0.6M | 12.3M |
| E4B | 5.8M | 15.2M | 0.7M | 21.7M |
| 31B | 12.1M | 28.4M | 1.1M | 41.6M |

### Phase 5: Inner Monologue + Dual-Stream

Research-frontier features from Moshi (Kyutai):

**Inner Monologue**: The model predicts text tokens alongside audio tokens. These text tokens are never spoken — they're the model's "thoughts" that guide linguistic quality. This dramatically reduces hallucination and improves coherence in generated speech.

**Dual-Stream**: Models both user and agent audio simultaneously, enabling:
- Full-duplex conversation (speak and listen at the same time)
- Natural barge-in handling
- Overlap and interruption modeling

**Token schedule** per 80ms frame:
```
[user_semantic] [user_acoustic_1..N] [text_token] [agent_semantic] [agent_acoustic_1..N]
```

**Vocabulary extension**: 12,288 new tokens (3 codebooks x 4096) added to Gemma's 256K vocabulary. Original embeddings are frozen; only audio embeddings are trainable.

### Phase 6: Hardware Acceleration

**Layer-Adaptive TurboQuant+**: First/last 4 layers in FP16 (quality-critical), middle layers at 3-bit TurboQuant (4.6x compression). Near-FP16 quality at ~3.5x total memory savings.

**libgemma_hw + IOSurface (default on macOS)**: Build with `make -C secret-apis libgemma_hw`. `IOSurfaceKVManager` in `hw_accel.py` uses real IOSurface memory when the dylib is present (CPU views stay locked until `release_all`). `native_hw.py` exposes Accelerate SGEMM and a **Metal selftest** (`newBufferWithBytesNoCopy` + compute shader) used by `prove-native-hw.py` and red-team Phase 6.

**mlx-server.py**: `GET /health` includes an `iosurface` object (dylib path, load status) and **`inference_tuning`** (speculative draft count, KV scheme, realtime flag). Optional staging: set `GEMMA_IOSURFACE_KV_BYTES` (e.g. `65536`); disable with `GEMMA_IOSURFACE_KV=0`. For a full env/benchmark checklist, see [Guide 08: Inference SOTA roadmap](08-inference-sota-roadmap.md).

**realtime-ws.py**: After shared models load, logs a one-line `native_hw.health_payload_for_http` summary on macOS.

**IOSurface Zero-Copy KV**: Shared memory between GPU and ANE for split inference. GPU does prefill, ANE does decode — no memory copies between processors. Metal wrapping of IOSurface-backed pointers is proven in-library; MLX KV still uses its own buffers unless you bridge explicitly.

**EAGLE Draft Head (research)**: Sketch in `hw_accel.py` — lightweight head for audio-conditioned speculation at the **text-token** level; not the same as the **Voxtral MLP draft heads** (`voxtral_speculative.py`) used for frame-level codec prediction in TTS.

### Phase 7: Fish Codec + True Speech-to-Speech

Standing on Fish Audio's shoulders for true STS — no text bottleneck.

**Architecture: MOSS-Speech layer splitting on Fish's codec**

```
User Audio (44.1kHz)
  → Fish DAC Encode → [cb0..cb7] tokens (~21 Hz, 8 FSQ groups)
  → Extract cb0 (semantic) → Project to Gemma embedding space
  → Gemma (frozen, layer-split):
      Layers 0..K:   shared (text + speech, frozen)
      Layers K..N:   text branch (frozen → inner monologue)
      Speech layers: trainable (audio token generation)
  → Fish Fast AR: cb0 → cb1..cb7 (depth decoder)
  → Fish DAC Decode → Agent Audio (44.1kHz)
```

**Why Fish's codec over SNAC:**

| | SNAC | Fish DAC |
|---|---|---|
| Codebooks | 3 × 4096 | 8 groups × 1000 (FSQ) |
| Frame rate | 12 Hz | ~21 Hz |
| Sample rate | 24 kHz | 44.1 kHz |
| Training data | - | 10M+ hours |
| Alignment | - | GRPO |
| Depth decoder | ~2M (hand-trained) | 400M (pre-trained Fast AR) |
| Quality | Good | SOTA (lowest WER on Seed-TTS Eval) |

**Why MOSS-Speech layer splitting:**
- Gemma's text weights stay frozen → no catastrophic forgetting
- Lower layers are shared between text and speech (general representations)
- Upper layers split: text branch = inner monologue, speech branch = audio generation
- Inner monologue dramatically improves linguistic quality (Moshi's key finding)
- Speech layers are small (~50M) — train in hours, not days

**Usage:**

```bash
# Architecture validation (no models needed)
python3 scripts/fish_sts.py

# As TTS backend in the WebSocket server (auto-loads trained weights)
python3 scripts/realtime-ws.py --tts fish

# Codec test
python3 scripts/codec.py --codec fish
```

**Training:**

Two codec modes: `snac` (fast iteration, 3 codebooks, 4096 vocab) or `fish` (production quality, 8 FSQ groups, 1000 vocab).

```bash
# SNAC proxy training (quick, lower-fidelity)
python3 scripts/train-fish-sts.py all --codec snac

# Fish DAC training (production quality — requires extract step first)
python3 scripts/train-fish-sts.py extract \
  --input data/libritts-codec-train-full-eos.jsonl \
  --output data/libritts-fish-dac-tokens.jsonl
python3 scripts/train-fish-sts.py all --codec fish

# Individual phases (both codecs)
python3 scripts/train-fish-sts.py phase-a --codec fish --iters 2000
python3 scripts/train-fish-sts.py phase-b --codec fish --iters 20000
python3 scripts/train-fish-sts.py phase-c --codec fish --iters 50000
```

Weights are saved to `adapters/fish-sts/{phase-a,phase-b,phase-c}/`. The pipeline auto-resolves weights in order: Phase C → B → A. `config_from_weights()` infers codec dimensions from saved weights, so switching between SNAC and Fish DAC trained weights is automatic.

**Fish DAC loader (`fish_dac_loader.py`):** Standalone loader that reconstructs Fish Audio's `FireflyArchitecture` (ConvNeXtEncoder + FSQ quantizer + HiFiGAN) from the raw `.pth` checkpoint, enabling full 8-codebook, 44.1kHz encode/decode without installing `fish-speech`.

**Proving and evaluation:**

```bash
# Comprehensive proof (codec roundtrip, projection alignment, full audio→audio, duplex state)
python3 scripts/prove-fish-sts.py

# Quantitative evaluation (WER via Whisper re-transcription, MOS proxy, speaker similarity, RTF)
python3 scripts/eval_sts.py --pipeline fish
python3 scripts/eval_sts.py --pipeline cascaded  # Voxtral TTS baseline for comparison
```

**What's wired end-to-end in `FishSTSPipeline.process_audio`:**

1. Codec encode (Fish DAC or SNAC fallback)
2. `AudioInputProjection`: cb0 → Gemma embedding space
3. Gemma shared layers (frozen lower N layers via `_make_shared_fn`)
4. Duplex state prediction (LISTEN / SPEAK / INTERRUPT)
5. `generate_cb0`: speech branch AR generation
6. Inner monologue extraction (Gemma frozen upper layers → text logits)
7. `FishFastAR` depth decode: cb0 → full codebook codes
8. Codec decode → audio waveform

**Key research references:**

| Paper | Key Idea | How We Use It |
|-------|----------|---------------|
| Fish Audio S2 (arXiv:2603.08823) | Dual-AR TTS, 8-group FSQ, GRPO | Codec + Fast AR depth model |
| MOSS-Speech (arXiv:2510.00499) | True STS, layer splitting, no text guidance | Layer splitting for Gemma |
| Moshi (arXiv:2410.00037) | Inner monologue, dual-stream | Inner monologue training signal |

## WebSocket Protocol

The realtime WebSocket API (`ws://localhost:8742/v1/realtime`) follows the OpenAI Realtime API shape:

### Client Messages

| Type | Fields | Description |
|------|--------|-------------|
| `audio.chunk` | `data` (base64 PCM s16le 24kHz) | Stream audio input |
| `audio.commit` | — | End of utterance, trigger response |
| `text.input` | `text` | Direct text input (no ASR) |
| `config` | `voice`, `vad_threshold`, `system_prompt` | Update session config |
| `session.close` | — | End session |

### Server Messages

| Type | Fields | Description |
|------|--------|-------------|
| `session.created` | `session_id`, `capabilities` | Connection established |
| `transcript.partial` | `text`, `confidence` | Streaming ASR partial |
| `transcript.final` | `text` | Final ASR transcript |
| `response.start` | — | LLM generation starting |
| `text.delta` | `text` | Streaming text token |
| `text.done` | `text` | Full response text |
| `audio.chunk` | `data`, `seq`, `format` | Audio output chunk |
| `audio.done` | — | Audio generation complete |
| `response.done` | `latency` | Response metrics |

## Benchmarking

```bash
# Full speech pipeline benchmark
python3 scripts/speech-bench.py --rounds 5 --json

# Compare text backends
python3 scripts/bench-all-backends.py

# Test codec roundtrip
python3 scripts/codec.py --codec snac --duration 2.0

# Validate speech model architecture
python3 scripts/speech_model.py
python3 scripts/speech_encoder.py
python3 scripts/speech_decoder.py

# Test hardware acceleration
python3 scripts/hw_accel.py
```

### Voice Targets

| Metric | Phase 1 | Phase 4 | Phase 5 | Phase 6 | Phase 7 |
|--------|---------|---------|---------|---------|---------|
| E2E latency | < 1.2s | < 500ms | < 300ms | < 200ms | < 250ms |
| TTFT | < 500ms | < 200ms | < 150ms | < 100ms | < 120ms |
| RTF | < 1.0 | < 0.5 | < 0.3 | < 0.2 | < 0.3 |
| Memory | < 8 GB | < 6 GB | < 6 GB | < 5 GB | < 7 GB |
| Audio quality | Kokoro | SNAC 3cb | SNAC 3cb | SNAC 3cb | Fish 10cb |
| Text bottleneck | Yes | Yes | Partial | Partial | **None** |

## Research References

| Paper | Key Idea | Applied In |
|-------|----------|-----------|
| Moshi (arXiv:2410.00037) | Full-duplex, inner monologue, dual-stream | Phase 5 |
| Freeze-Omni (arXiv:2411.00774) | Frozen LLM + speech adapters | Phase 4 |
| VITA-Audio (arXiv:2505.03739) | Interleaved text-audio tokens | Phase 5 |
| TurboQuant (arXiv:2504.19874) | KV cache compression | Phase 6 |
| SNAC (arXiv:2410.14411) | Multi-scale neural audio codec | Phase 3 |
| EAGLE (arXiv:2508.08192) | Speculative decoding at scale | Phase 6 |
| SpeakStream (arXiv:2505.19206) | Streaming TTS from LLM output | Phase 1 |
| Kokoro | 82M param TTS, StyleTTS 2 | Phase 1 |
| DuplexMamba (arXiv:2502.11123) | Mamba-based duplex speech | Future |
| Llama-Mimi (arXiv:2509.14882) | Interleaved semantic+acoustic tokens | Phase 5 |
| MOSS-Speech (arXiv:2510.00499) | True S2S without text guidance | Phase 7 |
| Fish Audio S2 (arXiv:2603.08823) | Dual-AR TTS, 8-group FSQ codec, GRPO | Phase 7 |
| Voxtral Realtime (arXiv:2602.11298) | Streaming ASR via vLLM WebSocket | Phase 1 alt |

## File Map

```
scripts/
├── mlx-server.py            # Phase 0: Fixed text LLM server (OpenAI API)
├── speech-server.py          # Phase 1: Cascaded ASR + LLM + TTS pipeline
├── speech-bench.py           # Phase 1: E2E speech latency benchmark
├── realtime-ws.py            # Phase 2: WebSocket bidirectional API
├── codec.py                  # Phase 3: Neural audio codec (SNAC/Mimi/EnCodec)
├── speech_encoder.py         # Phase 4: Speech encoder adapter (MLX nn.Module)
├── speech_decoder.py         # Phase 4: Speech decoder + state predictor
├── train-speech-adapter.py   # Phase 4: 3-stage training pipeline
├── speech_model.py           # Phase 5: Full S2S model (inner monologue, dual-stream)
├── hw_accel.py               # Phase 6: TurboQuant+, IOSurface, EAGLE
├── fish_sts.py               # Phase 7: True STS on Fish Audio's codec + MOSS-Speech
├── fish_dac_loader.py        # Phase 7: Standalone Fish Audio DAC codec loader
├── train-fish-sts.py         # Phase 7: STS training pipeline (--codec snac|fish)
├── eval_sts.py               # Phase 7: Eval harness (WER, MOS, speaker sim, RTF)
├── voxtral_speculative.py    # Voxtral draft heads + speculative decoding
├── draft_heads_resolve.py    # Auto-resolve draft head weights
└── benchmark-tts.py          # TTS benchmark (standard vs speculative vs denoised)

configs/
├── example-training-config.json   # Text fine-tuning config
└── speech-adapter-config.json     # Speech adapter training config
```
