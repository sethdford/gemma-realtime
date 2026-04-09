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

## Architecture

### Phase 1-2: Cascaded Pipeline

```
Microphone -> Silero VAD -> Whisper ASR -> text
text -> Gemma E4B (mlx-server.py) -> text deltas (SSE stream)
text deltas -> Sentence Buffer -> Kokoro TTS -> Speaker
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

**IOSurface Zero-Copy KV**: Shared memory between GPU and ANE for split inference. GPU does prefill, ANE does decode — no memory copies between processors.

**EAGLE Draft Head**: Lightweight (2-layer, 256-dim) head trained via online distillation to predict target model hidden states. Enables audio-conditioned speculative decoding at 4+ tokens per step.

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

| Metric | Phase 1 | Phase 4 | Phase 5 | Phase 6 |
|--------|---------|---------|---------|---------|
| E2E latency | < 1.2s | < 500ms | < 300ms | < 200ms |
| TTFT | < 500ms | < 200ms | < 150ms | < 100ms |
| RTF | < 1.0 | < 0.5 | < 0.3 | < 0.2 |
| Memory | < 8 GB | < 6 GB | < 6 GB | < 5 GB |

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
| MOSS-Speech (arXiv:2510.00499) | True S2S without text guidance | Future |
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
└── hw_accel.py               # Phase 6: TurboQuant+, IOSurface, EAGLE

configs/
├── example-training-config.json   # Text fine-tuning config
└── speech-adapter-config.json     # Speech adapter training config
```
