<p align="center">
  <img src="docs/assets/logo.svg" alt="gemma-realtime" width="120" />
</p>

<h1 align="center">gemma-realtime</h1>

<p align="center">
  <strong>Personalize Gemma 4. Make it real-time. Run it locally.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &nbsp;&bull;&nbsp;
  <a href="guides/01-quickstart.md">Guides</a> &nbsp;&bull;&nbsp;
  <a href="#benchmark-results">Benchmarks</a> &nbsp;&bull;&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;&bull;&nbsp;
  <a href="https://sethdford.github.io/gemma-realtime">Website</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Apple_Silicon-M1%2B-000?logo=apple&logoColor=white" alt="Apple Silicon" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/MLX-Inference-FF6F00?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSIjZmZmIj48cGF0aCBkPSJNMTIgMkw0IDdWMTdMMTIgMjJMMjAgMTdWN0wxMiAyWiIvPjwvc3ZnPg==" alt="MLX" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/badge/Gemma_4-E4B%20%7C%20E2B%20%7C%2031B-4285F4?logo=google&logoColor=white" alt="Gemma 4" />
</p>

---

Fine-tune Google's Gemma 4 on your own conversations (iMessage, Facebook Messenger, etc.) and serve it at **real-time voice speeds** on Apple Silicon. No cloud. No API keys. Your data never leaves your machine.

## Benchmark Results

Measured on M4 Max (128GB unified memory), Gemma 4 E4B 4-bit quantization:

```
==========================================================================================
  HEAD-TO-HEAD BENCHMARK RESULTS — Gemma 4 E4B on Apple Silicon
==========================================================================================

  Backend                        TTFT P50   TTFT P95   TPS P50  TPS Mean    Verdict
  ──────────────────────────  ────────  ────────  ───────  ───────  ────────
  MLX Server (mlx_lm)               154ms      889ms   111.6    111.4   REAL-TIME
  Ollama (Go + llama.cpp)           141ms      150ms   107.9    102.7   REAL-TIME
  llama.cpp Metal                   136ms      141ms    94.0     94.1   REAL-TIME

  All backends generate text 20-30x faster than human speech (RTF < 0.05).
==========================================================================================
```

> **111 tokens/sec** with a personalized fine-tuned model. That's Gemini Live territory — running locally on your Mac.

## The Problem

You want an AI that sounds like *you* — not a generic chatbot. And you want it fast enough for real-time voice conversation. Until now, that required cloud APIs and sending your data to someone else's servers.

## The Solution

```
Your Messages ──→ Fine-Tune ──→ Serve Locally ──→ Real-Time Voice
  (iMessage)       (5 min)      (111 tok/s)       (your style)
  (Facebook)
  (WhatsApp)
```

**gemma-realtime** gives you the complete pipeline:

1. **Extract** your conversations from iMessage, Facebook, or any messaging platform
2. **Fine-tune** Gemma 4 with LoRA in minutes on Apple Silicon
3. **Serve** at real-time speeds through your choice of optimized backend
4. **Benchmark** to prove it meets voice latency targets

## Quick Start

```bash
# Install
pip install mlx mlx-lm

# Extract your data (pick one or both)
python3 scripts/extract-imessage.py
python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export

# Prepare training data
python3 scripts/prepare-training-data.py --voice

# Fine-tune (5-15 min on M4 Max)
python3 scripts/finetune-gemma.py --target e4b --data data/finetune

# Serve with real-time optimizations
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/training-data/adapters/seth-lora-e4b \
  --realtime

# Prove it works
python3 scripts/voice-bench.py
```

## Architecture

```
                         ┌──────────────────────────┐
                         │     Your Conversations    │
                         │  iMessage · Facebook · …  │
                         └────────────┬─────────────┘
                                      │
                              extract & prepare
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      Training Data        │
                         │    train.jsonl (JSONL)     │
                         └────────────┬─────────────┘
                                      │
                          LoRA fine-tune (SFT + DPO)
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
             ┌────────────┐  ┌──────────────┐  ┌──────────────┐
             │  E4B (4B)   │  │  E2B (2B)    │  │  31B (dense) │
             │  110 tok/s  │  │  180 tok/s   │  │  20 tok/s    │
             │  Voice      │  │  Draft       │  │  Quality     │
             └──────┬─────┘  └──────┬───────┘  └──────┬───────┘
                    │               │                  │
                    └───────┬───────┘                  │
                            ▼                          ▼
                    ┌──────────────┐          ┌──────────────┐
                    │  MLX Server  │          │  MLX Server  │
                    │  Ollama      │          │  (high qual) │
                    │  llama.cpp   │          └──────────────┘
                    │  vLLM Metal  │
                    └──────────────┘
                     Real-Time Voice
```

### Model Targets

| Target | Params | Speed | Use Case |
|--------|--------|-------|----------|
| **E4B** | 4B | ~110 tok/s | Real-time voice, daily driver |
| **E2B** | 2B | ~180 tok/s | Speculative decode draft model |
| **31B** | 31B | ~20 tok/s | Highest quality, complex reasoning |

### Serving Backends

| Backend | Language | Key Advantage | Best For |
|---------|----------|---------------|----------|
| **MLX Server** | Python | Highest throughput, LoRA hot-swap | Development |
| **Ollama** | Go | Zero GIL, most consistent latency | Production |
| **llama.cpp** | C++ | Lowest TTFT, fused Metal kernels | First-word speed |
| **vLLM Metal** | Python | Paged attention, continuous batching | Multi-user |
| **ANE+GPU** | Python | Dual-compute speculative decode | Experimental |

## Key Discoveries

Things we learned making Gemma 4 real-time on Apple Silicon:

### 1. `mlx_lm` vs `mlx_vlm` — 10x speedup

Switching from the multimodal `mlx_vlm` import to text-only `mlx_lm` gave a **10x speedup** (13 → 110+ tok/s). The VLM path adds numpy synchronization overhead even for text-only inference.

### 2. PLE-safe quantization is critical

Gemma 4 uses `ScaledLinear` layers that most community quantizations corrupt. Only PLE-safe quants (which skip these layers) produce correct output. The scripts detect and warn about broken models automatically.

### 3. Go eliminates the Python GIL bottleneck

Ollama wraps llama.cpp in a Go server — no GIL serialization on token dispatch. Result: the most consistent P50-to-P95 latency spread of any backend.

### 4. Fused Metal kernels matter for TTFT

llama.cpp's fused RoPE+attention shaders give the fastest time-to-first-token. For voice UX, the first word is what the user feels.

### 5. TurboQuant 3-bit KV cache

Compresses the key-value cache by 4.6x with ~2% quality loss. Critical for long conversations on memory-constrained devices.

## Project Structure

```
scripts/
├── extract-imessage.py          # Extract iMessage conversations (macOS)
├── extract-facebook.py          # Extract Facebook Messenger data
├── prepare-training-data.py     # Combine sources → train/valid splits
├── finetune-gemma.py            # LoRA pipeline (SFT + DPO + quantize)
├── mlx-server.py                # MLX inference server (OpenAI-compatible)
├── ollama-serve.sh              # Ollama serving script
├── llamacpp-serve.sh            # llama.cpp Metal server
├── vllm-metal-serve.sh          # vLLM Metal server
├── ane-gpu-bridge.py            # ANE+GPU dual-compute bridge
├── voice-bench.py               # Single-backend voice benchmark
└── bench-all-backends.py        # Head-to-head comparison

guides/
├── 01-quickstart.md             # Running in 10 minutes
├── 02-data-preparation.md       # iMessage, Facebook, WhatsApp, custom
├── 03-fine-tuning.md            # LoRA deep dive
├── 04-real-time-serving.md      # All 5 backends explained
└── 05-benchmarking.md           # Measuring and interpreting results
```

## Hardware Requirements

| Hardware | E2B (2B) | E4B (4B) | 31B |
|----------|----------|----------|-----|
| **Minimum** | M1, 8GB | M1 Pro, 16GB | M2 Max, 64GB |
| **Recommended** | M2+, 16GB | M3 Pro+, 36GB | M4 Max, 128GB |
| **Expected TPS** | 150-200 | 80-120 | 15-25 |

Apple Silicon's unified memory is the key enabler — GPU reads model weights directly from main memory with no copying.

## Guides

| Guide | Description |
|-------|-------------|
| [Quick Start](guides/01-quickstart.md) | Get running in 10 minutes |
| [Data Preparation](guides/02-data-preparation.md) | Extract from iMessage, Facebook, WhatsApp |
| [Fine-Tuning](guides/03-fine-tuning.md) | LoRA hyperparameters, DPO, quantization |
| [Real-Time Serving](guides/04-real-time-serving.md) | Backend setup and optimization |
| [Benchmarking](guides/05-benchmarking.md) | Measure TTFT, TPS, RTF |

## Privacy

Everything runs locally. Your conversation data, training process, and inference all happen on your Mac:

- Extracted JSONL files contain your messages — treat as sensitive
- The LoRA adapter encodes your communication style — keep private
- No network calls during extraction, training, or inference
- The `.gitignore` excludes all data and model files by default

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is most needed:
- More data extractors (Telegram, Discord, Signal, WhatsApp native)
- CoreML/ANE optimization for the draft model
- Windows/Linux support (currently macOS-focused)
- Voice pipeline integration (STT → inference → TTS)

## License

MIT. See [LICENSE](LICENSE).

Gemma models are licensed under [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).
