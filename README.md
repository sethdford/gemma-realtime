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
python3 scripts/extract_imessage_pairs.py
python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export

# Prepare training data
python3 scripts/prepare-training-data.py --voice

# Fine-tune (5-15 min on M4 Max)
python3 scripts/finetune-gemma.py --target e4b --data data/finetune

# Serve with real-time optimizations
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/adapters/persona \
  --realtime

# Prove it works
python3 scripts/voice-bench.py
```

### h-uman Integration

The MLX server is the default inference backend for [h-uman](https://github.com/user/h-uman). When `~/.human/config.json` exists, the server auto-reads model, adapter, port, and TurboQuant+ settings — no flags needed:

```bash
# Start via h-uman (auto-detects gemma-realtime)
~/.human/bin/human-serve.sh start

# Or run mlx-server.py directly (reads ~/.human/config.json)
python3 scripts/mlx-server.py
```

See [Real-Time Serving Guide](guides/04-real-time-serving.md#h-uman-integration) for config details.

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

### 5. TurboQuant+ KV cache compression

[TurboQuant+](https://github.com/TheTom/turboquant_plus) compresses the KV cache **3.8-6.4x** using PolarQuant + Walsh-Hadamard rotation. Integrated via the MLX port — `TurboKVCache` is a drop-in replacement for `mlx-lm`'s KVCache with zero framework changes. The `--realtime` flag auto-enables 4-bit TurboQuant+; add `--kv-asymmetric` for FP16 keys (best quality, less compression).

| Config | Compression | Quality (vs FP16) | Decode Speed |
|--------|------------|-------------------|-------------|
| turbo4 (4-bit) | 3.8x | +0.23% PPL | 97-100% baseline |
| turbo3 (3-bit) | 4.6x | +1.06% PPL | 90-93% baseline |
| asymmetric (K=FP16, V=turbo4) | ~2x | +0.51% PPL | 99% baseline |

```bash
# Install TurboQuant+ MLX fork
pip install git+https://github.com/TheTom/mlx.git@feature/turboquant-plus

# Real-time mode auto-enables TurboQuant+ 4-bit
python3 scripts/mlx-server.py --model mlx-community/gemma-4-e4b-it-4bit --realtime

# Asymmetric mode (best for Q4_K_M models — preserves K precision)
python3 scripts/mlx-server.py --model mlx-community/gemma-4-e4b-it-4bit --kv-bits 4 --kv-asymmetric
```

### 6. The AMX/SME2 coprocessor is 77x faster than NEON

Apple Silicon has an undocumented matrix coprocessor (AMX on M1-M3, SME2 on M4+) that Accelerate's BLAS uses internally. Direct benchmarking shows 2.5 TFLOPS FP32 — that's every matmul in every transformer layer running at GPU-like speeds on the CPU.

### 7. IOSurface enables zero-copy KV cache sharing

Apple's IOSurface lets CPU, GPU, and ANE access the same physical memory with no `memcpy`. The hybrid pipeline uses this for zero-copy KV cache: GPU writes during prefill, ANE reads during decode. Measured 5+ TB/s effective bandwidth.

## Secret APIs: The Hidden Performance Stack

We reverse-engineered, built, and benchmarked the undocumented hardware features that make real-time LLM inference possible on Apple Silicon:

| Layer | What It Is | Result |
|-------|-----------|--------|
| **AMX/SME2** | Undocumented CPU matrix coprocessor | **77x** over NEON, 2.5 TFLOPS FP32 |
| **Neural Engine** | Private `_ANEClient` API (67 classes discovered) | 15.8 TFLOPS FP16, 6.6 TFLOPS/W |
| **Direct ANE** | Bypass CoreML via `_ANEInMemoryModelDescriptor` | In-memory MIL compilation, training proven |
| **IOSurface** | Zero-copy shared memory across CPU/GPU/ANE | **5+ TB/s** effective bandwidth |
| **Metal 4 Tensor** | MTLTensor + Shader ML + ML Command Encoder | Full CoreML on GPU timeline |
| **M5 Neural Accel** | Per-GPU-core Neural Accelerators (10-40 units) | **4x** peak AI compute vs M4 |
| **Metal Dynamic** | MTLFunctionConstant kernel specialization | Fused attention for all Gemma configs |
| **Hybrid Pipeline** | GPU prefill + ANE decode + zero-copy KV cache | **1,333 tok/s**, 53x real-time margin |

```bash
# Build and run all 8 secret API benchmarks
cd secret-apis && make all && make bench
```

See [Guide 06: Secret APIs](guides/06-secret-apis.md) for the full deep dive, including the [maderix/ANE](https://github.com/maderix/ANE) reverse engineering work that proved training on the Neural Engine is possible.

## Project Structure

```
scripts/
├── extract_imessage_pairs.py    # Extract iMessage conversations (macOS)
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

secret-apis/
├── amx_matmul.c                 # AMX/SME2 coprocessor benchmark
├── amx.h                        # Reverse-engineered AMX instruction encodings
├── sme2_matmul.c                # ARM SME2 detection and benchmark
├── ane_probe.m                  # Neural Engine private API discovery
├── ane_direct.m                 # Direct ANE access (maderix/ANE findings, 67 classes)
├── iosurface_bridge.m           # IOSurface zero-copy bridge + Metal compute
├── metal_dynamic.m              # Dynamic kernel compilation + fused attention
├── metal4_tensor.m              # Metal 4 Tensor APIs + M5 Neural Accelerator probe
├── hybrid_pipeline.m            # Full GPU+ANE hybrid inference pipeline
├── bench_all_secrets.sh         # Run all benchmarks with report generation
└── Makefile                     # Build system

guides/
├── 01-quickstart.md             # Running in 10 minutes
├── 02-data-preparation.md       # iMessage, Facebook, WhatsApp, custom
├── 03-fine-tuning.md            # LoRA deep dive
├── 04-real-time-serving.md      # All 5 backends explained
├── 05-benchmarking.md           # Measuring and interpreting results
└── 06-secret-apis.md            # Apple Silicon secret performance stack
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
