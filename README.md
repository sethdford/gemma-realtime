# gemma-realtime

**Personalize Gemma 4 and make it real-time on Apple Silicon.**

Fine-tune Google's Gemma 4 models on your own conversation data (iMessage, Facebook Messenger, etc.) and serve them at real-time voice speeds on a Mac — no cloud required.

## Benchmark Results (M4 Max, 128GB)

| Backend | TTFT P50 | TPS P50 | TPS Mean | RTF | Verdict |
|---------|----------|---------|----------|-----|---------|
| **MLX Server** (mlx_lm) | 154ms | **111.6** | 111.4 | 0.029 | **REAL-TIME** |
| **Ollama** (Go + llama.cpp) | 141ms | 107.9 | 102.7 | 0.031 | **REAL-TIME** |
| **llama.cpp** Metal | **136ms** | 94.0 | 94.1 | 0.040 | **REAL-TIME** |

All three backends generate text **20-30x faster than a human speaks** (RTF < 0.05). That's Gemini Live territory, running locally on your Mac with your personalized fine-tuned model.

## Quick Start

```bash
# 1. Install dependencies
pip install mlx mlx-lm

# 2. Extract your conversation data
python3 scripts/extract-imessage.py                    # iMessage (macOS)
python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export  # Facebook

# 3. Prepare training data
python3 scripts/prepare-training-data.py --voice

# 4. Fine-tune Gemma 4 E4B (real-time voice model)
python3 scripts/finetune-gemma.py --target e4b --data data/finetune

# 5. Serve with real-time optimizations
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/training-data/adapters/seth-lora-e4b \
  --realtime

# 6. Benchmark
python3 scripts/voice-bench.py
```

## Why This Exists

Google's Gemma 4 is the first open model family with dedicated real-time edge variants (E4B and E2B). Combined with Apple Silicon's unified memory and MLX, you can run a personalized AI that:

- **Sounds like you** — fine-tuned on your actual conversations
- **Responds in real-time** — 100+ tokens/sec, <200ms time-to-first-token
- **Runs locally** — no cloud, no API keys, no data leaving your machine
- **Costs nothing** — after the one-time fine-tuning (5-15 minutes on M4 Max)

## Architecture

```
Your Data                    Fine-Tuning              Real-Time Serving
─────────                    ───────────              ─────────────────
iMessage ─┐                  ┌─ LoRA SFT ──┐         MLX Server (Python)
Facebook ─┼─→ train.jsonl ──→│             │──→      Ollama (Go + llama.cpp)
Custom   ─┘                  └─ DPO ───────┘         llama.cpp (C++ Metal)
```

### Model Targets

| Target | Model | Speed | Use Case |
|--------|-------|-------|----------|
| **E4B** | gemma-4-e4b-it | ~110 tok/s | Real-time voice, daily driver |
| **E2B** | gemma-4-e2b-it | ~180 tok/s | Speculative decode draft model |
| **31B** | gemma-4-31b-it | ~20 tok/s | Highest quality, complex tasks |

### Serving Backends

| Backend | Language | Key Advantage | Best For |
|---------|----------|---------------|----------|
| **MLX Server** | Python | Highest throughput, LoRA hot-swap | Development, fine-tuned models |
| **Ollama** | Go | Zero Python overhead, easiest setup | Production, consistency |
| **llama.cpp** | C++ | Lowest TTFT, fused Metal kernels | Latency-critical first response |
| **vLLM Metal** | Python | Paged attention, continuous batching | Multi-user serving |
| **ANE+GPU Bridge** | Python | Dual-compute speculative decode | Experimental, max throughput |

## Project Structure

```
scripts/
├── extract-imessage.py      # Extract iMessage conversations (macOS)
├── extract-facebook.py      # Extract Facebook Messenger data
├── prepare-training-data.py # Combine sources into train/valid splits
├── finetune-gemma.py        # LoRA fine-tuning pipeline (SFT + DPO)
├── mlx-server.py            # MLX inference server (OpenAI-compatible)
├── ollama-serve.sh          # Ollama serving script
├── llamacpp-serve.sh        # llama.cpp Metal server script
├── vllm-metal-serve.sh      # vLLM Metal server script
├── ane-gpu-bridge.py        # ANE+GPU dual-compute bridge
├── voice-bench.py           # Single-backend voice benchmark
└── bench-all-backends.py    # Head-to-head multi-backend comparison

guides/
├── 01-quickstart.md         # Get running in 10 minutes
├── 02-data-preparation.md   # Download and prepare personal data
├── 03-fine-tuning.md        # LoRA fine-tuning deep dive
├── 04-real-time-serving.md  # Serving backends and optimization
└── 05-benchmarking.md       # Measure and compare performance

configs/
└── example-training-config.json  # Reference configuration
```

## Guides

1. **[Quick Start](guides/01-quickstart.md)** — Get a personalized Gemma running in 10 minutes
2. **[Data Preparation](guides/02-data-preparation.md)** — Extract data from iMessage, Facebook, and more
3. **[Fine-Tuning](guides/03-fine-tuning.md)** — LoRA training, DPO, quantization, speculative decoding
4. **[Real-Time Serving](guides/04-real-time-serving.md)** — MLX, Ollama, llama.cpp setup and optimization
5. **[Benchmarking](guides/05-benchmarking.md)** — Measure TTFT, TPS, RTF and compare backends

## Key Discoveries

Things we learned while making this work:

1. **`mlx_lm` vs `mlx_vlm` matters enormously** — switching from the multimodal VLM import to the text-only LM import gave a 10x speedup (13 → 110+ tok/s). The VLM path adds numpy synchronization overhead even for text-only inference.

2. **PLE-safe quantization is critical for Gemma 4** — Gemma 4 uses `ScaledLinear` layers (Per-Layer Embedding) that most community quantizations corrupt. Only PLE-safe quants (which skip these layers) produce correct output.

3. **Ollama's Go HTTP server eliminates the Python GIL** — Python's GIL serializes token dispatch in the MLX server. Ollama wraps the same llama.cpp Metal backend in a Go server, completely bypassing this bottleneck.

4. **llama.cpp has the lowest TTFT** — Its fused Metal kernels (RoPE+attention in one shader) give the fastest time-to-first-token, which matters most for voice UX.

5. **TurboQuant 3-bit KV cache** compresses the key-value cache by 4.6x with ~2% quality loss, enabling much longer context on memory-constrained devices.

## Hardware Requirements

| Hardware | E2B (2B) | E4B (4B) | 31B |
|----------|----------|----------|-----|
| **Minimum** | M1, 8GB | M1 Pro, 16GB | M2 Max, 64GB |
| **Recommended** | M2+, 16GB | M3 Pro+, 36GB | M4 Max, 128GB |
| **Expected TPS** | 150-200 | 80-120 | 15-25 |

Unified memory is key — Apple Silicon shares memory between CPU and GPU, so the full model weights are accessible without copying.

## License

MIT. See [LICENSE](LICENSE).

The Gemma models themselves are licensed under [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).
