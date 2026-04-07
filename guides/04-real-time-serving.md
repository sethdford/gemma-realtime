# Real-Time Serving

How to serve your fine-tuned Gemma 4 at real-time voice speeds on Apple Silicon.

## Real-Time Targets

For voice-quality real-time, you need:

| Metric | Target | What It Means |
|--------|--------|---------------|
| **TTFT** | < 200ms | Time to first token — "feels instant" |
| **TPS** | > 50 tok/s | Tokens per second — faster than speech |
| **RTF** | < 1.0 | Real-time factor — generation faster than speaking |

All three backends below achieve these targets with Gemma 4 E4B on M4 Max.

## Backend 1: MLX Server (Recommended for Development)

**Strengths:** Highest throughput, LoRA hot-swap, native MLX, real-time optimizations.

```bash
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/training-data/adapters/seth-lora-e4b \
  --realtime
```

### Key Flags

| Flag | Effect |
|------|--------|
| `--realtime` | Enable TurboQuant 3-bit KV + optimized settings |
| `--adapter-path` | Load LoRA adapter weights |
| `--kv-bits 3` | TurboQuant KV cache (4.6x smaller, ~2% quality loss) |
| `--speculative-draft MODEL` | Enable speculative decoding with a draft model |
| `--speculative-tokens 4` | Number of draft tokens per step |

### Speculative Decoding

Use a small E2B draft model to propose tokens that E4B verifies in parallel:

```bash
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/training-data/adapters/seth-lora-e4b \
  --speculative-draft mlx-community/gemma-4-e2b-it-4bit \
  --speculative-draft-adapter ~/.human/training-data/adapters/seth-lora-e2b \
  --realtime
```

This can give a ~1.5-2x speedup when the draft model's predictions closely match the target (which is why training both on the same data matters).

### How It Works

The MLX server uses `mlx_lm` for text inference (not `mlx_vlm`). This is critical — the `mlx_vlm` path adds numpy synchronization overhead that drops throughput from 110 to 13 tok/s.

When `--realtime` is set:
1. KV cache is compressed to 3-bit via TurboQuant
2. Prompt caching is enabled (system prompt KV reused across requests)
3. Generation parameters are optimized for low latency

### API

OpenAI-compatible at `http://127.0.0.1:8741/v1/chat/completions`:

```bash
curl http://127.0.0.1:8741/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "hey"}],
    "stream": true,
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Health check: `GET /health` — returns model info, hardware, and TPS stats.

## Backend 2: Ollama (Recommended for Production)

**Strengths:** Zero Python GIL overhead, most consistent P95 latency, easiest setup.

### Install

```bash
brew install ollama
```

### Serve

```bash
./scripts/ollama-serve.sh
```

Or manually:

```bash
ollama serve &
ollama run gemma3:4b ""  # pre-load into GPU memory
```

### With LoRA Adapter

Ollama supports LoRA via a Modelfile. The `ollama-serve.sh` script handles this:

```bash
./scripts/ollama-serve.sh --adapter ~/.human/training-data/adapters/seth-lora-e4b
```

This creates a custom Ollama model (`human-e4b-lora`) with your adapter baked in.

### Why Ollama Is Fast

Ollama wraps llama.cpp's Metal backend in a Go HTTP server. Go has real threading (no GIL), so token dispatch is truly parallel. On benchmarks, Ollama shows the tightest P95 spread — almost no variance between requests.

### API

Same OpenAI-compatible endpoint, but requires the `model` field:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma3:4b",
    "messages": [{"role": "user", "content": "hey"}],
    "stream": true
  }'
```

## Backend 3: llama.cpp (Lowest Latency)

**Strengths:** Lowest TTFT, fused Metal kernels, flash attention v2, tightest P95.

### Install

```bash
brew install llama.cpp
```

### Serve

```bash
./scripts/llamacpp-serve.sh --port 8742
```

The script automatically:
- Downloads the GGUF model if not present (from ggml-org on HuggingFace)
- Disables thinking tokens for voice mode (`--reasoning-budget 0`)
- Enables flash attention
- Locks the model in memory (`--mlock`)

### With Fine-Tuned GGUF

If you exported a GGUF from fine-tuning:

```bash
./scripts/llamacpp-serve.sh \
  --gguf ~/.human/training-data/adapters/seth-lora-e4b-q4.gguf \
  --port 8742
```

### Thinking Tokens

Gemma 4 models can generate "thinking" tokens (`<|think|>...`) before responding. For voice, this adds unwanted latency. The script disables this by default. To re-enable:

```bash
./scripts/llamacpp-serve.sh --think --port 8742
```

### Why llama.cpp Has the Lowest TTFT

llama.cpp's Metal backend has hand-optimized compute shaders:
- Fused RoPE+attention in a single kernel (fewer GPU dispatches)
- Flash attention v2 (reduced memory bandwidth)
- Optimized KV cache management
- No Python overhead whatsoever

## Backend 4: vLLM Metal (Multi-User)

**Strengths:** Paged attention, continuous batching, chunked prefill.

```bash
pip install vllm  # requires Python 3.10+
./scripts/vllm-metal-serve.sh --port 8743
```

Best for scenarios where multiple clients query the same model simultaneously. Paged attention prevents memory fragmentation; continuous batching maximizes GPU utilization.

## Backend 5: ANE+GPU Bridge (Experimental)

**Strengths:** Dual-compute speculative decoding using Neural Engine + GPU.

```bash
pip install coremltools
python3 scripts/ane-gpu-bridge.py --target e4b --draft e2b --port 8744
```

This experimental bridge runs the E2B draft model on the Neural Engine (via CoreML) while the E4B target runs on the GPU (via MLX). Apple Silicon's unified memory means zero-copy data sharing between the two compute units.

In practice, CoreML conversion of Gemma 4 is still unreliable, so the bridge falls back to MLX for the draft model. As CoreML support improves, this should deliver the highest throughput by fully utilizing both ANE and GPU in parallel.

## Choosing a Backend

| Situation | Recommendation |
|-----------|---------------|
| Development + testing | MLX Server (LoRA hot-swap) |
| Production voice app | Ollama (consistency) |
| Latency-critical first word | llama.cpp (lowest TTFT) |
| Multiple concurrent users | vLLM Metal (batching) |
| Fine-tuned with LoRA | MLX Server (native adapter loading) |
| Just want it to work | Ollama (brew install + one command) |

## Running Multiple Backends

Each backend uses a different default port:

| Backend | Default Port |
|---------|-------------|
| MLX Server | 8741 |
| Ollama | 11434 |
| llama.cpp | 8742 |
| vLLM | 8743 |
| ANE+GPU | 8744 |

You can run all of them simultaneously and benchmark:

```bash
python3 scripts/bench-all-backends.py
```
