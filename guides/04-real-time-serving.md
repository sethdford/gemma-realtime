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
| `--realtime` | Enable TurboQuant+ 4-bit KV + optimized settings |
| `--adapter-path` | Load LoRA adapter weights |
| `--kv-bits 4` | TurboQuant+ 4-bit KV (3.8x compression, +0.23% PPL) |
| `--kv-bits 3` | TurboQuant+ 3-bit KV (4.6x compression, +1.06% PPL) |
| `--kv-asymmetric` | Keep keys at FP16, compress only values (best quality) |
| `--speculative-draft MODEL` | Enable speculative decoding with a draft model |
| `--speculative-tokens 4` | Number of draft tokens per step |

### TurboQuant+ KV Cache Compression

[TurboQuant+](https://github.com/TheTom/turboquant_plus) uses PolarQuant + Walsh-Hadamard rotation to compress the KV cache with near-zero quality loss. The MLX port provides `TurboKVCache` as a drop-in replacement.

```bash
# Install TurboQuant+ (MLX fork with TurboKVCache)
pip install git+https://github.com/TheTom/mlx.git@feature/turboquant-plus
```

**Choosing the right config:**

| Your Model | Recommended Config | Why |
|-----------|-------------------|-----|
| Q8_0+ weights | `--kv-bits 4` (symmetric) | Full turbo compression, ~0.2% PPL |
| Q4_K_M weights | `--kv-bits 4 --kv-asymmetric` | Preserves K precision, ~0.5% PPL |
| Memory pressure | `--kv-bits 3` | Maximum compression (4.6x), ~1% PPL |
| Extreme memory | `--kv-bits 2` | 6.4x compression, ~6% PPL |

The `--realtime` flag auto-selects `--kv-bits 4` for the best speed/quality tradeoff. Asymmetric mode (`--kv-asymmetric`) keeps keys at FP16 while compressing values — this matters because K precision controls attention routing via softmax.

### Speculative Decoding

Use a small draft model to propose tokens that the target verifies in parallel:

```bash
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --speculative-draft mlx-community/gemma-4-e2b-it-4bit \
  --realtime
```

This can give a ~1.5-2x speedup when the draft model's predictions closely match the target (which is why training both on the same data matters).

**Important:** The draft model must use the same architecture family as the target. Gemma 4's sliding-window attention requires a Gemma 4 draft model (e.g., a future E2B variant). Cross-generation drafting (e.g., Gemma 3 1B draft with Gemma 4 target) will fail due to incompatible cache layouts. The server handles this gracefully and falls back to standard generation.

### How It Works

The MLX server uses `mlx_lm` for text inference (not `mlx_vlm`). This is critical — the `mlx_vlm` path adds numpy synchronization overhead that drops throughput from 110 to 13 tok/s.

When `--realtime` is set:
1. KV cache is compressed to 4-bit via TurboQuant+ (`TurboKVCache` from MLX fork)
2. During prefill, raw FP16 is stored; on first decode, compressed to packed TurboQuant storage
3. Cross-turn prompt caching: system prompt KV is preserved and reused across requests (trim-and-reuse)
4. Generation parameters are optimized for low latency

TurboQuant+ architecture: prefill stores FP16, decode compresses to packed storage and seeds an internal KVCache with decoded FP16. Subsequent decode tokens use pre-allocated buffers (zero-alloc slice-assign). This gives 97-100% baseline decode speed with 3.8x memory savings.

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

### h-uman Integration

The MLX server integrates with [h-uman](https://github.com/user/h-uman) as its local inference backend. When `~/.human/config.json` exists, the server auto-reads `mlx_local` settings for model, adapter, port, TurboQuant+, and speculative decoding defaults.

**Config (`~/.human/config.json`):**

```json
{
  "mlx_local": {
    "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "adapter_path": "~/.human/adapters/persona",
    "port": 8741,
    "realtime": true,
    "kv_bits": 4,
    "kv_asymmetric": false,
    "speculative_draft": "",
    "speculative_draft_adapter": ""
  }
}
```

**Start via h-uman's service manager:**

```bash
~/.human/bin/human-serve.sh start    # auto-detects gemma-realtime mlx-server.py
~/.human/bin/human-serve.sh status   # shows TurboQuant+, tok/s, hardware
~/.human/bin/human-serve.sh stop     # graceful shutdown
```

The h-uman CLI auto-starts the server when it detects `mlx_local` as the provider and port 8741 is not listening. Voice mode (STT → MLX → TTS) connects to the same endpoint.

**Priority order:** CLI args > environment variables > `~/.human/config.json` > built-in defaults.

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
