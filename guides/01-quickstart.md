# Quick Start

Get a personalized Gemma 4 running at real-time voice speeds in ~10 minutes.

## Prerequisites

- macOS with Apple Silicon (M1 or later)
- Python 3.10+
- At least 16GB unified memory (for E4B model)

## Step 1: Install MLX

```bash
pip install mlx mlx-lm
```

## Step 2: Extract Your Data

Pick at least one data source:

**iMessage** (easiest if you're on macOS):
```bash
python3 scripts/extract-imessage.py
```

**Facebook Messenger** (if you have a data export):
```bash
python3 scripts/extract-facebook.py --export ~/Downloads/facebook-export
```

You'll need to [download your Facebook data](https://www.facebook.com/dyi) first — select JSON format, Messages only.

## Step 3: Prepare Training Data

```bash
python3 scripts/prepare-training-data.py --voice
```

This merges all sources, deduplicates, adds a system prompt, and splits into train/valid.

## Step 4: Fine-Tune

```bash
python3 scripts/finetune-gemma.py --target e4b --data data/finetune
```

This runs LoRA fine-tuning on the Gemma 4 E4B model (~5-15 minutes on M4 Max). It will:
1. Stop any running MLX server (they compete for GPU memory)
2. Run supervised fine-tuning (SFT)
3. Run Direct Preference Optimization (DPO) if DPO data exists
4. Version the adapter
5. Restart the MLX server with your fine-tuned adapter

## Step 5: Serve

```bash
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/training-data/adapters/seth-lora-e4b \
  --realtime
```

The `--realtime` flag enables:
- TurboQuant 3-bit KV cache compression
- Optimized generation settings for low latency

## Step 6: Test It

```bash
# Quick test
curl http://127.0.0.1:8741/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hey whats up"}], "stream": true}'

# Full benchmark
python3 scripts/voice-bench.py
```

## What's Next?

- **[Data Preparation](02-data-preparation.md)** — Add more data sources for better personalization
- **[Fine-Tuning](03-fine-tuning.md)** — Tune hyperparameters, train all model sizes, speculative decoding
- **[Real-Time Serving](04-real-time-serving.md)** — Try Ollama or llama.cpp for even better performance
- **[Benchmarking](05-benchmarking.md)** — Compare backends and measure improvements
