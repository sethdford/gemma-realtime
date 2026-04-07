# Fine-Tuning Guide

Deep dive into LoRA fine-tuning Gemma 4 on Apple Silicon with MLX.

## How It Works

We use **LoRA** (Low-Rank Adaptation) to fine-tune Gemma 4 without modifying the base model weights. Instead, small adapter matrices (~20-50MB) are trained that modify the model's behavior. This means:

- Training is fast (5-15 minutes for E4B, 20-60 minutes for 31B)
- Memory efficient (the base model stays in 4-bit quantization)
- You can swap adapters without re-downloading the model
- Multiple personas can share the same base model

## Model Targets

### E4B — Real-Time Voice (Recommended for voice)

```bash
python3 scripts/finetune-gemma.py --target e4b --data data/finetune
```

| Setting | Value | Why |
|---------|-------|-----|
| Model | gemma-4-e4b-it-4bit | 4B params, optimized for edge |
| Iterations | 1,200 | Enough for style capture |
| Batch size | 4 | Fits M3 Pro+ comfortably |
| Learning rate | 2e-5 | Standard LoRA LR for small models |
| LoRA rank | 32 | Higher rank captures more nuance |
| LoRA layers | 12 | Cover the full attention stack |
| Sequence length | 4,096 | Full multi-turn context |

### E2B — Speculative Decode Draft

```bash
python3 scripts/finetune-gemma.py --target e2b --data data/finetune
```

Train the 2B model on the same data as E4B. When used as a speculative decode draft, having the same style distribution maximizes the acceptance rate (target model accepts more draft tokens).

### 31B — Highest Quality

```bash
python3 scripts/finetune-gemma.py --target 31b --data data/finetune
```

Use for text-based interactions where latency isn't critical. Produces the most nuanced, contextually aware responses.

### All Three at Once

```bash
python3 scripts/finetune-gemma.py --train-all --data data/finetune --realtime-first
```

Trains E4B, E2B, then 31B sequentially. `--realtime-first` prioritizes the voice models so you can start testing while 31B trains.

## Training Pipeline

The `finetune-gemma.py` script runs a multi-phase pipeline:

### Phase 1: Supervised Fine-Tuning (SFT)

Standard LoRA training with masked prompts (only trains on your responses, not the input context):

```bash
python3 scripts/finetune-gemma.py --target e4b --data data/finetune
```

**Key flags:**
- `--mask-prompt` (default: on) — Only compute loss on assistant responses
- `--grad-checkpoint` (default: on) — Trade compute for memory (essential for 31B)
- `--resume` — Continue from a previous adapter checkpoint

### Phase 2: Direct Preference Optimization (DPO)

If DPO pairs exist (preferred vs rejected responses), a DPO pass sharpens the boundary between your style and generic AI responses:

```bash
python3 scripts/finetune-gemma.py --target e4b --data data/finetune --dpo
```

DPO is enabled by default if `data/finetune/dpo/pairs.jsonl` or a `dpo_pairs.db` exists. Use `--sft-only` to skip.

### Phase 3: Adapter Versioning

Each training run automatically versions the adapter (v1, v2, v3...) and maintains a `seth-lora-current` symlink pointing to the latest.

### Phase 4: Quantization (Optional)

Export a fused, PLE-safe quantized model:

```bash
# MLX format (for mlx-server.py)
python3 scripts/finetune-gemma.py --target e4b --data data/finetune --quantize

# GGUF format (for llama.cpp / Ollama)
python3 scripts/finetune-gemma.py --target e4b --data data/finetune --quantize --quant-format gguf
```

### Phase 5: Speculative Draft Training (Optional)

Train an E2B draft model that mirrors your E4B target:

```bash
python3 scripts/finetune-gemma.py --target e4b --data data/finetune --speculative-draft
```

## PLE-Safe Quantization

Gemma 4 has a critical architectural detail: `ScaledLinear` layers (Per-Layer Embedding). Most community quantizations quantize these layers, corrupting model output.

**PLE-safe models skip ScaledLinear quantization.** The fine-tuning script handles this automatically when you use `--quantize`.

Known PLE-safe model IDs:
- `FakeRockert543/gemma-4-e4b-it-MLX-4bit`
- `FakeRockert543/gemma-4-e4b-it-MLX-8bit`
- `FakeRockert543/gemma-4-e2b-it-MLX-4bit`

Known broken model IDs (avoid):
- `mlx-community/gemma-4-e4b-it-4bit` (PLE layers corrupted)
- `unsloth/gemma-4-e4b-it-4bit` (same issue)

The `mlx-server.py` script warns you if you load a broken quantization.

## Hyperparameter Tuning

### Learning Rate

- **Too high** (>5e-5 for E4B): Catastrophic forgetting. Model becomes incoherent.
- **Too low** (<5e-7): Barely learns your style. Wastes training time.
- **Sweet spot**: 2e-5 for E4B/E2B, 1e-6 for 31B.

### LoRA Rank

- **Rank 8**: Minimal style capture. Good for simple adjustments.
- **Rank 16**: Default for 31B. Captures vocabulary and phrasing.
- **Rank 32**: Default for E4B/E2B. Full style capture including timing patterns.
- **Rank 64**: Overkill for most. Only if you have 10,000+ training pairs.

### Iterations

Monitor training loss. You want:
- Loss steadily decreasing for the first 60-70% of iterations
- Loss flattening (not increasing) for the last 30%
- If loss starts increasing: you're overfitting. Reduce iterations.

Typical iteration counts:
- 500 training pairs → 400-600 iterations
- 2,000 training pairs → 800-1,200 iterations
- 5,000+ training pairs → 1,200-1,500 iterations

### Batch Size

Limited by GPU memory:
- **M1/M2 (8-16GB)**: batch_size=1 for 31B, 2 for E4B
- **M3/M4 Pro (36GB)**: batch_size=2 for 31B, 4 for E4B
- **M4 Max (128GB)**: batch_size=4 for 31B, 8 for E4B

## Resuming Training

If training is interrupted, resume from the last checkpoint:

```bash
python3 scripts/finetune-gemma.py --target e4b --data data/finetune --resume
```

This loads the existing `adapters.safetensors` and continues from where it left off.

## Custom System Prompts

The default system prompt is generic. For better results, customize it:

```bash
python3 scripts/prepare-training-data.py \
  --system-prompt "You are [Your Name]. You're direct, use lowercase, and keep things casual. You love talking about code and music."
```

The system prompt is prepended to every training conversation, teaching the model when to activate your persona.

## Monitoring Training

The script reports loss every 5 steps and runs validation every 20 steps. Watch for:

```
Step    5 | Train Loss: 2.341
Step   10 | Train Loss: 1.892  ← good, decreasing
Step   15 | Train Loss: 1.654
Step   20 | Train Loss: 1.523 | Valid Loss: 1.601
Step   25 | Train Loss: 1.498
...
Step  800 | Train Loss: 0.891 | Valid Loss: 0.923  ← healthy gap
Step 1000 | Train Loss: 0.834 | Valid Loss: 0.952  ← valid rising = stop soon
```

A healthy train/valid gap is <0.2. If valid loss rises while train loss drops, you're overfitting.

## After Training

The adapter is saved to `~/.human/training-data/adapters/seth-lora-e4b/` (or your custom path).

**Test it immediately:**

```bash
python3 scripts/mlx-server.py \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ~/.human/training-data/adapters/seth-lora-e4b \
  --realtime

# In another terminal:
curl http://127.0.0.1:8741/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hey whats up"}], "stream": true}'
```

Does it sound like you? If not, check:
1. Was the training data diverse enough?
2. Did you use `--mask-prompt`? (Should be on)
3. Is the loss low enough? (Should be <1.0 for E4B)
4. Are you using a PLE-safe base model?
