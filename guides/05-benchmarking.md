# Benchmarking Guide

How to measure and compare inference performance across backends.

## Metrics

| Metric | What | Target | Why It Matters |
|--------|------|--------|----------------|
| **TTFT** | Time to First Token | < 200ms | User perceives response as "instant" |
| **TPS** | Tokens Per Second | > 50 | Faster than human speech (~3 words/sec) |
| **RTF** | Real-Time Factor | < 1.0 | Generation faster than speaking the output |
| **E2E** | End-to-End Latency | < 3000ms | Full response delivered quickly |

**RTF calculation:** generation_time / (word_count / 2.5). An RTF of 0.03 means the model generates text 33x faster than a human could speak it.

## Single Backend Benchmark

```bash
# Benchmark the MLX server
python3 scripts/voice-bench.py

# Benchmark a specific endpoint
python3 scripts/voice-bench.py --endpoint http://127.0.0.1:8742

# More rounds for statistical significance
python3 scripts/voice-bench.py --rounds 20 --warmup 5

# Benchmark Ollama (needs --model for Ollama)
python3 scripts/voice-bench.py --endpoint http://127.0.0.1:11434 --model gemma3:4b

# JSON output for programmatic analysis
python3 scripts/voice-bench.py --json > results.json
```

### What It Tests

The benchmark sends 20 real voice-style prompts ("hey what's up", "tell me something interesting", "I'm feeling kind of stressed", etc.) with streaming enabled and measures:

1. **TTFT** — Time from request to first SSE chunk with content
2. **TPS** — Total tokens generated / generation time
3. **RTF** — Generation time / estimated speech duration (words / 2.5 wps)
4. **E2E** — Total time from request to [DONE]

### Reading Results

```
======================================================================
  RESULTS (10 valid / 10 total)
======================================================================
  TTFT        P50=  154.0ms  P95=  889.0ms  P99=  889.0ms  mean=  247.0ms  [target: 200ms → PASS]
  TPS         P50=  111.6 t/s  P95=  120.0 t/s  P99=  120.0 t/s  mean=  111.4 t/s  [target: 50 t/s → PASS]
  RTF         P50=    0.029x  P95=    0.040x  P99=    0.040x  mean=    0.031x
  E2E         P50= 1200.0ms  P95= 2100.0ms  P99= 2100.0ms  mean= 1350.0ms

  Voice readiness:
    REAL-TIME READY — TTFT and TPS both meet voice targets
```

- **P50** (median) is the most reliable indicator
- **P95** shows worst-case latency (important for user experience)
- A large P50-to-P95 gap indicates inconsistency (common on first request due to model warmup)

## Head-to-Head Comparison

Compare all running backends at once:

```bash
python3 scripts/bench-all-backends.py
```

Or specify which backends to test:

```bash
python3 scripts/bench-all-backends.py --backends mlx,ollama,llamacpp --rounds 15
```

### Example Output

```
==========================================================================================
  HEAD-TO-HEAD BENCHMARK RESULTS
==========================================================================================

  Backend                        TTFT P50   TTFT P95   TPS P50  TPS Mean   RTF P50    Verdict
  ──────────────────────────  ────────  ────────  ───────  ───────  ───────  ────────
  MLX Server (mlx_lm)               154ms      889ms   111.6    111.4    0.029  REALTIME  >>>>>>>>>>
  Ollama (Go + llama.cpp)           141ms      150ms   107.9    102.7    0.031  REALTIME  >>>>>>>>>>
  llama.cpp Metal                   136ms      141ms    94.0     94.1    0.040  REALTIME  >>>>>>>>>

  ────────────────────────────────────────────────────────────────────────────────────────
  Fastest backend: MLX Server (mlx_lm) at 111.6 tok/s
  Real-time ready: MLX Server (mlx_lm), Ollama (Go + llama.cpp), llama.cpp Metal
==========================================================================================
```

### Using --compare

The single-backend `voice-bench.py` also supports comparison mode:

```bash
python3 scripts/voice-bench.py --compare --rounds 15
```

## Interpreting Results

### MLX has the highest P50 TPS but worst P95 TTFT

The first request after loading warms up the Metal GPU pipeline. Subsequent requests are fast. If P95 TTFT matters (it does for voice), Ollama or llama.cpp are more consistent.

### Ollama has the tightest P50-P95 spread

Go's threading model delivers consistent latency. No GIL means no contention between HTTP handling and token generation.

### llama.cpp has the lowest TTFT

Fused Metal kernels reduce GPU dispatch overhead. The first token arrives fastest, but sustained throughput is slightly lower than MLX.

### Low TPS on some rounds

If you see 0 TPS or very low numbers on individual rounds, check:
1. **Thinking tokens** — llama.cpp models may generate `<|think|>` tokens that count as empty content. Use `--reasoning-budget 0`.
2. **Wrong model** — Ollama serves multiple models; make sure the benchmark requests the right one.
3. **Memory pressure** — If another process is competing for unified memory, TPS drops. Close memory-heavy apps.

## Automated CI Benchmarking

For tracking performance over time:

```bash
# JSON output with all raw data
python3 scripts/bench-all-backends.py --rounds 20 --json > bench-$(date +%Y%m%d).json
```

You can track these metrics across fine-tuning iterations to see if adapter changes affect inference speed.

## Hardware Scaling

Expected performance by hardware tier:

| Chip | Memory | E4B TPS | E2B TPS | 31B TPS |
|------|--------|---------|---------|---------|
| M1 | 8GB | ~30 | ~60 | OOM |
| M1 Pro | 16GB | ~50 | ~90 | ~5 |
| M2 Pro | 32GB | ~65 | ~110 | ~10 |
| M3 Max | 64GB | ~85 | ~140 | ~18 |
| M4 Max | 128GB | ~110 | ~180 | ~22 |

These are approximate. Actual results depend on quantization, KV cache size, context length, and concurrent load.
