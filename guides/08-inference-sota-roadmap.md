# Guide 08: Inference SOTA — knobs, env vars, and benchmarks

This guide ties **latency-oriented inference research** to **concrete controls in this repo**: environment variables, server flags, health JSON, and benchmark scripts. Use it as a checklist when tuning voice or text throughput on Apple Silicon.

## Quick inventory

| Area | What to use in this repo | See also |
|------|---------------------------|----------|
| Text throughput | `mlx-server.py`: speculative draft (E2B), `MLX_SPECULATIVE_TOKENS` / `GEMMA_SPECULATIVE_TOKENS`, TurboQuant KV, `--kv-asymmetric` / `GEMMA_KV_ASYMMETRIC` | `guides/04-real-time-serving.md`, `guides/05-benchmarking.md` |
| Voice TTFB vs phrase quality | `GEMMA_MIN_FLUSH_CHARS`, `GEMMA_MAX_BUFFER_CHARS` in `speech-server.py` and `realtime-ws.py` | `guides/07-speech-pipeline.md` |
| Native IOSurface / Metal staging | `GEMMA_IOSURFACE_KV`, `GEMMA_IOSURFACE_KV_BYTES`; `prove-native-hw.py`, `/health` → `iosurface` | `guides/07-speech-pipeline.md` (Phase 6), `guides/06-secret-apis.md` |
| Sanity check | `python3 scripts/sota_env_check.py` | — |

## MLX server: environment variables

| Variable | Effect |
|----------|--------|
| `MLX_MODEL`, `MLX_PORT`, `MLX_ADAPTER_PATH` | Defaults for `--model`, `--port`, `--adapter-path` (also merged with `~/.human/config.json` when present). |
| `MLX_SPECULATIVE_TOKENS` or `GEMMA_SPECULATIVE_TOKENS` | Default for `--speculative-tokens` (draft proposals per speculative step). Higher can raise throughput until acceptance drops; **restart the server** after changing. |
| `GEMMA_KV_ASYMMETRIC=1` | Same as `--kv-asymmetric` (asymmetric TurboQuant: FP16 keys, compressed values). |
| `GEMMA_IOSURFACE_KV=1` | Enable IOSurface KV staging hooks after model load (macOS + `libgemma_hw`). |
| `GEMMA_IOSURFACE_KV_BYTES` | Optional staging buffer size (bytes). |

`GET /health` includes **`inference_tuning`**: `realtime_voice_mode`, `speculative_draft_tokens`, `speculative_active`, `kv_bits`, `kv_scheme`, `turboquant_active`, plus existing hardware and IOSurface fields.

## Voice pipeline: flush and buffer sizes

Sentence-boundary buffering avoids speaking half-phrases but adds latency until enough text arrives.

| Variable | Default | Effect |
|--------|---------|--------|
| `GEMMA_MIN_FLUSH_CHARS` | `12` | Minimum characters before flushing a chunk to TTS (lower → earlier audio, choppier phrases). |
| `GEMMA_MAX_BUFFER_CHARS` | `120` | Cap buffer growth before forced flush. |

Set these **before** starting `speech-server.py` or `realtime-ws.py` (they are read at import time).

## Speculative draft token sweep (MLX server)

The server reads speculative draft count **at startup**. To compare throughput vs `N`:

1. For each candidate `N`, start the server with `GEMMA_SPECULATIVE_TOKENS=N` (and the same `--speculative-draft` model).
2. Run a short benchmark against the running server.

Example loop (adjust model and paths):

```bash
for n in 2 3 4 6 8; do
  echo "=== draft_tokens=$n ==="
  GEMMA_SPECULATIVE_TOKENS=$n python3 scripts/mlx-server.py \
    --model mlx-community/gemma-4-e4b-it-4bit \
    --speculative-draft mlx-community/gemma-4-e2b-it-4bit &
  PID=$!
  sleep 20
  python3 scripts/bench-speculative-sweep.py --url http://127.0.0.1:8741 --rounds 8
  kill $PID
  wait $PID 2>/dev/null || true
done
```

Single snapshot (current server config + median tok/s):

```bash
python3 scripts/bench-speculative-sweep.py --url http://127.0.0.1:8741 --rounds 10 --json
```

## Research themes (off-repo pointers)

These are **not** all implemented as first-class switches here; they inform what to watch upstream (MLX, `mlx-lm`, TurboQuant) and what this stack already exposes.

- **Speculative decoding** — E2B draft + E4B target in `mlx-server.py`; tune `GEMMA_SPECULATIVE_TOKENS` and measure acceptance indirectly via tok/s and latency.
- **KV compression** — TurboQuant+ bits and asymmetric mode via CLI / `GEMMA_KV_ASYMMETRIC`; realtime preset tightens KV for voice.
- **Streaming TTS** — Phrase buffering and codec paths in `guides/07-speech-pipeline.md`; Voxtral draft heads are separate from LM speculative decoding.
- **Hardware / IOSurface** — `libgemma_hw` + `native_hw.py` for staging and Metal selftests; CI `native-hw-macos` job when applicable.

## Verify your environment

```bash
python3 scripts/sota_env_check.py
```

Reports platform, optional `mlx` / `mlx_lm` / TurboQuant imports, `libgemma_hw` presence, and relevant environment variables.
