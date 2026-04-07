# Contributing

Thanks for your interest in improving gemma-realtime. Here's how to help.

## Areas Where Help Is Needed

**High Impact:**
- More data extractors (Telegram, Discord, Signal, WhatsApp native API)
- CoreML/ANE optimization for the draft model (currently falls back to MLX)
- Windows/Linux support (currently macOS-focused due to Apple Silicon)
- Voice pipeline integration (STT + inference + TTS end-to-end)

**Medium Impact:**
- Better auto-detection of PLE-safe model variants
- Training data quality metrics and filtering
- Multi-GPU support for M-series Ultra chips
- Benchmark result archiving and regression tracking

**Documentation:**
- Guides for additional messaging platforms
- Video walkthrough of the full pipeline
- Troubleshooting guide for common issues

## Development Setup

```bash
git clone https://github.com/h-uman/gemma-realtime.git
cd gemma-realtime
pip install mlx mlx-lm
```

## Code Style

- Python: standard library preferred, minimal dependencies
- Shell: bash 3.2 compatible (macOS default), no associative arrays
- Scripts should be self-contained and runnable independently

## Testing Changes

```bash
# Verify all scripts parse correctly
python3 -m py_compile scripts/*.py
bash -n scripts/*.sh

# Run the benchmark
python3 scripts/voice-bench.py --rounds 5
```

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/telegram-extractor`)
3. Make your changes
4. Test on Apple Silicon if possible
5. Open a PR with a clear description

## Privacy

Never commit:
- Training data (JSONL files with personal messages)
- Model adapters (.safetensors files)
- Benchmark results that contain message previews
- API keys or credentials

The `.gitignore` already excludes these, but please verify before committing.
