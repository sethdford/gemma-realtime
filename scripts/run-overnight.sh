#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "  Voxtral Draft Head Overnight Pipeline"
echo "  Started: $(date)"
echo "==================================================================="
echo ""

# Step 1: Collect training data from LibriSpeech (8-step denoising for quality)
echo "[1/3] Collecting training data from LibriSpeech test-clean..."
echo "      2,620 sentences, 8-step denoising, adaptive frame caps"
echo "      Checkpoints every 200 sentences"
echo ""
COLLECT_FLAGS=(--librispeech --output data/draft-pairs-libri.npz --denoise-steps 8 --voice cheerful_male)
if [ -f data/draft-pairs-libri.checkpoint.npz ] && [ ! -f data/draft-pairs-libri.npz ]; then
  echo "      Resuming from data/draft-pairs-libri.checkpoint.npz"
  COLLECT_FLAGS+=(--resume)
fi
python3 scripts/collect-draft-data.py "${COLLECT_FLAGS[@]}"

echo ""
echo "==================================================================="
echo "[1/3] Data collection complete at $(date)"
echo "==================================================================="
echo ""

# Step 2: Train draft heads on full dataset
echo "[2/3] Training 3 draft heads (20 epochs, batch=256, lr=1e-3)..."
echo ""
python3 scripts/voxtral_speculative.py train \
    --data data/draft-pairs-libri.npz \
    --output adapters/draft-heads/heads-libri.safetensors \
    --heads 3 \
    --epochs 20 \
    --batch-size 256 \
    --lr 1e-3

echo ""
echo "==================================================================="
echo "[2/3] Training complete at $(date)"
echo "==================================================================="
echo ""

# Step 3: Run benchmark
echo "[3/3] Running benchmark (8-step, 4-step, speculative)..."
echo ""
python3 scripts/benchmark-tts.py \
    --precision 6bit \
    --output proof-artifacts/benchmark-libri.json

echo ""
echo "==================================================================="
echo "  Pipeline complete at $(date)"
echo "==================================================================="
