#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "==================================================================="
echo "  Overnight Pipeline — Voxtral Draft Heads + Fish STS"
echo "  Started: $(date)"
echo "==================================================================="
echo ""

# ─── Part 1: Voxtral Draft Heads ─────────────────────────────────────────────

echo "==================================================================="
echo "  Part 1: Voxtral Draft Head Pipeline"
echo "==================================================================="
echo ""

# Step 1: Collect training data from LibriSpeech (8-step denoising for quality)
echo "[1/6] Collecting Voxtral training data from LibriSpeech test-clean..."
echo "      2,620 sentences, 8-step denoising, adaptive frame caps"
echo ""
COLLECT_FLAGS=(--librispeech --output data/draft-pairs-libri.npz --denoise-steps 8 --voice cheerful_male)
if [ -f data/draft-pairs-libri.checkpoint.npz ] && [ ! -f data/draft-pairs-libri.npz ]; then
  echo "      Resuming from data/draft-pairs-libri.checkpoint.npz"
  COLLECT_FLAGS+=(--resume)
fi
python3 scripts/collect-draft-data.py "${COLLECT_FLAGS[@]}"

echo ""
echo "[1/6] Voxtral data collection complete at $(date)"
echo ""

# Step 2: Train draft heads on full dataset
echo "[2/6] Training 3 Voxtral draft heads..."
echo ""
python3 scripts/voxtral_speculative.py train \
    --data data/draft-pairs-libri.npz \
    --output adapters/draft-heads/heads-libri.safetensors \
    --heads 3 \
    --epochs 20 \
    --batch-size 256 \
    --lr 1e-3

echo ""
echo "[2/6] Voxtral training complete at $(date)"
echo ""

# Step 3: Voxtral benchmark
echo "[3/6] Running Voxtral TTS benchmark..."
echo ""
python3 scripts/benchmark-tts.py \
    --precision 6bit \
    --output proof-artifacts/benchmark-libri.json

echo ""
echo "[3/6] Voxtral benchmark complete at $(date)"
echo ""

# ─── Part 2: Fish STS Pipeline ───────────────────────────────────────────────

echo "==================================================================="
echo "  Part 2: Fish DAC True STS Pipeline"
echo "==================================================================="
echo ""

# Step 4: Extract Fish DAC tokens (skip if data exists)
if [ -f data/libritts-fish-dac-tokens.jsonl ]; then
  FISH_LINES=$(wc -l < data/libritts-fish-dac-tokens.jsonl)
  echo "[4/6] Fish DAC tokens exist ($FISH_LINES lines) — skipping extraction"
else
  echo "[4/6] Extracting Fish DAC tokens from LibriTTS..."
  python3 scripts/train-fish-sts.py extract \
      --input data/libritts-codec-train-full-eos.jsonl \
      --output data/libritts-fish-dac-tokens.jsonl
fi

echo ""
echo "[4/6] Fish DAC extraction complete at $(date)"
echo ""

# Step 5: Train Fish STS (Phase A → B → C)
echo "[5/6] Training Fish STS (Phase A → B → C)..."
echo ""
python3 scripts/train-fish-sts.py all --codec fish

echo ""
echo "[5/6] Fish STS training complete at $(date)"
echo ""

# Step 6: Evaluate both pipelines
echo "[6/6] Running evaluation (Fish STS + Cascaded)..."
echo ""
python3 scripts/eval_sts.py --pipeline fish --max-samples 10
python3 scripts/eval_sts.py --pipeline cascaded --max-samples 10

echo ""
echo "==================================================================="
echo "  Overnight pipeline complete at $(date)"
echo "  Proof artifacts: proof-artifacts/"
echo "  Fish STS weights: adapters/fish-sts/phase-c/fish_sts_final.safetensors"
echo "==================================================================="
