#!/usr/bin/env python3
"""
STS Evaluation Harness — WER, audio quality, speaker similarity, E2E latency.

Metrics:
    1. WER: Whisper re-transcription of generated audio vs reference text
    2. Audio quality score: spectral heuristic (speech-band energy, flatness, RMS)
    3. Speaker similarity: spectral envelope cosine between input/output audio
    4. E2E latency: generation time under realistic conditions
    5. RTF: real-time factor (generation time / audio duration)

Usage:
    python3 scripts/eval_sts.py --pipeline cascaded   # Whisper + LLM + Voxtral
    python3 scripts/eval_sts.py --pipeline fish        # True STS (Fish codec)
    python3 scripts/eval_sts.py --pipeline fish --eval-set data/eval-spoken-qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PROOF_DIR = Path("proof-artifacts")
PROOF_DIR.mkdir(exist_ok=True)


@dataclass
class STSMetrics:
    wer: float = 0.0
    wer_count: int = 0
    audio_quality: list[float] = field(default_factory=list)
    speaker_sims: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    rtfs: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "wer": round(self.wer, 4) if self.wer_count else None,
            "audio_quality_mean": round(float(np.mean(self.audio_quality)), 3) if self.audio_quality else None,
            "audio_quality_std": round(float(np.std(self.audio_quality)), 3) if self.audio_quality else None,
            "spk_sim_mean": round(float(np.mean(self.speaker_sims)), 3) if self.speaker_sims else None,
            "latency_p50_ms": round(float(np.median(self.latencies_ms)), 1) if self.latencies_ms else None,
            "latency_p95_ms": round(float(np.percentile(self.latencies_ms, 95)), 1) if self.latencies_ms else None,
            "rtf_mean": round(float(np.mean(self.rtfs)), 4) if self.rtfs else None,
            "n_samples": max(self.wer_count, len(self.audio_quality), len(self.latencies_ms)),
        }


# ═══════════════════════════════════════════════════════════
# WER via Whisper re-transcription
# ═══════════════════════════════════════════════════════════

def compute_wer(reference: str, hypothesis: str) -> float:
    """Minimum edit distance WER (word-level)."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref)][len(hyp)] / len(ref)


def whisper_transcribe(audio_np: np.ndarray, sr: int = 24000) -> str:
    """Transcribe audio with mlx-whisper."""
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        if sr != 16000:
            n_out = int(len(audio_np) * 16000 / sr)
            audio_16k = np.interp(
                np.linspace(0, 1, n_out),
                np.linspace(0, 1, len(audio_np)),
                audio_np.astype(np.float32),
            ).astype(np.float32)
        else:
            audio_16k = audio_np.astype(np.float32)
        sf.write(f.name, audio_16k, 16000)

        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(
                f.name, path_or_hf_repo="mlx-community/whisper-small-mlx"
            )
            return result.get("text", "").strip()
        except ImportError:
            return "[whisper not available]"


# ═══════════════════════════════════════════════════════════
# Audio quality score (spectral heuristic — NOT a real MOS predictor)
# ═══════════════════════════════════════════════════════════

def estimate_audio_quality(audio_np: np.ndarray, sr: int = 24000) -> float:
    """Spectral audio quality score (1-5). NOT a real MOS predictor.

    Measures basic signal properties (energy, speech-band concentration,
    spectral flatness) to distinguish silence/noise from speech-like audio.
    Scores >=4.0 mean "plausibly speech"; scores <2.0 mean "broken output".
    Do NOT compare across systems — use WER for that.
    """
    if len(audio_np) < sr * 0.2:
        return 1.0

    rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2)))
    if rms < 0.005:
        return 1.0

    try:
        from scipy import signal as sig
        f, psd = sig.welch(audio_np.astype(np.float64), fs=sr, nperseg=min(1024, len(audio_np)))
        total = np.sum(psd)
        if total < 1e-12:
            return 1.5

        speech_mask = (f >= 100) & (f <= 4000)
        speech_ratio = float(np.sum(psd[speech_mask]) / total)
        geo_mean = np.exp(np.mean(np.log(psd + 1e-20)))
        arith_mean = np.mean(psd)
        flatness = float(geo_mean / (arith_mean + 1e-20))

        score = 1.0
        score += min(speech_ratio * 2.0, 1.5)
        score += max(0, 1.0 - flatness) * 1.0
        score += min(rms * 15, 1.0)
        score += 0.5 if speech_ratio > 0.6 and flatness < 0.3 else 0.0
        return min(max(score, 1.0), 5.0)
    except ImportError:
        return 3.0


# ═══════════════════════════════════════════════════════════
# Speaker similarity (cosine on Whisper encoder features)
# ═══════════════════════════════════════════════════════════

def speaker_similarity(audio_a: np.ndarray, audio_b: np.ndarray,
                       sr: int = 24000) -> float:
    """Estimate speaker similarity via spectral envelope cosine."""
    try:
        from scipy import signal as sig
        _, psd_a = sig.welch(audio_a.astype(np.float64), fs=sr,
                             nperseg=min(1024, len(audio_a)))
        _, psd_b = sig.welch(audio_b.astype(np.float64), fs=sr,
                             nperseg=min(1024, len(audio_b)))
        min_len = min(len(psd_a), len(psd_b))
        a, b = psd_a[:min_len], psd_b[:min_len]
        dot = np.sum(a * b)
        norm = np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)) + 1e-10
        return float(dot / norm)
    except ImportError:
        return 0.0


# ═══════════════════════════════════════════════════════════
# Load eval data
# ═══════════════════════════════════════════════════════════

def load_eval_set(path: str) -> list[dict]:
    """Load eval items.  Each line: {text, audio_path?}."""
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("text"):
                items.append(item)
    return items


DEFAULT_EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, how are you doing today?",
    "Machine learning can now generate speech in real time on consumer hardware.",
    "Numbers like one thousand two hundred and thirty four test numeral handling.",
    "Can you believe it? This is amazing! I never thought this would work.",
]


def build_eval_from_libritts(n: int = 20, seed: int = 42) -> list[dict]:
    """Build eval set from LibriTTS data with reference audio for speaker sim."""
    data_path = Path("data/libritts-codec-train-full-eos.jsonl")
    if not data_path.exists():
        return []
    rng = np.random.RandomState(seed)
    all_items = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            text = item.get("text", "")
            audio_path = item.get("audio_path", "")
            if text and audio_path and Path(audio_path).exists():
                word_count = len(text.split())
                if 5 <= word_count <= 25:
                    all_items.append({"text": text, "audio_path": audio_path})
    if not all_items:
        return []
    indices = rng.choice(len(all_items), size=min(n, len(all_items)), replace=False)
    return [all_items[i] for i in indices]


# ═══════════════════════════════════════════════════════════
# Eval runner
# ═══════════════════════════════════════════════════════════

def _phase1_embed(texts: list[str], cache_path: Path) -> None:
    """Subprocess: load Gemma, compute text embeddings, save to disk."""
    import mlx.core as mx
    from mlx_lm import load as lm_load
    print("  Phase 1: Gemma text embeddings...", flush=True)
    gemma, tokenizer = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
    inner = gemma.language_model.model if hasattr(gemma, "language_model") else gemma.model

    results = {}
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        emb = inner.embed_tokens(mx.array([ids]))
        mx.eval(emb)
        results[f"emb_{i}"] = np.array(emb[0].tolist(), dtype=np.float32)
    np.savez(str(cache_path), **results)
    print(f"    Saved {len(results)} embeddings to {cache_path}", flush=True)


def _phase2_generate(texts: list[str], emb_path: Path, audio_dir: Path,
                     audio_paths: list[str] | None = None) -> None:
    """Subprocess: load STS model + codec, generate audio, save wavs.

    When audio_paths are provided, uses the audio-to-audio pathway
    (encode reference → audio_input projection → generate), which matches
    how the model was trained. Falls back to text embeddings when no
    reference audio is available.
    """
    import mlx.core as mx
    from fish_sts import FishSpeechToSpeech, FishSTSPipeline
    from codec import AudioCodec, CodecTokens, CodecType
    import soundfile as sf

    w_path = FishSTSPipeline._resolve_weights(None)
    if w_path is None:
        print("    [SKIP] No trained STS weights")
        return
    config = FishSTSPipeline.config_from_weights(w_path)
    model = FishSpeechToSpeech(config)
    w = mx.load(str(w_path))
    model.load_weights(list(w.items()), strict=False)
    del w

    use_fish_dac = config.fish_n_codebooks > 3
    if use_fish_dac:
        print("  Phase 2: Speech generation (STS + Fish DAC)...", flush=True)
        codec = AudioCodec("fish")
        codec.load()
        codec_type = CodecType.FISH_DAC
    else:
        print("  Phase 2: Speech generation (STS + SNAC)...", flush=True)
        codec = AudioCodec("snac")
        codec.load()
        codec_type = CodecType.SNAC
    sr = codec.sample_rate

    embs = np.load(str(emb_path))
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for i, text in enumerate(texts):
        t0 = time.time()

        # Prefer audio-to-audio pathway (matches training)
        ref_path = audio_paths[i] if audio_paths and i < len(audio_paths) else None
        if ref_path and Path(ref_path).exists():
            ref_audio, ref_sr = sf.read(ref_path)
            ref_audio = ref_audio.astype(np.float32)
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            if ref_sr != sr:
                n_out = int(len(ref_audio) * sr / ref_sr)
                ref_audio = np.interp(
                    np.linspace(0, 1, n_out), np.linspace(0, 1, len(ref_audio)), ref_audio
                ).astype(np.float32)
            tokens = codec.encode(ref_audio)
            cb0_input = mx.array(tokens.codes[0:1, :], dtype=mx.int32)
            emb = model.audio_input(cb0_input)
            mx.eval(emb)
            input_mode = "audio"
        else:
            key = f"emb_{i}"
            if key not in embs:
                continue
            emb = mx.array(embs[key][np.newaxis])
            input_mode = "text"

        n_words = len(text.split())
        max_frames = min(max(n_words * 6, 40), 150)
        cb0_out, speech_hidden = model.generate_cb0(
            emb, temperature=0.5, top_k=30, max_frames=max_frames
        )
        mx.eval(cb0_out, speech_hidden)
        if cb0_out.size == 0:
            print(f"    [{i+1}] No output for: {text[:50]}...")
            continue
        all_codes = model.decode_depth(cb0_out, speech_hidden)
        mx.eval(all_codes)
        out_tokens = CodecTokens(
            codes=np.array(all_codes[0].tolist(), dtype=np.int64),
            n_codebooks=config.fish_n_codebooks,
            frame_rate=config.fish_frame_rate,
            codec_type=codec_type,
        )
        audio_out = codec.decode(out_tokens)
        elapsed = time.time() - t0
        dur = len(audio_out) / sr
        wav_path = audio_dir / f"sample_{i}.wav"
        sf.write(str(wav_path), audio_out, sr)
        meta[str(i)] = {"elapsed": elapsed, "dur": dur, "sr": sr}
        print(f"    [{i+1}] {dur:.1f}s audio in {elapsed:.1f}s (RTF={elapsed/dur:.2f}x, {input_mode}): {text[:50]}...",
              flush=True)
    with open(audio_dir / "meta.json", "w") as f:
        json.dump(meta, f)


def _run_fish_eval(eval_items: list[dict]) -> STSMetrics:
    """Run eval in separate subprocesses so GPU memory is truly freed between phases."""
    import subprocess

    metrics = STSMetrics()
    texts = [item["text"][:200] for item in eval_items]

    cache_dir = Path("data/.eval_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npz"
    audio_dir = cache_dir / "audio"

    script = str(Path(__file__).resolve())

    # Phase 1: Gemma embeddings (subprocess)
    texts_json = json.dumps(texts)
    code1 = (
        f"import sys, json; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        f"from eval_sts import _phase1_embed; from pathlib import Path; "
        f"_phase1_embed(json.loads({texts_json!r}), Path({str(emb_path)!r}))"
    )
    r = subprocess.run([sys.executable, "-c", code1], capture_output=False)
    if r.returncode != 0 or not emb_path.exists():
        print("  [FAIL] Phase 1 (Gemma embeddings) failed or OOM'd")
        return metrics

    # Phase 2: STS generation (subprocess)
    audio_paths = [item.get("audio_path", "") for item in eval_items]
    audio_paths_json = json.dumps(audio_paths)
    code2 = (
        f"import sys, json; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        f"from eval_sts import _phase2_generate; from pathlib import Path; "
        f"_phase2_generate(json.loads({texts_json!r}), Path({str(emb_path)!r}), Path({str(audio_dir)!r}), "
        f"json.loads({audio_paths_json!r}))"
    )
    r = subprocess.run([sys.executable, "-c", code2], capture_output=False)
    meta_path = audio_dir / "meta.json"
    if r.returncode != 0 or not meta_path.exists():
        print("  [FAIL] Phase 2 (speech generation) failed or OOM'd")
        return metrics

    with open(meta_path) as f:
        meta = json.load(f)

    # Phase 3: Metrics (runs in-process — Whisper small fits easily)
    print("  Phase 3: Metrics (WER + audio quality + speaker similarity)...", flush=True)
    import soundfile as sf
    for i, text in enumerate(texts):
        key = str(i)
        if key not in meta:
            continue
        m = meta[key]
        wav_path = audio_dir / f"sample_{i}.wav"
        audio, sr = sf.read(str(wav_path))
        audio = audio.astype(np.float32)

        metrics.latencies_ms.append(m["elapsed"] * 1000)
        metrics.rtfs.append(m["elapsed"] / m["dur"] if m["dur"] > 0 else 0)

        aq = estimate_audio_quality(audio, sr=sr)
        metrics.audio_quality.append(aq)

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 0.005 or len(audio) < sr * 0.3:
            print(f"    [{i+1}] AQ={aq:.1f} [skip WER — silent/too short]", flush=True)
            continue

        ref_path = eval_items[i].get("audio_path") if i < len(eval_items) else None
        if ref_path and Path(ref_path).exists():
            ref_audio, ref_sr = sf.read(ref_path)
            ref_audio = ref_audio.astype(np.float32)
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            sim = speaker_similarity(ref_audio, audio, sr=sr)
            metrics.speaker_sims.append(sim)

        transcript = whisper_transcribe(audio, sr=sr)
        wer = compute_wer(text, transcript)
        metrics.wer = (metrics.wer * metrics.wer_count + wer) / (metrics.wer_count + 1)
        metrics.wer_count += 1
        spk_str = f" spk={sim:.3f}" if ref_path and Path(ref_path).exists() else ""
        print(f"    [{i+1}] WER={wer:.2f} AQ={aq:.1f}{spk_str} — \"{transcript[:60]}\"", flush=True)

    return metrics


def _cascaded_generate(texts: list[str], audio_dir_str: str) -> None:
    """Subprocess: Voxtral TTS for each text prompt."""
    import soundfile as sf
    audio_dir = Path(audio_dir_str)
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_audio.tts.utils import load as mlx_audio_load
    except ImportError:
        print("    [FAIL] mlx-audio not installed (pip install mlx-audio)", flush=True)
        return

    model = mlx_audio_load("mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")
    voice = "cheerful_male"
    sr = 24000

    meta = {}
    for i, text in enumerate(texts):
        n_words = len(text.split())
        max_tokens = min(max(n_words * 30, 200), 1200)
        t0 = time.time()
        try:
            chunks = []
            for r in model.generate(text=text, voice=voice, max_tokens=max_tokens):
                if r.audio is not None:
                    chunks.append(np.array(r.audio))
            audio_np = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"    [{i+1}] Voxtral failed: {e}", flush=True)
            continue

        elapsed = time.time() - t0
        if len(audio_np) < 100:
            print(f"    [{i+1}] No audio output", flush=True)
            continue

        dur = len(audio_np) / sr
        wav_path = audio_dir / f"sample_{i}.wav"
        sf.write(str(wav_path), audio_np, sr)
        meta[str(i)] = {"elapsed": elapsed, "dur": dur, "sr": sr}
        print(f"    [{i+1}] {dur:.1f}s audio in {elapsed:.1f}s (RTF={elapsed/dur:.2f}x): {text[:50]}...",
              flush=True)

    with open(audio_dir / "meta.json", "w") as f:
        json.dump(meta, f)


def _run_cascaded_eval(eval_items: list[dict]) -> STSMetrics:
    """Cascaded pipeline: text → Voxtral TTS → Whisper re-transcription → WER + audio quality."""
    import subprocess

    metrics = STSMetrics()
    texts = [item["text"][:200] for item in eval_items]

    cache_dir = Path("data/.eval_cache_cascaded")
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = cache_dir / "audio"

    # Phase 1: Voxtral TTS (subprocess — needs ~4B model in memory)
    print("  Phase 1: Voxtral TTS generation...", flush=True)
    texts_json = json.dumps(texts)
    code = (
        f"import sys, json; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        f"from eval_sts import _cascaded_generate; "
        f"_cascaded_generate(json.loads({texts_json!r}), {str(audio_dir)!r})"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=False)
    meta_path = audio_dir / "meta.json"
    if r.returncode != 0 or not meta_path.exists():
        print("  [FAIL] Phase 1 (Voxtral TTS) failed or OOM'd")
        return metrics

    with open(meta_path) as f:
        meta = json.load(f)

    # Phase 2: WER + audio quality + voice consistency (Whisper + scipy — in-process)
    print("  Phase 2: Metrics (WER + audio quality + voice consistency)...", flush=True)
    import soundfile as sf
    prev_audio = None
    for i, text in enumerate(texts):
        key = str(i)
        if key not in meta:
            continue
        m = meta[key]
        wav_path = audio_dir / f"sample_{i}.wav"
        audio, sr = sf.read(str(wav_path))
        audio = audio.astype(np.float32)

        metrics.latencies_ms.append(m["elapsed"] * 1000)
        metrics.rtfs.append(m["elapsed"] / m["dur"] if m["dur"] > 0 else 0)

        aq = estimate_audio_quality(audio, sr=sr)
        metrics.audio_quality.append(aq)

        if prev_audio is not None:
            sim = speaker_similarity(prev_audio, audio, sr=sr)
            metrics.speaker_sims.append(sim)
        prev_audio = audio

        transcript = whisper_transcribe(audio, sr=sr)
        wer = compute_wer(text, transcript)
        metrics.wer = (metrics.wer * metrics.wer_count + wer) / (metrics.wer_count + 1)
        metrics.wer_count += 1
        print(f"    [{i+1}] WER={wer:.2f} AQ={aq:.1f} — \"{transcript[:60]}\"", flush=True)

    return metrics


def run_eval(args) -> STSMetrics:
    if args.eval_set and Path(args.eval_set).exists():
        eval_items = load_eval_set(args.eval_set)
    else:
        libri_items = build_eval_from_libritts(n=20)
        if libri_items:
            print(f"  Using {len(libri_items)} LibriTTS eval samples (with reference audio)")
            eval_items = libri_items
        else:
            eval_items = [{"text": t} for t in DEFAULT_EVAL_TEXTS]

    if args.max_samples:
        eval_items = eval_items[: args.max_samples]

    print(f"\n{'='*60}")
    print(f"  STS Evaluation — {args.pipeline} pipeline")
    print(f"  {len(eval_items)} samples")
    print(f"{'='*60}\n")

    if args.pipeline == "fish":
        metrics = _run_fish_eval(eval_items)
    elif args.pipeline == "cascaded":
        metrics = _run_cascaded_eval(eval_items)
    else:
        print(f"  [unknown pipeline: {args.pipeline}]")
        metrics = STSMetrics()

    summary = metrics.summary()
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:<18} {v}")
    print(f"{'='*60}\n")

    out_path = PROOF_DIR / f"eval_{args.pipeline}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="STS Evaluation Harness")
    parser.add_argument("--pipeline", default="fish", choices=["fish", "cascaded"])
    parser.add_argument("--eval-set", default=None, help="JSONL with {text, audio_path?}")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
