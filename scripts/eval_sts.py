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

Fish cb0 sampling (same env as ``fish_sts.FishSTSPipeline``): FISH_STS_CB0_TEMPERATURE,
FISH_STS_CB0_TOP_K, FISH_STS_CB0_TOP_P, FISH_STS_CB0_REPETITION_PENALTY,
FISH_STS_CB0_REP_WINDOW, FISH_STS_CB0_MAX_FRAMES.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PROOF_DIR = Path("proof-artifacts")
PROOF_DIR.mkdir(exist_ok=True)


@dataclass
class STSMetrics:
    wer: float = 0.0
    wer_count: int = 0
    wer_skipped_short_ref: int = 0
    audio_quality: list[float] = field(default_factory=list)
    speaker_sims: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    rtfs: list[float] = field(default_factory=list)
    hypothesis_diversity: list[float] = field(default_factory=list)
    failure_tags: Counter = field(default_factory=Counter)
    # Phase-A conversational gate metrics (specs/phase-a-speech-lane). Each is a
    # list of per-sample observations; aggregated in summary().
    #   turn_take            : 1.0 if the agent yielded/took the turn correctly (AC-3)
    #   interruption_avoidance: 1.0 if SPEAK was held through a non-terminal barge-in (AC-4)
    #   ttfa_ms              : time-to-first-audio per turn, ms (AC-5)
    #   self_correction      : 1.0 if final intent matched the corrected intent (AC-6)
    turn_take: list[float] = field(default_factory=list)
    interruption_avoidance: list[float] = field(default_factory=list)
    ttfa_ms: list[float] = field(default_factory=list)
    self_correction: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "wer": round(self.wer, 4) if self.wer_count else None,
            "wer_n_scored": self.wer_count,
            "wer_skipped_short_ref": self.wer_skipped_short_ref,
            "hypothesis_diversity_mean": round(float(np.mean(self.hypothesis_diversity)), 3)
            if self.hypothesis_diversity else None,
            "audio_quality_mean": round(float(np.mean(self.audio_quality)), 3) if self.audio_quality else None,
            "audio_quality_std": round(float(np.std(self.audio_quality)), 3) if self.audio_quality else None,
            "spk_sim_mean": round(float(np.mean(self.speaker_sims)), 3) if self.speaker_sims else None,
            "latency_p50_ms": round(float(np.median(self.latencies_ms)), 1) if self.latencies_ms else None,
            "latency_p95_ms": round(float(np.percentile(self.latencies_ms, 95)), 1) if self.latencies_ms else None,
            "rtf_mean": round(float(np.mean(self.rtfs)), 4) if self.rtfs else None,
            # Phase-A conversational gates (None until populated):
            "turn_take_rate": round(float(np.mean(self.turn_take)), 4) if self.turn_take else None,
            "interruption_avoidance": round(float(np.mean(self.interruption_avoidance)), 4)
            if self.interruption_avoidance else None,
            "ttfa_p50_ms": round(float(np.median(self.ttfa_ms)), 1) if self.ttfa_ms else None,
            "ttfa_p95_ms": round(float(np.percentile(self.ttfa_ms, 95)), 1) if self.ttfa_ms else None,
            "self_correction_pass1": round(float(np.mean(self.self_correction)), 4)
            if self.self_correction else None,
            "failure_tags": dict(self.failure_tags),
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


def hypothesis_word_diversity(transcript: str) -> float | None:
    """Unique-token ratio for collapse spotting (None if too short)."""
    words = transcript.lower().split()
    if len(words) < 3:
        return None
    return len(set(words)) / len(words)


def _failure_tags_for_sample(
    *,
    reference: str,
    transcript: str,
    audio_quality: float,
    duration_s: float,
) -> list[str]:
    tags: list[str] = []
    ref_words = reference.strip().split()
    hyp_words = transcript.strip().split()
    if len(ref_words) < 2:
        tags.append("short_reference")
    if not transcript.strip():
        tags.append("empty_hypothesis")
    if duration_s < 0.35:
        tags.append("too_short_audio")
    if audio_quality < 2.0:
        tags.append("low_audio_quality")
    if len(hyp_words) >= 8:
        uniq_ratio = len(set(w.lower() for w in hyp_words)) / max(1, len(hyp_words))
        if uniq_ratio < 0.35:
            tags.append("repetition_collapse")
    return tags


def _compute_scorecard(summary: dict) -> dict:
    """Balanced score combining intelligibility, latency, diversity, and stability."""
    wer = summary.get("wer")
    lat50 = summary.get("latency_p50_ms")
    lat95 = summary.get("latency_p95_ms")
    div = summary.get("hypothesis_diversity_mean")
    aq = summary.get("audio_quality_mean")
    fail_tags = summary.get("failure_tags") or {}
    n = max(1, int(summary.get("n_samples") or 1))
    bad = sum(int(v) for k, v in fail_tags.items() if k in {"empty_hypothesis", "repetition_collapse", "low_audio_quality"})
    bad_rate = bad / n

    wer_score = max(0.0, min(1.0, 1.0 - ((wer if wer is not None else 3.0) / 3.0)))
    lat50_score = max(0.0, min(1.0, 1.0 - ((float(lat50 or 4000.0) - 900.0) / 2600.0)))
    lat95_score = max(0.0, min(1.0, 1.0 - ((float(lat95 or 8000.0) - 2200.0) / 6000.0)))
    div_score = max(0.0, min(1.0, float(div or 0.0)))
    aq_score = max(0.0, min(1.0, (float(aq or 0.0) - 1.0) / 4.0))
    stability = max(0.0, 1.0 - bad_rate)

    overall = (
        0.38 * wer_score
        + 0.20 * lat50_score
        + 0.10 * lat95_score
        + 0.12 * div_score
        + 0.12 * aq_score
        + 0.08 * stability
    )
    return {
        "overall": round(float(overall), 4),
        "components": {
            "wer_score": round(float(wer_score), 4),
            "latency_p50_score": round(float(lat50_score), 4),
            "latency_p95_score": round(float(lat95_score), 4),
            "diversity_score": round(float(div_score), 4),
            "audio_quality_score": round(float(aq_score), 4),
            "stability_score": round(float(stability), 4),
        },
        "conversational": _conversational_gates(summary),
    }


# Phase-A exit gates vs the mid-2026 frontier (specs/phase-a-speech-lane,
# docs/research/2026-06-06-s2s-L1-L5-roadmap.md). higher_is_better says which
# direction passes; threshold is the bar the chosen lane must clear.
CONVERSATIONAL_GATES = {
    "turn_take_rate": {"threshold": 0.95, "higher_is_better": True},          # AC-3 (FDB-v3: Cascaded 1.00, Gemini 3.1 0.78)
    "interruption_avoidance": {"threshold": 0.135, "higher_is_better": True}, # AC-4 (FDB-v3 leader GPT-Realtime 0.135)
    "ttfa_p50_ms": {"threshold": 400.0, "higher_is_better": False},           # AC-5 (<400 ms p50, realistic audio)
    "self_correction_pass1": {"threshold": 0.60, "higher_is_better": True},   # AC-6 (FDB-v3 leader GPT-Realtime 0.588)
}


def _conversational_gates(summary: dict) -> dict:
    """Per-gate {value, threshold, pass} for the Phase-A lane decision.

    pass is None when the metric was not measured (value None) — an unmeasured
    gate is neither pass nor fail, so the decision step knows to collect it.
    """
    out: dict = {}
    for key, spec in CONVERSATIONAL_GATES.items():
        value = summary.get(key)
        thr = spec["threshold"]
        if value is None:
            passed = None
        elif spec["higher_is_better"]:
            passed = bool(value >= thr) if key == "turn_take_rate" else bool(value > thr)
        else:
            passed = bool(value < thr)
        out[key] = {"value": value, "threshold": thr, "pass": passed}
    return out


def whisper_transcribe(audio_np: np.ndarray, sr: int = 24000) -> str:
    """Transcribe audio with mlx-whisper."""
    import tempfile
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        x = audio_np.astype(np.float32).reshape(-1)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        # Very quiet STS / codec outputs often decode to Whisper-empty transcripts.
        if peak > 1e-6:
            x = x * min(0.95 / peak, 50.0)

        if sr != 16000:
            n_out = int(len(x) * 16000 / sr)
            audio_16k = np.interp(
                np.linspace(0, 1, n_out),
                np.linspace(0, 1, len(x)),
                x,
            ).astype(np.float32)
        else:
            audio_16k = x
        sf.write(f.name, audio_16k, 16000)

        try:
            import mlx_whisper
            # Default mlx_whisper temperature=(0..1) retries often yield **no segments**
            # on short or quirky STS waveforms; greedy decode is a better eval gate.
            result = mlx_whisper.transcribe(
                f.name,
                path_or_hf_repo="mlx-community/whisper-small-mlx",
                language="en",
                temperature=0.0,
                no_speech_threshold=0.0,
                logprob_threshold=-2.0,
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

BUNDLES = {
    "smoke": 20,
    "dev": 100,
    "gate": 200,
}


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
                     audio_paths: list[str] | None = None,
                     use_gemma: bool = False,
                     weights_path: str | None = None) -> None:
    """Subprocess: load STS model + codec, generate audio, save wavs.

    When audio_paths are provided, uses the audio-to-audio pathway
    (encode reference → audio_input projection → generate), which matches
    how the model was trained. Falls back to text embeddings when no
    reference audio is available.

    When use_gemma is True, also loads Gemma's frozen shared layers and
    forwards embeddings through them before generation (produces
    semantically-rich context matching the training pipeline).
    """
    import mlx.core as mx
    from fish_sts import FishSpeechToSpeech, FishSTSPipeline, cb0_sampler_env_overrides
    from codec import AudioCodec, CodecTokens, CodecType
    import soundfile as sf

    w_path = FishSTSPipeline._resolve_weights(weights_path)
    if w_path is None:
        print("    [SKIP] No trained STS weights")
        return
    try:
        config = FishSTSPipeline.config_from_weights(str(w_path))
    except Exception:
        # Early checkpoints (e.g., phase-a) may not contain full depth keys.
        # Fall back to any full checkpoint for architecture shape inference.
        base_full = FishSTSPipeline._resolve_weights(None)
        if base_full is None:
            raise
        config = FishSTSPipeline.config_from_weights(str(base_full))
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

    _gemma_shared = None
    if use_gemma:
        from mlx_lm import load as lm_load
        print("    Loading Gemma for shared layer inference...", flush=True)
        _gemma, _tok = lm_load("mlx-community/gemma-4-26b-a4b-it-4bit")
        _inner = _gemma.language_model.model if hasattr(_gemma, "language_model") else _gemma.model
        _split_idx = min(28, len(_inner.layers) // 2)
        _shared_layers = _inner.layers[:_split_idx]

        def _gemma_shared(embeddings):
            h = embeddings
            for layer in _shared_layers:
                out = layer(h, mask=None)
                h = out[0] if isinstance(out, tuple) else out
            return mx.stop_gradient(h)

        print(f"    Gemma shared layers: {_split_idx}", flush=True)

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
            if _gemma_shared is not None:
                emb = _gemma_shared(emb)
                mx.eval(emb)
            input_mode = "audio"
        else:
            key = f"emb_{i}"
            if key not in embs:
                continue
            emb = mx.array(embs[key][np.newaxis])
            if _gemma_shared is not None:
                emb = _gemma_shared(emb)
                mx.eval(emb)
            input_mode = "text"

        n_words = len(text.split())
        samp = cb0_sampler_env_overrides(
            temperature=0.3,
            top_k=30,
            repetition_penalty=1.2,
            rep_window=16,
            max_frames_cap=400,
            top_p=0.92,
        )
        max_frames = min(max(n_words * 8, 60), samp["max_frames_cap"])
        cb0_out, speech_hidden = model.generate_cb0(
            emb,
            temperature=samp["temperature"],
            top_k=samp["top_k"],
            max_frames=max_frames,
            repetition_penalty=samp["repetition_penalty"],
            rep_window=samp["rep_window"],
            top_p=float(samp.get("top_p", 1.0)),
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


def _run_fish_eval(eval_items: list[dict], use_gemma: bool = False,
                   weights_path: str | None = None) -> tuple[STSMetrics, list[dict]]:
    """Run eval in separate subprocesses so GPU memory is truly freed between phases."""
    import subprocess

    metrics = STSMetrics()
    samples: list[dict] = []
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
        return metrics, samples

    # Phase 2: STS generation (subprocess)
    audio_paths = [item.get("audio_path", "") for item in eval_items]
    audio_paths_json = json.dumps(audio_paths)
    code2 = (
        f"import sys, json; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        f"from eval_sts import _phase2_generate; from pathlib import Path; "
        f"_phase2_generate(json.loads({texts_json!r}), Path({str(emb_path)!r}), Path({str(audio_dir)!r}), "
        f"json.loads({audio_paths_json!r}), use_gemma={use_gemma}, weights_path={weights_path!r})"
    )
    r = subprocess.run([sys.executable, "-c", code2], capture_output=False)
    meta_path = audio_dir / "meta.json"
    if r.returncode != 0 or not meta_path.exists():
        print("  [FAIL] Phase 2 (speech generation) failed or OOM'd")
        return metrics, samples

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
        duration_s = len(audio) / max(1, sr)

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 0.005 or len(audio) < sr * 0.3:
            metrics.failure_tags.update(["silent_or_too_short"])
            samples.append({
                "index": i + 1,
                "wer": None,
                "aq": round(float(aq), 3),
                "spk_sim": None,
                "duration_s": round(float(duration_s), 3),
                "rtf": round(float(m["elapsed"] / m["dur"]) if m["dur"] > 0 else 0.0, 4),
                "transcript": "",
                "tags": ["silent_or_too_short"],
            })
            print(f"    [{i+1}] AQ={aq:.1f} [skip WER — silent/too short]", flush=True)
            continue

        ref_path = eval_items[i].get("audio_path") if i < len(eval_items) else None
        sim = None
        if ref_path and Path(ref_path).exists():
            ref_audio, ref_sr = sf.read(ref_path)
            ref_audio = ref_audio.astype(np.float32)
            if ref_audio.ndim > 1:
                ref_audio = ref_audio.mean(axis=1)
            sim = speaker_similarity(ref_audio, audio, sr=sr)
            metrics.speaker_sims.append(sim)

        transcript = whisper_transcribe(audio, sr=sr)
        div = hypothesis_word_diversity(transcript)
        if div is not None:
            metrics.hypothesis_diversity.append(div)
        tags = _failure_tags_for_sample(
            reference=text,
            transcript=transcript,
            audio_quality=aq,
            duration_s=duration_s,
        )
        metrics.failure_tags.update(tags)

        ref_words = text.strip().split()
        if len(ref_words) < 2:
            metrics.wer_skipped_short_ref += 1
            spk_str = f" spk={sim:.3f}" if ref_path and Path(ref_path).exists() else ""
            samples.append({
                "index": i + 1,
                "wer": None,
                "aq": round(float(aq), 3),
                "spk_sim": round(float(sim), 3) if sim is not None else None,
                "duration_s": round(float(duration_s), 3),
                "rtf": round(float(m["elapsed"] / m["dur"]) if m["dur"] > 0 else 0.0, 4),
                "transcript": transcript,
                "tags": tags,
            })
            print(
                f"    [{i+1}] [skip WER — ref <2 words] AQ={aq:.1f}{spk_str} — \"{transcript[:60]}\"",
                flush=True,
            )
            continue

        wer = compute_wer(text, transcript)
        metrics.wer = (metrics.wer * metrics.wer_count + wer) / (metrics.wer_count + 1)
        metrics.wer_count += 1
        spk_str = f" spk={sim:.3f}" if ref_path and Path(ref_path).exists() else ""
        samples.append({
            "index": i + 1,
            "wer": round(float(wer), 4),
            "aq": round(float(aq), 3),
            "spk_sim": round(float(sim), 3) if sim is not None else None,
            "duration_s": round(float(duration_s), 3),
            "rtf": round(float(m["elapsed"] / m["dur"]) if m["dur"] > 0 else 0.0, 4),
            "transcript": transcript,
            "tags": tags,
        })
        print(f"    [{i+1}] WER={wer:.2f} AQ={aq:.1f}{spk_str} — \"{transcript[:60]}\"", flush=True)

    return metrics, samples


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


def _run_cascaded_eval(eval_items: list[dict]) -> tuple[STSMetrics, list[dict]]:
    """Cascaded pipeline: text → Voxtral TTS → Whisper re-transcription → WER + audio quality."""
    import subprocess

    metrics = STSMetrics()
    samples: list[dict] = []
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
        return metrics, samples

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
        duration_s = len(audio) / max(1, sr)

        if prev_audio is not None:
            sim = speaker_similarity(prev_audio, audio, sr=sr)
            metrics.speaker_sims.append(sim)
        prev_audio = audio

        transcript = whisper_transcribe(audio, sr=sr)
        div = hypothesis_word_diversity(transcript)
        if div is not None:
            metrics.hypothesis_diversity.append(div)
        tags = _failure_tags_for_sample(
            reference=text,
            transcript=transcript,
            audio_quality=aq,
            duration_s=duration_s,
        )
        metrics.failure_tags.update(tags)

        ref_words = text.strip().split()
        if len(ref_words) < 2:
            metrics.wer_skipped_short_ref += 1
            samples.append({
                "index": i + 1,
                "wer": None,
                "aq": round(float(aq), 3),
                "spk_sim": None,
                "duration_s": round(float(duration_s), 3),
                "rtf": round(float(m["elapsed"] / m["dur"]) if m["dur"] > 0 else 0.0, 4),
                "transcript": transcript,
                "tags": tags,
            })
            print(
                f"    [{i+1}] [skip WER — ref <2 words] AQ={aq:.1f} — \"{transcript[:60]}\"",
                flush=True,
            )
            continue

        wer = compute_wer(text, transcript)
        metrics.wer = (metrics.wer * metrics.wer_count + wer) / (metrics.wer_count + 1)
        metrics.wer_count += 1
        samples.append({
            "index": i + 1,
            "wer": round(float(wer), 4),
            "aq": round(float(aq), 3),
            "spk_sim": None,
            "duration_s": round(float(duration_s), 3),
            "rtf": round(float(m["elapsed"] / m["dur"]) if m["dur"] > 0 else 0.0, 4),
            "transcript": transcript,
            "tags": tags,
        })
        print(f"    [{i+1}] WER={wer:.2f} AQ={aq:.1f} — \"{transcript[:60]}\"", flush=True)

    return metrics, samples


def run_eval(args) -> STSMetrics:
    if args.eval_set and Path(args.eval_set).exists():
        eval_items = load_eval_set(args.eval_set)
    else:
        bundle_n = BUNDLES.get(args.bundle, 20)
        libri_items = build_eval_from_libritts(n=bundle_n, seed=args.seed)
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
        metrics, samples = _run_fish_eval(
            eval_items,
            use_gemma=getattr(args, 'use_gemma', False),
            weights_path=getattr(args, "weights", None),
        )
    elif args.pipeline == "cascaded":
        metrics, samples = _run_cascaded_eval(eval_items)
    else:
        print(f"  [unknown pipeline: {args.pipeline}]")
        metrics = STSMetrics()
        samples = []

    summary = metrics.summary()
    scorecard = _compute_scorecard(summary)
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:<18} {v}")
    print(f"  sota_score         {scorecard['overall']}")
    print(f"{'='*60}\n")

    out_path = PROOF_DIR / f"eval_{args.pipeline}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")
    score_path = PROOF_DIR / f"eval_{args.pipeline}_scorecard.json"
    score_out = {
        "bundle": args.bundle,
        "seed": args.seed,
        "pipeline": args.pipeline,
        "summary": summary,
        "scorecard": scorecard,
    }
    with open(score_path, "w") as f:
        json.dump(score_out, f, indent=2)
    print(f"  Saved to {score_path}")
    samples_path = PROOF_DIR / f"eval_{args.pipeline}_samples.jsonl"
    with open(samples_path, "w") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved to {samples_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="STS Evaluation Harness")
    parser.add_argument("--pipeline", default="fish", choices=["fish", "cascaded"])
    parser.add_argument("--eval-set", default=None, help="JSONL with {text, audio_path?}")
    parser.add_argument("--bundle", default="smoke", choices=sorted(BUNDLES.keys()),
                        help="Built-in eval size when --eval-set is omitted (default: smoke=20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed for built-in LibriTTS bundle")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-gemma", action="store_true",
                        help="Route through Gemma's shared layers (FishSTSPipeline.process_audio)")
    parser.add_argument("--weights", default=None,
                        help="Optional fish_sts .safetensors path (fish pipeline only)")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
