#!/usr/bin/env python3
"""Phase-A live lane scoreboard (specs/phase-a-speech-lane, Task 10).

Drives a lane's live inference over the rendered self-correction audio fixtures and
produces the conversational scoreboard the lane decision reads. This is the model-
in-the-loop glue (kept OUT of the pure conversational_runner): ASR the scenario
audio -> LLM emits the FINAL intent as structured JSON (D8: tool-call grounding) ->
score args against corrected_intent via conversational_scoring.match_value.

Cascade lane = Whisper ASR + the live mlx-server Gemma (localhost:8741).
Vendor lane = disqualified by construction (persona_consistency.vendor_*).

  python3 scripts/run_phase_a_scoreboard.py --lane cascade
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_correction import load_self_correction_scenarios
from conversational_runner import TurnRecord, conversational_scorecard
from render_scenario_audio import DEFAULT_OUT as AUDIO_DIR, render

PROOF = Path(__file__).resolve().parent.parent / "proof-artifacts"
PROOF.mkdir(exist_ok=True)
LLM_URL = "http://localhost:8741/v1/chat/completions"


def asr(wav_path: str, whisper_repo: str) -> str:
    import soundfile as sf
    import mlx_whisper
    a, sr = sf.read(wav_path)
    if a.ndim > 1:
        a = a.mean(axis=1)
    n16 = int(len(a) * 16000 / sr)
    x = np.interp(np.linspace(0, len(a), n16, endpoint=False), np.arange(len(a)), a).astype(np.float32)
    return mlx_whisper.transcribe(x, path_or_hf_repo=whisper_repo)["text"].strip()


def llm_extract_final_intent(transcript: str, keys: list[str], timeout: float = 180.0,
                             retries: int = 1) -> tuple[dict, float]:
    """Constrain the LLM to emit the user's FINAL intent as JSON over the expected keys
    (D8 structured-action grounding). Returns (intent_dict, latency_s). Raises on
    repeated transport failure so the caller can isolate the scenario."""
    sys_prompt = (
        "You extract the user's FINAL request after any self-corrections. "
        "The user may change their mind mid-utterance ('no wait', 'actually', 'I meant'); "
        "always use the LAST stated value. Output ONLY compact JSON, no prose."
    )
    user = f'Extract these fields as JSON {{{", ".join(repr(k) for k in keys)}}} from:\n"{transcript}"'
    body = json.dumps({
        "model": "gemma",
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        "max_tokens": 64, "temperature": 0,
    }).encode()
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            content = out["choices"][0]["message"]["content"].strip()
            return _parse_json(content), time.time() - t0
        except Exception as e:  # noqa: BLE001 — transport/timeout; retry then surface
            last_err = e
    raise RuntimeError(f"LLM extract failed after {retries + 1} attempts: {last_err}")


def _parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        obj = json.loads(s[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_cascade(scenarios: list[dict], whisper_repo: str, max_samples: int | None) -> tuple[list[TurnRecord], int]:
    items = scenarios[:max_samples] if max_samples else scenarios
    records: list[TurnRecord] = []
    errors = 0
    for i, s in enumerate(items, 1):
        wav = AUDIO_DIR / f"{s['id']}.wav"
        keys = list(s["expected_final"].keys())
        try:
            transcript = asr(str(wav), whisper_repo)
            final_intent, lat = llm_extract_final_intent(transcript, keys)
            note = f"asr=\"{transcript[:44]}\" -> {final_intent} ({lat:.1f}s)"
        except Exception as e:  # noqa: BLE001 — isolate the scenario; empty intent scores as fail
            final_intent, errors = {}, errors + 1
            note = f"ERROR ({type(e).__name__}) -> scored as fail"
        records.append(TurnRecord(states=[], should_respond=True, ttfa_ms=None,
                                  self_correction=(final_intent, s)))
        print(f"  [{i}/{len(items)} {s['id']}] {note}", flush=True)
    return records, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="cascade", choices=["cascade"])
    ap.add_argument("--whisper", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    scenarios = load_self_correction_scenarios()
    if not (AUDIO_DIR / f"{scenarios[0]['id']}.wav").exists():
        print("Audio fixtures missing — rendering first...", flush=True)
        render(scenarios)

    print(f"=== Phase-A live scoreboard: lane={args.lane} (whisper={args.whisper}) ===", flush=True)
    records, errors = run_cascade(scenarios, args.whisper, args.max_samples)
    board = conversational_scorecard(records)
    board["lane"] = args.lane
    board["n_scenarios"] = len(records)
    board["n_errors"] = errors
    board["whisper"] = args.whisper
    out = PROOF / f"lane-scoreboard-{args.lane}.json"
    out.write_text(json.dumps(board, indent=2), encoding="utf-8")

    s = board["summary"]
    g = board["scorecard"]["conversational"]
    print("\n=== RESULT ===")
    print(f"  self_correction_pass1 : {s['self_correction_pass1']}  "
          f"(gate >{g['self_correction_pass1']['threshold']} -> pass={g['self_correction_pass1']['pass']})")
    print(f"  scenarios={len(records)}  transport_errors={errors}")
    print(f"  frontier ref (FDB-v3 GPT-Realtime): 0.588 ; cascaded baseline: 0.176")
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
