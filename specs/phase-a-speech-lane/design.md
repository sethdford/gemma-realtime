# Phase-A Speech-Lane Decision — Design

> Approved requirements: `specs/phase-a-speech-lane/requirements.md`
> Companion roadmap: `docs/research/2026-06-06-s2s-L1-L5-roadmap.md`
> All file:line refs verified by read-only exploration of the repo (2026-06-06).

## Summary of approach

Phase A is a **measurement-driven decision**, not a build. We extend the *existing* harness (`scripts/eval_sts.py`) into a three-lane scoreboard, add the four missing conversational metrics, and add a persona-portability probe. The decision rule is then mechanical: the lane that (a) preserves the frozen-Gemma+LoRA persona across modalities AND (b) clears the frontier exit gates becomes primary; cascade stays as the runtime floor; native vendor S2S is excluded from the persona path (proven, not asserted, by AC-2).

## Components

| Component | What it does | Where it lives | New/Existing |
|---|---|---|---|
| **Lane scoreboard** | One command runs all 3 lanes on identical inputs → JSON | extend `scripts/eval_sts.py` (`--pipeline {fish,cascaded,vendor}`, currently `{fish,cascaded}` @ eval_sts.py:855) | extend |
| **Conversational metrics** | turn-take rate, interruption avoidance, TTFA p50, self-correction Pass@1 | new fields in `STSMetrics` (eval_sts.py:40) + `_compute_scorecard` (eval_sts.py:125) | new |
| **Realistic-audio set** | LibriSpeech + noise(15 dB SNR) + accent set + 21 self-correction scenarios | new `data/realistic-audio/` + loader | new |
| **Persona-portability probe** | Same Gemma+LoRA → trait vector in text path vs speech lane; cosine compare | new `scripts/persona_consistency.py` | new |
| **Vendor baseline adapter** | Thin gpt-realtime client for scoreboard contrast (AC-1/AC-2 only) | new `scripts/vendor_s2s_baseline.py` | new |
| **Duplex state source** | LISTEN/SPEAK/INTERRUPT already predicted | `DuplexStatePredictor` (speech_decoder.py:316), `predict_state()` (fish_sts.py:641) | existing (reuse) |
| **Latency instrumentation** | per-turn TTFA/E2E already emitted | `first_audio_ms` (realtime-ws.py:910), `build_turn_record` (speech_metrics.py:23) | existing (aggregate) |
| **Lane switch** | startup backend select + auto fish→cascade fallback on load error | `--tts` (realtime-ws.py:1019), `self._cascaded` fallback (sota_pipeline.py:452) | existing (extend to config) |
| **Decision record** | the ADR with thresholds + reversal trigger | `specs/phase-a-speech-lane/DECISION.md` (written at end) | new |

## Data flow (scoreboard run)

1. Loader yields N≥100 realistic-audio samples per branch (clean + noise + accent + self-correction subsets).
2. For each lane ∈ {fish, cascaded, vendor}: feed identical audio → capture (transcript, output audio, per-turn metrics JSON via `speech_metrics`).
3. Compute per-sample: WER (eval_sts.py:73), audio-quality (eval_sts.py:209), speaker-sim (eval_sts.py:251), **TTFA** (from `first_audio_ms`), **turn-take** (from `DuplexStatePredictor` transitions vs ground-truth boundaries), **interruption-avoidance** (state==SPEAK held through a barge-in token), **self-correction Pass@1** (final intent matches corrected intent on the 21 scenarios).
4. Aggregate → `STSMetrics.summary()` (add p50/p95 for TTFA, rates for turn-take/interruption/self-correction).
5. Persona probe (separate): run a fixed persona prompt through (text path) and (each speech lane → ASR re-transcribe → trait-score); cosine the trait vectors.
6. Emit `proof-artifacts/lane-scoreboard.json` + a markdown summary; write `DECISION.md`.

## Decisions

- **D1 — Extend `eval_sts.py`, do not build a new framework** (serves AC-1, AC-3/4/5/6; Constraint "harness reuse"). It already has the dataclass, scorecard, JSON emit, and `--pipeline` switch; we add fields + a third lane. Rejected: standing up Full-Duplex-Bench as an external dep (heavier, not Apple-Silicon-native).
- **D2 — Reuse the existing `DuplexStatePredictor` for turn-take/interruption metrics** (serves AC-3, AC-4). States LISTEN/SPEAK/INTERRUPT already exist (speech_decoder.py:316, fish_sts.py:641); we *measure* transitions, we do not rebuild prediction. Rejected: a separate VAD-based turn detector (would measure a different thing than the model actually uses).
- **D3 — Persona portability is the decisive, falsifiable test** (serves AC-2). The fish lane loads the *same* `gemma, tokenizer` as text (realtime-ws.py:610–613) and shares the first `split_layer` layers (`_make_shared_fn`, fish_sts.py:883) → persona *should* transfer. The vendor lane *cannot* load Gemma+LoRA → must fail AC-2 by construction. This is what excludes vendor S2S, with evidence. Rejected: excluding vendor on assertion alone.
- **D4 — Cascade is the floor, selectable by flag** (serves AC-7, Constraint "fallback guarantee"). `--tts` already selects backend (realtime-ws.py:1019) and fish→cascade auto-fallback exists at load (sota_pipeline.py:452). We add a **config-level** lane field + a `which-lane-active` health signal so fallback is observable, not silent. Rejected: mid-session hot-swap (out of scope; restart-to-switch is acceptable for a floor).
- **D5 — Aggregate existing latency instrumentation rather than re-instrument** (serves AC-5). `first_audio_ms` + `append_turn_metrics` already exist (realtime-ws.py:910, speech_metrics.py); we only add p50/p95 rollup to the scorecard. Rejected: new timing hooks (duplicate, drift risk).
- **D6 — AC-6 is a feasibility probe in Phase A** (per approved default). We measure baseline self-correction Pass@1 on all lanes (to confirm the gap) and assert the fish lane's `DuplexStatePredictor` exposes the INTERRUPT state needed for a future rollback engine — proving the lane *doesn't preclude* L4 repair. Full rollback engine = Phase B. Rejected: building rollback now (scope creep past a decision spec).
- **D7 — gpt-realtime as the vendor baseline** (per approved default; serves AC-1/AC-2). Strongest L3 (FDB-v3 Pass@1 0.600, tool F1 0.876) → hardest contrast. Rejected: Nova 2 Sonic (cheapest, but a weaker expressivity contrast).

## Addendum (2026-06-06, from the live smoke run)

Running `eval_sts.py --pipeline cascaded --bundle smoke` on-device confirmed it is
a **batch TTS/STS quality** harness: per-sample timing exposes `elapsed` (total
utterance synth) + `dur` only — **no first-audio timestamp**. Therefore the four
conversational gates do NOT come from this loop:

- **TTFA, turn-take, interruption, self-correction** require a **separate
  streaming/conversational runner** that drives the `duplex-predictor` adapter
  (LISTEN/SPEAK/INTERRUPT) and captures `first_audio_ms` from the streaming path
  (consistent with D5). `eval_sts.py` keeps reporting the *quality* gates (WER,
  AQ, speaker-sim, total latency) + emits the conversational gates as `null`
  (correctly unmeasured) until the streaming runner populates them.
- **Persona probe default (AC-2):** a heuristic **trait-vector cosine** — a fixed
  trait set scored from text via word-boundary lexical markers (per
  `substring-classifier-pitfalls`), cosine ≥ 0.85 = "same persona." No h-uman
  scorer exists to reuse, so this is the spec default. Pure/testable without a model.

Scoring cores built this round (model-free, unit-tested): `conversational_scoring.py`
(self-correction + turn-take + interruption) and `persona_consistency.py`. The live
model-driving glue (audio→states, lane runs) is the remaining integration boundary.

## Risks

- **R1 — Persona trait-scorer doesn't exist yet** (Explore: "NOT FOUND" in scripts/ or tests/). Mitigation: AC-2 default threshold (cosine ≥ 0.85) on a defined trait vector; if h-uman has a scorer elsewhere, wire it; else build a minimal one in `persona_consistency.py`. This is the critical-path new code.
- **R2 — Self-correction scenarios are scarce** (FDB-v3 used only 21). Mitigation: port/recreate the 21-scenario structure; report Pass@1 as feasibility signal, not a hard gate, in Phase A (D6).
- **R3 — Vendor baseline needs cloud + audio egress**, conflicting with AC-8. Mitigation: vendor lane runs *only* in the scoreboard (AC-1/AC-2), clearly flagged as comparison-only and never on the product hot path; AC-8 applies to the chosen primary lane.
- **R4 — Realistic-audio set construction is non-trivial.** Mitigation: start with LibriSpeech + additive noise (15 dB SNR to match τ-Voice) + a small accent subset; expand later. Bounded so it doesn't balloon Phase A.
- **R5 — Turn-take/interruption metric definitions could drift from FDB-v3.** Mitigation: mirror FDB-v3 definitions exactly (turn-take rate = % turns yielded correctly; interruption avoidance = SPEAK held through non-terminal barge-in) so numbers are comparable to the published leaderboard.
- **R6 — Lane decision could be premature if fish lane is under-trained.** Mitigation: AC-9 reversal trigger — if fish can't clear AC-3/4/6 within the agreed window, cascade-with-micro-turns becomes primary and fish stays in R&D.
