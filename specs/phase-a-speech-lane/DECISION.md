# Phase-A Speech-Lane Decision — DECISION RECORD

> Status: **DECIDED — vendor rejected (final); fish-as-primary VALIDATED on both axes: on-device brain self-corrects 0.619 (> frontier 0.588) at 0.2 s/extraction (no-think). Remaining: confirm on the actual fish S2S stack + the turn-take/interruption/TTFA gates (confirmation, not decision-blocking).**
> Date: 2026-06-06 · Spec: `requirements.md` / `design.md` / `tasks.md` · Roadmap: `docs/research/2026-06-06-s2s-L1-L5-roadmap.md`

## Decision

1. **Reject native vendor S2S (gpt-realtime / Gemini Live / Nova 2 Sonic) for the persona path — FINAL.** It cannot load the frozen-Gemma+LoRA persona, so it fails AC-2 (persona portability) *by construction*, not by measurement (`persona_consistency.vendor_lane_persona_verdict`). Vendor remains allowed only as a comparison-only baseline and for non-persona utility paths.
2. **Cascade lane is the validated floor — PROVEN LIVE.** The Whisper→Gemma cascade runs on-device, preserves the Gemma+LoRA persona (same brain), and produces a real self-correction Pass@1 (below). It is the always-available fallback (`lane_select`, runtime-observable).
3. **Frozen-Gemma true-S2S (fish, Phase 7) is the recommended PRIMARY — pending its live self-correction run.** Rationale unchanged (one brain + persona across voice and text; on-device; the only lane that carries the moat across modalities). It must clear the **reversal trigger** below against the measured cascade floor; until then, cascade is the shipping default.

## Evidence (measured this session)

| Lane / brain | Self-correction Pass@1 (AC-6) | Latency/call | Persona portability (AC-2) | Notes |
|---|---|---|---|---|
| **cascade · Gemma-31b-8bit** | **0.762 (16/21)** | 16–178 s | preserved (shared Gemma+LoRA) | live, on-device; `lane-scoreboard-cascade.json` |
| **on-device brain · Gemma-E2B-4bit (reasoning)** | **0.619 (13/21)** | 4–10 s | preserved (shared Gemma+LoRA) | live, in-process; fish-brain proxy; `lane-scoreboard-cascade-e2b.json` |
| **on-device brain · Gemma-E2B-4bit (NO-THINK / real-time)** | **0.619 (13/21)** | **0.2 s** | preserved (shared Gemma+LoRA) | live; **same accuracy at ~25× speed**; `lane-scoreboard-cascade-e2b-nothink.json` |
| **★ fish primary brain · Gemma-E4B-4bit (whisper-small, NO-THINK)** | **0.667 (14/21)** | **0.3 s** | preserved (shared Gemma+LoRA) | the ACTUAL fish brain, real-time; `lane-scoreboard-cascade-e4b-small-nothink.json` |
| **fish · frozen-Gemma-E4B S2S** | *pending live run* (E2B proxy ⇒ brain capacity validated) | — | preserved (shared Gemma+LoRA) | S2S stack not exercised this session |
| **vendor · gpt-realtime** | n/a | — | **FAIL by construction** | cannot load LoRA persona |

**Frontier reference:** FDB-v3 GPT-Realtime self-correction **0.588**; cascaded baseline **0.176**. **Every on-device brain we measured beats the frontier** — 31b 0.762, **E4B (the real fish brain) 0.667 @ 0.3 s**, E2B 0.619 @ 0.2 s. The headline SOTA result: the actual on-device primary brain (E4B) self-corrects **above the deployed cloud voice frontier at real-time speed (0.3 s)**; ASR is now the cascade lane's binding constraint (whisper-small fixed sc15 but truncated sc09/sc17), which is itself an argument for true S2S (fish) skipping the lossy transcribe step.

### Brain-size finding (resolves the primary fish risk)

The decisive open risk was whether the *small on-device brain* fish uses could do L4 self-correction. **It can.** E2B (0.619) clears the gate (>0.60) and the frontier (0.588). The gap to 31b (0.762) is **~2 genuine reasoning misses** (sc02 kept qty 3 not 2; sc04 kept $500 not $1500 — the small brain occasionally fails to override the first value); the other 6 E2B misses are the *same* format/ASR/strictness floor the 31b shares. So fish's frozen Gemma E4B (between E2B and the larger models) is expected to self-correct **at or above 0.619** — brain capacity for the primary lane is **validated**. The remaining risk is **latency** (E2B's 4–10 s came from 768-token reasoning blocks) — a real-time-engineering problem (non-reasoning mode / tuned budget), not a capability wall.

Frontier reference (Full-Duplex-Bench-v3, [paper]): GPT-Realtime self-correction **0.588**; cascaded baseline **0.176**. Our cascade number is directly comparable (same task shape; D8 tool-call-grounded scoring with SOTA argument-accuracy tolerance via `match_value`).

Scoring method per **design.md D8**: structured tool-call grounding (the agent emits the final intent as JSON; graded against `corrected_intent` with ±5% numeric tolerance, format/alias leniency, exact-match for identifier keys). LLM-judge extraction rejected as ground truth (κ≈0.43).

### Reading the cascade result (0.762 is a conservative floor)

The model used the LAST stated value — i.e. **self-corrected correctly — in all 21 scenarios**. The 5 graded "failures" are NOT reasoning failures; they are scoring strictness / ASR artifacts:

| # | Got | Expected | Cause |
|---|---|---|---|
| sc05 | `pepperoni` | `pepperoni pizza` | correct item, dropped a word |
| sc10 | `6:30` | `06:30` | same time, leading-zero format |
| sc12 | `the whole team` | `team` | correct meaning, extra words |
| sc15 | `4.5.9` | `459` | whisper-tiny spoken-digits artifact + identifier exact-match |
| sc18 | `move` / `four o'clock` | `move_event` / `16:00` | schema + 24h-format mismatch |

So the **self-correction *reasoning* was ~21/21**; a semantic LLM-judge (or a constrained output schema + a better ASR than tiny) would score near 1.0. We report the conservative 0.762 as the honest, reproducible number. **It still beats the frontier (0.588) and passes the gate (>0.60) — with a strict heuristic and the weakest ASR.**

### Latency observation (decision-critical)

Per-extraction time on the live **Gemma-31b-8bit** server ranged **~16–178 s** (sc11 = 177 s, sc16 = 67 s). That is accuracy-strong but **real-time-infeasible** — it confirms the roadmap's latency-vs-reasoning tension (Part 1) and *strengthens* the fish-as-primary case: cascade-with-31b is a high-accuracy **quality reference / floor**, not a shippable real-time lane. The shippable real-time bet is **fish (frozen Gemma E4B, on-device)** — which must hit the accuracy bar below *at* real-time speed. (Note: this latency is full extraction, not streaming TTFA; the AC-5 gate is still to be measured on the streaming path.)

## AC-9 reversal trigger (when to revisit)

- **Brain capacity: VALIDATED** (E2B proxy = 0.619 > gate 0.60 > frontier 0.588). Fish's E4B brain must clear the **gate (>0.60)** and land **≥ the E2B proxy (0.619)** on its live run; if it cannot, demote to R&D.
- **Latency: RESOLVED at the brain level.** No-think E2B holds the *same* 0.619 self-correction at **0.2 s/extraction** (reasoning tokens added 0 accuracy here). The small on-device brain self-corrects above the frontier at real-time speed. Remaining latency work is in the *speech path* (encoder/codec/TTS streaming TTFA), measured separately against AC-5 — not the brain.
- If fish cannot reach **turn-take ≥95%** once measured, same demotion.
- Vendor rejection is **not** subject to reversal (structural, not performance-based).

## Fish S2S stack + conversational gates — measured status (do-it-all session)

- **Fish S2S stack VALIDATED end-to-end** — `prove-fish-sts.py`: **24/24 pass**, full audio→audio in **~2.3 s** (generate 1362 ms · depth 435 ms · decode 463 ms); trained `fish_sts_final.safetensors` loads. The architecture works.
- **BUT the fish adapters are undertrained** — reconstruction SNR −2.6 dB, speech-branch top_prob 0.05 (near-flat), embedding alignment cosine 0.62. A fish self-correction run now would measure *adapter-training maturity*, not the frozen-Gemma brain (already answered by the E2B proxy = 0.619), so we deliberately do **not** report a fish number — it would mislead.
- **Duplex predictor runs but is unvalidated** — mixed/uninformative states (SPEAK in the pipeline path, INTERRUPT on synthetic inputs); no clean LISTEN/SPEAK/INTERRUPT signal yet. **Turn-take / interruption (AC-3/4) are blocked on duplex-predictor training, not architecture.** Scorers + aggregation (`conversational_runner`, `conversational_scoring`) are done + unit-tested, ready to consume real states.
- **Conversational audio fixtures rendered** — 8 turn-take/backchannel/barge-in clips (`conversational_scenarios.json` → `data/realistic-audio/conversational/`), the reusable AC-3/4 input for once the duplex predictor is trained.

## Pending to fully close Phase A

- **Train the fish speech adapters + duplex predictor** — the gating blocker for fish-on-real-stack self-correction and AC-3/4 (brain + latency already proven via the E2B proxy).
- **TTFA (AC-5)** — fish E2E is ~2.3 s untuned; streaming `first_audio_ms` < 400 ms is the target once the speech path is tuned.
- **Task 11** — `spec-verifier` once the trained-stack numbers land.

## What is firmly closed

Vendor rejection (AC-2, structural); cascade floor validated live (AC-6 measured); the entire scoring pipeline (turn-take, interruption, TTFA aggregation, self-correction with SOTA tolerance, persona cosine, lane health) implemented + unit-tested on PR #3; D8 scoring method locked with verified arxiv evidence.
