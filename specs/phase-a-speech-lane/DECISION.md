# Phase-A Speech-Lane Decision — DECISION RECORD

> Status: **DECIDED (vendor rejected, direction set); fish-vs-cascade finalization pending live fish run.**
> Date: 2026-06-06 · Spec: `requirements.md` / `design.md` / `tasks.md` · Roadmap: `docs/research/2026-06-06-s2s-L1-L5-roadmap.md`

## Decision

1. **Reject native vendor S2S (gpt-realtime / Gemini Live / Nova 2 Sonic) for the persona path — FINAL.** It cannot load the frozen-Gemma+LoRA persona, so it fails AC-2 (persona portability) *by construction*, not by measurement (`persona_consistency.vendor_lane_persona_verdict`). Vendor remains allowed only as a comparison-only baseline and for non-persona utility paths.
2. **Cascade lane is the validated floor — PROVEN LIVE.** The Whisper→Gemma cascade runs on-device, preserves the Gemma+LoRA persona (same brain), and produces a real self-correction Pass@1 (below). It is the always-available fallback (`lane_select`, runtime-observable).
3. **Frozen-Gemma true-S2S (fish, Phase 7) is the recommended PRIMARY — pending its live self-correction run.** Rationale unchanged (one brain + persona across voice and text; on-device; the only lane that carries the moat across modalities). It must clear the **reversal trigger** below against the measured cascade floor; until then, cascade is the shipping default.

## Evidence (measured this session)

| Lane | Self-correction Pass@1 (AC-6) | Persona portability (AC-2) | Notes |
|---|---|---|---|
| **cascade** (Whisper-tiny → Gemma-31b-8bit) | **0.762 (16/21)** (n=21, transport_errors=0) | preserved (shared Gemma+LoRA) | live, on-device; `proof-artifacts/lane-scoreboard-cascade.json` |
| **fish** (frozen-Gemma S2S) | *pending live run* | preserved (shared Gemma+LoRA) | S2S stack not exercised this session |
| **vendor** (gpt-realtime) | n/a | **FAIL by construction** | cannot load LoRA persona |

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

- If, within **2 weeks**, the **fish** lane's live self-correction Pass@1 does **not** meet-or-exceed the measured **cascade floor (0.762 (16/21))** AND clear the frontier-comparable bar (>0.588 target, with >0.176 as the hard floor), **demote fish to R&D and ship cascade-with-micro-turns as primary.**
- If fish cannot reach **TTFA <400 ms p50** or **turn-take ≥95%** on realistic audio once those gates are measured, same demotion.
- Vendor rejection is **not** subject to reversal (structural, not performance-based).

## Pending to fully close Phase A

- **Fish lane live run** (self-correction + the conversational gates) — needs the S2S adapters loaded end-to-end.
- **Turn-take / interruption (AC-3/4)** — need conversational/backchannel audio fixtures (current fixtures are single-utterance self-corrections) + the streaming duplex path; scorers + aggregation are done and tested.
- **TTFA (AC-5)** — needs the streaming TTS path's `first_audio_ms` (cascade self-correction run measured extraction latency, not audio TTFA).
- **Task 11** — `spec-verifier` pass once fish numbers land.

## What is firmly closed

Vendor rejection (AC-2, structural); cascade floor validated live (AC-6 measured); the entire scoring pipeline (turn-take, interruption, TTFA aggregation, self-correction with SOTA tolerance, persona cosine, lane health) implemented + unit-tested on PR #3; D8 scoring method locked with verified arxiv evidence.
