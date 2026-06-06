# Phase-A Speech-Lane Decision — Requirements

> Type: Architecture Decision Spec (ADR-style, validated by measurement)
> Companion: `docs/research/2026-06-06-s2s-L1-L5-roadmap.md`
> Decision under test: **Productionize Phase 7 frozen-Gemma true-S2S as the primary lane for the L4–L5 humanness moat, with Phase 1–2 VAD-free cascade as the always-available floor; reject native vendor S2S for the core persona path.**

## Context

gemma-realtime has built three speech lanes (cascade P1–2, frozen-Gemma adapters P4–6, true-S2S P7). The frontier (gpt-realtime, Gemini Live, Nova 2 Sonic, Cartesia, Hume EVI 3) is a strong **L3** and a weak **L4**: even the leader (GPT-Realtime) fails >40% of self-corrections and has 13.5% interruption avoidance (Full-Duplex-Bench-v3). L1–L3 are commoditized; the moat is **L4→L5** (pragmatic repair, persona stability, structural prosody, indistinguishability). The strategic question Phase A must settle: **which lane do we bet the moat on, and what is the floor?**

## User stories

- As the **gemma-realtime maintainer**, I want a measured, reversible decision on the primary speech lane, so that we invest the L4–L5 effort on an architecture that can actually carry it.
- As the **h-uman persona owner**, I want the speech path to preserve the *same Gemma brain + LoRA persona* used for text, so that voice and text are the same identity (the unifying moat) and persona work compounds across modalities.
- As an **on-device / privacy-sensitive user**, I want inference to run locally on Apple Silicon with no per-minute cloud cost or audio leaving the device.
- As an **operator**, I want an always-available fallback lane so a regression in the experimental S2S path never takes the product to zero.
- As a **reviewer**, I want each lane scored against the same gates on the same realistic-audio harness, so the decision rests on evidence not advocacy.

## Acceptance criteria

> Each AC is a measurable gate. "PASS" = demonstrated by the named harness/script on the named hardware (Apple Silicon, MLX), on a realistic-audio test set (noise, accents, disfluency), reported with ≥100 samples per branch.

- [ ] **AC-1 (Lane scoreboard exists):** A reproducible benchmark scores Phase 7 (frozen-Gemma S2S), Phase 1–2 (cascade), and one native-vendor S2S baseline on identical inputs via a single command, emitting a JSON scoreboard.
- [ ] **AC-2 (Persona portability — the decisive criterion):** A test demonstrates the *same* Gemma + LoRA persona produces measurably consistent persona-trait behavior in BOTH the text path and the chosen speech lane (trait-vector cosine ≥ agreed threshold), AND demonstrates this is impossible/absent in the native-vendor lane. This AC is what disqualifies vendor S2S.
- [ ] **AC-3 (Turn-take rate ≥95%):** The chosen primary lane achieves turn-take rate ≥95% on the local Full-Duplex-Bench-style harness (frontier: Cascaded 100%, Gemini 3.1 78%).
- [ ] **AC-4 (Interruption avoidance > 13.5%):** The chosen primary lane beats the frontier leader's interruption-avoidance (GPT-Realtime 13.5%) on the same harness.
- [ ] **AC-5 (TTFA < 400 ms p50 on realistic audio):** Time-to-first-audio p50 < 400 ms measured end-to-end on Apple Silicon with realistic (not clean-studio) input.
- [ ] **AC-6 (Self-correction Pass@1 > 0.60):** On mid-utterance intent-change scenarios, the primary lane beats GPT-Realtime's 0.588 — i.e. state-rollback works better than the frontier. (This is the L4 capability the decision is meant to enable; may be deferred to Phase B but the lane must not architecturally preclude it.)
- [ ] **AC-7 (Floor lane proven):** Phase 1–2 cascade passes a smoke + latency gate (<1.2 s E2E) and can be selected at runtime as a fallback without code changes (config/flag only).
- [ ] **AC-8 (On-device, zero cloud egress):** The primary lane runs fully local on Apple Silicon — verified no audio bytes leave the device and no per-minute external API is on the hot path.
- [ ] **AC-9 (Decision record is reversible):** The decision doc states the exact metric thresholds that would trigger a re-evaluation (e.g. if Phase 7 cannot beat AC-3/4/6 within N weeks, fall back to cascade-with-micro-turns as primary).

## Non-goals

- Not building the full L4 repair engine here — Phase A only proves the *lane can carry it* (AC-6 may be a feasibility probe, full delivery is Phase B).
- Not beating Cartesia's 40 ms TTS latency — latency parity is AC-5 (<400 ms), not latency leadership.
- Not multilingual / polyglot voices (Nova 2 Sonic territory) — English-first.
- Not choosing the neural codec (SNAC vs Fish DAC) — that's a sub-decision inside the Phase 7 lane, tracked separately.
- Not committing to a specific cloud vendor for *non-persona* utility paths (e.g. a vendor ASR could still be a swappable component) — the decision is about the *core persona conversational path*.

## Constraints

- **Hardware:** Apple Silicon; MLX runtime; IOSurface/ANE acceleration available (Phase 6).
- **Brain invariant:** Gemma stays **frozen + LoRA**; speech modules are additive (Freeze-Omni / MOSS-Speech layer-split). No catastrophic forgetting of the text persona.
- **Eval rigor:** ≥100 samples per branch (per τ-voice guidance); realistic-audio set (noise/accents/disfluency), not clean studio.
- **Harness reuse:** extend existing `scripts/eval_sts.py` (WER, MOS proxy, speaker sim, RTF) rather than a new framework; add turn-taking/interruption/self-correction metrics.
- **Fallback guarantee:** cascade lane must remain runnable at all times (operational safety).
- **Reversibility:** decision must name the measurable trigger to revisit.

## Open questions for sign-off

1. **Persona-consistency threshold (AC-2):** what trait-vector similarity counts as "same persona across modalities"? (Proposal: reuse h-uman's existing persona-trait scoring if present; else cosine ≥ 0.85 on a defined trait vector.)
2. **Realistic-audio test set:** do we have one, or does Phase A include building it (LibriSpeech + noise augmentation + accent set)?
3. **Native-vendor baseline for AC-1/AC-2:** which one — gpt-realtime (best agentic) or Nova 2 Sonic (cheapest)? (Proposal: gpt-realtime, as the strongest L3 to contrast against.)
4. **AC-6 scope:** feasibility probe in Phase A vs full delivery in Phase B? (Proposal: probe only — prove the lane doesn't preclude state rollback.)
