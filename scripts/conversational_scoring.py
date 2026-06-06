#!/usr/bin/env python3
"""Pure scoring core for Phase-A conversational gates (specs/phase-a-speech-lane).

Currently: self-correction Pass@1 (AC-6, Task 7). The live runner extracts the
agent's final intent per scenario and feeds it here; keeping the scoring logic
model-free makes it unit-testable and keeps the metric definition in one place.

A scenario passes iff the agent's final intent matches the *corrected* intent on
every ``expected_final`` key — i.e. it did NOT lock in the pre-correction value,
which is the frontier failure mode (GPT-Realtime fails 40%+; see roadmap Part 2).

Turn-take and interruption-avoidance scoring live with the live runner (Task 3),
where the duplex state-sequence representation is concrete — not scaffolded here
against an unknown shape.
"""
from __future__ import annotations

from typing import Iterable


def score_self_correction(final_intent: dict, scenario: dict) -> bool:
    """True iff ``final_intent`` matches the scenario's ``expected_final`` on every key.

    Only ``expected_final`` keys are graded; extra fields on the agent's intent
    are ignored. A missing/wrong key fails the scenario.
    """
    expected = scenario.get("expected_final") or {}
    if not expected:
        raise ValueError(f"scenario {scenario.get('id')!r} has no expected_final to score against")
    return all(final_intent.get(k) == v for k, v in expected.items())


def self_correction_pass1(pairs: Iterable[tuple[dict, dict]]) -> float | None:
    """Pass@1 over (final_intent, scenario) pairs. None if there are no pairs.

    None (not 0.0) signals "unmeasured" so the Phase-A decision step does not read
    an empty run as a failing gate (mirrors STSMetrics tri-state).
    """
    pairs = list(pairs)
    if not pairs:
        return None
    passed = sum(1 for final_intent, scenario in pairs if score_self_correction(final_intent, scenario))
    return passed / len(pairs)


# ── Duplex states (must match speech_decoder.DuplexStatePredictor) ─────────────
LISTEN, SPEAK, INTERRUPT = 0, 1, 2


def score_turn_take(states: Iterable[int], should_respond: bool) -> bool:
    """Did the agent take/yield the turn correctly (AC-3)?

    ``states`` is the duplex-predictor state sequence for one turn. If the agent
    *should* respond, passing means a SPEAK state appears; if it should yield
    (e.g. a backchannel turn), passing means it never took the floor (no SPEAK).
    """
    spoke = SPEAK in list(states)
    return spoke if should_respond else (not spoke)


def turn_take_rate(turns: Iterable[tuple[Iterable[int], bool]]) -> float | None:
    """Mean turn-take correctness over (states, should_respond) turns. None if empty."""
    turns = list(turns)
    if not turns:
        return None
    return sum(1 for states, should in turns if score_turn_take(states, should)) / len(turns)


def score_interruption_avoidance(states_during_bargein: Iterable[int]) -> bool:
    """For a NON-terminal barge-in (backchannel / noise that should NOT stop the
    agent): True iff the agent kept speaking — no INTERRUPT/LISTEN (AC-4).

    Feed only non-terminal barge-in windows; terminal (genuine) interruptions are
    a separate responsiveness metric, out of scope for the avoidance gate.
    """
    return not any(s in (INTERRUPT, LISTEN) for s in states_during_bargein)


def interruption_avoidance_rate(events: Iterable[Iterable[int]]) -> float | None:
    """Fraction of non-terminal barge-ins the agent correctly held through. None if empty."""
    events = list(events)
    if not events:
        return None
    return sum(1 for w in events if score_interruption_avoidance(w)) / len(events)
