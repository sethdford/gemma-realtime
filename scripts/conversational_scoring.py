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


import re as _re

NUMERIC_TOL = 0.05  # ±5% — matches Full-Duplex-Bench-v3 GPT-4o argument-accuracy tolerance


def _as_number(v) -> float | None:
    """Parse a number from int/float or a money/grouped string ($1,500 -> 1500). None if not numeric."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = _re.sub(r"[,$\s]", "", v)
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _norm_str(v) -> str:
    """Case/whitespace/punctuation-insensitive string normal form."""
    return _re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()


def match_value(expected, actual, *, numeric_tol: float = NUMERIC_TOL, aliases: dict | None = None) -> bool:
    """SOTA argument-accuracy match (FDB-v3 semantics): lenient on FORMAT, strict on VALUE.

    - numbers match within ±numeric_tol ("1500" == 1500 == "$1,500"; 1450 within 5%);
    - strings match case/punctuation-insensitively, with an optional alias map
      (e.g. {"vegas": "las vegas"}); exact-equality is the floor.
    """
    if expected == actual:
        return True
    en, an = _as_number(expected), _as_number(actual)
    if en is not None and an is not None:
        if en == 0.0:
            return abs(an) < 1e-9
        return abs(an - en) / abs(en) <= numeric_tol
    ne, na = _norm_str(expected), _norm_str(actual)
    if ne == na:
        return True
    if aliases:
        al = {_norm_str(k): _norm_str(v) for k, v in aliases.items()}
        return al.get(na) == ne or al.get(ne) == na
    return False


def score_self_correction(final_intent: dict, scenario: dict, *, semantic: bool = True,
                          aliases: dict | None = None) -> bool:
    """True iff ``final_intent`` matches the scenario's ``expected_final`` on every key.

    Default (``semantic=True``) uses SOTA argument-accuracy tolerance (see match_value):
    lenient on format, strict on the corrected VALUE — so an agent that correctly
    self-corrects to 1500 but emits "$1,500" is NOT falsely failed. ``semantic=False``
    falls back to exact equality. Only ``expected_final`` keys are graded; extra
    fields are ignored; a missing/wrong key fails.
    """
    expected = scenario.get("expected_final") or {}
    if not expected:
        raise ValueError(f"scenario {scenario.get('id')!r} has no expected_final to score against")
    if not semantic:
        return all(final_intent.get(k) == v for k, v in expected.items())
    return all(
        match_value(v, final_intent.get(k), numeric_tol=_numeric_tol_for_key(k), aliases=aliases)
        for k, v in expected.items()
    )


# ±5% tolerance is a MAGNITUDE concept — it must NOT apply to identifiers, or
# order_id 456 would "match" 459. Identifier keys get exact-numeric (format-
# insensitive, zero tolerance): "456" == 456 but != 459.
IDENTIFIER_TOKENS = {
    "id", "code", "number", "no", "phone", "zip", "account", "acct",
    "order", "ssn", "confirmation", "pin", "sku", "tracking",
}


def _numeric_tol_for_key(key: str) -> float:
    tokens = _re.split(r"[^a-z0-9]+", str(key).lower())
    return 0.0 if any(t in IDENTIFIER_TOKENS for t in tokens) else NUMERIC_TOL


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
