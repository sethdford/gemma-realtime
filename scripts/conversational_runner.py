#!/usr/bin/env python3
"""Streaming conversational runner for the Phase-A conversational gates
(specs/phase-a-speech-lane, Tasks 3/4/7).

`eval_sts.py` is a batch *quality* harness; the conversational gates (turn-take,
interruption avoidance, TTFA, self-correction) need a *streaming* run. This module
provides:

  • the model-FREE aggregation core — per-turn results -> STSMetrics -> scorecard,
    reusing the committed scorers and the eval_sts metric schema (tested);
  • `run_lane_conversational(...)` — the integration hook that drives a lane's live
    inference (audio -> duplex states via the duplex-predictor + first_audio capture).
    That part needs a model + audio fixtures and is the remaining boundary.

Keeping aggregation pure means the lane decision's conversational scoreboard is
fully unit-tested before any GPU run; only the audio->states glue is deferred.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_sts import STSMetrics, _compute_scorecard
from conversational_scoring import (
    score_turn_take, score_interruption_avoidance, score_self_correction,
)


@dataclass
class TurnRecord:
    """One conversational turn's observations from a lane run.

    states              : duplex LISTEN/SPEAK/INTERRUPT sequence for the turn (AC-3)
    should_respond      : ground truth — should the agent take the floor this turn?
    backchannel_windows : state windows during NON-terminal barge-ins (AC-4); each
                          is one backchannel event the agent should speak through
    ttfa_ms             : measured time-to-first-audio for the turn, ms (AC-5)
    self_correction     : (final_intent, scenario) for a self-correction turn (AC-6)
    """
    states: list[int]
    should_respond: bool = True
    backchannel_windows: list[list[int]] = field(default_factory=list)
    ttfa_ms: float | None = None
    self_correction: tuple[dict, dict] | None = None


def aggregate_conversational(records, into: STSMetrics | None = None) -> STSMetrics:
    """Populate STSMetrics' conversational lists from turn records via the scorers."""
    m = into if into is not None else STSMetrics()
    for r in records:
        m.turn_take.append(1.0 if score_turn_take(r.states, r.should_respond) else 0.0)
        for window in r.backchannel_windows:
            m.interruption_avoidance.append(1.0 if score_interruption_avoidance(window) else 0.0)
        if r.ttfa_ms is not None:
            m.ttfa_ms.append(float(r.ttfa_ms))
        if r.self_correction is not None:
            final_intent, scenario = r.self_correction
            m.self_correction.append(1.0 if score_self_correction(final_intent, scenario) else 0.0)
    return m


def conversational_scorecard(records) -> dict:
    """Aggregate + score: {summary, scorecard} (same shape eval_sts emits)."""
    m = aggregate_conversational(records)
    summary = m.summary()
    return {"summary": summary, "scorecard": _compute_scorecard(summary)}


def run_lane_conversational(lane: str, scenarios, *, model_ctx=None):  # pragma: no cover
    """INTEGRATION HOOK (deferred): drive a lane's live inference to produce
    TurnRecords — audio -> duplex-predictor states (speech_decoder.DuplexStatePredictor
    / fish_sts.predict_state) + first_audio_ms capture, replay self-correction
    scenarios. Needs a model + audio fixtures; not unit-tested here.

    Returns a list[TurnRecord] to feed conversational_scorecard().
    """
    raise NotImplementedError(
        "run_lane_conversational requires a live fish/cascade model + audio fixtures; "
        "the aggregation core (aggregate_conversational/conversational_scorecard) is ready."
    )
