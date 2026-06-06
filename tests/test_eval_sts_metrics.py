"""Phase-A spec (specs/phase-a-speech-lane): the lane scoreboard must expose the
four conversational gate metrics — turn-take rate (AC-3), interruption avoidance
(AC-4), TTFA p50/p95 (AC-5), and self-correction Pass@1 (AC-6) — in
STSMetrics.summary() and reflect them in the scorecard.

These pin the metric *schema* the Phase-A decision reads. They do not require a
running model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import eval_sts  # noqa: E402
from eval_sts import STSMetrics, _compute_scorecard  # noqa: E402


def _populated() -> STSMetrics:
    m = STSMetrics()
    # turn-take: 19/20 turns yielded correctly -> 0.95 (AC-3 gate)
    m.turn_take = [1.0] * 19 + [0.0]
    # interruption avoidance: held SPEAK through non-terminal barge-in 3/20 -> 0.15 (> 13.5% AC-4)
    m.interruption_avoidance = [1.0] * 3 + [0.0] * 17
    # TTFA samples (ms) -> p50 should be < 400 (AC-5)
    m.ttfa_ms = [120.0, 200.0, 350.0, 380.0, 410.0]
    # self-correction: 13/21 passed -> ~0.619 (> 0.60 AC-6)
    m.self_correction = [1.0] * 13 + [0.0] * 8
    return m


def test_summary_exposes_conversational_gate_metrics():
    s = _populated().summary()
    # AC-3 turn-take rate
    assert s["turn_take_rate"] == 0.95
    # AC-4 interruption avoidance
    assert s["interruption_avoidance"] == 0.15
    # AC-5 TTFA percentiles
    assert s["ttfa_p50_ms"] == 350.0
    assert s["ttfa_p95_ms"] is not None and s["ttfa_p95_ms"] >= s["ttfa_p50_ms"]
    # AC-6 self-correction Pass@1
    assert abs(s["self_correction_pass1"] - (13 / 21)) < 1e-3


def test_empty_metrics_are_none_not_crash():
    s = STSMetrics().summary()
    for k in ("turn_take_rate", "interruption_avoidance", "ttfa_p50_ms", "ttfa_p95_ms", "self_correction_pass1"):
        assert k in s
        assert s[k] is None


def test_scorecard_reports_conversational_gates_against_frontier_thresholds():
    s = _populated().summary()
    card = _compute_scorecard(s)
    conv = card.get("conversational")
    assert conv is not None, "scorecard must include a conversational gate block"
    # Each gate carries the measured value, the threshold, and a pass/fail bit.
    assert conv["turn_take_rate"]["value"] == 0.95
    assert conv["turn_take_rate"]["pass"] is True            # >= 0.95 (AC-3)
    assert conv["interruption_avoidance"]["pass"] is True    # > 0.135 (AC-4)
    assert conv["ttfa_p50_ms"]["pass"] is True               # < 400 (AC-5)
    assert conv["self_correction_pass1"]["pass"] is True     # > 0.60 (AC-6)


def test_scorecard_marks_failing_gates():
    m = STSMetrics()
    m.turn_take = [1.0] * 7 + [0.0] * 3          # 0.70 < 0.95 -> fail AC-3
    m.interruption_avoidance = [0.0] * 10        # 0.0 -> fail AC-4
    m.ttfa_ms = [900.0, 1000.0]                  # p50 950 -> fail AC-5
    m.self_correction = [1.0] * 5 + [0.0] * 16   # ~0.238 -> fail AC-6
    conv = _compute_scorecard(m.summary())["conversational"]
    assert conv["turn_take_rate"]["pass"] is False
    assert conv["interruption_avoidance"]["pass"] is False
    assert conv["ttfa_p50_ms"]["pass"] is False
    assert conv["self_correction_pass1"]["pass"] is False
