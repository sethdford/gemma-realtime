"""Phase-A spec (specs/phase-a-speech-lane) Tasks 3/4/7: the conversational runner
aggregates per-turn results (duplex states, TTFA, self-correction) into the same
STSMetrics + scorecard the lane decision reads. Pure orchestration tested with
synthetic turn records; the audio->model->states call is the integration hook.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from conversational_runner import TurnRecord, aggregate_conversational, conversational_scorecard  # noqa: E402
from conversational_scoring import LISTEN, SPEAK, INTERRUPT  # noqa: E402

SCN = {
    "id": "x", "domain": "d", "correction_type": "t", "utterance": "u",
    "original_intent": {"dest": "Boston"}, "corrected_intent": {"dest": "NYC"},
    "expected_final": {"dest": "NYC"},
}


def _passing_records():
    recs = []
    # 20 turns, 19 correct turn-take -> 0.95 (AC-3 pass)
    for i in range(20):
        recs.append(TurnRecord(
            states=[LISTEN, SPEAK] if i != 0 else [LISTEN, LISTEN],  # first turn wrongly silent
            should_respond=True,
            backchannel_windows=[],
            ttfa_ms=300.0,                       # < 400 p50 (AC-5)
            self_correction=({"dest": "NYC"}, SCN),  # corrected -> pass
        ))
    # interruption: 8 backchannels, 2 wrongly yielded -> 0.75 (> 0.135 AC-4)
    recs[0].backchannel_windows = [[SPEAK, SPEAK]] * 6 + [[SPEAK, INTERRUPT]] * 2
    return recs


def test_aggregate_populates_all_conversational_lists():
    m = aggregate_conversational(_passing_records())
    s = m.summary()
    assert s["turn_take_rate"] == 0.95
    assert s["ttfa_p50_ms"] == 300.0
    assert s["self_correction_pass1"] == 1.0
    assert abs(s["interruption_avoidance"] - 0.75) < 1e-9


def test_scorecard_gates_pass_for_good_run():
    card = conversational_scorecard(_passing_records())["scorecard"]["conversational"]
    assert card["turn_take_rate"]["pass"] is True
    assert card["interruption_avoidance"]["pass"] is True
    assert card["ttfa_p50_ms"]["pass"] is True
    assert card["self_correction_pass1"]["pass"] is True


def test_scorecard_gates_fail_for_bad_run():
    bad = [TurnRecord(states=[LISTEN], should_respond=True, backchannel_windows=[[SPEAK, INTERRUPT]],
                      ttfa_ms=900.0, self_correction=({"dest": "Boston"}, SCN)) for _ in range(5)]
    card = conversational_scorecard(bad)["scorecard"]["conversational"]
    assert card["turn_take_rate"]["pass"] is False          # never spoke
    assert card["interruption_avoidance"]["pass"] is False  # always yielded
    assert card["ttfa_p50_ms"]["pass"] is False             # 900 > 400
    assert card["self_correction_pass1"]["pass"] is False   # locked original


def test_empty_run_leaves_gates_unmeasured():
    card = conversational_scorecard([])["scorecard"]["conversational"]
    for gate in card.values():
        assert gate["pass"] is None
