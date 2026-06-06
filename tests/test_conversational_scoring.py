"""Phase-A spec (specs/phase-a-speech-lane) Task 7 / AC-6: pure scoring core for
self-correction Pass@1. The live runner extracts the agent's final intent per
scenario and calls these; scoring logic is testable without a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from conversational_scoring import score_self_correction, self_correction_pass1  # noqa: E402
from self_correction import load_self_correction_scenarios  # noqa: E402


def _scn(**kw):
    base = {
        "id": "x", "domain": "d", "correction_type": "t", "utterance": "u",
        "original_intent": {"dest": "Boston"},
        "corrected_intent": {"dest": "New York"},
        "expected_final": {"dest": "New York"},
    }
    base.update(kw)
    return base


def test_final_matching_corrected_passes():
    assert score_self_correction({"dest": "New York"}, _scn()) is True


def test_final_matching_original_fails():
    # Locking in the pre-correction value is the frontier failure mode (40%+).
    assert score_self_correction({"dest": "Boston"}, _scn()) is False


def test_missing_or_wrong_key_fails():
    assert score_self_correction({}, _scn()) is False
    assert score_self_correction({"dest": "Chicago"}, _scn()) is False


def test_extra_keys_ignored_when_expected_satisfied():
    # Only expected_final keys are graded; extra agent fields are allowed.
    assert score_self_correction({"dest": "New York", "cabin": "economy"}, _scn()) is True


def test_multikey_expected_requires_all():
    scn = _scn(
        corrected_intent={"action": "move_event", "time": "16:00"},
        expected_final={"action": "move_event", "time": "16:00"},
    )
    assert score_self_correction({"action": "move_event", "time": "16:00"}, scn) is True
    assert score_self_correction({"action": "move_event", "time": "15:00"}, scn) is False


def test_pass1_aggregates_mean():
    scn = _scn()
    finals = [{"dest": "New York"}, {"dest": "New York"}, {"dest": "Boston"}]
    pairs = [(f, scn) for f in finals]
    assert abs(self_correction_pass1(pairs) - (2 / 3)) < 1e-9


def test_pass1_empty_is_none():
    assert self_correction_pass1([]) is None


def test_scores_against_real_fixture_perfect_and_zero():
    scenarios = load_self_correction_scenarios()
    # An agent that always lands the corrected intent scores 1.0 on the real set.
    perfect = [(s["corrected_intent"], s) for s in scenarios]
    assert self_correction_pass1(perfect) == 1.0
    # An agent that locks in the original intent scores 0.0 (the failure mode).
    worst = [(s["original_intent"], s) for s in scenarios]
    assert self_correction_pass1(worst) == 0.0


# ── Turn-take (AC-3) ──────────────────────────────────────────────────────────
from conversational_scoring import (  # noqa: E402
    LISTEN, SPEAK, INTERRUPT,
    score_turn_take, turn_take_rate,
    score_interruption_avoidance, interruption_avoidance_rate,
)


def test_turn_take_should_respond():
    # Agent should respond and did (a SPEAK appears) -> pass.
    assert score_turn_take([LISTEN, LISTEN, SPEAK, SPEAK], should_respond=True) is True
    # Should respond but stayed silent -> fail.
    assert score_turn_take([LISTEN, LISTEN, LISTEN], should_respond=True) is False


def test_turn_take_should_yield():
    # Backchannel turn: agent should NOT take the floor; staying in LISTEN passes.
    assert score_turn_take([LISTEN, LISTEN], should_respond=False) is True
    # Barged in when it should have stayed quiet -> fail.
    assert score_turn_take([LISTEN, SPEAK], should_respond=False) is False


def test_turn_take_rate_aggregates():
    turns = [([SPEAK], True), ([LISTEN], True), ([LISTEN], False), ([SPEAK], False)]
    # 1st pass, 2nd fail (should respond, didn't), 3rd pass, 4th fail -> 2/4
    assert turn_take_rate(turns) == 0.5
    assert turn_take_rate([]) is None


# ── Interruption avoidance (AC-4) ─────────────────────────────────────────────
def test_interruption_avoidance_holds_through_backchannel():
    # Non-terminal barge-in (backchannel): correct behavior = keep speaking.
    assert score_interruption_avoidance([SPEAK, SPEAK, SPEAK]) is True
    # Wrongly yielded to a backchannel -> not avoided.
    assert score_interruption_avoidance([SPEAK, INTERRUPT, LISTEN]) is False


def test_interruption_avoidance_rate():
    events = [[SPEAK, SPEAK], [SPEAK, INTERRUPT], [SPEAK, SPEAK]]  # 2/3 held
    assert abs(interruption_avoidance_rate(events) - (2 / 3)) < 1e-9
    assert interruption_avoidance_rate([]) is None
