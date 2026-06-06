"""Phase-A spec (specs/phase-a-speech-lane) Task 6 / AC-2: persona-portability
default — a heuristic trait-vector cosine. The same Gemma+LoRA persona rendered
through the text path vs a speech lane should yield a high cosine; a different
persona should not. Word-boundary lexical scoring (no model needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from persona_consistency import (  # noqa: E402
    trait_vector, cosine, persona_consistency, DEFAULT_THRESHOLD, TRAITS,
)

FORMAL_A = "Thank you, I sincerely appreciate it. Regarding your request, I shall follow up. Furthermore, regards."
FORMAL_B = "I appreciate your note. However, regarding the matter, I shall respond. Sincerely, thank you."
CASUAL = "yeah lol gonna grab food, hey wanna come? kinda hungry haha, whatever ok"


def test_identical_text_cosine_one_and_consistent():
    r = persona_consistency(FORMAL_A, FORMAL_A)
    assert r["cosine"] == 1.0
    assert r["consistent"] is True


def test_same_persona_high_cosine():
    r = persona_consistency(FORMAL_A, FORMAL_B)
    assert r["cosine"] >= DEFAULT_THRESHOLD, r["cosine"]
    assert r["consistent"] is True


def test_different_persona_not_consistent():
    r = persona_consistency(FORMAL_A, CASUAL)
    assert r["cosine"] < DEFAULT_THRESHOLD, r["cosine"]
    assert r["consistent"] is False
    # And it should be clearly lower than the same-persona pair.
    assert r["cosine"] < persona_consistency(FORMAL_A, FORMAL_B)["cosine"]


def test_vector_dimensionality_matches_trait_set():
    assert len(trait_vector(FORMAL_A)) == len(TRAITS)


def test_word_boundary_no_substring_false_positive():
    # 'informally' must not trigger any formal marker via substring; whole-word only.
    v_plain = trait_vector("the the the the")          # no markers
    v_substr = trait_vector("informally informally")    # contains 'formal' as substring
    assert v_plain == v_substr == [0.0] * len(TRAITS)


def test_empty_and_zero_vectors():
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 1.0          # both no-signal => identical
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0          # one no-signal => orthogonal
    r = persona_consistency("the the the", "the the the")
    assert r["cosine"] == 1.0 and r["consistent"] is True


# ── Task-6 live probe orchestration (AC-2) ────────────────────────────────────
from persona_consistency import probe_persona_portability, vendor_lane_persona_verdict  # noqa: E402


def test_probe_on_device_lanes_preserve_persona():
    out = probe_persona_portability(FORMAL_A, {"fish": FORMAL_B, "cascade": FORMAL_A})
    assert out["lanes"]["fish"]["consistent"] is True
    assert out["lanes"]["cascade"]["consistent"] is True
    assert set(out["portable_lanes"]) == {"fish", "cascade"}


def test_probe_casual_lane_render_fails():
    out = probe_persona_portability(FORMAL_A, {"fish": CASUAL})
    assert out["lanes"]["fish"]["consistent"] is False
    assert "fish" not in out["portable_lanes"]


def test_probe_vendor_fails_by_construction():
    # vendor is disqualified regardless of what it renders (can't carry the LoRA).
    out = probe_persona_portability(FORMAL_A, {"fish": FORMAL_B, "vendor": FORMAL_A})
    v = out["lanes"]["vendor"]
    assert v["consistent"] is False and v["by_construction"] is True
    assert v["cosine"] is None and "LoRA" in v["reason"]
    assert out["portable_lanes"] == ["fish"]


def test_vendor_verdict_standalone():
    v = vendor_lane_persona_verdict()
    assert v["consistent"] is False and v["by_construction"] is True
