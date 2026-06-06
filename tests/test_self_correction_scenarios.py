"""Phase-A spec (specs/phase-a-speech-lane) Task 7 / AC-6: the self-correction
scenario set must load, be the expected size, and every scenario must be a real
mid-utterance intent change with a well-formed expected_final.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from self_correction import load_self_correction_scenarios  # noqa: E402


def test_loads_21_scenarios():
    s = load_self_correction_scenarios()
    assert len(s) == 21, f"expected 21 self-correction scenarios (FDB-v3 size), got {len(s)}"


def test_every_scenario_is_a_real_correction_with_expected_final():
    for sc in load_self_correction_scenarios():
        assert sc["original_intent"] != sc["corrected_intent"], sc["id"]
        # expected_final must point at the corrected value, never the original.
        for k, v in sc["expected_final"].items():
            assert sc["corrected_intent"].get(k) == v, f"{sc['id']}: expected_final[{k}] != corrected_intent"
            if k in sc["original_intent"]:
                assert sc["original_intent"][k] != v, f"{sc['id']}: expected_final[{k}] matches the ORIGINAL (no correction)"


def test_ids_unique_and_domains_varied():
    s = load_self_correction_scenarios()
    ids = [x["id"] for x in s]
    assert len(set(ids)) == len(ids), "duplicate scenario ids"
    domains = {x["domain"] for x in s}
    assert len(domains) >= 5, f"want varied domains, got {domains}"
