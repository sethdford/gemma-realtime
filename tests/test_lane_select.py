"""Phase-A spec (specs/phase-a-speech-lane) Task 8 / AC-7: the active speech lane
must be resolvable from the existing --tts flag and observable via a health
signal, so cascade fallback is never silent.

Pure-local: no model, no audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lane_select  # noqa: E402
from lane_select import resolve_lane, lane_health, LANES  # noqa: E402


def test_existing_tts_flags_map_to_three_lanes():
    # The current --tts choices (realtime-ws.py:1019) must each resolve to a lane.
    assert resolve_lane("fish") == "fish"
    for cascade_alias in ("kokoro", "kokoro-onnx", "voxtral", "f5", "native"):
        assert resolve_lane(cascade_alias) == "cascade", cascade_alias
    assert resolve_lane("gpt-realtime") == "vendor"


def test_precedence_explicit_over_env_over_default():
    assert resolve_lane("fish", env="voxtral", default="cascade") == "fish"
    assert resolve_lane(None, env="voxtral", default="fish") == "cascade"  # env wins over default
    assert resolve_lane(None, env=None, default="fish") == "fish"


def test_vendor_never_auto_selected_as_primary():
    # vendor is comparison-only (AC-8 tension); it must never be the default/env primary.
    assert resolve_lane(None, env=None, default="vendor") != "vendor"
    assert LANES["vendor"]["comparison_only"] is True
    assert LANES["fish"]["on_device"] is True and LANES["cascade"]["on_device"] is True
    assert LANES["vendor"]["on_device"] is False


def test_health_signal_reports_active_lane_no_fallback():
    h = lane_health(requested="fish", active="fish")
    assert h["requested_tts"] == "fish"
    assert h["active_lane"] == "fish"
    assert h["on_device"] is True
    assert h["fallback_occurred"] is False
    assert h["fallback_reason"] is None


def test_health_signal_makes_fallback_observable():
    # fish requested but cascade active => fallback must be flagged, not silent.
    h = lane_health(requested="fish", active="cascade", fallback_reason="Fish DAC load failed")
    assert h["active_lane"] == "cascade"
    assert h["fallback_occurred"] is True
    assert h["fallback_reason"] == "Fish DAC load failed"
    assert h["is_floor"] is True


def test_unknown_flag_falls_back_to_cascade_floor():
    assert resolve_lane("some-unknown-backend") == "cascade"
