#!/usr/bin/env python3
"""Speech-lane resolution + health signal for the Phase-A lane decision.

specs/phase-a-speech-lane (Task 8 / AC-7). Maps the existing ``--tts`` backend
flag (realtime-ws.py:1019) onto the three strategic lanes from
docs/research/2026-06-06-s2s-L1-L5-roadmap.md, and emits a ``which-lane-active``
health dict so the cascade fallback (sota_pipeline.py:452) is observable rather
than silent.

Lanes:
  fish    — frozen-Gemma layer-split true-S2S (primary moat lane, on-device)
  cascade — VAD-free / Whisper->Gemma->{Kokoro,Voxtral,...} (floor, on-device)
  vendor  — native vendor S2S (gpt-realtime/Gemini/Nova) — COMPARISON ONLY,
            never on the product hot path (cannot carry frozen-Gemma+LoRA persona)
"""
from __future__ import annotations

LANES: dict[str, dict] = {
    "fish": {"name": "fish-s2s", "on_device": True, "is_primary": True, "is_floor": False, "comparison_only": False},
    "cascade": {"name": "cascade", "on_device": True, "is_primary": False, "is_floor": True, "comparison_only": False},
    "vendor": {"name": "vendor-s2s", "on_device": False, "is_primary": False, "is_floor": False, "comparison_only": True},
}

# Map the current --tts choices (and common vendor names) onto lanes. Unknown
# backends resolve to the cascade floor — the always-available safe default.
_ALIASES: dict[str, str] = {
    "fish": "fish",
    "kokoro": "cascade",
    "kokoro-onnx": "cascade",
    "voxtral": "cascade",
    "f5": "cascade",
    "native": "cascade",
    "cascade": "cascade",
    "cascaded": "cascade",
    "vendor": "vendor",
    "gpt-realtime": "vendor",
    "gpt-realtime-2": "vendor",
    "gemini-live": "vendor",
    "nova-sonic": "vendor",
    "nova-2-sonic": "vendor",
}

_FLOOR = "cascade"


def _to_lane(value: str | None) -> str | None:
    if not value:
        return None
    return _ALIASES.get(value.strip().lower())


def resolve_lane(requested: str | None = None, env: str | None = None, default: str = "fish") -> str:
    """Resolve the active lane: explicit ``requested`` > ``env`` > ``default``.

    Vendor is comparison-only and must never be auto-selected as a primary via
    env/default — only an explicit ``requested`` vendor backend reaches it.
    Unknown values fall back to the cascade floor.
    """
    explicit = _to_lane(requested)
    if explicit is not None:
        return explicit
    if requested:  # requested something, but it's unknown -> safe floor
        return _FLOOR

    env_lane = _to_lane(env)
    if env_lane is not None and not LANES[env_lane]["comparison_only"]:
        return env_lane

    default_lane = _to_lane(default)
    if default_lane is not None and not LANES[default_lane]["comparison_only"]:
        return default_lane

    return _FLOOR


def lane_health(requested: str | None, active: str, fallback_reason: str | None = None) -> dict:
    """A JSON-serializable ``which-lane-active`` health signal.

    ``fallback_occurred`` is True whenever the active lane differs from what the
    request resolved to (e.g. fish requested, cascade loaded) — making the
    sota_pipeline.py fish->cascade fallback observable instead of silent.
    """
    requested_lane = resolve_lane(requested) if requested else active
    info = LANES.get(active, LANES[_FLOOR])
    fell_back = (active != requested_lane) or (fallback_reason is not None)
    return {
        "requested_tts": requested,
        "requested_lane": requested_lane,
        "active_lane": active,
        "lane_name": info["name"],
        "on_device": info["on_device"],
        "is_floor": info["is_floor"],
        "comparison_only": info["comparison_only"],
        "fallback_occurred": bool(fell_back),
        "fallback_reason": fallback_reason,
    }
