#!/usr/bin/env python3
"""Persona-portability default scorer (specs/phase-a-speech-lane, Task 6 / AC-2).

The decisive Phase-A criterion: the SAME frozen-Gemma+LoRA persona must produce a
consistent persona signature in BOTH the text path and the chosen speech lane
(vendor native-S2S cannot, since it can't carry the LoRA). h-uman has no persona
scorer to reuse, so this is the spec default: a heuristic **trait-vector cosine**.

Each trait is scored from text by counting WHOLE-WORD lexical markers (per
~/.claude/rules/substring-classifier-pitfalls.md — substring matching would let
"informal" trigger "formal"). The trait value is (high_markers - low_markers)
normalized by token count. Two texts are "same persona" iff cosine >= 0.85.

This is intentionally lightweight: the live probe renders one prompt through text
vs a speech lane (→ASR transcribe), scores both, and compares. Swap in a learned
trait classifier later without changing the cosine/decision interface.
"""
from __future__ import annotations

import math
import re

DEFAULT_THRESHOLD = 0.85

# Whole-word markers per trait: (high-signal, low-signal).
TRAITS: dict[str, tuple[set[str], set[str]]] = {
    "formality": (
        {"therefore", "however", "regarding", "furthermore", "sincerely", "kindly", "shall", "regards"},
        {"gonna", "wanna", "yeah", "lol", "hey", "kinda", "cuz", "gotta", "ok"},
    ),
    "warmth": (
        {"thanks", "thank", "appreciate", "glad", "happy", "wonderful", "care", "lovely", "welcome"},
        {"whatever", "ugh", "stop", "annoyed", "no"},
    ),
    "enthusiasm": (
        {"great", "awesome", "excited", "amazing", "love", "fantastic", "yay"},
        {"meh", "fine", "sure"},
    ),
    "directness": (
        {"must", "need", "now", "immediately", "do", "stop"},
        {"maybe", "perhaps", "might", "possibly", "sort"},
    ),
    "verbosity": (
        {"additionally", "moreover", "specifically", "furthermore", "detail", "elaborate"},
        {"briefly", "quick", "short", "just"},
    ),
    "humor": (
        {"lol", "haha", "funny", "joke", "hilarious", "kidding"},
        set(),
    ),
}

_TOKEN_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def trait_vector(text: str) -> list[float]:
    """Signed trait scores in TRAITS order: (#high - #low) / #tokens per trait."""
    toks = _tokens(text)
    n = max(1, len(toks))
    counts = {t: 0 for t in toks}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    vec: list[float] = []
    for high, low in TRAITS.values():
        score = sum(counts.get(w, 0) for w in high) - sum(counts.get(w, 0) for w in low)
        vec.append(score / n)
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity with explicit zero-vector handling: two no-signal vectors
    are treated as identical (1.0); one zero + one non-zero is orthogonal (0.0)."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def persona_consistency(text_a: str, text_b: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compare two renderings; ``consistent`` iff cosine >= threshold (AC-2 default)."""
    va, vb = trait_vector(text_a), trait_vector(text_b)
    c = cosine(va, vb)
    return {
        "cosine": round(c, 4),
        "consistent": bool(c >= threshold),
        "threshold": threshold,
        "vector_a": va,
        "vector_b": vb,
        "traits": list(TRAITS.keys()),
    }


VENDOR_REASON = (
    "vendor native-S2S cannot load the frozen-Gemma+LoRA persona; "
    "persona portability fails by construction (AC-2)"
)


def vendor_lane_persona_verdict() -> dict:
    """The structural fact that disqualifies vendor S2S from the persona path: it
    owns the brain, so the same LoRA persona cannot be carried — no render needed."""
    return {"lane": "vendor", "cosine": None, "consistent": False,
            "by_construction": True, "reason": VENDOR_REASON}


def probe_persona_portability(reference_render: str, lane_renders: dict[str, str],
                              threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Task-6 live probe (orchestration): compare each lane's render of the same
    persona prompt against the text-path ``reference_render``.

    On-device lanes (fish/cascade) are scored by trait-vector cosine; a ``vendor``
    lane is failed by construction (it can't carry the LoRA). The live caller
    supplies the renders (text path + each speech lane → ASR transcript).
    """
    lanes: dict[str, dict] = {}
    for lane, text in lane_renders.items():
        if lane == "vendor":
            lanes[lane] = vendor_lane_persona_verdict()
        else:
            r = persona_consistency(reference_render, text, threshold)
            lanes[lane] = {"lane": lane, "cosine": r["cosine"],
                           "consistent": r["consistent"], "by_construction": False}
    return {
        "reference_render": reference_render,
        "threshold": threshold,
        "lanes": lanes,
        "portable_lanes": [l for l, v in lanes.items() if v["consistent"]],
    }
