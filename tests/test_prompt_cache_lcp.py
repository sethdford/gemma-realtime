"""Truth-table tests for prompt_cache_lcp planning (pure, no mlx).

Pins the 2026-07-25 design: cross-turn KV reuse by longest-common-token-prefix
instead of whole-system-prompt hash (which hit 0% on h-uman's per-turn-varying
prompts).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prompt_cache_lcp import MODES, common_prefix_len, parse_mode, plan_reuse


# ── parse_mode ──────────────────────────────────────────────────────────

def test_parse_mode_known_values():
    for m in MODES:
        assert parse_mode(m) == m
        assert parse_mode(m.upper()) == m


def test_parse_mode_none_and_empty_use_default():
    assert parse_mode(None) == "shadow"
    assert parse_mode("") == "shadow"
    assert parse_mode(None, default="off") == "off"


def test_parse_mode_unknown_fails_closed_to_off():
    assert parse_mode("on") == "off"
    assert parse_mode("live!") == "off"
    assert parse_mode("1") == "off"


# ── common_prefix_len ───────────────────────────────────────────────────

def test_lcp_identical():
    assert common_prefix_len([1, 2, 3], [1, 2, 3]) == 3


def test_lcp_divergent_midway():
    assert common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_lcp_no_overlap():
    assert common_prefix_len([1, 2], [3, 4]) == 0


def test_lcp_empty_and_length_mismatch():
    assert common_prefix_len([], [1, 2]) == 0
    assert common_prefix_len([1, 2, 3], [1, 2]) == 2


# ── plan_reuse ──────────────────────────────────────────────────────────

def test_first_request_full_reset_of_empty_cache():
    assert plan_reuse(None, [1, 2, 3], 0) == (0, 0)
    assert plan_reuse([], [1, 2, 3], 0) == (0, 0)


def test_stable_head_varying_tail_reuses_head():
    prev = [1, 2, 3, 4, 5, 6]          # head=1..4, tail=5,6
    new = [1, 2, 3, 4, 7, 8, 9]        # same head, new tail
    cache = len(prev) + 10             # cache holds prompt + 10 generated toks
    reuse, trim = plan_reuse(prev, new, cache)
    assert reuse == 4
    assert trim == cache - 4


def test_identical_prompt_leaves_one_token_to_prefill():
    prev = [1, 2, 3, 4]
    new = [1, 2, 3, 4]
    reuse, trim = plan_reuse(prev, new, len(prev))
    assert reuse == 3                  # capped at len(new) - 1
    assert trim == 1


def test_no_common_prefix_full_reset():
    prev = [1, 2, 3]
    cache = 3 + 5
    assert plan_reuse(prev, [9, 8, 7], cache) == (0, cache)


def test_cache_smaller_than_prev_prompt_resets():
    # Inconsistent bookkeeping (e.g. cache reinitialized elsewhere) — never
    # reuse KV that can't correspond to prev_tokens.
    assert plan_reuse([1, 2, 3, 4], [1, 2, 3, 4, 5], 2) == (0, 2)


def test_new_prompt_shorter_than_lcp_cap():
    prev = [1, 2, 3, 4, 5]
    new = [1, 2]                       # lcp=2 but must leave 1 to prefill
    reuse, trim = plan_reuse(prev, new, 5)
    assert reuse == 1
    assert trim == 4


def test_empty_new_prompt_resets():
    assert plan_reuse([1, 2], [], 2) == (0, 2)
