"""Truth-table tests for prompt_cache_lcp planning (pure, no mlx).

Pins the 2026-07-25 design: cross-turn KV reuse by longest-common-token-prefix
instead of whole-system-prompt hash (which hit 0% on h-uman's per-turn-varying
prompts).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prompt_cache_lcp import (
    DEFAULT_SLOT_FLOOR,
    MODES,
    choose_slot,
    slot_evict_value,
    parse_evict_policy,
    common_prefix_len,
    parse_mode,
    parse_slots,
    plan_reuse,
)


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


# ── parse_slots ─────────────────────────────────────────────────────────

def test_parse_slots_default_and_clamping():
    assert parse_slots(None) == 2
    assert parse_slots("") == 2
    assert parse_slots(3) == 3
    assert parse_slots("3") == 3
    assert parse_slots(0) == 1                 # floor at 1 (single-slot v1)
    assert parse_slots(99) == 4                # each slot is real KV memory
    assert parse_slots("garbage") == 2


# ── choose_slot ─────────────────────────────────────────────────────────
#
# The pool exists so the daemon's big-persona-head slot SURVIVES interleaved
# small-probe traffic on :8741. Every scenario below is a distillation of
# that traffic shape.

BIG = list(range(1000, 6000))                  # 5000-token daemon prompt
PROBE = list(range(30))                        # small judge/gate probe
PREAMBLE = BIG[:9]                             # shared chat-template preamble


def test_choose_slot_picks_max_lcp_above_floor():
    # Slot 0 holds the big head, slot 1 a probe. A same-caller big prompt
    # (same head, new tail) must route to slot 0 with the full head reuse.
    new = BIG[:4000] + list(range(90000, 90100))
    idx, reuse = choose_slot([BIG, PROBE], [1, 2], new)
    assert idx == 0
    assert reuse == 4000


def test_choose_slot_below_floor_prefers_empty_slot():
    # A probe sharing only the 9-token preamble with the big slot must NOT
    # evict it while an empty slot exists.
    new = PREAMBLE + list(range(70000, 70020))
    idx, reuse = choose_slot([BIG, None], [1, 0], new)
    assert idx == 1
    assert reuse == 0                          # fresh reset, no reuse


def test_choose_slot_below_floor_no_empty_evicts_victim_not_best():
    # Pool full, probe matches both slots only at the preamble (< floor).
    # The probe slot is recycled; the big slot survives even though it has
    # the (marginally) longer LCP.
    # NOTE: this case does NOT discriminate between policies — the victim is
    # both the least-recently-used AND the smallest, so 'lru' and 'value'
    # agree. The discriminating cases are below.
    new = PREAMBLE + list(range(80000, 80020))
    old_probe = PREAMBLE + list(range(60000, 60020))
    idx, reuse = choose_slot([BIG, old_probe], [5, 2], new)
    assert idx == 1
    assert reuse == 0


# ── eviction policy: value vs lru (2026-07-27) ──────────────────────────
#
# Prod logging showed plain LRU destroying exactly what the pool exists to
# protect. Verbatim from mlx-server-launchd.log:
#
#   [lcp evict] slot 1: dropping 4240 kv pos (4227 prev toks), idle 2 reqs;
#               incoming 219 toks; pool[0:166t/0a, 1:4227t/2a, 2:10t/1a]
#
# A 219-token request evicted a 4227-token cached prefix because that slot
# happened to be the oldest. The tests below pin the fix against that exact
# pool state.

def test_value_policy_keeps_big_stale_head_over_small_fresh_probe():
    # THE production scenario, replayed: the big head is the STALEST slot, so
    # LRU targets it. Value must pick the 10-token slot instead.
    head = list(range(1000, 5227))             # 4227 tokens, age 2 (stalest)
    mid = list(range(20000, 20166))            # 166 tokens,  age 0
    probe = list(range(30000, 30010))          # 10 tokens,   age 1
    new = list(range(90000, 90219))            # 219 tokens, shares nothing
    last_used = [5, 3, 4]                      # now=5 → ages [0, 2, 1]

    idx, reuse = choose_slot([mid, head, probe], last_used, new)
    assert idx == 2, "value policy must evict the 10-token slot"
    assert reuse == 0

    # And prove the policies actually disagree here — otherwise this test
    # would pass without the fix and prove nothing.
    legacy, _ = choose_slot([mid, head, probe], last_used, new, policy="lru")
    assert legacy == 1, "legacy LRU evicts the 4227-token head (the bug)"


def test_lru_policy_remains_available_as_rollback():
    big = list(range(1000, 3000))
    small = list(range(10))
    # big is stale (age 2), small is fresh (age 0)
    assert choose_slot([big, small], [1, 3], list(range(70000, 70050)),
                       policy="lru")[0] == 0      # oldest
    assert choose_slot([big, small], [1, 3], list(range(70000, 70050)),
                       policy="value")[0] == 1    # least valuable


def test_value_policy_still_prefers_an_empty_slot_first():
    # An empty slot costs nothing to recycle and must always win.
    big = list(range(1000, 3000))
    idx, reuse = choose_slot([big, None], [3, 1], list(range(70000, 70050)))
    assert idx == 1
    assert reuse == 0


def test_value_decays_so_no_slot_is_immortal():
    # A big entry outranks a fresh small one only while size/(age+1) is
    # larger. 4227 vs a fresh 200-token slot crosses over around age 20.
    head = list(range(4227))
    fresh = list(range(50000, 50200))          # 200 tokens, age 0
    new = list(range(90000, 90100))

    # age 10 → 4227/11 = 384 > 200: the head is still worth protecting.
    assert choose_slot([head, fresh], [0, 10], new)[0] == 1
    # age 30 → 4227/31 = 136 < 200: the stale head is now the victim.
    assert choose_slot([head, fresh], [0, 30], new)[0] == 0


def test_reuse_selection_is_unchanged_by_policy():
    # Only the choice of VICTIM moved; a hit above the floor must be
    # identical under both policies.
    new = BIG[:4000] + list(range(90000, 90100))
    assert (choose_slot([BIG, PROBE], [1, 2], new)
            == choose_slot([BIG, PROBE], [1, 2], new, policy="lru"))


def test_slot_evict_value_shape():
    assert slot_evict_value(None, 0) == 0.0        # empty slots die first
    assert slot_evict_value([], 5) == 0.0
    assert slot_evict_value(list(range(100)), 0) == 100.0
    assert slot_evict_value(list(range(100)), 1) == 50.0
    assert slot_evict_value(list(range(100)), 9) == 10.0


def test_parse_evict_policy_defaults_and_rejects_junk():
    assert parse_evict_policy(None) == "value"
    assert parse_evict_policy("lru") == "lru"
    assert parse_evict_policy("VALUE") == "value"
    assert parse_evict_policy("nonsense") == "value"   # unknown → default
    assert parse_evict_policy("") == "value"


def test_choose_slot_identical_prompt_caps_at_len_minus_one():
    idx, reuse = choose_slot([BIG, None], [1, 0], list(BIG))
    assert idx == 0
    assert reuse == len(BIG) - 1               # mlx_lm needs 1 token to prefill


def test_choose_slot_all_empty_uses_first_slot():
    assert choose_slot([None, None, None], [0, 0, 0], BIG) == (0, 0)


def test_choose_slot_tie_breaks_to_most_recent():
    # Two slots with an identical qualifying LCP: prefer the most recently
    # used so the LRU slot stays available for eviction.
    head = list(range(200))
    a = head + [11, 12, 13]
    b = head + [21, 22, 23]
    new = head + [31, 32, 33]
    idx, reuse = choose_slot([a, b], [1, 2], new)
    assert idx == 1
    assert reuse == 200


def test_choose_slot_floor_is_inclusive():
    prev = list(range(DEFAULT_SLOT_FLOOR)) + [500]
    new = list(range(DEFAULT_SLOT_FLOOR)) + [600]
    idx, reuse = choose_slot([prev, None], [1, 0], new)
    assert (idx, reuse) == (0, DEFAULT_SLOT_FLOOR)


def test_choose_slot_short_prompt_never_reuses_negative():
    # A 1-token prompt can never reuse (cap = len-1 = 0) → fresh slot.
    idx, reuse = choose_slot([BIG, None], [1, 0], [BIG[0]])
    assert reuse == 0
    assert idx == 1                            # empty slot, big survives
