"""LCP slot pool v2 (2026-10-02): configurable slot count, a dedicated scratch
slot for small prompts, and X-HU-Purpose request logging.

Why: the 2026-10-01 latency profile measured 28% of main-reply calls evicted
with 3 slots. Small prompts (fact extract, judges, classifiers, proposer)
churned the pool; an evicted main reply re-prefills ~5k tokens (~16 s at
~316 tok/s). Worse than eviction, a small prompt that shares a long slot's
head REUSES that slot and plan_reuse trims it to the short LCP — destroying
the long prefix just as surely.

Model-free: real mlx_lm KVCache objects (offset bookkeeping only) and a stub
1-token-per-word tokenizer, the same harness as test_mlx_server_lcp_wiring.py.
"""
import importlib.util
import io
import os
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)

from prompt_cache_lcp import (  # noqa: E402
    DEFAULT_SCRATCH_MAX_TOKENS,
    parse_scratch_max_tokens,
    route_request,
)


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_scratch_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


class FakeProc:
    """1 token per word, no BOS — deterministic and model-free."""
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [hash(w) % 100000 for w in text.split()]


def _words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


# A ~5k-token reply prompt would make the tests slow to read; 600 words is
# comfortably above the 400-token scratch threshold, which is what matters.
HEAD_A = _words("a", 600)
HEAD_B = _words("b", 600)
REPLY_A1 = HEAD_A + " " + _words("turn1_", 20)
REPLY_A2 = HEAD_A + " " + _words("turn2_", 20)
REPLY_B1 = HEAD_B + " " + _words("bturn1_", 20)
REPLY_B2 = HEAD_B + " " + _words("bturn2_", 20)
# Small prompt that SHARES 150 tokens of HEAD_A (>= the 64-token floor): the
# dangerous shape — without scratch it reuses slot 0 and trims it to 150.
SMALL_SHARING_HEAD = _words("a", 150) + " " + _words("judge", 30)


def _ntok(text):
    return len(FakeProc().encode(text))


# ── pure routing ────────────────────────────────────────────────────────

BIG = list(range(1000, 6000))
SMALL = list(range(1000, 1150)) + list(range(90000, 90030))  # shares 150 of BIG


def test_parse_scratch_max_tokens():
    assert parse_scratch_max_tokens(None) == DEFAULT_SCRATCH_MAX_TOKENS == 400
    assert parse_scratch_max_tokens("250") == 250
    assert parse_scratch_max_tokens("0") == 0          # disabled
    assert parse_scratch_max_tokens(-5) == 0
    assert parse_scratch_max_tokens(10 ** 9) == 8192
    assert parse_scratch_max_tokens("junk") == 400


def test_route_small_prompt_goes_to_scratch_even_when_it_shares_the_head():
    prev = [BIG, None, None]
    used = [1, 0, 0]
    # Control: legacy routing (no scratch, no guard) reuses slot 0 — trims it.
    idx, reuse, scratch = route_request(prev, used, SMALL, guard_min_tokens=0)
    assert (idx, reuse, scratch) == (0, 150, False)
    # With scratch: the small prompt never touches slot 0.
    idx, reuse, scratch = route_request(prev, used, SMALL, scratch_idx=2,
                                        scratch_max_tokens=400)
    assert (idx, reuse, scratch) == (2, 0, True)


def test_route_small_prompt_reuses_scratch_prefix_above_floor():
    prev = [BIG, None, SMALL]
    idx, reuse, scratch = route_request(prev, [1, 0, 2], SMALL,
                                        scratch_idx=2, scratch_max_tokens=400)
    assert (idx, scratch) == (2, True)
    assert reuse == len(SMALL) - 1                     # identical: leave 1 to prefill


def test_route_long_prompt_uses_best_lcp_and_never_the_scratch():
    other = list(range(7000, 12000))
    prev = [other, BIG, BIG]                           # scratch (idx 2) "matches" best too
    idx, reuse, scratch = route_request(prev, [5, 3, 9], BIG[:4000] + [1, 2, 3],
                                        scratch_idx=2, scratch_max_tokens=400)
    assert (idx, reuse, scratch) == (1, 4000, False)


def test_route_long_prompt_eviction_never_picks_scratch():
    # All regular slots full of unrelated prompts; scratch is the cheapest
    # victim by value but must not be chosen for a long prompt.
    prev = [list(range(10000, 15000)), list(range(20000, 25000)), [1, 2, 3]]
    idx, reuse, scratch = route_request(prev, [1, 2, 3], BIG,
                                        scratch_idx=2, scratch_max_tokens=400)
    assert scratch is False and idx in (0, 1) and reuse == 0


def test_route_scratch_disabled_is_choose_slot():
    prev = [BIG, None, None]
    assert route_request(prev, [1, 0, 0], SMALL, scratch_idx=2,
                         scratch_max_tokens=0, guard_min_tokens=0) == (0, 150, False)


# ── server wiring ───────────────────────────────────────────────────────

@pytest.fixture()
def srv():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import KVCache
    srv = _load_server()
    srv.processor = FakeProc()
    srv.lm_prompt_cache = [KVCache()]
    srv.turbo_cache = None
    srv.lcp_mode = "live"
    srv.lm_cache_slot_count = 6
    srv.lcp_scratch_max_tokens = 400
    return srv


def _prefill(srv, payload, generated=0):
    srv.lm_prompt_cache[0].offset += len(payload) + generated
    srv._after_prefill([])


def test_pool_has_six_regular_slots_plus_scratch(srv):
    srv._plan_cache_for_prompt([], REPLY_A1)
    assert len(srv.lm_cache_slots) == 7
    assert srv._lcp_scratch_index(7) == 6
    assert srv._lcp_last["slots"] == 7


def test_small_prompts_never_evict_or_trim_slot0_long_prefix(srv):
    _prefill(srv, srv._plan_cache_for_prompt([], REPLY_A1), generated=9)
    slot0 = srv.lm_cache_slots[0]
    assert srv._lcp_last["slot"] == 0
    held_tokens = list(slot0["prev_tokens"])
    held_offset = slot0["cache"][0].offset
    assert held_offset == _ntok(REPLY_A1) + 9          # precondition: long prefix cached

    # 25 small prompts, including the dangerous head-sharing shape.
    for i in range(25):
        text = SMALL_SHARING_HEAD if i % 2 == 0 else _words(f"probe{i}_", 40)
        _prefill(srv, srv._plan_cache_for_prompt([], text), generated=5)
        assert srv._lcp_last["slot"] == 6 and srv._lcp_last["scratch"] is True
        assert slot0["cache"][0].offset == held_offset
        assert slot0["prev_tokens"] == held_tokens

    payload = srv._plan_cache_for_prompt([], REPLY_A2)
    assert srv._lcp_last["slot"] == 0
    assert _ntok(REPLY_A2) - len(payload) == 600       # whole head reused


def test_without_scratch_the_head_sharing_small_prompt_trims_slot0(srv):
    # Non-vacuity control for the test above: the hazard is real.
    srv.lcp_scratch_max_tokens = 0
    _prefill(srv, srv._plan_cache_for_prompt([], REPLY_A1), generated=9)
    srv._plan_cache_for_prompt([], SMALL_SHARING_HEAD)
    assert srv._lcp_last["slot"] == 0
    assert srv.lm_cache_slots[0]["cache"][0].offset == 150   # long prefix destroyed


def test_long_prompts_still_use_best_lcp_selection(srv):
    _prefill(srv, srv._plan_cache_for_prompt([], REPLY_A1), generated=9)   # slot 0
    _prefill(srv, srv._plan_cache_for_prompt([], REPLY_B1), generated=9)   # slot 1
    _prefill(srv, srv._plan_cache_for_prompt([], _words("x", 30)), generated=3)  # scratch

    payload = srv._plan_cache_for_prompt([], REPLY_B2)
    assert srv._lcp_last["slot"] == 1 and srv._lcp_last["scratch"] is False
    assert _ntok(REPLY_B2) - len(payload) == 600
    _prefill(srv, payload, generated=9)

    payload = srv._plan_cache_for_prompt([], REPLY_A2)
    assert srv._lcp_last["slot"] == 0
    assert _ntok(REPLY_A2) - len(payload) == 600


def test_shadow_mode_routes_small_prompts_to_scratch(srv):
    srv.lcp_mode = "shadow"
    msgs = [{"role": "system", "content": "s"}]
    srv._plan_cache_for_prompt(msgs, REPLY_A1)
    assert srv._lcp_last["slot"] == 0
    srv._plan_cache_for_prompt(msgs, SMALL_SHARING_HEAD)
    assert srv._lcp_last["slot"] == 6 and srv._lcp_last["scratch"] is True
    srv._plan_cache_for_prompt(msgs, REPLY_A2)
    assert srv._lcp_last["slot"] == 0 and srv._lcp_last["would_reuse"] == 600


def test_pool_config_precedence():
    srv = _load_server()
    f = srv._lcp_pool_config_raw
    assert f({}, {}) == (None, None)                          # -> defaults 6 / 400
    assert f({"prompt_cache_slots": 3}, {}) == (3, None)      # prod config pins 3
    assert f({"prompt_cache_slots": 3}, {"MLX_PROMPT_CACHE_SLOTS": "4"})[0] == "4"
    assert f({"prompt_cache_slots": 3},
             {"MLX_PROMPT_CACHE_SLOTS": "4", "MLX_LCP_SLOTS": "6"})[0] == "6"
    assert f({"prompt_cache_slots": 3}, {"MLX_LCP_SLOTS": "  "})[0] == 3
    assert f({"prompt_cache_scratch_max_tokens": 200},
             {"MLX_LCP_SCRATCH_MAX_TOKENS": "0"})[1] == "0"   # env can disable
    assert f({"prompt_cache_scratch_max_tokens": 200}, {})[1] == 200


# ── X-HU-Purpose logging ────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("reply", "reply"),
    ("Fact-Extract", "fact_extract"),
    ("judge.v2", "judge_v"),
    ("reply\r\nINJECTED: prompt text", "replyinjected_prompt_tex"),
    ("x" * 40, "x" * 24),
    ("1234!!", None),
    ("", None),
    (None, None),
])
def test_sanitize_purpose(raw, expected):
    srv = _load_server()
    out = srv._sanitize_purpose(raw)
    assert out == expected
    if out is not None:
        import re
        assert re.fullmatch(r"[a-z_]{1,24}", out)


def _non_stream_log_line(srv, monkeypatch, capsys, headers):
    monkeypatch.setattr(srv, "generate_response", lambda *a, **k: ("hey there", 12, 3))
    monkeypatch.setattr(srv, "_reseed_sampling", lambda: None)
    handler = object.__new__(srv.ChatHandler)
    handler.headers = headers
    handler.wfile = io.BytesIO()
    handler._send_json = lambda code, obj, headers=None: None
    handler._handle_non_stream({"messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 8}, "chatcmpl-test", 0.0)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  -> ")]
    assert len(lines) == 1
    return lines[0]


def test_per_request_log_line_carries_sanitized_purpose(monkeypatch, capsys):
    srv = _load_server()
    line = _non_stream_log_line(srv, monkeypatch, capsys,
                                {"X-HU-Purpose": "Fact-Extract\nevil"})
    assert " purpose=fact_extractevil " in line
    assert "\nevil" not in line


def test_per_request_log_line_without_purpose_header(monkeypatch, capsys):
    srv = _load_server()
    line = _non_stream_log_line(srv, monkeypatch, capsys, {})
    assert "purpose=" not in line


# ── review round (2026-10-02): per-slot cap, reuse-trim guard, shadow policy,
#    base-weights isolation, adapter-swap clearing ─────────────────────────

def test_route_oversize_prompt_never_reuses_and_goes_to_scratch():
    prev = [BIG, None, None]
    long_prompt = BIG + list(range(50000, 58000))      # 13k tokens, shares all of BIG
    assert route_request(prev, [1, 0, 0], long_prompt, scratch_idx=2,
                         scratch_max_tokens=400, slot_max_tokens=12000) == (2, 0, True)
    # Without scratch: the least valuable regular slot (an empty one), no reuse.
    assert route_request(prev, [1, 0, 0], long_prompt,
                         slot_max_tokens=12000) == (1, 0, False)
    # Control: uncapped, it would reuse slot 0's whole prefix.
    assert route_request(prev, [1, 0, 0], long_prompt)[:2] == (0, 5000)


def test_route_guard_refuses_shredding_a_big_slot():
    mid = BIG[:100] + list(range(60000, 60600))        # 700 toks, shares 100
    prev, used = [BIG, None, None], [1, 0, 0]
    # Control: guard off -> reuse slot 0 and trim it 5000 -> 100.
    assert route_request(prev, used, mid, scratch_idx=2, scratch_max_tokens=400,
                         guard_min_tokens=0) == (0, 100, False)
    # Guard on, scratch available, prompt < 2000 -> scratch.
    assert route_request(prev, used, mid, scratch_idx=2,
                         scratch_max_tokens=400) == (2, 0, True)
    # Guard on, no scratch -> least valuable OTHER regular slot, reset.
    assert route_request(prev, used, mid) == (1, 0, False)
    # Big prompt sharing < half -> other regular slot even with scratch on.
    big_other = BIG[:1000] + list(range(70000, 73000))
    assert route_request(prev, used, big_other, scratch_idx=2,
                         scratch_max_tokens=400) == (1, 0, False)


def test_route_guard_allows_reuse_of_at_least_half_and_single_slot_legacy():
    half_plus = BIG[:2600] + list(range(80000, 80400))
    assert route_request([BIG, None], [1, 0], half_plus)[:2] == (0, 2600)
    mid = BIG[:100] + list(range(60000, 60600))
    # One regular slot, no scratch: nowhere else to go, legacy trim allowed.
    assert route_request([BIG], [1], mid) == (0, 100, False)
    # Slot at/below 2000 tokens is not protected.
    small_slot = BIG[:2000]
    assert route_request([small_slot, None], [1, 0], mid)[:2] == (0, 100)


@pytest.fixture()
def capped(srv):
    srv.lm_cache_slot_count = 1
    srv.lcp_slot_max_tokens = 1000
    return srv


def test_oversize_prompt_is_not_cached_and_scratch_is_reallocated_next(capped, capsys):
    srv = capped
    _prefill(srv, srv._plan_cache_for_prompt([], REPLY_A1), generated=9)   # slot 0
    slot0_tokens = list(srv.lm_cache_slots[0]["prev_tokens"])
    slot0_offset = srv.lm_cache_slots[0]["cache"][0].offset
    scratch = srv.lm_cache_slots[1]
    scratch_cache_before = scratch["cache"]

    huge = _words("h", 1200)                                              # > cap
    payload = srv._plan_cache_for_prompt([], huge)
    assert len(payload) == 1200                         # full prefill, no reuse
    assert srv._lcp_last["slot"] == 1 and srv._lcp_last["cached"] is False
    assert scratch["cache"] is not scratch_cache_before  # freed + reallocated
    _prefill(srv, payload, generated=50)
    assert scratch["prev_tokens"] == []                 # NOT cached
    assert srv.lm_cache_slots[0]["prev_tokens"] == slot0_tokens
    assert srv.lm_cache_slots[0]["cache"][0].offset == slot0_offset

    grown = scratch["cache"]
    capsys.readouterr()
    srv._plan_cache_for_prompt([], _words("tiny", 20))
    assert srv._lcp_last["slot"] == 1
    assert scratch["cache"] is not grown                # peak 1250 > 1000 -> reallocated
    assert scratch["cache"][0].offset == 0
    assert "[lcp cap] slot 1" in capsys.readouterr().out


def test_recycling_a_slot_whose_peak_exceeded_cap_reallocates_not_trims(capped):
    srv = capped
    first = _words("p", 900)                            # under cap...
    _prefill(srv, srv._plan_cache_for_prompt([], first), generated=300)  # ...peak 1200
    slot0 = srv.lm_cache_slots[0]
    grown = slot0["cache"]
    srv._plan_cache_for_prompt([], _words("q", 700))    # unrelated, recycles slot 0
    assert srv._lcp_last["slot"] == 0
    assert slot0["cache"] is not grown and slot0["cache"][0].offset == 0
    assert slot0["peak"] == 0


def test_recycling_under_cap_still_trims_in_place(capped):
    srv = capped
    _prefill(srv, srv._plan_cache_for_prompt([], _words("p", 700)), generated=100)
    slot0 = srv.lm_cache_slots[0]
    same = slot0["cache"]
    srv._plan_cache_for_prompt([], _words("q", 700))
    assert slot0["cache"] is same and same[0].offset == 0   # trimmed, not reallocated


def test_guard_keeps_big_slot_intact_in_live_wiring(srv):
    big_head = _words("g", 2500)
    _prefill(srv, srv._plan_cache_for_prompt([], big_head + " " + _words("t1_", 10)),
             generated=9)
    slot0 = srv.lm_cache_slots[0]
    held = slot0["cache"][0].offset
    mid = _words("g", 100) + " " + _words("m", 600)     # 700 toks, shares 100
    srv._plan_cache_for_prompt([], mid)
    assert srv._lcp_last["scratch"] is True
    assert slot0["cache"][0].offset == held


def test_shadow_routing_honours_evict_policy(srv):
    srv.lcp_mode = "shadow"
    srv.lcp_scratch_max_tokens = 0
    srv.lm_cache_slot_count = 2
    msgs = [{"role": "system", "content": "s"}]

    def run(policy):
        srv.lcp_evict_policy = policy
        srv._lcp_shadow_slots = None
        srv._plan_cache_for_prompt(msgs, _words("a", 600))   # slot 0, older, big
        srv._plan_cache_for_prompt(msgs, _words("b", 100))   # slot 1, newer, small
        srv._plan_cache_for_prompt(msgs, _words("c", 500))   # miss -> victim
        return srv._lcp_last["slot"]

    assert run("value") == 1        # 100/1 < 600/2: evict the small one
    assert run("lru") == 0          # legacy: oldest regardless of size


def _slot_snapshot(srv):
    return [(id(s["cache"]), s["cache"][0].offset,
             None if s["prev_tokens"] is None else list(s["prev_tokens"]),
             s["last_used"]) for s in srv.lm_cache_slots]


def test_base_weights_requests_never_touch_any_slot_including_scratch(srv):
    _prefill(srv, srv._plan_cache_for_prompt([], REPLY_A1), generated=9)
    _prefill(srv, srv._plan_cache_for_prompt([], _words("s", 30)), generated=4)
    assert srv.lm_cache_slots[6]["prev_tokens"]          # scratch is populated
    before = _slot_snapshot(srv)
    last = dict(srv._lcp_last)

    with srv._base_weights_request():
        for text in (_words("s", 30), _words("other", 50), REPLY_A2):
            out = srv._plan_cache_for_prompt([], text)
            assert out == text                           # string: no slot planning
            assert "prompt_cache" not in srv._kv_kwargs()  # private throwaway cache
            srv._after_prefill([])

    assert _slot_snapshot(srv) == before
    assert srv._lcp_last == last


def test_disable_adapter_request_runs_under_base_weights(srv):
    seen = []
    handler = object.__new__(srv.ChatHandler)
    handler._handle_non_stream = lambda req, rid, t0: seen.append(
        (srv._base_weights_depth, "prompt_cache" in srv._kv_kwargs()))
    handler._do_chat_completion({"messages": [], "disable_adapter": True}, "id", 0.0)
    handler._do_chat_completion({"messages": []}, "id", 0.0)
    assert seen[0] == (1, False)                         # base: no shared cache
    assert seen[1][0] == 0 and seen[1][1] is True        # adapter: shared cache back


def test_adapter_swap_clears_the_scratch_slot(srv, tmp_path):
    import json as _json
    _prefill(srv, srv._plan_cache_for_prompt([], _words("s", 30)), generated=4)
    scratch = srv.lm_cache_slots[srv._lcp_scratch_index(len(srv.lm_cache_slots))]
    assert scratch["prev_tokens"] and srv._cache_offset(scratch["cache"]) > 0

    (tmp_path / "adapters.safetensors").write_bytes(b"")
    srv._apply_adapter_weights = lambda path: 42
    body = _json.dumps({"adapter_path": str(tmp_path)}).encode()
    handler = object.__new__(srv.ChatHandler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    responses = []
    handler._send_json = lambda code, obj: responses.append((code, obj))
    handler._handle_adapter_swap()

    assert responses and responses[0][0] == 200
    assert scratch["prev_tokens"] is None
    assert srv._cache_offset(scratch["cache"]) == 0
    # A small prompt after the swap gets no reuse from pre-swap KV.
    payload = srv._plan_cache_for_prompt([], _words("s", 30))
    assert len(payload) == 30 and srv._lcp_last["slot"] == 6


def test_slot_max_config_precedence_and_parse():
    from prompt_cache_lcp import parse_slot_max_tokens
    srv = _load_server()
    assert srv._lcp_slot_max_raw({}, {}) is None
    assert parse_slot_max_tokens(None) == 12000
    assert parse_slot_max_tokens("0") == 0
    assert parse_slot_max_tokens("junk") == 12000
    assert srv._lcp_slot_max_raw({"prompt_cache_slot_max_tokens": 9000}, {}) == 9000
    assert srv._lcp_slot_max_raw({"prompt_cache_slot_max_tokens": 9000},
                                 {"MLX_LCP_SLOT_MAX_TOKENS": "16000"}) == "16000"
