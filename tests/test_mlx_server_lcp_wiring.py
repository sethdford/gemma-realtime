"""Wiring tests for mlx-server's token-LCP prompt-cache reuse (2026-07-25).

Exercises _plan_cache_for_prompt/_after_prefill against real mlx_lm KVCache
objects (offset bookkeeping only — no model, no tensors) with a stub
tokenizer. Requires mlx_lm; skipped where it isn't installed.

Multi-slot upgrade: :8741 serves interleaved callers (the h-uman daemon's
~4K-token persona turns mixed with 162-547-token probes), so a single slot
degrades to reusing only the chat-template preamble. The pool exists so the
daemon's big-head slot SURVIVES the interleaved probe traffic — that's the
central scenario tested here.
"""
import importlib.util
import os
import sys

import pytest

mlx_lm = pytest.importorskip("mlx_lm")
from mlx_lm.models.cache import KVCache  # noqa: E402

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


class FakeProc:
    """1 token per word, no BOS — deterministic and model-free."""
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [hash(w) % 100000 for w in text.split()]


@pytest.fixture()
def srv():
    srv = _load_server()
    srv.processor = FakeProc()
    srv.lm_prompt_cache = [KVCache()]
    srv.turbo_cache = None
    srv.lcp_mode = "live"
    return srv


HEAD = "persona head words " * 50            # 150 stable tokens (>= slot floor)
P1 = HEAD + "alpha one memory sections here"
P2 = HEAD + "beta two totally different middle content"
PROBE_B = "judge this reply yes or no"       # 6 tokens, well below the floor
PROBE_C = "gate check tiny probe request"    # ditto, different content


def _ntok(text):
    return len(FakeProc().encode(text))


def _prefill(srv, payload, generated=0):
    """Simulate prefill+generation: active cache offset grows by prompt+generated."""
    srv.lm_prompt_cache[0].offset += len(payload) + generated
    srv._after_prefill([])


# ── v1 behaviors that must survive the multi-slot refactor ─────────────

def test_first_request_prefills_everything(srv):
    payload = srv._plan_cache_for_prompt([], P1)
    assert isinstance(payload, list)
    assert len(payload) == _ntok(P1)


def test_stable_head_reused_across_varying_tails(srv):
    payload1 = srv._plan_cache_for_prompt([], P1)
    _prefill(srv, payload1, generated=20)

    payload2 = srv._plan_cache_for_prompt([], P2)
    reuse = _ntok(P2) - len(payload2)
    assert reuse == 150                      # the head (divergence at word 151)
    assert srv.lm_prompt_cache[0].offset == reuse  # generated tail trimmed off


def test_identical_prompt_leaves_one_token(srv):
    payload1 = srv._plan_cache_for_prompt([], P2)
    _prefill(srv, payload1, generated=7)
    payload2 = srv._plan_cache_for_prompt([], P2)
    assert len(payload2) == 1


def test_single_slot_no_overlap_fully_resets(srv):
    srv.lm_cache_slot_count = 1              # v1-equivalent configuration
    payload1 = srv._plan_cache_for_prompt([], P1)
    _prefill(srv, payload1, generated=3)
    srv._plan_cache_for_prompt([], "zzz completely different prompt")
    assert srv.lm_prompt_cache[0].offset == 0


def test_shadow_mode_returns_string_and_measures(srv):
    srv.lcp_mode = "shadow"
    srv.lm_prompt_cache[0].offset = 10
    out = srv._plan_cache_for_prompt([{"role": "system", "content": "s"}], P1)
    assert out == P1                          # legacy path manages the cache
    assert srv._lcp_last["mode"] == "shadow"
    assert srv._lcp_last["prompt_tokens"] == _ntok(P1)


def test_off_mode_returns_string(srv):
    srv.lcp_mode = "off"
    out = srv._plan_cache_for_prompt([{"role": "system", "content": "s"}], P1)
    assert out == P1


def test_turbo_cache_disables_live_reuse(srv):
    srv.turbo_cache = [KVCache()]            # any active turbo cache
    out = srv._plan_cache_for_prompt([{"role": "system", "content": "s"}], P1)
    assert out == P1                          # falls back to legacy string path


# ── multi-slot: the interleaved-traffic scenario ────────────────────────

def test_probe_routes_to_empty_slot_and_leaves_big_cache_untouched(srv):
    payload1 = srv._plan_cache_for_prompt([], P1)
    _prefill(srv, payload1, generated=20)
    big_cache = srv.lm_prompt_cache
    big_offset = big_cache[0].offset

    payload_b = srv._plan_cache_for_prompt([], PROBE_B)
    assert len(payload_b) == _ntok(PROBE_B)   # full prefill, no reuse
    assert srv._lcp_last["slot"] == 1
    assert srv.lm_prompt_cache is not big_cache   # routed away from big slot
    assert big_cache[0].offset == big_offset      # big KV state untouched


def test_interleaved_big_small_keeps_big_slot_reuse_high(srv):
    # big A, small B, big A', small C, big A'' — the soak sequence.
    _prefill(srv, srv._plan_cache_for_prompt([], P1), generated=20)   # A
    _prefill(srv, srv._plan_cache_for_prompt([], PROBE_B), generated=5)   # B

    payload = srv._plan_cache_for_prompt([], P2)                      # A'
    assert _ntok(P2) - len(payload) == 150     # shared head fully reused
    assert srv._lcp_last["slot"] == 0
    _prefill(srv, payload, generated=9)

    _prefill(srv, srv._plan_cache_for_prompt([], PROBE_C), generated=5)   # C
    assert srv._lcp_last["slot"] == 1          # LRU probe slot recycled

    payload = srv._plan_cache_for_prompt([], P1)                      # A''
    assert _ntok(P1) - len(payload) == 150     # big slot survived both probes
    assert srv._lcp_last["slot"] == 0


def test_live_reports_slot_stats(srv):
    srv._plan_cache_for_prompt([], P1)
    assert srv._lcp_last["mode"] == "live"
    assert srv._lcp_last["slot"] == 0
    assert srv._lcp_last["slots"] == 2


def test_shadow_simulates_slot_pool(srv):
    srv.lcp_mode = "shadow"
    msgs = [{"role": "system", "content": "s"}]
    srv._plan_cache_for_prompt(msgs, P1)       # A  → sim slot 0
    srv._plan_cache_for_prompt(msgs, PROBE_B)  # B  → sim slot 1
    assert srv._lcp_last["slot"] == 1
    assert srv._lcp_last["would_reuse"] == 0

    srv._plan_cache_for_prompt(msgs, P2)       # A' → sim slot 0, head hit
    assert srv._lcp_last["slot"] == 0
    assert srv._lcp_last["slots"] == 2
    assert srv._lcp_last["would_reuse"] == 150
    assert srv._lcp_last["slot_hit"] is True


# ── fail-safes ──────────────────────────────────────────────────────────

def test_partial_trim_reinitializes_only_that_slot(srv):
    _prefill(srv, srv._plan_cache_for_prompt([], P1), generated=20)
    srv._trim_cache = lambda cache_list, n: 0  # trim silently fails

    payload = srv._plan_cache_for_prompt([], P2)
    assert len(payload) == _ntok(P2)           # reuse abandoned, prefill all
    assert srv._lcp_last["reused"] == 0
    assert srv.lm_prompt_cache[0].offset == 0  # slot rebuilt fresh
    assert srv.lm_prompt_cache is srv.lm_cache_slots[0]["cache"]


def test_invalidate_cross_turn_caches_resets_everything(srv):
    # Arm the shadow simulation FIRST (its legacy call invalidates the
    # active slot's bookkeeping), then live slot state, then legacy fields.
    srv.lcp_mode = "shadow"
    srv._plan_cache_for_prompt([{"role": "system", "content": "s"}], P1)
    srv.lcp_mode = "live"
    _prefill(srv, srv._plan_cache_for_prompt([], P1), generated=11)
    srv._cached_system_hash = 12345
    srv._cached_system_ntokens = 77
    assert srv.lm_cache_slots[0]["prev_tokens"] is not None
    assert srv._lcp_shadow_slots is not None

    srv._invalidate_cross_turn_caches("test")

    # KV computed under old weights must be unreachable everywhere.
    for slot in srv.lm_cache_slots:
        assert slot["prev_tokens"] is None
        assert srv._cache_offset(slot["cache"]) == 0
    assert srv._lcp_shadow_slots is None
    assert srv._cached_system_hash is None
    assert srv._cached_system_ntokens == 0
    assert srv._lcp_pending_tokens is None


def test_invalidate_is_safe_with_no_cache_state(srv):
    srv.lm_prompt_cache = None
    srv.lm_cache_slots = None
    srv._invalidate_cross_turn_caches("test")  # must not raise


def test_adapter_swap_invalidates_caches(srv, tmp_path):
    # The wiring test: the HTTP swap handler itself must reset cache state,
    # because swapped LoRA weights make every cached KV entry stale.
    import io
    import json as _json

    _prefill(srv, srv._plan_cache_for_prompt([], P1), generated=11)
    srv._cached_system_hash = 12345
    srv._cached_system_ntokens = 77
    assert srv._cache_offset(srv.lm_cache_slots[0]["cache"]) > 0

    (tmp_path / "adapters.safetensors").write_bytes(b"")
    srv._apply_adapter_weights = lambda path: 42  # weights swap "succeeds"

    body = _json.dumps({"adapter_path": str(tmp_path)}).encode()
    handler = object.__new__(srv.ChatHandler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    responses = []
    handler._send_json = lambda code, obj: responses.append((code, obj))

    handler._handle_adapter_swap()

    assert responses and responses[0][0] == 200
    for slot in srv.lm_cache_slots:
        assert slot["prev_tokens"] is None
        assert srv._cache_offset(slot["cache"]) == 0
    assert srv._cached_system_hash is None
    assert srv._cached_system_ntokens == 0


def test_legacy_path_invalidates_active_slot_bookkeeping(srv):
    _prefill(srv, srv._plan_cache_for_prompt([], P1))
    assert srv.lm_cache_slots[0]["prev_tokens"] is not None
    # A legacy-managed request (e.g. speculative path) mutates the active
    # slot's cache — its prev-token bookkeeping must be dropped.
    srv._prepare_cache_for_request([{"role": "system", "content": "s"}])
    assert srv.lm_cache_slots[0]["prev_tokens"] is None
