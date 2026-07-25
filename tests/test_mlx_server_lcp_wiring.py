"""Wiring tests for mlx-server's token-LCP prompt-cache reuse (2026-07-25).

Exercises _plan_cache_for_prompt/_after_prefill against real mlx_lm KVCache
objects (offset bookkeeping only — no model, no tensors) with a stub
tokenizer. Requires mlx_lm; skipped where it isn't installed.
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


HEAD = "persona head words " * 50            # 150 stable tokens
P1 = HEAD + "alpha one memory sections here"
P2 = HEAD + "beta two totally different middle content"


def _prefill(srv, payload, generated=0):
    """Simulate prefill+generation: cache offset grows by prompt+generated."""
    srv.lm_prompt_cache[0].offset += len(payload) + generated
    srv._after_prefill([])


def test_first_request_prefills_everything(srv):
    payload = srv._plan_cache_for_prompt([], P1)
    assert isinstance(payload, list)
    assert len(payload) == len(FakeProc().encode(P1))


def test_stable_head_reused_across_varying_tails(srv):
    payload1 = srv._plan_cache_for_prompt([], P1)
    _prefill(srv, payload1, generated=20)

    toks2 = FakeProc().encode(P2)
    payload2 = srv._plan_cache_for_prompt([], P2)
    reuse = len(toks2) - len(payload2)
    assert reuse == 150                      # the head (divergence at word 151)
    assert srv.lm_prompt_cache[0].offset == reuse  # generated tail trimmed off


def test_identical_prompt_leaves_one_token(srv):
    payload1 = srv._plan_cache_for_prompt([], P2)
    _prefill(srv, payload1, generated=7)
    payload2 = srv._plan_cache_for_prompt([], P2)
    assert len(payload2) == 1


def test_no_overlap_fully_resets(srv):
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
    assert srv._lcp_last["prompt_tokens"] == len(FakeProc().encode(P1))


def test_off_mode_returns_string(srv):
    srv.lcp_mode = "off"
    out = srv._plan_cache_for_prompt([{"role": "system", "content": "s"}], P1)
    assert out == P1


def test_legacy_path_invalidates_lcp_bookkeeping(srv):
    payload1 = srv._plan_cache_for_prompt([], P1)
    _prefill(srv, payload1)
    assert srv._lcp_prev_tokens is not None
    # A legacy-managed request (e.g. speculative path) must clear prev tokens
    srv._prepare_cache_for_request([{"role": "system", "content": "s"}])
    assert srv._lcp_prev_tokens is None


def test_turbo_cache_disables_live_reuse(srv):
    srv.turbo_cache = [KVCache()]            # any active turbo cache
    out = srv._plan_cache_for_prompt([{"role": "system", "content": "s"}], P1)
    assert out == P1                          # falls back to legacy string path
