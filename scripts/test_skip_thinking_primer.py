#!/usr/bin/env python3
"""Unit tests for the reply-first serving-config fix in mlx-server.py.

Covers _resolve_skip_thinking_primer / _request_skip_thinking_primer and that
prepare_prompt_lm honors the per-request primer-skip — the Task-0 serving-config
lever that makes streamed casual tiers (stream_strip=false) reply FIRST instead
of front-loading Gemma's forced `<|channel>thought` deliberation.

Plain-runner pattern (no pytest). Run:
    python3 scripts/test_skip_thinking_primer.py

mlx-server.py is loaded via importlib (hyphenated filename is not import-by-name).
Its heavy deps (mlx_lm/mlx_vlm) are imported lazily inside functions, and the
model is None at module scope, so importing requires no live model / no mlx.
"""
import importlib.util
import os
import sys
from pathlib import Path

_SERVER_PATH = Path(__file__).parent / "mlx-server.py"
_spec = importlib.util.spec_from_file_location("mlx_server_under_test", _SERVER_PATH)
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)


# --- _resolve_skip_thinking_primer: the pure decision (req_flag, no_think) ----

def test_resolve_casual_stream_strip_false_skips_primer():
    # Casual tier sends stream_strip=false → reply-first → skip the primer,
    # regardless of the global no-think config.
    assert srv._resolve_skip_thinking_primer(False, False) is True
    assert srv._resolve_skip_thinking_primer(False, True) is True
    print("✓ resolve_casual_stream_strip_false_skips_primer")


def test_resolve_heavy_stream_strip_true_falls_through_to_global():
    # Heavy tier (stream_strip=true) must NOT opt into reply-first; it keeps the
    # server's global behavior (think-first unless no-think is configured).
    assert srv._resolve_skip_thinking_primer(True, False) is False
    assert srv._resolve_skip_thinking_primer(True, True) is True
    print("✓ resolve_heavy_stream_strip_true_falls_through_to_global")


def test_resolve_unspecified_none_falls_through_to_global():
    # No per-request override → preserve EXACT legacy default (skip iff no-think).
    assert srv._resolve_skip_thinking_primer(None, False) is False
    assert srv._resolve_skip_thinking_primer(None, True) is True
    print("✓ resolve_unspecified_none_falls_through_to_global")


def test_resolve_always_returns_bool():
    for rf in (True, False, None):
        for nt in (True, False):
            assert isinstance(srv._resolve_skip_thinking_primer(rf, nt), bool)
    print("✓ resolve_always_returns_bool")


# --- _request_skip_thinking_primer: extract from request body + env -----------

def _with_disable_thinking(value):
    """Context-free helper: set/clear GEMMA_DISABLE_THINKING, return prior value."""
    prior = os.environ.get("GEMMA_DISABLE_THINKING")
    if value is None:
        os.environ.pop("GEMMA_DISABLE_THINKING", None)
    else:
        os.environ["GEMMA_DISABLE_THINKING"] = value
    return prior


def test_request_casual_skips_primer_even_without_no_think():
    prior = _with_disable_thinking(None)  # no-think OFF globally
    try:
        assert srv._request_skip_thinking_primer({"stream_strip": False}) is True
    finally:
        _with_disable_thinking(prior)
    print("✓ request_casual_skips_primer_even_without_no_think")


def test_request_heavy_keeps_global_default():
    prior = _with_disable_thinking(None)  # no-think OFF globally
    try:
        # stream_strip=true → not casual → fall through → no-think OFF → don't skip
        assert srv._request_skip_thinking_primer({"stream_strip": True}) is False
    finally:
        _with_disable_thinking(prior)
    print("✓ request_heavy_keeps_global_default")


def test_request_absent_field_follows_env():
    prior = _with_disable_thinking("1")  # no-think ON globally
    try:
        assert srv._request_skip_thinking_primer({}) is True
        assert srv._request_skip_thinking_primer({"stream_strip": "notabool"}) is True
    finally:
        _with_disable_thinking(prior)

    prior = _with_disable_thinking(None)  # no-think OFF globally
    try:
        assert srv._request_skip_thinking_primer({}) is False
        assert srv._request_skip_thinking_primer(None) is False
    finally:
        _with_disable_thinking(prior)
    print("✓ request_absent_field_follows_env")


# --- prepare_prompt_lm: the kwarg actually reaches the chat template ----------

class _FakeProcessor:
    """Records the kwargs apply_chat_template was called with."""
    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return "PROMPT"


def _with_fake_processor():
    fake = _FakeProcessor()
    prior = srv.processor
    srv.processor = fake
    return fake, prior


def test_prepare_prompt_skip_true_sets_enable_thinking():
    prior_dt = _with_disable_thinking(None)
    fake, prior_proc = _with_fake_processor()
    try:
        srv.prepare_prompt_lm([{"role": "user", "content": "hey"}],
                              skip_thinking_primer=True)
        assert fake.last_kwargs.get("enable_thinking") is True, fake.last_kwargs
    finally:
        srv.processor = prior_proc
        _with_disable_thinking(prior_dt)
    print("✓ prepare_prompt_skip_true_sets_enable_thinking")


def test_prepare_prompt_skip_false_omits_enable_thinking():
    prior_dt = _with_disable_thinking(None)
    fake, prior_proc = _with_fake_processor()
    try:
        srv.prepare_prompt_lm([{"role": "user", "content": "hey"}],
                              skip_thinking_primer=False)
        # legacy: when not skipping, the kwarg is simply absent (template default)
        assert "enable_thinking" not in fake.last_kwargs, fake.last_kwargs
    finally:
        srv.processor = prior_proc
        _with_disable_thinking(prior_dt)
    print("✓ prepare_prompt_skip_false_omits_enable_thinking")


def test_prepare_prompt_default_none_preserves_legacy_behavior():
    # skip_thinking_primer=None → resolves to (no-think configured?).
    # no-think OFF → no kwarg; no-think ON → enable_thinking=True. EXACT legacy.
    fake, prior_proc = _with_fake_processor()
    try:
        prior_dt = _with_disable_thinking(None)
        srv.prepare_prompt_lm([{"role": "user", "content": "hey"}])
        assert "enable_thinking" not in fake.last_kwargs
        _with_disable_thinking(prior_dt)

        prior_dt = _with_disable_thinking("1")
        srv.prepare_prompt_lm([{"role": "user", "content": "hey"}])
        assert fake.last_kwargs.get("enable_thinking") is True
        _with_disable_thinking(prior_dt)
    finally:
        srv.processor = prior_proc
    print("✓ prepare_prompt_default_none_preserves_legacy_behavior")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    sys.exit(main())
