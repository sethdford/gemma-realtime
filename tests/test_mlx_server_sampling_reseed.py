"""Every request must sample from a fresh RNG state, not the thread-default one.

MLX's default random state is thread-local and every new thread starts from
the same seed. ChatHandler runs each request on a fresh ThreadingMixIn
thread, so before 2026-09-13 identical prompts returned identical text at any
temperature (three temp=5.0 calls: byte-identical), and every best-of-N /
retry / eval "sample" was the same draw. _reseed_sampling() reseeds from
entropy per generation; GEMMA_SAMPLING_RESEED=0 keeps the old determinism.
"""
import importlib.util
import os
import sys
import threading

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_server():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_sampling_reseed_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


def _draw_in_new_thread(reseed, out):
    """What a request thread sees: (reseeded?, first 4 random ints)."""
    import mlx.core as mx

    def body():
        did = reseed()
        out.append((did, mx.random.randint(0, 1_000_000, (4,)).tolist()))

    t = threading.Thread(target=body)
    t.start()
    t.join()


def test_fresh_threads_share_the_default_seed_without_reseed(monkeypatch):
    """The failure mode, pinned: two request threads draw the same sequence."""
    srv = _load_server()
    monkeypatch.setenv("GEMMA_SAMPLING_RESEED", "0")
    out = []
    _draw_in_new_thread(srv._reseed_sampling, out)
    _draw_in_new_thread(srv._reseed_sampling, out)
    assert [d for d, _ in out] == [False, False]
    assert out[0][1] == out[1][1], "MLX thread-default RNG no longer identical — revisit the reseed rationale"


def test_reseed_gives_each_request_thread_a_different_sequence(monkeypatch):
    srv = _load_server()
    monkeypatch.delenv("GEMMA_SAMPLING_RESEED", raising=False)
    out = []
    _draw_in_new_thread(srv._reseed_sampling, out)
    _draw_in_new_thread(srv._reseed_sampling, out)
    assert [d for d, _ in out] == [True, True]
    assert out[0][1] != out[1][1]


def _draws_on_spawned_thread(sampler, n=8):
    """The worker-thread shape: a flat distribution sampled n times on a
    fresh thread. Runs on the CPU device so the thread has a stream."""
    import mlx.core as mx

    out = []

    def body():
        mx.set_default_device(mx.cpu)
        with mx.stream(mx.cpu):
            logits = mx.log(mx.array([[0.2] * 100]))
            out.extend(int(sampler(logits).item()) for _ in range(n))

    t = threading.Thread(target=body)
    t.start()
    t.join()
    return out


def test_server_sampler_advances_on_a_spawned_thread():
    """The failure mode: mlx_lm's compiled sampler froze on the worker thread
    (one draw for every token). The server's own sampler must vary."""
    srv = _load_server()
    draws = _draws_on_spawned_thread(srv._make_sampler(1.0))
    assert len(draws) == 8
    assert len(set(draws)) >= 3, draws


def test_server_sampler_is_argmax_at_zero_temperature():
    import mlx.core as mx

    srv = _load_server()
    logits = mx.log(mx.array([[0.1, 0.7, 0.2]]))
    assert int(srv._make_sampler(0.0)(logits).item()) == 1
    assert int(srv._make_sampler(None)(logits).item()) == 1
    assert int(srv._make_sampler("bogus")(logits).item()) == 1


def test_generation_sites_use_the_server_sampler():
    src = open(os.path.join(_SCRIPTS, "mlx-server.py"), encoding="utf-8").read()
    assert "make_sampler(temp=temperature)" not in src
    assert src.count("= _make_sampler(temperature)") == 3  # the three lm-path generation sites


def test_both_request_handlers_reseed():
    """The call must sit on the request path, not just exist."""
    src = open(os.path.join(_SCRIPTS, "mlx-server.py"), encoding="utf-8").read()
    assert src.count("_reseed_sampling()") >= 2
    for handler in ("def _handle_non_stream", "def _handle_stream("):
        body = src[src.index(handler):][:1500]
        assert "_reseed_sampling()" in body, f"{handler} does not reseed"
