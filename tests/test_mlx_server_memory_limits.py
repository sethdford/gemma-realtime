"""MLX memory limits + /health memory telemetry (2026-10-31).

Background: the Mac reached 135 GB resident on 128 GB and was force-restarted.
MLX's defaults let a process grow past physical RAM and keep freed buffers
cached, and the GLM server set neither limit. These tests pin:
  * MLX_CACHE_LIMIT_GB / MLX_MEMORY_LIMIT_GB map to set_cache_limit /
    set_memory_limit in bytes, and unset means NOTHING is called (no
    behaviour change by default);
  * invalid or non-positive values are ignored, never raised;
  * the snapshot reads active/peak/cache and degrades to {} on error;
  * /health reports the worker's snapshot and limits without calling mx.
Model-free: a fake mx module stands in for mlx.core.
"""
import importlib.util
import os
import sys

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
GIB = 1024 ** 3


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location(
        "mlx_server_memlimit_under_test", os.path.join(_SCRIPTS, "mlx-server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeMx:
    def __init__(self, active=0, peak=0, cache=0, boom=False):
        self.calls = []
        self._vals = (active, peak, cache)
        self._boom = boom

    def set_cache_limit(self, n):
        self.calls.append(("cache", n))
        return 7 * GIB

    def set_memory_limit(self, n):
        self.calls.append(("memory", n))
        return 9 * GIB

    def get_active_memory(self):
        if self._boom:
            raise RuntimeError("no metal")
        return self._vals[0]

    def get_peak_memory(self):
        return self._vals[1]

    def get_cache_memory(self):
        return self._vals[2]


def test_unset_env_calls_nothing(srv):
    mx = FakeMx()
    assert srv._apply_mlx_memory_limits(mx, {}) == {}
    assert mx.calls == []


def test_limits_applied_in_bytes(srv):
    mx = FakeMx()
    got = srv._apply_mlx_memory_limits(mx, {"MLX_CACHE_LIMIT_GB": "4", "MLX_MEMORY_LIMIT_GB": "70.5"})
    assert got == {"cache_limit_gb": 4.0, "memory_limit_gb": 70.5}
    assert mx.calls == [("cache", 4 * GIB), ("memory", int(70.5 * GIB))]


@pytest.mark.parametrize("raw", ["", "  ", "abc", "-3"])
def test_bad_values_ignored(srv, raw):
    mx = FakeMx()
    assert srv._apply_mlx_memory_limits(mx, {"MLX_CACHE_LIMIT_GB": raw}) == {}
    assert mx.calls == []


def test_cache_limit_zero_disables_cache(srv):
    """0 is mlx's documented 'no cache' value, the main lever after a spike."""
    mx = FakeMx()
    assert srv._apply_mlx_memory_limits(mx, {"MLX_CACHE_LIMIT_GB": "0"}) == {"cache_limit_gb": 0.0}
    assert mx.calls == [("cache", 0)]


def test_memory_limit_zero_rejected_and_logged(srv, capsys):
    mx = FakeMx()
    assert srv._apply_mlx_memory_limits(mx, {"MLX_MEMORY_LIMIT_GB": "0"}) == {}
    assert mx.calls == []
    assert "ignoring MLX_MEMORY_LIMIT_GB" in capsys.readouterr().out


def test_seed_after_load_fills_snapshot(srv, monkeypatch):
    monkeypatch.setattr(srv, "_mem_snapshot", {})
    monkeypatch.setattr(srv, "_mem_limits", {})
    srv._seed_memory_after_load(FakeMx(active=56 * GIB, peak=57 * GIB, cache=1 * GIB))
    assert srv._health_memory_block()["active_gb"] == 56.0


def test_seed_warns_when_memory_limit_below_model(srv, monkeypatch, capsys):
    monkeypatch.setattr(srv, "_mem_snapshot", {})
    monkeypatch.setattr(srv, "_mem_limits", {"memory_limit_gb": 40.0})
    srv._seed_memory_after_load(FakeMx(active=56 * GIB))
    assert "WARNING MLX_MEMORY_LIMIT_GB=40" in capsys.readouterr().out


def test_seed_quiet_when_memory_limit_above_model(srv, monkeypatch, capsys):
    monkeypatch.setattr(srv, "_mem_snapshot", {})
    monkeypatch.setattr(srv, "_mem_limits", {"memory_limit_gb": 80.0})
    srv._seed_memory_after_load(FakeMx(active=56 * GIB))
    assert "WARNING" not in capsys.readouterr().out


def _src():
    return open(os.path.join(_SCRIPTS, "mlx-server.py"), encoding="utf-8").read()


def test_main_applies_limits_before_load_and_seeds_after():
    src = _src()
    body = src[src.index("\ndef main():"):]
    apply_at = body.index("_apply_mlx_memory_limits(")
    load_at = body.index("load_model(args.model")
    seed_at = body.index("_seed_memory_after_load(")
    assert apply_at < load_at < seed_at


def test_worker_refreshes_snapshot_after_each_chat():
    src = _src()
    start = src.index("def _do_chat_completion(")
    body = src[start:src.index("\ndef ", start + 1)]
    fin = body[body.index("finally:"):]
    assert "_mlx_memory_snapshot(" in fin


def test_missing_setter_is_ignored(srv):
    class NoSetters:
        pass
    assert srv._apply_mlx_memory_limits(NoSetters(), {"MLX_CACHE_LIMIT_GB": "4"}) == {}


def test_snapshot_reports_gb(srv):
    mx = FakeMx(active=int(56.25 * GIB), peak=60 * GIB, cache=int(3.5 * GIB))
    assert srv._mlx_memory_snapshot(mx) == {"active_gb": 56.25, "peak_gb": 60.0, "cache_gb": 3.5}


def test_snapshot_degrades_to_empty(srv):
    assert srv._mlx_memory_snapshot(FakeMx(boom=True)) == {}


def test_health_block_reads_cached_state_only(srv, monkeypatch):
    monkeypatch.setattr(srv, "_mem_snapshot", {"active_gb": 56.0, "peak_gb": 61.0, "cache_gb": 2.0})
    monkeypatch.setattr(srv, "_mem_limits", {"cache_limit_gb": 4.0})
    assert srv._health_memory_block() == {
        "active_gb": 56.0, "peak_gb": 61.0, "cache_gb": 2.0, "limits": {"cache_limit_gb": 4.0}}


def test_health_handler_never_calls_mx_directly():
    """The /health branch must read _health_memory_block(), not mx.get_*()."""
    src = open(os.path.join(_SCRIPTS, "mlx-server.py"), encoding="utf-8").read()
    start = src.index('if self.path == "/health":')
    end = src.index("self._send_json(200, health", start)
    block = src[start:end]
    assert "_health_memory_block()" in block
    assert "get_active_memory" not in block and "get_peak_memory" not in block
