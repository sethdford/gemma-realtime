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


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "-3"])
def test_bad_values_ignored(srv, raw):
    mx = FakeMx()
    assert srv._apply_mlx_memory_limits(mx, {"MLX_CACHE_LIMIT_GB": raw}) == {}
    assert mx.calls == []


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
