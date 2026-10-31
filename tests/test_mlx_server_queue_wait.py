"""Per-request admission-queue wait in the server log (2026-10-31).

Background: h-uman replies took p50 5.1 s / p90 14.5 s for ~9 output tokens.
The logged elapsed time is measured from do_POST's t0, before
admission_queue.submit_and_wait, so it mixes queue wait and generation, and
the log alone could not say which. These tests pin the ' qwait=<s>s' tag:
  * _queue_wait_tag formats started - t0, clamps at 0, and returns '' on bad
    input (logging must never fail a request);
  * _do_chat_completion stamps the tag before generating, so a request that
    waited shows that wait on its log line;
  * all three chat log lines carry it.
Model-free: no weights load.
"""
import importlib.util
import os
import sys
import time

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
_SERVER_PATH = os.path.join(_SCRIPTS, "mlx-server.py")


def _load_server():
    spec = importlib.util.spec_from_file_location("mlx_server_queue_wait_under_test", _SERVER_PATH)
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    return srv


@pytest.fixture(scope="module")
def srv():
    return _load_server()


@pytest.mark.parametrize("t0,started,expected", [
    (100.0, 100.0, " qwait=0.00s"),
    (100.0, 112.346, " qwait=12.35s"),
    (100.0, 100.004, " qwait=0.00s"),
    (100.0, 99.0, " qwait=0.00s"),      # clock stepped back: clamp, never negative
])
def test_queue_wait_tag_formats_wait(srv, t0, started, expected):
    assert srv._queue_wait_tag(t0, started) == expected


@pytest.mark.parametrize("t0,started", [(None, 1.0), ("x", 1.0), (1.0, object())])
def test_queue_wait_tag_bad_input_is_empty(srv, t0, started):
    assert srv._queue_wait_tag(t0, started) == ""


def test_do_chat_completion_stamps_wait_before_generating(srv, monkeypatch):
    """A request that sat 7 s in the queue logs qwait=7.00s. The stamp is
    taken when the worker STARTS the request, not when it finishes."""
    seen = {}

    def fake_non_stream(self, req, resp_id, t0):
        seen["tag_during_generation"] = self._qwait_tag()

    monkeypatch.setattr(srv.ChatHandler, "_handle_non_stream", fake_non_stream, raising=True)
    monkeypatch.setattr(srv.time, "time", lambda: 1007.0)
    handler = srv.ChatHandler.__new__(srv.ChatHandler)   # no socket needed
    handler.headers = {}
    srv.ChatHandler._do_chat_completion(handler, {"messages": []}, "chatcmpl-test", 1000.0)

    assert seen["tag_during_generation"] == " qwait=7.00s"


def test_qwait_tag_empty_before_worker_ran(srv):
    handler = srv.ChatHandler.__new__(srv.ChatHandler)
    assert handler._qwait_tag() == ""


def test_every_chat_log_line_carries_qwait():
    src = open(_SERVER_PATH, encoding="utf-8").read()
    lines = [ln for ln in src.splitlines() if "{self._purpose_tag()}" in ln and "{preview}" in ln]
    assert len(lines) == 3, lines
    for ln in lines:
        assert "{self._purpose_tag()}{self._qwait_tag()}" in ln, ln
