"""Tests for AdmissionQueue (Task 10, reopened) — a bounded priority job
queue in front of exactly one dedicated worker thread.

NO mlx import, NO model load, NO port binding. Uses a FakeModel with a
sleep-based generate() to stand in for the real generation/embedding calls.
mlx-server.py itself is intentionally never imported here: it runs
`_install_mlx_lm_patches()` (which imports mlx.core) at module import time,
and mlx-server holds ~54 GB wired on :8741 — a second loader in this
process would risk exactly the crash this task exists to prevent. Every
symbol under test (AdmissionQueue, QueueFull, parse_priority_header,
queue_full_http_response) lives in priority_lock.py, which has no mlx
dependency at all.
"""
import threading
import time

import pytest
from priority_lock import (
    LIVE, BATCH,
    AdmissionQueue, QueueFull,
    parse_priority_header, queue_full_http_response,
)


class FakeModel:
    """Stand-in for the real generate()/embed() calls. Records, under a
    non-blocking lock, whether it was ever entered by two callers at once —
    a `RuntimeError` there means the test failed to enforce single-worker
    access, not that the model raised. Also records every distinct thread
    identity that ever called in, so tests can assert not just "one at a
    time" but "always the same one thread"."""

    def __init__(self, delay=0.03):
        self.delay = delay
        self._reentry_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self.max_concurrent = 0
        self.concurrent = 0
        self.threads_seen = set()
        self.calls = []

    def generate(self, tag, gate=None):
        """gate, if given, is a threading.Event: generate() blocks on it
        instead of sleeping a fixed delay, so a test can hold a job "in the
        model" for exactly as long as it needs, deterministically."""
        entered = self._reentry_lock.acquire(blocking=False)
        if not entered:
            raise RuntimeError(f"FakeModel.generate re-entered concurrently for {tag!r}")
        try:
            with self._depth_lock:
                self.concurrent += 1
                self.max_concurrent = max(self.max_concurrent, self.concurrent)
                self.threads_seen.add(threading.get_ident())
            if gate is not None:
                gate.wait(5)
            else:
                time.sleep(self.delay)
            self.calls.append(tag)
            return tag
        finally:
            with self._depth_lock:
                self.concurrent -= 1
            self._reentry_lock.release()


def _run_in_thread(fn, *args, **kwargs):
    t = threading.Thread(target=fn, args=args, kwargs=kwargs)
    t.start()
    return t


# --- (1) interactive admitted before queued batch work ---------------------

def test_interactive_admitted_before_queued_batch_jobs():
    q = AdmissionQueue(batch_cap=10, live_cap=10)
    model = FakeModel()
    order = []
    order_lock = threading.Lock()

    first_started = threading.Event()
    release_first = threading.Event()  # deterministic hold — no sleep-guessing

    def first_job():
        first_started.set()
        model.generate("first", gate=release_first)

    t_first = _run_in_thread(q.submit_and_wait, BATCH, first_job, name="first")
    assert first_started.wait(2), "first (occupying) job never started"
    time.sleep(0.02)  # make sure the worker is actually inside model.generate

    def make(tag, prio):
        def _run():
            model.generate(tag)
            with order_lock:
                order.append(tag)
        return _run

    batch_threads = [
        _run_in_thread(q.submit_and_wait, BATCH, make(f"b{i}", BATCH), f"b{i}")
        for i in range(3)
    ]
    time.sleep(0.05)  # give all three batch jobs time to queue behind "first"

    live_thread = _run_in_thread(q.submit_and_wait, LIVE, make("live", LIVE), "live")
    time.sleep(0.05)
    assert q.snapshot()["live_waiting"] == 1
    assert q.snapshot()["batch_waiting"] == 4  # "first" (running) + b0 + b1 + b2 (queued)

    release_first.set()  # NOW let "first" finish — live and b0-b2 are all queued
    t_first.join(2)
    live_thread.join(2)
    for t in batch_threads:
        t.join(2)

    assert order[0] == "live", order
    assert sorted(order[1:]) == ["b0", "b1", "b2"], order
    q.shutdown()


# --- (2) exactly one thread ever enters the model ---------------------------

def test_only_one_worker_thread_ever_touches_the_model():
    q = AdmissionQueue(batch_cap=25, live_cap=25)
    model = FakeModel(delay=0.01)

    def submit(i):
        prio = LIVE if i % 2 == 0 else BATCH
        q.submit_and_wait(prio, lambda: model.generate(f"job{i}"), name=f"job{i}")

    threads = [_run_in_thread(submit, i) for i in range(20)]
    for t in threads:
        t.join(3)

    assert model.max_concurrent == 1, "more than one job was inside the model at once"
    assert len(model.threads_seen) == 1, (
        f"model was entered from {len(model.threads_seen)} distinct threads; "
        "expected exactly one dedicated worker thread"
    )
    assert sorted(model.calls) == sorted(f"job{i}" for i in range(20))
    q.shutdown()


# --- (3) batch over-cap -> 503 + Retry-After, model never touched ----------

def test_batch_over_cap_returns_503_with_retry_after_and_skips_the_model():
    q = AdmissionQueue(batch_cap=1, live_cap=10)
    model = FakeModel(delay=0.2)

    occupying_started = threading.Event()

    def occupy():
        occupying_started.set()
        model.generate("occupying")

    t_occupy = _run_in_thread(q.submit_and_wait, BATCH, occupy, "occupying")
    assert occupying_started.wait(2)
    time.sleep(0.03)  # the occupying job now counts against batch_cap=1

    with pytest.raises(QueueFull) as exc_info:
        q.submit_and_wait(BATCH, lambda: model.generate("rejected"), name="rejected")

    code, body, headers = queue_full_http_response(exc_info.value)
    assert code == 503
    assert "Retry-After" in headers
    assert int(headers["Retry-After"]) > 0
    assert body["error"] == "server busy"

    t_occupy.join(2)
    assert "rejected" not in model.calls, "rejected batch job reached the model"
    assert model.calls == ["occupying"]

    # A LIVE submission is unaffected by the batch cap.
    q.submit_and_wait(LIVE, lambda: model.generate("live-after-reject"), name="live")
    assert "live-after-reject" in model.calls
    q.shutdown()


def test_can_admit_reflects_capacity_without_enqueuing():
    q = AdmissionQueue(batch_cap=1, live_cap=1)
    model = FakeModel(delay=0.15)
    started = threading.Event()

    t = _run_in_thread(q.submit_and_wait, BATCH, lambda: (started.set(), model.generate("x")), "x")
    assert started.wait(2)
    time.sleep(0.02)

    assert q.can_admit(BATCH) is False
    assert q.can_admit(LIVE) is True  # separate cap/class, unaffected by batch fullness

    t.join(2)
    assert q.can_admit(BATCH) is True  # slot freed once the job completed
    q.shutdown()


# --- (4) X-HU-Priority header parsing: missing -> interactive, unknown -> interactive

def test_priority_header_parsing_missing_and_unknown_default_to_interactive():
    assert parse_priority_header(None) == LIVE
    assert parse_priority_header("") == LIVE
    assert parse_priority_header("   ") == LIVE
    assert parse_priority_header("interactive") == LIVE
    assert parse_priority_header("INTERACTIVE") == LIVE
    assert parse_priority_header("urgent") == LIVE          # unknown value -> interactive
    assert parse_priority_header("live") == LIVE            # legacy header value -> interactive
    assert parse_priority_header("batch") == BATCH
    assert parse_priority_header("  Batch  ") == BATCH
    assert parse_priority_header("BATCH") == BATCH
