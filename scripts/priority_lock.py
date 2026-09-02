"""PriorityLock — a re-entrant mutex where "live" waiters are admitted ahead of
"batch" waiters. Task 10 of h-uman's 2026-09-01 plan: the serving queue was
99.8% internal machinery (judges, arena, proposer); a real reply had to wait
behind it. Admission order is the ONLY thing this changes — the critical
section is still fully serialized, exactly as with the plain Lock it replaces.

Re-entrant so the whole request handler can be wrapped once at the top while
the existing inner `with model_lock:` sites keep working unchanged.

Task 10 REOPENED (2026-09-02): PriorityLock alone was shipped once behind
ThreadingMixIn (commit dc154d5) and caused a SIGSEGV crash loop (891e1b0,
"LastExitStatus 11" x36 in one night). PriorityLock's mutual exclusion was
never the bug — the crash was *different* per-connection accept threads each
taking a turn calling into mx/Metal. That is a thread-AFFINITY violation, not
a locking bug: a correct mutex proves only one caller is inside at a time, it
proves nothing about which THREAD that caller is. AdmissionQueue below is the
fix: exactly one persistent worker thread, created once, ever executes a job.
Accept/connection threads may read sockets and enqueue, but the model is only
ever touched by that one thread for the life of the process.
"""
import heapq
import itertools
import threading
from contextlib import contextmanager

LIVE = "live"
BATCH = "batch"


class PriorityLock:
    def __init__(self):
        self._cv = threading.Condition()
        self._owner = None
        self._depth = 0
        self.live_waiting = 0
        self.batch_waiting = 0

    def acquire(self, priority=BATCH):
        me = threading.get_ident()
        with self._cv:
            if self._owner == me:
                self._depth += 1
                return
            if priority == LIVE:
                self.live_waiting += 1
                try:
                    while self._owner is not None:
                        self._cv.wait()
                finally:
                    self.live_waiting -= 1
            else:
                self.batch_waiting += 1
                try:
                    # A batch waiter yields while any live waiter is queued.
                    while self._owner is not None or self.live_waiting > 0:
                        self._cv.wait()
                finally:
                    self.batch_waiting -= 1
            self._owner = me
            self._depth = 1

    def release(self):
        with self._cv:
            if self._owner != threading.get_ident():
                raise RuntimeError("PriorityLock released by non-owner")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._cv.notify_all()

    # `with lock:` keeps the plain-Lock semantics (batch priority).
    def __enter__(self):
        self.acquire(BATCH)
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    @contextmanager
    def held(self, priority):
        self.acquire(LIVE if priority == LIVE else BATCH)
        try:
            yield self
        finally:
            self.release()

    def snapshot(self):
        with self._cv:
            return {"live_waiting": self.live_waiting, "batch_waiting": self.batch_waiting,
                    "held": self._owner is not None}


def parse_priority_header(raw_value):
    """Map a raw X-HU-Priority header value to an admission priority.

    "batch" (case-insensitive, trimmed) -> BATCH. Everything else — missing,
    empty, "interactive", or any unrecognized value/typo -> LIVE.

    Unknown values fail open toward LIVE (interactive), not BATCH: a typo in
    a batch caller's header must not let it cut ahead of real interactive
    traffic, but a typo in an interactive caller's header (or simply no
    header, the daemon's default) must never get stuck behind batch work.
    """
    return BATCH if (raw_value or "").strip().lower() == "batch" else LIVE


class QueueFull(Exception):
    """Raised by AdmissionQueue.submit_and_wait when the request's priority
    class is already at its admission cap. Raised BEFORE the job is queued —
    the model is never touched for a rejected request."""


def queue_full_http_response(exc, retry_after_seconds=2):
    """Shape a QueueFull into the (status, body, headers) triple the HTTP
    layer sends back. Lives here — not in mlx-server.py — so the exact
    production shaping is importable and testable without importing mlx."""
    return (
        503,
        {"error": "server busy", "detail": str(exc)},
        {"Retry-After": str(retry_after_seconds)},
    )


class AdmissionQueue:
    """A bounded priority job queue in front of exactly one dedicated worker
    thread.

    Why this exists in addition to PriorityLock: PriorityLock proves mutual
    exclusion (only one caller inside at a time) but says nothing about
    thread IDENTITY. mlx-server's 2026-09-02 SIGSEGV-under-ThreadingMixIn
    crash loop happened despite PriorityLock serializing correctly — the
    crash was different accept-per-connection threads each taking a turn
    inside mx/Metal. AdmissionQueue fixes the actual hazard: the worker
    thread is created ONCE in __init__ and is the ONLY thread that ever
    invokes a submitted job, for the lifetime of the queue. Callers
    (HTTP accept threads) may read sockets, parse JSON, and block waiting —
    they must never call into the model themselves.

    It also bounds the backlog per priority class, so a runaway batch
    producer (reindexer, eval harness) cannot wedge interactive traffic
    behind an unbounded queue (see the 2026-07-25 retry-amplification doom
    loop). submit_and_wait raises QueueFull synchronously, before the job is
    ever enqueued, once a class is at capacity — the caller turns that into a
    503 + Retry-After (see queue_full_http_response) without the model ever
    being touched.

    Admission order: LIVE jobs run before any BATCH job still queued behind
    them; FIFO within a class (a monotonic sequence counter breaks heap
    ties). A LIVE job that arrives while a BATCH job is already EXECUTING
    still waits for that one job to finish — this queue governs admission
    order for waiting work, not preemption of a job already running.
    """

    def __init__(self, batch_cap=4, live_cap=64, name="mlx-admission-worker"):
        self._cv = threading.Condition()
        self._heap = []
        self._seq = itertools.count()
        self._batch_cap = batch_cap
        self._live_cap = live_cap
        self._live_depth = 0
        self._batch_depth = 0
        self._active = None
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name=name, daemon=True)
        self._worker.start()

    def can_admit(self, priority):
        """Best-effort pre-check so an HTTP handler can reject a request
        (503) BEFORE reading a possibly-large body. submit_and_wait
        re-checks the same caps atomically under the lock, so a race here
        only costs a wasted body read — it can never let an over-cap
        request slip through, nor wrongly reject one that would fit."""
        with self._cv:
            if priority == BATCH:
                return self._batch_depth < self._batch_cap
            return self._live_depth < self._live_cap

    def submit_and_wait(self, priority, run, name="job"):
        """Enqueue `run` (a zero-arg callable) and block the calling thread
        until the single worker thread has executed it. Raises QueueFull
        immediately, without enqueuing anything, if `priority`'s class is at
        capacity. Any exception `run` itself raises is captured on the
        worker thread and re-raised here, on the CALLING thread, so normal
        try/except at the call site keeps working."""
        rank = 0 if priority == LIVE else 1
        done = threading.Event()
        box = {}

        def _wrapped():
            try:
                run()
            except BaseException as exc:  # noqa: BLE001 — re-raised below, on the caller's thread
                box["exc"] = exc
            finally:
                done.set()

        with self._cv:
            if self._closed:
                raise RuntimeError("admission queue is shut down")
            if priority == BATCH:
                if self._batch_depth >= self._batch_cap:
                    raise QueueFull(f"batch queue at capacity ({self._batch_cap})")
                self._batch_depth += 1
            else:
                if self._live_depth >= self._live_cap:
                    raise QueueFull(f"live queue at capacity ({self._live_cap})")
                self._live_depth += 1
            heapq.heappush(self._heap, (rank, next(self._seq), priority, name, _wrapped))
            self._cv.notify()

        done.wait()
        if "exc" in box:
            raise box["exc"]

    def snapshot(self):
        with self._cv:
            return {"live_waiting": self._live_depth, "batch_waiting": self._batch_depth,
                    "active": self._active}

    def shutdown(self, timeout=2):
        """Stop the worker thread once the backlog drains. Not needed in
        normal server operation (the worker is a daemon thread); provided so
        tests can tear down cleanly instead of leaking a parked thread per
        test."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=timeout)

    def _worker_loop(self):
        while True:
            with self._cv:
                while not self._heap and not self._closed:
                    self._cv.wait()
                if not self._heap and self._closed:
                    return
                rank, seq, priority, name, fn = heapq.heappop(self._heap)
                self._active = name
            try:
                fn()
            finally:
                with self._cv:
                    if priority == BATCH:
                        self._batch_depth -= 1
                    else:
                        self._live_depth -= 1
                    self._active = None
