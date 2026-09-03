"""PriorityLock — a re-entrant mutex where "live" waiters are admitted ahead of
"batch" waiters. Task 10 of h-uman's 2026-09-01 plan: the serving queue was
99.8% internal machinery (judges, arena, proposer); a real reply had to wait
behind it. Admission order is the ONLY thing this changes — the critical
section is still fully serialized, exactly as with the plain Lock it replaces.

Re-entrant so the whole request handler can be wrapped once at the top while
the existing inner `with model_lock:` sites keep working unchanged.
"""
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
