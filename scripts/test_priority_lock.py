import threading, time
from priority_lock import PriorityLock, LIVE, BATCH

def test_live_waiter_admitted_before_queued_batch_waiters():
    lock = PriorityLock(); order = []
    lock.acquire(BATCH)                     # main thread holds it
    def batch(i):
        with lock.held(BATCH): order.append(f"b{i}")
    def live():
        with lock.held(LIVE): order.append("live")
    bs = [threading.Thread(target=batch, args=(i,)) for i in range(5)]
    for t in bs: t.start()
    time.sleep(0.05)                        # all five batch waiters queued
    lt = threading.Thread(target=live); lt.start(); time.sleep(0.05)
    assert lock.snapshot()["live_waiting"] == 1
    lock.release()                          # who gets it next?
    lt.join(2); [t.join(2) for t in bs]
    assert order[0] == "live", order
    assert sorted(order[1:]) == ["b0","b1","b2","b3","b4"]

def test_reentrant_for_owner_and_plain_with_still_works():
    lock = PriorityLock()
    with lock:
        with lock.held(LIVE):
            with lock: pass
    assert lock.snapshot() == {"live_waiting": 0, "batch_waiting": 0, "held": False}

def test_release_by_non_owner_raises():
    lock = PriorityLock(); lock.acquire(LIVE)
    err = []
    def other():
        try: lock.release()
        except RuntimeError as e: err.append(e)
    t = threading.Thread(target=other); t.start(); t.join(2)
    assert err; lock.release()
