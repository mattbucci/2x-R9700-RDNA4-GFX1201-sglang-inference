"""Behaviour check for bounded_work_queue. Usage: check_bounded_work_queue.py <module path>."""
import importlib.util
import sys
import threading
import time


def load(path):
    spec = importlib.util.spec_from_file_location("module_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def spawn(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def main():
    m = load(sys.argv[1])
    q = m.BoundedWorkQueue(3)
    assert q.capacity == 3 and q.empty() and not q.full() and len(q) == 0
    for i in range(3):
        q.put(i)
    assert q.full() and len(q) == 3
    try:
        q.put_nowait(99)
        raise SystemExit("put on a full queue succeeded")
    except m.QueueFull:
        pass
    start = time.monotonic()
    try:
        q.put(99, timeout=0.2)
        raise SystemExit("put on a full queue succeeded")
    except m.QueueFull:
        assert 0.15 <= time.monotonic() - start < 2.0
    assert [q.get(), q.get_nowait(), q.get()] == [0, 1, 2]
    try:
        q.get(timeout=0.1)
        raise SystemExit("get on an empty queue succeeded")
    except m.QueueEmpty:
        pass
    try:
        m.BoundedWorkQueue(0)
        raise SystemExit("zero capacity accepted")
    except ValueError:
        pass

    # a blocked producer is released by a consumer
    q = m.BoundedWorkQueue(1)
    q.put("a")
    done = []
    t = spawn(lambda: (q.put("b", timeout=5), done.append(True)))
    time.sleep(0.2)
    assert q.get(timeout=1) == "a"
    t.join(2)
    assert done == [True] and q.get(timeout=1) == "b"

    # a blocked consumer is released by a producer
    got = []
    t = spawn(lambda: got.append(q.get(timeout=5)))
    time.sleep(0.2)
    q.put("c")
    t.join(2)
    assert got == ["c"]

    # batches, task tracking and drain
    q = m.BoundedWorkQueue(5)
    for i in range(5):
        q.put(i)
    assert q.get_batch(3, timeout=1) == [0, 1, 2]
    assert q.join(timeout=0.1) is False
    for _ in range(3):
        q.task_done()
    assert q.drain() == [3, 4] and q.join(timeout=1) is True
    try:
        q.task_done()
        raise SystemExit("extra task_done accepted")
    except ValueError:
        pass

    # close: producers refused, consumers drain and then stop
    q = m.BoundedWorkQueue(4)
    q.put(1)
    q.put(2)
    q.close()
    q.close()
    assert q.closed
    try:
        q.put(3, timeout=1)
        raise SystemExit("put on a closed queue succeeded")
    except m.QueueClosed:
        pass
    assert list(q) == [1, 2]
    try:
        q.get(timeout=1)
        raise SystemExit("get on a closed empty queue succeeded")
    except m.QueueClosed:
        pass

    # close releases a producer blocked on a full queue
    q = m.BoundedWorkQueue(1)
    q.put("x")
    outcome = []

    def producer():
        try:
            q.put("y", timeout=6)
            outcome.append("stored")
        except m.QueueClosed:
            outcome.append("closed")
        except m.QueueFull:
            outcome.append("full")

    t = spawn(producer)
    time.sleep(0.2)
    q.close()
    t.join(2)
    assert outcome == ["closed"], outcome

    # close releases every consumer blocked on an empty queue
    q = m.BoundedWorkQueue(2)
    results = []
    lock = threading.Lock()

    def consumer():
        try:
            q.get(timeout=6)
            res = "item"
        except m.QueueClosed:
            res = "closed"
        except m.QueueEmpty:
            res = "empty"
        with lock:
            results.append(res)

    threads = [spawn(consumer) for _ in range(4)]
    time.sleep(0.3)
    q.close()
    deadline = time.monotonic() + 2
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    with lock:
        assert results == ["closed"] * 4, results
    print("ok")


if __name__ == "__main__":
    main()
