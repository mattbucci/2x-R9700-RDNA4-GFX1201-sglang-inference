"""Behaviour check for min_heap_scheduler. Usage: check_min_heap_scheduler.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])
import random

s = m.Scheduler()
assert s.peek() is None and len(s) == 0
for tid, pri in [("a", 5), ("b", 1), ("c", 3), ("d", 1)]:
    s.schedule(tid, pri, payload=tid.upper())
assert len(s) == 4 and "a" in s and "z" not in s
assert s.peek().task_id == "b"
assert s.pending_ids() == ["b", "d", "c", "a"]
assert s.priority_of("c") == 3 and s.priority_of("z") is None
try:
    s.schedule("a", 9)
except ValueError:
    pass
else:
    raise AssertionError("duplicate id accepted")
s.reschedule("a", 0)
assert s.peek().task_id == "a"
s.reschedule("a", 2)
assert s.pending_ids() == ["b", "d", "a", "c"]
assert s.cancel("d") is True and s.cancel("d") is False
assert [t.task_id for t in s.pop_due(2)] == ["b", "a"]
first = s.pop()
assert first.task_id == "c" and first.payload == "C"
try:
    s.pop()
except IndexError:
    pass
else:
    raise AssertionError("pop on empty scheduler")

# cancel a task whose replacement must move towards the root
s = m.Scheduler()
for tid, pri in [("t1", 1), ("t10", 10), ("t2", 2), ("t11", 11), ("t12", 12), ("t4", 4), ("t3", 3)]:
    s.schedule(tid, pri)
assert s.cancel("t11") is True
order = [t.priority for t in s.drain()]
assert order == [1, 2, 3, 4, 10, 12], order

# randomised comparison with a sorted reference
rng = random.Random(7)
for _ in range(200):
    s = m.Scheduler()
    ref = {}
    for i in range(rng.randint(1, 25)):
        ref[f"k{i}"] = (rng.randint(0, 30), i)
        s.schedule(f"k{i}", ref[f"k{i}"][0])
    for _ in range(rng.randint(0, 10)):
        tid = rng.choice(sorted(ref)) if ref else None
        if tid is None:
            break
        if rng.random() < 0.5:
            assert s.cancel(tid)
            del ref[tid]
        else:
            pri = rng.randint(0, 30)
            s.reschedule(tid, pri)
            ref[tid] = (pri, ref[tid][1])
    expected = [tid for tid, _ in sorted(ref.items(), key=lambda kv: kv[1])]
    assert [t.task_id for t in s.drain()] == expected
print("ok")
