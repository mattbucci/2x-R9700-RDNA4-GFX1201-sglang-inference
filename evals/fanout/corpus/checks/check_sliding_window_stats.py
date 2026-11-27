"""Behaviour check for sliding_window_stats. Usage: check_sliding_window_stats.py <module path>."""
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
import statistics

w = m.SlidingWindow(3)
assert len(w) == 0 and w.size == 3 and not w.is_full()
for fn in (w.min, w.max, w.mean, w.median):
    try:
        fn()
    except ValueError:
        pass
    else:
        raise AssertionError("statistic on empty window")
w.push(4)
w.push(2)
assert w.values() == [4, 2] and w.mean() == 3 and w.median() == 3
assert w.min() == 2 and w.max() == 4 and w.range() == 2

# the oldest value of a full window still counts
w = m.SlidingWindow(3)
w.extend([9, 1, 2])
assert w.is_full() and w.total_pushed == 3
assert w.max() == 9, w.max()
assert w.median() == 2 and w.mean() == 4
w.push(3)
assert w.values() == [1, 2, 3] and w.max() == 3 and w.min() == 1
w = m.SlidingWindow(3)
w.extend([1, 5, 6])
assert w.min() == 1, w.min()

assert m.rolling_max([1, 3, 2, 5, 4, 1, 0], 3) == [3, 5, 5, 5, 4]
one = m.SlidingWindow(1)
one.push(7)
assert one.min() == 7 and one.max() == 7
w.reset()
assert len(w) == 0 and w.total_pushed == 0
try:
    w.push(float("nan"))
except ValueError:
    pass
else:
    raise AssertionError("NaN accepted")
try:
    m.SlidingWindow(0)
except ValueError:
    pass
else:
    raise AssertionError("zero size accepted")

rng = random.Random(11)
for _ in range(100):
    size = rng.randint(1, 8)
    w = m.SlidingWindow(size)
    seen = []
    for _ in range(40):
        v = rng.randint(-5, 5)
        seen.append(v)
        w.push(v)
        ref = seen[-size:]
        assert w.values() == ref
        assert w.min() == min(ref) and w.max() == max(ref)
        assert w.median() == statistics.median(ref)
        assert abs(w.mean() - sum(ref) / len(ref)) < 1e-12
print("ok")
