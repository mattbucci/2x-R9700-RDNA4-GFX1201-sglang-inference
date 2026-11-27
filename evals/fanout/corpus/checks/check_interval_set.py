import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])

S = m.IntervalSet

s = S([(5, 7), (1, 3)])
assert list(s) == [(1, 3), (5, 7)]
s.add(3, 5)  # adjacent on both sides
assert list(s) == [(1, 7)] and len(s) == 1 and s.measure() == 6
s.add(10, 12)
s.add(0, 1)
assert list(s) == [(0, 7), (10, 12)]
s.add(4, 4)
assert list(s) == [(0, 7), (10, 12)]

# membership, half-open
assert s.contains(0) and s.contains(6) and 10 in s and 11 in s
assert not s.contains(-1) and not s.contains(8) and not s.contains(9)
assert not s.contains(7), "end of a half-open interval is not a member"
assert 12 not in s
assert s.find(11) == (10, 12)
try:
    s.find(12)
except KeyError:
    pass
else:
    raise AssertionError("expected KeyError")

assert s.contains_range(0, 7) and s.contains_range(10, 12) and s.contains_range(3, 3)
assert not s.contains_range(6, 8) and not s.contains_range(6, 11)
assert s.overlaps(6, 10) and s.overlaps(9, 11) and not s.overlaps(7, 10) and not s.overlaps(12, 20)

# remove: split, trim, span
t = S([(0, 10)])
t.remove(3, 5)
assert list(t) == [(0, 3), (5, 10)]
t.remove(0, 1)
t.remove(9, 20)
assert list(t) == [(1, 3), (5, 9)]
t.remove(2, 6)
assert list(t) == [(1, 2), (6, 9)]
t.remove(2, 6)
assert list(t) == [(1, 2), (6, 9)]
t.remove(-5, 50)
assert list(t) == [] and t.measure() == 0

a = S([(0, 5), (10, 15)])
b = S([(3, 12), (20, 22)])
assert list(a.merge(b)) == [(0, 15), (20, 22)]
assert list(a.intersection(b)) == [(3, 5), (10, 12)]
assert list(a.difference(b)) == [(0, 3), (12, 15)]
assert a.gaps(-2, 17) == [(-2, 0), (5, 10), (15, 17)]
assert a.gaps(1, 4) == [] and a.gaps(6, 8) == [(6, 8)]
assert a == S([(10, 15), (0, 2), (2, 5)]) and a != b
c = a.copy()
c.clear()
assert len(c) == 0 and len(a) == 2
try:
    a.add(5, 1)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
print("ok")
