"""Behaviour check for bloom_filter. Usage: check_bloom_filter.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])
import math

assert m.optimal_num_bits(1000, 0.01) == 9586
assert m.optimal_num_hashes(9586, 1000) == 7
b = m.BloomFilter.for_capacity(1000, 0.01)
assert b.num_bits == 9586 and b.num_hashes == 7
assert b.bit_count() == 0 and b.estimated_cardinality() == 0.0
assert "x" not in b
assert b.add("x") is True and b.add("x") is False
assert "x" in b and b"x" in b
items = [f"item-{i}" for i in range(500)]
b.clear()
assert b.update(items) >= 495
assert all(item in b for item in items)
false_hits = sum(1 for i in range(5000) if f"other-{i}" in b)
assert false_hits < 100, false_hits
assert 0.0 < b.fill_ratio() < 0.5
assert 0.0 < b.false_positive_rate() < 0.01
estimate = b.estimated_cardinality()
assert abs(estimate - 500) < 25, estimate

c = m.BloomFilter(9586, 7)
c.update(f"second-{i}" for i in range(300))
u = b.union(c)
assert all(item in u for item in items) and "second-299" in u
assert u.bit_count() >= max(b.bit_count(), c.bit_count())
assert abs(u.estimated_cardinality() - 800) < 40
assert "second-1" not in b or "second-2" not in b
try:
    b.union(m.BloomFilter(64, 7))
except ValueError:
    pass
else:
    raise AssertionError("incompatible union accepted")

small = m.BloomFilter(10, 2)
small.update(range(200))
assert small.bit_count() == 10 and small.estimated_cardinality() == math.inf
clone = b.copy()
clone.clear()
assert clone.bit_count() == 0 and b.bit_count() > 0
for bad in ((0, 1), (8, 0)):
    try:
        m.BloomFilter(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError("bad geometry accepted")
print("ok")
