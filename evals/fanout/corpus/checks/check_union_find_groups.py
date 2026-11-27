"""Behaviour check for union_find_groups. Usage: check_union_find_groups.py <module path>."""
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

d = m.DisjointSets(["a", "b", "c"])
assert len(d) == 3 and d.component_count == 3
assert d.add("a") is False and d.add("d") is True
assert d.union("a", "b") is True and d.union("b", "a") is False
assert d.connected("a", "b") and not d.connected("a", "c")
assert not d.connected("a", "nope")
assert d.component_size("a") == 2 and d.component_size("c") == 1
assert d.groups() == [["a", "b"], ["c"], ["d"]]
assert d.members("b") == ["a", "b"]
assert d.largest_group() == ["a", "b"]
assert d.size_histogram() == {2: 1, 1: 2}
assert not d.is_single_group()
try:
    d.find("nope")
except KeyError:
    pass
else:
    raise AssertionError("unknown item accepted")

# merging two pairs through their non-representative members
e = m.DisjointSets()
assert e.union_all([(1, 2), (3, 4)]) == 2
assert e.union(2, 4) is True
assert e.component_count == 1 and e.is_single_group()
assert e.groups() == [[1, 2, 3, 4]]
for item in (1, 2, 3, 4):
    assert e.component_size(item) == 4, (item, e.component_size(item))
assert e.size_histogram() == {4: 1}

rng = random.Random(3)
for _ in range(100):
    n = rng.randint(2, 30)
    f = m.DisjointSets(range(n))
    for _ in range(rng.randint(0, 40)):
        f.union(rng.randrange(n), rng.randrange(n))
    groups = f.groups()
    assert sum(len(g) for g in groups) == n
    assert len(groups) == f.component_count
    for g in groups:
        for item in g:
            assert f.component_size(item) == len(g)
print("ok")
