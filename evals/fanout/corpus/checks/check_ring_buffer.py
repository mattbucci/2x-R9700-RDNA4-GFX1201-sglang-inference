import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])

R = m.RingBuffer

r = R(3)
assert r.is_empty() and not r.is_full() and r.capacity == 3
assert r.push(1) is False and r.push(2) is False and r.push(3) is False
assert r.is_full() and list(r) == [1, 2, 3]
assert r.push(4) is True
assert list(r) == [2, 3, 4] and r[0] == 2 and r[-1] == 4 and r[1] == 3
assert r.oldest() == 2 and r.newest() == 4
assert r.extend([5, 6]) == 2 and r.to_list() == [4, 5, 6]
assert r.last(2) == [5, 6] and r.last(10) == [4, 5, 6] and r.last(0) == []
assert 5 in r and 1 not in r

assert r.popleft() == 4
assert list(r) == [5, 6] and len(r) == 2
r.push(7)
assert list(r) == [5, 6, 7]

# pop removes the newest item
p = R(5)
p.extend([1, 2, 3])
got = p.pop()
assert got == 3, f"pop returned {got!r}, expected 3"
assert list(p) == [1, 2] and len(p) == 2
p.push(9)
assert list(p) == [1, 2, 9]
full = R(3)
full.extend([1, 2, 3, 4])
assert full.pop() == 4 and full.pop() == 3 and full.to_list() == [2]
assert full.pop() == 2 and full.is_empty()

for fn in (full.pop, full.popleft, lambda: full[0], lambda: r[3], lambda: r[-4]):
    try:
        fn()
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError")

s = R(2, overwrite=False)
s.push("a")
s.push("b")
try:
    s.push("c")
except m.BufferFullError:
    pass
else:
    raise AssertionError("expected BufferFullError")
assert s.to_list() == ["a", "b"]
s[0] = "z"
assert s.to_list() == ["z", "b"]

d = R(4)
d.extend(range(6))
assert d.drop_oldest(3) == 3 and d.to_list() == [5]
assert d.drop_oldest(5) == 1 and d.is_empty()
d.extend([1, 2])
d.clear()
assert len(d) == 0 and d.to_list() == []
try:
    R(0)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError")
print("ok")
