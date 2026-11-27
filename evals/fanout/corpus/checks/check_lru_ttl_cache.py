import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


clock = Clock()
c = m.LruTtlCache(2, 10.0, clock=clock)
c.put("a", 1)
c.put("b", 2)
assert c.get("a") == 1
assert len(c) == 2
c.put("c", 3)  # evicts b, the least recently used
assert c.get("b") is None
assert c.keys() == ["a", "c"]
assert c.stats.evictions == 1

# expiry boundary
clock.t = 9.999
assert "a" in c
clock.t = 10.0
assert "a" not in c and c.get("a", "gone") == "gone"
assert c.stats.expirations >= 1
assert len(c) == 0

# per-entry ttl, touch, ttl_remaining, peek
c.clear()
c.put("x", 1, ttl=5)
c.put("y", 2, ttl=50)
assert c.ttl_remaining("x") == 5
clock.t = 14.0
assert c.touch("x", ttl=20) is True
assert c.ttl_remaining("x") == 20
clock.t = 16.0
assert c.peek("x") == 1
assert c.keys() == ["y", "x"]
assert c.peek("y") == 2 and c.keys() == ["y", "x"]

# updating an existing key in a full cache must not evict anything
d = m.LruTtlCache(2, 100.0, clock=clock)
d.put("a", 1)
d.put("b", 2)
d.put("b", 22)
assert d.peek("a") == 1, "update of an existing key evicted another entry"
assert d.peek("b") == 22
assert d.stats.evictions == 0
assert d.keys() == ["a", "b"]

# a full cache with an expired entry frees it instead of evicting a live one
e = m.LruTtlCache(2, 100.0, clock=clock)
e.put("live", 1)
e.put("short", 2, ttl=1)
clock.t += 2
e.put("new", 3)
assert e.keys() == ["live", "new"] and e.stats.evictions == 0

# get_or_compute, delete, purge
calls = []
assert e.get_or_compute("k", lambda: calls.append(1) or 7) == 7
assert e.get_or_compute("k", lambda: calls.append(1) or 8) == 7
assert len(calls) == 1
assert e.delete("k") is True and e.delete("k") is False
e.put("z", 1, ttl=1)
clock.t += 5
assert e.purge_expired() == 1
for bad in (lambda: m.LruTtlCache(0, 1), lambda: m.LruTtlCache(1, 0), lambda: e.put("q", 1, ttl=0)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
print("ok")
