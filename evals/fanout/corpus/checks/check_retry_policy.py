import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    return module


m = load(sys.argv[1])

P = m.RetryPolicy

p = P(base_delay=1.0, multiplier=2.0, max_delay=100.0, max_attempts=5)
assert [p.nominal_delay(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]
assert p.schedule() == [1.0, 2.0, 4.0, 8.0]
assert P(base_delay=1.0, max_delay=3.0, max_attempts=5).schedule() == [1.0, 2.0, 3.0, 3.0]
assert p.worst_case_total() == 15.0
assert P(base_delay=0.0, max_attempts=3).schedule() == [0.0, 0.0]

# jitter stays inside its bounds and under the cap
j = P(base_delay=1.0, multiplier=2.0, max_delay=4.0, jitter=0.5, max_attempts=8)
assert j.delay(1, rng=lambda: 0.0) == 0.5
assert j.delay(1, rng=lambda: 0.5) == 1.0
assert j.delay_bounds(1) == (0.5, 1.5)
assert j.delay_bounds(5) == (4.0, 4.0)
for attempt in range(1, 8):
    low, high = j.delay_bounds(attempt)
    for u in (0.0, 0.25, 0.5, 0.75, 0.999999):
        d = j.delay(attempt, rng=lambda: u)
        assert d <= j.max_delay + 1e-12, f"attempt {attempt}: delay {d} exceeds max_delay"
        assert low - 1e-9 <= d <= high + 1e-9, f"attempt {attempt}: {d} outside {low}..{high}"


class World:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def flaky(fail_times):
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] <= fail_times:
            raise OSError("boom %d" % state["n"])
        return state["n"]

    return fn


w = World()
seen = []
r = m.Retrier(p, sleep=w.sleep, clock=w.clock, on_retry=lambda a, e, s: seen.append((a, s)))
assert r.call(flaky(2)) == 3
assert w.sleeps == [1.0, 2.0] and r.last_attempts == 3 and seen == [(1, 1.0), (2, 2.0)]

w = World()
r = m.Retrier(p, sleep=w.sleep, clock=w.clock)
try:
    r.call(flaky(99))
except m.RetryError as err:
    assert err.attempts == 5 and err.reason == "attempts exhausted"
    assert isinstance(err.last_exception, OSError) and err.__cause__ is err.last_exception
else:
    raise AssertionError("expected RetryError")
assert w.sleeps == [1.0, 2.0, 4.0, 8.0]

w = World()
dl = P(base_delay=1.0, multiplier=2.0, max_delay=100.0, max_attempts=10, deadline=5.0)
r = m.Retrier(dl, sleep=w.sleep, clock=w.clock)
try:
    r.call(flaky(99))
except m.RetryError as err:
    assert err.attempts == 3 and err.reason == "deadline exceeded"
else:
    raise AssertionError("expected RetryError")
assert w.sleeps == [1.0, 2.0]

w = World()
r = m.Retrier(p, sleep=w.sleep, clock=w.clock, retry_on=(KeyError,))
try:
    r.call(flaky(1))
except OSError:
    pass
else:
    raise AssertionError("expected OSError to propagate")
assert w.sleeps == [] and r.wrap(lambda x: x + 1)(1) == 2
for kw in ({"jitter": 1.0}, {"multiplier": 0.5}, {"max_attempts": 0}, {"deadline": 0}):
    try:
        P(**kw)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
print("ok")
