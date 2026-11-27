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
b = m.TokenBucket(rate=2.0, capacity=10.0, clock=clock)
assert b.available() == 10.0
assert b.try_acquire(4) is True and b.available() == 6.0
assert b.try_acquire(7) is False and b.available() == 6.0
clock.t = 1.0
assert b.available() == 8.0
clock.t = 100.0
assert b.available() == 10.0  # burst is capped

assert b.try_acquire(10) is True
assert b.time_until(3) == 1.5
assert b.available() == 0.0
clock.t = 101.5
assert b.try_acquire(3) is True

# reserve goes into debt and reports the wait at the refill rate
r = m.TokenBucket(rate=2.0, capacity=10.0, clock=clock)
assert r.reserve(10) == 0.0
wait = r.reserve(4)
assert abs(wait - 2.0) < 1e-9, f"reserve wait {wait}, expected 2.0"
assert r.available() == -4.0
assert r.try_acquire(1) is False
clock.t += 2.0
assert abs(r.available()) < 1e-9
r.cancel(3)
assert abs(r.available() - 3.0) < 1e-9

# acquire sleeps for the reserved wait
slept = []
q = m.TokenBucket(rate=4.0, capacity=4.0, clock=clock, initial=0.0)
assert q.acquire(2, sleep=slept.append) is True
assert len(slept) == 1 and abs(slept[0] - 0.5) < 1e-9
assert q.acquire(4, sleep=slept.append, max_wait=0.1) is False and len(slept) == 1

q.set_rate(8.0)
assert q.rate == 8.0
q.reset()
assert q.available() == 4.0

k = m.KeyedLimiter(rate=1.0, capacity=2.0, clock=clock)
assert k.allow("u") and k.allow("u") and not k.allow("u")
assert k.allow("v") and len(k) == 2
assert abs(k.retry_after("u") - 1.0) < 1e-9
clock.t += 10
assert k.prune_full() == 2 and len(k) == 0
assert k.forget("u") is False
for bad in (lambda: m.TokenBucket(0, 1), lambda: m.TokenBucket(1, 0), lambda: b.try_acquire(11), lambda: b.reserve(0)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
print("ok")
