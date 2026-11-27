"""Token-bucket rate limiting with burst capacity.

A :class:`TokenBucket` accumulates tokens at ``rate`` tokens per second up to
``capacity``, which bounds the burst a caller can spend at once.  Three ways
of spending are offered:

* :meth:`TokenBucket.try_acquire` spends tokens only if enough are available
  right now and never waits;
* :meth:`TokenBucket.reserve` always spends, letting the balance go negative,
  and reports how long the caller has to wait before acting on the
  reservation;
* :meth:`TokenBucket.acquire` reserves and then sleeps for the reported time.

The clock and the sleep function are injectable.  The bucket is not
thread-safe; wrap calls in a lock when sharing it between threads.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Hashable, Optional


class TokenBucket:
    """Rate limiter refilled continuously at a fixed rate."""

    def __init__(
        self,
        rate: float,
        capacity: float,
        clock: Callable[[], float] = time.monotonic,
        initial: Optional[float] = None,
    ) -> None:
        """Create a bucket.

        Args:
            rate: tokens added per second; must be positive.
            capacity: largest balance the bucket can hold; must be positive.
            clock: callable returning the current time in seconds.
            initial: starting balance; defaults to a full bucket.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if initial is not None and not 0 <= initial <= capacity:
            raise ValueError("initial must lie between 0 and capacity")
        self._rate = float(rate)
        self._capacity = float(capacity)
        self._clock = clock
        self._tokens = self._capacity if initial is None else float(initial)
        self._last = clock()

    @property
    def rate(self) -> float:
        """Refill rate in tokens per second."""
        return self._rate

    @property
    def capacity(self) -> float:
        """Maximum balance, which is also the largest single request."""
        return self._capacity

    def _refill(self) -> None:
        """Credit the tokens earned since the last refill."""
        now = self._clock()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = now

    def _validate(self, tokens: float) -> None:
        """Reject requests that can never be satisfied."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self._capacity:
            raise ValueError("request exceeds bucket capacity")

    def available(self) -> float:
        """Current balance; negative while reservations are outstanding."""
        self._refill()
        return self._tokens

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Spend ``tokens`` if the balance covers them; never waits.

        Returns ``True`` and deducts the tokens on success, otherwise leaves
        the balance untouched and returns ``False``.
        """
        self._validate(tokens)
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def time_until(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` would be available, without spending."""
        self._validate(tokens)
        self._refill()
        deficit = tokens - self._tokens
        if deficit <= 0:
            return 0.0
        return deficit / self._rate

    def reserve(self, tokens: float = 1.0) -> float:
        """Spend ``tokens`` unconditionally and return the required wait.

        The balance may become negative.  The return value is the number of
        seconds after which the refill will have covered the reservation, or
        0.0 when the tokens were available immediately.
        """
        self._validate(tokens)
        self._refill()
        self._tokens -= tokens
        if self._tokens >= 0:
            return 0.0
        return -self._tokens / self._capacity

    def cancel(self, tokens: float = 1.0) -> None:
        """Return tokens from a reservation that will not be used."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        self._refill()
        self._tokens = min(self._capacity, self._tokens + tokens)

    def acquire(
        self,
        tokens: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        max_wait: Optional[float] = None,
    ) -> bool:
        """Reserve ``tokens`` and sleep until the reservation is covered.

        When ``max_wait`` is given and the required wait is longer, nothing
        is spent and ``False`` is returned.
        """
        if max_wait is not None and self.time_until(tokens) > max_wait:
            return False
        wait = self.reserve(tokens)
        if wait > 0:
            sleep(wait)
        return True

    def set_rate(self, rate: float) -> None:
        """Change the refill rate; tokens already earned are kept."""
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._refill()
        self._rate = float(rate)

    def reset(self) -> None:
        """Refill the bucket completely and forget outstanding reservations."""
        self._tokens = self._capacity
        self._last = self._clock()


class KeyedLimiter:
    """Independent token buckets created on demand, one per key."""

    def __init__(
        self,
        rate: float,
        capacity: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter whose buckets all share ``rate`` and ``capacity``."""
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self._rate = rate
        self._capacity = capacity
        self._clock = clock
        self._buckets: Dict[Hashable, TokenBucket] = {}

    def __len__(self) -> int:
        """Number of keys that currently own a bucket."""
        return len(self._buckets)

    def bucket(self, key: Hashable) -> TokenBucket:
        """Return the bucket for ``key``, creating a full one if needed."""
        found = self._buckets.get(key)
        if found is None:
            found = TokenBucket(self._rate, self._capacity, clock=self._clock)
            self._buckets[key] = found
        return found

    def allow(self, key: Hashable, tokens: float = 1.0) -> bool:
        """Whether ``key`` may spend ``tokens`` now; spends them if so."""
        return self.bucket(key).try_acquire(tokens)

    def retry_after(self, key: Hashable, tokens: float = 1.0) -> float:
        """Seconds until ``key`` could spend ``tokens``."""
        return self.bucket(key).time_until(tokens)

    def forget(self, key: Hashable) -> bool:
        """Drop the bucket for ``key``; return whether one existed."""
        return self._buckets.pop(key, None) is not None

    def prune_full(self) -> int:
        """Drop buckets that have refilled completely; return the count."""
        idle = [key for key, found in self._buckets.items() if found.available() >= found.capacity]
        for key in idle:
            del self._buckets[key]
        return len(idle)
