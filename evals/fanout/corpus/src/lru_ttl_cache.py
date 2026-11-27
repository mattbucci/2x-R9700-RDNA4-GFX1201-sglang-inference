"""Least-recently-used cache with a per-entry time to live.

The cache holds at most ``capacity`` entries.  Every entry carries its own
expiry instant, computed from the cache clock when the entry is written.  An
entry is expired once the clock has reached its expiry instant; expired
entries are never returned and are dropped lazily when they are touched, or
eagerly through :meth:`LruTtlCache.purge_expired`.

Recency is updated by successful reads (:meth:`LruTtlCache.get`), by writes
and by :meth:`LruTtlCache.touch`.  :meth:`LruTtlCache.peek` and the
``in`` operator leave the recency order alone.

The clock is injectable so that tests and simulations can drive time
explicitly; it must return seconds as a monotonically non-decreasing float.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic, Hashable, List, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_MISSING: Any = object()


@dataclass
class _Entry(Generic[V]):
    """A stored value together with the instant at which it expires."""

    value: V
    expires_at: float


@dataclass
class CacheStats:
    """Counters describing how a cache has been used."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def lookups(self) -> int:
        """Total number of counted lookups."""
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        """Fraction of lookups that were hits, or 0.0 before any lookup."""
        if self.lookups == 0:
            return 0.0
        return self.hits / self.lookups


class LruTtlCache(Generic[K, V]):
    """Bounded mapping with least-recently-used eviction and entry expiry."""

    def __init__(
        self,
        capacity: int,
        default_ttl: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a cache.

        Args:
            capacity: maximum number of entries, at least 1.
            default_ttl: lifetime in seconds for entries written without an
                explicit ``ttl``; must be positive.
            clock: callable returning the current time in seconds.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self._capacity = capacity
        self._default_ttl = float(default_ttl)
        self._clock = clock
        self._entries: "OrderedDict[K, _Entry[V]]" = OrderedDict()
        self._stats = CacheStats()

    @property
    def capacity(self) -> int:
        """Maximum number of entries the cache holds."""
        return self._capacity

    @property
    def stats(self) -> CacheStats:
        """Usage counters; the returned object is updated in place."""
        return self._stats

    def __len__(self) -> int:
        """Number of entries that have not expired."""
        now = self._clock()
        return sum(1 for entry in self._entries.values() if not self._expired(entry, now))

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` is present and not expired; recency is unchanged."""
        entry = self._entries.get(key)  # type: ignore[call-overload]
        return entry is not None and not self._expired(entry, self._clock())

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Return the value for ``key`` and mark it most recently used.

        A missing or expired key returns ``default`` and counts as a miss.
        """
        entry = self._entries.get(key)
        if entry is None:
            self._stats.misses += 1
            return default
        if self._expired(entry, self._clock()):
            del self._entries[key]
            self._stats.expirations += 1
            self._stats.misses += 1
            return default
        self._entries.move_to_end(key)
        self._stats.hits += 1
        return entry.value

    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Return the value for ``key`` without touching recency or counters."""
        entry = self._entries.get(key)
        if entry is None or self._expired(entry, self._clock()):
            return default
        return entry.value

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Store ``value`` under ``key`` and mark it most recently used.

        Writing an existing key replaces its value and restarts its lifetime
        without affecting any other entry.  Writing a new key into a full
        cache first drops expired entries and, if the cache is still full,
        evicts the least recently used one.
        """
        lifetime = self._lifetime(ttl)
        now = self._clock()
        if len(self._entries) >= self._capacity:
            self._make_room(now)
        self._entries[key] = _Entry(value, now + lifetime)
        self._entries.move_to_end(key)

    def get_or_compute(self, key: K, factory: Callable[[], V], ttl: Optional[float] = None) -> V:
        """Return the cached value for ``key``, computing and storing it on a miss."""
        value = self.get(key, _MISSING)
        if value is _MISSING:
            value = factory()
            self.put(key, value, ttl)
        return value

    def touch(self, key: K, ttl: Optional[float] = None) -> bool:
        """Restart the lifetime of a live entry and mark it most recently used.

        Returns ``False`` when the key is missing or already expired.
        """
        lifetime = self._lifetime(ttl)
        now = self._clock()
        entry = self._entries.get(key)
        if entry is None:
            return False
        if self._expired(entry, now):
            del self._entries[key]
            self._stats.expirations += 1
            return False
        entry.expires_at = now + lifetime
        self._entries.move_to_end(key)
        return True

    def ttl_remaining(self, key: K) -> Optional[float]:
        """Seconds until ``key`` expires, or ``None`` if it is not live."""
        entry = self._entries.get(key)
        now = self._clock()
        if entry is None or self._expired(entry, now):
            return None
        return entry.expires_at - now

    def delete(self, key: K) -> bool:
        """Remove ``key``; return whether a live entry was removed."""
        entry = self._entries.pop(key, None)
        return entry is not None and not self._expired(entry, self._clock())

    def clear(self) -> None:
        """Remove every entry; counters are kept."""
        self._entries.clear()

    def keys(self) -> List[K]:
        """Live keys ordered from least to most recently used."""
        now = self._clock()
        return [key for key, entry in self._entries.items() if not self._expired(entry, now)]

    def purge_expired(self) -> int:
        """Drop every expired entry and return how many were dropped."""
        return self._purge(self._clock())

    def _lifetime(self, ttl: Optional[float]) -> float:
        """Resolve an optional per-entry ttl against the default."""
        if ttl is None:
            return self._default_ttl
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        return float(ttl)

    @staticmethod
    def _expired(entry: _Entry[V], now: float) -> bool:
        """Whether ``entry`` has reached its expiry instant."""
        return now >= entry.expires_at

    def _purge(self, now: float) -> int:
        """Drop entries expired at ``now``; return the number dropped."""
        stale = [key for key, entry in self._entries.items() if self._expired(entry, now)]
        for key in stale:
            del self._entries[key]
        self._stats.expirations += len(stale)
        return len(stale)

    def _make_room(self, now: float) -> None:
        """Free one slot, preferring expired entries over a live eviction."""
        self._purge(now)
        if len(self._entries) >= self._capacity:
            self._entries.popitem(last=False)
            self._stats.evictions += 1
