"""Rolling statistics over the most recent values of a stream.

:class:`SlidingWindow` keeps the last ``size`` values that were pushed and
reports their minimum, maximum, mean and median.

* Minimum and maximum are answered in ``O(1)`` from two monotonic deques.
  Each deque stores ``(sequence number, value)`` pairs; values that can never
  become the extreme again are discarded as soon as a newer value dominates
  them, and entries that have left the window are dropped from the front.
* The median comes from a list that is kept sorted with ``bisect``.
* The mean is recomputed with ``math.fsum`` so that long streams do not
  accumulate rounding error.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from typing import Deque, Iterable, List, Tuple


class _MonotonicDeque:
    """Deque of ``(seq, value)`` pairs whose values are monotonic.

    With ``keep_max`` the values decrease from front to back, so the front is
    the maximum of the live entries; otherwise they increase and the front is
    the minimum.
    """

    def __init__(self, keep_max: bool) -> None:
        self._keep_max = keep_max
        self._entries: Deque[Tuple[int, float]] = deque()

    def push(self, seq: int, value: float) -> None:
        """Append a value, discarding older entries that it dominates."""
        entries = self._entries
        if self._keep_max:
            while entries and entries[-1][1] <= value:
                entries.pop()
        else:
            while entries and entries[-1][1] >= value:
                entries.pop()
        entries.append((seq, value))

    def expire(self, oldest: int) -> None:
        """Drop entries that precede sequence number ``oldest``."""
        entries = self._entries
        while entries and entries[0][0] <= oldest:
            entries.popleft()

    def front(self) -> float:
        """Return the extreme value of the live entries."""
        return self._entries[0][1]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()


class SlidingWindow:
    """Fixed-size window over a stream of numbers.

    Args:
        size: number of most recent values that the window retains.

    Raises:
        ValueError: if ``size`` is not positive.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("window size must be positive")
        self._size = size
        self._values: Deque[float] = deque()
        self._sorted: List[float] = []
        self._min = _MonotonicDeque(keep_max=False)
        self._max = _MonotonicDeque(keep_max=True)
        self._count = 0

    def __len__(self) -> int:
        """Return how many values the window currently holds."""
        return len(self._values)

    @property
    def size(self) -> int:
        """Capacity of the window."""
        return self._size

    @property
    def total_pushed(self) -> int:
        """Number of values pushed since creation or the last reset."""
        return self._count

    def is_full(self) -> bool:
        """Return ``True`` once the window holds ``size`` values."""
        return len(self._values) == self._size

    def push(self, value: float) -> None:
        """Add a value, evicting the oldest one when the window is full.

        Raises:
            ValueError: if ``value`` is NaN, which has no defined order.
        """
        if value != value:
            raise ValueError("NaN cannot be added to the window")
        if len(self._values) == self._size:
            evicted = self._values.popleft()
            del self._sorted[bisect_left(self._sorted, evicted)]
        self._values.append(value)
        insort(self._sorted, value)
        self._min.push(self._count, value)
        self._max.push(self._count, value)
        self._count += 1
        oldest = self._count - self._size
        self._min.expire(oldest)
        self._max.expire(oldest)

    def extend(self, values: Iterable[float]) -> None:
        """Push every value in order."""
        for value in values:
            self.push(value)

    def _require_values(self) -> None:
        """Raise when a statistic is requested from an empty window."""
        if not self._values:
            raise ValueError("window is empty")

    def min(self) -> float:
        """Return the smallest value in the window."""
        self._require_values()
        return self._min.front()

    def max(self) -> float:
        """Return the largest value in the window."""
        self._require_values()
        return self._max.front()

    def range(self) -> float:
        """Return the spread between the largest and smallest value."""
        return self.max() - self.min()

    def mean(self) -> float:
        """Return the arithmetic mean of the window."""
        self._require_values()
        return math.fsum(self._values) / len(self._values)

    def median(self) -> float:
        """Return the median of the window.

        For an even number of values this is the mean of the two middle ones.
        """
        self._require_values()
        ordered = self._sorted
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    def values(self) -> List[float]:
        """Return the window contents, oldest first."""
        return list(self._values)

    def reset(self) -> None:
        """Forget every value."""
        self._values.clear()
        self._sorted.clear()
        self._min.clear()
        self._max.clear()
        self._count = 0


def rolling_max(values: Iterable[float], size: int) -> List[float]:
    """Return the maximum of each full window of ``size`` over ``values``."""
    window = SlidingWindow(size)
    result: List[float] = []
    for value in values:
        window.push(value)
        if window.is_full():
            result.append(window.max())
    return result
