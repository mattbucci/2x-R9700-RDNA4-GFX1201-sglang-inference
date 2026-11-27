"""Set of integers represented as disjoint half-open intervals.

An :class:`IntervalSet` stores a set of integers as a sorted list of
half-open ranges ``[start, end)``: ``start`` belongs to the range and ``end``
does not.  The representation is kept canonical at all times: ranges are
sorted, pairwise disjoint and never adjacent, so ``[1, 3)`` followed by
``[3, 5)`` is always stored as the single range ``[1, 5)``.  Two sets holding
the same integers therefore compare equal.

All point and range queries run in ``O(log n)`` through binary search over
the range boundaries; insertion and removal are ``O(n)`` in the worst case
because of list splicing.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Iterable, Iterator, List, Tuple

Interval = Tuple[int, int]


def _check(start: int, end: int) -> None:
    """Reject ranges whose end precedes their start."""
    if end < start:
        raise ValueError(f"invalid interval [{start}, {end})")


class IntervalSet:
    """Mutable set of integers stored as canonical half-open intervals."""

    def __init__(self, intervals: Iterable[Interval] = ()) -> None:
        """Create a set holding the union of ``intervals``."""
        self._starts: List[int] = []
        self._ends: List[int] = []
        for start, end in intervals:
            self.add(start, end)

    def copy(self) -> "IntervalSet":
        """Return an independent set with the same contents."""
        clone = IntervalSet()
        clone._starts = list(self._starts)
        clone._ends = list(self._ends)
        return clone

    def __len__(self) -> int:
        """Number of stored (maximal) intervals."""
        return len(self._starts)

    def __iter__(self) -> Iterator[Interval]:
        """Iterate over the stored intervals in ascending order."""
        return iter(zip(self._starts, self._ends))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalSet):
            return NotImplemented
        return self._starts == other._starts and self._ends == other._ends

    def __repr__(self) -> str:
        body = ", ".join(f"[{s}, {e})" for s, e in self)
        return f"IntervalSet({body})"

    def measure(self) -> int:
        """Number of integers contained in the set."""
        return sum(end - start for start, end in self)

    def add(self, start: int, end: int) -> None:
        """Add every integer in ``[start, end)``.

        Stored intervals that overlap or touch the new range are merged with
        it.  An empty range is accepted and changes nothing.
        """
        _check(start, end)
        if start == end:
            return
        # Stored intervals lo..hi-1 overlap or touch [start, end).
        lo = bisect_left(self._ends, start)
        hi = bisect_right(self._starts, end)
        if lo < hi:
            start = min(start, self._starts[lo])
            end = max(end, self._ends[hi - 1])
        self._starts[lo:hi] = [start]
        self._ends[lo:hi] = [end]

    def remove(self, start: int, end: int) -> None:
        """Remove every integer in ``[start, end)``.

        Stored intervals that straddle a boundary of the range are trimmed;
        an interval strictly containing the range is split in two.
        """
        _check(start, end)
        if start == end:
            return
        # Stored intervals lo..hi-1 share at least one integer with the range.
        lo = bisect_right(self._ends, start)
        hi = bisect_left(self._starts, end)
        if lo >= hi:
            return
        kept_starts: List[int] = []
        kept_ends: List[int] = []
        if self._starts[lo] < start:
            kept_starts.append(self._starts[lo])
            kept_ends.append(start)
        if self._ends[hi - 1] > end:
            kept_starts.append(end)
            kept_ends.append(self._ends[hi - 1])
        self._starts[lo:hi] = kept_starts
        self._ends[lo:hi] = kept_ends

    def contains(self, value: int) -> bool:
        """Whether the integer ``value`` is a member of the set."""
        index = bisect_right(self._starts, value) - 1
        return index >= 0 and value <= self._ends[index]

    __contains__ = contains

    def contains_range(self, start: int, end: int) -> bool:
        """Whether every integer in ``[start, end)`` is a member.

        An empty range is contained in every set.
        """
        _check(start, end)
        if start == end:
            return True
        index = bisect_right(self._starts, start) - 1
        return index >= 0 and end <= self._ends[index]

    def overlaps(self, start: int, end: int) -> bool:
        """Whether at least one integer in ``[start, end)`` is a member."""
        _check(start, end)
        if start == end:
            return False
        index = bisect_right(self._ends, start)
        return index < len(self._starts) and self._starts[index] < end

    def find(self, value: int) -> Interval:
        """Return the stored interval containing ``value``.

        Raises:
            KeyError: if ``value`` is not a member of the set.
        """
        index = bisect_right(self._starts, value) - 1
        if index < 0 or value >= self._ends[index]:
            raise KeyError(value)
        return self._starts[index], self._ends[index]

    def merge(self, other: "IntervalSet") -> "IntervalSet":
        """Return a new set holding the union of both sets."""
        if len(self) >= len(other):
            result, smaller = self.copy(), other
        else:
            result, smaller = other.copy(), self
        for start, end in smaller:
            result.add(start, end)
        return result

    def intersection(self, other: "IntervalSet") -> "IntervalSet":
        """Return a new set holding the integers present in both sets."""
        result = IntervalSet()
        i = j = 0
        while i < len(self._starts) and j < len(other._starts):
            start = max(self._starts[i], other._starts[j])
            end = min(self._ends[i], other._ends[j])
            if start < end:
                result._starts.append(start)
                result._ends.append(end)
            # Advance whichever interval finishes first.
            if self._ends[i] < other._ends[j]:
                i += 1
            else:
                j += 1
        return result

    def difference(self, other: "IntervalSet") -> "IntervalSet":
        """Return a new set holding the members of this set absent from ``other``."""
        result = self.copy()
        for start, end in other:
            result.remove(start, end)
        return result

    def gaps(self, start: int, end: int) -> List[Interval]:
        """Return the maximal sub-ranges of ``[start, end)`` not in the set."""
        _check(start, end)
        missing: List[Interval] = []
        cursor = start
        for lo, hi in self:
            if hi <= cursor:
                continue
            if lo >= end:
                break
            if lo > cursor:
                missing.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < end:
            missing.append((cursor, end))
        return missing

    def clear(self) -> None:
        """Remove every member."""
        self._starts.clear()
        self._ends.clear()
