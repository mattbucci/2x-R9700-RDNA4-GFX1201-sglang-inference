"""Hand-written binary search helpers for sorted sequences.

The functions mirror the vocabulary of the C++ standard library:

* :func:`lower_bound` is the first position whose element is not less than
  the value,
* :func:`upper_bound` is the first position whose element is greater than
  the value.

Everything else is built on these two: membership tests, first / last
occurrence, counting the elements inside a closed range, neighbours of a
value (floor, ceiling, closest) and insertion that keeps a list sorted.

Two helpers deal with rotated arrays, i.e. ascending arrays of distinct
elements that were cut at some point and had the two parts swapped, such as
``[30, 40, 50, 10, 20]``.

An optional ``key`` function extracts the comparison key from each element,
as in :func:`sorted`; search values are always given as keys.
"""

from __future__ import annotations

from typing import Any, Callable, List, MutableSequence, Optional, Sequence, TypeVar

T = TypeVar("T")
KeyFunc = Optional[Callable[[Any], Any]]


def _key_of(item: Any, key: KeyFunc) -> Any:
    """Apply the optional key function to ``item``."""
    return item if key is None else key(item)


def lower_bound(items: Sequence[T], value: Any, key: KeyFunc = None) -> int:
    """Return the first index ``i`` with ``items[i] >= value``.

    Returns ``len(items)`` when every element is smaller than ``value``.
    """
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if _key_of(items[mid], key) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(items: Sequence[T], value: Any, key: KeyFunc = None) -> int:
    """Return the first index ``i`` with ``items[i] > value``.

    Returns ``len(items)`` when no element is greater than ``value``.
    """
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < _key_of(items[mid], key):
            hi = mid
        else:
            lo = mid + 1
    return lo


def contains(items: Sequence[T], value: Any, key: KeyFunc = None) -> bool:
    """Return ``True`` when ``value`` occurs in the sorted sequence."""
    pos = lower_bound(items, value, key)
    return pos < len(items) and _key_of(items[pos], key) == value


def index_of(items: Sequence[T], value: Any, key: KeyFunc = None) -> int:
    """Return the index of the first occurrence of ``value`` or ``-1``."""
    pos = lower_bound(items, value, key)
    if pos < len(items) and _key_of(items[pos], key) == value:
        return pos
    return -1


def last_index_of(items: Sequence[T], value: Any, key: KeyFunc = None) -> int:
    """Return the index of the last occurrence of ``value`` or ``-1``."""
    pos = upper_bound(items, value, key) - 1
    if pos >= 0 and _key_of(items[pos], key) == value:
        return pos
    return -1


def count_equal(items: Sequence[T], value: Any, key: KeyFunc = None) -> int:
    """Return how many elements compare equal to ``value``."""
    return upper_bound(items, value, key) - lower_bound(items, value, key)


def count_in_range(items: Sequence[T], low: Any, high: Any, key: KeyFunc = None) -> int:
    """Return how many elements lie in the closed range ``[low, high]``.

    An empty range (``high < low``) contains no elements.
    """
    if high < low:
        return 0
    return upper_bound(items, high, key) - lower_bound(items, low, key)


def floor_value(items: Sequence[T], value: Any, key: KeyFunc = None) -> Optional[T]:
    """Return the largest element that is ``<= value``, or ``None``."""
    pos = upper_bound(items, value, key)
    return items[pos - 1] if pos > 0 else None


def ceiling_value(items: Sequence[T], value: Any, key: KeyFunc = None) -> Optional[T]:
    """Return the smallest element that is ``>= value``, or ``None``."""
    pos = lower_bound(items, value, key)
    return items[pos] if pos < len(items) else None


def closest_value(items: Sequence[float], value: float) -> Optional[float]:
    """Return the element nearest to ``value``.

    Ties go to the smaller element.  Returns ``None`` for an empty sequence.
    """
    if not items:
        return None
    pos = lower_bound(items, value)
    if pos == 0:
        return items[0]
    if pos == len(items):
        return items[-1]
    before, after = items[pos - 1], items[pos]
    return before if value - before <= after - value else after


def insert_sorted(items: MutableSequence[T], value: T, key: KeyFunc = None) -> int:
    """Insert ``value`` so that ``items`` stays sorted and return its index.

    The value goes after any elements that compare equal, so insertion order
    is preserved among equal keys.
    """
    pos = upper_bound(items, _key_of(value, key), key)
    items.insert(pos, value)
    return pos


def insert_unique(items: MutableSequence[T], value: T, key: KeyFunc = None) -> bool:
    """Insert ``value`` unless an element with the same key exists.

    Returns ``True`` when the value was inserted.
    """
    wanted = _key_of(value, key)
    pos = lower_bound(items, wanted, key)
    if pos < len(items) and _key_of(items[pos], key) == wanted:
        return False
    items.insert(pos, value)
    return True


def merge_sorted(left: Sequence[T], right: Sequence[T]) -> List[T]:
    """Merge two sorted sequences into one sorted list (stable)."""
    out: List[T] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def rotation_point(items: Sequence[T]) -> int:
    """Return the index of the smallest element of a rotated array.

    The array must hold distinct elements.  An array that is not rotated,
    and an empty one, give ``0``.
    """
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[hi] < items[mid]:
            lo = mid + 1
        else:
            hi = mid
    return max(lo, 0)


def search_rotated(items: Sequence[T], target: T) -> int:
    """Return the index of ``target`` in a rotated array, or ``-1``.

    The array must hold distinct elements.  Each step decides which half of
    the current window is in ascending order and whether the target can lie
    inside that half.
    """
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[lo] < items[mid]:
            if items[lo] <= target < items[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if items[mid] < target <= items[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def unrotate(items: Sequence[T]) -> List[T]:
    """Return the ascending order of a rotated array as a new list."""
    pivot = rotation_point(items)
    return list(items[pivot:]) + list(items[:pivot])
