"""Fixed-capacity ring buffer.

A :class:`RingBuffer` keeps the most recent ``capacity`` items in a
pre-allocated list.  Items are addressed logically: index 0 is the oldest
stored item and index ``len(buffer) - 1`` is the newest, regardless of where
they sit in the underlying storage.

Two modes control what happens when an item is pushed into a full buffer:

* overwrite mode (the default) silently replaces the oldest item;
* strict mode raises :class:`BufferFullError` and leaves the buffer intact.

Items can be removed from either end, read by logical index (negative
indices count from the newest item), iterated from oldest to newest, and the
last ``n`` items can be copied out as a list.
"""

from __future__ import annotations

from typing import Generic, Iterable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class BufferFullError(Exception):
    """Raised when pushing into a full buffer that does not overwrite."""


class RingBuffer(Generic[T]):
    """Bounded FIFO storage over a circular list."""

    def __init__(self, capacity: int, overwrite: bool = True) -> None:
        """Create an empty buffer.

        Args:
            capacity: number of items the buffer can hold, at least 1.
            overwrite: whether a push into a full buffer replaces the oldest
                item instead of raising.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._overwrite = overwrite
        self._slots: List[Optional[T]] = [None] * capacity
        self._head = 0  # storage index of the oldest item
        self._size = 0

    @property
    def capacity(self) -> int:
        """Maximum number of items held."""
        return self._capacity

    @property
    def overwrite(self) -> bool:
        """Whether pushes into a full buffer replace the oldest item."""
        return self._overwrite

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        """Whether the buffer holds no items."""
        return self._size == 0

    def is_full(self) -> bool:
        """Whether the buffer holds ``capacity`` items."""
        return self._size == self._capacity

    def push(self, item: T) -> bool:
        """Append ``item`` as the newest element.

        Returns ``True`` when the oldest item was overwritten to make room.

        Raises:
            BufferFullError: the buffer is full and overwrite mode is off.
        """
        if self._size == self._capacity:
            if not self._overwrite:
                raise BufferFullError(f"buffer holds {self._capacity} items")
            self._slots[self._head] = item
            self._head = (self._head + 1) % self._capacity
            return True
        self._slots[(self._head + self._size) % self._capacity] = item
        self._size += 1
        return False

    def extend(self, items: Iterable[T]) -> int:
        """Push every item in order; return how many old items were overwritten."""
        overwritten = 0
        for item in items:
            if self.push(item):
                overwritten += 1
        return overwritten

    def popleft(self) -> T:
        """Remove and return the oldest item.

        Raises:
            IndexError: the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("pop from an empty ring buffer")
        item = self._slots[self._head]
        self._slots[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        return item  # type: ignore[return-value]

    def pop(self) -> T:
        """Remove and return the newest item.

        Raises:
            IndexError: the buffer is empty.
        """
        if self._size == 0:
            raise IndexError("pop from an empty ring buffer")
        tail = (self._head + self._size) % self._capacity
        item = self._slots[tail]
        self._slots[tail] = None
        self._size -= 1
        return item  # type: ignore[return-value]

    def _storage_index(self, index: int) -> int:
        """Translate a logical index into a storage index, checking bounds."""
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("ring buffer index out of range")
        return (self._head + index) % self._capacity

    def __getitem__(self, index: int) -> T:
        """Return the item at logical ``index``; 0 is the oldest item."""
        if not isinstance(index, int):
            raise TypeError("ring buffer indices must be integers")
        return self._slots[self._storage_index(index)]  # type: ignore[return-value]

    def __setitem__(self, index: int, item: T) -> None:
        """Replace the item at logical ``index``."""
        if not isinstance(index, int):
            raise TypeError("ring buffer indices must be integers")
        self._slots[self._storage_index(index)] = item

    def oldest(self) -> T:
        """Return the oldest item without removing it."""
        return self[0]

    def newest(self) -> T:
        """Return the newest item without removing it."""
        return self[-1]

    def __iter__(self) -> Iterator[T]:
        """Iterate from the oldest item to the newest.

        The iterator works on a snapshot, so the buffer may be modified
        while it is being consumed.
        """
        return iter(self.to_list())

    def __contains__(self, item: object) -> bool:
        return any(stored == item for stored in self.to_list())

    def to_list(self) -> List[T]:
        """Copy the contents into a list ordered oldest to newest."""
        first = self._capacity - self._head
        if self._size <= first:
            chunk = self._slots[self._head:self._head + self._size]
        else:
            chunk = self._slots[self._head:] + self._slots[:self._size - first]
        return chunk  # type: ignore[return-value]

    def last(self, count: int) -> List[T]:
        """Return up to ``count`` newest items, ordered oldest to newest.

        Fewer items are returned when the buffer holds fewer than ``count``.
        """
        if count < 0:
            raise ValueError("count must not be negative")
        take = min(count, self._size)
        return [self[index] for index in range(self._size - take, self._size)]

    def drop_oldest(self, count: int) -> int:
        """Discard up to ``count`` oldest items; return how many were dropped."""
        if count < 0:
            raise ValueError("count must not be negative")
        dropped = min(count, self._size)
        for _ in range(dropped):
            self._slots[self._head] = None
            self._head = (self._head + 1) % self._capacity
        self._size -= dropped
        return dropped

    def clear(self) -> None:
        """Remove every item."""
        self._slots = [None] * self._capacity
        self._head = 0
        self._size = 0

    def __repr__(self) -> str:
        return f"RingBuffer({self.to_list()!r}, capacity={self._capacity})"
