"""Thread-safe bounded work queue with close semantics.

The queue hands work items from producer threads to consumer threads.  It
holds at most ``capacity`` items: producers block (optionally with a timeout)
while the queue is full and consumers block while it is empty.

Closing the queue is the shutdown signal.  After :meth:`BoundedWorkQueue.close`

* producers get :class:`QueueClosed` immediately,
* consumers keep receiving the items that are still queued, and
* once the queue is drained every consumer gets :class:`QueueClosed`,
  including the ones that were already blocked when the queue was closed.

Completion tracking follows the ``task_done`` / ``join`` convention of the
standard library queue so a coordinator can wait until all accepted work has
been processed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class QueueClosed(Exception):
    """Raised when an operation cannot proceed because the queue is closed."""


class QueueFull(Exception):
    """Raised when ``put`` could not find a free slot before its timeout."""


class QueueEmpty(Exception):
    """Raised when ``get`` could not find an item before its timeout."""


def _deadline(timeout: Optional[float]) -> Optional[float]:
    """Convert a relative timeout into a monotonic deadline.

    ``None`` means "wait forever"; negative timeouts behave like zero.
    """
    if timeout is None:
        return None
    return time.monotonic() + max(0.0, timeout)


def _remaining(deadline: Optional[float]) -> Optional[float]:
    """Return the seconds left until ``deadline`` (``None`` for no deadline)."""
    if deadline is None:
        return None
    return deadline - time.monotonic()


class BoundedWorkQueue(Generic[T]):
    """A FIFO queue of bounded size shared between threads."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._items: Deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._all_done = threading.Condition(self._lock)
        self._closed = False
        self._unfinished = 0

    @property
    def capacity(self) -> int:
        """Maximum number of items the queue holds at once."""
        return self._capacity

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        with self._lock:
            return self._closed

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def full(self) -> bool:
        """Return ``True`` when no free slot is available right now."""
        with self._lock:
            return len(self._items) >= self._capacity

    def empty(self) -> bool:
        """Return ``True`` when no item is queued right now."""
        with self._lock:
            return not self._items

    def put(self, item: T, timeout: Optional[float] = None) -> None:
        """Append ``item``, waiting for a free slot if necessary.

        Raises :class:`QueueFull` when the timeout expires first and
        :class:`QueueClosed` when the queue is, or becomes, closed.
        """
        deadline = _deadline(timeout)
        with self._not_full:
            while True:
                if self._closed:
                    raise QueueClosed("cannot put on a closed queue")
                if len(self._items) < self._capacity:
                    break
                remaining = _remaining(deadline)
                if remaining is not None and remaining <= 0:
                    raise QueueFull("no free slot before the timeout")
                self._not_full.wait(remaining)
            self._items.append(item)
            self._unfinished += 1
            self._not_empty.notify()

    def put_nowait(self, item: T) -> None:
        """Append ``item`` without waiting; raise :class:`QueueFull` if full."""
        self.put(item, timeout=0)

    def _wait_for_item(self, deadline: Optional[float]) -> None:
        """Block until an item is queued.  The lock must be held."""
        while not self._items:
            if self._closed:
                raise QueueClosed("queue is closed and drained")
            remaining = _remaining(deadline)
            if remaining is not None and remaining <= 0:
                raise QueueEmpty("no item before the timeout")
            self._not_empty.wait(remaining)

    def get(self, timeout: Optional[float] = None) -> T:
        """Remove and return the oldest item, waiting if necessary.

        Raises :class:`QueueEmpty` when the timeout expires first and
        :class:`QueueClosed` when the queue is closed and has no items left.
        """
        deadline = _deadline(timeout)
        with self._not_empty:
            self._wait_for_item(deadline)
            item = self._items.popleft()
            self._not_full.notify()
            return item

    def get_nowait(self) -> T:
        """Remove and return an item without waiting."""
        return self.get(timeout=0)

    def get_batch(self, max_items: int, timeout: Optional[float] = None) -> List[T]:
        """Return up to ``max_items`` items in FIFO order.

        Waits like :meth:`get` for the first item, then takes whatever else is
        already queued without waiting again.
        """
        if max_items <= 0:
            raise ValueError("max_items must be a positive integer")
        deadline = _deadline(timeout)
        with self._not_empty:
            self._wait_for_item(deadline)
            batch: List[T] = []
            while self._items and len(batch) < max_items:
                batch.append(self._items.popleft())
            self._not_full.notify(len(batch))
            return batch

    def task_done(self) -> None:
        """Mark one previously fetched item as fully processed."""
        with self._lock:
            if self._unfinished <= 0:
                raise ValueError("task_done() called more often than put()")
            self._unfinished -= 1
            if self._unfinished == 0:
                self._all_done.notify_all()

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait until every accepted item has been marked done.

        Returns ``True`` when all work finished and ``False`` on timeout.
        """
        deadline = _deadline(timeout)
        with self._all_done:
            while self._unfinished:
                remaining = _remaining(deadline)
                if remaining is not None and remaining <= 0:
                    return False
                self._all_done.wait(remaining)
            return True

    def close(self) -> None:
        """Stop accepting items and release every blocked thread.

        Closing is idempotent.  Items already queued stay available to
        consumers.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._not_empty.notify()
            self._not_full.notify_all()

    def drain(self) -> List[T]:
        """Remove and return all queued items without processing them.

        The removed items no longer count as unfinished work, so a pending
        :meth:`join` is released if nothing else is outstanding.
        """
        with self._lock:
            removed = list(self._items)
            self._items.clear()
            self._unfinished -= len(removed)
            if removed:
                self._not_full.notify_all()
            if self._unfinished == 0:
                self._all_done.notify_all()
            return removed

    def __iter__(self) -> Iterator[T]:
        """Yield items until the queue is closed and drained."""
        while True:
            try:
                item = self.get()
            except QueueClosed:
                return
            yield item
