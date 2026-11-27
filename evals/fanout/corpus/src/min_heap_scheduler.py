"""Priority task scheduler backed by a hand-written binary min-heap.

Tasks are identified by a string id and ordered by a numeric priority, lowest
first (a priority is typically a due time).  Tasks of equal priority run in
the order in which they were scheduled.  Besides the usual push and pop the
scheduler supports cancelling and rescheduling a task by id in ``O(log n)``,
which is why the heap is implemented here instead of using ``heapq``: every
move inside the array is mirrored in an id-to-position index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    """A scheduled unit of work."""

    task_id: str
    priority: float
    payload: Any = None
    seq: int = 0

    def sort_key(self) -> tuple:
        """Return the heap ordering key: priority first, then arrival."""
        return (self.priority, self.seq)


class Scheduler:
    """Min-heap scheduler with cancel and reschedule by task id."""

    def __init__(self) -> None:
        self._heap: List[Task] = []
        self._index: Dict[str, int] = {}
        self._next_seq = 0

    def __len__(self) -> int:
        """Return the number of pending tasks."""
        return len(self._heap)

    def __contains__(self, task_id: object) -> bool:
        """Return ``True`` when a task with this id is pending."""
        return task_id in self._index

    # ------------------------------------------------------------------
    # heap primitives
    # ------------------------------------------------------------------

    def _less(self, i: int, j: int) -> bool:
        """Return ``True`` when the task at ``i`` must run before ``j``."""
        return self._heap[i].sort_key() < self._heap[j].sort_key()

    def _swap(self, i: int, j: int) -> None:
        """Exchange two heap slots and keep the position index in sync."""
        heap = self._heap
        heap[i], heap[j] = heap[j], heap[i]
        self._index[heap[i].task_id] = i
        self._index[heap[j].task_id] = j

    def _sift_up(self, idx: int) -> bool:
        """Move the task at ``idx`` towards the root while it is too low.

        Returns:
            ``True`` when the task changed position.
        """
        moved = False
        while idx > 0:
            parent = (idx - 1) // 2
            if not self._less(idx, parent):
                break
            self._swap(idx, parent)
            idx = parent
            moved = True
        return moved

    def _sift_down(self, idx: int) -> None:
        """Move the task at ``idx`` towards the leaves while it is too high."""
        size = len(self._heap)
        while True:
            left = 2 * idx + 1
            right = left + 1
            smallest = idx
            if left < size and self._less(left, smallest):
                smallest = left
            if right < size and self._less(right, smallest):
                smallest = right
            if smallest == idx:
                return
            self._swap(idx, smallest)
            idx = smallest

    def _restore(self, idx: int) -> None:
        """Re-establish the heap order around a slot whose key changed."""
        if not self._sift_up(idx):
            self._sift_down(idx)

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def schedule(self, task_id: str, priority: float, payload: Any = None) -> Task:
        """Add a task.

        Raises:
            ValueError: if a task with the same id is already pending.
        """
        if task_id in self._index:
            raise ValueError(f"task {task_id!r} is already scheduled")
        task = Task(task_id, priority, payload, self._next_seq)
        self._next_seq += 1
        self._heap.append(task)
        idx = len(self._heap) - 1
        self._index[task_id] = idx
        self._sift_up(idx)
        return task

    def peek(self) -> Optional[Task]:
        """Return the next task without removing it, or ``None`` if idle."""
        return self._heap[0] if self._heap else None

    def pop(self) -> Task:
        """Remove and return the task with the lowest priority.

        Raises:
            IndexError: if no task is pending.
        """
        if not self._heap:
            raise IndexError("pop from an empty scheduler")
        top = self._heap[0]
        last = self._heap.pop()
        del self._index[top.task_id]
        if self._heap:
            self._heap[0] = last
            self._index[last.task_id] = 0
            self._sift_down(0)
        return top

    def cancel(self, task_id: str) -> bool:
        """Remove a pending task by id.

        Returns:
            ``True`` when the task was pending and has been removed.
        """
        idx = self._index.pop(task_id, None)
        if idx is None:
            return False
        last = self._heap.pop()
        if idx < len(self._heap):
            self._heap[idx] = last
            self._index[last.task_id] = idx
            self._sift_down(idx)
        return True

    def reschedule(self, task_id: str, priority: float) -> None:
        """Change the priority of a pending task.

        The task keeps its arrival order among tasks of equal priority.

        Raises:
            KeyError: if no task with this id is pending.
        """
        idx = self._index[task_id]
        self._heap[idx].priority = priority
        self._restore(idx)

    def priority_of(self, task_id: str) -> Optional[float]:
        """Return the priority of a pending task, or ``None`` if unknown."""
        idx = self._index.get(task_id)
        if idx is None:
            return None
        return self._heap[idx].priority

    def pop_due(self, now: float) -> List[Task]:
        """Remove and return every task whose priority is at most ``now``.

        Tasks are returned in execution order.
        """
        due: List[Task] = []
        while self._heap and self._heap[0].priority <= now:
            due.append(self.pop())
        return due

    def drain(self) -> List[Task]:
        """Remove and return all pending tasks in execution order."""
        tasks: List[Task] = []
        while self._heap:
            tasks.append(self.pop())
        return tasks

    def pending_ids(self) -> List[str]:
        """Return the ids of all pending tasks in execution order.

        The scheduler is left unchanged.
        """
        ordered = sorted(self._heap, key=Task.sort_key)
        return [task.task_id for task in ordered]
