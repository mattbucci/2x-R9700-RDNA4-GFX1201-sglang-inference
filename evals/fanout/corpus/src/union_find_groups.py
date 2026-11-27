"""Disjoint-set forest (union-find) over arbitrary hashable items.

The structure keeps a partition of the items it has seen.  Two items can be
merged into one group with :meth:`DisjointSets.union`; membership questions
are answered with :meth:`DisjointSets.find` and
:meth:`DisjointSets.connected`.  Union by rank and path compression keep
both operations close to constant amortised time.

Group listings are deterministic: groups, and the members inside a group,
appear in the order in which the items were first added.

Example::

    sets = DisjointSets(["a", "b", "c"])
    sets.union("a", "b")
    sets.connected("a", "b")     # True
    sets.component_size("a")     # 2
    sets.groups()                # [["a", "b"], ["c"]]
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Tuple


class DisjointSets:
    """Partition of hashable items into disjoint groups.

    Args:
        items: initial items; each starts in a group of its own.
    """

    def __init__(self, items: Iterable[Hashable] = ()) -> None:
        self._parent: Dict[Hashable, Hashable] = {}
        self._rank: Dict[Hashable, int] = {}
        self._size: Dict[Hashable, int] = {}
        self._count = 0
        for item in items:
            self.add(item)

    def __len__(self) -> int:
        """Return the number of items (not groups)."""
        return len(self._parent)

    def __contains__(self, item: object) -> bool:
        """Return ``True`` when ``item`` has been added."""
        return item in self._parent

    @property
    def component_count(self) -> int:
        """Number of disjoint groups."""
        return self._count

    def add(self, item: Hashable) -> bool:
        """Add ``item`` as a singleton group.

        Returns:
            ``True`` when the item was new, ``False`` when already present.
        """
        if item in self._parent:
            return False
        self._parent[item] = item
        self._rank[item] = 0
        self._size[item] = 1
        self._count += 1
        return True

    def find(self, item: Hashable) -> Hashable:
        """Return the representative of the group that contains ``item``.

        Every node visited on the way is re-pointed at the representative.

        Raises:
            KeyError: if ``item`` was never added.
        """
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        node = item
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(self, a: Hashable, b: Hashable) -> bool:
        """Merge the groups of ``a`` and ``b``.

        Items that have not been seen before are added first.

        Returns:
            ``True`` when two different groups were merged, ``False`` when
            both items already shared a group.
        """
        self.add(a)
        self.add(b)
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[b]
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1
        self._count -= 1
        return True

    def union_all(self, pairs: Iterable[Tuple[Hashable, Hashable]]) -> int:
        """Apply :meth:`union` to every pair.

        Returns:
            The number of pairs that merged two different groups.
        """
        merged = 0
        for a, b in pairs:
            if self.union(a, b):
                merged += 1
        return merged

    def connected(self, a: Hashable, b: Hashable) -> bool:
        """Return ``True`` when both items are known and share a group."""
        if a not in self._parent or b not in self._parent:
            return False
        return self.find(a) == self.find(b)

    def component_size(self, item: Hashable) -> int:
        """Return the number of items in the group of ``item``.

        Raises:
            KeyError: if ``item`` was never added.
        """
        return self._size[self.find(item)]

    def members(self, item: Hashable) -> List[Hashable]:
        """Return all items in the group of ``item`` in insertion order.

        Raises:
            KeyError: if ``item`` was never added.
        """
        root = self.find(item)
        return [other for other in self._parent if self.find(other) == root]

    def groups(self) -> List[List[Hashable]]:
        """Return every group as a list of items.

        Groups are ordered by their earliest item; members keep insertion
        order.
        """
        by_root: Dict[Hashable, List[Hashable]] = {}
        for item in list(self._parent):
            by_root.setdefault(self.find(item), []).append(item)
        return list(by_root.values())

    def largest_group(self) -> List[Hashable]:
        """Return the group with the most members.

        Among groups of equal size the one with the earliest item wins.  An
        empty structure yields an empty list.
        """
        best: List[Hashable] = []
        for group in self.groups():
            if len(group) > len(best):
                best = group
        return best

    def size_histogram(self) -> Dict[int, int]:
        """Return a mapping from group size to the number of such groups."""
        histogram: Dict[int, int] = {}
        for item, parent in self._parent.items():
            if item == parent:
                size = self._size[item]
                histogram[size] = histogram.get(size, 0) + 1
        return histogram

    def is_single_group(self) -> bool:
        """Return ``True`` when all items belong to one group.

        An empty structure is not considered a single group.
        """
        return self._count == 1
