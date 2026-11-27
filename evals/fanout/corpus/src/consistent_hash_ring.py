"""Consistent hashing with virtual nodes.

Each node is placed on a 64-bit ring at several pseudo-random points (its
virtual nodes).  A key belongs to the node that owns the first point at or
after the key's own hash, wrapping around at the end of the ring.  Adding or
removing a node therefore only moves the keys next to that node's points.

Hashing uses MD5 purely as a stable, well mixed function; placement is the
same in every process and on every platform.
"""

from __future__ import annotations

import bisect
import hashlib
from typing import Dict, Iterable, Iterator, List, Optional

_RING_SIZE = 1 << 64


class RingError(ValueError):
    """Raised for invalid ring configuration or lookups on an empty ring."""


def ring_hash(text: str) -> int:
    """Map a string to a point on the ring, in ``[0, 2**64)``."""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class HashRing:
    """A consistent hash ring over string node names.

    ``vnodes`` is the number of ring points per unit of node weight.  More
    points give a more even key distribution at the cost of memory.
    """

    def __init__(self, nodes: Optional[Iterable[str]] = None, vnodes: int = 100) -> None:
        if vnodes < 1:
            raise RingError("vnodes must be at least 1")
        self._vnodes = vnodes
        self._nodes: Dict[str, int] = {}
        self._keys: List[int] = []
        self._owners: Dict[int, str] = {}
        for node in nodes or ():
            self.add_node(node)

    def __repr__(self) -> str:
        return f"HashRing(nodes={self.nodes!r}, vnodes={self._vnodes})"

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._nodes

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._nodes))

    @property
    def nodes(self) -> List[str]:
        """The node names, sorted."""
        return sorted(self._nodes)

    def weight(self, node: str) -> int:
        """Return the weight ``node`` was added with."""
        if node not in self._nodes:
            raise KeyError(node)
        return self._nodes[node]

    def point_count(self, node: str) -> int:
        """Return how many ring points ``node`` currently owns."""
        if node not in self._nodes:
            raise KeyError(node)
        return sum(1 for owner in self._owners.values() if owner == node)

    def add_node(self, node: str, weight: int = 1) -> None:
        """Place ``node`` on the ring with ``vnodes * weight`` points.

        A point that is already taken stays with its current owner; with a
        64-bit ring this is vanishingly rare.
        """
        if node in self._nodes:
            raise RingError(f"node {node!r} is already on the ring")
        if weight < 1:
            raise RingError("weight must be at least 1")
        for replica in range(self._vnodes * weight):
            point = ring_hash(f"{node}#{replica}")
            if point in self._owners:
                continue
            self._owners[point] = node
            bisect.insort(self._keys, point)
        self._nodes[node] = weight

    def remove_node(self, node: str) -> None:
        """Take ``node`` and all of its points off the ring."""
        if node not in self._nodes:
            raise KeyError(node)
        del self._nodes[node]
        self._keys = [point for point in self._keys if self._owners[point] != node]
        self._owners = {point: self._owners[point] for point in self._keys}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def _start_index(self, key: str) -> int:
        """Index of the first ring point at or after the hash of ``key``."""
        index = bisect.bisect_left(self._keys, ring_hash(key))
        if index == len(self._keys):
            return 0
        return index

    def get_node(self, key: str) -> str:
        """Return the node responsible for ``key``."""
        if not self._keys:
            raise RingError("the ring has no nodes")
        return self._owners[self._keys[self._start_index(key)]]

    def get_nodes(self, key: str, count: int) -> List[str]:
        """Return up to ``count`` distinct nodes for ``key``, primary first.

        The walk starts at the key's primary point and continues clockwise,
        skipping points whose node was already chosen.  When fewer than
        ``count`` nodes exist, all of them are returned.
        """
        if count < 1:
            raise RingError("count must be at least 1")
        if not self._keys:
            raise RingError("the ring has no nodes")
        count = min(count, len(self._nodes))
        start = self._start_index(key)
        chosen: List[str] = []
        for offset in range(len(self._keys)):
            point = self._keys[(start + offset) % len(self._nodes)]
            owner = self._owners[point]
            if owner in chosen:
                continue
            chosen.append(owner)
            if len(chosen) == count:
                break
        return chosen

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def distribution(self, keys: Iterable[str]) -> Dict[str, int]:
        """Count how many of ``keys`` each node is responsible for.

        Every node appears in the result, including nodes that own no key.
        """
        counts = {node: 0 for node in self._nodes}
        if not self._keys:
            return counts
        for key in keys:
            counts[self.get_node(key)] += 1
        return counts

    def ownership(self) -> Dict[str, float]:
        """Return the fraction of the ring each node owns; values sum to 1."""
        shares = {node: 0.0 for node in self._nodes}
        if not self._keys:
            return shares
        if len(self._keys) == 1:
            shares[self._owners[self._keys[0]]] = 1.0
            return shares
        for index, point in enumerate(self._keys):
            previous = self._keys[index - 1]
            arc = (point - previous) % _RING_SIZE
            shares[self._owners[point]] += arc / _RING_SIZE
        return shares

    def moved_keys(self, keys: Iterable[str], other: "HashRing") -> List[str]:
        """Return the keys whose responsible node differs on ``other``."""
        return [key for key in keys if self.get_node(key) != other.get_node(key)]
