"""Bloom filter on a packed bit array.

A Bloom filter answers "have I seen this item?" with no false negatives and
a tunable false-positive rate, using a fixed amount of memory.  This
implementation derives its ``k`` bit positions from one 128-bit BLAKE2b
digest by double hashing (Kirsch and Mitzenmacher): position ``i`` is
``(h1 + i * h2) mod m``.

Two filters built with the same geometry can be merged with
:meth:`BloomFilter.union`, and the number of distinct items that were added
can be estimated from the fraction of bits that are set.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Iterator


def _encode(item: Any) -> bytes:
    """Turn an item into the bytes that are hashed."""
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    return repr(item).encode("utf-8")


def optimal_num_bits(capacity: int, error_rate: float) -> int:
    """Return the bit count that holds ``capacity`` items at ``error_rate``."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not 0.0 < error_rate < 1.0:
        raise ValueError("error_rate must be between 0 and 1")
    bits = -capacity * math.log(error_rate) / (math.log(2) ** 2)
    return max(1, math.ceil(bits))


def optimal_num_hashes(num_bits: int, capacity: int) -> int:
    """Return the hash count that minimises false positives."""
    return max(1, round(num_bits / capacity * math.log(2)))


class BloomFilter:
    """Fixed-size Bloom filter.

    Args:
        num_bits: size ``m`` of the bit array.
        num_hashes: number ``k`` of bit positions per item.
    """

    def __init__(self, num_bits: int, num_hashes: int) -> None:
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        self._num_bits = num_bits
        self._num_hashes = num_hashes
        self._bits = bytearray((num_bits + 7) // 8)

    @classmethod
    def for_capacity(cls, capacity: int, error_rate: float = 0.01) -> "BloomFilter":
        """Build a filter sized for ``capacity`` items at ``error_rate``."""
        num_bits = optimal_num_bits(capacity, error_rate)
        return cls(num_bits, optimal_num_hashes(num_bits, capacity))

    @property
    def num_bits(self) -> int:
        """Size of the bit array."""
        return self._num_bits

    @property
    def num_hashes(self) -> int:
        """Number of bit positions used per item."""
        return self._num_hashes

    def _positions(self, item: Any) -> Iterator[int]:
        """Yield the bit positions of ``item``."""
        digest = hashlib.blake2b(_encode(item), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big") | 1
        for i in range(self._num_hashes):
            yield (h1 + i * h2) % self._num_bits

    def _get_bit(self, position: int) -> bool:
        """Return the bit at ``position``."""
        return bool(self._bits[position >> 3] & (1 << (position & 7)))

    def _set_bit(self, position: int) -> None:
        """Set the bit at ``position``."""
        self._bits[position >> 3] |= 1 << (position & 7)

    def add(self, item: Any) -> bool:
        """Add ``item`` to the filter.

        Returns:
            ``True`` when at least one bit changed, which proves the item
            had not been added before; ``False`` when the item was already
            (possibly) present.
        """
        changed = False
        for position in self._positions(item):
            if not self._get_bit(position):
                self._set_bit(position)
                changed = True
        return changed

    def update(self, items: Iterable[Any]) -> int:
        """Add every item and return how many of them changed a bit."""
        return sum(1 for item in items if self.add(item))

    def __contains__(self, item: Any) -> bool:
        """Return ``False`` when ``item`` was definitely never added."""
        return all(self._get_bit(position) for position in self._positions(item))

    def bit_count(self) -> int:
        """Return the number of bits that are set."""
        return sum(bin(byte).count("1") for byte in self._bits)

    def fill_ratio(self) -> float:
        """Return the fraction of bits that are set."""
        return self.bit_count() / self._num_bits

    def false_positive_rate(self) -> float:
        """Return the current probability of a false positive.

        A lookup of an absent item is a false positive when all ``k`` of its
        positions happen to be set.
        """
        return self.fill_ratio() ** self._num_hashes

    def estimated_cardinality(self) -> float:
        """Estimate how many distinct items have been added.

        Uses the Swamidass-Baldi estimator ``-(m / k) * ln(1 - X / m)``
        where ``X`` is the number of set bits.  A saturated filter carries no
        information about its item count and yields infinity.
        """
        set_bits = self.bit_count()
        if set_bits == 0:
            return 0.0
        if set_bits >= self._num_bits:
            return math.inf
        empty_fraction = 1 - set_bits // self._num_bits
        return -(self._num_bits / self._num_hashes) * math.log(empty_fraction)

    def is_compatible(self, other: "BloomFilter") -> bool:
        """Return ``True`` when ``other`` has the same geometry."""
        return (
            self._num_bits == other._num_bits
            and self._num_hashes == other._num_hashes
        )

    def union(self, other: "BloomFilter") -> "BloomFilter":
        """Return a new filter that contains the items of both filters.

        Raises:
            ValueError: if the filters differ in size or hash count.
        """
        if not self.is_compatible(other):
            raise ValueError("filters must share num_bits and num_hashes")
        merged = BloomFilter(self._num_bits, self._num_hashes)
        merged._bits = bytearray(a | b for a, b in zip(self._bits, other._bits))
        return merged

    def copy(self) -> "BloomFilter":
        """Return an independent copy of this filter."""
        clone = BloomFilter(self._num_bits, self._num_hashes)
        clone._bits = bytearray(self._bits)
        return clone

    def clear(self) -> None:
        """Reset the filter to its empty state."""
        self._bits = bytearray(len(self._bits))
