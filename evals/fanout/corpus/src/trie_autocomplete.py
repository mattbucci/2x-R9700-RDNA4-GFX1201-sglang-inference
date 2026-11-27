"""Prefix trie with weighted autocomplete.

The trie stores words together with a numeric weight (for example a query
frequency).  It supports insertion, deletion with pruning of unused branches,
prefix queries and retrieval of the ``k`` heaviest completions of a prefix.

Example::

    trie = Trie()
    trie.insert("car", 5)
    trie.insert("cart", 2)
    trie.insert("cat", 9)
    trie.complete("ca", k=2)   # ["cat", "car"]
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterator, List, Optional, Tuple


class _Node:
    """A single trie node; one node per character on a stored path."""

    __slots__ = ("children", "is_word", "weight")

    def __init__(self) -> None:
        self.children: Dict[str, "_Node"] = {}
        self.is_word: bool = False
        self.weight: float = 0.0


class Trie:
    """Weighted prefix trie.

    Args:
        case_sensitive: when ``False`` every word and prefix is folded to
            lower case before it is stored or looked up.
    """

    def __init__(self, case_sensitive: bool = True) -> None:
        self._root = _Node()
        self._size = 0
        self._case_sensitive = case_sensitive

    def __len__(self) -> int:
        """Return the number of stored words."""
        return self._size

    def __contains__(self, word: object) -> bool:
        """Return ``True`` when ``word`` is stored as a complete word."""
        if not isinstance(word, str):
            return False
        node = self._find(self._normalize(word))
        return node is not None and node.is_word

    def _normalize(self, text: str) -> str:
        """Apply the case policy of this trie to ``text``."""
        return text if self._case_sensitive else text.lower()

    def _find(self, key: str) -> Optional[_Node]:
        """Return the node reached by following ``key``, or ``None``."""
        node = self._root
        for ch in key:
            child = node.children.get(ch)
            if child is None:
                return None
            node = child
        return node

    def insert(self, word: str, weight: float = 1.0) -> bool:
        """Store ``word`` with ``weight``.

        An existing word keeps its place and has its weight replaced.

        Returns:
            ``True`` when the word was new, ``False`` when it was updated.

        Raises:
            ValueError: if ``word`` is empty.
        """
        key = self._normalize(word)
        if not key:
            raise ValueError("cannot insert an empty word")
        node = self._root
        for ch in key:
            child = node.children.get(ch)
            if child is None:
                child = _Node()
                node.children[ch] = child
            node = child
        created = not node.is_word
        node.is_word = True
        node.weight = float(weight)
        if created:
            self._size += 1
        return created

    def weight(self, word: str) -> Optional[float]:
        """Return the weight of ``word`` or ``None`` when it is not stored."""
        node = self._find(self._normalize(word))
        if node is None or not node.is_word:
            return None
        return node.weight

    def increment(self, word: str, amount: float = 1.0) -> float:
        """Add ``amount`` to the weight of ``word``, inserting it if needed.

        Returns:
            The new weight of the word.
        """
        current = self.weight(word)
        if current is None:
            self.insert(word, amount)
            return float(amount)
        updated = current + amount
        self.insert(word, updated)
        return updated

    def delete(self, word: str) -> bool:
        """Remove ``word`` and prune the branch that only it was using.

        Returns:
            ``True`` when the word was stored and has been removed.
        """
        key = self._normalize(word)
        path: List[Tuple[_Node, str]] = []
        node = self._root
        for ch in key:
            child = node.children.get(ch)
            if child is None:
                return False
            path.append((node, ch))
            node = child
        if not node.is_word:
            return False
        node.is_word = False
        node.weight = 0.0
        self._size -= 1
        for parent, ch in reversed(path):
            child = parent.children[ch]
            if child.children:
                break
            del parent.children[ch]
        return True

    def has_prefix(self, prefix: str) -> bool:
        """Return ``True`` when at least one stored word starts with ``prefix``."""
        if self._size == 0:
            return False
        return self._find(self._normalize(prefix)) is not None

    def _walk(self, node: _Node, prefix: str) -> Iterator[Tuple[str, float]]:
        """Yield ``(word, weight)`` below ``node`` in lexicographic order."""
        if node.is_word:
            yield prefix, node.weight
        for ch in sorted(node.children):
            yield from self._walk(node.children[ch], prefix + ch)

    def iter_words(self, prefix: str = "") -> Iterator[Tuple[str, float]]:
        """Iterate over ``(word, weight)`` pairs that start with ``prefix``.

        Words are produced in lexicographic order.
        """
        key = self._normalize(prefix)
        node = self._find(key)
        if node is None:
            return
        yield from self._walk(node, key)

    def count_prefix(self, prefix: str) -> int:
        """Return how many stored words start with ``prefix``."""
        return sum(1 for _ in self.iter_words(prefix))

    def complete(self, prefix: str, k: int = 5) -> List[str]:
        """Return up to ``k`` completions of ``prefix``, heaviest first.

        Words of equal weight are ordered lexicographically.  The prefix
        itself is included when it is a stored word.
        """
        if k <= 0:
            return []
        best = heapq.nsmallest(
            k,
            self.iter_words(prefix),
            key=lambda item: (-item[1], item[0]),
        )
        return [word for word, _ in best]

    def longest_common_prefix(self) -> str:
        """Return the longest prefix shared by every stored word."""
        chars: List[str] = []
        node = self._root
        while len(node.children) == 1 and not node.is_word:
            ch, child = next(iter(node.children.items()))
            chars.append(ch)
            node = child
        return "".join(chars)

    def clear(self) -> None:
        """Remove every word."""
        self._root = _Node()
        self._size = 0
