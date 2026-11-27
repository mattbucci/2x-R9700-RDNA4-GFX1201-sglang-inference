"""Line diff based on the longest common subsequence.

The module compares two sequences of lines and produces

* the edit script (:func:`diff_ops`) as a list of ``equal`` / ``delete`` /
  ``insert`` operations,
* hunks with surrounding context (:func:`group_hunks`), and
* a unified-style text rendering (:func:`unified_diff`).

The algorithm is the classic quadratic dynamic programme.  It is meant for
configuration files and source files of moderate size, where a minimal and
predictable edit script matters more than raw speed.

Hunk headers follow the unified diff convention: ``@@ -start,count
+start,count @@`` with 1-based start lines.  When a side contributes no lines
to a hunk its start is the number of lines that precede the hunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Sequence, Tuple

EQUAL = "equal"
DELETE = "delete"
INSERT = "insert"

_PREFIX = {EQUAL: " ", DELETE: "-", INSERT: "+"}


class Op(NamedTuple):
    """One step of an edit script.

    ``a_index`` and ``b_index`` are the 0-based positions reached in the old
    and the new sequence when the operation is emitted.
    """

    tag: str
    a_index: int
    b_index: int
    text: str


@dataclass
class Hunk:
    """A run of changes together with its context lines."""

    a_start: int
    a_count: int
    b_start: int
    b_count: int
    lines: List[str] = field(default_factory=list)

    def header(self) -> str:
        """Return the ``@@ -a,b +c,d @@`` line for this hunk."""
        return f"@@ -{self.a_start},{self.a_count} +{self.b_start},{self.b_count} @@"


def lcs_table(a: Sequence[str], b: Sequence[str]) -> List[List[int]]:
    """Build the suffix LCS table.

    ``table[i][j]`` is the length of the longest common subsequence of
    ``a[i:]`` and ``b[j:]``.
    """
    rows, cols = len(a), len(b)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        row = table[i]
        below = table[i + 1]
        for j in range(cols - 1, -1, -1):
            if a[i] == b[j]:
                row[j] = below[j + 1] + 1
            else:
                row[j] = max(below[j], row[j + 1])
    return table


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Return the length of the longest common subsequence of ``a`` and ``b``."""
    return lcs_table(a, b)[0][0]


def diff_ops(a: Sequence[str], b: Sequence[str]) -> List[Op]:
    """Return the edit script that turns ``a`` into ``b``.

    Deletions are emitted before insertions when both are possible, which
    gives the familiar "old lines first" layout of replaced blocks.
    """
    table = lcs_table(a, b)
    ops: List[Op] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            ops.append(Op(EQUAL, i, j, a[i]))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            ops.append(Op(DELETE, i, j, a[i]))
            i += 1
        else:
            ops.append(Op(INSERT, i, j, b[j]))
            j += 1
    while i < len(a):
        ops.append(Op(DELETE, i, j, a[i]))
        i += 1
    while j < len(b):
        ops.append(Op(INSERT, i, j, b[j]))
        j += 1
    return ops


def longest_common_subsequence(a: Sequence[str], b: Sequence[str]) -> List[str]:
    """Return one longest common subsequence of ``a`` and ``b``."""
    return [op.text for op in diff_ops(a, b) if op.tag == EQUAL]


def apply_ops(ops: Sequence[Op]) -> List[str]:
    """Rebuild the new sequence from an edit script."""
    return [op.text for op in ops if op.tag != DELETE]


def original_from_ops(ops: Sequence[Op]) -> List[str]:
    """Rebuild the old sequence from an edit script."""
    return [op.text for op in ops if op.tag != INSERT]


def diff_stats(ops: Sequence[Op]) -> Tuple[int, int]:
    """Return ``(inserted, deleted)`` line counts of an edit script."""
    inserted = sum(1 for op in ops if op.tag == INSERT)
    deleted = sum(1 for op in ops if op.tag == DELETE)
    return inserted, deleted


def similarity_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """Return a similarity score in ``[0, 1]``; two empty inputs score 1."""
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    return 2.0 * lcs_length(a, b) / total


def _change_clusters(ops: Sequence[Op], context: int) -> List[Tuple[int, int]]:
    """Group changed operations whose context windows touch or overlap.

    Returns ``(first, last)`` index pairs into ``ops``; both ends are changed
    operations.  Two changes share a cluster when at most ``2 * context``
    unchanged lines separate them.
    """
    changed = [k for k, op in enumerate(ops) if op.tag != EQUAL]
    if not changed:
        return []
    clusters: List[Tuple[int, int]] = []
    first = last = changed[0]
    for k in changed[1:]:
        if k - last - 1 > 2 * context:
            clusters.append((first, last))
            first = k
        last = k
    clusters.append((first, last))
    return clusters


def _build_hunk(window: Sequence[Op]) -> Hunk:
    """Turn a slice of the edit script into a :class:`Hunk`."""
    a_count = sum(1 for op in window if op.tag != INSERT)
    b_count = sum(1 for op in window if op.tag != DELETE)
    head = window[0]
    a_start = head.a_index + 1 if a_count else head.a_index
    b_start = head.b_index + 1 if b_count else head.b_index
    lines = [_PREFIX[op.tag] + op.text for op in window]
    return Hunk(a_start, a_count, b_start, b_count, lines)


def group_hunks(ops: Sequence[Op], context: int = 3) -> List[Hunk]:
    """Split an edit script into hunks.

    Every hunk carries up to ``context`` unchanged lines before its first
    change and after its last change (fewer at the edges of the file).
    """
    if context < 0:
        raise ValueError("context must not be negative")
    hunks: List[Hunk] = []
    for first, last in _change_clusters(ops, context):
        lo = max(0, first - context)
        hi = min(len(ops), last + context)
        hunks.append(_build_hunk(ops[lo:hi]))
    return hunks


def unified_diff(
    a: Sequence[str],
    b: Sequence[str],
    context: int = 3,
    from_name: str = "a",
    to_name: str = "b",
) -> List[str]:
    """Render the difference between ``a`` and ``b`` as unified-style lines.

    Returns an empty list when the inputs are identical.
    """
    hunks = group_hunks(diff_ops(a, b), context)
    if not hunks:
        return []
    out = [f"--- {from_name}", f"+++ {to_name}"]
    for hunk in hunks:
        out.append(hunk.header())
        out.extend(hunk.lines)
    return out


def diff_text(old: str, new: str, context: int = 3) -> str:
    """Diff two multi-line strings and return the rendering as one string."""
    lines = unified_diff(old.splitlines(), new.splitlines(), context)
    return "\n".join(lines)
