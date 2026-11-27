"""Parse and re-render pipe tables in the GitHub flavoured markdown style.

A table is a header row, a delimiter row that fixes the number of columns
and their alignment, and any number of body rows::

    | Name  | Qty |
    | :---- | --: |
    | apple |   3 |

Cells are stored unescaped.  A literal pipe inside a cell is written ``\\|``
in the markdown source and is escaped again on output.  Column widths are
measured in characters of the rendered (escaped) cell text.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

ALIGN_NONE = "none"
ALIGN_LEFT = "left"
ALIGN_CENTER = "center"
ALIGN_RIGHT = "right"
_ALIGNMENTS = (ALIGN_NONE, ALIGN_LEFT, ALIGN_CENTER, ALIGN_RIGHT)

# Narrowest column that still leaves room for a ":-:" delimiter cell.
MIN_WIDTH = 3

_DELIMITER_RE = re.compile(r":?-+:?")


class TableError(ValueError):
    """Raised when text is not a well-formed table or a row does not fit."""


def split_row(line: str) -> List[str]:
    """Split one table line into unescaped, stripped cell texts.

    Pipes preceded by a backslash belong to the cell.  One leading and one
    trailing pipe are treated as the table border and produce no cell.
    """
    text = line.strip()
    cells: List[str] = []
    current: List[str] = []
    ends_with_separator = False
    position = 0
    while position < len(text):
        char = text[position]
        if char == "\\" and text[position + 1 : position + 2] == "|":
            current.append("|")
            ends_with_separator = False
            position += 2
            continue
        if char == "|":
            cells.append("".join(current))
            current = []
            ends_with_separator = True
        else:
            current.append(char)
            ends_with_separator = False
        position += 1
    cells.append("".join(current))
    if ends_with_separator:
        cells.pop()
    if text.startswith("|") and cells:
        cells.pop(0)
    return [cell.strip() for cell in cells]


def escape_cell(text: str) -> str:
    """Return ``text`` as it must appear inside a table cell."""
    return text.replace("|", "\\|")


def parse_alignment(cell: str) -> str:
    """Return the alignment encoded by one delimiter-row cell."""
    spec = cell.strip()
    if not _DELIMITER_RE.fullmatch(spec):
        raise TableError(f"invalid delimiter cell {cell!r}")
    left = spec.startswith(":")
    right = spec.endswith(":")
    if left and right:
        return ALIGN_CENTER
    if right:
        return ALIGN_RIGHT
    if left:
        return ALIGN_LEFT
    return ALIGN_NONE


def _pad(text: str, width: int, alignment: str) -> str:
    """Pad ``text`` with spaces to ``width`` according to ``alignment``."""
    if alignment == ALIGN_RIGHT:
        return text.rjust(width)
    if alignment == ALIGN_CENTER:
        missing = max(width - len(text), 0)
        before = missing // 2
        return " " * before + text + " " * (missing - before)
    return text.ljust(width)


def _delimiter_cell(width: int, alignment: str) -> str:
    """Build the delimiter-row cell for a column of the given width."""
    if alignment == ALIGN_CENTER:
        return ":" + "-" * (width - 2) + ":"
    if alignment == ALIGN_RIGHT:
        return "-" * width + ":"
    if alignment == ALIGN_LEFT:
        return ":" + "-" * (width - 1)
    return "-" * width


class Table:
    """A markdown table: headers, per-column alignment and body rows."""

    def __init__(
        self,
        headers: Sequence[str],
        alignments: Optional[Sequence[str]] = None,
        rows: Iterable[Sequence[str]] = (),
    ) -> None:
        if not headers:
            raise TableError("a table needs at least one column")
        self.headers: List[str] = list(headers)
        if alignments is None:
            alignments = [ALIGN_NONE] * len(self.headers)
        if len(alignments) != len(self.headers):
            raise TableError("one alignment per column is required")
        for alignment in alignments:
            if alignment not in _ALIGNMENTS:
                raise TableError(f"unknown alignment {alignment!r}")
        self.alignments: List[str] = list(alignments)
        self.rows: List[List[str]] = []
        for row in rows:
            self.add_row(row)

    @classmethod
    def parse(cls, text: str) -> "Table":
        """Parse markdown table text; blank lines are ignored.

        Body rows shorter than the header are padded with empty cells and
        longer rows are cut to the header width, as markdown renderers do.
        """
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise TableError("a table needs a header row and a delimiter row")
        headers = split_row(lines[0])
        delimiter = split_row(lines[1])
        if not headers or len(delimiter) != len(headers):
            raise TableError("header and delimiter rows differ in column count")
        table = cls(headers, [parse_alignment(cell) for cell in delimiter])
        for line in lines[2:]:
            table.add_row(split_row(line)[: len(headers)])
        return table

    def add_row(self, cells: Sequence[str]) -> None:
        """Append a body row, padding missing trailing cells with ``""``."""
        if len(cells) > len(self.headers):
            raise TableError(
                f"row has {len(cells)} cells, table has {len(self.headers)} columns"
            )
        row = list(cells)
        row.extend([""] * (len(self.headers) - len(row)))
        self.rows.append(row)

    def _column_index(self, header: str) -> int:
        try:
            return self.headers.index(header)
        except ValueError:
            raise KeyError(header) from None

    def column(self, header: str) -> List[str]:
        """Return the body cells of the column titled ``header``."""
        index = self._column_index(header)
        return [row[index] for row in self.rows]

    def set_alignment(self, header: str, alignment: str) -> None:
        """Change the alignment of the column titled ``header``."""
        if alignment not in _ALIGNMENTS:
            raise TableError(f"unknown alignment {alignment!r}")
        self.alignments[self._column_index(header)] = alignment

    def column_widths(self) -> List[int]:
        """Return the rendered width of every column, at least ``MIN_WIDTH``."""
        widths = [max(MIN_WIDTH, len(escape_cell(header))) for header in self.headers]
        for row in self.rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(escape_cell(cell)))
        return widths

    def _render_row(self, cells: Sequence[str], widths: Sequence[int]) -> str:
        padded = [
            _pad(escape_cell(cell), width, alignment)
            for cell, width, alignment in zip(cells, widths, self.alignments)
        ]
        return "| " + " | ".join(padded) + " |"

    def render(self) -> str:
        """Return the table as aligned markdown text without a final newline."""
        widths = self.column_widths()
        delimiter = [
            _delimiter_cell(width, alignment)
            for width, alignment in zip(widths, self.alignments)
        ]
        lines = [self._render_row(self.headers, widths)]
        lines.append("| " + " | ".join(delimiter) + " |")
        lines.extend(self._render_row(row, widths) for row in self.rows)
        return "\n".join(lines)


def format_table(text: str) -> str:
    """Parse ``text`` as a markdown table and return it re-rendered."""
    return Table.parse(text).render()
