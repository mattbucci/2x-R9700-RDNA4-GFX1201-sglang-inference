"""Small CSV reader and writer in the style of RFC 4180.

Supported syntax:

* records are separated by ``\\n``, ``\\r\\n`` or a lone ``\\r``; the last
  record may or may not be followed by a line break;
* fields are separated by a single-character delimiter;
* a field may be enclosed in quote characters, in which case it can contain
  delimiters, line breaks and quote characters, the latter written twice;
* a line without any character is not a record and is skipped, whereas a
  line holding only a quoted empty field is a record with one empty field;
* an empty field at the end of a record is kept, so ``a,`` has two fields.

In strict mode (the default) malformed quoting raises :class:`CsvError`; in
lenient mode stray characters are kept as literal text.  Writing is the
inverse of reading: ``parse(dumps(rows)) == rows`` for any list of non-empty
rows of strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence


class CsvError(ValueError):
    """Malformed CSV input; ``line`` is the 1-based physical line number."""

    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


@dataclass(frozen=True)
class Dialect:
    """Lexical conventions of a CSV flavour.

    Attributes:
        delimiter: single character separating fields.
        quote: single character enclosing quoted fields.
        strict: whether malformed quoting is an error.
        lineterminator: record separator used when writing.
    """

    delimiter: str = ","
    quote: str = '"'
    strict: bool = True
    lineterminator: str = "\r\n"

    def __post_init__(self) -> None:
        for name in ("delimiter", "quote"):
            value = getattr(self, name)
            if len(value) != 1 or value in "\r\n":
                raise ValueError(f"{name} must be a single non-newline character")
        if self.delimiter == self.quote:
            raise ValueError("delimiter and quote must differ")
        if self.lineterminator not in ("\n", "\r\n", "\r"):
            raise ValueError("unsupported lineterminator")


EXCEL = Dialect()
UNIX = Dialect(lineterminator="\n")
TSV = Dialect(delimiter="\t", lineterminator="\n")


def iter_records(text: str, dialect: Dialect = EXCEL) -> Iterator[List[str]]:
    """Yield the records of ``text`` one at a time as lists of fields.

    Raises:
        CsvError: on malformed quoting in strict mode, or when the input
            ends inside a quoted field (in either mode).
    """
    delimiter, quote = dialect.delimiter, dialect.quote
    row: List[str] = []
    field: List[str] = []
    in_quotes = False  # inside a quoted field
    closed = False  # the current field was quoted and its closing quote was read
    started = False  # the current record has consumed at least one character
    line = 1
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_quotes:
            if ch != quote:
                if ch == "\n" or (ch == "\r" and text[i + 1:i + 2] != "\n"):
                    line += 1
                field.append(ch)
                i += 1
            elif i + 1 < n and text[i + 1] == quote:
                field.append(quote)
                i += 2
            else:
                in_quotes = False
                closed = True
                i += 1
            continue
        if ch == delimiter:
            row.append("".join(field))
            field = []
            closed = False
            started = True
        elif ch == "\n" or ch == "\r":
            if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
                i += 1
            if started:
                row.append("".join(field))
                yield row
            row, field = [], []
            closed = started = False
            line += 1
        elif closed:
            if dialect.strict:
                raise CsvError("unexpected character after closing quote", line)
            field.append(ch)
        elif ch == quote and not field:
            in_quotes = True
            started = True
        elif ch == quote and dialect.strict:
            raise CsvError("quote character inside an unquoted field", line)
        else:
            field.append(ch)
            started = True
        i += 1
    if in_quotes:
        raise CsvError("input ends inside a quoted field", line)
    if field:
        row.append("".join(field))
        yield row


def parse(text: str, dialect: Dialect = EXCEL) -> List[List[str]]:
    """Parse ``text`` completely and return its records."""
    return list(iter_records(text, dialect))


def parse_dicts(text: str, dialect: Dialect = EXCEL) -> List[Dict[str, str]]:
    """Parse ``text`` whose first record is a header.

    Every following record becomes a dictionary keyed by the header names.

    Raises:
        ValueError: the header has duplicate names or a record does not have
            exactly one field per header name.
    """
    records = iter_records(text, dialect)
    try:
        header = next(records)
    except StopIteration:
        return []
    if len(set(header)) != len(header):
        raise ValueError("duplicate column name in header")
    result: List[Dict[str, str]] = []
    for number, record in enumerate(records, start=2):
        if len(record) != len(header):
            raise ValueError(
                f"record {number} has {len(record)} fields, expected {len(header)}"
            )
        result.append(dict(zip(header, record)))
    return result


def needs_quoting(value: str, dialect: Dialect = EXCEL) -> bool:
    """Whether ``value`` must be quoted to survive a round trip."""
    return any(ch in value for ch in (dialect.delimiter, dialect.quote, "\n", "\r"))


def format_field(value: str, dialect: Dialect = EXCEL) -> str:
    """Render one field, quoting and escaping it when required."""
    if not needs_quoting(value, dialect):
        return value
    quote = dialect.quote
    return quote + value.replace(quote, quote + quote) + quote


def format_row(row: Sequence[str], dialect: Dialect = EXCEL) -> str:
    """Render one record without its line terminator.

    A record consisting of a single empty field is written as an empty
    quoted field so that it is not mistaken for a blank line.
    """
    if len(row) == 0:
        raise ValueError("a record needs at least one field")
    if len(row) == 1 and row[0] == "":
        return dialect.quote * 2
    return dialect.delimiter.join(format_field(value, dialect) for value in row)


def dumps(rows: Iterable[Sequence[str]], dialect: Dialect = EXCEL) -> str:
    """Render ``rows`` as CSV text; every record ends with the terminator."""
    return "".join(format_row(row, dialect) + dialect.lineterminator for row in rows)


def column(records: Sequence[Sequence[str]], index: int, default: str = "") -> List[str]:
    """Extract one column, substituting ``default`` for short records."""
    if index < 0:
        raise ValueError("index must not be negative")
    return [record[index] if index < len(record) else default for record in records]
