"""ISO 8601 duration parsing, formatting and arithmetic.

Supported syntax is the designator form ``PnYnMnWnDTnHnMnS`` with integer
components, for example ``P1Y2M10DT2H30M`` or ``PT45S``.  A leading ``-``
marks a negative duration (an extension that many APIs accept); a leading
``+`` is allowed and ignored.

Durations with calendar components have no exact length, so conversion to
seconds uses fixed conventions that are common for scheduling and retention
settings:

* one year is 365 days,
* one month is 30 days,
* one week is 7 days.

Arithmetic is carried out on the second count and yields a normalised
duration made of days, hours, minutes and seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365

_DIGITS = "0123456789"
_DATE_ORDER = "YMWD"
_TIME_ORDER = "HMS"
_DATE_FIELDS = {"Y": "years", "M": "months", "W": "weeks", "D": "days"}
_TIME_FIELDS = {"H": "hours", "M": "minutes", "S": "seconds"}
_FIELD_NAMES = ("years", "months", "weeks", "days", "hours", "minutes", "seconds")


class DurationError(ValueError):
    """Raised for text that is not a valid duration or for invalid values."""


@dataclass(frozen=True)
class Duration:
    """A duration split into ISO 8601 components.

    All components are non-negative integers; the sign of the whole duration
    is carried by ``negative``.
    """

    years: int = 0
    months: int = 0
    weeks: int = 0
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    negative: bool = False

    def __post_init__(self) -> None:
        for name in _FIELD_NAMES:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DurationError(f"{name} must be a non-negative integer")

    def is_zero(self) -> bool:
        """Return ``True`` when every component is zero."""
        return not any(getattr(self, name) for name in _FIELD_NAMES)

    def total_seconds(self) -> int:
        """Return the signed length in seconds under the module conventions."""
        days = self.days
        days += self.weeks * DAYS_PER_WEEK
        days += self.months * DAYS_PER_MONTH
        days += self.years * DAYS_PER_YEAR
        total = days * SECONDS_PER_DAY
        total += self.hours * SECONDS_PER_HOUR
        total += self.minutes * SECONDS_PER_HOUR
        total += self.seconds
        return -total if self.negative else total

    def normalized(self) -> "Duration":
        """Return the equivalent duration in days, hours, minutes, seconds."""
        return from_seconds(self.total_seconds())

    def __add__(self, other: "Duration") -> "Duration":
        if not isinstance(other, Duration):
            return NotImplemented
        return from_seconds(self.total_seconds() + other.total_seconds())

    def __sub__(self, other: "Duration") -> "Duration":
        if not isinstance(other, Duration):
            return NotImplemented
        return from_seconds(self.total_seconds() - other.total_seconds())

    def __neg__(self) -> "Duration":
        return from_seconds(-self.total_seconds())

    def scaled(self, factor: int) -> "Duration":
        """Return this duration multiplied by an integer factor."""
        if not isinstance(factor, int) or isinstance(factor, bool):
            raise DurationError("factor must be an integer")
        return from_seconds(self.total_seconds() * factor)

    def __str__(self) -> str:
        return format_duration(self)


def from_seconds(total: int) -> Duration:
    """Build a normalised duration from a signed number of seconds."""
    negative = total < 0
    rest = abs(total)
    days, rest = divmod(rest, SECONDS_PER_DAY)
    hours, rest = divmod(rest, SECONDS_PER_HOUR)
    minutes, seconds = divmod(rest, SECONDS_PER_MINUTE)
    return Duration(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        negative=negative,
    )


def _parse_section(section: str, order: str, fields: Dict[str, str]) -> Dict[str, int]:
    """Parse the date half or the time half of a duration.

    ``order`` lists the designators that are legal in this half in the order
    in which they must appear; each may occur at most once.
    """
    result: Dict[str, int] = {}
    digits = ""
    last_rank = -1
    for ch in section:
        if ch in _DIGITS:
            digits += ch
            continue
        rank = order.find(ch)
        if rank < 0:
            raise DurationError(f"unexpected designator {ch!r}")
        if not digits:
            raise DurationError(f"designator {ch!r} has no number")
        if rank <= last_rank:
            raise DurationError(f"designator {ch!r} is repeated or out of order")
        result[fields[ch]] = int(digits)
        digits = ""
        last_rank = rank
    if digits:
        raise DurationError("number without a designator")
    return result


def parse_duration(text: str) -> Duration:
    """Parse an ISO 8601 duration string.

    Raises :class:`DurationError` when the text is not a valid duration.
    """
    if not isinstance(text, str):
        raise DurationError("duration must be a string")
    body = text.strip()
    negative = body.startswith("-")
    if body[:1] in ("+", "-"):
        body = body[1:]
    if not body.startswith("P"):
        raise DurationError("duration must start with 'P'")
    date_part, separator, time_part = body[1:].partition("T")
    if separator and not time_part:
        raise DurationError("'T' must be followed by a time component")
    values = _parse_section(date_part, _DATE_ORDER, _DATE_FIELDS)
    values.update(_parse_section(time_part, _TIME_ORDER, _TIME_FIELDS))
    if not values:
        raise DurationError("duration has no components")
    return Duration(negative=negative, **values)


def format_duration(duration: Duration) -> str:
    """Format a duration as an ISO 8601 string.

    Zero components are omitted; the zero duration is written ``PT0S``.
    Components are written as stored, so format a :meth:`Duration.normalized`
    value to get carried units.
    """
    if duration.is_zero():
        return "PT0S"
    date_part = ""
    for designator in _DATE_ORDER:
        value = getattr(duration, _DATE_FIELDS[designator])
        if value:
            date_part += f"{value}{designator}"
    time_part = ""
    for designator in _TIME_ORDER:
        value = getattr(duration, _TIME_FIELDS[designator])
        if value:
            time_part += f"{value}{designator}"
    sign = "-" if duration.negative else ""
    if time_part:
        return f"{sign}P{date_part}T{time_part}"
    return f"{sign}P{date_part}"


def to_seconds(text: str) -> int:
    """Parse ``text`` and return its length in seconds."""
    return parse_duration(text).total_seconds()


def seconds_to_iso(total: int) -> str:
    """Format a number of seconds as a normalised ISO 8601 duration."""
    return format_duration(from_seconds(total))


def compare(left: Duration, right: Duration) -> int:
    """Three-way comparison by length: ``-1``, ``0`` or ``1``."""
    a = left.total_seconds()
    b = right.total_seconds()
    return (a > b) - (a < b)


def sum_durations(*texts: str) -> str:
    """Add several duration strings and return the normalised result."""
    total = sum(to_seconds(text) for text in texts)
    return seconds_to_iso(total)
