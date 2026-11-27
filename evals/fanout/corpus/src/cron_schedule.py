"""Five-field cron expressions: parsing and next-fire-time computation.

Supported syntax per field: ``*``, single values, ranges (``a-b``), steps
(``*/n``, ``a-b/n``, ``a/n``) and comma separated lists of those.  Month and
day-of-week fields also accept three-letter English names.  Day of week runs
from 0 (Sunday) to 6 (Saturday); 7 is accepted as another spelling of Sunday.

As in classic cron, when both the day-of-month and the day-of-week field are
restricted (neither starts with ``*``), a day matches when either field does.

All arithmetic is done on the wall-clock fields of the given ``datetime``;
daylight saving transitions are not modelled.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, FrozenSet, List, NamedTuple, Set

_MONTH_NAMES = {
    name: number
    for number, name in enumerate(
        "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1
    )
}
_DAY_NAMES = {name: number for number, name in enumerate("sun mon tue wed thu fri sat".split())}

# How many years ahead next_fire searches before giving up.
_SEARCH_YEARS = 8


class CronError(ValueError):
    """Raised for malformed expressions and schedules that never fire."""


class FieldSpec(NamedTuple):
    """Name, inclusive bounds and symbolic names of one cron field."""

    name: str
    low: int
    high: int
    names: Dict[str, int]


MINUTE = FieldSpec("minute", 0, 59, {})
HOUR = FieldSpec("hour", 0, 23, {})
DAY_OF_MONTH = FieldSpec("day of month", 1, 31, {})
MONTH = FieldSpec("month", 1, 12, _MONTH_NAMES)
DAY_OF_WEEK = FieldSpec("day of week", 0, 7, _DAY_NAMES)


def _parse_value(text: str, spec: FieldSpec) -> int:
    """Parse one number or symbolic name and check it against the bounds."""
    lowered = text.lower()
    if lowered in spec.names:
        return spec.names[lowered]
    if not (text.isascii() and text.isdigit()):
        raise CronError(f"invalid {spec.name} value {text!r}")
    value = int(text)
    if not spec.low <= value <= spec.high:
        raise CronError(
            f"{spec.name} value {value} outside {spec.low}-{spec.high}"
        )
    return value


def _parse_part(part: str, spec: FieldSpec) -> Set[int]:
    """Expand one list element (value, range or stepped range) to a set."""
    step = 1
    base = part
    stepped = "/" in part
    if stepped:
        base, step_text = part.split("/", 1)
        if not (step_text.isascii() and step_text.isdigit()) or int(step_text) == 0:
            raise CronError(f"invalid {spec.name} step {step_text!r}")
        step = int(step_text)
    if base == "*":
        start, end = spec.low, spec.high
    elif "-" in base:
        first, last = base.split("-", 1)
        start, end = _parse_value(first, spec), _parse_value(last, spec)
        if start > end:
            raise CronError(f"{spec.name} range {base!r} runs backwards")
    else:
        start = _parse_value(base, spec)
        end = spec.high if stepped else start
    return set(range(start, end + 1, step))


def parse_field(text: str, spec: FieldSpec) -> FrozenSet[int]:
    """Return the set of values one cron field matches."""
    values: Set[int] = set()
    for part in text.split(","):
        if not part:
            raise CronError(f"empty element in {spec.name} field {text!r}")
        values |= _parse_part(part, spec)
    if spec is DAY_OF_WEEK:
        values = {value % 7 for value in values}
    return frozenset(values)


class CronSchedule:
    """A parsed five-field cron expression."""

    def __init__(self, expression: str) -> None:
        parts = expression.split()
        if len(parts) != 5:
            raise CronError(
                f"expected 5 fields (minute hour day month weekday), got {len(parts)}"
            )
        self.expression = " ".join(parts)
        self.minutes = parse_field(parts[0], MINUTE)
        self.hours = parse_field(parts[1], HOUR)
        self.days = parse_field(parts[2], DAY_OF_MONTH)
        self.months = parse_field(parts[3], MONTH)
        self.weekdays = parse_field(parts[4], DAY_OF_WEEK)
        self._day_restricted = not parts[2].startswith("*")
        self._weekday_restricted = not parts[4].startswith("*")

    def __repr__(self) -> str:
        return f"CronSchedule({self.expression!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CronSchedule):
            return NotImplemented
        return self.expression == other.expression

    def __hash__(self) -> int:
        return hash(self.expression)

    def _day_matches(self, day: date) -> bool:
        """Apply the day-of-month and day-of-week fields to a calendar day."""
        month_day_ok = day.day in self.days
        weekday_ok = day.isoweekday() % 7 in self.weekdays
        if self._day_restricted and self._weekday_restricted:
            return month_day_ok or weekday_ok
        return month_day_ok and weekday_ok

    def matches(self, moment: datetime) -> bool:
        """True when the schedule fires in the minute containing ``moment``."""
        return (
            moment.month in self.months
            and self._day_matches(moment.date())
            and moment.hour in self.hours
            and moment.minute in self.minutes
        )

    def next_fire(self, after: datetime) -> datetime:
        """Return the first fire time strictly later than ``after``.

        The result has zero seconds and microseconds and keeps the ``tzinfo``
        of ``after``.  Raises :class:`CronError` when nothing fires within the
        search horizon, for example for ``0 0 31 2 *``.
        """
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        horizon = after.year + _SEARCH_YEARS
        while candidate.year <= horizon:
            if candidate.month not in self.months:
                if candidate.month == 12:
                    year, month = candidate.year + 1, 1
                else:
                    year, month = candidate.year, candidate.month + 1
                candidate = candidate.replace(
                    year=year, month=month, day=1, hour=0, minute=0
                )
                continue
            if not self._day_matches(candidate.date()):
                candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if candidate.hour not in self.hours:
                candidate = candidate + timedelta(hours=1)
                continue
            if candidate.minute not in self.minutes:
                candidate = candidate + timedelta(minutes=1)
                continue
            return candidate
        raise CronError(
            f"{self.expression!r} does not fire within {_SEARCH_YEARS} years"
        )

    def upcoming(self, after: datetime, count: int) -> List[datetime]:
        """Return the next ``count`` fire times after ``after``, in order."""
        if count < 0:
            raise ValueError("count must not be negative")
        times: List[datetime] = []
        moment = after
        for _ in range(count):
            moment = self.next_fire(moment)
            times.append(moment)
        return times

    def fires_between(self, start: datetime, end: datetime) -> int:
        """Count the fire times in the half-open interval ``(start, end]``."""
        fired = 0
        moment = start
        while True:
            try:
                moment = self.next_fire(moment)
            except CronError:
                return fired
            if moment > end:
                return fired
            fired += 1


def next_fire(expression: str, after: datetime) -> datetime:
    """Parse ``expression`` and return its first fire time after ``after``."""
    return CronSchedule(expression).next_fire(after)
