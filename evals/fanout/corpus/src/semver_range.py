"""Semantic version parsing, precedence and range matching.

Versions follow Semantic Versioning 2.0.0: ``MAJOR.MINOR.PATCH`` with an
optional pre-release (``-rc.1``) and optional build metadata (``+build.5``).
Precedence rules:

* major, minor and patch compare numerically;
* a version with a pre-release has lower precedence than the same version
  without one (``1.0.0-rc.1 < 1.0.0``);
* pre-release identifiers compare left to right: numeric identifiers
  numerically, alphanumeric ones in ASCII order, a numeric identifier is
  lower than an alphanumeric one, and when all shared identifiers are equal
  the longer list is higher;
* build metadata never affects precedence.

Ranges are comparator sets joined by ``||``.  A set is a whitespace-separated
list of comparators that must all hold: ``>=1.2.0 <2.0.0``, ``^1.2.3``,
``~1.2.3``, ``=1.0.0``, a bare version, or ``*``.  Versions inside a range
must be complete (no ``1.x`` shorthands).  Pre-release versions take part in
range matching by plain precedence.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

_IDENT = r"[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*"
_VERSION_RE = re.compile(
    r"^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    rf"(?:-({_IDENT}))?(?:\+({_IDENT}))?$"
)
_OPERATOR_GAP_RE = re.compile(r"(>=|<=|>|<|=|\^|~)\s+")
_OPERATORS = (">=", "<=", ">", "<", "=")


def _cmp(left, right) -> int:
    """Three-way comparison returning -1, 0 or 1."""
    return (left > right) - (left < right)


def _is_numeric(identifier: str) -> bool:
    """Return ``True`` for an identifier made only of ASCII digits."""
    return identifier.isascii() and identifier.isdigit()


@dataclass(frozen=True)
class Version:
    """A parsed semantic version."""

    major: int
    minor: int
    patch: int
    prerelease: Tuple[str, ...] = ()
    build: Tuple[str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Version":
        """Parse ``text`` into a :class:`Version`.

        A leading ``v`` is accepted.

        Raises:
            ValueError: if ``text`` is not a valid semantic version.
        """
        match = _VERSION_RE.match(text.strip())
        if match is None:
            raise ValueError(f"invalid semantic version: {text!r}")
        major, minor, patch, pre, build = match.groups()
        prerelease = tuple(pre.split(".")) if pre else ()
        for identifier in prerelease:
            if _is_numeric(identifier) and len(identifier) > 1 and identifier[0] == "0":
                raise ValueError(f"numeric identifier with leading zero: {text!r}")
        return cls(
            int(major),
            int(minor),
            int(patch),
            prerelease,
            tuple(build.split(".")) if build else (),
        )

    @property
    def core(self) -> Tuple[int, int, int]:
        """The ``(major, minor, patch)`` triple."""
        return (self.major, self.minor, self.patch)

    def is_prerelease(self) -> bool:
        """Return ``True`` when the version carries a pre-release tag."""
        return bool(self.prerelease)

    def __str__(self) -> str:
        text = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            text += "-" + ".".join(self.prerelease)
        if self.build:
            text += "+" + ".".join(self.build)
        return text


def _compare_identifiers(left: str, right: str) -> int:
    """Compare two pre-release identifiers."""
    left_numeric = _is_numeric(left)
    right_numeric = _is_numeric(right)
    if left_numeric and right_numeric:
        return _cmp(int(left), int(right))
    if left_numeric:
        return -1
    if right_numeric:
        return 1
    return _cmp(left, right)


def _compare_prerelease(left: Tuple[str, ...], right: Tuple[str, ...]) -> int:
    """Compare the pre-release parts of two versions with an equal core."""
    if not left or not right:
        return _cmp(bool(left), bool(right))
    for left_id, right_id in zip(left, right):
        result = _compare_identifiers(left_id, right_id)
        if result != 0:
            return result
    return _cmp(len(left), len(right))


def compare(left: Version, right: Version) -> int:
    """Return -1, 0 or 1 as ``left`` is lower, equal or higher in precedence."""
    result = _cmp(left.core, right.core)
    if result != 0:
        return result
    return _compare_prerelease(left.prerelease, right.prerelease)


@dataclass(frozen=True)
class Comparator:
    """A single ``operator version`` constraint."""

    operator: str
    version: Version

    def matches(self, candidate: Version) -> bool:
        """Return ``True`` when ``candidate`` satisfies this comparator."""
        result = compare(candidate, self.version)
        if self.operator == ">=":
            return result >= 0
        if self.operator == "<=":
            return result <= 0
        if self.operator == ">":
            return result > 0
        if self.operator == "<":
            return result < 0
        return result == 0


def _caret_upper(version: Version) -> Version:
    """Return the exclusive upper bound of ``^version``.

    The bound bumps the left-most non-zero component.
    """
    if version.major > 0:
        return Version(version.major + 1, 0, 0)
    if version.minor > 0:
        return Version(0, version.minor + 1, 0)
    return Version(0, 0, version.patch + 1)


def _expand(token: str) -> List[Comparator]:
    """Translate one range token into primitive comparators."""
    if token == "*":
        return []
    if token.startswith("^"):
        base = Version.parse(token[1:])
        return [Comparator(">=", base), Comparator("<", _caret_upper(base))]
    if token.startswith("~"):
        base = Version.parse(token[1:])
        upper = Version(base.major, base.minor + 1, 0)
        return [Comparator(">=", base), Comparator("<", upper)]
    for operator in _OPERATORS:
        if token.startswith(operator):
            return [Comparator(operator, Version.parse(token[len(operator):]))]
    return [Comparator("=", Version.parse(token))]


class VersionRange:
    """A parsed range: alternatives of comparator sets."""

    def __init__(self, alternatives: List[List[Comparator]]) -> None:
        self._alternatives = alternatives

    @classmethod
    def parse(cls, text: str) -> "VersionRange":
        """Parse a range expression.

        Raises:
            ValueError: if a version inside the range is malformed.
        """
        alternatives: List[List[Comparator]] = []
        for part in text.split("||"):
            tokens = _OPERATOR_GAP_RE.sub(r"\1", part.strip()).split()
            comparators: List[Comparator] = []
            for token in tokens:
                comparators.extend(_expand(token))
            alternatives.append(comparators)
        return cls(alternatives)

    def matches(self, version: Version) -> bool:
        """Return ``True`` when ``version`` satisfies any comparator set."""
        return any(
            all(comparator.matches(version) for comparator in comparators)
            for comparators in self._alternatives
        )


def satisfies(version: str, range_text: str) -> bool:
    """Return ``True`` when the version string lies inside the range."""
    return VersionRange.parse(range_text).matches(Version.parse(version))


def sort_versions(versions: Iterable[str]) -> List[str]:
    """Return the version strings sorted by ascending precedence."""
    parsed = [(Version.parse(text), text) for text in versions]
    parsed.sort(key=functools.cmp_to_key(lambda a, b: compare(a[0], b[0])))
    return [text for _, text in parsed]


def max_satisfying(versions: Iterable[str], range_text: str) -> Optional[str]:
    """Return the highest version inside the range, or ``None``."""
    version_range = VersionRange.parse(range_text)
    matching = [v for v in versions if version_range.matches(Version.parse(v))]
    if not matching:
        return None
    return sort_versions(matching)[-1]
