"""Money as an integer number of cents, with explicit rounding.

Amounts are never stored as floats.  Every operation that can produce a
fraction of a cent takes a :class:`Rounding` mode, and :meth:`Money.allocate`
splits an amount by integer ratios so that the parts always add up to the
original amount.

Only currencies with two decimal places are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from typing import Iterable, List, Sequence, Union

Factor = Union[int, str, Decimal, Fraction]

CENTS_PER_UNIT = 100


class Rounding(Enum):
    """How to resolve a result that falls between two whole cents."""

    DOWN = "down"            # toward zero
    UP = "up"                # away from zero
    HALF_UP = "half_up"      # nearest cent, ties away from zero
    HALF_EVEN = "half_even"  # nearest cent, ties to the even cent


class CurrencyMismatch(ValueError):
    """Raised when two amounts in different currencies are combined."""


def divide_rounded(numerator: int, denominator: int, mode: Rounding) -> int:
    """Return ``numerator / denominator`` rounded to an integer.

    Rounding is symmetric around zero: the magnitude is rounded and the sign
    is applied afterwards, so ``-x`` always rounds to the negation of ``x``.
    """
    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    negative = (numerator < 0) != (denominator < 0)
    divisor = abs(denominator)
    quotient, remainder = divmod(abs(numerator), divisor)
    if remainder:
        twice = 2 * remainder
        if mode is Rounding.UP:
            bump = True
        elif mode is Rounding.DOWN:
            bump = False
        elif mode is Rounding.HALF_UP:
            bump = twice > divisor
        else:
            bump = twice > divisor or (twice == divisor and quotient % 2 == 1)
        if bump:
            quotient += 1
    return -quotient if negative else quotient


@dataclass(frozen=True)
class Money:
    """An amount of ``cents`` in one ISO 4217 ``currency``."""

    cents: int
    currency: str

    def __post_init__(self) -> None:
        if isinstance(self.cents, bool) or not isinstance(self.cents, int):
            raise TypeError("cents must be an integer")
        code = self.currency
        if not (isinstance(code, str) and len(code) == 3 and code.isascii()
                and code.isalpha() and code.isupper()):
            raise ValueError(f"invalid currency code {code!r}")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls, currency: str) -> "Money":
        """Return the zero amount in ``currency``."""
        return cls(0, currency)

    @classmethod
    def from_string(
        cls, text: str, currency: str, rounding: Rounding = Rounding.HALF_EVEN
    ) -> "Money":
        """Parse a decimal amount in major units, such as ``"12.34"``.

        Digits beyond the second decimal place are rounded with ``rounding``.
        """
        try:
            value = Decimal(text.strip())
        except InvalidOperation:
            raise ValueError(f"not a decimal amount: {text!r}") from None
        if not value.is_finite():
            raise ValueError(f"not a finite amount: {text!r}")
        exact = Fraction(value) * CENTS_PER_UNIT
        return cls(divide_rounded(exact.numerator, exact.denominator, rounding), currency)

    # ------------------------------------------------------------------
    # Currency checks and comparisons
    # ------------------------------------------------------------------
    def _require_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(
                f"cannot combine {self.currency} with {other.currency}"
            )

    def __lt__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.cents < other.cents

    def __le__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.cents <= other.cents

    def __gt__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.cents > other.cents

    def __ge__(self, other: "Money") -> bool:
        self._require_same_currency(other)
        return self.cents >= other.cents

    @property
    def is_zero(self) -> bool:
        """True for an amount of exactly zero cents."""
        return self.cents == 0

    @property
    def is_negative(self) -> bool:
        """True for an amount below zero."""
        return self.cents < 0

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: "Money") -> "Money":
        self._require_same_currency(other)
        return Money(self.cents + other.cents, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._require_same_currency(other)
        return Money(self.cents - other.cents, self.currency)

    def __neg__(self) -> "Money":
        return Money(-self.cents, self.currency)

    def multiply(self, factor: Factor, rounding: Rounding = Rounding.HALF_EVEN) -> "Money":
        """Return the amount times ``factor``, rounded to whole cents.

        ``factor`` may be an ``int``, a decimal string such as ``"0.175"``,
        a ``Decimal`` or a ``Fraction``.  Floats are rejected because their
        binary value is rarely the number the caller wrote down.
        """
        if isinstance(factor, float):
            raise TypeError("multiply by a string, Decimal or Fraction, not a float")
        ratio = Fraction(factor)
        cents = divide_rounded(self.cents * ratio.numerator, ratio.denominator, rounding)
        return Money(cents, self.currency)

    def allocate(self, ratios: Sequence[int]) -> List["Money"]:
        """Split the amount in proportion to ``ratios``; parts sum to the total.

        Each part first receives its share rounded toward zero.  The cents
        left over go one each to the parts with the largest discarded
        fraction, earlier parts winning ties.  A part with ratio 0 receives
        nothing.
        """
        if not ratios:
            raise ValueError("at least one ratio is required")
        for ratio in ratios:
            if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 0:
                raise ValueError("ratios must be non-negative integers")
        total = sum(ratios)
        if total == 0:
            raise ValueError("ratios must not all be zero")
        magnitude = abs(self.cents)
        shares = [magnitude * ratio // total for ratio in ratios]
        fractions = [magnitude * ratio % total for ratio in ratios]
        leftover = magnitude - sum(shares)
        by_fraction = sorted(range(len(ratios)), key=lambda i: (-fractions[i], i))
        for index in by_fraction[:leftover]:
            shares[index] += 1
        sign = -1 if self.cents < 0 else 1
        return [Money(sign * share, self.currency) for share in shares]

    def split(self, parts: int) -> List["Money"]:
        """Split the amount into ``parts`` portions that differ by at most a cent."""
        if isinstance(parts, bool) or not isinstance(parts, int) or parts < 1:
            raise ValueError("parts must be a positive integer")
        return self.allocate([1] * parts)

    def format(self) -> str:
        """Return a plain representation such as ``-12.05 EUR``."""
        units, minor = divmod(abs(self.cents), CENTS_PER_UNIT)
        sign = "-" if self.cents < 0 else ""
        return f"{sign}{units}.{minor:02d} {self.currency}"


def sum_money(amounts: Iterable[Money], currency: str) -> Money:
    """Add up ``amounts``, all of which must be in ``currency``."""
    total = Money.zero(currency)
    for amount in amounts:
        total = total + amount
    return total
