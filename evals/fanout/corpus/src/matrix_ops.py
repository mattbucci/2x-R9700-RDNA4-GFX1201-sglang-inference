"""Dense matrices in pure Python.

A small, dependency-free matrix type for exact or floating point work.
Entries may be ``int``, ``float`` or ``fractions.Fraction``.  Matrices that
hold only integers and fractions stay exact: the determinant eliminates over
``Fraction`` values, so no rounding error is introduced.

Instances are treated as values: every operation returns a new matrix and the
stored rows are never handed out without copying.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float, Fraction]


class MatrixError(ValueError):
    """Raised for shape mismatches and invalid matrix arguments."""


class Matrix:
    """A rectangular matrix stored in row-major order."""

    __slots__ = ("_data", "_rows", "_cols")

    def __init__(self, rows: Iterable[Sequence[Number]]) -> None:
        data = [list(row) for row in rows]
        if not data or not data[0]:
            raise MatrixError("a matrix needs at least one row and one column")
        width = len(data[0])
        for index, row in enumerate(data):
            if len(row) != width:
                raise MatrixError(
                    f"row {index} has {len(row)} entries, expected {width}"
                )
        self._data: List[List[Number]] = data
        self._rows = len(data)
        self._cols = width

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls, size: int) -> "Matrix":
        """Return the ``size`` x ``size`` identity matrix."""
        if size < 1:
            raise MatrixError("identity size must be at least 1")
        return cls([[1 if r == c else 0 for c in range(size)] for r in range(size)])

    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        """Return a ``rows`` x ``cols`` matrix filled with zeros."""
        if rows < 1 or cols < 1:
            raise MatrixError("matrix dimensions must be at least 1")
        return cls([[0] * cols for _ in range(rows)])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        """The ``(rows, columns)`` pair."""
        return (self._rows, self._cols)

    @property
    def is_square(self) -> bool:
        """True when the matrix has as many rows as columns."""
        return self._rows == self._cols

    def __getitem__(self, position: Tuple[int, int]) -> Number:
        row, col = position
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise IndexError(f"position {position} outside {self.shape}")
        return self._data[row][col]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self._data))

    def __repr__(self) -> str:
        return f"Matrix({self._data!r})"

    def to_lists(self) -> List[List[Number]]:
        """Return the entries as a fresh list of row lists."""
        return [list(row) for row in self._data]

    def row(self, index: int) -> List[Number]:
        """Return a copy of row ``index``."""
        if not 0 <= index < self._rows:
            raise IndexError(f"row {index} outside {self.shape}")
        return list(self._data[index])

    def column(self, index: int) -> List[Number]:
        """Return column ``index`` as a list."""
        if not 0 <= index < self._cols:
            raise IndexError(f"column {index} outside {self.shape}")
        return [row[index] for row in self._data]

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def transpose(self) -> "Matrix":
        """Return the transpose: entry ``(r, c)`` moves to ``(c, r)``."""
        return Matrix([list(col) for col in zip(*self._data)])

    def add(self, other: "Matrix") -> "Matrix":
        """Return the entrywise sum with a matrix of the same shape."""
        self._require_same_shape(other, "add")
        return Matrix(
            [[a + b for a, b in zip(mine, theirs)]
             for mine, theirs in zip(self._data, other._data)]
        )

    def subtract(self, other: "Matrix") -> "Matrix":
        """Return the entrywise difference with a matrix of the same shape."""
        self._require_same_shape(other, "subtract")
        return Matrix(
            [[a - b for a, b in zip(mine, theirs)]
             for mine, theirs in zip(self._data, other._data)]
        )

    def scale(self, factor: Number) -> "Matrix":
        """Return the matrix with every entry multiplied by ``factor``."""
        return Matrix([[factor * value for value in row] for row in self._data])

    def multiply(self, other: "Matrix") -> "Matrix":
        """Return the matrix product ``self @ other``."""
        if self._cols != other._rows:
            raise MatrixError(
                f"cannot multiply {self.shape} by {other.shape}: "
                "inner dimensions differ"
            )
        columns = list(zip(*other._data))
        return Matrix(
            [[sum(a * b for a, b in zip(row, col)) for col in columns]
             for row in self._data]
        )

    def __matmul__(self, other: "Matrix") -> "Matrix":
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.multiply(other)

    def trace(self) -> Number:
        """Return the sum of the main diagonal of a square matrix."""
        self._require_square("trace")
        return sum(self._data[i][i] for i in range(self._rows))

    def determinant(self) -> Number:
        """Return the determinant, computed by Gaussian elimination.

        Each column uses the remaining row with the largest magnitude entry
        as its pivot row.  Integer and ``Fraction`` matrices are eliminated
        exactly; an integer result is returned as ``int``.  Any ``float``
        entry switches the whole computation to floating point.
        """
        self._require_square("determinant")
        size = self._rows
        exact = all(
            isinstance(value, (int, Fraction)) for row in self._data for value in row
        )
        work = [
            [Fraction(value) if exact else float(value) for value in row]
            for row in self._data
        ]
        sign = 1
        result: Number = Fraction(1) if exact else 1.0
        for col in range(size):
            pivot = max(range(col, size), key=lambda r: work[r][col])
            if work[pivot][col] == 0:
                return 0 if exact else 0.0
            if pivot != col:
                work[col], work[pivot] = work[pivot], work[col]
                sign = -sign
            result *= work[col][col]
            for r in range(col + 1, size):
                factor = work[r][col] / work[col][col]
                if factor == 0:
                    continue
                for k in range(col + 1, size):
                    work[r][k] -= factor * work[col][k]
        result *= sign
        if exact and result.denominator == 1:
            return int(result)
        return result

    def power(self, exponent: int) -> "Matrix":
        """Return the matrix raised to a non-negative integer power.

        Uses binary exponentiation, so the number of multiplications grows
        with the logarithm of ``exponent``.  The zeroth power is the identity.
        """
        self._require_square("power")
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise MatrixError("exponent must be an integer")
        if exponent < 0:
            raise MatrixError("negative powers are not supported")
        result = Matrix.identity(self._rows)
        base = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result.multiply(base)
            remaining >>= 1
            if remaining:
                base = base.multiply(base)
        return result

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _require_square(self, operation: str) -> None:
        if not self.is_square:
            raise MatrixError(f"{operation} needs a square matrix, got {self.shape}")

    def _require_same_shape(self, other: "Matrix", operation: str) -> None:
        if self.shape != other.shape:
            raise MatrixError(
                f"cannot {operation} matrices of shape {self.shape} and {other.shape}"
            )
