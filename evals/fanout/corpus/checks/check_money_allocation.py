"""Behaviour checks for money_allocation. Usage: check_money_allocation.py <module path>."""
import importlib.util
import sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def raises(exc, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc:
        return True
    return False



def main():
    from decimal import Decimal
    from fractions import Fraction

    m = load(sys.argv[1], "money_allocation_under_test")
    Money, R = m.Money, m.Rounding
    usd = lambda cents: Money(cents, "USD")

    d = m.divide_rounded
    assert [d(7, 2, mode) for mode in (R.DOWN, R.UP, R.HALF_EVEN)] == [3, 4, 4]
    assert [d(5, 2, mode) for mode in (R.DOWN, R.UP, R.HALF_EVEN)] == [2, 3, 2]
    assert [d(-7, 2, mode) for mode in (R.DOWN, R.UP, R.HALF_EVEN)] == [-3, -4, -4]
    assert [d(10, 3, mode) for mode in R] == [3, 4, 3, 3]
    assert [d(11, 3, mode) for mode in R] == [3, 4, 4, 4]
    assert [d(6, 3, mode) for mode in R] == [2, 2, 2, 2]
    assert d(7, -2, R.DOWN) == -3 and d(-7, -2, R.UP) == 4
    assert d(5, 2, R.HALF_UP) == 3 and d(7, 2, R.HALF_UP) == 4
    assert d(-5, 2, R.HALF_UP) == -3 and d(1, 2, R.HALF_UP) == 1
    assert raises(ZeroDivisionError, d, 1, 0, R.DOWN)

    assert Money.from_string("12.34", "USD") == usd(1234)
    assert Money.from_string(" -0.5 ", "USD") == usd(-50)
    assert Money.from_string("1.005", "USD") == usd(100)
    assert Money.from_string("1.015", "USD") == usd(102)
    assert Money.from_string("1.005", "USD", R.HALF_UP) == usd(101)
    assert Money.from_string("1.0049", "USD", R.HALF_UP) == usd(100)
    assert Money.from_string("1.001", "USD", R.UP) == usd(101)
    assert raises(ValueError, Money.from_string, "abc", "USD")
    assert raises(ValueError, Money.from_string, "NaN", "USD")
    assert raises(ValueError, Money, 1, "usd")
    assert raises(ValueError, Money, 1, "US")
    assert raises(TypeError, Money, 1.5, "USD")

    assert usd(150) + usd(275) == usd(425)
    assert usd(150) - usd(275) == usd(-125) and -usd(5) == usd(-5)
    assert usd(1) < usd(2) and usd(2) >= usd(2) and usd(3) > usd(2) and usd(2) <= usd(2)
    assert raises(m.CurrencyMismatch, lambda: usd(1) + Money(1, "EUR"))
    assert raises(m.CurrencyMismatch, lambda: usd(1) < Money(1, "EUR"))
    assert usd(1) != Money(1, "EUR")
    assert usd(0).is_zero and usd(-1).is_negative and not usd(1).is_negative

    assert usd(1000).multiply("0.175") == usd(175)
    assert usd(5).multiply("0.5") == usd(2) and usd(15).multiply("0.5") == usd(8)
    assert usd(5).multiply(Decimal("0.5"), R.HALF_UP) == usd(3)
    assert usd(-5).multiply(Fraction(1, 2), R.HALF_UP) == usd(-3)
    assert usd(333).multiply(3) == usd(999)
    assert raises(TypeError, usd(5).multiply, 0.5)

    assert usd(100).allocate([1, 1, 1]) == [usd(34), usd(33), usd(33)]
    assert usd(5).allocate([3, 7]) == [usd(2), usd(3)]
    assert usd(100).allocate([0, 1, 2]) == [usd(0), usd(33), usd(67)]
    assert usd(-100).allocate([1, 1, 1]) == [usd(-34), usd(-33), usd(-33)]
    assert usd(1).allocate([1, 2, 1]) == [usd(0), usd(1), usd(0)]
    for cents in (0, 1, 7, 99, 1001, -250):
        for ratios in ([1], [1, 2], [3, 3, 3, 1], [5, 0, 95], [7, 11, 13, 17]):
            parts = usd(cents).allocate(ratios)
            assert m.sum_money(parts, "USD") == usd(cents), (cents, ratios)
    assert raises(ValueError, usd(1).allocate, [])
    assert raises(ValueError, usd(1).allocate, [0, 0])
    assert raises(ValueError, usd(1).allocate, [1, -1])

    assert usd(10).split(3) == [usd(4), usd(3), usd(3)]
    assert raises(ValueError, usd(10).split, 0)
    assert usd(-1205).format() == "-12.05 USD" and usd(7).format() == "0.07 USD"
    assert raises(m.CurrencyMismatch, m.sum_money, [Money(1, "EUR")], "USD")
    print("ok")


if __name__ == "__main__":
    main()
