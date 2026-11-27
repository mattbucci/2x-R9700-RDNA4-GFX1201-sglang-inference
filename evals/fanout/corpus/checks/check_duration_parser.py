"""Behaviour check for duration_parser. Usage: check_duration_parser.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("module_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    m = load(sys.argv[1])
    d = m.parse_duration("P1Y2M3W4DT5H6M7S")
    assert (d.years, d.months, d.weeks, d.days) == (1, 2, 3, 4)
    assert (d.hours, d.minutes, d.seconds, d.negative) == (5, 6, 7, False)
    assert m.format_duration(d) == "P1Y2M3W4DT5H6M7S" == str(d)
    assert m.parse_duration("P2M").months == 2 and m.parse_duration("PT2M").minutes == 2
    assert m.to_seconds("PT45S") == 45
    assert m.to_seconds("PT2H") == 7200
    assert m.to_seconds("P1D") == 86400
    assert m.to_seconds("P1W") == 7 * 86400
    assert m.to_seconds("P1M") == 30 * 86400
    assert m.to_seconds("P1Y") == 365 * 86400
    assert m.to_seconds("-P1DT1S") == -86401
    assert m.to_seconds("+PT1S") == 1
    assert m.to_seconds("PT30M") == 1800, m.to_seconds("PT30M")
    assert m.to_seconds("P1DT2H30M15S") == 86400 + 7200 + 1800 + 15
    for bad in ("", "P", "PT", "1D", "P1", "PD", "P1S", "PT1D", "P1D1Y", "P1D1D",
                "P1.5D", "P-1D", "PT1H1H", "P1DT", "T1H"):
        try:
            m.parse_duration(bad)
            raise SystemExit("accepted invalid duration %r" % bad)
        except m.DurationError:
            pass
    assert m.seconds_to_iso(0) == "PT0S"
    assert m.seconds_to_iso(90061) == "P1DT1H1M1S"
    assert m.seconds_to_iso(-3600) == "-PT1H"
    assert m.seconds_to_iso(86400) == "P1D"
    for total in (1, 59, 60, 61, 3599, 3600, 3661, 86399, 86400, 123456, -98765):
        assert m.to_seconds(m.seconds_to_iso(total)) == total, total
    assert m.format_duration(m.parse_duration("PT90M").normalized()) == "PT1H30M"
    a, b = m.parse_duration("PT1H"), m.parse_duration("PT20M")
    assert str(a + b) == "PT1H20M"
    assert str(a - b) == "PT40M"
    assert str(b - a) == "-PT40M"
    assert str(-a) == "-PT1H"
    assert str(b.scaled(3)) == "PT1H"
    assert m.compare(a, b) == 1 and m.compare(b, a) == -1
    assert m.compare(m.parse_duration("PT60M"), a) == 0
    assert m.sum_durations("PT45M", "PT30M", "P1D") == "P1DT1H15M"
    assert m.Duration().is_zero() and str(m.Duration(negative=True)) == "PT0S"
    try:
        m.Duration(hours=-1)
        raise SystemExit("negative component accepted")
    except m.DurationError:
        pass
    print("ok")


if __name__ == "__main__":
    main()
