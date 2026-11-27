"""Behaviour checks for cron_schedule. Usage: check_cron_schedule.py <module path>."""
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
    from datetime import datetime

    m = load(sys.argv[1], "cron_schedule_under_test")
    Cron, CronError = m.CronSchedule, m.CronError

    assert m.parse_field("*/15", m.MINUTE) == {0, 15, 30, 45}
    assert m.parse_field("1,3,5-7", m.HOUR) == {1, 3, 5, 6, 7}
    assert m.parse_field("10-20/5", m.MINUTE) == {10, 15, 20}
    assert m.parse_field("50/5", m.MINUTE) == {50, 55}
    assert m.parse_field("*", m.DAY_OF_MONTH) == set(range(1, 32))
    assert m.parse_field("7", m.DAY_OF_WEEK) == {0}
    assert m.parse_field("*", m.DAY_OF_WEEK) == set(range(7))
    assert m.parse_field("mon-fri", m.DAY_OF_WEEK) == {1, 2, 3, 4, 5}
    assert m.parse_field("JAN,dec", m.MONTH) == {1, 12}

    for bad in ("60 * * * *", "* * *", "*/0 * * * *", "5-1 * * * *",
                "* 24 * * *", "* * 0 * *", "* * * 13 *", "1,,2 * * * *", "x * * * *"):
        assert raises(CronError, Cron, bad), bad

    def nxt(expr, *when):
        return m.next_fire(expr, datetime(*when))

    assert nxt("*/5 * * * *", 2024, 3, 10, 10, 5) == datetime(2024, 3, 10, 10, 10)
    assert nxt("*/5 * * * *", 2024, 3, 10, 10, 4, 30) == datetime(2024, 3, 10, 10, 5)
    assert nxt("* * * * *", 2024, 12, 31, 23, 59, 59) == datetime(2025, 1, 1, 0, 0)
    assert nxt("30 14 1 * *", 2024, 1, 15, 9, 0) == datetime(2024, 2, 1, 14, 30)
    assert nxt("0 0 29 2 *", 2023, 1, 1) == datetime(2024, 2, 29)
    assert nxt("0 0 1 1 *", 2024, 6, 1, 12, 0) == datetime(2025, 1, 1)
    assert nxt("0 12 * * 0", 2024, 3, 8, 13, 0) == datetime(2024, 3, 10, 12, 0)
    assert nxt("0 12 * * sun", 2024, 3, 10, 12, 0) == datetime(2024, 3, 17, 12, 0)
    assert nxt("0 0 13 * 5", 2024, 9, 1) == datetime(2024, 9, 6)
    assert nxt("0 0 13 * 5", 2024, 9, 6) == datetime(2024, 9, 13)
    assert nxt("0 0 13 * *", 2024, 9, 1) == datetime(2024, 9, 13)

    assert nxt("0 9 * * *", 2024, 3, 10, 8, 30) == datetime(2024, 3, 10, 9, 0)
    assert nxt("15 14 * * *", 2024, 3, 11, 10, 40) == datetime(2024, 3, 11, 14, 15)
    assert nxt("0 */6 * * *", 2024, 3, 11, 1, 1) == datetime(2024, 3, 11, 6, 0)
    assert nxt("5 0 * 8 *", 2024, 7, 31, 23, 50) == datetime(2024, 8, 1, 0, 5)
    assert raises(CronError, nxt, "0 0 31 2 *", 2024, 1, 1)

    s = Cron("0 9,17 * * 1-5")
    assert s.matches(datetime(2024, 3, 11, 9, 0, 42))
    assert not s.matches(datetime(2024, 3, 10, 9, 0))
    assert not s.matches(datetime(2024, 3, 11, 9, 1))
    assert s.upcoming(datetime(2024, 3, 8, 9, 0), 3) == [
        datetime(2024, 3, 8, 17, 0), datetime(2024, 3, 11, 9, 0), datetime(2024, 3, 11, 17, 0)]
    assert s.upcoming(datetime(2024, 3, 8, 9, 0), 0) == []
    assert s.fires_between(datetime(2024, 3, 11, 0, 0), datetime(2024, 3, 12, 17, 0)) == 4
    assert Cron("0  9 * * *") == Cron("0 9 * * *")
    print("ok")


if __name__ == "__main__":
    main()
