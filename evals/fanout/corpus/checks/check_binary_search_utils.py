"""Behaviour check for binary_search_utils. Usage: check_binary_search_utils.py <module path>."""
import bisect
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
    data = [1, 2, 2, 2, 5, 7, 7, 10]
    for v in range(-1, 13):
        assert m.lower_bound(data, v) == bisect.bisect_left(data, v), v
        assert m.upper_bound(data, v) == bisect.bisect_right(data, v), v
        assert m.contains(data, v) == (v in data), v
        assert m.count_equal(data, v) == data.count(v), v
    assert m.lower_bound([], 3) == 0 and m.upper_bound([], 3) == 0
    assert m.index_of(data, 2) == 1 and m.last_index_of(data, 2) == 3
    assert m.index_of(data, 3) == -1 and m.last_index_of(data, 3) == -1
    for lo in range(-1, 12):
        for hi in range(-1, 12):
            assert m.count_in_range(data, lo, hi) == sum(1 for x in data if lo <= x <= hi)
    assert m.floor_value(data, 6) == 5 and m.floor_value(data, 0) is None
    assert m.ceiling_value(data, 6) == 7 and m.ceiling_value(data, 11) is None
    assert m.closest_value(data, 6) == 5 and m.closest_value(data, 9) == 10
    assert m.closest_value([], 1) is None and m.closest_value(data, 100) == 10
    pairs = [(1, "a"), (2, "b"), (2, "c"), (4, "d")]
    assert m.insert_sorted(pairs, (2, "z"), key=lambda p: p[0]) == 3
    assert pairs == [(1, "a"), (2, "b"), (2, "c"), (2, "z"), (4, "d")]
    nums = [1, 3, 5]
    assert m.insert_unique(nums, 3) is False and m.insert_unique(nums, 4) is True
    assert nums == [1, 3, 4, 5]
    assert m.merge_sorted([1, 4, 6], [2, 4, 9]) == [1, 2, 4, 4, 6, 9]
    assert m.rotation_point([30, 40, 50, 10, 20]) == 3
    assert m.rotation_point([]) == 0 and m.rotation_point([1, 2, 3]) == 0
    assert m.unrotate([30, 40, 50, 10, 20]) == [10, 20, 30, 40, 50]
    assert m.search_rotated([30, 40, 50, 10, 20], 10) == 3
    assert m.search_rotated([], 1) == -1
    for n in range(1, 9):
        base = [10 * (i + 1) for i in range(n)]
        for r in range(n):
            arr = base[r:] + base[:r]
            for idx, val in enumerate(arr):
                assert m.search_rotated(arr, val) == idx, (arr, val)
            for missing in (5, 15, 10 * n + 5):
                assert m.search_rotated(arr, missing) == -1, (arr, missing)
    print("ok")


if __name__ == "__main__":
    main()
