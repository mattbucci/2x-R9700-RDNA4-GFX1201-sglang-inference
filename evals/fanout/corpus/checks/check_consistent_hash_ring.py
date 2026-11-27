"""Behaviour checks for consistent_hash_ring. Usage: check_consistent_hash_ring.py <module path>."""
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
    m = load(sys.argv[1], "consistent_hash_ring_under_test")
    HashRing, RingError = m.HashRing, m.RingError
    keys = [f"user:{i}" for i in range(600)]

    ring = HashRing(["alpha", "beta", "gamma"], vnodes=50)
    assert len(ring) == 3 and "beta" in ring and "delta" not in ring
    assert ring.nodes == ["alpha", "beta", "gamma"]
    assert ring.point_count("alpha") == 50
    assert all(ring.get_node(k) in ring for k in keys)
    assert [ring.get_node(k) for k in keys] == [
        HashRing(["gamma", "alpha", "beta"], vnodes=50).get_node(k) for k in keys]

    counts = ring.distribution(keys)
    assert sum(counts.values()) == len(keys)
    assert all(count > 60 for count in counts.values()), counts
    shares = ring.ownership()
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert all(0.15 < share < 0.6 for share in shares.values()), shares

    for k in keys:
        assert ring.get_nodes(k, 1) == [ring.get_node(k)], k
        pair = ring.get_nodes(k, 2)
        assert len(pair) == 2 and pair[0] == ring.get_node(k) and pair[0] != pair[1], k
        assert sorted(ring.get_nodes(k, 3)) == ["alpha", "beta", "gamma"], k
        assert ring.get_nodes(k, 3)[:2] == pair, k
        assert len(ring.get_nodes(k, 10)) == 3, k
    assert raises(RingError, ring.get_nodes, "x", 0)

    # The second replica is the primary once the first node is gone.
    smaller = HashRing(["alpha", "beta", "gamma"], vnodes=50)
    smaller.remove_node("beta")
    assert smaller.nodes == ["alpha", "gamma"]
    for k in keys:
        before = ring.get_nodes(k, 2)
        if before[0] == "beta":
            assert smaller.get_node(k) == before[1], k
        else:
            assert smaller.get_node(k) == before[0], k
    assert raises(KeyError, smaller.remove_node, "beta")

    bigger = HashRing(["alpha", "beta", "gamma"], vnodes=50)
    bigger.add_node("delta")
    moved = ring.moved_keys(keys, bigger)
    assert moved and len(moved) < len(keys) // 2
    assert all(bigger.get_node(k) == "delta" for k in moved)
    assert raises(RingError, bigger.add_node, "delta")

    weighted = HashRing(vnodes=40)
    weighted.add_node("small")
    weighted.add_node("large", weight=4)
    assert weighted.weight("large") == 4 and weighted.point_count("large") == 160
    spread = weighted.distribution(keys)
    assert spread["large"] > 2 * spread["small"], spread
    weighted.remove_node("large")
    assert all(weighted.get_node(k) == "small" for k in keys)
    assert abs(weighted.ownership()["small"] - 1.0) < 1e-9 and len(weighted.ownership()) == 1

    empty = HashRing()
    assert raises(RingError, empty.get_node, "x")
    assert empty.distribution(keys) == {}
    assert raises(RingError, HashRing, ["a"], 0)
    single = HashRing(["only"], vnodes=1)
    assert single.get_nodes("k", 3) == ["only"] and single.ownership() == {"only": 1.0}
    print("ok")


if __name__ == "__main__":
    main()
