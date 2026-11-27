"""Behaviour check for topo_sort_graph. Usage: check_topo_sort_graph.py <module path>."""
import importlib.util
import sys


def load(path):
    spec = importlib.util.spec_from_file_location("module_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def longest_depths(edges, nodes):
    deps = {n: set() for n in nodes}
    for node, dep in edges:
        deps[node].add(dep)
    memo = {}

    def depth(n):
        if n not in memo:
            memo[n] = 1 + max((depth(d) for d in deps[n]), default=-1)
        return memo[n]

    return {n: depth(n) for n in nodes}


def main():
    m = load(sys.argv[1])
    g = m.DependencyGraph()
    assert g.levels() == [] and g.topological_order() == []
    g.add_dependency("app", "lib")
    g.add_dependency("lib", "core")
    g.add_dependency("app", "core")
    g.add_node("docs")
    assert g.nodes() == ["app", "core", "docs", "lib"] and len(g) == 4
    assert g.topological_order() == ["core", "docs", "lib", "app"]
    assert g.dependencies_of("app") == ["core", "lib"]
    assert g.dependents_of("core") == ["app", "lib"]
    assert g.roots() == ["core", "docs"] and g.leaves() == ["app", "docs"]
    assert g.transitive_dependencies("app") == {"lib", "core"}
    assert g.build_order_for(["lib"]) == ["core", "lib"]
    assert not g.has_cycle() and g.find_cycle() is None
    assert g.levels() == [["core", "docs"], ["lib"], ["app"]]
    try:
        g.add_dependency("x", "x")
        raise SystemExit("self dependency accepted")
    except ValueError:
        pass

    # uneven diamond: d needs the long chain a -> b and the short root c
    h = m.DependencyGraph()
    edges = [("b", "a"), ("d", "b"), ("d", "c")]
    for node, dep in edges:
        h.add_dependency(node, dep)
    assert h.topological_order() == ["a", "b", "c", "d"]
    assert h.levels() == [["a", "c"], ["b"], ["d"]], h.levels()

    # larger layered graph compared with a longest-path reference
    edges = [("n1", "n0"), ("n2", "n1"), ("n3", "n2"), ("n9", "n3"), ("n9", "n8"),
             ("n5", "n4"), ("n6", "n5"), ("n6", "n0"), ("n7", "n6"), ("n7", "n8"),
             ("n3", "n8"), ("n2", "n4")]
    k = m.DependencyGraph()
    for node, dep in edges:
        k.add_dependency(node, dep)
    want = longest_depths(edges, k.nodes())
    layers = k.levels()
    for depth, layer in enumerate(layers):
        assert layer == sorted(layer)
        for name in layer:
            assert want[name] == depth, (name, depth, want[name])
    assert sum(len(layer) for layer in layers) == len(k)
    order = k.topological_order()
    for node, dep in edges:
        assert order.index(dep) < order.index(node)

    # cycles
    c = m.DependencyGraph()
    c.add_dependency("a", "b")
    c.add_dependency("b", "c")
    c.add_dependency("c", "a")
    c.add_dependency("z", "a")
    assert c.has_cycle() and c.find_cycle() == ["a", "b", "c", "a"]
    for call in (c.topological_order, c.levels):
        try:
            call()
            raise SystemExit("cycle not reported")
        except m.CycleError as err:
            assert err.cycle == ["a", "b", "c", "a"]
    c.remove_node("c")
    assert not c.has_cycle() and "c" not in c
    assert c.topological_order() == ["b", "a", "z"]
    assert c.levels() == [["b"], ["a"], ["z"]]
    print("ok")


if __name__ == "__main__":
    main()
