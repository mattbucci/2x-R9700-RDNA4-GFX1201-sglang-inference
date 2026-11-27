"""Dependency graph with topological ordering.

:class:`DependencyGraph` records "X depends on Y" relations between named
nodes and answers the questions a build or task scheduler asks:

* in which order can everything run (:meth:`DependencyGraph.topological_order`),
* which nodes can run in parallel (:meth:`DependencyGraph.levels`),
* is there a cycle and which nodes form it (:meth:`DependencyGraph.find_cycle`),
* what does one target need (:meth:`DependencyGraph.transitive_dependencies`,
  :meth:`DependencyGraph.build_order_for`).

All results are deterministic: wherever several nodes are equally eligible
they are reported in ascending name order.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, Iterator, List, Optional, Set


class CycleError(Exception):
    """Raised when an ordering is requested for a graph that has a cycle.

    ``cycle`` lists the nodes of one cycle; the first node is repeated at the
    end, and every node depends on the one that follows it.
    """

    def __init__(self, cycle: List[str]) -> None:
        super().__init__("dependency cycle: " + " -> ".join(cycle))
        self.cycle = cycle


class DependencyGraph:
    """A directed graph of named nodes and their dependencies."""

    def __init__(self) -> None:
        self._deps: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = {}

    def add_node(self, name: str) -> None:
        """Register ``name``; adding an existing node is a no-op."""
        self._deps.setdefault(name, set())
        self._dependents.setdefault(name, set())

    def add_dependency(self, node: str, depends_on: str) -> None:
        """Record that ``node`` depends on ``depends_on``.

        Both nodes are created when missing.  Adding the same relation twice
        has no further effect.
        """
        if node == depends_on:
            raise ValueError(f"{node!r} cannot depend on itself")
        self.add_node(node)
        self.add_node(depends_on)
        self._deps[node].add(depends_on)
        self._dependents[depends_on].add(node)

    def remove_node(self, name: str) -> None:
        """Remove ``name`` and every relation that mentions it."""
        if name not in self._deps:
            raise KeyError(name)
        for dep in self._deps.pop(name):
            self._dependents[dep].discard(name)
        for child in self._dependents.pop(name):
            self._deps[child].discard(name)

    def __contains__(self, name: object) -> bool:
        return name in self._deps

    def __len__(self) -> int:
        return len(self._deps)

    def __iter__(self) -> Iterator[str]:
        return iter(self.nodes())

    def nodes(self) -> List[str]:
        """Return all node names in ascending order."""
        return sorted(self._deps)

    def dependencies_of(self, name: str) -> List[str]:
        """Return the direct dependencies of ``name`` in ascending order."""
        return sorted(self._deps[name])

    def dependents_of(self, name: str) -> List[str]:
        """Return the nodes that directly depend on ``name``."""
        return sorted(self._dependents[name])

    def topological_order(self) -> List[str]:
        """Return the nodes so that each one follows all its dependencies.

        Among the nodes that are ready at the same time the smallest name
        comes first.  Raises :class:`CycleError` when no order exists.
        """
        waiting = {name: len(deps) for name, deps in self._deps.items()}
        ready = [name for name, count in waiting.items() if count == 0]
        heapq.heapify(ready)
        order: List[str] = []
        while ready:
            node = heapq.heappop(ready)
            order.append(node)
            for child in self._dependents[node]:
                waiting[child] -= 1
                if waiting[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(self._deps):
            cycle = self.find_cycle()
            raise CycleError(cycle if cycle is not None else [])
        return order

    def find_cycle(self) -> Optional[List[str]]:
        """Return one dependency cycle, or ``None`` when the graph is acyclic.

        The result starts and ends with the same node, for example
        ``["a", "b", "c", "a"]`` when a depends on b, b on c and c on a.
        """
        unvisited, active, done = 0, 1, 2
        state = {name: unvisited for name in self._deps}
        for root in sorted(self._deps):
            if state[root] != unvisited:
                continue
            state[root] = active
            path = [root]
            pending = [iter(sorted(self._deps[root]))]
            while path:
                descended = False
                for dep in pending[-1]:
                    if state[dep] == active:
                        return path[path.index(dep):] + [dep]
                    if state[dep] == unvisited:
                        state[dep] = active
                        path.append(dep)
                        pending.append(iter(sorted(self._deps[dep])))
                        descended = True
                        break
                if not descended:
                    state[path.pop()] = done
                    pending.pop()
        return None

    def has_cycle(self) -> bool:
        """Return ``True`` when the graph contains a dependency cycle."""
        return self.find_cycle() is not None

    def levels(self) -> List[List[str]]:
        """Group the nodes into layers that can be processed in parallel.

        Layer 0 holds the nodes without dependencies; every other node sits
        one layer above its deepest dependency.  Each layer is sorted by
        name.  Raises :class:`CycleError` for a cyclic graph.
        """
        order = self.topological_order()
        depth = {name: 0 for name in order}
        for node in order:
            for child in self._dependents[node]:
                depth[child] = depth[node] + 1
        layers: List[List[str]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
        for name in order:
            layers[depth[name]].append(name)
        return [sorted(layer) for layer in layers]

    def transitive_dependencies(self, name: str) -> Set[str]:
        """Return everything ``name`` needs, directly or indirectly.

        The node itself is not part of the result (unless it sits on a cycle
        through itself).
        """
        if name not in self._deps:
            raise KeyError(name)
        seen: Set[str] = set()
        stack = list(self._deps[name])
        while stack:
            dep = stack.pop()
            if dep in seen:
                continue
            seen.add(dep)
            stack.extend(self._deps[dep] - seen)
        return seen

    def build_order_for(self, targets: Iterable[str]) -> List[str]:
        """Return the order in which to build ``targets`` and their needs.

        Nodes that no target requires are left out.
        """
        needed: Set[str] = set()
        for target in targets:
            needed.add(target)
            needed |= self.transitive_dependencies(target)
        return [name for name in self.topological_order() if name in needed]

    def roots(self) -> List[str]:
        """Return the nodes without dependencies."""
        return sorted(name for name, deps in self._deps.items() if not deps)

    def leaves(self) -> List[str]:
        """Return the nodes nothing depends on."""
        return sorted(name for name, kids in self._dependents.items() if not kids)
