"""The kit's simulator (facility.sim.simulate), rewritten for speed.

Same dispatch rules and the same distance lookups, so it returns the same
makespan; the search uses it as its objective and the kit's own scorer still
checks every layout we keep.
"""

from __future__ import annotations

import heapq
import math

from facility.safety import check
from facility.sim import CM_PER_MIN, HANDLING_MIN, needed


class Distances:
    """Shortest grid walks, one Dijkstra tree per source, looked up like the kit's."""

    def __init__(self, grid):
        self.adj, self.n, self.trees = grid.adj, len(grid.free), {}

    def tree(self, src):
        dist = [math.inf] * self.n
        dist[src] = 0.0
        heap = [(0.0, src)]
        adj, pop, push = self.adj, heapq.heappop, heapq.heappush
        while heap:
            d, i = pop(heap)
            if d > dist[i]:
                continue
            for j, w in adj[i]:
                nd = d + w
                if nd < dist[j]:
                    dist[j] = nd
                    push(heap, (nd, j))
        self.trees[src] = dist
        return dist

    def __call__(self, a, b):
        if a == b:
            return 0.0
        trees = self.trees
        if b in trees:
            a, b = b, a
        t = trees.get(a)
        if t is None:
            t = self.tree(a)
        return t[b]


def simulate(runs, units, start, dist, operators):
    """Makespan of the runs, as facility.sim.simulate computes it."""
    plan, handling = runs["runs"], runs.get("handling_min", HANDLING_MIN)
    free = {u: 0.0 for kind in units.values() for u, _ in kind}
    tmin = {t: 0.0 for t in units}
    crew_idle = [0.0] * operators
    crew_at = [start] * operators
    waiting = {n: (run["start_min"], 0, start) for n, run in enumerate(plan) if run["steps"]}
    order = {n: (run["start_min"], n) for n, run in enumerate(plan)}
    finished = [run["start_min"] for run in plan]
    kinds = [[s["at"] for s in run["steps"]] for run in plan]
    while waiting:
        opens = {}
        for n, (ready, s, _) in waiting.items():
            m = tmin[kinds[n][s]]
            opens[n] = ready if ready > m else m
        now = max(min(opens.values()), min(crew_idle))
        n = min((r for r, o in opens.items() if o <= now), key=order.__getitem__)
        _, s, here = waiting.pop(n)
        step = plan[n]["steps"][s]
        o = min((dist(crew_at[k], here), k) for k in range(operators) if crew_idle[k] <= now)[1]
        at = crew_at[o]
        kind = step["at"]
        unit, there = min((u for u in units[kind] if free[u[0]] <= now), key=lambda u: dist(here, u[1]))
        end = now + (dist(at, here) + dist(here, there)) / CM_PER_MIN + step.get("handling_min", handling)
        done = end + step["minutes"]
        crew_idle[o], crew_at[o] = (done if step.get("attended") else end), there
        free[unit] = done
        tmin[kind] = min(free[u] for u, _ in units[kind])
        if s + 1 < len(plan[n]["steps"]):
            waiting[n] = (done, s + 1, there)
        else:
            finished[n] = done
    return round(max(finished, default=0.0), 3)


def score(floor, catalog, runs, layout):
    """Makespan in minutes, or None if the layout breaks a rule or misses a type."""
    violations, items, way = check(floor, catalog, layout)
    if violations:
        return None
    installed = {i.type for i in items}
    if any(t not in installed for t in needed(runs)):
        return None
    units = {}
    for i in items:
        units.setdefault(i.type, []).append((i.id, way.access[i.id]))
    return simulate(runs, units, way.start, Distances(way.grid), floor["operators"])
