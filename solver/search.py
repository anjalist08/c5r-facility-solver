"""Build a first layout greedily, then improve it against the kit's own simulator."""

from __future__ import annotations

import math
import random
import time

from facility import score as kit_score
from facility import solve as kit_baseline
from facility.safety import SERVICES, is_bench, on_bench

from .build import Problem


def constrained(spec):
    needs = spec.get("needs", ())
    return any(n in SERVICES for n in needs) or "dedicated-circuit" in needs or spec.get("volts") != 120


def order(P, rng=None):
    """Required types: hard-to-place first, then by traffic."""

    def key(t):
        spec = P.catalog[t]
        noise = rng.random() * 30 if rng else 0
        return (not constrained(spec), -(spec.get("watts", 0) >= 1000), -P.traffic[t] - P.busy[t] / 10 + noise)

    return sorted(P.types, key=key)


def extras(P):
    """Extra units worth trying, most useful first: busy machines, then busy stops."""
    out = []
    for t in P.types:
        for n in range(2, P.inventory.get(t, 0) + 1):
            out.append((P.busy[t] / n + P.traffic[t] * 0.3 / n, t))
    return [t for _, t in sorted(out, reverse=True)]


def construct(P, first=(), rng=None, jitter=0.0, log=None, rounds=8):
    """Place one unit of every needed type. A type that cannot be placed moves
    to the front of the order and the build starts over (squeaky wheel)."""
    front = list(first)
    for _ in range(rounds):
        objs, failed = [], None
        for t in front + [t for t in order(P, rng) if t not in front]:
            new = P.add_unit(objs, t, rng, jitter)
            if new is None:
                failed = t
                break
            objs = new
        if failed is None:
            return objs, front
        if log:
            log(f"  could not place {failed}; placing it earlier")
        if failed in front:
            front.remove(failed)
        front.insert(0, failed)
    return None, front


def evaluate(P, objs):
    report = kit_score(P.floor, P.catalog, P.runs, P.layout(objs))
    return report["minutes"] if report["valid"] else None


def add_extras(P, objs, m, log, deadline):
    """Add extra units greedily while each one helps."""
    tried = set()
    for t in extras(P):
        if time.time() > deadline:
            break
        if t in tried or P.counts(objs)[t] >= P.inventory.get(t, 0):
            continue
        new = P.add_unit(objs, t)
        if new is None:
            tried.add(t)
            continue
        m2 = evaluate(P, new)
        if m2 is not None and m2 < m - 1e-9:
            log(f"  + {t}: {m2}")
            objs, m = new, m2
        else:
            tried.add(t)
    return objs, m


def mutate(P, objs, rng):
    """A random neighbor of the layout (may be None or invalid)."""
    move = rng.random()
    n = len(objs)
    if move < 0.35:
        # move one standing object (a floor unit or a whole bench) elsewhere
        k = rng.randrange(n)
        o = objs[k]
        rest = objs[:k] + objs[k + 1 :]
        template = o.copy()
        return P.place_standing(rest, template, rng, jitter=1.5, tries=20)
    if move < 0.6:
        # move one bench item to another bench or a new one
        benches = [k for k, o in enumerate(objs) if o.load]
        if not benches:
            return None
        k = rng.choice(benches)
        i = rng.randrange(len(objs[k].load))
        new = [o.copy() for o in objs]
        t = new[k].load.pop(i)
        if not new[k].load:
            new.pop(k)
        elif not any(P.catalog[q].get("watts") for q in new[k].load):
            new[k].outlet = None
        return P.add_unit(new, t, rng, jitter=1.5)
    if move < 0.7:
        # reorder items on a bench
        benches = [k for k, o in enumerate(objs) if len(o.load) > 1]
        if not benches:
            return None
        k = rng.choice(benches)
        new = [o.copy() for o in objs]
        rng.shuffle(new[k].load)
        return new
    if move < 0.85:
        # one more unit of a type with stock left
        counts = P.counts(objs)
        spare = [t for t in P.types if counts[t] < P.inventory.get(t, 0)]
        if not spare:
            return None
        return P.add_unit(objs, rng.choice(spare), rng, jitter=1.5)
    # one unit fewer of a type that has several
    counts = P.counts(objs)
    many = [t for t in P.types if counts[t] > 1]
    if not many:
        return None
    t = rng.choice(many)
    spots = [(k, -1) for k, o in enumerate(objs) if o.type == t]
    spots += [(k, i) for k, o in enumerate(objs) for i, q in enumerate(o.load) if q == t]
    k, i = rng.choice(spots)
    new = [o.copy() for o in objs]
    if i < 0:
        new.pop(k)
    else:
        new[k].load.pop(i)
        if not new[k].load:
            new.pop(k)
        elif not any(P.catalog[q].get("watts") for q in new[k].load):
            new[k].outlet = None
    return new


def improve(P, objs, m, deadline, rng, log, temp=1.0):
    """Simulated annealing on the simulated makespan; returns the best layout seen."""
    best, best_m = objs, m
    cur, cur_m = objs, m
    start = time.time()
    span = max(deadline - start, 1e-9)
    steps = 0
    while time.time() < deadline:
        steps += 1
        new = mutate(P, cur, rng)
        if new is None:
            continue
        m2 = evaluate(P, new)
        if m2 is None:
            continue
        t = temp * max(1 - (time.time() - start) / span, 0.01)
        if m2 <= cur_m or rng.random() < math.exp((cur_m - m2) / t):
            cur, cur_m = new, m2
            if m2 < best_m - 1e-9:
                best, best_m = new, m2
                log(f"  {best_m}  ({steps} steps)")
    return best, best_m


def solve(floor, catalog, runs, seconds=120, seed=0, log=None):
    """Return (layout, minutes) for the best valid layout found in about `seconds`."""
    log = log or (lambda *a: None)
    P = Problem(floor, catalog, runs)
    t0 = time.time()
    deadline = t0 + seconds
    best_layout, best = None, math.inf
    try:
        base = kit_baseline(floor, catalog, runs)
        report = kit_score(floor, catalog, runs, base)
        if report["valid"]:
            best_layout, best = base, report["minutes"]
            log(f"kit baseline {best}")
    except ValueError as error:
        log(f"kit baseline failed: {error}")
    rng = random.Random(seed)
    objs, front = construct(P, log=log)
    if objs is None:
        objs, _ = construct(P, front, rng, 1.0, log, rounds=20)
    if objs is None:
        log("greedy failed; keeping the kit baseline")
        return best_layout, best
    m = evaluate(P, objs)
    log(f"greedy {m}  ({time.time() - t0:.0f} s)")
    if m is None:
        return best_layout, best
    objs, m = add_extras(P, objs, m, log, t0 + seconds * 0.4)
    log(f"extras {m}  ({time.time() - t0:.0f} s)")
    objs, m = improve(P, objs, m, deadline, rng, log)
    log(f"search {m}  ({time.time() - t0:.0f} s)")
    if m < best:
        best_layout, best = P.layout(objs), m
    return best_layout, best
