"""Build a first layout greedily, then improve it against the kit's own simulator."""

from __future__ import annotations

import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from facility import score as kit_score
from facility import solve as kit_baseline
from facility.geometry import footprint, front_zone
from facility.safety import SERVICES, is_bench

from .build import Obj, Problem


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


def targets(P):
    """How many units of each type to start with: enough that no machine is
    much busier than the operators are."""
    handling = P.runs.get("handling_min", 0.5)
    work = sum(
        s.get("handling_min", handling) + (s["minutes"] if s.get("attended") else 0)
        for run in P.runs["runs"]
        for s in run["steps"]
    )
    span = max(1.15 * work / P.floor["operators"], max((r["start_min"] for r in P.runs["runs"]), default=0))
    return {t: max(1, min(P.inventory.get(t, 1), math.ceil(P.busy[t] / (0.7 * span)))) for t in P.types}


def construct(P, counts, first=(), rng=None, jitter=0.0, log=None, rounds=8):
    """Place counts[t] units of every needed type (the first one of each must
    fit). A type that cannot be placed moves to the front of the order and the
    build starts over (squeaky wheel)."""
    front = list(first)
    for _ in range(rounds):
        objs, failed = [], None
        for t in front + [t for t in order(P, rng) if t not in front]:
            for n in range(counts.get(t, 1)):
                new = P.add_unit(objs, t, rng, jitter)
                if new is None:
                    if n == 0:
                        failed = t
                    break
                objs = new
            if failed:
                break
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
    if move < 0.65:
        # swap the loads of two benches of the same type
        benches = [k for k, o in enumerate(objs) if is_bench(P.catalog[o.type])]
        if len(benches) < 2:
            return None
        a, b = rng.sample(benches, 2)
        if objs[a].type != objs[b].type:
            return None
        new = [o.copy() for o in objs]
        new[a].load, new[b].load = new[b].load, new[a].load
        new[a].outlet, new[b].outlet = None, None
        for k in (a, b):
            if any(P.catalog[q].get("watts") for q in new[k].load):
                rest = new[:k] + new[k + 1 :]
                zones = [front_zone(footprint(P.catalog[o.type], o.x, o.y, o.rot), o.rot) for o in new]
                choices = P.outlet_choices(P.rects(new[k]), P.power_state(rest), zones)
                if not choices:
                    return None
                new[k].outlet = choices[0]
        return new
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
        weights = [P.busy[t] / counts[t] + 1 for t in spare]
        return P.add_unit(objs, rng.choices(spare, weights)[0], rng, jitter=1.5)
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


def from_layout(P, layout):
    """Objects of a layout file (bench items stay on their bench, in order)."""
    objs, benches = [], {}
    for it in layout["items"]:
        if "bench" in it:
            continue
        o = Obj(it["type"], it["x"], it["y"], it["rot"], it.get("outlet"))
        objs.append(o)
        benches[it["id"]] = o
    for it in layout["items"]:
        if "bench" in it:
            benches[it["bench"]].load.append(it["type"])
    return objs


def worker(floor, catalog, runs, seconds, seed, start=None, verbose=True):
    """One search: greedy build (seed 0 plain, others jittered) or `start`, then annealing."""
    t0 = time.time()
    tag = f"[{floor['id']}#{seed}]"
    log = (lambda *a: print(tag, *a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    P = Problem(floor, catalog, runs)
    rng = random.Random(seed)
    if start is not None:
        objs = from_layout(P, start)
        m = evaluate(P, objs)
        log(f"start {m}")
    else:
        counts = targets(P)
        if seed:
            counts = {t: max(1, min(P.inventory.get(t, 1), n + rng.choice((-1, 0, 0, 1)))) for t, n in counts.items()}
        log("targets " + " ".join(f"{t}x{n}" for t, n in counts.items() if n > 1))
        jitter = 0.0 if seed == 0 else 0.5
        objs, front = construct(P, counts, rng=rng, jitter=jitter, log=log)
        if objs is None:
            objs, _ = construct(P, counts, front, rng, 1.0, log, rounds=20)
        if objs is None:
            log("greedy failed")
            return None, math.inf
        m = evaluate(P, objs)
        log(f"greedy {m}  ({time.time() - t0:.0f} s)")
    if m is None:
        return None, math.inf
    if start is None:
        objs, m = add_extras(P, objs, m, log, t0 + seconds * 0.3)
        log(f"extras {m}  ({time.time() - t0:.0f} s)")
    objs, m = improve(P, objs, m, t0 + seconds, rng, log, temp=0.5 if start is not None else 1.0)
    log(f"search {m}  ({time.time() - t0:.0f} s)")
    return P.layout(objs), m


def baseline(floor, catalog, runs):
    try:
        layout = kit_baseline(floor, catalog, runs)
    except ValueError:
        return None, math.inf
    report = kit_score(floor, catalog, runs, layout)
    return (layout, report["minutes"]) if report["valid"] else (None, math.inf)


def solve(floor, catalog, runs, seconds=120, seed=0, workers=1, start=None, log=None):
    """Return (layout, minutes) for the best valid layout found in about `seconds`:
    `workers` searches in parallel, with the kit baseline as the fallback."""
    log = log or (lambda *a: None)
    with ProcessPoolExecutor(workers + 1) as pool:
        base = pool.submit(baseline, floor, catalog, runs)
        jobs = [pool.submit(worker, floor, catalog, runs, seconds, seed + k, start) for k in range(workers)]
        results = [base.result()] + [job.result() for job in jobs]
    log(f"kit baseline {results[0][1]}")
    best_layout, best = None, math.inf
    for layout, m in results:
        if layout is not None and m < best:
            report = kit_score(floor, catalog, runs, layout)
            if report["valid"]:
                best_layout, best = layout, report["minutes"]
    return best_layout, best
