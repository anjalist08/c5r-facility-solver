"""Layout model and greedy placement.

A layout is a list of standing objects (`Obj`): floor units and benches, each
bench carrying a list of bench-equipment types in order. Objects are placed
one at a time at the candidate spot that keeps walking short (weighted by how
often runs go from one type to another), checked against every safety rule.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from facility.geometry import CLEAR, ROTS, door_middle, footprint, front_zone, gap, inside, length_inside, mount
from facility.geometry import nearest_point, overlaps, point
from facility.safety import (
    NETWORK_REACH,
    OUTLET_REACH,
    PLUGS,
    SEPARATION,
    SERVICE_REACH,
    SERVICES,
    capacity,
    is_bench,
    keepouts,
    obstacles,
    on_bench,
    place,
    rules,
    swings,
    validate,
    walkway,
)
from facility.sim import needed

CELL = 10
WALL_STEP = 10
INNER_STEP = 20
WORK = 60  # the scorer's work position: 60 cm out from the middle of the front
EPS = 1e-6


@dataclass
class Obj:
    type: str
    x: float
    y: float
    rot: int
    outlet: str | None = None
    load: list = field(default_factory=list)

    def copy(self):
        return Obj(self.type, self.x, self.y, self.rot, self.outlet, list(self.load))


def work_point(rect, rot):
    x, y, w, d = rect
    return {
        0: (x + w / 2, y + d + WORK),
        90: (x - WORK, y + d / 2),
        180: (x + w / 2, y - WORK),
        270: (x + w + WORK, y + d / 2),
    }[rot]


class Occupancy:
    """Cells of 10 cm touched by rectangles, with prefix sums: a quick, sound
    test that a rectangle overlaps one of them (only cells wholly inside it count)."""

    def __init__(self, width, depth, rects):
        self.cols, self.rows = math.ceil(width / CELL), math.ceil(depth / CELL)
        marks = [[0] * self.cols for _ in range(self.rows)]
        for x, y, w, d in rects:
            if w <= 0 or d <= 0:
                continue
            c0, c1 = max(int(math.floor(x / CELL + EPS)), 0), min(int(math.ceil((x + w) / CELL - EPS)), self.cols)
            r0, r1 = max(int(math.floor(y / CELL + EPS)), 0), min(int(math.ceil((y + d) / CELL - EPS)), self.rows)
            for r in range(r0, r1):
                row = marks[r]
                for c in range(c0, c1):
                    row[c] = 1
        self.sums = [[0] * (self.cols + 1) for _ in range(self.rows + 1)]
        for r in range(self.rows):
            acc, above, here = 0, self.sums[r], self.sums[r + 1]
            for c in range(self.cols):
                acc += marks[r][c]
                here[c + 1] = above[c + 1] + acc

    def hits(self, rect):
        x, y, w, d = rect
        c0, c1 = max(int(math.ceil(x / CELL - EPS)), 0), min(int(math.floor((x + w) / CELL + EPS)), self.cols)
        r0, r1 = max(int(math.ceil(y / CELL - EPS)), 0), min(int(math.floor((y + d) / CELL + EPS)), self.rows)
        if c0 >= c1 or r0 >= r1:
            return False
        s = self.sums
        return s[r1][c1] - s[r0][c1] - s[r1][c0] + s[r0][c0] > 0


class Problem:
    def __init__(self, floor, catalog, runs):
        self.floor, self.catalog, self.runs = floor, catalog, runs
        self.width, self.depth = floor["width"], floor["depth"]
        self.fixed = [r for _, r in obstacles(floor)]
        self.swings = [r for _, r in swings(floor)]
        self.keepouts = [r for _, r in keepouts(floor)]
        self.circuits = {c["id"]: c for c in floor["circuits"]}
        self.outlets = {o["id"]: o for o in floor["outlets"]}
        self.inventory = floor["inventory"]
        self.types = needed(runs)
        self.pair = defaultdict(Counter)
        self.first = Counter()
        self.traffic = Counter()
        self.busy = Counter()
        handling = runs.get("handling_min", 0.5)
        for run in runs["runs"]:
            prev = None
            for s in run["steps"]:
                t = s["at"]
                self.traffic[t] += 1
                self.busy[t] += s["minutes"] + s.get("handling_min", handling)
                if prev is None:
                    self.first[t] += 1
                elif prev != t:
                    self.pair[prev][t] += 1
                    self.pair[t][prev] += 1
                prev = t
        door = floor["doors"][0]
        mx, my = door_middle(door, self.depth)
        if door["y"] == 0:
            self.start = (mx, WORK)
        elif door["y"] == self.depth:
            self.start = (mx, self.depth - WORK)
        elif door["x"] == 0:
            self.start = (WORK, my)
        else:
            self.start = (self.width - WORK, my)
        self.benches = sorted(
            (t for t in self.inventory if t in catalog and is_bench(catalog[t])),
            key=lambda t: catalog[t]["w"] * catalog[t]["d"],
        )

    # ---- geometry of a layout -------------------------------------------------

    def spec(self, t):
        return self.catalog[t]

    def rects(self, o):
        """[(type, rect)] for every unit on the object: the floor unit itself, or each bench item."""
        spec = self.catalog[o.type]
        rect = footprint(spec, o.x, o.y, o.rot)
        if not is_bench(spec):
            return [(o.type, rect)]
        out, used = [], 0.0
        for t in o.load:
            out.append((t, mount(rect, o.rot, used, self.catalog[t]["w"])))
            used += self.catalog[t]["w"]
        return out

    def layout(self, objs):
        items, seen = [], Counter()

        def name(t):
            seen[t] += 1
            return f"{t}-{seen[t]}"

        for o in objs:
            oid = name(o.type)
            item = {"id": oid, "type": o.type, "x": o.x, "y": o.y, "rot": o.rot}
            if o.outlet:
                item["outlet"] = o.outlet
            items.append(item)
            items += [{"id": name(t), "type": t, "bench": oid} for t in o.load]
        return {"items": items}

    def valid(self, objs):
        layout = self.layout(objs)
        if validate(self.floor, self.catalog, layout):
            return False
        placed = place(self.catalog, layout)
        return not rules(self.floor, placed) and not walkway(self.floor, placed).violations

    def counts(self, objs):
        n = Counter()
        for o in objs:
            n[o.type] += 1
            n.update(o.load)
        return n

    # ---- cost -------------------------------------------------------------------

    def unit_points(self, objs, skip=None):
        """type -> [work point] of every placed unit."""
        pts = defaultdict(list)
        for k, o in enumerate(objs):
            if k == skip:
                continue
            for t, r in self.rects(o):
                pts[t].append(work_point(r, o.rot))
        return pts

    def unit_cost(self, t, p, pts):
        cost = self.first[t] * math.dist(p, self.start)
        for q, w in self.pair[t].items():
            if pts.get(q):
                cost += w * min(math.dist(p, r) for r in pts[q])
        # a little pull to the door for everything, so the lab stays compact
        return cost + 0.05 * self.traffic[t] * math.dist(p, self.start)

    # ---- power ------------------------------------------------------------------

    def power_state(self, objs):
        load, plugs, dedicated, users = Counter(), Counter(), set(), Counter()
        for o in objs:
            for t, _ in self.rects(o):
                spec = self.catalog[t]
                if spec.get("watts") and o.outlet:
                    c = self.outlets[o.outlet]["circuit"]
                    load[c] += spec["watts"]
                    plugs[o.outlet] += 1
                    users[c] += 1
                    if "dedicated-circuit" in spec.get("needs", ()):
                        dedicated.add(c)
        return load, plugs, dedicated, users

    def cords(self, objs):
        out = []
        for o in objs:
            if not o.outlet:
                continue
            a = (self.outlets[o.outlet]["x"], self.outlets[o.outlet]["y"])
            for t, r in self.rects(o):
                if self.catalog[t].get("watts"):
                    out.append((a, nearest_point(a, r)))
        return out

    def outlet_choices(self, units, power, zones, keep=None):
        """Outlets that can feed `units` [(type, rect)], best fit first."""
        powered = [(t, r) for t, r in units if self.catalog[t].get("watts")]
        if not powered:
            return [None]
        load, plugs, dedicated, users = power
        kinds = {(self.catalog[t]["volts"], self.catalog[t]["phases"]) for t, _ in powered}
        if len(kinds) > 1:
            return []
        volts, phases = kinds.pop()
        watts = sum(self.catalog[t]["watts"] for t, _ in powered)
        alone = any("dedicated-circuit" in self.catalog[t].get("needs", ()) for t, _ in powered)
        out = []
        for oid, o in self.outlets.items():
            if keep is not None and oid != keep:
                continue
            c = self.circuits[o["circuit"]]
            if (c["volts"], c["phases"]) != (volts, phases):
                continue
            if plugs[oid] + len(powered) > PLUGS:
                continue
            spare = capacity(c) - load[c["id"]] - watts
            if spare < -EPS:
                continue
            if c["id"] in dedicated or (alone and (users[c["id"]] or len(powered) > 1)):
                continue
            a = (o["x"], o["y"])
            if any(gap(point(o), r) > OUTLET_REACH for _, r in powered):
                continue
            if not all(self.cord_ok(a, nearest_point(a, r), zones) for _, r in powered):
                continue
            out.append((spare, oid))
        return [oid for _, oid in sorted(out)]

    def cord_ok(self, a, b, zones):
        if any(length_inside(a, b, z, True) > EPS for z in self.swings):
            return False
        return not any(length_inside(a, b, z, False) > EPS for z in zones)

    # ---- other rules on one object ----------------------------------------------

    def services_ok(self, t, rect):
        for need in self.catalog[t].get("needs", ()):
            if need == "network":
                pts, reach = self.floor["network"], NETWORK_REACH
            elif need in SERVICES:
                pts, reach = [s for s in self.floor["services"] if s["kind"] == need], SERVICE_REACH
            else:
                continue
            if min((gap(point(p), rect) for p in pts), default=math.inf) > reach:
                return False
        return True

    def room_of(self, rect):
        x, y, w, d = rect
        for r in self.floor.get("rooms", ()):
            if inside((x - r["x"], y - r["y"], w, d), r["w"], r["d"]):
                return r["id"]
        return None

    def tags_ok(self, units, others, zone=None):
        """Fire (rule 10) and PCR (rule 11) for new units against the placed ones."""
        rooms = bool(self.floor.get("rooms"))
        for t, r in units:
            tags = self.catalog[t].get("tags", ())
            if "flammable" in tags and zone is not None and any(overlaps(zone, s) for s in self.swings):
                return False
            if rooms and ("pre-pcr" in tags or "post-pcr" in tags):
                room = self.room_of(r)
                if room is None:
                    return False
            for q, s in others + [u for u in units if u[1] is not r]:
                qt = self.catalog[q].get("tags", ())
                for a, b in (("flammable", "heat"), ("heat", "flammable")):
                    if a in tags and b in qt and gap(r, s) < SEPARATION - EPS:
                        return False
                for a, b in (("pre-pcr", "post-pcr"), ("post-pcr", "pre-pcr")):
                    if a in tags and b in qt:
                        if not rooms and gap(r, s) < SEPARATION - EPS:
                            return False
                        if rooms and self.room_of(s) == self.room_of(r):
                            return False
        return True

    # ---- placing ----------------------------------------------------------------

    def candidates(self, spec, objs):
        """Spots (x, y, rot) for a standing object that clear items, zones, obstacles, doors and keep-outs."""
        standing = [(footprint(self.catalog[o.type], o.x, o.y, o.rot), o.rot) for o in objs]
        rects = [r for r, _ in standing]
        zones = [front_zone(r, rot) for r, rot in standing]
        hard = Occupancy(self.width, self.depth, rects + self.fixed)
        soft = Occupancy(self.width, self.depth, zones + self.swings + self.keepouts)
        width, depth = self.width, self.depth
        seen = set()
        for rot in ROTS:
            w, d = (spec["w"], spec["d"]) if rot in (0, 180) else (spec["d"], spec["w"])
            xs = set(_steps(0, width - w, INNER_STEP))
            ys = set(_steps(0, depth - d, INNER_STEP))
            # flush against walls and against other things (packing)
            for x, y, rw, rd in rects + self.fixed + zones + self.swings + self.keepouts:
                xs.update((x + rw, x - w))
                ys.update((y + rd, y - d))
            xs.update((0, width - w))
            ys.update((0, depth - d))
            xs = sorted(x for x in xs if -EPS <= x <= width - w + EPS)
            ys = sorted(y for y in ys if -EPS <= y <= depth - d + EPS)
            for x in xs:
                for y in ys:
                    rect = (x, y, w, d)
                    zone = front_zone(rect, rot)
                    if not inside(zone, width, depth):
                        continue
                    if hard.hits(rect) or soft.hits(rect) or hard.hits(zone):
                        continue
                    key = (round(x, 3), round(y, 3), rot)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield x, y, rot, rect, zone

    def workable(self, rect, zone, rot, objs):
        """Some spot 45-120 cm out in the front zone is clear of walls, things and door swings
        (the scorer's work position needs a walkable cell there, 45 cm from door swings)."""
        x, y, w, d = zone
        blocks = [footprint(self.catalog[o.type], o.x, o.y, o.rot) for o in objs] + self.fixed + [rect]
        if rot in (0, 180):
            along = [x + w * f for f in (0.5, 0.25, 0.75, 0.0, 1.0)]
            out = [y + d - v for v in (60, 75, 90)] if rot == 180 else [y + v for v in (60, 75, 90)]
            pts = [(a, b) for b in out for a in along]
        else:
            along = [y + d * f for f in (0.5, 0.25, 0.75, 0.0, 1.0)]
            out = [x + w - v for v in (60, 75, 90)] if rot == 90 else [x + v for v in (60, 75, 90)]
            pts = [(a, b) for a in out for b in along]
        for px, py in pts:
            if not (45 <= px <= self.width - 45 and 45 <= py <= self.depth - 45):
                continue
            p = (px, py, 0, 0)
            if all(gap(p, s) >= 45 for s in self.swings) and all(gap(p, b) >= 45 - EPS for b in blocks):
                return True
        return False

    def exact_clear(self, rect, zone, objs):
        for o in objs:
            r = footprint(self.catalog[o.type], o.x, o.y, o.rot)
            z = front_zone(r, o.rot)
            if overlaps(rect, r) or overlaps(zone, r) or overlaps(rect, z):
                return False
        for r in self.fixed:
            if overlaps(rect, r) or overlaps(zone, r):
                return False
        return not any(overlaps(rect, r) for r in self.swings + self.keepouts)

    def place_standing(self, objs, template, rng=None, jitter=0.0, tries=60):
        """Put `template` (a floor unit, or a bench with its load) at the best spot; new list or None."""
        spec = self.catalog[template.type]
        pts = self.unit_points(objs)
        scored = []
        for x, y, rot, rect, zone in self.candidates(spec, objs):
            probe = Obj(template.type, x, y, rot, None, template.load)
            units = self.rects(probe)
            cost = sum(self.unit_cost(t, work_point(r, rot), pts) for t, r in units)
            if jitter and rng:
                cost *= 1 + jitter * rng.random()
            scored.append((cost, x, y, rot, rect, zone))
        scored.sort(key=lambda s: s[0])
        power = self.power_state(objs)
        others = [u for o in objs for u in self.rects(o)]
        all_zones = [front_zone(footprint(self.catalog[o.type], o.x, o.y, o.rot), o.rot) for o in objs]
        cords = self.cords(objs)
        attempts, failed = 0, []
        for cost, x, y, rot, rect, zone in scored:
            if any(r == rot and abs(fx - x) < 25 and abs(fy - y) < 25 for fx, fy, r in failed):
                continue
            if not self.exact_clear(rect, zone, objs):
                continue
            if not self.workable(rect, zone, rot, objs):
                continue
            if any(length_inside(a, b, zone, False) > EPS for a, b in cords):
                continue
            probe = Obj(template.type, x, y, rot, None, list(template.load))
            units = self.rects(probe)
            if not all(self.services_ok(t, r) for t, r in units):
                continue
            if not self.tags_ok(units, others, zone):
                continue
            for outlet in self.outlet_choices(units, power, all_zones + [zone])[:2]:
                probe.outlet = outlet
                trial = objs + [probe]
                attempts += 1
                if self.valid(trial):
                    return trial
                failed.append((x, y, rot))
                break
            if attempts >= tries:
                return None
        return None

    def bench_fits(self, bench, t):
        b, s = self.catalog[bench.type], self.catalog[t]
        width = sum(self.catalog[q]["w"] for q in bench.load) + s["w"]
        kg = sum(self.catalog[q].get("kg", 0) for q in bench.load) + s.get("kg", 0)
        return width <= b["w"] + EPS and s["d"] <= b["d"] + EPS and kg <= b["max_kg"] + EPS

    def add_to_bench(self, objs, k, t):
        """Append `t` to bench objs[k]; new list or None."""
        bench = objs[k]
        if not self.bench_fits(bench, t):
            return None
        new = bench.copy()
        new.load.append(t)
        rest = objs[:k] + objs[k + 1 :]
        units = self.rects(new)
        mine = units[-1]
        if not self.services_ok(*mine):
            return None
        if not self.tags_ok([mine], [u for o in rest for u in self.rects(o)] + units[:-1]):
            return None
        powered_before = [u for u in units[:-1] if self.catalog[u[0]].get("watts")]
        if self.catalog[t].get("watts"):
            power = self.power_state(rest)
            zones = [front_zone(footprint(self.catalog[o.type], o.x, o.y, o.rot), o.rot) for o in objs]
            keep = bench.outlet if powered_before else None
            choices = self.outlet_choices(units, power, zones, keep)
            if not choices:
                return None
            new.outlet = choices[0]
        trial = rest[:k] + [new] + rest[k:]
        return trial if self.valid(trial) else None

    def add_unit(self, objs, t, rng=None, jitter=0.0, new_bench=True):
        """Install one more unit of type t wherever it costs least; new list or None."""
        spec = self.catalog[t]
        if not on_bench(spec):
            return self.place_standing(objs, Obj(t, 0, 0, 0), rng, jitter)
        pts = self.unit_points(objs)
        options = []
        for k, o in enumerate(objs):
            if is_bench(self.catalog[o.type]) and self.bench_fits(o, t):
                probe = o.copy()
                probe.load.append(t)
                t2, r = self.rects(probe)[-1]
                cost = self.unit_cost(t, work_point(r, o.rot), pts)
                if jitter and rng:
                    cost *= 1 + jitter * rng.random()
                options.append((cost, k))
        options.sort()
        for cost, k in options[:6]:
            trial = self.add_to_bench(objs, k, t)
            if trial:
                if not new_bench:
                    return trial
                best = (cost, trial)
                break
        else:
            best = None
        if new_bench:
            used = self.counts(objs)
            for b in self.benches:
                if used[b] >= self.inventory[b]:
                    continue
                if spec["d"] > self.catalog[b]["d"] + EPS or spec.get("kg", 0) > self.catalog[b]["max_kg"]:
                    continue
                trial = self.place_standing(objs, Obj(b, 0, 0, 0, None, [t]), rng, jitter)
                if trial:
                    p = work_point(self.rects(trial[-1])[0][1], trial[-1].rot)
                    cost = self.unit_cost(t, p, pts)
                    # sharing a bench saves floor space: take the shared spot unless a new bench is much better
                    if best is None or cost < 0.7 * best[0] - 50:
                        return trial
                    break
        return best[1] if best else None


def _steps(lo, hi, step):
    v = lo
    while v <= hi + EPS:
        yield v
        v += step
