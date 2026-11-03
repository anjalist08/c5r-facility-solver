"""Solve and score floor plans with the kit's scorer; keep the best valid layouts."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import KIT, search
from facility import score  # noqa: E402  (KIT is on sys.path)
from facility.__main__ import load, load_floor, load_runs, runs_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
BEST = ROOT / "submissions"
CATALOG = KIT / "catalog.json"


def inputs(plan, runs=None):
    plan = Path(plan)
    floor = load_floor(plan)
    return floor, load(CATALOG), load_runs(runs or runs_path(plan, floor))


def solve(plan, runs=None, seconds=120, seed=0, quiet=False):
    floor, catalog, runs = inputs(plan, runs)
    log = (lambda *a: None) if quiet else (lambda *a: print(f"[{floor['id']}]", *a, file=sys.stderr, flush=True))
    layout, minutes = search.solve(floor, catalog, runs, seconds=seconds, seed=seed, log=log)
    return floor["id"], layout, minutes


def bench_one(plan, seconds, seed):
    start = time.time()
    name, layout, _ = solve(plan, seconds=seconds, seed=seed)
    floor, catalog, runs = inputs(plan)
    report = score(floor, catalog, runs, layout) if layout else {"valid": False, "violations": ["no layout"]}
    OUT.mkdir(exist_ok=True)
    if layout:
        (OUT / f"{name}.json").write_text(json.dumps(layout, indent=1) + "\n")
    (OUT / f"{name}.report.json").write_text(json.dumps(report))
    return name, plan, report, time.time() - start


def bench(args):
    plans = args.floorplans or sorted(str(p) for p in (KIT / "floorplans").glob("*.json"))
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(bench_one, plans, [args.seconds] * len(plans), [args.seed] * len(plans)))
    for name, plan, report, took in results:
        if report["valid"]:
            new = keep(plan, json.loads((OUT / f"{name}.json").read_text()))
            mark = f"  NEW BEST -> submissions/{name}.json" if new else ""
            print(f"{name:6} {report['minutes']:9.3f} min  {report['walk_m']:7.1f} m walked  ({took:.0f} s){mark}")
        else:
            print(f"{name:6}  INVALID  " + "\n    ".join(report["violations"][:8]))
    return 0 if all(report["valid"] for *_, report, _ in results) else 1


def keep(plan, layout):
    """Save `layout` as submissions/<id>.json if the kit's scorer finds it valid and
    better than submissions/scores.json. Returns the new score, or None."""
    floor, catalog, runs = inputs(plan)
    report = score(floor, catalog, runs, layout)
    if not report["valid"]:
        return None
    BEST.mkdir(exist_ok=True)
    scores_path = BEST / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    name = floor["id"]
    if report["minutes"] >= scores.get(name, float("inf")) - 1e-9:
        return None
    (BEST / f"{name}.json").write_text(json.dumps(layout, indent=1) + "\n")
    scores[name] = report["minutes"]
    scores_path.write_text(json.dumps(dict(sorted(scores.items())), indent=2) + "\n")
    return report["minutes"]
