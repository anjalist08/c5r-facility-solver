"""CLI.

    python3 -m solver solve FLOORPLAN [RUNS] > layout.json   # print our layout
    python3 -m solver bench [FLOORPLAN ...]                  # solve and score every floor plan
    python3 -m solver keep FLOORPLAN LAYOUT                  # save LAYOUT if it beats submissions/
"""

from __future__ import annotations

import argparse
import json
import sys

from .bench import bench, keep, solve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m solver")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("solve", help="print the layout for a floor plan")
    one.add_argument("floorplan")
    one.add_argument("runs", nargs="?")
    one.add_argument("--start", help="improve this layout instead of building one")
    many = commands.add_parser("bench", help="solve and score floor plans (default: all in the kit)")
    many.add_argument("floorplans", nargs="*")
    for command in (one, many):
        command.add_argument("--seconds", type=float, default=120, help="search time per floor plan")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--workers", type=int, default=2, help="parallel searches per floor plan")
    saving = commands.add_parser("keep", help="save a layout to submissions/ if it is valid and better")
    saving.add_argument("floorplan")
    saving.add_argument("layout")
    args = parser.parse_args(argv)
    if args.command == "bench":
        return bench(args)
    if args.command == "keep":
        with open(args.layout) as f:
            minutes = keep(args.floorplan, json.load(f))
        print(f"kept: {minutes} minutes" if minutes else "not kept (invalid or not better)")
        return 0
    name, layout, minutes = solve(args.floorplan, args.runs, args.seconds, args.seed, args.workers, args.start)
    if layout is None:
        print(f"error: no valid layout for {name}", file=sys.stderr)
        return 1
    print(f"[{name}] best {minutes} minutes", file=sys.stderr)
    print(json.dumps(layout, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
