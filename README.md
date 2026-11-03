# c5r-facility-solver

A solver for **Facility Planning** at the C5R × Stanford hackathon: it places lab
equipment in a floor plan so that every safety rule holds and the simulated day
of runs finishes as early as possible.

The kit itself (scorer, simulator, floor plans, catalog) belongs to the
organizers and is not in this repo. Copy its `facility` folder into `kit/`:

```sh
git clone https://github.com/anjalist08/c5r-facility-solver && cd c5r-facility-solver
mkdir kit && cp -R path/to/c5r-stanford-kit/facility kit/facility
```

Or point `C5R_KIT` at the kit's `facility` folder. Python 3.9+, standard library only.

## Use

```sh
python3 -m solver solve kit/facility/floorplans/h0.json > layout.json   # 120 s search
python3 -m solver solve FLOORPLAN RUNS --seconds 300 > layout.json
python3 -m solver bench                       # solve + score every floor plan in the kit
python3 -m solver keep FLOORPLAN layout.json  # save a layout if it beats submissions/
```

`bench` writes each layout and report to `out/` and copies every valid layout
that beats `submissions/scores.json` into `submissions/<id>.json`. Every layout
is checked with the kit's own scorer; the kit baseline is the fallback.

## Scores (local scorer, minutes, lower is better)

| Floor plan | Site name | This solver | Kit baseline |
| --- | --- | --- | --- |
| h0 | Main lab | 181.3 | 292.2 |
| b1 | Process lab | 190.0 | 298.7 |
| b2 | PCR and QC suite | 152.5 | 253.5 |

## How it works

- **Greedy build.** Units are placed one at a time, hard-to-place types first
  (services, 208 V, dedicated circuits), at the spot that minimizes walking:
  distance to the units that runs visit just before or after, weighted by how
  often, plus distance to the door for first steps. Bench equipment either
  joins an existing bench or gets a new one. Candidate spots come from a coarse
  grid plus positions flush against walls and other items; a prefix-sum
  occupancy map rejects overlaps fast. Each placement is checked with the kit's
  rules (power, cords, services, fire, PCR, walkways).
- **Squeaky wheel.** A type that cannot be placed moves to the front of the
  order and the build restarts.
- **Search.** Simulated annealing on the kit's simulated makespan: move a unit
  or bench, move an item between benches, reorder a bench, add or remove units.
