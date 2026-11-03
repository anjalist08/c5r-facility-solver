"""Solver for the C5R x Stanford Facility Planning problem.

The kit's `facility` package (geometry, safety rules, simulator, baseline) is
imported from the kit folder: `kit/facility` in this repo, or the path in
$C5R_KIT.
"""

import os
import sys
from pathlib import Path

KIT = Path(os.environ.get("C5R_KIT", Path(__file__).resolve().parent.parent / "kit" / "facility"))

if not (KIT / "facility").is_dir():
    sys.exit(f"kit not found at {KIT}: copy the C5R kit's facility folder to kit/facility or set C5R_KIT to it")
sys.path.insert(0, str(KIT))
