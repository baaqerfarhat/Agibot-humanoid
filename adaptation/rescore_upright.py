"""Re-score the adaptation experiments requiring the robot to still be standing.

`box_metrics` decides `placed` from the box alone -- it is low, and it is within
BOX_HELD_DIST of the root -- and `success = held[-1] or placed`. Nothing in that asks
whether the ROBOT is still up. A run that lifts the box, falls, and drops it beside
itself therefore scores as a completed placement, which is what the MuJoCo mirror made
visible: baseline_noisy seed 600 scores success=True with the robot face-down.

Early termination is unambiguous here -- `_rollout` breaks only on `root_z < FALL_HEIGHT`
and nothing else -- so `survival < steps` means the robot fell, full stop.

This does NOT change any metric. It reports both numbers side by side so the effect on
each comparison is visible:

    as-scored   success as recorded by box_metrics
    upright     success AND the episode ran its full length without falling

Usage:  python3 adaptation/rescore_upright.py [--runs adaptation/isaac_runs] [--all]
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(runs: str):
    out = []
    pattern = os.path.join(runs, "**", "adapt_experiments_summary.json")
    for f in sorted(glob.glob(pattern, recursive=True)):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        steps = d.get("steps")
        if not steps:
            continue
        name = os.path.basename(os.path.dirname(f)) or "(top level)"
        for mode, rows in (d.get("results") or {}).items():
            rec = [r for r in rows if isinstance(r, dict) and "success" in r]
            if rec:
                out.append((name, mode, steps, rec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "isaac_runs"))
    ap.add_argument("--all", action="store_true", help="include unaffected conditions")
    a = ap.parse_args()

    groups = load(a.runs)
    if not groups:
        raise SystemExit(f"no summaries under {a.runs}")

    n_tot = loose_tot = strict_tot = 0
    print(f"{'run dir':<26} {'mode':<20} {'n':>3} {'as-scored':>10} {'upright':>8}  delta")
    for name, mode, steps, rec in groups:
        n = len(rec)
        loose = sum(bool(r["success"]) for r in rec)
        strict = sum(bool(r["success"]) and r.get("survival", steps) >= steps for r in rec)
        n_tot += n
        loose_tot += loose
        strict_tot += strict
        if loose != strict or a.all:
            print(f"{name:<26} {mode:<20} {n:>3} {100*loose/n:>9.0f}% {100*strict/n:>7.0f}%"
                  f"  {100*(strict-loose)/n:>+5.0f} pts")

    print(f"\n{n_tot} episodes: {loose_tot} scored successful ({100*loose_tot/n_tot:.1f}%), "
          f"{strict_tot} of those finished upright ({100*strict_tot/n_tot:.1f}%).")
    print(f"{loose_tot - strict_tot} successes ({100*(loose_tot-strict_tot)/n_tot:.1f}% of all "
          f"episodes) are runs in which the robot fell.")


if __name__ == "__main__":
    main()
