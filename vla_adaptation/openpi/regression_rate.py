"""The regression rate, scored against a denominator that can actually regress.

The paper reports safety as broken/all-episodes -- "275 fixed, 7 broken, a 1.0% regression
rate". That denominator includes every episode the frozen policy had *already failed*, which
the correction cannot break. In the headline four-suite cell, 92 of 120 episodes (77%) are in
that category.

Scored against the episodes that were actually working:

    headline cell      1/120 = 0.8%   ->   1/28  = 3.6%
    all law files     34/1635 = 2.1%  ->  34/528 = 6.4%

and the three-backbone transfer table's "zero broken in 180 paired episodes" is really zero
broken in SIX opportunities, because seven of its nine cells have a frozen policy at exactly
0/20. Its one-sided 95% upper bound on the regression rate is 39%.

Neither denominator is wrong, but they answer different questions, and only the second answers
"if I switch this on, how often does it break something that was working". Report both.

Pure standard library. Runs with nothing installed.

    python3 openpi/regression_rate.py                 # headline cell + transfer table + sweep
    python3 openpi/regression_rate.py FILE [FILE ...] # any paired result files
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

HEADLINE = ["results/suites/libero_spatial_rotonly_paired.json",
            "results/suites/libero_goal_rotonly_n40.json",
            "results/suites/libero_object_rotonly_paired.json",
            "results/suites/libero_10_rotonly_n40.json"]

TRANSFER = [("pi0.5", "results/phase05/map_rot010.json", None),
            ("pi0.5", "results/phase05/tmag_015.json", None),
            ("pi0.5", "results/phase05/gsev_g020.json", "adaptive_gain"),
            ("OFT", "results/oft/oft_rot010.json", None),
            ("OFT", "results/oft/oft_tra015.json", None),
            ("OFT", "results/oft/oft_gain020.json", "adaptive_gain"),
            ("GR00T", "results/groot/groot_rot010.json", None),
            ("GR00T", "results/groot/groot_tra015.json", None),
            ("GR00T", "results/groot/groot_gain020.json", "adaptive_gain")]


def load(path, arm_b=None):
    arms = json.loads(pathlib.Path(path).read_text())["arms"]
    a = "frozen_faulted"
    b = arm_b or ("adaptive" if "adaptive" in arms else
                  next(k for k in arms if k != a))
    ka = {(e["task"], e["init"]): int(e["ok"]) for e in arms[a]["per_ep"]}
    kb = {(e["task"], e["init"]): int(e["ok"]) for e in arms[b]["per_ep"]}
    ks = sorted(set(ka) & set(kb))
    return (len(ks),
            sum(ka[k] for k in ks),                                  # opportunities
            sum(1 for k in ks if kb[k] and not ka[k]),               # fixed
            sum(1 for k in ks if ka[k] and not kb[k]))               # broken


def upper95(broken, opportunities):
    """One-sided 95% upper bound on a binomial rate, by bisection on the exact tail."""
    if opportunities == 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(math.comb(opportunities, i) * mid ** i * (1 - mid) ** (opportunities - i)
                   for i in range(broken + 1))
        lo, hi = (mid, hi) if tail > 0.05 else (lo, mid)
    return hi


def report(label, rows):
    n = sum(r[0] for r in rows); opp = sum(r[1] for r in rows)
    fx = sum(r[2] for r in rows); br = sum(r[3] for r in rows)
    print(f"\n{label}")
    print(f"  episodes {n}, fixed {fx}, broken {br}")
    print(f"  frozen policy had already failed {n - opp} of {n} "
          f"({100 * (n - opp) / max(n, 1):.0f}%) -- those cannot regress")
    print(f"  regression / all episodes    {br}/{n} = {100 * br / max(n, 1):.2f}%")
    print(f"  regression / opportunities   {br}/{opp} = "
          f"{100 * br / max(opp, 1):.2f}%   <- the number that answers the safety question")
    print(f"  one-sided 95% upper bound on the true rate: {100 * upper95(br, opp):.1f}%")


def main(argv):
    root = pathlib.Path(__file__).resolve().parent.parent
    if argv:
        report("supplied files", [load(p) for p in argv])
        return 0
    report("HEADLINE four-suite cell", [load(root / f) for f in HEADLINE])
    report("TRANSFER table (3 backbones x 3 fault families, one calibration)",
           [load(root / f, arm) for _, f, arm in TRANSFER])
    rows = []
    for p in sorted(root.glob("results/**/*.json")):
        if any(s in str(p) for s in ("/hardware/", "/ace", "/oracle/", "/gate04/", "norm_channels")):
            continue
        if p.name.startswith(("base_ki", "abl_")):
            continue                                   # baselines and ablations, not the law
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        a = d.get("arms")
        if not isinstance(a, dict) or "frozen_faulted" not in a:
            continue
        b = next((k for k in a if k != "frozen_faulted"), None)
        if not b or not all(isinstance(a[k], dict) and "per_ep" in a[k]
                            for k in ("frozen_faulted", b)):
            continue
        rows.append(load(p))
    report(f"ALL paired law files ({len(rows)} files, baselines and ablations excluded)", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
