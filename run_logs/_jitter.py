"""Leg-target jitter per run: mrad/step and how often the step reverses sign.

The v5 commit claims 17.2 mrad/step and a much lower reversal rate than the
iter-9000 policy it replaces (205 mrad/step, reversing on 67% of steps). This
measures the same two numbers on hardware, over the policy phase only.
"""
import csv
import glob
import json
import os

import numpy as np

LEG = ("hip", "knee", "ankle")


def leg_target_cols(meta):
    names = meta["joint_names"]
    per = meta["columns_per_joint"]
    tgt = per.index("tgt")
    cols = {}
    for i, n in enumerate(names):
        if any(k in n for k in LEG):
            cols[n] = 1 + i * len(per) + tgt
    return cols


for f in sorted(glob.glob("2026*_box_pickup_*.meta.json")):
    meta = json.load(open(f))
    csv_path = meta["csv"]
    if not os.path.exists(csv_path) or not meta.get("engage"):
        continue
    cols = leg_target_cols(meta)

    with open(csv_path) as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        try:
            ph = header.index("phase")
        except ValueError:
            ph = None
        # A Ctrl+C can leave the final row truncated mid-write, so require a
        # row to be long enough to index before trusting any field in it.
        width = max(cols.values()) + 1
        rows = [
            r
            for r in rdr
            if len(r) >= width and (ph is None or r[ph] == "policy")
        ]

    if len(rows) < 20:
        continue

    idx = sorted(cols.values())
    arr = np.array([[float(r[c]) for c in idx] for r in rows])
    d = np.diff(arr, axis=0)
    mrad = np.abs(d).mean() * 1e3
    both = d[:-1] * d[1:]
    reversal = (both < 0).mean() * 100.0

    pol = os.path.basename(str(meta.get("policy", "?")))
    pol = pol.replace("x2_box_policy_", "").replace(".npz", "")
    print(
        "%s  %-26s gain=%-4s ticks=%4d  |dtgt| = %6.1f mrad/step  "
        "reversals = %4.1f%%"
        % (f[9:15], pol, meta.get("gain_scale", "?"), len(rows), mrad, reversal)
    )
