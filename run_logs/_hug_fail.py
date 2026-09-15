"""Where in the box-hug clip do the incomplete runs stop, and in what attitude?

The clip's phase clock (deploy_x2_box_hug.py):
STAND 0.00  LOWER 1.02  GRASP 2.52  LIFT 3.62  HOLD 5.62  LOWER_BOX 6.62
SET_DOWN 8.62  RELEASE 9.02  DONE 9.72 ... clip ends 12.22 s, then --hold-end-seconds
(2.0 s default) of DONE, so a full run is 711 ticks at 50 Hz.

Two thresholds can stop a run early, and they are easy to confuse -- read both
before calling a run a fall:
  --roll-abort 0.6   |pelvis_roll| > 0.6 rad
  --tilt-abort -0.5  pelvis projected_gravity z RISES above -0.5 (~60 deg lean;
                     -1.0 is upright, so less negative means more tilted)
A run that ends near upright on both, in DONE, after the 9.72 s sequence is
already over, was almost certainly stopped by hand rather than aborted.
"""
import csv
import glob
import json
import os
import sys

PHASES = [
    (0.00, "STAND"), (1.02, "LOWER"), (2.52, "GRASP"), (3.62, "LIFT"),
    (5.62, "HOLD"), (6.62, "LOWER_BOX"), (8.62, "SET_DOWN"),
    (9.02, "RELEASE"), (9.72, "DONE"),
]
COMPLETE = 711
DONE_TICK = 486
ROLL_ABORT = 0.6
TILT_ABORT = -0.5


def phase_at(t):
    name = "?"
    for start, n in PHASES:
        if t >= start:
            name = n
    return name


def verdict(ticks, roll, gz):
    if abs(roll) >= ROLL_ABORT - 0.07:
        return "ROLL abort"
    if gz == gz and gz > TILT_ABORT - 0.02:
        return "TILT abort (pitched over)"
    if ticks >= DONE_TICK:
        return "stopped by hand in DONE (sequence finished)"
    return "stopped early, attitude looks fine"


pattern = sys.argv[1] if len(sys.argv) > 1 else "2026*"
for f in sorted(glob.glob("%s_box_hug_*.meta.json" % pattern)):
    m = json.load(open(f))
    path = m["csv"]
    if not os.path.exists(path):
        continue
    with open(path) as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        idx = {k: (header.index(k) if k in header else None)
               for k in ("phase", "pelvis_roll", "proj_g_z")}
        last, n = None, 0
        for r in rdr:
            i = idx["phase"]
            if i is not None and len(r) > i and r[i] == "policy":
                n += 1
                last = r
    if n >= COMPLETE or last is None:
        continue

    def get(key):
        i = idx[key]
        if i is None or len(last) <= i:
            return float("nan")
        try:
            return float(last[i])
        except ValueError:
            return float("nan")

    roll, gz = get("pelvis_roll"), get("proj_g_z")
    t = n / 50.0
    print("%s gain=%-5s stopped %5.2fs in %-9s roll=%+.3f proj_g_z=%+.3f  %s"
          % (f[9:15], m.get("gain_scale"), t, phase_at(t), roll, gz,
             verdict(n, roll, gz)))
