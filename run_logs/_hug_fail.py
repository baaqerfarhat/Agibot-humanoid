"""Where in the box-hug clip do the incomplete runs stop, and in what attitude?

Many Sep 14 runs end within a few ticks of each other, which points at one place
in the clip rather than at chance. The clip's phase clock (deploy_x2_box_hug.py):
STAND 0.00  LOWER 1.02  GRASP 2.52  LIFT 3.62  HOLD 5.62  LOWER_BOX 6.62
SET_DOWN 8.62  RELEASE 9.02  DONE 9.72 ... clip ends 12.22 s.
"""
import csv
import glob
import json
import os

PHASES = [
    (0.00, "STAND"), (1.02, "LOWER"), (2.52, "GRASP"), (3.62, "LIFT"),
    (5.62, "HOLD"), (6.62, "LOWER_BOX"), (8.62, "SET_DOWN"),
    (9.02, "RELEASE"), (9.72, "DONE"),
]
ROLL_ABORT = 0.6
TILT_ABORT = -0.5


def phase_at(t):
    name = "?"
    for start, n in PHASES:
        if t >= start:
            name = n
    return name


for f in sorted(glob.glob("20260914_*_box_hug_*.meta.json")):
    m = json.load(open(f))
    path = m["csv"]
    if not os.path.exists(path):
        continue
    with open(path) as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        ph = header.index("phase") if "phase" in header else None
        ri = header.index("pelvis_roll") if "pelvis_roll" in header else None
        last = None
        n = 0
        for r in rdr:
            if ph is not None and len(r) > ph and r[ph] == "policy":
                n += 1
                last = r
    if n >= 711 or last is None:
        continue
    t = n / 50.0
    roll = float(last[ri]) if ri is not None and len(last) > ri else float("nan")
    print(
        "%s gain=%-5s stopped at %5.2fs in %-9s  final pelvis_roll=%+.3f rad%s"
        % (
            f[9:15],
            m.get("gain_scale"),
            t,
            phase_at(t),
            roll,
            "  <-- past roll abort" if abs(roll) > ROLL_ABORT else "",
        )
    )
