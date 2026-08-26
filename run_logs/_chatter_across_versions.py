#!/usr/bin/env python3
"""Per-tick target chatter for every box policy we have a sim rollout for.

The reference is the floor: it is what a joint has to do to perform the motion.
Anything above it is the policy adding motion of its own. Direction reversals are
the sharper measure -- a policy tracking a smooth trajectory reverses a joint only
at the turning points, so a high reversal rate is oscillation by definition and
cannot be explained by the task being fast.

v33 is the reference point that matters: it is the one that ran on hardware
without jitter.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

CLIPS = {
    "walk_feasible": "sub3_largebox_003_walk_feasible.npz",
    "clean_grasp": None,
    "clean_retrain": None,
    "v33": None,
}
MOT = Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking")
LEG = ("hip", "knee", "ankle")


def chatter(tgt, vel, leg):
    dt = np.diff(tgt, axis=0)
    rev = np.sign(dt[1:]) * np.sign(dt[:-1]) < 0
    dv = np.diff(vel, axis=0)
    vrev = np.sign(dv[1:]) * np.sign(dv[:-1]) < 0
    return (np.abs(dt[:, leg]).mean() * 1000, 100 * rev[:, leg].mean(),
            100 * vrev[:, leg].mean(), np.abs(vel[:, leg]).mean())


def main():
    files = sorted(glob.glob("/tmp/x2_box*rollout.npz"))
    # newest checkpoint per experiment only
    best = {}
    for f in files:
        m = re.match(r".*/x2_box_(.+)_iter(\d+)_rollout\.npz", f)
        if not m:
            continue
        name, it = m.group(1), int(m.group(2))
        if name not in best or it > best[name][0]:
            best[name] = (it, f)

    # the reference floor for the walking clip
    c = np.load(MOT / "sub3_largebox_003_walk_feasible.npz", allow_pickle=True)
    q = np.asarray(c["joint_pos"])[:, 7:]
    jn = [str(x) for x in c["joint_names"]]
    leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]
    r = chatter(q, np.asarray(c["joint_vel"]), leg)
    print(f"{'policy':34s} {'|dtgt|':>8} {'reversals':>10} {'accel rev':>10} {'|vel|':>7}")
    print(f"{'-'*72}")
    print(f"{'REFERENCE (walk_feasible clip)':34s} {r[0]:7.1f}  {r[1]:9.1f}% {r[2]:9.1f}% {r[3]:7.2f}")
    print()
    rows = []
    for name, (it, f) in best.items():
        d = np.load(f, allow_pickle=True)
        if "dof_pos_target" not in d.files:
            continue
        rows.append((name, it, chatter(d["dof_pos_target"], d["dof_vel"], leg)))
    for name, it, (a, b, cc, v) in sorted(rows, key=lambda x: x[2][1]):
        tag = f"{name} @{it}"
        note = ""
        if name.startswith("v33"):
            note = "  <- ran on hardware clean"
        if name.startswith("walk_feasible_v16"):
            note = "  <- jittered on hardware"
        print(f"{tag:34s} {a:7.1f}  {b:9.1f}% {cc:9.1f}% {v:7.2f}{note}")


if __name__ == "__main__":
    main()
