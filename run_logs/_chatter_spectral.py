#!/usr/bin/env python3
"""Separate oscillation from task motion, so policies on different clips compare.

Raw per-tick target motion conflates two things: a walking clip legitimately moves
the legs faster than an in-place one, and a shaking policy moves them faster still.
Comparing v16 on the 511-frame walking clip against v33 on the 733-frame in-place
clip by |dtgt| therefore penalises v16 for the task.

A box pickup is a sub-2 Hz motion. Anything a joint target does above 5 Hz is not
the task, so high-passing there leaves only what the policy added, and that number
is comparable across clips. The reference clip run through the same filter gives
the floor -- non-zero only because retargeting leaves some numerical roughness.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

LEG = ("hip", "knee", "ankle")
FS = 50.0
CUT = 5.0


def highpass_rms(x: np.ndarray, cols) -> float:
    """RMS of the >5 Hz content of each column, averaged over `cols`, in mrad."""
    n = len(x)
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    X = np.fft.rfft(x[:, cols] - x[:, cols].mean(axis=0), axis=0)
    X[f < CUT] = 0.0
    hi = np.fft.irfft(X, n=n, axis=0)
    return float(np.sqrt((hi ** 2).mean()) * 1000)


def main():
    mot = Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
               "x2_31dof/whole_body_tracking")
    print(f"RMS of >{CUT:.0f} Hz content in the LEG position targets (mrad).")
    print("This is oscillation only: the pickup itself lives below 2 Hz.\n")

    # reference floors, both clips
    for tag, fn in (("walking clip (v6-v16)", "sub3_largebox_003_walk_feasible.npz"),
                    ("in-place clip (v33/v5)", "sub3_largebox_003_mj_w_obj.npz")):
        p = mot / fn
        if not p.exists():
            p = mot / "box_multispeed" / fn
        if not p.exists():
            continue
        c = np.load(p, allow_pickle=True)
        jn = [str(x) for x in c["joint_names"]]
        leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]
        q = np.asarray(c["joint_pos"])[:, 7:]
        print(f"  REFERENCE {tag:26s} {highpass_rms(q, leg):6.2f}")
    print()

    best = {}
    for f in sorted(glob.glob("/tmp/x2_box*rollout.npz")):
        m = re.match(r".*/x2_box_(.+)_iter(\d+)_rollout\.npz", f)
        if not m:
            continue
        name, it = m.group(1), int(m.group(2))
        if name not in best or it > best[name][0]:
            best[name] = (it, f)

    rows = []
    for name, (it, f) in best.items():
        d = np.load(f, allow_pickle=True)
        if "dof_pos_target" not in d.files:
            continue
        meta = json.loads(str(d["_metadata_json"]))
        jn = meta["dof_names"]
        leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]
        rows.append((name, it, highpass_rms(d["dof_pos_target"], leg),
                     highpass_rms(d["dof_pos"], leg)))

    print(f"{'policy':34s} {'target':>8} {'measured':>9}")
    for name, it, a, b in sorted(rows, key=lambda x: x[2]):
        note = ""
        if name.startswith("v33"):
            note = "  <- ran on hardware clean"
        if name.startswith("walk_feasible_v16"):
            note = "  <- jittered on hardware"
        print(f"{name+' @'+str(it):34s} {a:8.2f} {b:9.2f}{note}")

    # and the hardware run itself, same filter
    here = Path(__file__).resolve().parent
    hw = here / "20260825_173138_box_pickup_x2_box_policy_walk_feasible_v16_iter30500.csv"
    lines = hw.read_text().splitlines()
    hdr = lines[0].split(",")
    rowsx = [ln for ln in lines[1:] if ln.count(",") == len(hdr) - 1]
    raw = np.genfromtxt(rowsx, delimiter=",", dtype=float)
    col = {c: i for i, c in enumerate(hdr)}
    ph = np.array([r.split(",")[col["phase"]] for r in rowsx])
    m = ph == "policy"
    meta = json.loads(hw.with_suffix("").with_suffix(".meta.json").read_text())
    jn = meta["joint_names"]
    leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]
    tgt = np.stack([raw[m, col[f"{j}__tgt"]] for j in jn], axis=1)
    pos = np.stack([raw[m, col[f"{j}__pos_meas"]] for j in jn], axis=1)
    print(f"\n{'HARDWARE v16 @30500':34s} {highpass_rms(tgt, leg):8.2f} {highpass_rms(pos, leg):9.2f}")


if __name__ == "__main__":
    main()
