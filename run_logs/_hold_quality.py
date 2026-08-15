#!/usr/bin/env python3
"""Is a static-pose hold trustworthy enough to difference against the model?

The comparison in _static_pose_compare.py assumes three things that the hold log
can confirm or deny: the robot was engaged, it actually reached the commanded
pose, and it had stopped moving. It also assumes the robot was standing on its
own two feet in the same way the sim does, and the giveaway when it was not is
the torso attitude -- a pose that settles upright in sim but sits pitched over on
hardware is not the same load case, and differencing the two says nothing.

Usage:  ~/baaqer_ws/mjlab/.venv/bin/python run_logs/_hold_quality.py <csv> [...]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rp(qx, qy, qz, qw):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return roll, pitch


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob(os.path.join(HERE, "*static_pose_id*.csv")))
    for name in args:
        p = name if os.path.isabs(name) else os.path.join(HERE, name)
        rows = list(csv.DictReader(open(p)))
        meta = json.load(open(p.replace(".csv", ".meta.json")))
        jn = meta["joint_names"]
        settle = float(meta.get("settle_s", 1.5))
        print(f"\n=== {os.path.basename(p)}")
        print(f"    engage={meta.get('engage')}  frames={meta.get('frames')}  "
              f"gain={meta.get('gain_scale')}")
        if not meta.get("engage"):
            print("    DRY RUN -- nothing was commanded, the robot never moved. "
                  "Not usable.")
            continue
        phases = sorted({r["phase"] for r in rows if r["phase"].startswith("hold")},
                        key=lambda s: int(s[4:]))
        print(f"    {'pose':>5s}{'n':>5s}{'|q-tgt|max':>12s}{'joint':>24s}"
              f"{'|qvel|max':>11s}{'roll':>8s}{'pitch':>8s}{'tau drift':>11s}")
        for ph in phases:
            sel = [r for r in rows if r["phase"] == ph]
            t0 = float(sel[0]["t_s"])
            sel = [r for r in sel if float(r["t_s"]) - t0 >= settle] or sel
            q = np.array([[float(r[f"{n}__pos_meas"]) for n in jn] for r in sel])
            tg = np.array([[float(r[f"{n}__tgt"]) for n in jn] for r in sel])
            dq = np.array([[float(r[f"{n}__vel_meas"]) for n in jn] for r in sel])
            tau = np.array([[float(r[f"{n}__eff_meas"]) for n in jn] for r in sel])
            err = np.abs(q - tg).mean(axis=0)
            w = max(len(sel) // 4, 1)
            drift = np.abs(tau[-w:].mean(axis=0) - tau[:w].mean(axis=0)).max()
            roll, pitch = rp(*[np.mean([float(r[f"base_quat_{k}"]) for r in sel])
                               for k in "xyzw"])
            print(f"    {ph[4:]:>5s}{len(sel):5d}{err.max():12.3f}"
                  f"{jn[int(np.argmax(err))]:>24s}{np.abs(dq).max():11.3f}"
                  f"{roll:+8.3f}{pitch:+8.3f}{drift:11.2f}")
        print("    (|q-tgt| in rad: above ~0.1 the robot is not in the pose it was "
              "asked for.\n     tau drift in Nm: above a few Nm it was still "
              "settling, or being moved.)")


if __name__ == "__main__":
    main()
