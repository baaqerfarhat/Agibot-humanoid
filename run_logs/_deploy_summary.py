#!/usr/bin/env python3
"""One line per deploy run: how far it got, whether it stayed up, what saturated.

Written to compare the action-clip A/B, where the question is narrow -- does
bounding the ankle-roll and wrist actions change how far the motion gets before
the robot leans past recovery -- and the runs are otherwise identical.

Columns:
  frames    last motion frame the policy reached; 584 finishes the motion and the
            deploy then holds, so anything >=584 got all the way through
  t_pol     seconds spent in the policy phase
  end_up    torso tilt over the last half second, in rad. Small means it was still
            standing when the run ended
  rise      worst tilt from frame 180 on. The deep bend before that reaches 80 deg
            by design, so only the rise and carry say anything about balance
  ARroll    largest |right_ankle_roll| reached; its URDF stop is 0.263
  wp_err    mean |waist_pitch - target|, the droop the waist never closes

Attitude comes from the gravity direction in the torso frame, not from a
world-frame roll and pitch. These logs carry 100-130 deg of yaw, which leaks into
a naive roll/pitch and made every run look like it was pitched 85 deg over.

Usage: ~/baaqer_ws/mjlab/.venv/bin/python run_logs/_deploy_summary.py [glob]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np


def proj_gravity(qx, qy, qz, qw):
    """World -z expressed in the body frame. Upright is (0, 0, -1)."""
    return np.array([
        -2 * (qx * qz - qw * qy),
        -2 * (qy * qz + qw * qx),
        -(1 - 2 * (qx * qx + qy * qy)),
    ])


def lean_tilt(q):
    g = proj_gravity(*q)
    return (np.arctan2(-g[0], -g[2]), np.arctan2(g[1], -g[2]),
            np.arccos(np.clip(-g[2], -1, 1)))


def main() -> None:
    HERE = os.path.dirname(os.path.abspath(__file__))
    pat = sys.argv[1] if len(sys.argv) > 1 else "*box_pickup*.csv"
    files = sorted(glob.glob(os.path.join(HERE, pat)))
    print(f"{'run':>7s}{'policy':>10s}{'clip':>6s}{'gain':>6s}{'frames':>8s}"
          f"{'t_pol':>7s}{'end_up':>7s}{'carry':>7s}{'ARroll':>8s}"
          f"{'wp_err':>8s}")
    for p in files:
        try:
            meta = json.load(open(p.replace(".csv", ".meta.json")))
        except FileNotFoundError:
            continue
        rows = list(csv.DictReader(open(p)))
        jn = meta["joint_names"]
        pol = [r for r in rows if r["phase"] == "policy"]
        tag = os.path.basename(p)[9:15]
        name = os.path.basename(meta.get("policy", "?")).replace(
            "x2_box_policy_", "").replace(".npz", "")[:13]
        if not pol:
            last = rows[-1]["phase"] if rows else "empty"
            print(f"{tag:>7s}{name:>10s}{meta.get('action_clip', 0):6.0f}"
                  f"{meta.get('gain_scale', 1):6.2f}{'-':>8s}{'-':>7s}"
                  f"{'stuck@' + last:>10s}")
            continue
        fr = np.array([int(r["frame"]) for r in pol])
        t = np.array([float(r["t_s"]) for r in pol])
        lean, tilt, off = zip(*[lean_tilt([float(r[f"base_quat_{k}"]) for k in "xyzw"])
                                for r in pol])
        ar = np.abs([float(r["right_ankle_roll_joint__pos_meas"]) for r in pol]).max()
        wp = np.abs([float(r["waist_pitch_joint__pos_meas"])
                     - float(r["waist_pitch_joint__tgt"]) for r in pol]).mean()
        off = np.asarray(off)
        end_up = off[t >= t[-1] - 0.5].mean()
        # Frames 160-360 are the standing carry: up out of the bend, box held, torso
        # near vertical. Before that the motion bends 80 deg forward on purpose, and
        # after ~400 it bends down again to set the box back down, so a tilt outside
        # this window says nothing about whether the robot was balanced.
        late = off[(fr >= 160) & (fr <= 360)]
        rise = late.max() if late.size else np.nan
        print(f"{tag:>7s}{name:>10s}{meta.get('action_clip', 0):6.0f}"
              f"{meta.get('gain_scale', 1):6.2f}{fr.max():8d}{t[-1] - t[0]:7.1f}"
              f"{end_up:7.2f}{rise:7.2f}{ar:8.3f}{wp:8.3f}"
              + ("" if fr.max() >= 584 else "   stopped early"))


if __name__ == "__main__":
    main()
