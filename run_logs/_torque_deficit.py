#!/usr/bin/env python3
"""Torque the robot actually applied vs the torque training would have applied.

Training (holosoma JointPositionActionTerm._compute_torques):
    tau = kp * (action*action_scale + default - q) - kd * dq,  clipped to +-effort_limit
and the position target is NEVER clipped to the joint limit.

Deploy (build_area_cmd) clips the position target to the mechanical limit and
sends effort=0, so:
    tau = kp * (clip(target, lo, hi) - q) - kd * dq
A target deliberately parked outside the limit -- the policy's way of asking for
saturated torque -- therefore becomes a near-zero torque request.

The effort limits are recoverable from the npz alone:
    action_scale = 0.25 * effort_limit / kp   =>   effort_limit = 4 * action_scale * kp
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analyze_clip import LIMITS  # noqa: E402
from _replay_deploy import HERE, REPO, Policy, replay  # noqa: E402

# frame windows of the 584-frame reference (hold_frame_range = [211, 311])
WINDOWS = [("hold upright", 0, 35), ("bend down", 35, 180), ("grasp + LIFT UP", 180, 215),
           ("hold at chest", 215, 311), ("set down", 311, 470), ("STAND BACK UP", 470, 584)]


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    jn = policy.meta["joint_names"]
    kp = np.array(policy.meta["joint_stiffness"], np.float32)
    kd = np.array(policy.meta["joint_damping"], np.float32)
    ascale = np.array(policy.meta["action_scale"], np.float32)
    effort = 4.0 * ascale * kp
    lo = np.array([LIMITS[n][0] for n in jn], np.float32)
    hi = np.array([LIMITS[n][1] for n in jn], np.float32)

    print("effort limits recovered from the npz (4 * action_scale * kp):")
    print("  " + "  ".join(f"{n.replace('_joint','')}={e:.1f}"
                           for n, e in zip(jn, effort) if "wrist" in n or "ankle" in n
                           or "waist" in n or "shoulder_roll" in n))
    print()

    files = sys.argv[1:] or ["20260812_122817_box_pickup_x2_box_policy_v33_iter253000.csv",
                             "20260812_122937_box_pickup_x2_box_policy_v33_iter253000.csv"]
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(HERE, f)
        R = replay(p, policy)
        if R is None:
            continue
        tgt, meas, vmeas, frame = R["tgt"], R["meas"], R["vmeas"], R["frame"]

        tau_train = np.clip(kp * (tgt - meas) - kd * vmeas, -effort, effort)
        tau_real = kp * (np.clip(tgt, lo, hi) - meas) - kd * vmeas
        tau_real = np.clip(tau_real, -effort, effort)

        print("=" * 104)
        print(f"{os.path.basename(p)}   leg_filter={R['meta']['leg_filter']} "
              f"gain={R['meta']['gain_scale']}  frames {frame[0]}-{frame[-1]}")
        print("=" * 104)
        satfrac = (np.abs(tau_train) > effort - 1e-3).mean(axis=0)
        print("  joints whose TRAINING torque is saturated (bang-bang by design):")
        for i in np.argsort(-satfrac)[:8]:
            if satfrac[i] <= 0.01:
                break
            deliv = np.abs(tau_real[:, i]).mean() / max(1e-6, np.abs(tau_train[:, i]).mean())
            print(f"      {jn[i]:28s} saturated {100*satfrac[i]:5.1f}% of ticks   "
                  f"limit {effort[i]:5.1f} Nm   robot delivered {100*deliv:5.1f}% of "
                  f"the intended |tau|")

        print("  torque delivered / intended, per motion window (mean |tau| ratio):")
        hdr = "      {:28s}".format("joint") + "".join(f"{w[0][:13]:>14s}" for w in WINDOWS)
        print(hdr)
        watch = ["left_ankle_roll_joint", "right_ankle_roll_joint", "waist_pitch_joint",
                 "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                 "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                 "left_wrist_roll_joint", "left_elbow_joint", "right_elbow_joint",
                 "left_knee_joint", "left_hip_pitch_joint"]
        for n in watch:
            i = jn.index(n)
            cells = ""
            for _, a, b in WINDOWS:
                m = (frame >= a) & (frame < b)
                if m.sum() < 3:
                    cells += f"{'-':>14s}"
                    continue
                num = np.abs(tau_real[m, i]).mean()
                den = np.abs(tau_train[m, i]).mean()
                r = 100.0 * num / max(1e-6, den)
                cells += f"{r:11.0f}%  " if den > 0.5 else f"{'~0':>14s}"
            print(f"      {n:28s}{cells}")

        # total torque deficit as a fraction, whole body, per window
        print("  whole-body: mean |tau_train| vs |tau_real| (Nm) per window:")
        for name, a, b in WINDOWS:
            m = (frame >= a) & (frame < b)
            if m.sum() < 3:
                continue
            print(f"      {name:16s} intended {np.abs(tau_train[m]).mean():6.2f}  "
                  f"delivered {np.abs(tau_real[m]).mean():6.2f}  "
                  f"deficit {100*(1-np.abs(tau_real[m]).mean()/max(1e-6,np.abs(tau_train[m]).mean())):5.1f}%")
        print()


if __name__ == "__main__":
    main()
