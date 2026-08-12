#!/usr/bin/env python3
"""Offline validation of `base_frame.PelvisEstimator` against recorded runs.

Replays the logged v33 hardware runs through the estimator and asks one question:
does the corrected `base_ang_vel` look like what the policy was TRAINED on?

The Isaac rollout logs `root_quat_xyzw` (the pelvis, which is holosoma's base),
so the training-time signal is recovered by finite-differencing that quaternion
and rotating into the pelvis frame -- the same quantity holosoma computes as
`quat_rotate_inverse(base_quat, robot_root_states[:, 10:13])`.

Run:
    python validate_base_frame.py
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

import base_frame as bf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "run_logs", "box_pickup_v33_last5")
SIM = os.path.join(REPO, "adaptation", "isaac_runs", "v33", "isaac_frozen_npz_seed600.npz")
POLICY = os.path.join(REPO, "box_pickup", "policy", "x2_box_policy_v33_iter253000.npz")
DT = 0.02
BEND = (60, 110)  # motion frames spanning the deep bend where every run fails


class J:
    __slots__ = ("position", "velocity")

    def __init__(self, p, v):
        self.position, self.velocity = p, v


def quat_rate_body(q: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity from a quaternion sequence (xyzw), central diff."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w = np.zeros((len(q), 3))
    for t in range(1, len(q) - 1):
        # relative rotation from t-1 to t+1 expressed in the body frame at t
        dq = bf._quat_xyzw_to_mat(q[t]).T @ (
            bf._quat_xyzw_to_mat(q[t + 1]) - bf._quat_xyzw_to_mat(q[t - 1])
        ) / (2 * dt)
        w[t] = np.array([dq[2, 1] - dq[1, 2], dq[0, 2] - dq[2, 0], dq[1, 0] - dq[0, 1]]) * 0.5
    w[0], w[-1] = w[1], w[-2]
    return w


def rpy(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(np.stack([roll, pitch], 1))


def main() -> None:
    meta = json.loads(str(np.load(POLICY, allow_pickle=True)["meta_json"]))
    JN = meta["joint_names"]
    wi = [JN.index(n) for n in bf.WAIST_JOINTS]

    # ---------------- training-time truth, from the Isaac pelvis ----------------
    s = np.load(SIM, allow_pickle=True)
    sim_w = quat_rate_body(s["root_quat_xyzw"].astype(float), DT)
    sim_frame = s["frame"]
    sim_bend = (sim_frame >= BEND[0]) & (sim_frame <= BEND[1])
    sim_pelvis_rpy = rpy(s["root_quat_xyzw"].astype(float))

    print("=" * 100)
    print("A. Per-axis base_ang_vel statistics (rad/s), pelvis/base frame")
    print("   sim = what training fed the policy | raw = what the robot fed it | fixed = corrected")
    print("=" * 100)

    raw_all, fix_all, corr_all = [], [], []
    raw_bend, fix_bend = [], []
    tor_rpy_bend, pel_rpy_bend = [], []
    mount_check = []

    for p in sorted(glob.glob(os.path.join(RUNS, "*.csv"))):
        rows = [r for r in csv.DictReader(open(p)) if r["phase"] == "policy"]
        if not rows:
            continue
        frame = np.array([int(r["frame"]) for r in rows])
        w_torso = np.array([[float(r[f"base_ang_vel_{a}"]) for a in "xyz"] for r in rows])
        q_torso = np.array([[float(r[f"base_quat_{a}"]) for a in ("x", "y", "z", "w")] for r in rows])
        qj = np.array([[float(r[f"{n}__pos_meas"]) for n in JN] for r in rows])
        dqj = np.array([[float(r[f"{n}__vel_meas"]) for n in JN] for r in rows])

        est = bf.PelvisEstimator()
        w_fix = np.zeros_like(w_torso)
        q_pel = np.zeros_like(q_torso)
        for t in range(len(rows)):
            jmap = {n: J(qj[t, i], dqj[t, i]) for n, i in zip(bf.WAIST_JOINTS, wi)}
            w_fix[t], q_pel[t] = est.update(q_torso[t], w_torso[t], jmap)

        b = (frame >= BEND[0]) & (frame <= BEND[1])
        raw_all.append(w_torso); fix_all.append(w_fix); corr_all.append(w_fix - w_torso)
        if b.sum():
            raw_bend.append(w_torso[b]); fix_bend.append(w_fix[b])
            tor_rpy_bend.append(rpy(q_torso[b])); pel_rpy_bend.append(rpy(q_pel[b]))
        # standing start: waist ~0 so pelvis and torso must agree; a mismatch here
        # would mean the IMU is not aligned with torso_link
        st = frame <= 20
        if st.sum():
            mount_check.append((rpy(q_torso[st]).mean(0), rpy(q_pel[st]).mean(0),
                                np.degrees(qj[st][:, wi].mean(0))))

    RAW = np.vstack(raw_all); FIX = np.vstack(fix_all); COR = np.vstack(corr_all)
    RB = np.vstack(raw_bend); FB = np.vstack(fix_bend)

    for ax, nm in enumerate(("x (roll)", "y (pitch)", "z (yaw)")):
        print(f"  {nm:11s} sim  mean {sim_w[:, ax].mean():+6.3f} std {sim_w[:, ax].std():5.3f} "
              f"|  raw mean {RAW[:, ax].mean():+6.3f} std {RAW[:, ax].std():5.3f} "
              f"|  fixed mean {FIX[:, ax].mean():+6.3f} std {FIX[:, ax].std():5.3f}")
    print()
    print("  Magnitude of the correction that was missing from every policy step:")
    print(f"    |fixed - raw| mean {np.abs(COR).mean():.3f} rad/s   "
          f"per-axis {np.abs(COR).mean(0).round(3)}   peak {np.abs(COR).max():.2f} rad/s")
    print(f"    relative to the raw signal: {np.abs(COR).mean() / np.abs(RAW).mean() * 100:.0f}% "
          "of the observation the policy was reading")

    print()
    print("=" * 100)
    print(f"B. Inside the failure window (motion frames {BEND[0]}-{BEND[1]}, the deep bend)")
    print("=" * 100)
    for ax, nm in enumerate(("x (roll)", "y (pitch)", "z (yaw)")):
        sb = sim_w[sim_bend, ax]
        print(f"  {nm:11s} sim mean {sb.mean():+6.3f} | raw mean {RB[:, ax].mean():+6.3f} "
              f"| fixed mean {FB[:, ax].mean():+6.3f}    "
              f"raw err vs sim {abs(RB[:, ax].mean()-sb.mean()):.3f} -> "
              f"fixed {abs(FB[:, ax].mean()-sb.mean()):.3f} rad/s")

    TR = np.vstack(tor_rpy_bend); PR = np.vstack(pel_rpy_bend)
    print()
    print("  Attitude during the bend (deg). The policy's motion_ref_ori_b legitimately")
    print("  tracks torso_link, but the ROLL-ABORT check should watch the pelvis:")
    print(f"    torso  roll {TR[:, 0].mean():+6.1f}  pitch {TR[:, 1].mean():+6.1f}")
    print(f"    pelvis roll {PR[:, 0].mean():+6.1f}  pitch {PR[:, 1].mean():+6.1f}   (reconstructed)")
    print(f"    sim pelvis roll {sim_pelvis_rpy[sim_bend, 0].mean():+6.1f}  "
          f"pitch {sim_pelvis_rpy[sim_bend, 1].mean():+6.1f}")

    print()
    print("=" * 100)
    print("C. IMU mount sanity: at the standing start the waist is ~0, so the")
    print("   reconstruction must be a no-op. A residual = IMU not aligned to torso_link.")
    print("=" * 100)
    for i, (t, pel, wj) in enumerate(mount_check):
        print(f"  run {i}: torso roll/pitch {t[0]:+5.2f}/{t[1]:+5.2f}  "
              f"pelvis {pel[0]:+5.2f}/{pel[1]:+5.2f}  waist yaw/pitch/roll "
              f"{wj[0]:+5.2f}/{wj[1]:+5.2f}/{wj[2]:+5.2f} deg")

    d = np.abs(np.vstack([m[1] - m[0] for m in mount_check])).max()
    print(f"\n  max |pelvis - torso| at standing start: {d:.2f} deg "
          f"({'consistent with the logged waist offset' if d < 6 else 'CHECK MOUNT'})")


if __name__ == "__main__":
    main()
