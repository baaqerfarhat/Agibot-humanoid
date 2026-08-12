#!/usr/bin/env python3
"""Quantify what the deploy-side joint-limit clip does to the policy command.

Two questions:

1. How much of the commanded position target is thrown away by the
   `np.clip(pos, lower_limit, upper_limit)` in build_area_cmd()?
   The policy uses deliberately-saturated position targets to ask for max
   torque; clipping the target to the mechanical limit converts a max-torque
   request into a ~zero-torque one.

2. Which observation dimensions are off the training distribution?
   The npz carries the empirical obs normalizer (mean/std) from training, so
   z = (obs_real - mean)/std is a direct per-dimension sim-vs-real gap measure.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _replay_deploy import HERE, REPO, Policy, replay  # noqa: E402

# from agibot_control_functions/robot_states_control.py robot_model
LIMITS = {
    "head_yaw_joint": (-0.366, 0.366), "head_pitch_joint": (-0.3838, 0.3838),
    "waist_yaw_joint": (-3.43, 2.382), "waist_pitch_joint": (-0.314, 0.314),
    "waist_roll_joint": (-0.488, 0.488),
    "left_shoulder_pitch_joint": (-3.08, 2.04), "left_shoulder_roll_joint": (-0.061, 2.993),
    "left_shoulder_yaw_joint": (-2.556, 2.556), "left_elbow_joint": (-2.3556, 0.0),
    "left_wrist_yaw_joint": (-2.556, 2.556), "left_wrist_pitch_joint": (-0.558, 0.558),
    "left_wrist_roll_joint": (-1.571, 0.724),
    "right_shoulder_pitch_joint": (-3.08, 2.04), "right_shoulder_roll_joint": (-2.993, 0.061),
    "right_shoulder_yaw_joint": (-2.556, 2.556), "right_elbow_joint": (-2.3556, 0.0),
    "right_wrist_yaw_joint": (-2.556, 2.556), "right_wrist_pitch_joint": (-0.558, 0.558),
    "right_wrist_roll_joint": (-0.724, 1.571),
    "left_hip_pitch_joint": (-2.704, 2.556), "left_hip_roll_joint": (-0.235, 2.906),
    "left_hip_yaw_joint": (-1.684, 3.430), "left_knee_joint": (0.0, 2.4073),
    "left_ankle_pitch_joint": (-0.803, 0.453), "left_ankle_roll_joint": (-0.2625, 0.2625),
    "right_hip_pitch_joint": (-2.704, 2.556), "right_hip_roll_joint": (-2.906, 0.235),
    "right_hip_yaw_joint": (-3.430, 1.684), "right_knee_joint": (0.0, 2.4073),
    "right_ankle_pitch_joint": (-0.803, 0.453), "right_ankle_roll_joint": (-0.2625, 0.2625),
}

OBS_GROUPS = [("actions", 31), ("base_ang_vel", 3), ("dof_pos", 31), ("dof_vel", 31),
              ("ref_pos", 31), ("ref_vel", 31), ("ori6", 6)]


def clip_report(R, policy):
    jn = R["jn"]
    raw, meas, tgt = R["raw"], R["meas"], R["tgt"]
    kp = np.array(policy.meta["joint_stiffness"], np.float32)
    kd = np.array(policy.meta["joint_damping"], np.float32)
    lo = np.array([LIMITS[n][0] for n in jn], np.float32)
    hi = np.array([LIMITS[n][1] for n in jn], np.float32)

    sent = np.clip(tgt, lo, hi)          # what actually left the deploy script
    lost = tgt - sent                    # command amplitude discarded by the clip
    vel = np.array([[0.0]])              # velocity term is commanded 0 in build_area_cmd

    # torque the robot's servo produces vs the torque Isaac would have produced
    # (Isaac's implicit PD does NOT clamp the target to the joint limit)
    tau_real = kp * (sent - meas) - kd * R.get("vmeas", np.zeros_like(meas))
    tau_sim = kp * (tgt - meas) - kd * R.get("vmeas", np.zeros_like(meas))

    print("  --- joint-limit clip on the OUTGOING command ---")
    frac = (np.abs(lost) > 1e-6).mean(axis=0)
    order = np.argsort(-frac)
    any_clip = frac[order[0]] > 0
    if not any_clip:
        print("      none")
    for i in order[:10]:
        if frac[i] <= 0:
            break
        k = int(np.argmax(np.abs(lost[:, i])))
        print(f"      {jn[i]:28s} clipped {100*frac[i]:5.1f}% of ticks   "
              f"max discarded {np.abs(lost[:,i]).max():6.2f} rad   "
              f"tau_sim {tau_sim[k,i]:+8.1f} -> tau_real {tau_real[k,i]:+7.1f} Nm "
              f"@frame {R['frame'][k]}")
    return lost, tau_sim, tau_real


def zscore_report(R, policy):
    """Rebuild the obs matrix for the run and z-score it against the training
    normalizer that shipped in the npz."""
    jn = R["jn"]
    default = np.array(policy.meta["default_joint_pos"], np.float32)
    mean, std = policy.mean, policy.std
    n = len(R["t"])
    # actions(31) angvel(3) dofpos(31) dofvel(31) refpos(31) refvel(31) ori6(6)
    obs = np.zeros((n, 164), np.float32)
    obs[1:, 0:31] = R["act"][:-1]
    obs[:, 31:34] = R["w"]
    obs[:, 34:65] = R["meas"] - default
    obs[:, 65:96] = R["vmeas"]
    obs[:, 96:127] = policy.ref_joint_pos[R["frame"]]
    obs[:, 127:158] = policy.ref_joint_vel[R["frame"]]
    obs[:, 158:164] = R["ori6"]
    z = (obs - mean) / std

    print("  --- observation z-score vs the TRAINING normalizer ---")
    labels = []
    for g, c in OBS_GROUPS:
        labels += [f"{g}:{jn[i] if c == 31 else i}" for i in range(c)]
    zmax = np.abs(z).max(axis=0)
    i0 = 0
    for g, c in OBS_GROUPS:
        blk = np.abs(z[:, i0:i0 + c])
        print(f"      {g:14s} |z| mean {blk.mean():5.2f}  p99 {np.percentile(blk,99):6.2f}  "
              f"max {blk.max():7.2f}")
        i0 += c
    print("      worst dims:")
    for i in np.argsort(-zmax)[:10]:
        k = int(np.argmax(np.abs(z[:, i])))
        print(f"        {labels[i]:42s} |z|max {zmax[i]:7.1f} @frame {R['frame'][k]:3d}  "
              f"(train mean {mean[i]:+.3f} std {std[i]:.3f}, real {obs[k,i]:+.3f})")
    return z


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    files = sys.argv[1:] or ["20260812_122817_box_pickup_x2_box_policy_v33_iter253000.csv",
                             "20260812_122937_box_pickup_x2_box_policy_v33_iter253000.csv"]
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(HERE, f)
        R = replay(p, policy)
        if R is None:
            continue
        print("=" * 100)
        print(f"{os.path.basename(p)}  leg_filter={R['meta']['leg_filter']} "
              f"gain={R['meta']['gain_scale']}")
        print("=" * 100)
        clip_report(R, policy)
        zscore_report(R, policy)
        print()


if __name__ == "__main__":
    main()
