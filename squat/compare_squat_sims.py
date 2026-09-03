#!/usr/bin/env python3
"""Compare mjlab vs Isaac Sim squat rollouts and write canvas_data.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

COMPARE_DIR = Path(__file__).resolve().parent / "compare"
MJLAB_NPZ = COMPARE_DIR / "mjlab_rollout.npz"
ISAAC_NPZ = COMPARE_DIR / "isaac_rollout.npz"
OUT_JSON = COMPARE_DIR / "canvas_data.json"

KEY_JOINTS = [
    "left_knee_joint",
    "right_knee_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]


def rmse(a, b) -> float:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt(np.mean(d * d)))


def maxabs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def downsample(arr, every: int = 5) -> list:
    return [round(float(x), 4) for x in np.asarray(arr)[::every]]


def success(z, roll, pitch, standing: float, target_min: float) -> dict:
    min_z = float(np.min(z))
    final_z = float(z[-1])
    max_rp = float(max(np.max(np.abs(roll)), np.max(np.abs(pitch))))
    reached = min_z <= target_min + 0.04
    stood = abs(final_z - standing) <= 0.08
    upright = max_rp < 0.45
    return {
        "min_height_m": round(min_z, 4),
        "final_height_m": round(final_z, 4),
        "max_roll_rad": round(float(np.max(np.abs(roll))), 4),
        "max_pitch_rad": round(float(np.max(np.abs(pitch))), 4),
        "reached_bottom": reached,
        "stood_back_up": stood,
        "stayed_upright": upright,
        "success": bool(reached and stood and upright),
    }


def main() -> None:
    mj = np.load(MJLAB_NPZ, allow_pickle=True)
    isa = np.load(ISAAC_NPZ, allow_pickle=True)
    n = min(len(mj["t"]), len(isa["t"]))
    names = [str(x) for x in mj["joint_names"]]
    standing = 0.69
    target_min = 0.69 * 0.40

    mj_z = mj["pelvis_height"][:n]
    is_z = isa["pelvis_height"][:n]
    # Isaac root z includes env origin; compare relative to each run's start.
    mj_z0 = mj_z - mj_z[0] + standing
    is_z0 = is_z - is_z[0] + standing

    mj_xy = np.linalg.norm(mj["root_pos"][:n, :2] - mj["root_pos"][0, :2], axis=1)
    is_xy = np.linalg.norm(isa["root_pos"][:n, :2] - isa["root_pos"][0, :2], axis=1)

    joint_rmse = {}
    for i, name in enumerate(names):
        joint_rmse[name] = round(rmse(mj["joint_pos"][:n, i], isa["joint_pos"][:n, i]), 4)
    top = sorted(joint_rmse.items(), key=lambda kv: -kv[1])[:10]

    dh = np.abs(mj_z0 - is_z0)
    first_2cm = int(np.argmax(dh > 0.02)) if np.any(dh > 0.02) else None
    first_2cm_t = None if first_2cm is None else round(float(mj["t"][first_2cm]), 3)

    mj_ok = success(mj_z, mj["roll"][:n], mj["pitch"][:n], standing, target_min)
    is_ok = success(is_z0, isa["roll"][:n], isa["pitch"][:n], standing, target_min)

    # Use absolute Isaac height for display if origin is ~0, else origin-corrected.
    isaac_height_plot = is_z0.tolist()
    mj_height_plot = mj_z.tolist()

    every = 5
    cats = [f"{x:.1f}" for x in mj["t"][:n:every]]

    key_series = {}
    for jn in KEY_JOINTS:
        if jn not in names:
            continue
        i = names.index(jn)
        key_series[jn] = {
            "mjlab": downsample(mj["joint_pos"][:n, i], every),
            "isaac": downsample(isa["joint_pos"][:n, i], every),
            "rmse_rad": joint_rmse[jn],
            "rmse_deg": round(joint_rmse[jn] * 180.0 / math.pi, 2),
        }

    action_l2 = np.linalg.norm(mj["action"][:n] - isa["action"][:n], axis=1)
    q_l2 = np.linalg.norm(mj["joint_pos"][:n] - isa["joint_pos"][:n], axis=1)

    data = {
        "n_steps": n,
        "dt": 0.02,
        "policy": "x2_squat_policy_40pct_iter16499.npz",
        "videos": {
            "mjlab": str(COMPARE_DIR / "mjlab.mp4"),
            "isaac": str(COMPARE_DIR / "isaac.mp4"),
        },
        "mjlab_success": mj_ok,
        "isaac_success": is_ok,
        "same_outcome": mj_ok["success"] == is_ok["success"],
        "metrics": {
            "pelvis_height_rmse_m": round(rmse(mj_z0, is_z0), 4),
            "pelvis_height_max_abs_m": round(maxabs(mj_z0, is_z0), 4),
            "joint_pos_rmse_rad": round(rmse(mj["joint_pos"][:n], isa["joint_pos"][:n]), 4),
            "joint_pos_max_abs_rad": round(maxabs(mj["joint_pos"][:n], isa["joint_pos"][:n]), 4),
            "action_rmse": round(rmse(mj["action"][:n], isa["action"][:n]), 4),
            "action_max_abs": round(maxabs(mj["action"][:n], isa["action"][:n]), 4),
            "xy_drift_mjlab_m": round(float(mj_xy.max()), 4),
            "xy_drift_isaac_m": round(float(is_xy.max()), 4),
            "first_height_gap_gt_2cm_s": first_2cm_t,
        },
        "top_joint_rmse_rad": [{"joint": k, "rmse_rad": v, "rmse_deg": round(v * 180 / math.pi, 2)} for k, v in top],
        "charts": {
            "t": cats,
            "height_mjlab": downsample(mj_height_plot, every),
            "height_isaac": downsample(isaac_height_plot, every),
            "roll_mjlab_deg": downsample(np.rad2deg(mj["roll"][:n]), every),
            "roll_isaac_deg": downsample(np.rad2deg(isa["roll"][:n]), every),
            "pitch_mjlab_deg": downsample(np.rad2deg(mj["pitch"][:n]), every),
            "pitch_isaac_deg": downsample(np.rad2deg(isa["pitch"][:n]), every),
            "xy_mjlab": downsample(mj_xy, every),
            "xy_isaac": downsample(is_xy, every),
            "action_l2": downsample(action_l2, every),
            "joint_l2": downsample(q_l2, every),
            "joints": key_series,
        },
        "notes": [
            "Same npz policy (iter 16499, 40% squat), 50 Hz, 5.0 s cycle + 1.5 s hold, wrap_cycle=False.",
            "Isaac boots the existing Holosoma X2 WBT env then drives the squat npz (not the WBT policy).",
            "Isaac PD / armature / friction swapped to mjlab values; box teleported away; standing z=0.69 m.",
            "Remaining differences are physics (PhysX vs MuJoCo), contacts (full vs feet-only), IMU site vs root.",
        ],
    }
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(json.dumps({k: data[k] for k in ("metrics", "mjlab_success", "isaac_success", "same_outcome")}, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
