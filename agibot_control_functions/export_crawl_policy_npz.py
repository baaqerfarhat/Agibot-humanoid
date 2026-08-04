#!/usr/bin/env python3
"""Convert a holosoma X2 slope-crawl WBT checkpoint (.pt) into a self-contained .npz.

Same format as `export_box_policy_npz.py`, but for the crawl observation
layout (includes chest/torso projected-gravity).

    python export_crawl_policy_npz.py \
        --checkpoint <run_dir>/model_49999.pt \
        --config     <run_dir>/holosoma_config.yaml \
        --motion     <holosoma>/.../crawl_slope_palmflat_mj.npz \
        --out        ../box_pickup/policy/x2_crawl_policy_v3.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml


def match_gain(joint_name: str, gain_dict: dict) -> float:
    matches = [k for k in gain_dict if k in joint_name]
    if not matches:
        raise KeyError(f"No gain entry matches joint {joint_name!r}")
    key = max(matches, key=len)
    return float(gain_dict[key])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = ck["actor_model_state_dict"]
    norm = ck["actor_obs_normalizer_state_dict"]

    layer_ids = sorted(
        {int(k.split(".")[2]) for k in actor if k.startswith("actor_module.module.") and k.endswith(".weight")}
    )
    weights = [actor[f"actor_module.module.{i}.weight"].numpy().astype(np.float32) for i in layer_ids]
    biases = [actor[f"actor_module.module.{i}.bias"].numpy().astype(np.float32) for i in layer_ids]

    mean = norm["_mean"].numpy().reshape(-1).astype(np.float32)
    std = norm["_std"].numpy().reshape(-1).astype(np.float32) + 1e-2

    cfg = yaml.safe_load(open(args.config))
    rob = cfg["robot"]
    joint_names = list(rob["dof_names"])
    default = [float(rob["init_state"]["default_joint_angles"][n]) for n in joint_names]
    stiffness = [match_gain(n, rob["control"]["stiffness"]) for n in joint_names]
    damping = [match_gain(n, rob["control"]["damping"]) for n in joint_names]
    effort = [float(e) for e in rob["dof_effort_limit_list"]]

    ctl = rob["control"]
    if ctl.get("action_scales_by_effort_limit_over_p_gain"):
        action_scale = [ctl["action_scale"] * e / kp for e, kp in zip(effort, stiffness)]
    else:
        action_scale = [float(ctl["action_scale"])] * len(joint_names)

    m = np.load(args.motion, allow_pickle=True)
    motion_joints = list(m["joint_names"])
    assert motion_joints == joint_names, "motion joint order != robot dof order"
    ref_joint_pos = m["joint_pos"][:, 7:].astype(np.float32)
    ref_joint_vel = m["joint_vel"][:, 6:].astype(np.float32)
    body_names = list(m["body_names"])
    ref_body = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["body_name_ref"][0]
    torso_idx = body_names.index(ref_body)
    q_wxyz = m["body_quat_w"][:, torso_idx].astype(np.float32)
    ref_quat_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    fps = int(np.asarray(m["fps"]).reshape(-1)[0])

    obs_dim = int(mean.shape[0])
    # Crawl actor obs (alphabetical): actions(31) + base_ang_vel(3) + dof_pos(31)
    # + dof_vel(31) + motion_command(62) + motion_ref_ori_b(6) + projected_gravity(3)
    expected = 2 * len(joint_names) + 6 + 3 + 3 * len(joint_names) + 3
    assert obs_dim == expected, f"obs_dim {obs_dim} != expected crawl dim {expected}"

    meta = {
        "task": "x2_crawl_slope_wbt",
        "joint_names": joint_names,
        "default_joint_pos": default,
        "action_scale": action_scale,
        "joint_stiffness": stiffness,
        "joint_damping": damping,
        "observation_names": [
            "actions",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "motion_command",
            "motion_ref_ori_b",
            "projected_gravity",
        ],
        "ref_body": ref_body,
        "motion_fps": fps,
        "motion_frames": int(ref_joint_pos.shape[0]),
        "control_hz": 50,
        "run_path": args.checkpoint,
        "obs_dim": obs_dim,
        "action_dim": int(weights[-1].shape[0]),
        "terrain": "slope",
        "notes": (
            "Prone hands-and-feet slope crawl. Start the robot near the reference "
            "crawl pose (not standing). Uses torso/chest IMU for gyro + projected gravity."
        ),
    }

    save = {
        "mean": mean,
        "std": std,
        "n_layers": np.array(len(weights), dtype=np.int64),
        "meta_json": np.array(json.dumps(meta)),
        "ref_joint_pos": ref_joint_pos,
        "ref_joint_vel": ref_joint_vel,
        "ref_quat_xyzw": ref_quat_xyzw,
    }
    for i, (w, b) in enumerate(zip(weights, biases)):
        save[f"W{i}"] = w
        save[f"b{i}"] = b
    np.savez(args.out, **save)

    print(f"[export] wrote {args.out}")
    print(f"[export] obs_dim={obs_dim} action_dim={meta['action_dim']} layers={[w.shape for w in weights]}")
    print(f"[export] motion: {meta['motion_frames']} frames @ {fps} Hz ({meta['motion_frames']/fps:.1f}s)")


if __name__ == "__main__":
    main()
