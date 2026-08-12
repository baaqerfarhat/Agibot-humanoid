#!/usr/bin/env python3
"""Convert a holosoma WBT box-pickup checkpoint (.pt) into a self-contained .npz.

Run in any environment with torch + yaml + numpy (e.g. the holosoma conda env):

    python export_box_policy_npz.py \
        --checkpoint <run_dir>/model_89500.pt \
        --config     <run_dir>/holosoma_config.yaml \
        --motion     <holosoma>/data/motions/x2_31dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz \
        --out        ../box_pickup/policy/x2_box_policy.npz

The output holds the obs normalizer, the actor MLP weights, the deployment
metadata (joint order, default pose, per-joint action scale, PD gains, obs term
order) AND the reference motion arrays the observation needs at runtime
(reference joint pos/vel and reference torso orientation per 50 Hz frame).
`deploy_x2_box_pickup.py` then needs only numpy on the robot.

Network (rsl_rl actor, ELU activations, normalization baked in):
    x = (obs - mean) / std
    for W, b in hidden:  x = elu(x @ W.T + b)
    action = x @ W_out.T + b_out
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml


def match_gain(joint_name: str, gain_dict: dict) -> float:
    """Resolve a per-joint gain from holosoma's substring-keyed dict."""
    matches = [k for k in gain_dict if k in joint_name]
    if not matches:
        raise KeyError(f"No gain entry matches joint {joint_name!r}")
    # Longest substring wins (e.g. 'ankle_pitch' over 'ankle').
    key = max(matches, key=len)
    return float(gain_dict[key])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="holosoma model_*.pt")
    ap.add_argument("--config", required=True, help="holosoma_config.yaml from the run dir")
    ap.add_argument("--motion", required=True, help="reference motion .npz used in training")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--hold-frames", type=int, nargs=2, default=None, metavar=("H0", "H1"),
                    help="50 Hz frame range of the clip's static HOLD segment "
                         "(for the hybrid carry hand-off), e.g. --hold-frames 161 261")
    args = ap.parse_args()

    # ---------------- checkpoint: actor MLP + obs normalizer ----------------
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = ck["actor_model_state_dict"]
    norm = ck["actor_obs_normalizer_state_dict"]

    layer_ids = sorted(
        {int(k.split(".")[2]) for k in actor if k.startswith("actor_module.module.") and k.endswith(".weight")}
    )
    weights = [actor[f"actor_module.module.{i}.weight"].numpy().astype(np.float32) for i in layer_ids]
    biases = [actor[f"actor_module.module.{i}.bias"].numpy().astype(np.float32) for i in layer_ids]

    mean = norm["_mean"].numpy().reshape(-1).astype(np.float32)
    # holosoma's EmpiricalNormalization computes (x - mean) / (std + eps) with
    # eps=1e-2; bake the eps in so the runtime does a plain (x - mean) / std.
    std = norm["_std"].numpy().reshape(-1).astype(np.float32) + 1e-2

    # ---------------- training config: joints, gains, scales ----------------
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

    # ---------------- reference motion ----------------
    m = np.load(args.motion, allow_pickle=True)
    motion_joints = list(m["joint_names"])
    assert motion_joints == joint_names, "motion joint order != robot dof order"
    ref_joint_pos = m["joint_pos"][:, 7:].astype(np.float32)  # strip root [xyz, wxyz]
    ref_joint_vel = m["joint_vel"][:, 6:].astype(np.float32)  # strip root [v, w]
    body_names = list(m["body_names"])
    ref_body = cfg["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["body_name_ref"][0]
    torso_idx = body_names.index(ref_body)
    # npz stores wxyz; convert to xyzw to match the IMU/holosoma runtime convention.
    q_wxyz = m["body_quat_w"][:, torso_idx].astype(np.float32)
    ref_quat_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    fps = int(np.asarray(m["fps"]).reshape(-1)[0])

    obs_dim = int(mean.shape[0])
    expected = 2 * len(joint_names) + 6 + 3 + 3 * len(joint_names)  # cmd(62)+ori(6)+angvel(3)+q/dq/act(93)
    assert obs_dim == expected, f"obs_dim {obs_dim} != expected {expected}"

    meta = {
        "task": "x2_box_pickup_wbt",
        "joint_names": joint_names,
        "default_joint_pos": default,
        "action_scale": action_scale,
        "joint_stiffness": stiffness,
        "joint_damping": damping,
        # Training clips the PD torque to these (clip_torques) but never clips the
        # position target to the joint limit, so the deploy side needs them to
        # reproduce the saturated-torque commands the policy relies on.
        "joint_effort_limit": effort,
        "clip_torques": bool(ctl.get("clip_torques", False)),
        # NOTE: holosoma concatenates group terms in ALPHABETICAL order.
        "observation_names": [
            "actions",             # previous raw action (31)
            "base_ang_vel",        # torso IMU gyro, base frame (3)
            "dof_pos",             # joint_pos - default (31)
            "dof_vel",             # joint_vel (31)
            "motion_command",      # [ref_joint_pos(31), ref_joint_vel(31)] at motion clock
            "motion_ref_ori_b",    # ref torso ori relative to actual torso, 6D (first 2 rot-mat cols)
        ],
        "ref_body": ref_body,
        "motion_fps": fps,
        "motion_frames": int(ref_joint_pos.shape[0]),
        "control_hz": 50,
        "run_path": args.checkpoint,
        "obs_dim": obs_dim,
        "action_dim": int(weights[-1].shape[0]),
    }
    if args.hold_frames is not None:
        meta["hold_frame_range"] = [int(args.hold_frames[0]), int(args.hold_frames[1])]

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
