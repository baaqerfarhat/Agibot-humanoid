#!/usr/bin/env python3
"""Roll the exported squat npz policy in mjlab and dump arrays + MP4.

Run from the mjlab checkout so `uv run` sees the project env:

    cd ~/baaqer_ws/mjlab
    MUJOCO_GL=egl uv run python ~/baaqer_ws/Agibot-humanoid/squat/dump_squat_mjlab.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import torch

SQUAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SQUAT_DIR))

from squat_policy_common import (  # noqa: E402
    CONTROL_DT,
    DEFAULT_POLICY,
    HOLD_S,
    NUM_STEPS,
    NumpyPolicy,
    rpy_from_xyzw,
    save_rollout,
)

OUT_DIR = SQUAT_DIR / "compare"
OUT_NPZ = OUT_DIR / "mjlab_rollout.npz"
OUT_MP4 = OUT_DIR / "mjlab.mp4"


def _actor_obs(td) -> np.ndarray:
    if "actor" in td.keys():
        t = td["actor"]
    elif "policy" in td.keys():
        t = td["policy"]
    else:
        raise KeyError(f"No actor/policy obs in TensorDict keys={list(td.keys())}")
    return t[0].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.squat.checkpoint import TASK_ID
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    policy_path = Path(os.environ.get("SQUAT_POLICY", str(DEFAULT_POLICY)))
    policy = NumpyPolicy(policy_path)
    meta = policy.meta
    print(f"[mjlab] policy={policy_path} obs_dim={meta['obs_dim']} joints={len(meta['joint_names'])}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = int(1e9)
    env_cfg.auto_reset = False
    env_cfg.terminations = {}
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.viewer.width = 640
    env_cfg.viewer.height = 480
    env_cfg.commands["squat"].squat_height_frac = float(meta["squat_height_frac"])
    env_cfg.commands["squat"].wrap_cycle = False

    agent_cfg = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    npz_names = list(meta["joint_names"])
    if joint_names != npz_names:
        raise RuntimeError(f"mjlab joint order != npz order\n{joint_names}\n{npz_names}")

    default_q = np.asarray(meta["default_joint_pos"], np.float32)
    scale = np.asarray(meta["action_scale"], np.float32)

    logs = {
        "t": [],
        "obs": [],
        "action": [],
        "target": [],
        "command": [],
        "joint_pos": [],
        "joint_vel": [],
        "root_pos": [],
        "root_quat_xyzw": [],
        "base_ang_vel": [],
        "projected_gravity": [],
        "pelvis_height": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
    }
    frames: list[np.ndarray] = []

    obs_td = wrapped.get_observations()
    for step in range(NUM_STEPS):
        obs_np = _actor_obs(obs_td)
        action = policy(obs_np)
        target = action * scale + default_q
        cmd = env.command_manager.get_term("squat")
        command = cmd.command[0].detach().cpu().numpy().astype(np.float32)

        pos = robot.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float32)
        quat_wxyz = robot.data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        q = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        dq = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
        grav = robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
        ang = robot.data.root_link_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        roll, pitch, yaw = rpy_from_xyzw(quat_xyzw)

        logs["t"].append(step * CONTROL_DT)
        logs["obs"].append(obs_np)
        logs["action"].append(action.astype(np.float32))
        logs["target"].append(target.astype(np.float32))
        logs["command"].append(command)
        logs["joint_pos"].append(q)
        logs["joint_vel"].append(dq)
        logs["root_pos"].append(pos)
        logs["root_quat_xyzw"].append(quat_xyzw)
        logs["base_ang_vel"].append(ang)
        logs["projected_gravity"].append(grav)
        logs["pelvis_height"].append(float(pos[2]))
        logs["roll"].append(roll)
        logs["pitch"].append(pitch)
        logs["yaw"].append(yaw)

        act_t = torch.from_numpy(action).to(device=device, dtype=torch.float32).unsqueeze(0)
        obs_td, _rew, dones, _extras = wrapped.step(act_t)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if step % 50 == 0:
            print(
                f"[mjlab] step {step:3d}/{NUM_STEPS}  z={pos[2]:.3f}  "
                f"cmd_h={command[2]:.3f}  done={int(dones[0])}"
            )

    stacked = {k: np.asarray(v) for k, v in logs.items()}
    stacked["joint_names"] = np.array(joint_names)
    stacked["simulator"] = np.array("mjlab")
    stacked["policy"] = np.array(str(policy_path))
    stacked["meta_json"] = np.array(json.dumps(meta))
    stacked["dt"] = np.array(CONTROL_DT)
    save_rollout(OUT_NPZ, **stacked)

    if frames:
        media.write_video(str(OUT_MP4), frames, fps=int(round(1.0 / CONTROL_DT)))
        print(f"[mjlab] wrote {OUT_MP4} ({len(frames)} frames)")
    else:
        print("[mjlab] WARN: no video frames")

    z = stacked["pelvis_height"]
    print(
        f"[mjlab] height min={z.min():.3f} final={z[-1]:.3f}  "
        f"max|roll|={np.abs(stacked['roll']).max():.3f}  "
        f"max|pitch|={np.abs(stacked['pitch']).max():.3f}  "
        f"xy_drift={np.linalg.norm(stacked['root_pos'][:, :2], axis=1).max():.3f}"
    )
    wrapped.close()


if __name__ == "__main__":
    main()
