#!/usr/bin/env python3
"""Replay a squat rollout NPZ in mjlab's renderer (no policy, kinematics only).

Used to visualize the Isaac Sim trajectory when Kit RTX cameras cannot run
headless on this machine.

    cd ~/baaqer_ws/mjlab
    MUJOCO_GL=egl uv run python ~/baaqer_ws/Agibot-humanoid/squat/replay_rollout_mjlab.py \
        --npz ~/baaqer_ws/Agibot-humanoid/squat/compare/isaac_rollout.npz \
        --out ~/baaqer_ws/Agibot-humanoid/squat/compare/isaac.mp4 \
        --label isaac
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mediapy as media
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.squat.checkpoint import TASK_ID
from mjlab.utils.torch import configure_torch_backends


def _set_state(env, robot, root_pos, quat_xyzw, q):
    # mjlab root quat is wxyz
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    data = robot.data
    # Write through the entity API when available.
    if hasattr(robot, "write_root_link_pose_to_sim"):
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, :3] = torch.as_tensor(root_pos, device=env.device)
        pose[0, 3:] = torch.as_tensor(quat_wxyz, device=env.device)
        robot.write_root_link_pose_to_sim(pose)
        vel = torch.zeros(1, 6, device=env.device)
        robot.write_root_link_velocity_to_sim(vel)
        jp = torch.as_tensor(q, device=env.device, dtype=torch.float32).unsqueeze(0)
        jv = torch.zeros_like(jp)
        robot.write_joint_state_to_sim(jp, jv)
    else:
        raise RuntimeError("robot has no write_*_to_sim")
    env.sim.forward()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    d = np.load(args.npz, allow_pickle=True)
    names = [str(x) for x in d["joint_names"]]
    q = d["joint_pos"]
    root = d["root_pos"].copy()
    quat = d["root_quat_xyzw"]
    # Isaac env origin may be 0; keep xy relative to start, z as recorded.
    root[:, :2] -= root[0, :2]
    # If z is ~0.69 at start, keep it. If origin-shifted, subtract start-z then add 0.69.
    if root[0, 2] > 0.4:
        pass
    else:
        root[:, 2] = root[:, 2] - root[0, 2] + 0.69

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False
    env_cfg.terminations = {}
    env_cfg.viewer.width = 640
    env_cfg.viewer.height = 480
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    robot = env.scene["robot"]
    env_names = list(robot.joint_names)
    lookup = {n: i for i, n in enumerate(names)}
    order = [lookup[n] for n in env_names]

    frames = []
    for i in range(len(q)):
        _set_state(env, robot, root[i], quat[i], q[i, order])
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if i % 50 == 0:
            print(f"[replay] {i}/{len(q)} z={root[i, 2]:.3f}")
    env.close()
    fps = int(round(1.0 / float(d["dt"])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(args.out), frames, fps=fps)
    print(f"[replay] wrote {args.out} ({len(frames)} frames @ {fps} fps) label={args.label}")


if __name__ == "__main__":
    main()
