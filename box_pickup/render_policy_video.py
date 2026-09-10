"""Render a trained box-pickup checkpoint to mp4 (no GLFW / viser)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="Mjlab-BoxPickup-Flat-X2-DR")
    ap.add_argument("--steps", type=int, default=1100)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 1
    cfg.viewer.height = 480
    cfg.viewer.width = 640
    rl_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), None, "cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=args.fps, macro_block_size=1) as w:
        for step in range(args.steps):
            with torch.no_grad():
                act = policy(obs)
            obs, _, _, _ = wrapped.step(act)
            frame = env.render()
            if frame is None:
                raise RuntimeError("env.render() returned None")
            w.append_data(np.asarray(frame))
            if step % 100 == 0:
                print(f"  frame {step}/{args.steps}")
    env.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
