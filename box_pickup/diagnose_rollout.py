"""Roll out a trained box-pickup checkpoint and log why the opening goes wrong.

grasp_fraction sits at exactly 0 after 11k iterations while the box gets knocked to
0.30 m, so the policy is hitting the box with something that is not a hand. This logs,
per step: the action magnitude the policy is asking for, how far the achieved joints are
from the reference, and every geom actually touching the box.

    cd ~/baaqer_ws/mjlab && uv run python <this> --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="Mjlab-BoxPickup-Flat-X2")
    ap.add_argument("--steps", type=int, default=140)
    args = ap.parse_args()

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = 1
    rl_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

    model = env.sim.mj_model
    gname = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"g{i}"
    box_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box/box_geom")

    from dataclasses import asdict

    from mjlab.rl import MjlabOnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), None, "cpu")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cpu")

    robot = env.scene["robot"]
    box = env.scene["box"]
    cmd = env.command_manager.get_term("motion")

    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    print(f"{'step':>5}{'|a|max':>9}{'|a|rms':>9}{'qerr':>8}{'boxZ':>8}{'pelvZ':>8}  box contacts")
    for step in range(args.steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = wrapped.step(act)

        f = max(int(cmd.time_steps[0]) - 1, 0)
        qerr = float((robot.data.joint_pos[0] - cmd.motion.joint_pos[f]).abs().max())
        a = act[0].numpy()

        d = env.sim.data
        ncon = int(np.asarray(d.nacon).reshape(-1)[0])
        pairs = np.asarray(d.contact.geom).reshape(-1, 2)
        dist = np.asarray(d.contact.dist).reshape(-1)
        hits = []
        for c in range(min(ncon, len(pairs))):
            if box_gid in pairs[c]:
                other = pairs[c][1] if pairs[c][0] == box_gid else pairs[c][0]
                hits.append(f"{gname(int(other))}({dist[c] * 1e3:+.0f}mm)")

        if step % 5 == 0 or hits:
            print(
                f"{step:>5}{np.abs(a).max():>9.2f}{np.sqrt((a**2).mean()):>9.2f}"
                f"{qerr:>8.2f}{box.data.root_link_pos_w[0, 2].item():>8.3f}"
                f"{robot.data.root_link_pos_w[0, 2].item():>8.3f}  "
                f"{', '.join(sorted(set(hits))) if hits else ''}"
            )

    env.close()


if __name__ == "__main__":
    main()
