"""Step the box-pickup env with no policy and report what the managers see.

Checks the things that silently break a training run: observation width, whether the
box lands on the floor instead of in it or through it, whether the three contact
sensors fire on the events they are named after, and whether every reward term
produces a finite number on step one.

    cd ~/baaqer_ws/mjlab && uv run python <this>
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-BoxPickup-Flat-X2")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

    obs, _ = env.reset()
    actor, critic = obs["actor"], obs["critic"]
    print("=" * 70)
    print("SHAPES")
    print("=" * 70)
    print(f"actor obs  {tuple(actor.shape)}   terms "
          f"{env.observation_manager.active_terms['actor']}")
    print(f"critic obs {tuple(critic.shape)}")
    print(f"action dim {env.action_manager.total_action_dim}")
    print(f"control    {1 / env.step_dt:.0f} Hz   sim dt {env.physics_dt}   "
          f"decimation {cfg.decimation}")
    print(f"episode    {cfg.episode_length_s} s = {env.max_episode_length} steps")

    cmd = env.command_manager.get_term("motion")
    robot, box = env.scene["robot"], env.scene["box"]
    origin_z = env.scene.env_origins[:, 2]

    print()
    print("=" * 70)
    print("RESET STATE (should match reference frame 0)")
    print("=" * 70)
    print(f"box z            {(box.data.root_link_pos_w[:, 2] - origin_z).tolist()}")
    print(f"reference box z  {(cmd.object_pos_w[:, 2] - origin_z).tolist()}")
    print(f"pelvis z         {(robot.data.root_link_pos_w[:, 2] - origin_z).tolist()}")
    print(f"reference pelvis {(cmd.body_pos_w[:, 0, 2] - origin_z).tolist()}")
    jerr = torch.abs(robot.data.joint_pos - cmd.joint_pos).max().item()
    print(f"joint pos error vs reference: {jerr:.6f} rad")

    print()
    print("=" * 70)
    print(f"ROLLOUT: {args.steps} steps of zero action (PD holds the default pose)")
    print("=" * 70)
    zero = torch.zeros(args.envs, env.action_manager.total_action_dim,
                       device=args.device)
    feet = env.scene.sensors["feet_ground_contact"]
    hfloor = env.scene.sensors["hand_ground_contact"]
    hbox = env.scene.sensors["hand_box_contact"]
    print(f"{'step':>5} {'boxZ':>7} {'pelvZ':>7} {'feetN':>7} {'handFloorN':>11} "
          f"{'handBoxN':>9} {'reward':>8} {'done':>5}")
    for i in range(args.steps):
        obs, rew, term, trunc, _ = env.step(zero)
        if i % 25 == 0 or i == args.steps - 1:
            fN = torch.norm(feet.data.force, dim=-1).sum(-1).mean().item()
            gN = torch.norm(hfloor.data.force, dim=-1).sum(-1).mean().item()
            bN = torch.norm(hbox.data.force, dim=-1).sum(-1).mean().item()
            print(f"{i:5d} {(box.data.root_link_pos_w[:, 2] - origin_z).mean():7.3f} "
                  f"{(robot.data.root_link_pos_w[:, 2] - origin_z).mean():7.3f} "
                  f"{fN:7.1f} {gN:11.1f} {bN:9.1f} {rew.mean():8.3f} "
                  f"{int((term | trunc).sum()):5d}")

    print()
    print("=" * 70)
    print("REWARD TERMS (mean over envs, one step)")
    print("=" * 70)
    env.reset()
    env.step(zero)
    for name in env.reward_manager.active_terms:
        term = env.reward_manager.get_term_cfg(name)
        val = term.func(env, **term.params)
        flag = "  NON-FINITE" if not torch.isfinite(val).all() else ""
        print(f"  {name:<24} weight {term.weight:+8.4g}   value {val.mean().item():+9.4f}"
              f"{flag}")

    print()
    print("METRICS")
    for name in env.metrics_manager.active_terms:
        print(f"  {name}")
    env.close()


if __name__ == "__main__":
    main()
