#!/usr/bin/env python3
"""Offline policy sanity test (NO robot needed).

Feed the policy a PERFECT in-distribution observation (upright, exactly at the
default pose, zero velocity) and vary only the command. This isolates the policy
from all hardware/sim-to-real conditioning.

If the export + feeding are correct we expect:
  - command = [0,0,0]      -> |action| small  (robot should stand still)
  - command = [vx>0,...]   -> action CHANGES (intent to walk/lean forward)
If the action is large at cmd 0, or barely changes with the command, the bug is
in the policy export / observation layout (not the robot).
"""
import json
import numpy as np
from deploy_x2_walk import NumpyPolicy

POLICY = "policies/x2_policy.npz"


def build_ideal_obs(meta, command):
    nj = len(meta["joint_names"])
    parts = []
    for name in meta["observation_names"]:
        if name == "base_lin_vel":
            parts.append(np.zeros(3, np.float32))
        elif name == "base_ang_vel":
            parts.append(np.zeros(3, np.float32))
        elif name == "projected_gravity":
            parts.append(np.array([0.0, 0.0, -1.0], np.float32))  # perfectly upright
        elif name == "joint_pos":
            parts.append(np.zeros(nj, np.float32))                # exactly at default
        elif name == "joint_vel":
            parts.append(np.zeros(nj, np.float32))
        elif name == "actions":
            parts.append(np.zeros(meta["action_dim"], np.float32))
        elif name == "command":
            parts.append(np.asarray(command, np.float32))
        else:
            raise ValueError(name)
    return np.concatenate(parts).astype(np.float32)


def main():
    policy = NumpyPolicy(POLICY)
    meta = policy.meta
    print("obs terms:", meta["observation_names"], " dim", meta["obs_dim"])
    print("command_names:", meta.get("command_names"))
    print("=" * 70)
    base = None
    for command in ([0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [-0.5, 0, 0], [0, 0, 1.0]):
        obs = build_ideal_obs(meta, command)
        act = policy(obs).reshape(-1)
        if base is None:
            base = act.copy()
        delta = float(np.mean(np.abs(act - base)))
        print(f"cmd={str(command):<14} |action| mean={np.mean(np.abs(act)):.4f} "
              f"max={np.max(np.abs(act)):.4f}   mean|act-act(cmd0)|={delta:.4f}")
    print("=" * 70)
    print("Interpretation:")
    print("  - cmd=[0,0,0] |action| should be SMALL (stand still at default).")
    print("  - increasing vx should make mean|act-act(cmd0)| grow (command has effect).")


if __name__ == "__main__":
    main()
