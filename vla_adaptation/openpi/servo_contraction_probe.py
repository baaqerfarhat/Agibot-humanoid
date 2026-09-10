"""Finite-difference checks of ALOHA's driven position servo, without policy inference.

Replays recorded healthy commands from identical simulator states. These are local,
finite-horizon measurements, not a proof of global contraction or VLA-loop stability.
Velocity perturbations explicitly test whether joint position alone is a sufficient state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np


def probe_aloha(log_path, *, seed=2500, episodes=2, checkpoints=(0, 50, 100, 150),
                horizons=(1, 5, 10, 25), epsilon=1e-4):
    import gymnasium as gym
    import gym_aloha  # noqa: F401

    data = json.loads(pathlib.Path(log_path).read_text())
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array")
    dm = env.unwrapped._env
    task, physics = dm.task, dm.physics
    # Observation rendering is irrelevant to replay and is deliberately disabled.
    task.get_observation = lambda p: dict(qpos=task.get_qpos(p), qvel=task.get_qvel(p),
                                         images={"top": np.zeros((1, 1, 3), np.uint8)})
    joints = np.arange(6)
    records, trajectories = [], []

    def capture():
        return dict(state=physics.get_state().copy(), time=float(physics.data.time),
                    ctrl=physics.data.ctrl.copy(), warm=physics.data.qacc_warmstart.copy(),
                    step_count=dm._step_count, reset_next=dm._reset_next_step)

    def restore(snap, axis=None, amount=0.0):
        physics.set_state(snap["state"])
        physics.data.time = snap["time"]
        physics.data.ctrl[:] = snap["ctrl"]
        if axis is not None:
            target = physics.data.qpos if axis < 6 else physics.data.qvel
            target[int(joints[axis % 6])] += amount
        physics.forward()
        physics.data.qacc_warmstart[:] = snap["warm"]
        dm._step_count = snap["step_count"]
        dm._reset_next_step = snap["reset_next"]

    def measured():
        return np.r_[physics.data.qpos[joints], physics.data.qvel[joints]]

    def replay(snap, commands, axis=None, amount=0.0):
        restore(snap, axis, amount)
        out = []
        for command in commands:
            dm.step(np.asarray(command, float))
            out.append(measured().copy())
        return np.asarray(out)

    try:
        for episode, donor in enumerate(data[:episodes]):
            commands = np.asarray(donor["u"], float)
            env.reset(seed=seed + episode)
            q_before, q_after, velocities, contacts = [], [], [], []
            for t, command in enumerate(commands):
                if t in checkpoints and t + max(horizons) <= len(commands):
                    snap = capture()
                    window = commands[t:t + max(horizons)]
                    baseline = replay(snap, window)
                    repeat = replay(snap, window)
                    repeat_error = float(np.max(np.abs(baseline - repeat)))
                    if repeat_error > 1e-9:
                        raise RuntimeError("simulator restoration is not deterministic")
                    jacobians = np.empty((len(window), 12, 12))
                    for axis in range(12):
                        plus = replay(snap, window, axis, epsilon)
                        minus = replay(snap, window, axis, -epsilon)
                        jacobians[:, :, axis] = (plus - minus) / (2 * epsilon)
                    for horizon in horizons:
                        jac = jacobians[horizon - 1]
                        records.append(dict(episode=episode, init=seed + episode,
                            command_donor_episode=episode, checkpoint=t, horizon=horizon,
                            time_seconds=0.02 * horizon, epsilon=epsilon,
                            restore_repeat_max_abs=repeat_error, jacobian=jac.tolist(),
                            position_from_position_norm=float(np.linalg.norm(jac[:6, :6], 2)),
                            position_from_velocity_norm=float(np.linalg.norm(jac[:6, 6:], 2)),
                            full_state_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(jac)))),
                            full_state_euclidean_norm=float(np.linalg.norm(jac, 2))))
                    restore(snap)
                q_before.append(task.get_qpos(physics).copy())
                velocities.append(task.get_qvel(physics).copy())
                dm.step(command)
                q_after.append(task.get_qpos(physics).copy())
                contacts.append(int(physics.data.ncon))
            trajectories.append(dict(task="gym_aloha/AlohaTransferCube-v0", init=seed + episode,
                command_donor_episode=episode, u=commands.tolist(), q_before=np.asarray(q_before).tolist(),
                q=np.asarray(q_after).tolist(), qvel_before=np.asarray(velocities).tolist(), contacts=contacts))
            print("completed ALOHA replay", episode, "probes", len(records), flush=True)
    finally:
        env.close()
    return dict(schema_version=1, experiment="driven_servo_finite_difference",
        source_log=str(log_path), source_sha256=hashlib.sha256(pathlib.Path(log_path).read_bytes()).hexdigest(),
        seed=seed, dt=0.02, tracked_joints=joints.tolist(), records=records, episodes=trajectories,
        limits=["Recorded commands are replayed; the VLA is not in the feedback loop.",
                "Position derivatives hold initial velocity fixed; velocity derivatives expose hidden state.",
                "Local finite-horizon Jacobians do not certify global contraction.",
                "A spectral radius below one alone is not Euclidean contraction."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=2500)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--checkpoints", default="0,50,100,150")
    parser.add_argument("--horizons", default="1,5,10,25")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    args = parser.parse_args()
    if args.episodes < 1 or not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("episodes and epsilon must be positive")
    result = probe_aloha(args.log, seed=args.seed, episodes=args.episodes,
        checkpoints=tuple(map(int, args.checkpoints.split(","))),
        horizons=tuple(map(int, args.horizons.split(","))), epsilon=args.epsilon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
