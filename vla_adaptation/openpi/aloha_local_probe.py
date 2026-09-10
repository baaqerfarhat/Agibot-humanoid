"""Same-state physical torque authority probes on all twelve ALOHA arm joints.

Stored policy commands are replayed in fresh seeded scenes, without policy queries.
Healthy responses and torque truth are privileged diagnostic oracles, not adaptation
inputs. The full 12-arm-coordinate endpoint is measured in radians, excluding grippers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib

import numpy as np

from aloha_joint_fault import joint_metadata

ARM_COORDINATES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


class Snapshot:
    DATA = ("ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart",
            "mocap_pos", "mocap_quat", "userdata")

    def __init__(self, dm):
        self.dm, self.physics = dm, dm.physics
        self.state = self.physics.get_state().copy()
        self.time = float(self.physics.data.time)
        self.arrays = {k: np.asarray(getattr(self.physics.data, k)).copy()
                       for k in self.DATA if hasattr(self.physics.data, k)}
        self.steps, self.reset_next = dm._step_count, dm._reset_next_step
        self.random = copy.deepcopy(dm.task.random.get_state())
        self.identity = hashlib.sha256(self.state.tobytes()).hexdigest()

    def restore(self):
        self.physics.set_state(self.state)
        self.physics.data.time = self.time
        for k, v in self.arrays.items():
            getattr(self.physics.data, k)[:] = v
        self.physics.forward()
        for k, v in self.arrays.items():
            getattr(self.physics.data, k)[:] = v
        self.dm._step_count, self.dm._reset_next_step = self.steps, self.reset_next
        self.dm.task.random.set_state(copy.deepcopy(self.random))


def bounded_span(B, disturbance, bound):
    from scipy.optimize import lsq_linear
    unconstrained = np.linalg.lstsq(B, -disturbance, rcond=None)[0]
    solution = unconstrained
    if np.any(np.abs(solution) > bound):
        fit = lsq_linear(B, -disturbance, bounds=(-bound, bound), tol=1e-12,
                         lsmr_tol=1e-12, max_iter=1000)
        if not fit.success:
            raise RuntimeError(f"bounded authority solve failed: {fit.message}")
        solution = fit.x
    norm = float(np.linalg.norm(disturbance))
    remaining = float(np.linalg.norm(disturbance + B @ solution))
    return dict(rank=int(np.linalg.matrix_rank(B)), singular_values=np.linalg.svd(B, compute_uv=False),
                correction=solution, bounded_remaining_norm=remaining,
                bounded_remaining_fraction=remaining / norm if norm else None)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, default=json_value, indent=2, allow_nan=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commands", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--task", default="gym_aloha/AlohaTransferCube-v0")
    ap.add_argument("--episodes", default="0,1")
    ap.add_argument("--seed", type=int, default=2500)
    ap.add_argument("--joints", default=",".join(map(str, range(12))))
    ap.add_argument("--checkpoints", default="0,60,140")
    ap.add_argument("--horizons", default="1,10")
    ap.add_argument("--equivalent-target", type=float, default=.02)
    ap.add_argument("--step", type=float, default=.005)
    ap.add_argument("--bound", type=float, default=.1)
    ap.add_argument("--tolerance", type=float, default=1e-10)
    args = ap.parse_args()
    episodes, joints, checkpoints, horizons = [sorted(set(map(int, value.split(","))))
        for value in (args.episodes, args.joints, args.checkpoints, args.horizons)]
    if min(checkpoints) < 0 or min(horizons) < 1 or not set(joints) <= set(range(12)):
        ap.error("nonnegative checkpoints, positive horizons and joints0..11 required")
    if args.step <= 0 or args.bound <= 0:
        ap.error("positive finite-difference step and correction bound required")
    stored = json.loads(args.commands.read_text())
    if isinstance(stored, dict):
        stored = stored["episodes"]
    import gymnasium as gym
    import gym_aloha  # noqa: F401
    env = gym.make(args.task, obs_type="pixels_agent_pos", render_mode="rgb_array")
    dm = env.unwrapped._env
    physics = dm.physics
    original_observation = dm.task.get_observation
    # Pure read-only observation work is removed; stepping remains dm_control.step.
    def state_observation(physics):
        return dict(qpos=dm.task.get_qpos(physics), qvel=dm.task.get_qvel(physics),
                    env_state=dm.task.get_env_state(physics),
                    images={"top": np.zeros((1, 1, 3), dtype=np.uint8)})
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, args=vars(args), probes=[], validation=[],
        protocol="same full DM state; fixed future recorded commands; torque constant over horizon",
        response="all twelve physical arm joint positions in radians; grippers excluded",
        notes=["Local privileged authority diagnostic; not policy success or global contraction.",
               "Healthy branch removes current torque from an already faulted checkpoint.",
               "Magnitude kp*.02 is an equivalent static position-target displacement before force limits."])
    try:
        for episode in episodes:
            commands = np.asarray(stored[episode]["u"], float)
            if max(checkpoints) + max(horizons) > len(commands):
                raise ValueError("recorded episode is shorter than requested probe endpoint")
            for joint in joints:
                dm.task.get_observation = state_observation
                env.reset(seed=args.seed + episode)
                physics.data.qfrc_applied[:] = 0.
                meta = joint_metadata(physics, joint)
                torque = meta["command_to_torque_gain"] * args.equivalent_target
                dof = meta["dof"]
                for step_index in range(max(checkpoints) + 1):
                    if step_index in checkpoints:
                        snapshot = Snapshot(dm)
                        # Compare the complete original observation path once per checkpoint.
                        dm.task.get_observation = original_observation
                        physics.data.qfrc_applied[dof] = torque
                        dm.step(commands[step_index])
                        full_state = physics.get_state().copy()
                        full_ctrl = physics.data.ctrl.copy()
                        snapshot.restore()
                        dm.task.get_observation = state_observation
                        physics.data.qfrc_applied[dof] = torque
                        dm.step(commands[step_index])
                        parity = dict(state_max=float(np.max(np.abs(full_state-physics.get_state()))),
                                      ctrl_max=float(np.max(np.abs(full_ctrl-physics.data.ctrl))))
                        snapshot.restore()
                        if max(parity.values()) > args.tolerance:
                            raise RuntimeError(f"state-only observation path failed parity: {parity}")
                        manifest["validation"].append(dict(episode=episode, joint=joint,
                                                           checkpoint=step_index, parity=parity))
                        for horizon in horizons:
                            future = commands[step_index:step_index+horizon]
                            def evaluate(delta, force, opposite=0.):
                                snapshot.restore()
                                actuator_forces = []
                                for command in future:
                                    action = command.copy()
                                    action[ARM_COORDINATES] += delta
                                    physics.data.qfrc_applied[dof] = force + opposite
                                    dm.step(action)
                                    actuator_forces.append(physics.data.actuator_force.copy())
                                return dict(response=dm.task.get_qpos(physics)[ARM_COORDINATES],
                                            full_state=physics.get_state().copy(),
                                            actuator_force=actuator_forces,
                                            snapshot_id=snapshot.identity)
                            zero = np.zeros(12)
                            raw = dict(metadata=dict(joint=joint, episode=episode, seed=args.seed+episode,
                                checkpoint=step_index, horizon=horizon, joint_metadata=meta, torque=torque),
                                step=args.step, bound=args.bound, healthy=evaluate(zero, 0.),
                                faulted=evaluate(zero, torque),
                                torque_oracle=evaluate(zero, torque, -torque), plus=[], minus=[])
                            for coordinate in range(12):
                                delta = zero.copy(); delta[coordinate] = args.step
                                raw["plus"].append(evaluate(delta, torque))
                                raw["minus"].append(evaluate(-delta, torque))
                            raw["repeat"] = evaluate(zero, torque)
                            healthy = raw["healthy"]["response"]
                            center = raw["faulted"]["response"]
                            disturbance = center - healthy
                            B = np.column_stack([(p["response"]-m["response"])/(2*args.step)
                                                  for p, m in zip(raw["plus"], raw["minus"])])
                            repeat = float(np.max(np.abs(raw["repeat"]["full_state"]-raw["faulted"]["full_state"])))
                            cancel = float(np.max(np.abs(raw["torque_oracle"]["full_state"]-raw["healthy"]["full_state"])))
                            if max(repeat, cancel) > args.tolerance:
                                raise RuntimeError(f"same-state restore/cancel failed: {repeat}, {cancel}")
                            summary = dict(metadata=raw["metadata"], disturbance=disturbance,
                                disturbance_norm=float(np.linalg.norm(disturbance)), response_map=B,
                                repeated_full_state_max=repeat, torque_cancel_full_state_max=cancel,
                                observation_parity=parity, authority={}, nonlinear_replays={})
                            subsets = {"left6": np.arange(6), "affected_side6": np.arange(6)+(joint//6)*6,
                                       "all12": np.arange(12)}
                            raw["replays"] = {}
                            for name, indices in subsets.items():
                                authority = bounded_span(B[:, indices], disturbance, args.bound)
                                summary["authority"][name] = authority
                                delta = zero.copy(); delta[indices] = authority["correction"]
                                replay = evaluate(delta, torque)
                                remaining = float(np.linalg.norm(replay["response"]-healthy))
                                raw["replays"][name] = dict(action=delta, evaluation=replay)
                                summary["nonlinear_replays"][name] = dict(remaining_norm=remaining,
                                    error_ratio=remaining/summary["disturbance_norm"] if summary["disturbance_norm"] else None)
                            # Analytic privileged servo check: target shift -tau/kp.
                            delta = zero.copy(); delta[joint] = -args.equivalent_target
                            replay = evaluate(delta, torque)
                            remaining = float(np.linalg.norm(replay["response"]-healthy))
                            raw["replays"]["static_target_oracle"] = dict(action=delta, evaluation=replay)
                            summary["nonlinear_replays"]["static_target_oracle"] = dict(remaining_norm=remaining,
                                error_ratio=remaining/summary["disturbance_norm"] if summary["disturbance_norm"] else None)
                            stem = f"seed{args.seed+episode}_j{joint}_s{step_index}_h{horizon}"
                            save(args.out/(stem+"_raw.json"), raw)
                            save(args.out/(stem+"_summary.json"), summary)
                            manifest["probes"].append(summary)
                        snapshot.restore()
                    physics.data.qfrc_applied[dof] = torque
                    dm.step(commands[step_index])
                save(args.out/"manifest.json", manifest)
                print(json.dumps(dict(episode=episode, joint=joint, torque=torque,
                                      probes=len(manifest["probes"]))), flush=True)
    finally:
        physics.data.qfrc_applied[:] = 0.
        dm.task.get_observation = original_observation
        env.close()


if __name__ == "__main__":
    main()
