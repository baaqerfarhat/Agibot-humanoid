"""Replay recorded LIBERO commands and probe joint-torque correction authority.

This diagnostic never contacts a policy server. At selected checkpoints it restores full
physics/controller/gripper state for healthy, torque-faulted, opposing-torque and Cartesian
action perturbation replays. The local least-squares correction uses privileged healthy
responses: it tests authority, and is NOT a deployable estimator or a global certificate.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import pathlib
import random

import numpy as np

from joint_diagnostics import collect_local_probe, analyze_local_probe, _json


class Snapshot:
    """Mutable state used by the robosuite 1.4 control/physics path, plus RNGs."""
    DATA = ("ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart", "mocap_pos", "mocap_quat", "userdata")
    ENV = ("timestep", "cur_time", "done", "_obs_cache", "_observables")

    def __init__(self, rs):
        self.rs, self.sim = rs, rs.sim
        self.memo = {id(o): o for o in (rs, rs.sim, rs.sim.model, rs.sim.data)}
        self.state = copy.deepcopy(self.sim.get_state())
        self.arrays = {k: np.asarray(getattr(self.sim.data, k)).copy()
                       for k in self.DATA if hasattr(self.sim.data, k)}
        self.env = {k: copy.deepcopy(getattr(rs, k), self.memo.copy()) for k in self.ENV if hasattr(rs, k)}
        self.robots = []
        for robot in rs.robots:
            memo = self.memo.copy(); memo[id(robot)] = robot
            self.robots.append(dict(
                controller=copy.deepcopy(robot.controller.__dict__, memo),
                buffers={k: copy.deepcopy(v, memo.copy()) for k, v in vars(robot).items()
                         if k.startswith("recent_") or k == "torques"},
                gripper=robot.gripper.current_action.copy()))
        self.objects = [(o, copy.deepcopy(o.__dict__, self.memo.copy()))
                        for o in getattr(rs, "tracking_object_states_change", [])]
        self.numpy_rng, self.python_rng = np.random.get_state(), random.getstate()
        self.identity = hashlib.sha256(self.state.flatten().tobytes()).hexdigest()

    def restore(self):
        self.sim.set_state(copy.deepcopy(self.state))
        for name, value in self.arrays.items():
            getattr(self.sim.data, name)[:] = value
        self.sim.forward()
        # Forward can overwrite acceleration warm starts; restore these last too.
        for name, value in self.arrays.items():
            getattr(self.sim.data, name)[:] = value
        for name, value in self.env.items():
            setattr(self.rs, name, copy.deepcopy(value, self.memo.copy()))
        for robot, stored in zip(self.rs.robots, self.robots):
            memo = self.memo.copy(); memo[id(robot)] = robot
            robot.controller.__dict__.clear()
            robot.controller.__dict__.update(copy.deepcopy(stored["controller"], memo))
            for name, value in stored["buffers"].items():
                setattr(robot, name, copy.deepcopy(value, memo.copy()))
            robot.gripper.current_action = stored["gripper"].copy()
        for obj, stored in self.objects:
            obj.__dict__.clear(); obj.__dict__.update(copy.deepcopy(stored, self.memo.copy()))
        np.random.set_state(self.numpy_rng); random.setstate(self.python_rng)


def pose(rs):
    from robosuite.utils.transform_utils import convert_quat
    sid = rs.robots[0].eef_site_id
    return (rs.sim.data.site_xpos[sid].copy(),
            convert_quat(rs.sim.data.get_body_xquat(rs.robots[0].robot_model.eef_name), to="xyzw").copy())


def observed_pose(rs):
    prefix = rs.robots[0].robot_model.naming_prefix
    return (np.asarray(rs._observables[prefix+"eef_pos"].obs).copy(),
            np.asarray(rs._observables[prefix+"eef_quat"].obs).copy())


def angle(q1, q2):
    return float(2 * np.arccos(np.clip(abs(np.dot(q1, q2)) / np.linalg.norm(q1) / np.linalg.norm(q2), 0, 1)))


def advance(rs, action, dof, fault, opposite=0.0):
    """Same controller/substep path as MujocoEnv.step, without image observation work."""
    rs.timestep += 1
    rs.sim.data.qfrc_applied[dof] = fault
    rs.sim.data.qfrc_applied[dof] += opposite
    policy_step = True
    for _ in range(int(rs.control_timestep / rs.model_timestep)):
        rs.sim.forward()
        rs._pre_action(action, policy_step)
        rs.sim.step()
        # Preserve proprioceptive sampling times. Camera observables are disabled below.
        rs._update_observables()
        policy_step = False
    rs.cur_time += rs.control_timestep
    rs._post_action(action)


def parity(rs, action, dof, torque, tolerance):
    """Compare full env.step and observation-free stepping from one full snapshot."""
    snapshot = Snapshot(rs)
    rs.sim.data.qfrc_applied[dof] = torque
    rs.step(action)
    full_qpos = rs.sim.data.qpos.copy(); full_qvel = rs.sim.data.qvel.copy()
    full_ctrl = rs.sim.data.ctrl.copy(); full_pose = pose(rs)
    snapshot.restore()
    advance(rs, action, dof, torque)
    got = dict(qpos_max=float(np.max(np.abs(rs.sim.data.qpos - full_qpos))),
               qvel_max=float(np.max(np.abs(rs.sim.data.qvel - full_qvel))),
               ctrl_max=float(np.max(np.abs(rs.sim.data.ctrl - full_ctrl))),
               position_norm=float(np.linalg.norm(pose(rs)[0] - full_pose[0])),
               quaternion_angle=angle(pose(rs)[1], full_pose[1]))
    snapshot.restore()
    if max(got[k] for k in ("qpos_max", "qvel_max", "ctrl_max", "position_norm")) > tolerance:
        raise RuntimeError(f"low-level stepping failed full-step parity: {got}")
    return got


def read_telemetry(path, source_arm):
    header, groups = None, collections.defaultdict(list)
    with pathlib.Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "header":
                header = row
            elif row.get("type") == "step" and row.get("arm") == source_arm:
                groups[(int(row["task"]), int(row["init"]))].append(row)
    if header is None or not groups:
        raise ValueError("telemetry header and requested source arm required")
    return header, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--telemetry", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--cases", default="0:45,6:45,0:46", help="task:init pairs")
    ap.add_argument("--checkpoints", default="0,20,60", help="zero-based rollout indices")
    ap.add_argument("--horizons", default="1,5")
    ap.add_argument("--source-arm", default="frozen_faulted")
    ap.add_argument("--suite", default=None, help="default: recorded suite; must match replay")
    ap.add_argument("--step", type=float, default=.02)
    ap.add_argument("--bound", type=float, default=.3)
    ap.add_argument("--tolerance", type=float, default=1e-8)
    ap.add_argument("--recorded-tolerance", type=float, default=1e-5)
    args = ap.parse_args()
    header, groups = read_telemetry(args.telemetry, args.source_arm)
    recorded = header["args"]
    kind, joint, torque = recorded["joint_fault"].split(":")
    joint, torque = int(joint), float(torque)
    if kind != "torque" or joint not in range(7):
        ap.error("a torque fault on arm joint 0..6 is required")
    suite_name = args.suite or recorded["suite"]
    if suite_name != recorded["suite"]:
        ap.error("suite must match the recorded command replay")
    cases = [tuple(map(int, x.split(":"))) for x in args.cases.split(",")]
    checkpoints = sorted(set(map(int, args.checkpoints.split(","))))
    horizons = sorted(set(map(int, args.horizons.split(","))))
    if min(checkpoints) < 0 or min(horizons) < 1:
        ap.error("nonnegative checkpoints and positive horizons required")
    # These imports happen only for an explicitly requested simulator experiment.
    import main as lm
    from libero.libero import benchmark
    from adaptive_law import estimator_step
    from so3 import rot_delta
    suite = benchmark.get_benchmark_dict()[suite_name]()
    OUT = np.asarray(header["constants"]["OUT"])
    W, inverse = np.asarray(header["config"]["W"]), np.asarray(header["config"]["M_inv"])
    M = np.asarray(header["config"]["M"])
    bias = header["config"].get("bias")
    bias = None if bias is None else np.asarray(bias)
    mask = np.asarray(header["config"]["mask"])
    kfir = int(header["constants"]["K_FIR"])
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, args=vars(args), recorded_args=recorded,
        protocol="fixed future recorded commands, constant correction over each horizon, no policy queries",
        metric="endpoint translation/rotation increment divided by recorded OUT", cases=[], probes=[])
    for task_id, init in cases:
        rows = groups[(task_id, init)]
        warmup = [r for r in rows if r["phase"] == "warmup"]
        rollout = [r for r in rows if r["phase"] == "rollout"]
        if not rollout or max(checkpoints) + max(horizons) > len(rollout):
            raise ValueError(f"not enough recorded steps in {task_id}:{init}")
        env, _ = lm._get_libero_env(suite.get_task(task_id), lm.LIBERO_ENV_RESOLUTION, 7)
        try:
            env.reset(); env.set_init_state(suite.get_task_init_states(task_id)[init])
            rs = env.env
            for name, observable in rs._observables.items():
                if any(token in name for token in ("image", "depth", "segmentation")):
                    observable.set_enabled(False)
            dof = rs.sim.model.get_joint_qvel_addr(rs.robots[0].robot_model.joints[joint])
            rs.sim.data.qfrc_applied[:] = 0
            reference = parity(rs, np.asarray(warmup[0]["command"]), dof, 0., args.tolerance)
            for row in warmup:
                advance(rs, np.asarray(row["command"]), dof, 0.)
            history = collections.deque([np.zeros(6)] * (kfir + 1), maxlen=kfir + 1)
            estimate = np.zeros(6)
            errors = []
            for index, row in enumerate(rollout):
                if index > max(checkpoints):
                    break
                if index in checkpoints:
                    state = Snapshot(rs)
                    check = parity(rs, np.asarray(row["command"]), dof, torque, args.tolerance)
                    state.restore()
                    x0, q0 = pose(rs)
                    for horizon in horizons:
                        future = [np.asarray(r["command"]) for r in rollout[index:index+horizon]]
                        def evaluate(delta, fault, opposite):
                            state.restore()
                            torques, raw_torques, positions = [], [], []
                            for command in future:
                                a = command.copy(); a[:6] += delta
                                advance(rs, a, dof, fault, opposite)
                                torques.append(rs.robots[0].torques.copy())
                                raw_torques.append(rs.robots[0].controller.torques.copy())
                                positions.append(pose(rs)[0])
                            x1, q1 = pose(rs)
                            response = np.r_[x1-x0, rot_delta(q0, q1)] / OUT
                            return dict(response=response, snapshot_id=state.identity,
                                        endpoint_position=x1, endpoint_quaternion=q1,
                                        clipped_torques=torques, raw_torques=raw_torques,
                                        positions=positions)
                        candidate = -estimate * mask
                        metadata = dict(joint=joint, torque=torque, suite=suite_name,
                            task=task_id, init=init, checkpoint=index, horizon=horizon,
                            source_arm=args.source_arm, endpoint_scale=OUT,
                            candidate="legacy estimate recomputed on the same uncorrected replay" if args.source_arm == "frozen_faulted"
                                      else "estimate recomputed on the recorded corrected replay",
                            full_step_parity=check)
                        probe = collect_local_probe(evaluate, step=args.step, torque=torque,
                            bound=args.bound, candidate=candidate, metadata=metadata)
                        analysis = analyze_local_probe(probe)
                        # Also test the translation-only local optimum in the actual nonlinear plant.
                        tc = np.zeros(6); tc[:3] = analysis["subsets"]["translation"]["bounded_correction"]
                        probe["translation_oracle"] = dict(action=tc, evaluation=evaluate(tc, torque, 0))
                        # Preserve the all-six estimate candidate separately from the applied mask.
                        probe["all_estimate_candidate"] = dict(action=-estimate, evaluation=evaluate(-estimate, torque, 0))
                        healthy = np.asarray(probe["healthy"]["response"])
                        dnorm = np.linalg.norm(np.asarray(probe["faulted"]["response"])-healthy)
                        for key in ("translation_oracle", "all_estimate_candidate"):
                            remaining = np.asarray(probe[key]["evaluation"]["response"])-healthy
                            analysis["replays"][key] = dict(action=probe[key]["action"],
                                remaining_norm=float(np.linalg.norm(remaining)),
                                error_ratio=float(np.linalg.norm(remaining)/dnorm) if dnorm else None)
                        if analysis["repeated_center_difference"] > args.tolerance or analysis["opposing_torque_error"] > args.tolerance:
                            raise RuntimeError("snapshot repeat or opposing-torque sanity failed")
                        stem = f"j{joint}_t{task_id}_i{init}_s{index}_h{horizon}"
                        (args.out / (stem+"_raw.json")).write_text(json.dumps(probe, default=_json, allow_nan=False))
                        (args.out / (stem+"_summary.json")).write_text(json.dumps(analysis, default=_json, indent=2, allow_nan=False))
                        manifest["probes"].append(analysis)
                        print(stem, "dist", round(dnorm, 6), "error ratios",
                              {k:round(v["error_ratio"], 4) for k,v in analysis["replays"].items()}, flush=True)
                    state.restore()
                before_x, before_q = observed_pose(rs)
                advance(rs, np.asarray(row["command"]), dof, torque)
                after_x, after_q = observed_pose(rs)
                errors.append(dict(step=index, position=float(np.linalg.norm(after_x-np.asarray(row["position"]))),
                                   quaternion=angle(after_q,np.asarray(row["quaternion"]))))
                if errors[-1]["position"] > args.recorded_tolerance:
                    raise RuntimeError(f"recorded replay diverged: {errors[-1]}")
                history.appendleft(np.asarray(row["nominal_command"])[:6])
                H = np.asarray(history)
                prediction = (W[:,:-1]@H.reshape(-1)+W[:,-1]) if W.shape[1]>kfir+2 else np.array(
                    [W[j,:kfir+1]@H[:,j]+W[j,-1] for j in range(6)])
                measured = np.r_[after_x-before_x,rot_delta(before_q,after_q)]/OUT
                estimate,_ = estimator_step(estimate,measured-prediction,inverse,
                    gamma=recorded["gamma"],dead=recorded["dead"],norm_r=recorded["norm_r"],
                    clip=recorded["clip"],bias=bias,mask=mask,M=M)
            manifest["cases"].append(dict(task=task_id,init=init,warmup_full_step_parity=reference,
                                          recorded_replay_errors=errors))
            (args.out/"manifest.json").write_text(json.dumps(manifest,default=_json,indent=2,allow_nan=False))
        finally:
            env.close()
    print("complete", len(manifest["probes"]), "probes", flush=True)


if __name__ == "__main__":
    main()
