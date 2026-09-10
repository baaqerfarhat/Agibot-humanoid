"""Interleaved ALOHA physical-joint torque screen, one fault/healthy cell per run.

All arms share healthy W/M/Q/R calibration and correct the twelve arm coordinates,
excluding both grippers. Torque truth is applied only by an environment wrapper.
No composite reference model is used or claimed qualified for all twelve joints.
Default ten-scenario cells are development screens, not confirmation outcomes.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from run_aloha_followup import ResetObserver
from run_joint_followup import atomic_json, arm_order, json_default, state_difference


ROOT = Path(__file__).resolve().parent.parent
CORRECTION_COORDINATES = list(range(6)) + list(range(7, 13))
COMMON = dict(gamma=.08, dead=.002, norm_r=.4, clip=.08, law="legacy",
              deadzone_mode="zero", corr=CORRECTION_COORDINATES,
              profile="step", prof_p=60., onset=0, damping=0., tracking_rate=0.)
ARM_CONFIGS = {
    "off": dict(adapt=False, baseline="none", norm_channels="all"),
    "legacy_allchannels": dict(adapt=True, baseline="none", norm_channels="all"),
    "legacy_correctedchannels": dict(adapt=True, baseline="none", norm_channels="corrected"),
    "kalman": dict(adapt=True, baseline="kalman", norm_channels="all"),
    "weighted": dict(adapt=True, baseline="weighted_dob", norm_channels="all"),
}
DEFAULT_ARMS = ("off", "legacy_allchannels", "legacy_correctedchannels", "kalman")
EXPECTED_GAINS = np.tile([800., 1600., 800., 10., 50., 20.], 2)


def shared_observer(artifact, law):
    """Validate observer matrices without treating a reference fit as qualification."""
    observer = {name: np.asarray(artifact["observer"][name], float) for name in ("W", "M", "Q", "R")}
    if law.NJ != 14:
        raise ValueError("the registered ALOHA arm/gripper mapping requires fourteen command channels")
    for name, value in observer.items():
        shape = (law.NJ, law.K_FIR+2) if name == "W" else (law.NJ, law.NJ)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"shared observer {name} must be finite with shape {shape}")
    if artifact["observer"].get("bias") is not None:
        raise ValueError("this screen requires the shared observer's declared zero bias correction")
    if (not np.isclose(artifact["observer"]["dt"], law.DT)
            or not np.isclose(artifact["observer"]["gamma"], COMMON["gamma"])):
        raise ValueError("shared observer dt/gamma differs from the registered settings")
    for name in ("Q", "R"):
        value = observer[name]
        if not np.allclose(value, value.T, rtol=1e-8, atol=1e-12):
            raise ValueError(f"shared observer {name} is not symmetric")
        if np.linalg.eigvalsh(value).min() < -1e-12:
            raise ValueError(f"shared observer {name} is not positive semidefinite")
    return observer


def robot_metadata(aloha, fault_module):
    physics = aloha.env.unwrapped._env.physics
    model = physics.model
    rows = []
    for joint in range(12):
        row = fault_module.joint_metadata(physics, joint)
        aid, jid = row["actuator"], row["joint_id"]
        row.update(control_limited=bool(model.actuator_ctrllimited[aid]),
                   control_range=np.asarray(model.actuator_ctrlrange[aid]).tolist(),
                   joint_limited=bool(model.jnt_limited[jid]),
                   joint_range=np.asarray(model.jnt_range[jid]).tolist(),
                   nominal_offset_rad=.02,
                   torque_nm=.02*row["command_to_torque_gain"])
        if row["command_coordinate"] != CORRECTION_COORDINATES[joint]:
            raise ValueError("live robot command mapping differs from the registered twelve-arm mapping")
        if not np.isclose(row["command_to_torque_gain"], EXPECTED_GAINS[joint], rtol=1e-9, atol=1e-9):
            raise ValueError("live position-actuator gain differs from the registered torque schedule")
        rows.append(row)
    return rows


class PhysicalTelemetry:
    """Read physical response after each step without changing commands or observations."""

    def __init__(self, env, metadata, callback):
        self.env, self.metadata, self.callback = env, metadata, callback
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, command):
        result = self.env.step(command)
        data = self.env.unwrapped._env.physics.data
        commands = np.asarray(command)
        stats = dict(force_limit=[], control_limit=[], requested_outside_control_range=[])
        for row in self.metadata:
            aid = row["actuator"]
            force_lo, force_hi = row["force_range"]
            ctrl_lo, ctrl_hi = row["control_range"]
            requested = commands[row["command_coordinate"]]
            stats["force_limit"].append(bool(row["force_limited"] and
                (data.actuator_force[aid] <= force_lo+1e-8 or data.actuator_force[aid] >= force_hi-1e-8)))
            stats["control_limit"].append(bool(row["control_limited"] and
                (data.ctrl[aid] <= ctrl_lo+1e-8 or data.ctrl[aid] >= ctrl_hi-1e-8)))
            stats["requested_outside_control_range"].append(bool(row["control_limited"] and
                (requested < ctrl_lo or requested > ctrl_hi)))
        self.callback(dict(type="physical_step", t=self.steps,
                           qpos=np.asarray(data.qpos).copy(), qvel=np.asarray(data.qvel).copy(),
                           qfrc_applied=np.asarray(data.qfrc_applied).copy(),
                           actuator_force=np.asarray(data.actuator_force).copy(),
                           actuator_control=np.asarray(data.ctrl).copy(), **stats))
        self.steps += 1
        return result


def persist(args, study):
    study["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    for name in args.arm_names:
        if name == "off":
            continue
        recorded = dict(study["args"], **ARM_CONFIGS[name], followup_arm=name,
                        corr_joints=",".join(map(str, CORRECTION_COORDINATES)),
                        static_corr=None, warm_start=False, freeze_after=None,
                        identify_episodes=None)
        view = dict(args=recorded, status=study["status"], study_id=study["study_id"],
                    shared_control_id=study["shared_control_id"],
                    shared_control_source="study.json:arms.off", source_hashes=study["source_hashes"],
                    pairing=study["pairing"], physical_fault=study["physical_fault"],
                    robot_joints=study["robot_joints"],
                    arms=dict(frozen_faulted=study["arms"]["off"], adaptive=study["arms"][name]))
        atomic_json(args.out_dir/f"{name}.json", view)
    atomic_json(args.out_dir/"study.json", study)


def execute(args, aloha, law, artifact, source_hashes, fault_module=None):
    if fault_module is None:
        import aloha_joint_fault as fault_module
    observer = shared_observer(artifact, law)
    metadata = robot_metadata(aloha, fault_module)
    physical_fault = None if args.healthy else dict(metadata[args.joint], fault="constant_external_torque")
    torque = 0.0 if args.healthy else physical_fault["torque_nm"]
    parsed = dict(vars(args), **COMMON)
    parsed.pop("arm_names", None)
    parsed.update(task=law.TASK, suite=law.TASK, dt=law.DT, horizon=law.HORIZON,
                  condition="healthy" if args.healthy else f"joint{args.joint}",
                  joint_fault=None if args.healthy else f"torque:{args.joint}:{torque}",
                  fault_vec=None, gain=None, sev=0.0, policy_rng_pinned=False,
                  reset_estimate=True, reset_covariance=True, composite_reference_used=False)
    parsed = json.loads(json.dumps(parsed, default=json_default))
    schedule = [dict(episode=i, task=0, init=i, seed=args.seed+i, arms=arm_order(args.arm_names, i))
                for i in range(args.episodes)]
    study_id = str(uuid.uuid4())
    study = dict(schema_version=1, study_id=study_id, stage=args.stage, status="planned",
                 args=parsed, source_hashes=source_hashes, arm_configs={n: ARM_CONFIGS[n] for n in args.arm_names},
                 shared_control_id=f"{study_id}:off", schedule=schedule, physical_fault=physical_fault,
                 robot_joints=metadata,
                 calibration_scope=dict(observer="shared fourteen-channel W/M/Q/R",
                    reference_used=False, source_reference_qualification=artifact.get("qualification"),
                    all_arm_reference="failed held-out model qualification; no composite arm in this screen"),
                 pairing=dict(mechanism="aloha_seeded_reset", valid_so_far=True, checked_states=0,
                    tolerance=args.state_tolerance, fields=["full_qpos", "full_qvel"],
                    snapshot_phase="after reset, before policy inference and first physical torque application",
                    policy_rng_pinned=False),
                 arms={name: dict(successes=0, n=0, per_ep=[], f_hat=[], traj=[],
                                 action_fault=[], diagnostics=[]) for name in args.arm_names})
    persist(args, study)
    previous_cap = law.MAX_STEPS
    law.MAX_STEPS = args.max_steps
    current = None
    try:
        with args.telemetry.open("x", encoding="utf-8") as telemetry:
            law.write_telemetry(telemetry, dict(type="header", experiment="aloha_physical_joint_followup",
                study_id=study_id, args=parsed, schedule=schedule, source_hashes=source_hashes,
                config=dict(common=COMMON, arm_configs=study["arm_configs"], observer=observer,
                            robot_joints=metadata, physical_fault=physical_fault),
                schema=dict(f_true="underlying episode logger's action-space fault; zero here, not torque truth",
                    physical_step="read-only simulator state and force/control limit indicators",
                    torque_scale="gain times 0.02 rad: nominal unsaturated servo offset, not equal torque or difficulty")))
            study["status"] = "running"
            persist(args, study)
            for batch in schedule:
                reference, reference_arm = None, None
                for order, name in enumerate(batch["arms"]):
                    current = dict(arm=name, episode=batch["episode"], seed=batch["seed"], task=0, init=batch["init"])
                    snapshot = {}
                    diagnostics = {key: np.zeros(12, dtype=int) for key in
                                   ("force_limit", "control_limit", "requested_outside_control_range")}
                    physical_steps = 0
                    def capture(captured):
                        nonlocal reference, reference_arm
                        if reference is None:
                            reference, reference_arm = captured, name
                        differences = state_difference(reference, captured)
                        valid = max(differences.values()) <= args.state_tolerance
                        snapshot.update(captured, comparison_to_arm=reference_arm,
                            exact_hash_match=captured["sha256"] == reference["sha256"],
                            pairing_valid=valid, **differences)
                        law.write_telemetry(telemetry, dict(type="scenario_start", **current, **snapshot))
                        study["pairing"]["checked_states"] += 1
                        if not valid:
                            study["pairing"].update(valid_so_far=False, mismatch=dict(**current, **snapshot))
                            raise RuntimeError(f"full-state pairing failed for {name} seed={batch['seed']}")
                    def physical_log(row):
                        nonlocal physical_steps
                        physical_steps += 1
                        for key in diagnostics:
                            diagnostics[key] += np.asarray(row[key], dtype=int)
                        law.write_telemetry(telemetry, dict(**current, **row))
                    context = contextlib.nullcontext() if args.healthy else fault_module.torque_fault(aloha, args.joint, torque)
                    with context:
                        injected_environment = aloha.env
                        aloha.env = PhysicalTelemetry(injected_environment, metadata, physical_log)
                        try:
                            success, estimate, trace = law.episode(ResetObserver(aloha, capture), batch["episode"],
                                W=observer["W"], M=observer["M"], M_inv=np.linalg.pinv(observer["M"]),
                                kf_q=observer["Q"], kf_r=observer["R"], fvec=np.zeros(law.NJ),
                                f_init=None, static_corr=None, freeze_after=None,
                                telemetry=telemetry, arm=name, **COMMON, **ARM_CONFIGS[name])
                        finally:
                            aloha.env = injected_environment
                    if not snapshot:
                        raise RuntimeError("episode did not call the required full-state reset observer")
                    estimates = np.asarray([row["f_hat"] for row in trace])
                    clipping = ((np.abs(estimates) >= COMMON["clip"]-1e-10).sum(axis=0).tolist()
                                if len(estimates) else [0]*law.NJ)
                    arm = study["arms"][name]
                    arm["n"] += 1
                    arm["successes"] += int(success)
                    arm["per_ep"].append(dict(task=0, init=batch["init"], actual_seed=batch["seed"],
                        episode=batch["episode"], order_index=order, ok=bool(success), initial_state=snapshot))
                    arm["f_hat"].append(np.asarray(estimate).tolist())
                    arm["traj"].append(estimates.tolist())
                    arm["action_fault"].append([row["f_true"] for row in trace])
                    arm["diagnostics"].append(dict(physical_steps=physical_steps,
                        estimate_clip_steps=clipping, **{key: value.tolist() for key, value in diagnostics.items()}))
                    telemetry.flush()
                    persist(args, study)
                    print(f"[{parsed['condition']}/{name}] seed={batch['seed']} success={int(success)} "
                          f"total={arm['successes']}/{arm['n']}", flush=True)
            if any(arm["n"] != args.episodes for arm in study["arms"].values()):
                raise RuntimeError("study ended with incomplete arms")
            study["status"] = "complete"
            persist(args, study)
    except BaseException as error:
        study.update(status="failed", failed_at=current, error=dict(type=type(error).__name__, message=str(error)))
        persist(args, study)
        raise
    finally:
        law.MAX_STEPS = previous_cap
    return study


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    condition = p.add_mutually_exclusive_group(required=True)
    condition.add_argument("--joint", type=int, choices=range(12), help="physical joint, left 0–5 then right 6–11")
    condition.add_argument("--healthy", action="store_true", help="run the one shared healthy cell")
    p.add_argument("--stage", choices=("dev", "confirm"), default="dev")
    p.add_argument("--seed", type=int, default=2900)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--reference-artifact", type=Path,
                   default=ROOT/"results/composite_followup/calibration/aloha_reference.json")
    p.add_argument("--prereg", type=Path, default=ROOT/"prereg_records/PREREG_COMPOSITE_JOINT_FOLLOWUP.md")
    p.add_argument("--selection-record", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--telemetry", type=Path)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--state-tolerance", type=float, default=1e-9)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.arm_names = [name.strip() for name in args.arms.split(",")]
    if (len(args.arm_names) != len(set(args.arm_names)) or "off" not in args.arm_names
            or any(name not in ARM_CONFIGS for name in args.arm_names)):
        p.error("--arms requires unique known arm names including the shared off arm")
    if args.seed < 0 or args.episodes < 1 or args.max_steps < 1:
        p.error("seed must be nonnegative; episodes and max-steps must be positive")
    if not np.isfinite(args.state_tolerance) or args.state_tolerance < 0:
        p.error("state tolerance must be finite and nonnegative")
    if args.stage == "confirm" and (args.selection_record is None or not args.selection_record.is_file()):
        p.error("confirmation requires an existing locked --selection-record")
    args.out_dir = args.out_dir.resolve()
    args.telemetry = (args.telemetry or args.out_dir/"telemetry.jsonl").resolve()
    targets = [args.out_dir/"study.json", args.out_dir/"run_config.json", args.out_dir/".run_claim", args.telemetry]
    targets.extend(args.out_dir/f"{name}.json" for name in args.arm_names if name != "off")
    sources = [Path(__file__), ROOT/"openpi/aloha_adapt.py", ROOT/"openpi/adaptive_law.py",
               ROOT/"openpi/aloha_joint_fault.py", ROOT/"openpi/run_aloha_followup.py",
               ROOT/"openpi/run_joint_followup.py", args.reference_artifact, args.prereg]
    if "weighted" in args.arm_names:
        sources.append(ROOT/"openpi/weighted_dob.py")
    if args.selection_record is not None:
        sources.append(args.selection_record)
    if any(path.exists() for path in targets):
        p.error("an output exists; use a fresh output directory and telemetry file")
    if len(set(targets)) != len(targets) or set(targets).intersection(path.resolve() for path in sources):
        p.error("output paths must be distinct from each other and source inputs")
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    artifact = json.loads(args.reference_artifact.read_text())
    import aloha_adapt as law
    shared_observer(artifact, law)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/".run_claim").open("x") as claim:
        claim.write(str(uuid.uuid4())+"\n")
    atomic_json(args.out_dir/"run_config.json", dict(args=vars(args), common=COMMON,
        arm_configs={name: ARM_CONFIGS[name] for name in args.arm_names}, source_hashes=hashes,
        locked_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    aloha = None
    try:
        aloha = law.Aloha(args.host, args.port, args.seed)
        execute(args, aloha, law, artifact, hashes)
    except BaseException as error:
        if not (args.out_dir/"study.json").exists():
            atomic_json(args.out_dir/"study.json", dict(status="failed", stage=args.stage,
                source_hashes=hashes, failed_at="environment setup", error=dict(type=type(error).__name__, message=str(error))))
        raise
    finally:
        if aloha is not None:
            aloha.env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
