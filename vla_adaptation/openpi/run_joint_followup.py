"""Interleaved, explicitly paired LIBERO joint-torque development/evaluation.

Reuses adaptive_law.run without changing the policy or its sampling RNG. A shared
off cohort is run once, not once per comparator. study.json is authoritative;
each candidate JSON is a standard frozen_faulted/adaptive view of those outcomes.
The views explicitly identify their shared controls and must not be pooled as
independent replications of the off arm.

This script runs rollouts when invoked. --selftest and --help launch no simulator.
It refuses to overwrite an existing study or telemetry file; incomplete studies
are retained and are not automatically resumed or scored as complete.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ARM_CONFIGS = {
    "off": dict(adapt=False, corr_dims=[0, 1, 2], baseline="none", correction_scale=1.0, freeze_after=None),
    "legacy_translation": dict(adapt=True, corr_dims=[0, 1, 2], baseline="none", correction_scale=1.0, freeze_after=None),
    "legacy_full": dict(adapt=True, corr_dims=[0, 1, 2, 3, 4, 5], baseline="none", correction_scale=1.0, freeze_after=None),
    "legacy_rotation": dict(adapt=True, corr_dims=[3, 4, 5], baseline="none", correction_scale=1.0, freeze_after=None),
    "dob_translation": dict(adapt=True, corr_dims=[0, 1, 2], baseline="dob", correction_scale=1.0, freeze_after=None),
    "legacy_half": dict(adapt=True, corr_dims=[0, 1, 2], baseline="none", correction_scale=0.5, freeze_after=None),
    "legacy_hold30": dict(adapt=True, corr_dims=[0, 1, 2], baseline="none", correction_scale=1.0, freeze_after=30),
    "weighted_translation": dict(adapt=True, corr_dims=[0, 1, 2], baseline="weighted_dob", correction_scale=1.0, freeze_after=None),
    "weighted_full": dict(adapt=True, corr_dims=[0, 1, 2, 3, 4, 5], baseline="weighted_dob", correction_scale=1.0, freeze_after=None),
}
DEFAULT_ARMS = ("off", "legacy_translation", "legacy_full", "legacy_rotation",
                "dob_translation", "legacy_half", "legacy_hold30")
COMMON = dict(gamma=0.08, dead=0.008, norm_r=0.15, clip=0.30, sev=0.0,
              bias=None, law="legacy", norm_channels="all", deadzone_mode="zero",
              profile="step", prof_p=60.0, onset=0)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path, payload):
    """Publish one complete JSON snapshot; never expose a truncated output file."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, default=json_default, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def arm_order(names, episode):
    """Prespecified cyclic order, reversed on odd episodes; outcomes play no role."""
    offset = episode % len(names)
    order = list(names[offset:]) + list(names[:offset])
    return list(reversed(order)) if episode % 2 else order


def scenario_state(env):
    """Capture all model qpos/qvel immediately before the first policy action."""
    robot_env = env.env if hasattr(env, "env") else env
    qpos = np.asarray(robot_env.sim.data.qpos, dtype="<f8").copy()
    qvel = np.asarray(robot_env.sim.data.qvel, dtype="<f8").copy()
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("nonfinite simulator state before policy action")
    digest = hashlib.sha256()
    digest.update(json.dumps([list(qpos.shape), list(qvel.shape)]).encode())
    digest.update(qpos.tobytes())
    digest.update(qvel.tobytes())
    from libero_reset import physics_fingerprint
    return dict(qpos=qpos.tolist(), qvel=qvel.tolist(), sha256=digest.hexdigest(),
                physics=physics_fingerprint(env))


def state_difference(reference, observed):
    differences = {}
    for name in ("qpos", "qvel"):
        a, b = np.asarray(reference[name]), np.asarray(observed[name])
        if a.shape != b.shape:
            raise ValueError(f"pairing state dimension changed: {name}")
        differences[f"max_abs_{name}"] = float(np.max(np.abs(a-b))) if a.size else 0.0
    if "physics" in reference or "physics" in observed:
        a = reference.get("physics", {}).get("fields", {})
        b = observed.get("physics", {}).get("fields", {})
        # qpos/qvel have the numeric tolerance above. Remaining model/data fields
        # must agree exactly; random fixture geometry is not in saved init states.
        keys = (set(a) | set(b)) - {"data.qpos", "data.qvel"}
        differences["nonstate_physics_mismatch"] = float(any(a.get(k) != b.get(k) for k in keys))
    return differences


def persist(args, study):
    study["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    # Each candidate view contains the same shared off outcomes, explicitly identified.
    for name in args.arm_names:
        if name == "off":
            continue
        config = ARM_CONFIGS[name]
        view_args = dict(study["args"], baseline=config["baseline"],
                         corr_dims=",".join(map(str, config["corr_dims"])),
                         correction_scale=config["correction_scale"], freeze_after=config["freeze_after"],
                         followup_arm=name, estimate_only=False, static_corr=None)
        view = dict(args=view_args, gamma=COMMON["gamma"], joint_fault=args.joint_fault,
                    status=study["status"], study_id=study["study_id"],
                    shared_control_id=study["shared_control_id"],
                    shared_control_source="study.json:arms.off",
                    source_hashes=study["source_hashes"],
                    pairing=study["pairing"],
                    arms=dict(frozen_faulted=study["arms"]["off"], adaptive=study["arms"][name]))
        atomic_json(args.out_dir / f"{name}.json", view)
    atomic_json(args.out_dir / "study.json", study)


def execute(args, probe, law, W, M, M_inv, source_hashes, healthy_covariance=None):
    """The ordered rollout loop, factored so its integrity can be tested without a simulator."""
    count = probe.suite.n_tasks
    scenarios = [(i % count, args.eval_init + i // count) for i in range(args.episodes)]
    for task, init in scenarios:
        _, _, inits = probe.env_for(task)
        if not 0 <= init < len(inits):
            raise ValueError(f"task {task} has {len(inits)} initial states; requested {init}")
    parsed = dict(vars(args), **COMMON)
    parsed.pop("arm_names", None)
    parsed.update(fault_vec=None, gain=None, joint_fault=args.joint_fault,
                  task_stride=1, policy_rng_pinned=False, reset_protocol="libero-reset-v1")
    parsed = json.loads(json.dumps(parsed, default=json_default))
    study_id = str(uuid.uuid4())
    schedule = [dict(episode=i, task=task, init=init, arms=arm_order(args.arm_names, i))
                for i, (task, init) in enumerate(scenarios)]
    study = dict(schema_version=1, study_id=study_id, stage=args.stage,
                 status="planned", args=parsed, source_hashes=source_hashes,
                 shared_control_id=f"{study_id}:off", schedule=schedule,
                 arm_configs={name: ARM_CONFIGS[name] for name in args.arm_names},
                 pairing=dict(mechanism="libero_set_init_state", checked_states=0,
                              valid_so_far=True, tolerance=args.state_tolerance,
                              snapshot_phase="before first policy action and before first torque injection",
                              fields=["full_qpos", "full_qvel", "model_and_data_fingerprint"], policy_rng_pinned=False),
                 arms={name: dict(successes=0, n=0, per_ep=[], f_hat=[], traj=[], f_true=[])
                       for name in args.arm_names})
    persist(args, study)
    header = dict(experiment="joint_followup_interleaved", study_id=study_id,
                  args=parsed, config=dict(W=W, M=M, M_inv=M_inv, bias=None,
                                          healthy_covariance=healthy_covariance, allocation_prior_std=0.1,
                                          common=COMMON, arm_configs=study["arm_configs"]),
                  source_hashes=source_hashes, schedule=schedule,
                  schema=dict(scenario_start="full qpos/qvel at first pre-act callback; no policy torque fault yet",
                              arm="actual named arm, with shared off cohort recorded once",
                              warmup="10 dummy-action steps, torque fault not yet applied"))
    current = None
    try:
        with law.telemetry_stream(args.telemetry, header) as telemetry:
            # The control handshake uses its own task0/init8 setup probe, not a scored episode.
            study["status"] = "configuring_policy"
            persist(args, study)
            ack = probe.control(dict(site=None, pin_rng=False))
            study["policy_control_ack"] = ack
            study["setup_probe"] = dict(task=0, init=8, scored=False)
            study["status"] = "running"
            persist(args, study)
            for batch in schedule:
                reference, reference_arm = None, None
                for name in batch["arms"]:
                    task, init, episode = batch["task"], batch["init"], batch["episode"]
                    current = dict(arm=name, episode=episode, task=task, init=init)
                    config = ARM_CONFIGS[name]
                    snapshot = {}

                    def observer(phase, values):
                        nonlocal reference, reference_arm
                        if phase != "before" or snapshot:
                            return
                        captured = scenario_state(values["env"])
                        if reference is None:
                            reference, reference_arm = captured, name
                        differences = state_difference(reference, captured)
                        valid = max(differences.values()) <= args.state_tolerance
                        snapshot.update(captured, comparison_to_arm=reference_arm,
                                        exact_hash_match=captured["sha256"] == reference["sha256"],
                                        pairing_valid=valid, **differences)
                        law.write_telemetry(telemetry, dict(type="scenario_start", **current,
                                                          t=values["t"], **snapshot))
                        study["pairing"]["checked_states"] += 1
                        if not valid:
                            study["pairing"]["valid_so_far"] = False
                            study["pairing"]["mismatch"] = dict(**current, **snapshot)
                            raise RuntimeError(f"scenario pairing failed before {name} task={task} init={init}: {differences}")

                    success, estimate, trace = law.run(
                        probe, task, init, COMMON["sev"], M_inv, W, COMMON["gamma"], config["adapt"],
                        max_steps=args.max_steps, dead=COMMON["dead"], norm_r=COMMON["norm_r"],
                        clip=COMMON["clip"], bias=None, corr_dims=config["corr_dims"],
                        law=COMMON["law"], M=M, baseline=config["baseline"],
                        joint_fault=args.joint_fault, norm_channels=COMMON["norm_channels"],
                        deadzone_mode=COMMON["deadzone_mode"], telemetry=telemetry,
                        episode=episode, arm=name, step_observer=observer,
                        freeze_after=config["freeze_after"], correction_scale=config["correction_scale"],
                        scenario_reset=True, kf_r=healthy_covariance)
                    if not snapshot:
                        raise RuntimeError("runner never supplied the required pre-act state callback")
                    arm = study["arms"][name]
                    arm["per_ep"].append(dict(task=task, init=init, ok=bool(success),
                                              episode=episode, order_index=batch["arms"].index(name),
                                              initial_state=snapshot))
                    arm["successes"] += int(success)
                    arm["n"] += 1
                    arm["f_hat"].append(np.asarray(estimate).tolist())
                    arm["traj"].append([row["f_hat"] for row in trace])
                    arm["f_true"].append([row["f_true"] for row in trace])
                    telemetry.flush()
                    persist(args, study)
                    print(f"[{name}] task={task} init={init} success={int(success)} "
                          f"total={arm['successes']}/{arm['n']}", flush=True)
            if any(arm["n"] != args.episodes for arm in study["arms"].values()):
                raise RuntimeError("study ended with incomplete arms")
            study["status"] = "complete"
            persist(args, study)
    except BaseException as exc:
        study.update(status="failed", failed_at=current,
                     error=dict(type=type(exc).__name__, message=str(exc)))
        persist(args, study)
        raise
    return study


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("dev", "confirm"), default="dev")
    p.add_argument("--joint", type=int, choices=range(7), default=5,
                   help="zero-based physical Panda joint index")
    p.add_argument("--torque", type=float, default=5.0,
                   help="constant physical torque in N.m; zero supplies a healthy control cell")
    p.add_argument("--eval-init", type=int, default=30)
    p.add_argument("--episodes", type=int, default=10, help="scenarios per arm, not total rollout count")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="ordered comma-separated names; off is required")
    p.add_argument("--suite", default="libero_spatial",
                   choices=("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"))
    p.add_argument("--log", type=Path, default=ROOT / "results/phase05/error_signal_so3.json")
    p.add_argument("--openloop", type=Path, default=ROOT / "results/phase05/openloop_so3.json")
    p.add_argument("--control", type=Path, default=Path("/tmp/ctl_joint_followup.json"))
    p.add_argument("--ack", type=Path, default=Path("/tmp/ack_joint_followup.json"))
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--telemetry", type=Path, help="default: OUT_DIR/telemetry.jsonl")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int,
                   help="default: benchmark episode cap (spatial 220, object 280, goal 300)")
    p.add_argument("--state-tolerance", type=float, default=1e-9)
    p.add_argument("--selection-record", type=Path,
                   help="required for confirm; locked candidate/cohort/multiplicity manifest")
    p.add_argument("--selftest", action="store_true")
    return p


def selftest():
    import contextlib
    import io
    from types import SimpleNamespace
    import unittest

    class Tests(unittest.TestCase):
        def test_schedule_and_state(self):
            names = list(ARM_CONFIGS)
            for i in range(20):
                self.assertEqual(sorted(arm_order(names, i)), sorted(names))
            self.assertNotEqual(arm_order(names, 0), arm_order(names, 1))
            env = SimpleNamespace(sim=SimpleNamespace(
                model=SimpleNamespace(body_pos=np.zeros((2, 3))),
                data=SimpleNamespace(qpos=np.zeros(12), qvel=np.zeros(11))))
            state = scenario_state(env)
            self.assertEqual(state["sha256"], scenario_state(env)["sha256"])
            env.sim.data.qpos[9] = 1
            self.assertEqual(state_difference(state, scenario_state(env))["max_abs_qpos"], 1)
            env.sim.model.body_pos[1, 0] = .1
            self.assertEqual(state_difference(state, scenario_state(env))["nonstate_physics_mismatch"], 1)

        def test_interleaving_and_shared_off_outputs(self):
            with tempfile.TemporaryDirectory() as directory:
                args = parser().parse_args(["--out-dir", directory, "--episodes", "2",
                                            "--arms", "off,legacy_half,legacy_hold30"])
                args.arm_names = args.arms.split(",")
                args.joint_fault = "torque:5:5.0"
                args.telemetry = args.out_dir / "telemetry.jsonl"
                data = SimpleNamespace(qpos=np.zeros(12), qvel=np.zeros(11))
                env = SimpleNamespace(sim=SimpleNamespace(data=data, model=SimpleNamespace()))
                probe = SimpleNamespace(suite=SimpleNamespace(n_tasks=10),
                                        env_for=lambda _: (env, "test", range(50)), control=lambda _: dict(pin_rng=False))
                calls = []
                def run(pr, task, init, *positional, **kwargs):
                    calls.append((task, kwargs["arm"], kwargs["freeze_after"], kwargs["correction_scale"]))
                    kwargs["step_observer"]("before", dict(env=env, t=10))
                    return True, np.zeros(6), [dict(f_hat=[0.0]*6, f_true=[0.0]*6)]
                @contextlib.contextmanager
                def telemetry(path, header):
                    with path.open("w") as stream:
                        stream.write(json.dumps(header, default=json_default)+"\n")
                        yield stream
                law = SimpleNamespace(run=run, telemetry_stream=telemetry,
                                      write_telemetry=lambda stream, row: stream.write(json.dumps(row)+"\n"))
                with contextlib.redirect_stdout(io.StringIO()):
                    result = execute(args, probe, law, np.zeros((6, 8)), np.eye(6), np.eye(6), {})
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(calls), 6)
                self.assertEqual([x[1] for x in calls[:3]], arm_order(args.arm_names, 0))
                self.assertEqual([x[1] for x in calls[3:]], arm_order(args.arm_names, 1))
                self.assertTrue(all(x[2] == 30 for x in calls if x[1] == "legacy_hold30"))
                self.assertTrue(all(x[3] == .5 for x in calls if x[1] == "legacy_half"))
                views = [json.loads((args.out_dir/f"{name}.json").read_text()) for name in args.arm_names[1:]]
                self.assertEqual(views[0]["shared_control_id"], views[1]["shared_control_id"])
                self.assertEqual(views[0]["arms"]["frozen_faulted"], views[1]["arms"]["frozen_faulted"])
                self.assertEqual(result["pairing"]["checked_states"], 6)

    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.selftest:
        return 0 if selftest() else 1
    args.arm_names = [name.strip() for name in args.arms.split(",")]
    if (not args.arm_names or len(set(args.arm_names)) != len(args.arm_names)
            or any(name not in ARM_CONFIGS for name in args.arm_names) or "off" not in args.arm_names):
        p.error("--arms must list unique known names and include off")
    if (args.episodes < 1 or args.eval_init < 0 or args.replan_steps < 1
            or (args.max_steps is not None and args.max_steps < 1)):
        p.error("episode count, step counts, and initial state are invalid")
    if not np.isfinite(args.state_tolerance) or args.state_tolerance < 0:
        p.error("--state-tolerance must be finite and nonnegative")
    if not np.isfinite(args.torque):
        p.error("--torque must be finite")
    if args.out_dir is None:
        p.error("--out-dir is required")
    if args.stage == "confirm" and (args.selection_record is None or not args.selection_record.is_file()):
        p.error("confirmation requires an existing --selection-record")
    args.out_dir = args.out_dir.resolve()
    args.telemetry = (args.telemetry or args.out_dir / "telemetry.jsonl").resolve()
    args.joint_fault = f"torque:{args.joint}:{args.torque}"
    targets = [args.out_dir / "study.json", args.telemetry] + [args.out_dir/f"{n}.json" for n in args.arm_names if n != "off"]
    if any(path.exists() for path in targets):
        p.error("an output already exists; use a fresh --out-dir/telemetry path to preserve prior results")
    inputs = [args.log.resolve(), args.openloop.resolve(), args.control.resolve(), args.ack.resolve()]
    if args.telemetry in inputs or len(set(targets)) != len(targets):
        p.error("telemetry/output paths must be distinct from each other and calibration/control paths")
    source_paths = [Path(__file__), ROOT / "openpi/adaptive_law.py", ROOT / "openpi/joint_fault.py",
                    ROOT / "openpi/paired_probe.py", ROOT / "openpi/libero_reset.py", args.log, args.openloop,
                    ROOT / "prereg_records/PREREG_COMPOSITE_JOINT_FOLLOWUP.md"]
    if args.selection_record is not None:
        source_paths.append(args.selection_record)
    if any(ARM_CONFIGS[name]["baseline"] == "weighted_dob" for name in args.arm_names):
        source_paths.append(ROOT / "openpi/weighted_dob.py")
    source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    import adaptive_law as law
    import paired_probe
    if args.max_steps is None:
        args.max_steps = paired_probe.SUITE_MAX[args.suite]
    W = law.fit_plant(args.log)
    calibration = json.loads(args.openloop.read_text())
    M = np.asarray(calibration["M"], float)
    if M.shape != (6, 6) or not np.isfinite(M).all():
        raise ValueError("M must be a finite six-by-six matrix")
    M_inv = np.linalg.pinv(M)
    healthy_covariance = None
    if any(ARM_CONFIGS[name]["baseline"] == "weighted_dob" for name in args.arm_names):
        healthy_covariance, covariance_report = law.healthy_residual_covariance(json.loads(args.log.read_text()), W)
        args.allocation_prior_std = 0.1
        args.healthy_covariance_samples = covariance_report["samples"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    probe = paired_probe.Probe(args)
    try:
        execute(args, probe, law, W, M, M_inv, source_hashes, healthy_covariance)
    finally:
        probe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
