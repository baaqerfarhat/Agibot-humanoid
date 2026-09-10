"""Interleaved ALOHA development with one shared off cohort per condition.

Every arm uses the same W/M/Q/R calibration artifact. Full MuJoCo qpos/qvel are
checked after reset, before policy inference or fault injection. The policy's
sampling RNG is not pinned. Outputs are unique, atomic progress snapshots; an
incomplete or failed study is retained and cannot be scored as complete.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

from run_joint_followup import atomic_json, arm_order, json_default, state_difference


ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "prereg_records/PREREG_COMPOSITE_JOINT_FOLLOWUP.md"
COMMON = dict(gamma=0.08, dead=0.002, norm_r=0.4, clip=0.08,
              law="legacy", deadzone_mode="zero", profile="step", prof_p=60.0,
              onset=0, corr=list(range(6)), damping=0.0)
ARM_CONFIGS = {
    "off": dict(adapt=False, baseline="none", norm_channels="all", tracking_strength=0.0),
    "legacy_all": dict(adapt=True, baseline="none", norm_channels="all", tracking_strength=0.0),
    "legacy_corrected": dict(adapt=True, baseline="none", norm_channels="corrected", tracking_strength=0.0),
    "dob": dict(adapt=True, baseline="dob", norm_channels="all", tracking_strength=0.0),
    "kalman": dict(adapt=True, baseline="kalman", norm_channels="all", tracking_strength=0.0),
    "composite_001": dict(adapt=True, baseline="composite", norm_channels="all", tracking_strength=0.01),
    "composite_005": dict(adapt=True, baseline="composite", norm_channels="all", tracking_strength=0.05),
    "composite_010": dict(adapt=True, baseline="composite", norm_channels="all", tracking_strength=0.1),
    "composite_025": dict(adapt=True, baseline="composite", norm_channels="all", tracking_strength=0.25),
}
FAULTS = {"healthy": np.zeros(14), "offset": np.r_[np.full(6, 0.02), np.zeros(8)]}


def scenario_state(env):
    """Read complete physical state, including the cube, without rendering hooks."""
    physics = env.unwrapped._env.physics
    qpos = np.asarray(physics.data.qpos, dtype="<f8").copy()
    qvel = np.asarray(physics.data.qvel, dtype="<f8").copy()
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("nonfinite full simulator state after reset")
    digest = hashlib.sha256()
    digest.update(json.dumps([list(qpos.shape), list(qvel.shape)]).encode())
    digest.update(qpos.tobytes())
    digest.update(qvel.tobytes())
    return dict(qpos=qpos.tolist(), qvel=qvel.tolist(), sha256=digest.hexdigest())


class ResetObserver:
    """Delegate to the real Aloha object, adding only a post-reset read callback."""

    def __init__(self, aloha, callback):
        self.aloha = aloha
        self.callback = callback

    def __getattr__(self, name):
        return getattr(self.aloha, name)

    def reset(self, episode):
        observation = self.aloha.reset(episode)
        self.callback(scenario_state(self.aloha.env))
        return observation


def calibrated_configs(artifact, names, law, *, artifact_path=None, source_log=None,
                       source_sensitivity=None):
    """Reject unsupported calibration before any simulated episode is started."""
    if not artifact.get("qualification", {}).get("allowed", False):
        raise ValueError("reference artifact failed healthy qualification")
    tracking = artifact["tracking"]
    observer = {key: np.asarray(artifact["observer"][key], float) for key in ("W", "M", "Q", "R")}
    for key, value in observer.items():
        expected = (law.NJ, law.K_FIR + 2) if key == "W" else (law.NJ, law.NJ)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"invalid shared observer {key}")
    if artifact["observer"].get("bias") is not None:
        raise ValueError("this study requires the declared no-bias shared observer")
    if (not np.isclose(tracking["dt"], law.DT) or
            not np.isclose(tracking["damping"], COMMON["damping"]) or
            not np.isclose(artifact["observer"]["dt"], law.DT) or
            not np.isclose(artifact["observer"]["gamma"], COMMON["gamma"])):
        raise ValueError("dt/damping/gamma differs from the declared calibration")
    if not np.array_equal(tracking["correction_mask"], np.r_[np.ones(6), np.zeros(8)]):
        raise ValueError("artifact correction mask differs from the left-six study")
    # These matrices generated the permitted raw feedback rates. Merely retaining
    # an 'allowed' flag while replacing R/Q/M would invalidate that screening.
    for actual, expected in ((observer["M"], tracking["H"]),
                             (observer["Q"], tracking["Q_step"]),
                             (observer["R"], tracking["R"])):
        if not np.array_equal(actual, np.asarray(expected)):
            raise ValueError("shared observer differs from qualified tracking matrices")
    configs = {}
    for name in names:
        config = dict(ARM_CONFIGS[name], tracking_rate=0.0)
        if config["baseline"] == "composite":
            candidates = [row for row in tracking["candidates"]
                          if np.isclose(row["strength"], config["tracking_strength"], rtol=0, atol=1e-14)]
            if len(candidates) != 1 or not candidates[0].get("qualified_for_experiment", False):
                raise ValueError(f"tracking strength for {name} is not qualified")
            config["tracking_rate"] = float(candidates[0]["tracking_rate"])
            if not np.isfinite(config["tracking_rate"]) or config["tracking_rate"] < 0:
                raise ValueError("qualified tracking rate must be finite and nonnegative")
        configs[name] = config
    if any(config["baseline"] == "composite" for config in configs.values()):
        from validate_composite_reference import validate_composite_artifact
        validate_composite_artifact(artifact, artifact_path=artifact_path,
            source_log=source_log, source_sensitivity=source_sensitivity)
    return observer, configs


def persist(args, study):
    study["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    for condition in args.condition_names:
        cohort = study["conditions"][condition]
        for name in args.arm_names:
            if name == "off":
                continue
            config = study["arm_configs"][name]
            recorded = dict(study["args"], **config)
            recorded.update(condition=condition, followup_arm=name,
                            fault_vec=",".join(str(float(x)) for x in FAULTS[condition]),
                            gain=None, corr_joints="0,1,2,3,4,5", static_corr=None,
                            warm_start=False, freeze_after=None, identify_episodes=None)
            view = dict(args=recorded, status=study["status"], study_id=study["study_id"],
                        shared_control_id=cohort["shared_control_id"],
                        shared_control_source=f"study.json:conditions.{condition}.arms.off",
                        source_hashes=study["source_hashes"], pairing=study["pairing"],
                        arms=dict(frozen_faulted=cohort["arms"]["off"], adaptive=cohort["arms"][name]))
            atomic_json(args.out_dir / f"{condition}_{name}.json", view)
    atomic_json(args.out_dir / "study.json", study)


def execute(args, aloha, law, artifact, source_hashes):
    observer, configs = calibrated_configs(artifact, args.arm_names, law,
        artifact_path=args.reference_artifact,
        source_log=getattr(args, "reference_source_log", None),
        source_sensitivity=getattr(args, "reference_source_sensitivity", None))
    M_inv = np.linalg.pinv(observer["M"])
    parsed = dict(vars(args), **COMMON)
    for key in ("arm_names", "condition_names"):
        parsed.pop(key, None)
    parsed.update(task=law.TASK, policy_rng_pinned=False, reset_estimate=True,
                  reset_covariance=True, reset_reference=True, dt=law.DT)
    parsed = json.loads(json.dumps(parsed, default=json_default))
    schedule = []
    for episode in range(args.episodes):
        conditions = arm_order(args.condition_names, episode)
        for position, condition in enumerate(conditions):
            schedule.append(dict(episode=episode, task=0, init=episode, seed=args.seed + episode,
                                 condition=condition, arms=arm_order(args.arm_names,
                                     episode * len(args.condition_names) + position)))
    study_id = str(uuid.uuid4())
    study = dict(schema_version=1, study_id=study_id, stage=args.stage, status="planned",
                 args=parsed, source_hashes=source_hashes, schedule=schedule, arm_configs=configs,
                 qualification=artifact["qualification"],
                 pairing=dict(mechanism="aloha_seeded_reset", checked_states=0, valid_so_far=True,
                              tolerance=args.state_tolerance, fields=["full_qpos", "full_qvel"],
                              snapshot_phase="after reset, before policy inference and fault injection",
                              policy_rng_pinned=False),
                 conditions={condition: dict(fault_vec=FAULTS[condition].tolist(),
                     shared_control_id=f"{study_id}:{condition}:off",
                     arms={name: dict(successes=0, n=0, per_ep=[], f_hat=[], traj=[], f_true=[])
                           for name in args.arm_names}) for condition in args.condition_names})
    persist(args, study)
    current = None
    previous_max_steps = law.MAX_STEPS
    law.MAX_STEPS = args.max_steps
    references = {}
    try:
        with args.telemetry.open("x", encoding="utf-8") as telemetry:
            law.write_telemetry(telemetry, dict(type="header", experiment="aloha_followup_interleaved",
                study_id=study_id, args=parsed, schedule=schedule, source_hashes=source_hashes,
                config=dict(common=COMMON, arm_configs=configs, observer=observer,
                            reference_model=artifact["model"], qualification=artifact["qualification"]),
                schema=dict(arm="condition/arm_name; one shared off per condition/seed",
                            scenario_start="complete physics qpos/qvel before the first policy action")))
            study["status"] = "running"
            persist(args, study)
            for batch in schedule:
                condition, episode = batch["condition"], batch["episode"]
                for order, name in enumerate(batch["arms"]):
                    current = dict(condition=condition, arm=name, episode=episode, task=0,
                                   init=episode, seed=batch["seed"])
                    snapshot = {}

                    def capture(captured):
                        if episode not in references:
                            references[episode] = (captured, f"{condition}/{name}")
                        reference, reference_arm = references[episode]
                        differences = state_difference(reference, captured)
                        valid = max(differences.values()) <= args.state_tolerance
                        snapshot.update(captured, comparison_to_arm=reference_arm,
                                        exact_hash_match=captured["sha256"] == reference["sha256"],
                                        pairing_valid=valid, **differences)
                        law.write_telemetry(telemetry, dict(type="scenario_start", **current, **snapshot))
                        study["pairing"]["checked_states"] += 1
                        if not valid:
                            study["pairing"].update(valid_so_far=False, mismatch=dict(**current, **snapshot))
                            raise RuntimeError(f"full-state pairing failed for {condition}/{name} seed={batch['seed']}")

                    config = configs[name]
                    kwargs = {key: value for key, value in config.items() if key != "tracking_strength"}
                    success, estimate, trace = law.episode(ResetObserver(aloha, capture), episode,
                        W=observer["W"], M=observer["M"], M_inv=M_inv,
                        kf_q=observer["Q"], kf_r=observer["R"], reference_model=artifact["model"],
                        fvec=FAULTS[condition], f_init=None, static_corr=None, freeze_after=None,
                        telemetry=telemetry, arm=f"{condition}/{name}", **COMMON, **kwargs)
                    if not snapshot:
                        raise RuntimeError("episode did not call the required reset observer")
                    arm = study["conditions"][condition]["arms"][name]
                    arm["n"] += 1
                    arm["successes"] += int(success)
                    arm["per_ep"].append(dict(task=0, init=episode, actual_seed=batch["seed"],
                                              episode=episode, order_index=order, ok=bool(success),
                                              initial_state=snapshot))
                    arm["f_hat"].append(np.asarray(estimate).tolist())
                    arm["traj"].append([row["f_hat"] for row in trace])
                    arm["f_true"].append([row["f_true"] for row in trace])
                    persist(args, study)
                    print(f"[{condition}/{name}] seed={batch['seed']} success={int(success)} "
                          f"total={arm['successes']}/{arm['n']}", flush=True)
            if any(arm["n"] != args.episodes for cohort in study["conditions"].values()
                   for arm in cohort["arms"].values()):
                raise RuntimeError("study ended with incomplete arms")
            study["status"] = "complete"
            persist(args, study)
    except BaseException as error:
        study.update(status="failed", failed_at=current,
                     error=dict(type=type(error).__name__, message=str(error)))
        persist(args, study)
        raise
    finally:
        law.MAX_STEPS = previous_max_steps
    return study


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("dev", "confirm"), default="dev")
    p.add_argument("--seed", type=int, default=2700)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--arms", default=",".join(ARM_CONFIGS))
    p.add_argument("--conditions", default="healthy,offset")
    p.add_argument("--reference-artifact", type=Path,
                   default=ROOT / "results/composite_followup/calibration/aloha_reference.json")
    p.add_argument("--reference-source-log", type=Path,
                   help="composite validation: portable copy with the recorded healthy-source SHA256")
    p.add_argument("--reference-source-sensitivity", type=Path,
                   help="composite validation: portable copy with the recorded M-source SHA256")
    p.add_argument("--prereg", type=Path, default=PREREG)
    p.add_argument("--selection-record", type=Path, help="required for confirmation")
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
    args.arm_names = [x.strip() for x in args.arms.split(",")]
    args.condition_names = [x.strip() for x in args.conditions.split(",")]
    for names, known in ((args.arm_names, ARM_CONFIGS), (args.condition_names, FAULTS)):
        if not names or len(names) != len(set(names)) or any(x not in known for x in names):
            p.error("arms and conditions must list unique known names")
    if "off" not in args.arm_names:
        p.error("the shared off arm is required")
    if args.seed < 0 or args.episodes < 1 or args.max_steps < 1:
        p.error("seed must be nonnegative; episodes and max-steps must be positive")
    if not np.isfinite(args.state_tolerance) or args.state_tolerance < 0:
        p.error("state tolerance must be finite and nonnegative")
    if args.stage == "confirm" and (args.selection_record is None or not args.selection_record.is_file()):
        p.error("confirmation requires an existing locked --selection-record")
    args.out_dir = args.out_dir.resolve()
    args.telemetry = (args.telemetry or args.out_dir / "telemetry.jsonl").resolve()
    targets = [args.out_dir / "study.json", args.out_dir / "run_config.json", args.telemetry,
               args.out_dir / ".run_claim"] + [args.out_dir / f"{condition}_{name}.json"
                   for condition in args.condition_names for name in args.arm_names if name != "off"]
    sources = [Path(__file__), ROOT / "openpi/aloha_adapt.py", ROOT / "openpi/adaptive_law.py",
               ROOT / "openpi/composite_observer.py", ROOT / "openpi/prepare_composite_reference.py",
               ROOT / "openpi/validate_composite_reference.py",
               ROOT / "openpi/run_joint_followup.py", args.reference_artifact, args.prereg]
    if args.selection_record is not None:
        sources.append(args.selection_record)
    if any(path.exists() for path in targets):
        p.error("an output already exists; choose a fresh output directory and telemetry file")
    if len(set(targets)) != len(targets) or set(targets).intersection(path.resolve() for path in sources):
        p.error("all output paths must be distinct from outputs and source inputs")
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    artifact = json.loads(args.reference_artifact.read_text())
    import aloha_adapt as law
    _, configs = calibrated_configs(artifact, args.arm_names, law,
        artifact_path=args.reference_artifact, source_log=args.reference_source_log,
        source_sensitivity=args.reference_source_sensitivity)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / ".run_claim").open("x") as claim:
        claim.write(str(uuid.uuid4()) + "\n")
    atomic_json(args.out_dir / "run_config.json", dict(args=vars(args), common=COMMON,
                arm_configs=configs, source_hashes=hashes,
                locked_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    aloha = None
    try:
        aloha = law.Aloha(args.host, args.port, args.seed)
        execute(args, aloha, law, artifact, hashes)
    except BaseException as error:
        if not (args.out_dir / "study.json").exists():
            atomic_json(args.out_dir / "study.json", dict(status="failed", stage=args.stage,
                source_hashes=hashes, failed_at="environment setup", error=dict(
                    type=type(error).__name__, message=str(error))))
        raise
    finally:
        if aloha is not None:
            aloha.env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
