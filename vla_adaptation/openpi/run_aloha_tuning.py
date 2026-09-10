"""Run a source-bound ALOHA tuning plan with interleaved, paired resets.

The candidate bank is reconstructed from healthy data before inference. The plan
selects already-qualified candidates, conditions and disjoint seeds. W/M and the
healthy position reference are shared; candidate Q/R/P0, clipping, damping and
tracking feedback are explicit. Kalman uses the SAME Joseph covariance update as
composite with tracking disabled. That execution route is recorded honestly.

Physical initial states are paired; policy sampling randomness is not pinned.
This runner does not choose parameters or infer a method ranking. Oracle is an
unranked diagnostic and is allowed only for a constant fault from episode start.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

import numpy as np

from run_joint_followup import atomic_json, arm_order, json_default, state_difference
from run_aloha_followup import ResetObserver

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULTS = dict(gamma=0.08, dead=0.002, norm_r=0.4, clip=0.08,
                norm_channels="all", damping=0.0, tracking_rate=0.0,
                ki=0.02, rls_lambda=0.95, rls_p0=1.0)
FAMILIES = {"off", "legacy", "dob", "rls", "integral_calibrated", "kalman", "composite", "oracle"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise RuntimeError("source changed during tuning study: " + str(path))


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a finite number")
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return float(value)


def normalize_plan(plan, dimension=14):
    """Validate the schedule before allocating outputs or contacting a server."""
    allowed = {"schema_version", "bank_sha256", "stage", "seed", "episodes", "max_steps",
               "correction_indices", "conditions", "candidate_names", "state_tolerance",
               "selection_record", "description", "excluded_seeds", "metadata"}
    if not isinstance(plan, dict) or set(plan) - allowed:
        raise ValueError("unknown or invalid plan fields")
    if plan.get("schema_version") != 1:
        raise ValueError("tuning plan schema_version must equal 1")
    if not re.fullmatch(r"[0-9a-f]{64}", plan.get("bank_sha256", "")):
        raise ValueError("plan requires the candidate bank SHA256")
    if plan.get("stage") not in ("tuning", "confirmation"):
        raise ValueError("stage must be tuning or confirmation")
    result = dict(plan)
    for key, default, minimum in (("seed", 0, 0), ("episodes", 0, 1), ("max_steps", 300, 1)):
        value = plan.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(key + " must be an integer >= " + str(minimum))
        result[key] = value
    excluded = plan.get("excluded_seeds", [])
    if not isinstance(excluded, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in excluded):
        raise ValueError("excluded_seeds must list nonnegative integers")
    if set(range(result["seed"], result["seed"] + result["episodes"])) & set(excluded):
        raise ValueError("study seeds overlap excluded calibration or previous study seeds")
    result["state_tolerance"] = _number(plan.get("state_tolerance", 1e-9), "state_tolerance")
    corrections = plan.get("correction_indices", list(range(6)))
    if (not isinstance(corrections, list) or not corrections or len(set(corrections)) != len(corrections)
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 or x >= dimension for x in corrections)):
        raise ValueError("correction_indices must be distinct valid coordinates")
    result["correction_indices"] = corrections
    names = plan.get("candidate_names", [])
    if (not isinstance(names, list) or len(names) < 2 or len(names) != len(set(names))
            or any(not isinstance(x, str) or not NAME.fullmatch(x) for x in names) or "off" not in names):
        raise ValueError("candidate_names must be unique safe identifiers including shared off")
    conditions, names = [], set()
    for row in plan.get("conditions", []):
        if not isinstance(row, dict) or set(row) - {"name", "kind", "fault_vec", "joint", "torque", "profile", "onset", "prof_p"}:
            raise ValueError("unknown or invalid condition fields")
        name = row.get("name", "")
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise ValueError("condition names must be unique safe identifiers")
        names.add(name)
        condition = dict(name=name, kind=row.get("kind", "offset"), profile=row.get("profile", "step"),
                         onset=row.get("onset", 0), prof_p=_number(row.get("prof_p", 60), "prof_p", positive=True))
        if condition["profile"] not in ("step", "ramp", "sine_bias", "intermittent"):
            raise ValueError("unknown disturbance profile")
        if isinstance(condition["onset"], bool) or not isinstance(condition["onset"], int) or condition["onset"] < 0:
            raise ValueError("condition onset must be a nonnegative integer")
        if condition["kind"] == "offset":
            if "joint" in row or "torque" in row:
                raise ValueError("an offset condition cannot also inject torque")
            vector = np.asarray(row.get("fault_vec"), float)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                raise ValueError("condition requires one finite fault offset per command coordinate")
            condition["fault_vec"] = vector.tolist()
        elif condition["kind"] == "torque":
            joint = row.get("joint")
            if isinstance(joint, bool) or not isinstance(joint, int) or not 0 <= joint < 12:
                raise ValueError("physical arm joint must be an integer in 0..11")
            torque = row.get("torque")
            if isinstance(torque, bool) or not isinstance(torque, (int, float)) or not np.isfinite(torque):
                raise ValueError("torque must be finite, in Nm")
            if "fault_vec" in row:
                recorded = np.asarray(row["fault_vec"], float)
                if recorded.shape != (dimension,) or np.any(recorded != 0):
                    raise ValueError("a torque condition cannot also inject a command offset")
            condition.update(joint=joint, torque=float(torque), fault_vec=np.zeros(dimension).tolist())
        else:
            raise ValueError("condition kind must be offset or torque")
        conditions.append(condition)
    if not conditions:
        raise ValueError("at least one condition is required")
    result["conditions"] = conditions
    if result["stage"] == "confirmation":
        record = result.get("selection_record", {})
        if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
                or not isinstance(record["path"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError("confirmation requires a hash-bound selection_record with path and sha256")
    return result


def select_candidates(plan, validated, dimension=14):
    """Check explicit numerical configurations returned by the healthy validator."""
    source = validated["candidates"]
    configs, signatures = {}, set()
    for name in plan["candidate_names"]:
        if name not in source:
            raise ValueError("candidate is missing or rejected by healthy qualification: " + name)
        row = dict(source[name])
        family = row.get("family")
        if family not in FAMILIES or (name == "off") != (family == "off"):
            raise ValueError("invalid family or shared off candidate name")
        if row.get("qualified", row.get("allowed", True)) is not True:
            raise ValueError("candidate was not qualified: " + name)
        config = dict(DEFAULTS)
        for key in DEFAULTS:
            if key in row:
                config[key] = row[key]
        config.update(name=name, family=family)
        for key in set(DEFAULTS) - {"norm_channels"}:
            config[key] = _number(config[key], key, positive=key not in {"dead", "damping", "tracking_rate"})
        if config["gamma"] > 1 or config["rls_lambda"] > 1:
            raise ValueError("gamma and RLS forgetting factor must be <= 1")
        if config["norm_channels"] not in ("all", "corrected"):
            raise ValueError("norm_channels must be all or corrected")
        if family != "composite" and config["tracking_rate"] != 0:
            raise ValueError("only composite candidates may use tracking feedback")
        if family not in ("kalman", "composite") and config["damping"] != 0:
            raise ValueError("damping is supported only by Kalman/composite")
        if family in ("kalman", "composite"):
            for key in ("Q", "R", "P0"):
                matrix = np.asarray(row.get(key), float)
                if key == "P0" and matrix.ndim == 0:
                    matrix = np.eye(dimension) * float(matrix)
                if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
                    raise ValueError(key + " must be a finite square parameter/observation covariance")
                if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-20):
                    raise ValueError(key + " must be symmetric")
                if np.linalg.eigvalsh(matrix).min() <= 0:
                    raise ValueError(key + " must be positive definite")
                config[key] = matrix.tolist()
        if family == "oracle" and any(c["profile"] != "step" or c["onset"] != 0 for c in plan["conditions"]):
            raise ValueError("static oracle diagnostic requires constant faults from episode start")
        # Persist all generator information, but hash only execution parameters.
        config["bank_candidate"] = row
        effective = {key: value for key, value in config.items() if key not in {"name", "bank_candidate"}}
        if family in ("kalman", "composite"):
            for key in ("gamma", "dead", "norm_r", "norm_channels", "ki", "rls_lambda", "rls_p0"):
                effective.pop(key)
        else:
            for key in ("damping", "tracking_rate"):
                effective.pop(key)
        signature = json.dumps(effective, sort_keys=True, default=json_default)
        if signature in signatures:
            raise ValueError("duplicate effective candidate configuration")
        signatures.add(signature)
        configs[name] = config
    return configs


def validate_runtime(plan, validated, law_dt=0.02):
    """Bind the experiment horizon, mask and seeds to the healthy bank's scope."""
    if list(validated["correction_indices"]) != plan["correction_indices"]:
        raise ValueError("plan correction mask differs from the qualified candidate bank")
    if not np.isclose(validated["dt"], law_dt, rtol=0, atol=1e-14):
        raise ValueError("episode control interval differs from candidate qualification")
    if validated["screened_steps"] < plan["max_steps"]:
        raise ValueError("episode horizon exceeds candidate qualification")
    seeds = set(range(plan["seed"], plan["seed"]+plan["episodes"]))
    if seeds & set(validated["calibration_seeds"]):
        raise ValueError("study seeds overlap candidate-bank calibration seeds")


def candidate_kwargs(candidate, observer, model, correction_indices):
    """Actual parameters passed to the unchanged ALOHA episode algorithm."""
    family = candidate["family"]
    baseline = "composite" if family in ("kalman", "composite") else (
        "none" if family in ("off", "legacy", "oracle") else family)
    return dict(adapt=family not in ("off", "oracle"), baseline=baseline,
        W=np.asarray(observer["W"]), M=np.asarray(observer["M"]),
        M_inv=np.linalg.pinv(observer["M"]),
        kf_q=np.asarray(candidate["Q"]) if "Q" in candidate else None,
        kf_r=np.asarray(candidate["R"]) if "R" in candidate else None,
        initial_covariance=np.asarray(candidate["P0"]) if "P0" in candidate else 1.0,
        reference_model=model, gamma=candidate["gamma"], dead=candidate["dead"],
        norm_r=candidate["norm_r"], clip=candidate["clip"], corr=correction_indices,
        norm_channels=candidate["norm_channels"], tracking_rate=candidate["tracking_rate"],
        damping=candidate["damping"], ki=candidate["ki"], rls_lambda=candidate["rls_lambda"],
        rls_p0=candidate["rls_p0"], law="legacy", deadzone_mode="zero",
        f_init=None, static_corr=None, freeze_after=None)


def oracle_offset(condition, metadata):
    """Diagnostic correction in action units; torque version assumes the servo gain."""
    oracle = np.asarray(condition["fault_vec"], float).copy()
    if condition["kind"] == "torque":
        joint = metadata[condition["joint"]]
        oracle[joint["command_coordinate"]] = condition["torque"] / joint["command_to_torque_gain"]
    return oracle


def fault_scale(condition, step):
    if step < condition["onset"]:
        return 0.0
    u, period = step-condition["onset"], condition["prof_p"]
    return {"step": 1.0, "ramp": min(1.0, u / max(period, 1e-9)),
            "sine_bias": 0.5 * (1 + np.sin(2*np.pi*u/max(period, 1e-9))),
            "intermittent": 1.0 if int(u // max(period, 1)) % 2 == 0 else 0.0}[condition["profile"]]


class CorrectionTelemetry:
    """Observe the actual logged correction while preserving raw output bytes."""

    def __init__(self, stream, arm, episode, dimension):
        self.stream, self.arm, self.episode = stream, arm, episode
        self.dimension = dimension
        self.pending = ""
        self.steps = 0
        self.energy = 0.0

    def write(self, text):
        count = self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            row = json.loads(line)
            if row.get("type") != "step":
                continue
            if row.get("arm") != self.arm or row.get("episode") != self.episode or row.get("t") != self.steps:
                raise RuntimeError("logged correction step has inconsistent episode/arm/order")
            correction = np.asarray(row.get("correction"), float)
            if correction.shape != (self.dimension,) or not np.isfinite(correction).all():
                raise RuntimeError("logged applied correction is not a finite command vector")
            self.energy += float(correction @ correction)
            if not np.isfinite(self.energy):
                raise RuntimeError("applied correction energy overflowed")
            self.steps += 1
        return count

    def flush(self):
        return self.stream.flush()


@contextlib.contextmanager
def condition_context(aloha, condition, fault_module):
    if condition["kind"] != "torque":
        yield
        return
    original = aloha.env
    class ProfiledTorque(fault_module.TorqueEnvironment):
        def step(self, command):
            self.magnitude = condition["torque"] * fault_scale(condition, self.steps)
            return super().step(command)
    wrapper = ProfiledTorque(original, condition["joint"], 0.0)
    aloha.env = wrapper
    try:
        yield
    finally:
        wrapper.restore()
        aloha.env = original


def persist(out_dir, study):
    study["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    for condition, cohort in study["conditions"].items():
        for name, config in study["arm_configs"].items():
            if name == "off":
                continue
            recorded = dict(study["args"], **config, condition=condition, followup_arm=name,
                            corr_joints=",".join(map(str, study["plan"]["correction_indices"])),
                            static_corr=None, warm_start=False, freeze_after=None, identify_episodes=None)
            definition = cohort["definition"]
            recorded.update(fault_vec=",".join(str(x) for x in definition["fault_vec"]),
                            profile=definition["profile"], onset=definition["onset"], prof_p=definition["prof_p"],
                            joint_fault=(f"torque:{definition['joint']}:{definition['torque']}"
                                         if definition["kind"] == "torque" else None),
                            oracle_diagnostic=config["family"] == "oracle")
            recorded["baseline"] = ("composite" if config["family"] in ("kalman", "composite")
                                    else ("none" if config["family"] in ("off", "legacy", "oracle")
                                          else config["family"]))
            if config["family"] == "oracle":
                recorded["static_corr"] = oracle_offset(definition, study["robot_joints"]).tolist()
            view = dict(args=recorded, status=study["status"], study_id=study["study_id"],
                        shared_control_id=cohort["shared_control_id"],
                        shared_control_source=f"study.json:conditions.{condition}.arms.off",
                        source_hashes=study["source_hashes"], pairing=study["pairing"],
                        arms=dict(frozen_faulted=cohort["arms"]["off"], adaptive=cohort["arms"][name]))
            atomic_json(out_dir / f"{condition}_{name}.json", view)
    atomic_json(out_dir / "study.json", study)


def execute(args, aloha, law, plan, validated, source_hashes, fault_module=None):
    """Execute a validated schedule, retaining partial data on every failure."""
    from run_aloha_joint_followup import PhysicalTelemetry, robot_metadata
    if fault_module is None:
        import aloha_joint_fault as fault_module
    validate_runtime(plan, validated, law.DT)
    configs = select_candidates(plan, validated, law.NJ)
    if aloha.seed != plan["seed"]:
        raise ValueError("client reset seed differs from the frozen plan")
    observer, model = validated["observer"], validated["reference_model"]
    metadata = robot_metadata(aloha, fault_module)
    schedule = []
    condition_names = [row["name"] for row in plan["conditions"]]
    definitions = {row["name"]: row for row in plan["conditions"]}
    for episode in range(plan["episodes"]):
        for position, condition in enumerate(arm_order(condition_names, episode)):
            schedule.append(dict(episode=episode, task=0, init=episode, seed=plan["seed"]+episode,
                condition=condition, arms=arm_order(plan["candidate_names"], episode*len(condition_names)+position)))
    identifier = str(uuid.uuid4())
    parsed = dict(task=law.TASK, suite=law.TASK, seed=plan["seed"], episodes=plan["episodes"],
                  max_steps=plan["max_steps"], dt=law.DT, horizon=law.HORIZON,
                  policy_rng_pinned=False, reset_estimate=True, reset_covariance=True, reset_reference=True,
                  plan=str(args.plan), candidate_bank=str(args.candidate_bank), stage=plan["stage"],
                  telemetry=str(args.telemetry), corr=plan["correction_indices"],
                  correction_coordinates=plan["correction_indices"],
                  kalman_execution="composite_step with tracking_rate=0; same covariance/damping/P0 support")
    study = dict(schema_version=1, study_id=identifier, stage=plan["stage"], status="planned",
        args=parsed, plan=plan, source_hashes=source_hashes, schedule=schedule, arm_configs=configs,
        qualification=validated["qualification"], robot_joints=metadata,
        pairing=dict(mechanism="aloha_seeded_reset", checked_states=0, valid_so_far=True,
                     tolerance=plan["state_tolerance"], fields=["full_qpos", "full_qvel"],
                     snapshot_phase="after reset, before policy inference or fault injection", policy_rng_pinned=False),
        conditions={name: dict(definition=definitions[name], fault_vec=definitions[name]["fault_vec"],
            shared_control_id=f"{identifier}:{name}:off",
            arms={name: dict(successes=0, n=0, per_ep=[], f_hat=[], traj=[], f_true=[], diagnostics=[])
                  for name in configs}) for name in condition_names})
    persist(args.out_dir, study)
    current, references, observed_keys = None, {}, set()
    previous_cap = law.MAX_STEPS
    law.MAX_STEPS = plan["max_steps"]
    try:
        with args.telemetry.open("x", encoding="utf-8") as telemetry:
            law.write_telemetry(telemetry, dict(type="header", experiment="aloha_parameter_tuning",
                study_id=identifier, args=parsed, plan=plan, schedule=schedule, source_hashes=source_hashes,
                config=dict(arm_configs=configs, observer=observer, reference_model=model,
                            qualification=validated["qualification"], robot_joints=metadata),
                schema=dict(arm="condition/candidate; one shared off per condition/seed",
                            oracle="Unranked diagnostic; static perfect additive offset or nominal -torque/kp",
                            f_true="Action-space injected offset; torque conditions record zero, not physical torque truth",
                            physical_step="Complete simulator state and force/control limit indicators",
                            applied_correction_energy="Per episode sum_t ||logged correction_t||^2, without dt; divide by applied_correction_steps for mean per-step energy")))
            study["status"] = "running"
            persist(args.out_dir, study)
            for batch in schedule:
                condition, episode = definitions[batch["condition"]], batch["episode"]
                for order, name in enumerate(batch["arms"]):
                    verify_sources(source_hashes)
                    key = (condition["name"], name, batch["seed"])
                    if key in observed_keys:
                        raise RuntimeError("duplicate scenario/candidate outcome key")
                    current = dict(condition=condition["name"], arm=name, episode=episode,
                                   task=0, init=episode, seed=batch["seed"])
                    snapshot = {}
                    diagnostics = {key: np.zeros(12, dtype=int) for key in
                                   ("force_limit", "control_limit", "requested_outside_control_range")}
                    physical_steps = 0
                    def capture(captured):
                        if snapshot:
                            raise RuntimeError("episode reset more than once")
                        if episode not in references:
                            references[episode] = (captured, f"{condition['name']}/{name}")
                        reference, reference_arm = references[episode]
                        differences = state_difference(reference, captured)
                        valid = max(differences.values()) <= plan["state_tolerance"]
                        snapshot.update(captured, comparison_to_arm=reference_arm,
                            exact_hash_match=captured["sha256"] == reference["sha256"], pairing_valid=valid, **differences)
                        law.write_telemetry(telemetry, dict(type="scenario_start", **current, **snapshot))
                        study["pairing"]["checked_states"] += 1
                        if not valid:
                            study["pairing"].update(valid_so_far=False, mismatch=dict(**current, **snapshot))
                            raise RuntimeError("full-state pairing failed for " + str(key))
                    def physical_log(row):
                        nonlocal physical_steps
                        physical_steps += 1
                        for field in diagnostics:
                            diagnostics[field] += np.asarray(row[field], dtype=int)
                        law.write_telemetry(telemetry, dict(**current, **row))
                    config = configs[name]
                    kwargs = candidate_kwargs(config, observer, model, plan["correction_indices"])
                    if config["family"] == "oracle":
                        kwargs["static_corr"] = oracle_offset(condition, metadata)
                    correction_log = CorrectionTelemetry(telemetry, f"{condition['name']}/{name}", episode, law.NJ)
                    with condition_context(aloha, condition, fault_module):
                        injected_environment = aloha.env
                        aloha.env = PhysicalTelemetry(injected_environment, metadata, physical_log)
                        try:
                            success, estimate, trace = law.episode(ResetObserver(aloha, capture), episode,
                                fvec=np.asarray(condition["fault_vec"]), profile=condition["profile"],
                                onset=condition["onset"], prof_p=condition["prof_p"], telemetry=correction_log,
                                arm=f"{condition['name']}/{name}", **kwargs)
                        finally:
                            aloha.env = injected_environment
                    if (not snapshot or not trace or physical_steps != len(trace)
                            or correction_log.steps != physical_steps or correction_log.pending):
                        raise RuntimeError("episode omitted reset, telemetry, or physical-step records")
                    verify_sources(source_hashes)
                    estimates = np.asarray([row["f_hat"] for row in trace])
                    if not np.isfinite(estimates).all() or not np.isfinite(estimate).all():
                        raise RuntimeError("nonfinite parameter estimate")
                    arm = study["conditions"][condition["name"]]["arms"][name]
                    arm["n"] += 1
                    arm["successes"] += int(success)
                    arm["per_ep"].append(dict(task=0, init=episode, actual_seed=batch["seed"], episode=episode,
                        order_index=order, ok=bool(success), initial_state=snapshot))
                    arm["f_hat"].append(np.asarray(estimate).tolist())
                    arm["traj"].append(estimates.tolist())
                    arm["f_true"].append([row["f_true"] for row in trace])
                    arm["diagnostics"].append(dict(physical_steps=physical_steps,
                        applied_correction_energy=correction_log.energy,
                        applied_correction_steps=correction_log.steps,
                        estimate_clip_steps=(np.abs(estimates) >= config["clip"]-1e-10).sum(axis=0).tolist(),
                        **{field: value.tolist() for field, value in diagnostics.items()}))
                    observed_keys.add(key)
                    persist(args.out_dir, study)
                    print(f"[{condition['name']}/{name}] seed={batch['seed']} success={int(success)} "
                          f"total={arm['successes']}/{arm['n']}", flush=True)
            expected = len(configs)*len(definitions)*plan["episodes"]
            if (len(observed_keys) != expected or study["pairing"]["checked_states"] != expected
                    or any(arm["n"] != plan["episodes"] for cohort in study["conditions"].values() for arm in cohort["arms"].values())):
                raise RuntimeError("study ended with incomplete or duplicated arms")
            verify_sources(source_hashes)
            study.update(status="complete", completed_source_recheck=True)
            persist(args.out_dir, study)
    except BaseException as error:
        study.update(status="failed", failed_at=current, error=dict(type=type(error).__name__, message=str(error)))
        persist(args.out_dir, study)
        raise
    finally:
        law.MAX_STEPS = previous_cap
    return study


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--candidate-bank", required=True, type=Path)
    p.add_argument("--reference-source-log", type=Path)
    p.add_argument("--reference-source-sensitivity", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--telemetry", type=Path)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8002)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.telemetry = (args.telemetry or args.out_dir / "telemetry.jsonl").resolve()
    plan = normalize_plan(json.loads(args.plan.read_text()))
    if digest(args.candidate_bank) != plan["bank_sha256"]:
        p.error("candidate bank SHA256 differs from the frozen plan")
    source_paths = [Path(__file__), ROOT/"openpi/aloha_adapt.py", ROOT/"openpi/adaptive_law.py",
        ROOT/"openpi/composite_observer.py", ROOT/"openpi/prepare_composite_reference.py",
        ROOT/"openpi/validate_composite_reference.py", ROOT/"openpi/prepare_composite_tuning.py",
        ROOT/"openpi/run_joint_followup.py", ROOT/"openpi/run_aloha_followup.py",
        ROOT/"openpi/run_aloha_joint_followup.py", ROOT/"openpi/aloha_joint_fault.py", args.plan, args.candidate_bank]
    targets = [args.out_dir/"study.json", args.out_dir/"run_config.json", args.out_dir/".run_claim", args.telemetry]
    targets += [args.out_dir/f"{condition['name']}_{name}.json" for condition in plan["conditions"]
                for name in plan["candidate_names"] if name != "off"]
    if any(path.exists() for path in targets):
        p.error("an output already exists; choose a fresh output directory and telemetry file")
    if len(targets) != len(set(targets)):
        p.error("output paths must be distinct")
    if plan["stage"] == "confirmation":
        selection = Path(plan["selection_record"]["path"])
        selection = selection if selection.is_absolute() else args.plan.resolve().parent / selection
        if digest(selection) != plan["selection_record"]["sha256"]:
            p.error("selection record SHA256 differs from the frozen plan")
        source_paths.append(selection)
    from prepare_composite_tuning import validate_candidate_bank
    bank = json.loads(args.candidate_bank.read_text())
    validated = validate_candidate_bank(bank, bank_path=args.candidate_bank,
        source_log=args.reference_source_log, source_sensitivity=args.reference_source_sensitivity)
    select_candidates(plan, validated)
    validate_runtime(plan, validated)
    hashes = {str(path.resolve()): digest(path) for path in source_paths}
    for path, expected in validated["sources"].items():
        resolved = str(Path(path).resolve())
        if resolved in hashes and hashes[resolved] != expected:
            p.error("conflicting source provenance")
        hashes[resolved] = expected
    if set(targets) & {Path(path) for path in hashes}:
        p.error("an output path collides with a source input")
    verify_sources(hashes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/".run_claim").open("x") as claim:
        claim.write(str(uuid.uuid4())+"\n")
    atomic_json(args.out_dir/"run_config.json", dict(args=vars(args), plan=plan,
        source_hashes=hashes, qualification=validated["qualification"],
        locked_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    import aloha_adapt as law
    aloha = None
    try:
        aloha = law.Aloha(args.host, args.port, plan["seed"])
        execute(args, aloha, law, plan, validated, hashes)
    except BaseException as error:
        if not (args.out_dir/"study.json").exists():
            atomic_json(args.out_dir/"study.json", dict(status="failed", stage=plan["stage"],
                source_hashes=hashes, failed_at="environment setup", error=dict(
                    type=type(error).__name__, message=str(error))))
        raise
    finally:
        if aloha is not None:
            aloha.env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
