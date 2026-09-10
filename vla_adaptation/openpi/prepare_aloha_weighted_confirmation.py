"""Prepare the fixed 13-cell ALOHA weighted transfer confirmation; never launch it.

Creates a separate registration-bound selection record, launch/collection plans,
an explicit collector API plan, and a prospective scoring manifest. All commands
are argv lists. Destination input verification performs no network or simulations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import aloha_adapt as law
from run_aloha_joint_followup import ARM_CONFIGS, COMMON, CORRECTION_COORDINATES, EXPECTED_GAINS, shared_observer


ROOT = Path(__file__).resolve().parent.parent
BASE = Path("/data/fxxie/vla/followup_20260908")
REGISTRATION = "prereg_records/PREREG_ALOHA_WEIGHTED_TRANSFER_CONFIRMATION.md"
ARMS = ["off", "legacy_allchannels", "weighted"]
CONDITIONS = ["healthy"]+[f"joint{joint}" for joint in range(12)]
EXPECTED_KEYS = [[0, seed] for seed in range(3000, 3020)]
FORBIDDEN_KEYS = [[0, seed] for seed in (*range(2600, 2610), *range(2700, 2720),
                                       *range(2800, 2820), *range(2900, 2920))]
FAMILIES = ("legacy_vs_off", "weighted_vs_off", "weighted_vs_legacy")
RUNNER_FILES = ("run_aloha_joint_followup.py", "aloha_adapt.py", "adaptive_law.py", "aloha_joint_fault.py",
                "run_aloha_followup.py", "run_joint_followup.py", "weighted_dob.py")
SNAPSHOT_FILES = RUNNER_FILES + ("libero_reset.py", "composite_observer.py", "collect_followup_results.py",
    "score_joint_followup.py", "mcnemar.py", "mcnemar_crosscheck.py", "prepare_aloha_weighted_confirmation.py",
    "test_prepare_aloha_weighted_confirmation.py", "validate_composite_reference.py", "prepare_composite_reference.py",
    "launch_followup_grid.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serialized(value):
    return (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_calibration(args):
    reference = json.loads(args.reference_artifact.read_text())
    observer = shared_observer(reference, law)
    require(reference.get("qualification", {}).get("allowed") is True
            and reference.get("model", {}).get("state_indices") == list(range(6)),
            "expected the existing left-six qualified artifact, used for observer matrices only")
    provenance = json.loads(args.healthy_provenance.read_text())
    healthy = json.loads(args.healthy_log.read_text())
    health_sha = digest(args.healthy_log)
    require(reference.get("source", {}).get("sha256") == health_sha
            and provenance.get("data_sha256") == health_sha, "healthy log/source provenance hash mismatch")
    logged = provenance.get("header", {}).get("args", {})
    require(logged.get("seed") == 2600 and logged.get("episodes") == 10 and logged.get("mode") == "log"
            and logged.get("fault_vec") is None and logged.get("gain") is None,
            "expected the healthy seed-2600 ten-episode collection")
    require(isinstance(healthy, list) and len(healthy) == 10, "healthy episode count differs")
    for episode in healthy:
        u, q = np.asarray(episode.get("u"), float), np.asarray(episode.get("q"), float)
        require(u.ndim == 2 and u.shape == q.shape and u.shape[1] == 14 and len(u) >= 7
                and np.isfinite(u).all() and np.isfinite(q).all(), "invalid healthy fourteen-channel transitions")
    require(reference["observer"].get("fit_episode_indices") == list(range(6))
            and reference.get("settings", {}).get("fit_episodes") == list(range(6))
            and reference["settings"].get("validation_episodes") == list(range(6, 10)),
            "observer/reference calibration and validation split differs from the fixed artifact")
    sensitivity = json.loads(args.sensitivity_artifact.read_text())
    require(reference["source"].get("sensitivity", {}).get("sha256") == digest(args.sensitivity_artifact),
            "historical sensitivity source hash mismatch")
    require(np.array_equal(observer["M"], np.asarray(sensitivity.get("M"), float)),
            "observer M differs from the linked sensitivity artifact")
    rejected = json.loads(args.rejected_reference.read_text())
    require(rejected.get("qualification", {}).get("allowed") is False,
            "retain the failed all-twelve reference qualification separately")
    used = {(0, seed) for seed in range(2600, 2610)}
    require(not used.intersection(map(tuple, EXPECTED_KEYS)), "confirmation overlaps healthy calibration")
    # Verify the currently imported dispatcher uses the fixed numeric regularizer;
    # its source bytes will also be bound in every confirmation result.
    from adaptive_law import estimator_step
    mask = np.isin(np.arange(14), CORRECTION_COORDINATES)
    _, diag = estimator_step(np.zeros(14), np.zeros(14), np.linalg.pinv(observer["M"]), gamma=.08,
        dead=.002, norm_r=.4, clip=.08, mask=mask, M=observer["M"], baseline="weighted_dob", kf_r=observer["R"])
    require(np.array_equal(np.asarray(diag["prior_std"]), np.full(14, .1))
            and np.array_equal(diag["active_input_indices"], CORRECTION_COORDINATES),
            "weighted regularizer or active input coordinates differ from the fixed transfer")
    return dict(observer_channels=14, corrected_coordinates=CORRECTION_COORDINATES,
        fit_seeds=list(range(2600, 2606)), validation_seeds=list(range(2606, 2610)),
        sensitivity_provenance="historical artifact reused by hash; complete original reset-seed metadata unavailable",
        reference_used=False, all_twelve_reference_qualification=rejected["qualification"],
        prior_std=.1, prior_units="numerical ALOHA command units; active twelve arm channels are radians")


def build(args):
    require(1 <= args.workers <= 6 and 0 < args.port < 65536, "invalid concurrency or policy port")
    require(all(path.is_absolute() for path in (args.snapshot_root, args.remote_root, args.python, args.upstream_root)),
            "remote runtime paths must be absolute")
    require(args.snapshot_root != args.remote_root, "frozen source and outcome roots must differ")
    require(ARM_CONFIGS.get("weighted", {}).get("baseline") == "weighted_dob", "weighted runner arm is unavailable")
    calibration = validate_calibration(args)
    source_root, destination = args.source_root.resolve(), args.out_dir.resolve()
    registration = args.registration or f"{args.prereg.name} SHA256 {digest(args.prereg)}"
    selection = dict(schema_version=1, stage="prospective_transfer_selection", candidate="weighted",
        baseline="weighted_dob", registration=registration,
        decision="fixed Panda-motivated allocation transfer; no weighted ALOHA outcome tuning",
        sigma=.1, sigma_units=calibration["prior_units"], gamma=.08, clip=.08,
        correction_coordinates=CORRECTION_COORDINATES, reference_feedback=False,
        reset_seeds=list(range(3000, 3020)), conditions=CONDITIONS, arms=ARMS)
    selection_bytes = serialized(selection)
    remote_selection = args.remote_root/"inputs/transfer_selection.json"
    remote_prereg = args.snapshot_root/REGISTRATION
    inputs = []
    for name in SNAPSHOT_FILES:
        path = source_root/"openpi"/name
        require(path.is_file(), f"missing frozen runtime/tool source: {path}")
        inputs.append(dict(role="code", local=str(path), remote=str(args.snapshot_root/"openpi"/name), sha256=digest(path)))
    inputs.append(dict(role="registration", local=str(args.prereg.resolve()), remote=str(remote_prereg), sha256=digest(args.prereg)))
    inputs.append(dict(role="selection", local=str(destination/"transfer_selection.json"), remote=str(remote_selection),
                       sha256=hashlib.sha256(selection_bytes).hexdigest()))
    for name, path in (("aloha_reference.json", args.reference_artifact), ("aloha_healthy_seed2600.json", args.healthy_log),
                       ("aloha_healthy_seed2600_provenance.json", args.healthy_provenance),
                       ("historical_sensitivity.json", args.sensitivity_artifact),
                       ("aloha_reference_bimanual_rejected.json", args.rejected_reference)):
        inputs.append(dict(role="calibration", local=str(path.resolve()), remote=str(args.remote_root/"inputs"/name), sha256=digest(path)))
    require(len({item["remote"] for item in inputs}) == len(inputs), "input paths alias")
    hashes = {item["remote"]: item["sha256"] for item in inputs}
    remote_reference = args.remote_root/"inputs/aloha_reference.json"
    expected_sources = [args.snapshot_root/"openpi"/name for name in RUNNER_FILES]+[remote_reference, remote_prereg, remote_selection]
    expected_hashes = {str(path): hashes[str(path)] for path in expected_sources}
    common = dict(COMMON, stage="confirm", seed=3000, episodes=20, task=law.TASK, suite=law.TASK,
        max_steps=300, state_tolerance=1e-9, dt=.02, horizon=10,
        fault_vec=None, gain=None, sev=0., policy_rng_pinned=False,
        reset_estimate=True, reset_covariance=True, composite_reference_used=False,
        reference_artifact=str(remote_reference), prereg=str(remote_prereg), selection_record=str(remote_selection))
    plan = dict(schema_version=1, study="aloha_weighted_transfer_confirmation", stage="confirmation", suite=law.TASK,
        pairing="aloha_seeded_reset", registration=registration, locked_before_outcomes=True,
        selection_record=str(remote_selection), expected_keys=EXPECTED_KEYS, forbidden_keys=FORBIDDEN_KEYS,
        cells=[], families=[dict(id=family, comparisons=[]) for family in FAMILIES])
    score = {key: value for key, value in plan.items() if key not in ("cells", "families")}
    score.update(runs=[], comparisons=[], family_sizes={family: 13 for family in FAMILIES}, shared_control_groups=[])
    jobs, collection = [], []
    for condition in CONDITIONS:
        joint = None if condition == "healthy" else int(condition[5:])
        torque = None if joint is None else float(EXPECTED_GAINS[joint]*.02)
        cell_args = dict(common, condition=condition, healthy=joint is None, joint=joint,
                         joint_fault=None if joint is None else f"torque:{joint}:{torque}")
        remote_output = args.remote_root/"cells"/condition
        remote_compact = args.remote_root/"compact"/condition
        argv = [str(args.python), str(args.snapshot_root/"openpi/run_aloha_joint_followup.py"),
            *( ["--healthy"] if joint is None else ["--joint", str(joint)] ),
            "--stage", "confirm", "--seed", "3000", "--episodes", "20", "--arms", ",".join(ARMS),
            "--reference-artifact", str(remote_reference), "--prereg", str(remote_prereg),
            "--selection-record", str(remote_selection), "--out-dir", str(remote_output),
            "--host", args.host, "--port", str(args.port), "--max-steps", "300", "--state-tolerance", "1e-9"]
        jobs.append(dict(id=condition, argv=argv, out_dir=str(remote_output),
                         log=str(args.remote_root/"logs"/f"{condition}.log"), expected_rollouts=60))
        collection.append(dict(id=condition, argv=[str(args.python), str(args.snapshot_root/"openpi/collect_followup_results.py"),
            "collect", str(remote_output/"study.json"), "--out-dir", str(remote_compact), "--hash-telemetry"],
            source_study=str(remote_output/"study.json"), local_compact_dir=str(destination/"compact"/condition)))
        plan["cells"].append(dict(id=condition, summary=f"compact/{condition}/summary.json", condition=condition,
            expected_args=cell_args, expected_arms=ARMS,
            expected_arm_args={name: dict(ARM_CONFIGS[name], corr_joints=",".join(map(str, CORRECTION_COORDINATES)),
                followup_arm=name, static_corr=None, freeze_after=None, warm_start=False, identify_episodes=None)
                for name in ARMS[1:]}))
        run_ids = []
        for name in ARMS[1:]:
            rid = f"{condition}_{name}"
            run_ids.append(rid)
            expected = dict(cell_args, **plan["cells"][-1]["expected_arm_args"][name])
            score["runs"].append(dict(id=rid, cell=condition, path=f"compact/{condition}/{name}.json",
                                      expected_args=expected, expected_source_hashes=expected_hashes))
        legacy, weighted = run_ids
        score["shared_control_groups"].append(run_ids)
        for family, left, right, left_ref, right_ref in (
            ("legacy_vs_off", "off", "legacy_allchannels", dict(run=legacy, arm="frozen_faulted"), dict(run=legacy, arm="adaptive")),
            ("weighted_vs_off", "off", "weighted", dict(run=weighted, arm="frozen_faulted"), dict(run=weighted, arm="adaptive")),
            ("weighted_vs_legacy", "legacy_allchannels", "weighted", dict(run=legacy, arm="adaptive"), dict(run=weighted, arm="adaptive"))):
            cid = f"{condition}_{family}"
            next(item for item in plan["families"] if item["id"] == family)["comparisons"].append(
                dict(id=cid, cell=condition, left=left, right=right))
            score["comparisons"].append(dict(id=cid, family=family, left=left_ref, right=right_ref))
    launch = dict(schema_version=1, stage="confirmation", registration=registration, locked_before_outcomes=True,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), concurrency=args.workers,
        allocation=dict(cells=13, arms=ARMS, scenarios_per_cell=20, total_rollouts=780),
        env=dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(args.egl_device), PYTHONUNBUFFERED="1",
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=":".join(map(str,
                (args.snapshot_root/"openpi", args.upstream_root/"packages/openpi-client/src")))),
        shared_policy_server=dict(host=args.host, port=args.port, policy_rng_pinned=False),
        snapshot_root=str(args.snapshot_root), remote_root=str(args.remote_root), frozen_inputs=inputs,
        calibration=calibration, selection=selection, family_sizes=score["family_sizes"], jobs=jobs,
        input_preflight_argv=[str(args.python), str(args.snapshot_root/"openpi/prepare_aloha_weighted_confirmation.py"),
                              "--verify-inputs", str(args.remote_root/"launch_manifest.json")],
        launch_argv=[str(args.python), str(args.snapshot_root/"openpi/launch_followup_grid.py"),
                     "--manifest", str(args.remote_root/"launch_manifest.json"),
                     "--status", str(args.remote_root/"status.json"), "--jobs", str(args.workers)])
    operations = dict(schema_version=1, expected_cells=13, expected_rollouts=780, expected_views=26,
        collection_jobs=collection, download=dict(remote=str(args.remote_root/"compact"), local=str(destination/"compact")),
        explicit_plan="collector_plan.json",
        collector_manifest_argv=["python3", str(source_root/"openpi/collect_followup_results.py"), "manifest",
            "--plan", str(destination/"collector_plan.json"), "--out-dir", str(destination/"collected_family_manifests")],
        score_argv=["python3", str(source_root/"openpi/score_joint_followup.py"), str(destination/"score_manifest.json"), "--json"],
        score_stdout=str(destination/"scores.json"), family_sizes=score["family_sizes"],
        required_checks=["all thirteen complete cells", "three arms and twenty reset seeds each",
            "full qpos/qvel pairing including cube", "matching shared controls and frozen source hashes",
            "all thirteen comparisons retained in every family"],
        retention="preserve authoritative source studies, complete telemetry, and rejected all-twelve reference")
    return {"transfer_selection.json": selection, "launch_manifest.json": launch,
            "collector_plan.json": plan, "collection_operations.json": operations, "score_manifest.json": score}


def verify_inputs(path):
    manifest = json.loads(Path(path).read_text())
    require(manifest.get("stage") == "confirmation" and manifest.get("locked_before_outcomes") is True,
            "expected a prospective confirmation manifest")
    for row in manifest["frozen_inputs"]:
        require(Path(row["remote"]).is_file() and digest(row["remote"]) == row["sha256"],
                f"missing or changed frozen input: {row['remote']}")
    for job in manifest["jobs"]:
        require(not Path(job["out_dir"]).exists(), f"confirmation outcome directory already exists: {job['out_dir']}")
    return dict(verified_inputs=len(manifest["frozen_inputs"]), prospective_cells=len(manifest["jobs"]))


def write_frozen_plan(args, artifacts):
    """Capture every planned input before writing a launchable plan.

    The original checkout may continue changing. Copy operations must read the
    captured paths in frozen_inputs, whose bytes are checked against the lock.
    """
    destination = args.out_dir.resolve()
    manifest = artifacts["launch_manifest.json"]
    captured = []
    for row in manifest["frozen_inputs"]:
        data = serialized(artifacts["transfer_selection.json"]) if row["role"] == "selection" else Path(row["local"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == row["sha256"],
                f"input changed while preparing the lock: {row['local']}")
        remote = Path(row["remote"])
        if row["role"] in ("code", "registration"):
            target = destination/"frozen_inputs/snapshot"/remote.relative_to(args.snapshot_root)
        else:
            target = destination/"frozen_inputs/inputs"/remote.name
        captured.append((row, target, data))
    require(len({target for _, target, _ in captured}) == len(captured), "captured input paths alias")
    destination.mkdir(parents=True, exist_ok=False)
    for row, target, data in captured:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        row["original_local"] = row["local"]
        row["local"] = str(target)
    for name, value in artifacts.items():
        with (destination/name).open("xb") as stream:
            stream.write(serialized(value))
    return dict(captured_inputs=len(captured), snapshot=str(destination/"frozen_inputs/snapshot"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-inputs", type=Path)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--snapshot-root", type=Path, default=BASE/"aloha_confirmation_repo")
    p.add_argument("--remote-root", type=Path, default=BASE/"runs/aloha_weighted_confirmation")
    p.add_argument("--out-dir", type=Path, default=ROOT/"results/composite_followup/aloha_weighted_confirmation_plan")
    p.add_argument("--prereg", type=Path, default=ROOT/REGISTRATION)
    p.add_argument("--registration")
    p.add_argument("--reference-artifact", type=Path, default=ROOT/"results/composite_followup/calibration/aloha_reference.json")
    p.add_argument("--healthy-log", type=Path, default=ROOT/"results/composite_followup/calibration/aloha_healthy_seed2600.json")
    p.add_argument("--healthy-provenance", type=Path, default=ROOT/"results/composite_followup/calibration/aloha_healthy_seed2600_provenance.json")
    p.add_argument("--sensitivity-artifact", type=Path, default=ROOT/"results/aloha/openloop.json")
    p.add_argument("--rejected-reference", type=Path, default=ROOT/"results/composite_followup/calibration/aloha_reference_bimanual.json")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--python", type=Path, default=Path("/data/fxxie/vla/envs/aloha/bin/python"))
    p.add_argument("--upstream-root", type=Path, default=Path("/data/fxxie/vla/openpi"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--egl-device", type=int, default=0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.verify_inputs:
        print(json.dumps(verify_inputs(args.verify_inputs)))
        return 0
    require(not args.out_dir.exists(), "output directory already exists; preserve the earlier prospective lock")
    artifacts = build(args)
    write_frozen_plan(args, artifacts)
    print(f"Prepared thirteen cells, 780 rollouts, three global families of thirteen. No launch: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
