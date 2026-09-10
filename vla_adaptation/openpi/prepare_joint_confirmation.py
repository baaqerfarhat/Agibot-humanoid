"""Prepare the prospective 24-cell Panda confirmation allocation; never launch it.

Select a candidate explicitly after development. Validate the three fresh healthy
calibrations, then write argv-based rollout/collection plans and one multi-suite
scoring manifest with global comparison families. Remote paths are declarations;
--verify-inputs runs on the destination filesystem before launch to verify every
frozen source/calibration/selection byte and reject existing outcome directories.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from openloop_id import nominal_commands
from run_joint_followup import ARM_CONFIGS, COMMON


ROOT = Path(__file__).resolve().parent.parent
SUITES = ("libero_spatial", "libero_object", "libero_goal")
CAPS = dict(libero_spatial=220, libero_object=280, libero_goal=300)
EXPECTED_KEYS = [[task, initial] for initial in (35, 36) for task in range(10)]
FORBIDDEN_KEYS = [[task, initial] for initial in (8, 20, 25, 26, 27, 28, 30, 31, 32, 45, 46, 47, 48)
                  for task in range(10)]
FAMILY_IDS = ("legacy_vs_off", "candidate_vs_off", "candidate_vs_legacy")
RUNNER_SOURCES = ("run_joint_followup.py", "adaptive_law.py", "joint_fault.py", "paired_probe.py", "libero_reset.py")
PREREG = "prereg_records/PREREG_COMPOSITE_JOINT_FOLLOWUP.md"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_calibration(root, suite):
    """Prove M links to this complete FIR and to one intact healthy command episode."""
    fir_path, matrix_path = Path(root)/suite/"fir.json", Path(root)/suite/"M.json"
    fir, matrix = json.loads(fir_path.read_text()), json.loads(matrix_path.read_text())
    for label, record in (("FIR", fir), ("M", matrix)):
        require(isinstance(record, dict) and record.get("status") == "complete", f"{suite}: {label} is incomplete")
        require(record.get("suite") == suite and record.get("reset_protocol") == "libero-reset-v1",
                f"{suite}: {label} suite or reset protocol mismatch")
    expected_calibration = [[task, 26] for task in range(10)]
    require(fir.get("calib_episodes") == expected_calibration,
            f"{suite}: FIR must use exactly tasks 0–9 at initial state 26")
    records = fir.get("records")
    require(isinstance(records, list) and len(records) == 1 and records[0].get("sev") == 0,
            f"{suite}: expected one healthy FIR record")
    nominal = records[0]
    require(nominal.get("episode_keys") == expected_calibration, f"{suite}: FIR episode keys disagree")
    lengths = nominal.get("ep_len")
    require(isinstance(lengths, list) and len(lengths) == 10
            and all(type(length) is int and length >= 7 for length in lengths), f"{suite}: invalid FIR episode boundaries")
    count = sum(lengths)
    require(nominal.get("n_steps") == count, f"{suite}: FIR step count disagrees with episode boundaries")
    require(isinstance(nominal.get("per_ep"), list) and
            [[row.get("task"), row.get("init")] for row in nominal["per_ep"]] == expected_calibration
            and [row.get("steps") for row in nominal["per_ep"]] == lengths,
            f"{suite}: FIR per-episode provenance disagrees with command boundaries")
    arrays = {name: np.asarray(nominal.get(name), float) for name in
              ("raw_a", "raw_d", "raw_exec", "raw_cmd", "raw_executed")}
    for name, value in arrays.items():
        require(value.shape == (count, 7 if name in ("raw_cmd", "raw_executed") else 6)
                and np.isfinite(value).all(), f"{suite}: malformed {name} or episode boundaries")
    require(np.array_equal(arrays["raw_a"], arrays["raw_cmd"][:, :6])
            and np.array_equal(arrays["raw_exec"], arrays["raw_a"])
            and np.array_equal(arrays["raw_executed"], arrays["raw_cmd"]), f"{suite}: healthy commands/executions disagree")
    require(matrix.get("probe_episodes") == [[0, 25]] and matrix.get("probe_task") == 0
            and matrix.get("probe_init") == 25, f"{suite}: M must probe task 0 at initial state 25")
    require(matrix.get("source_calib_episodes") == expected_calibration,
            f"{suite}: M source calibration scenarios differ from FIR")
    require(matrix.get("probe") == .02, f"{suite}: M must use the declared central probe magnitude .02")
    M = np.asarray(matrix.get("M"), float)
    require(M.shape == (6, 6) and np.isfinite(M).all(), f"{suite}: malformed M")
    source = matrix.get("command_source", {})
    require(type(source.get("episode_index")) is int and type(source.get("requested_steps")) is int,
            f"{suite}: M command-source provenance missing")
    _, recomputed = nominal_commands(fir, source["episode_index"], source["requested_steps"])
    require(source == recomputed and source["gripper"] == "recorded",
            f"{suite}: M command hash, gripper, or episode boundary mismatch")
    require(source["scenario"] in expected_calibration, f"{suite}: unknown M command donor")
    original_log = matrix.get("args", {}).get("log")
    require(isinstance(original_log, str) and matrix.get("source_hashes", {}).get(original_log) == digest(fir_path),
            f"{suite}: M source hash does not match this FIR artifact")
    used = {tuple(key) for key in expected_calibration + matrix["probe_episodes"] + [source["scenario"]]}
    require(not used.intersection(map(tuple, EXPECTED_KEYS)), f"{suite}: calibration/evaluation overlap")
    return dict(suite=suite, fir=dict(path=str(fir_path.resolve()), sha256=digest(fir_path)),
                M=dict(path=str(matrix_path.resolve()), sha256=digest(matrix_path)),
                fir_episodes=expected_calibration, probe_episodes=matrix["probe_episodes"], command_source=source)


def build(args):
    candidate = args.candidate
    require(candidate in ARM_CONFIGS and ARM_CONFIGS[candidate]["adapt"] and candidate != "legacy_translation",
            "candidate must name a known adapting arm distinct from legacy_translation")
    require(args.workers in (4, 6), "concurrency must be four or six jobs")
    require(args.registration and args.registration.strip(), "registration must identify the prospective lock")
    require(args.selection_record.is_file() and args.selection_record.read_text().strip(), "nonempty selection record required")
    try:
        selection = json.loads(args.selection_record.read_text())
    except json.JSONDecodeError:
        selection = None
    if isinstance(selection, dict) and "candidate" in selection:
        require(selection["candidate"] == candidate, "candidate disagrees with the selection record")
    for path in (args.snapshot_root, args.remote_root, args.upstream_root, args.python, args.control, args.ack):
        require(path.is_absolute(), "remote runtime paths must be absolute")
    require(args.snapshot_root != args.remote_root, "frozen code snapshot and output root must differ")
    require(0 < args.port < 65536 and args.control != args.ack, "policy endpoint/control paths are invalid")
    source_root, destination = args.source_root.resolve(), args.out_dir.resolve()
    require((source_root/PREREG).is_file(), "source preregistration is missing")
    calibration = {suite: validate_calibration(args.calibration_root, suite) for suite in SUITES}
    code_paths = sorted((source_root/"openpi").rglob("*.py")) + [source_root/PREREG]
    for name in (*RUNNER_SOURCES, "weighted_dob.py", "score_joint_followup.py", "collect_followup_results.py", "prepare_joint_confirmation.py"):
        require((source_root/"openpi"/name).is_file(), f"frozen source is missing {name}")
    inputs = [dict(role="code", local=str(path), remote=str(args.snapshot_root/path.relative_to(source_root)),
                   sha256=digest(path)) for path in code_paths]
    remote_selection = args.remote_root/"inputs"/args.selection_record.name
    inputs.append(dict(role="selection", local=str(args.selection_record.resolve()), remote=str(remote_selection),
                       sha256=digest(args.selection_record)))
    for suite in SUITES:
        for name, filename in (("fir", "fir.json"), ("M", "M.json")):
            record = calibration[suite][name]
            inputs.append(dict(role="calibration", local=record["path"],
                remote=str(args.remote_root/"calibration"/suite/filename), sha256=record["sha256"]))
    require(len({row["remote"] for row in inputs}) == len(inputs), "frozen input paths alias one another")
    hashes = {row["remote"]: row["sha256"] for row in inputs}
    environment = dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(args.egl_device), PYTHONUNBUFFERED="1",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        PYTHONPATH=":".join(map(str, (args.snapshot_root/"openpi", args.upstream_root/"third_party/libero",
                                      args.upstream_root/"examples/libero"))))
    arms = ["off", "legacy_translation", candidate]
    score = dict(schema_version=1, study="panda_three_benchmark_joint_confirmation", stage="confirmation",
        suites=list(SUITES), pairing="libero_set_init_state", expected_keys=EXPECTED_KEYS,
        forbidden_keys=FORBIDDEN_KEYS, registration=args.registration, locked_before_outcomes=True,
        selection_record=str(remote_selection), family_sizes={family: 24 for family in FAMILY_IDS},
        shared_control_groups=[], runs=[], comparisons=[])
    jobs, collections = [], []
    for condition in ("healthy", *[f"joint{joint}" for joint in range(7)]):
        for suite in SUITES:
            joint = 0 if condition == "healthy" else int(condition[5:])
            torque = 0. if condition == "healthy" else 5.
            cell = f"{suite}_{condition}"
            remote_output = args.remote_root/"cells"/suite/condition
            remote_compact = args.remote_root/"compact"/suite/condition
            fir = args.remote_root/"calibration"/suite/"fir.json"
            M = args.remote_root/"calibration"/suite/"M.json"
            argv = [str(args.python), str(args.snapshot_root/"openpi/run_joint_followup.py"),
                "--stage", "confirm", "--suite", suite, "--joint", str(joint), "--torque", str(torque),
                "--eval-init", "35", "--episodes", "20", "--arms", ",".join(arms),
                "--max-steps", str(CAPS[suite]), "--state-tolerance", "1e-9", "--replan-steps", "5",
                "--log", str(fir), "--openloop", str(M), "--selection-record", str(remote_selection),
                "--control", str(args.control), "--ack", str(args.ack), "--host", args.host,
                "--port", str(args.port), "--out-dir", str(remote_output)]
            jobs.append(dict(id=cell, suite=suite, condition=condition, argv=argv,
                log=str(args.remote_root/"logs"/f"{cell}.log"), out_dir=str(remote_output), expected_rollouts=60,
                expected_arms=arms, expected_keys=EXPECTED_KEYS))
            collections.append(dict(id=cell, argv=[str(args.python), str(args.snapshot_root/"openpi/collect_followup_results.py"),
                "collect", str(remote_output/"study.json"), "--out-dir", str(remote_compact), "--hash-telemetry"],
                source_study=str(remote_output/"study.json"),
                expected_compact_files=[str(remote_compact/name) for name in ("summary.json", "legacy_translation.json", f"{candidate}.json")],
                local_compact_dir=str(destination/"compact"/suite/condition)))
            run_ids = []
            expected_source_paths = [args.snapshot_root/"openpi"/name for name in RUNNER_SOURCES]
            expected_source_paths.extend((args.snapshot_root/PREREG, fir, M, remote_selection))
            if ARM_CONFIGS[candidate]["baseline"] == "weighted_dob":
                expected_source_paths.append(args.snapshot_root/"openpi/weighted_dob.py")
            for name in arms[1:]:
                config = ARM_CONFIGS[name]
                rid = f"{cell}_{name}"
                run_ids.append(rid)
                expected = dict(COMMON, stage="confirm", suite=suite, episodes=20, eval_init=35,
                    joint=joint, torque=torque, joint_fault=f"torque:{joint}:{torque}",
                    reset_protocol="libero-reset-v1", policy_rng_pinned=False,
                    max_steps=CAPS[suite], replan_steps=5, state_tolerance=1e-9,
                    baseline=config["baseline"], corr_dims=",".join(map(str, config["corr_dims"])),
                    correction_scale=config["correction_scale"], freeze_after=config["freeze_after"],
                    followup_arm=name, estimate_only=False, static_corr=None,
                    log=str(fir), openloop=str(M), selection_record=str(remote_selection))
                score["runs"].append(dict(id=rid, cell=cell, suite=suite,
                    path=f"compact/{suite}/{condition}/{name}.json", expected_args=expected,
                    expected_source_hashes={str(path): hashes[str(path)] for path in expected_source_paths}))
            legacy, selected = run_ids
            score["shared_control_groups"].append(run_ids)
            for family, left, right in (
                ("legacy_vs_off", dict(run=legacy, arm="frozen_faulted"), dict(run=legacy, arm="adaptive")),
                ("candidate_vs_off", dict(run=selected, arm="frozen_faulted"), dict(run=selected, arm="adaptive")),
                ("candidate_vs_legacy", dict(run=legacy, arm="adaptive"), dict(run=selected, arm="adaptive"))):
                score["comparisons"].append(dict(id=f"{cell}_{family}", family=family, left=left, right=right))
    aggregates = [dict(id="primary_joint5_candidate_vs_legacy", role="prespecified_primary",
        comparison_ids=[f"{suite}_joint5_candidate_vs_legacy" for suite in SUITES],
        family="primary_joint5_candidate_vs_legacy", family_tests=1, expected_scenarios=60,
        scenario_identity=["suite", "task", "init"], analysis="sum disjoint within-suite paired discordances; exact two-sided McNemar",
        interpretation="specific joint-5 harm-repair contrast; report every healthy/joint cell alongside it"),
        dict(id="secondary_joint5_candidate_vs_off", role="descriptive_secondary",
        comparison_ids=[f"{suite}_joint5_candidate_vs_off" for suite in SUITES], expected_scenarios=60,
        scenario_identity=["suite", "task", "init"], analysis="pool disjoint within-suite paired counts; no special primary significance claim")]
    score["planned_aggregates"] = aggregates
    launch = dict(schema_version=1, stage="confirmation", candidate=candidate,
        registration=args.registration, locked_before_outcomes=True,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), concurrency=args.workers,
        env=environment, shared_policy_server=dict(host=args.host, port=args.port,
            control=str(args.control), ack=str(args.ack), handshake="serialized by Probe.control file lock"),
        allocation=dict(suites=list(SUITES), cells=24, arms=arms, scenarios_per_cell=20, total_rollouts=1440),
        snapshot_root=str(args.snapshot_root), remote_root=str(args.remote_root), frozen_inputs=inputs,
        calibration=calibration, expected_keys=EXPECTED_KEYS, forbidden_keys=FORBIDDEN_KEYS,
        family_sizes=score["family_sizes"], planned_aggregates=aggregates, jobs=jobs,
        input_preflight_argv=[str(args.python), str(args.snapshot_root/"openpi/prepare_joint_confirmation.py"),
                              "--verify-inputs", str(args.remote_root/"launch_manifest.json")])
    collector = dict(schema_version=1, stage="confirmation", registration=args.registration,
        expected_cells=24, expected_rollouts=1440, expected_views=48, collection_jobs=collections,
        required_checks=["all 24 source studies complete", "all three arms and all 20 scenario keys per cell",
            "full physical fingerprint paired in every episode", "shared off views agree",
            "calibration/code/selection source hashes match frozen inputs", "all global families retain 24 comparisons"],
        score_manifest="score_manifest.json", family_sizes=score["family_sizes"], planned_aggregates=aggregates,
        score_argv=["python3", str(source_root/"openpi/score_joint_followup.py"), str(destination/"score_manifest.json"), "--json"],
        score_stdout=str(destination/"scores.json"),
        download=dict(remote=str(args.remote_root/"compact"), local=str(destination/"compact")),
        retention="keep authoritative studies and full telemetry remotely; compact records retain hashes",
        aggregate_note="Compute only after all 24 cells validate; pool within-suite paired discordances with suite-prefixed identities, never cross-suite pair matching.")
    return {"launch_manifest.json": launch, "score_manifest.json": score, "collector_plan.json": collector}


def verify_inputs(path):
    manifest = json.loads(Path(path).read_text())
    require(manifest.get("stage") == "confirmation" and manifest.get("locked_before_outcomes") is True,
            "expected a locked prospective confirmation manifest")
    for record in manifest["frozen_inputs"]:
        require(Path(record["remote"]).is_file() and digest(record["remote"]) == record["sha256"],
                f"missing or changed frozen input: {record['remote']}")
    for job in manifest["jobs"]:
        require(not Path(job["out_dir"]).exists(), f"confirmation output already exists: {job['out_dir']}")
    return dict(verified_inputs=len(manifest["frozen_inputs"]), prospective_cells=len(manifest["jobs"]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-inputs", type=Path)
    p.add_argument("--candidate", choices=[name for name, config in ARM_CONFIGS.items()
                   if config["adapt"] and name != "legacy_translation"])
    p.add_argument("--calibration-root", type=Path)
    p.add_argument("--snapshot-root", type=Path)
    p.add_argument("--remote-root", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--selection-record", type=Path)
    p.add_argument("--registration")
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--workers", type=int, choices=(4, 6), default=4)
    p.add_argument("--upstream-root", type=Path, default=Path("/data/fxxie/vla/openpi"))
    p.add_argument("--python", type=Path, default=Path("/data/fxxie/vla/envs/libero/bin/python"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--control", type=Path, default=Path("/tmp/ctl.json"))
    p.add_argument("--ack", type=Path, default=Path("/tmp/ack.json"))
    p.add_argument("--egl-device", type=int, default=0)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.verify_inputs is not None:
        print(json.dumps(verify_inputs(args.verify_inputs)))
        return 0
    for name in ("candidate", "calibration_root", "snapshot_root", "remote_root", "out_dir", "selection_record", "registration"):
        if getattr(args, name) is None:
            p.error("--"+name.replace("_", "-")+" is required for preparation")
    if args.out_dir.exists():
        p.error("output directory already exists; preserve the earlier prospective lock")
    artifacts = build(args)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in artifacts.items():
        with (args.out_dir/name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"Prepared 24 cells, 1440 rollouts, 72 per-cell contrasts; no rollouts launched. Files: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
