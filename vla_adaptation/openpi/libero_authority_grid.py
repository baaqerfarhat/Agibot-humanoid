"""Fresh-calibration Panda same-state torque authority grid, without policy queries.

Replay one healthy command donor in disjoint initial states, then compare torque-on,
torque-off, exact opposing torque and bounded Cartesian corrections from one snapshot.
The healthy branch and local correction oracle are privileged diagnostics. They do not
certify global matching, contraction, task success, or a deployable observer.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import pathlib

import numpy as np

from joint_diagnostics import _json, analyze_local_probe, collect_local_probe
from libero_local_probe import Snapshot, advance, observed_pose, parity, pose
from libero_reset import physics_fingerprint, reset_libero


def save(path, payload):
    path.write_text(json.dumps(payload, default=_json, indent=2, allow_nan=False))


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-root", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal")
    ap.add_argument("--inits", default="27,28")
    ap.add_argument("--joints", default="0,1,2,3,4,5,6")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--source-episode", type=int, default=0)
    ap.add_argument("--checkpoints", default="0,20,60")
    ap.add_argument("--horizons", default="1,5")
    ap.add_argument("--torque", type=float, default=5.)
    ap.add_argument("--step", type=float, default=.02)
    ap.add_argument("--bound", type=float, default=.3)
    ap.add_argument("--tolerance", type=float, default=1e-8)
    ap.add_argument("--candidate-observer", choices=("legacy", "weighted"), default="legacy")
    ap.add_argument("--reference-manifest", type=pathlib.Path,
                    help="optional original grid; verify identical physical probes before comparing candidates")
    args = ap.parse_args()
    suites = args.suites.split(",")
    inits, joints, checkpoints, horizons = [sorted(set(map(int, value.split(","))))
        for value in (args.inits, args.joints, args.checkpoints, args.horizons)]
    if (min(inits) < 0 or min(checkpoints) < 0 or min(horizons) < 1
            or not set(joints) <= set(range(7)) or args.step <= 0 or args.bound <= 0):
        ap.error("invalid initial states, joints, checkpoints, horizons or probe bounds")
    if args.out.exists():
        ap.error("output directory exists; preserve earlier or incomplete probe grids")
    import main as lm
    from libero.libero import benchmark
    from adaptive_law import fit_plant, estimator_step, healthy_residual_covariance, OUT, K_FIR
    from openloop_id import nominal_commands
    from so3 import rot_delta
    code = pathlib.Path(__file__).resolve().parent
    source_paths = [code/name for name in ("libero_authority_grid.py", "libero_local_probe.py",
        "libero_reset.py", "joint_diagnostics.py", "adaptive_law.py", "openloop_id.py", "so3.py")]
    if args.candidate_observer == "weighted":
        from weighted_dob import weighted_dob_step
        source_paths.append(code/"weighted_dob.py")
    reference = None
    if args.reference_manifest is not None:
        reference = json.loads(args.reference_manifest.read_text())
        if reference.get("status") != "complete":
            raise ValueError("reference grid must be complete")
        reference = {(r["metadata"]["suite"], r["metadata"]["joint"], r["metadata"]["init"],
                      r["metadata"]["checkpoint"], r["metadata"]["horizon"]): r for r in reference["probes"]}
    args.out.mkdir(parents=True)
    manifest = dict(schema_version=1, status="running", args=vars(args), source_hashes=hashes(source_paths),
        reset_protocol="libero-reset-v1", metric="endpoint translation/.05m and rotation/.5rad",
        protocol="fixed future recorded commands; same full simulator/controller state for all branches",
        estimator=dict(law="legacy", gamma=.08, dead=.008, norm_r=.15, clip=.3,
                       norm_channels="all", bias=None, candidate_support="translation and all6"),
        suites=[], scenarios=[], probes=[],
        limitations=["The donor is a fresh healthy calibration episode, replayed in another initial state.",
                     "No historical trajectory parity is claimed and no policy server is contacted.",
                     "Healthy branch removes current torque from an already faulted checkpoint.",
                     "Each candidate correction is held fixed over the probe horizon; branch observers are not updated.",
                     "Local authority and privileged correction do not establish task success or global stability."])
    if args.candidate_observer == "weighted":
        manifest["estimator"] = dict(law="weighted_dob", gamma=.08, clip=.3, prior_std=.1,
            bias=None, covariance="healthy_residual_covariance from the same FIR data and fitted W",
            candidate_support="independent translation and full masks, allocations and residual filter states",
            replay_names=dict(candidate="weighted_translation", all_estimate_candidate="weighted_full"))
    if args.reference_manifest is not None:
        manifest["reference_hashes"] = hashes([args.reference_manifest])
    save(args.out/"manifest.json", manifest)
    try:
        for suite_name in suites:
            folder = args.calibration_root/suite_name
            fir_path, m_path = folder/"fir.json", folder/"M.json"
            fir, sensitivity = json.loads(fir_path.read_text()), json.loads(m_path.read_text())
            for artifact in (fir, sensitivity):
                if artifact.get("status") != "complete" or artifact.get("suite") != suite_name:
                    raise ValueError("complete matching-suite calibration artifacts required")
                if artifact.get("reset_protocol") != "libero-reset-v1":
                    raise ValueError("fresh scenario reset calibration required")
            for init in inits:
                if [args.task, init] in fir["calib_episodes"] + sensitivity["probe_episodes"]:
                    raise ValueError("authority evaluation must use disjoint calibration initial states")
            needed = max(checkpoints) + max(horizons)
            commands, donor = nominal_commands(fir, args.source_episode, needed)
            if len(commands) < needed or commands.shape[1] != 7:
                raise ValueError("donor must contain enough full seven-coordinate commands")
            W, M = fit_plant(fir_path), np.asarray(sensitivity["M"])
            inverse = np.linalg.pinv(M)
            calibration = dict(suite=suite_name, donor=donor, W=W, M=M,
                M_condition=float(np.linalg.cond(M)), calibration_hashes=hashes([fir_path, m_path]))
            R = covariance_report = None
            if args.candidate_observer == "weighted":
                R, covariance_report = healthy_residual_covariance(fir, W)
                calibration.update(R=R, covariance_calibration=covariance_report)
            manifest["suites"].append(calibration)
            suite = benchmark.get_benchmark_dict()[suite_name]()
            env, _ = lm._get_libero_env(suite.get_task(args.task), 32, 7)
            try:
                for init in inits:
                    reference_physics = None
                    for joint in joints:
                        _, provenance = reset_libero(env, suite.get_task_init_states(args.task)[init],
                                                     suite=suite_name, task=args.task, init=init)
                        rs = env.env
                        for name, observable in rs._observables.items():
                            if any(token in name for token in ("image", "depth", "segmentation")):
                                observable.set_enabled(False)
                        dof = rs.sim.model.get_joint_qvel_addr(rs.robots[0].robot_model.joints[joint])
                        for _ in range(10):
                            advance(rs, np.asarray(lm.LIBERO_DUMMY_ACTION), dof, 0.)
                        fingerprint = physics_fingerprint(env)
                        if reference_physics is None:
                            reference_physics = fingerprint
                        if fingerprint != reference_physics:
                            raise RuntimeError("joint branches did not start from identical warmup physics")
                        manifest["scenarios"].append(dict(suite=suite_name, task=args.task, init=init,
                            joint=joint, reset=provenance, warmup_physics=fingerprint))
                        history = collections.deque([np.zeros(6)]*(K_FIR+1), maxlen=K_FIR+1)
                        estimate = np.zeros(6)
                        masks = dict(translation=np.array([1., 1., 1., 0., 0., 0.]), full=np.ones(6))
                        weighted_estimates = {name: np.zeros(6) for name in masks}
                        weighted_states = {name: None for name in masks}
                        weighted_diagnostics = {name: None for name in masks}
                        for index in range(max(checkpoints)+1):
                            if index in checkpoints:
                                snapshot = Snapshot(rs)
                                check = parity(rs, commands[index], dof, args.torque, args.tolerance)
                                snapshot.restore()
                                x0, q0 = pose(rs)
                                for horizon in horizons:
                                    future = commands[index:index+horizon]
                                    def evaluate(delta, fault, opposite):
                                        snapshot.restore()
                                        torques = []
                                        for command in future:
                                            action = command.copy(); action[:6] += delta
                                            advance(rs, action, dof, fault, opposite)
                                            torques.append(rs.robots[0].torques.copy())
                                        x1, q1 = pose(rs)
                                        return dict(response=np.r_[x1-x0, rot_delta(q0, q1)]/OUT,
                                            full_state=rs.sim.get_state().flatten().copy(),
                                            ctrl=rs.sim.data.ctrl.copy(), clipped_torques=torques,
                                            snapshot_id=snapshot.identity)
                                    mask = np.array([1., 1., 1., 0., 0., 0.])
                                    translation_candidate, full_candidate = -estimate*mask, -estimate
                                    if args.candidate_observer == "weighted":
                                        translation_candidate = -weighted_estimates["translation"]
                                        full_candidate = -weighted_estimates["full"]
                                    metadata = dict(suite=suite_name, task=args.task, init=init, joint=joint,
                                        checkpoint=index, horizon=horizon, torque=args.torque,
                                        full_step_parity=check, estimate_before=estimate.copy(),
                                        endpoint_scale=OUT, donor=donor)
                                    if args.candidate_observer == "weighted":
                                        metadata["legacy_estimate_before"] = metadata.pop("estimate_before")
                                        metadata.update(candidate_observer="weighted",
                                            weighted_estimates_before={name: value.copy() for name, value in weighted_estimates.items()},
                                            weighted_diagnostics=copy.deepcopy(weighted_diagnostics))
                                    raw = collect_local_probe(evaluate, step=args.step, torque=args.torque,
                                        bound=args.bound, candidate=translation_candidate, metadata=metadata)
                                    report = analyze_local_probe(raw)
                                    translation = np.zeros(6)
                                    translation[:3] = report["subsets"]["translation"]["bounded_correction"]
                                    for key, correction in (("translation_oracle", translation),
                                                            ("all_estimate_candidate", full_candidate)):
                                        evaluation = evaluate(correction, args.torque, 0.)
                                        raw[key] = dict(action=correction, evaluation=evaluation)
                                        remaining = float(np.linalg.norm(evaluation["response"]-raw["healthy"]["response"]))
                                        dnorm = report["all_probed_inputs"]["disturbance_norm"]
                                        report["replays"][key] = dict(action=correction, remaining_norm=remaining,
                                            error_ratio=remaining/dnorm if dnorm else None)
                                    if args.candidate_observer == "weighted":
                                        factor = np.linalg.cholesky(R)
                                        disturbance = raw["faulted"]["response"]-raw["healthy"]["response"]
                                        weighted_norm = float(np.linalg.norm(np.linalg.solve(factor, disturbance)))
                                        for key in ("candidate", "all_estimate_candidate"):
                                            remaining = raw[key]["evaluation"]["response"]-raw["healthy"]["response"]
                                            weighted_remaining = float(np.linalg.norm(np.linalg.solve(factor, remaining)))
                                            report["replays"][key].update(physical_R_weighted_remaining_norm=weighted_remaining,
                                                physical_R_weighted_error_ratio=weighted_remaining/weighted_norm if weighted_norm else None)
                                    full_checks = {}
                                    for name, a, b in (("repeat", "faulted_repeat", "faulted"),
                                                       ("opposing_torque", "torque_oracle", "healthy")):
                                        for field in ("full_state", "ctrl"):
                                            full_checks[name+"_"+field+"_max"] = float(np.max(
                                                np.abs(np.asarray(raw[a][field])-np.asarray(raw[b][field]))))
                                    report["full_state_checks"] = full_checks
                                    if max(full_checks.values()) > args.tolerance:
                                        raise RuntimeError(f"full-state restore or torque cancellation failed: {full_checks}")
                                    stem = f"{suite_name}_j{joint}_i{init}_s{index}_h{horizon}"
                                    if reference is not None:
                                        ref = reference[(suite_name, joint, init, index, horizon)]
                                        old_raw = json.loads((args.reference_manifest.parent/(stem+"_raw.json")).read_text())
                                        reference_checks = {name+"_"+field+"_max": float(np.max(np.abs(
                                            np.asarray(raw[name][field])-np.asarray(old_raw[name][field]))))
                                            for name in ("healthy", "faulted") for field in ("full_state", "ctrl")}
                                        reference_checks["response_map_max"] = float(np.max(np.abs(
                                            np.asarray(report["response_map"])-np.asarray(ref["response_map"]))))
                                        if max(reference_checks.values()) > args.tolerance:
                                            raise RuntimeError(f"original grid physical comparison failed: {reference_checks}")
                                        report.update(reference_physics_checks=reference_checks,
                                                      reference_replays=ref["replays"])
                                    save(args.out/(stem+"_raw.json"), raw)
                                    save(args.out/(stem+"_summary.json"), report)
                                    manifest["probes"].append(report)
                                snapshot.restore()
                            before_x, before_q = observed_pose(rs)
                            advance(rs, commands[index], dof, args.torque)
                            after_x, after_q = observed_pose(rs)
                            history.appendleft(commands[index, :6])
                            H = np.asarray(history)
                            prediction = np.array([W[j, :K_FIR+1]@H[:, j]+W[j, -1] for j in range(6)])
                            measured = np.r_[after_x-before_x, rot_delta(before_q, after_q)]/OUT
                            residual = measured-prediction
                            estimate, _ = estimator_step(estimate, residual, inverse,
                                gamma=.08, dead=.008, norm_r=.15, clip=.3, bias=None,
                                mask=np.array([1., 1., 1., 0., 0., 0.]), M=M)
                            if args.candidate_observer == "weighted":
                                for name, active in masks.items():
                                    weighted_estimates[name], diag = weighted_dob_step(
                                        weighted_estimates[name], residual, M, mask=active, R=R,
                                        gamma=.08, clip=.3, state=weighted_states[name], prior_std=.1, bias=None)
                                    weighted_states[name] = diag["estimator_state"]
                                    weighted_diagnostics[name] = diag
                        save(args.out/"manifest.json", manifest)
                        print(json.dumps(dict(suite=suite_name, init=init, joint=joint,
                                              complete_probes=len(manifest["probes"]))), flush=True)
            finally:
                env.sim.data.qfrc_applied[:] = 0.
                env.close()
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        save(args.out/"manifest.json", manifest)


if __name__ == "__main__":
    main()
