"""Compare frozen dynamic/FIR measurements on every completed ALOHA trajectory.

No controller is rerun. Fault truth enters only diagnostic errors/identities.
The affine model and FIR coefficients are frozen in the telemetry header; the
dynamic noise covariance uses the six declared healthy fitting episodes only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def array(value):
    return np.asarray(value, dtype=float)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def moments(values):
    values = array(values)
    return dict(samples=len(values), mean=values.mean(0),
                rms=np.sqrt(np.mean(values ** 2, axis=0)),
                rms_norm=float(np.sqrt(np.mean(np.sum(values ** 2, axis=1)))),
                max_norm=float(np.max(np.linalg.norm(values, axis=1))))


def summarize(rows):
    metrics = {key: moments([row[key] for row in rows]) for key in rows[0] if key != "t"}
    dyn = metrics["dynamic_estimate_error"]["rms_norm"]
    fir = metrics["fir_equivalent_error"]["rms_norm"]
    metrics["dynamic_to_fir_error_rms_ratio"] = dyn / fir if fir else None
    metrics["fraction_dynamic_error_norm_smaller"] = float(np.mean([
        np.linalg.norm(row["dynamic_estimate_error"]) < np.linalg.norm(row["fir_equivalent_error"])
        for row in rows]))
    return metrics


def lag_one_correlation(episodes):
    before = np.concatenate([item[:-1] for item in episodes if len(item) > 1])
    after = np.concatenate([item[1:] for item in episodes if len(item) > 1])
    return [float(np.corrcoef(before[:, j], after[:, j])[0, 1])
            if np.std(before[:, j]) and np.std(after[:, j]) else None
            for j in range(before.shape[1])]


def calibration_report(episodes, model, observer):
    A, B, offset = (array(model[key]) for key in ("A", "B", "offset"))
    B_inv = np.linalg.inv(B)
    M_inv = np.linalg.inv(array(observer["M"]))
    W = array(observer["W"])
    lag = W.shape[1] - 2
    dynamic, fir, paired_dynamic = [], [], []
    for episode in episodes:
        q, before, command = (array(episode[key]) for key in ("q", "q_before", "u"))
        if not (q.shape == before.shape == command.shape) or q.shape[1] != 14:
            raise ValueError("expected finite, aligned 14-joint healthy q/q_before/u")
        if not all(np.all(np.isfinite(item)) for item in (q, before, command)):
            raise ValueError("nonfinite healthy sample")
        residual = q[:, :6] - before[:, :6] @ A.T - command[:, :6] @ B.T - offset
        dynamic.append(residual)
        fir_residual = np.asarray([
            q[t] - np.sum(W[:, :-1] * command[t-lag:t+1][::-1].T, axis=1) - W[:, -1]
            for t in range(lag, len(q))])
        fir.append((fir_residual @ M_inv.T)[:, :6])
        paired_dynamic.append(residual[lag:] @ B_inv.T)
    train = np.concatenate(dynamic[:6])
    sample_cov = np.cov(train, rowvar=False, ddof=1)
    eigenvalues, vectors = np.linalg.eigh(sample_cov)
    floor = max(1e-12, float(eigenvalues[-1]) * 1e-9)
    R_dynamic = (vectors * np.maximum(eigenvalues, floor)) @ vectors.T
    R_dynamic = (R_dynamic + R_dynamic.T) * 0.5
    action_cov = B_inv @ R_dynamic @ B_inv.T
    R_fir = array(observer["R"])
    fir_action_cov = (M_inv @ R_fir @ M_inv.T)[:6, :6]
    split_report = {}
    for label, indices in (("fit", list(range(6))), ("previous_reference_validation", list(range(6, 10)))):
        dyn_eps = [dynamic[index] @ B_inv.T for index in indices]
        fir_eps = [fir[index] for index in indices]
        paired = [paired_dynamic[index] for index in indices]
        split_report[label] = dict(episode_indices=indices,
            dynamic_measurement_residual=moments(np.concatenate([dynamic[index] for index in indices])),
            dynamic_equivalent_action_noise=moments(np.concatenate(dyn_eps)),
            dynamic_equivalent_action_noise_t_ge_fir_lag=moments(np.concatenate(paired)),
            fir_equivalent_action_noise_t_ge_fir_lag=moments(np.concatenate(fir_eps)),
            dynamic_action_noise_lag1_correlation=lag_one_correlation(dyn_eps),
            fir_action_noise_lag1_correlation=lag_one_correlation(fir_eps))
    return dict(fit_episode_indices=list(range(6)), covariance_fit_samples=len(train),
        covariance_centering="sample covariance is centered; the measured residual/estimate is NOT bias-subtracted",
        dynamic_residual_mean=train.mean(0), dynamic_sample_covariance=sample_cov,
        covariance_eigenvalue_floor=floor, dynamic_R=R_dynamic,
        dynamic_R_eigenvalues=np.linalg.eigvalsh(R_dynamic),
        dynamic_R_condition=float(np.linalg.cond(R_dynamic)),
        dynamic_equivalent_action_covariance=action_cov,
        dynamic_equivalent_action_std=np.sqrt(np.diag(action_cov)),
        fir_equivalent_action_covariance=fir_action_cov,
        fir_equivalent_action_std=np.sqrt(np.diag(fir_action_cov)),
        splits=split_report, fir_lag=lag,
        validation_status="episodes6-9 were already used for the original .005-rad reference gate; this is not fresh qualification")


def analyze(telemetry, summary_path, calibration_log):
    summary = json.loads(Path(summary_path).read_text())
    episodes = json.loads(Path(calibration_log).read_text())
    if summary.get("status") != "complete" or not summary.get("scoreable"):
        raise ValueError("a completed scoreable summary is required")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise ValueError("expected the frozen ten-episode healthy calibration log")
    known = {(condition, name, ep["init"]): ep["ok"]
             for condition, cell in summary["cells"].items()
             for name, arm in cell["arms"].items() for ep in arm["per_ep"]}
    pooled, condition_pooled = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    per_episode, seen = [], set()
    header, current, rows = None, None, []
    digest = hashlib.sha256()
    integrity = dict(command_identity_max_error=0., dynamic_truth_identity_max_error=0.)

    def finish(key, selected_rows):
        if key is None:
            return
        if key not in known or key in seen:
            raise ValueError("unexpected or repeated episode")
        seen.add(key)
        windows = dict(first_1=selected_rows[:1], first_5=selected_rows[:5],
            first_20=selected_rows[:20], steps_20_100=selected_rows[20:100],
            steps_100_end=selected_rows[100:], last_100=selected_rows[-100:])
        record = dict(condition=key[0], arm=key[1], episode=key[2],
                      success=known[key], steps=len(selected_rows), windows={})
        for window, values in windows.items():
            if not values:
                continue
            pooled[key[:2]][window].extend(values)
            condition_pooled[key[0]][window].extend(values)
            stats = summarize(values)
            record["windows"][window] = {metric: stats[metric] for metric in (
                "dynamic_estimate_error", "fir_equivalent_error", "recorded_observer_error",
                "dynamic_to_fir_error_rms_ratio", "fraction_dynamic_error_norm_smaller")}
        per_episode.append(record)

    with Path(telemetry).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row.get("type") == "header":
                if header is not None:
                    raise ValueError("multiple telemetry headers")
                header = row
                model, observer = row["config"]["reference_model"], row["config"]["observer"]
                if model["state_indices"] != list(range(6)) or model["command_indices"] != list(range(6)):
                    raise ValueError("requires the frozen left-six position reference")
                A, B, offset = (array(model[key]) for key in ("A", "B", "offset"))
                if np.linalg.matrix_rank(B) != 6:
                    raise ValueError("dynamic map is not invertible")
                B_inv, M_inv = np.linalg.inv(B), np.linalg.inv(array(observer["M"]))
                continue
            if row.get("type") != "step":
                continue
            if header is None:
                raise ValueError("missing telemetry header")
            condition, name = row["arm"].split("/", 1)
            key = condition, name, row["episode"]
            if key != current:
                finish(current, rows)
                current, rows = key, []
            if row["t"] != len(rows):
                raise ValueError("missing or reordered steps")
            q, before = array(row["measured"]), array(row["joint_before"])
            nominal, command, truth = (array(row[key]) for key in ("nominal_command", "command", "f_true"))
            residual = q[:6] - A @ before[:6] - B @ nominal[:6] - offset
            dynamic = B_inv @ residual
            fir = (M_inv @ array(row["r"]))[:6]
            error = dynamic - truth[:6]
            model_error = q[:6] - A @ before[:6] - B @ command[:6] - offset
            integrity["command_identity_max_error"] = max(integrity["command_identity_max_error"],
                float(np.max(np.abs(command - nominal - truth))))
            integrity["dynamic_truth_identity_max_error"] = max(integrity["dynamic_truth_identity_max_error"],
                float(np.max(np.abs(error - B_inv @ model_error))))
            values = dict(t=row["t"], dynamic_measurement=residual,
                dynamic_equivalent_estimate=dynamic, fir_equivalent_estimate=fir,
                dynamic_estimate_error=error, fir_equivalent_error=fir-truth[:6],
                recorded_observer_error=array(row["f_hat"])[:6]-truth[:6],
                actual_fault=truth[:6], executed_net_offset=array(row["correction"])[:6]+truth[:6])
            if not all(np.all(np.isfinite(value)) for value in values.values()):
                raise ValueError("nonfinite diagnostic data")
            rows.append(values)
    finish(current, rows)
    if seen != set(known):
        raise ValueError("missing declared episodes")
    if digest.hexdigest() != summary["source_telemetry"]["sha256"]:
        raise ValueError("telemetry hash differs from frozen completed collection")
    if max(integrity.values()) > 1e-10:
        raise ValueError("measurement arithmetic failed: " + str(integrity))
    singular = np.linalg.svd(B, compute_uv=False)
    return dict(schema_version=1, status="complete_diagnostics", study_id=summary["study_id"],
        source_study=summary["source_study"], source_summary=fingerprint(summary_path),
        source_telemetry=dict(path=str(Path(telemetry).resolve()), sha256=digest.hexdigest(), bytes=Path(telemetry).stat().st_size),
        source_calibration_log=fingerprint(calibration_log), source_script=fingerprint(__file__),
        args=header["args"], reference_model=model, original_qualification=header["config"]["qualification"],
        formulas=dict(dynamic_residual="q_after - A @ q_before - B @ nominal_corrected_command - offset",
            dynamic_equivalent="solve(B, dynamic_residual)", fir_equivalent="solve(M, recorded_FIR_residual)[:6]",
            diagnostic_error="equivalent_estimate - recorded_actual_fault[:6]",
            timing="post-step measurement; any new causal controller would apply its estimate from the next step"),
        B_diagnostics=dict(rank=int(np.linalg.matrix_rank(B)), singular_values=singular,
            condition=float(np.linalg.cond(B)), inverse_spectral_norm=float(np.linalg.norm(B_inv, 2))),
        calibration=calibration_report(episodes, model, observer), arithmetic_integrity=integrity,
        windows=dict(first_1="t=0", first_5="0<=t<5", first_20="0<=t<20", steps_20_100="20<=t<100",
            steps_100_end="100<=t<episode length", last_100="last min(100,length); windows overlap"),
        aggregation="Pooled step-weighted descriptive moments; retained per-episode summaries. Arms have independently sampled policy actions.",
        arms={condition: {name: dict(episodes=summary["cells"][condition]["arms"][name]["n"],
            successes=summary["cells"][condition]["arms"][name]["successes"],
            windows={window: summarize(values) for window, values in pooled[(condition, name)].items()})
            for name in summary["cells"][condition]["arms"]} for condition in summary["cells"]},
        condition_pooled={condition: {window: summarize(values) for window, values in items.items()}
            for condition, items in condition_pooled.items()}, per_episode=per_episode,
        limits=["This is a frozen measurement-model counterfactual on recorded trajectories; no new policy/observer rollout or task success claim.",
            "Fault truth is used only for diagnostic errors; it is absent from either equivalent estimate and covariance fit.",
            "The fitted position model omits physical velocity/contact state; healthy prediction qualification is not identifiability or fault-distribution qualification.",
            "Dynamic covariance is estimated only on healthy fit episodes0-5; episodes6-9 are reused validation diagnostics, not fresh qualification.",
            "Direct dynamic and FIR equivalent measurements are unfiltered, unbounded and not directly executable controllers; observer estimates are recorded postupdate references only.",
            "Changing the measurement model changes information supplied to a comparator; a prospective estimator comparison must share the residual/model/noise assumptions.",
            "Moment ratios do not demonstrate a task-level causal effect; no gains, thresholds, model coefficients or policy outcomes were changed."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--calibration-log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    result = analyze(args.telemetry, args.summary, args.calibration_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(result, stream, default=json_default, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(out=str(path), episodes=len(result["per_episode"]),
        integrity=result["arithmetic_integrity"], B_condition=result["B_diagnostics"]["condition"])))


if __name__ == "__main__":
    main()
