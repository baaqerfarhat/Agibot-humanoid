"""Prepare a joint reference and lock a calibration-only composite gain grid.

The input is a healthy list of {u,q[,q_before]} episodes, or a dictionary with an
``episodes`` list. q is the achieved position AFTER its corresponding command.
Whole episodes, rather than timesteps, are split between fit and validation.

Optional --openloop enables JOINT-POSITION FIR observation-noise calibration.
This is appropriate for the ALOHA runner. A LIBERO joint-reference log does not
by itself provide the Cartesian FIR observations used by adaptive_law.py, and
must not be used as a substitute for those observations.

Rejected qualifications are written to --out and exit with status 2. Limits are
explicit inputs, never selected from task outcomes or relaxed by this script.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import composite_observer as composite


def index_list(value):
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("provide distinct nonnegative comma-separated indices")
    return result


def float_list(value):
    result = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not result or any(not np.isfinite(x) or x < 0 for x in result):
        raise argparse.ArgumentTypeError("provide finite nonnegative comma-separated strengths")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("tracking strengths must be distinct")
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def fit_joint_fir_noise(episodes, fit_indices, lag=6):
    """Same per-channel least-squares FIR convention as aloha_adapt.fit_plant.

    Centering the residual estimates measurement NOISE; it does not subtract a
    healthy mean from the observer. The mean is retained as diagnostic metadata.
    """
    if not isinstance(lag, int) or lag < 0:
        raise ValueError("FIR lag must be a nonnegative integer")
    first = episodes[fit_indices[0]]
    n = np.asarray(first["q"]).shape[1]
    X, Y = [[] for _ in range(n)], [[] for _ in range(n)]
    for index in fit_indices:
        u = np.asarray(episodes[index]["u"], dtype=float)
        q = np.asarray(episodes[index]["q"], dtype=float)
        if q.ndim != 2 or u.shape != q.shape or q.shape[1] != n:
            raise ValueError("joint FIR noise calibration requires equal joint q/u dimensions")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(u)):
            raise ValueError("joint FIR calibration data must be finite")
        for t in range(lag, len(q)):
            window = u[t - lag:t + 1][::-1]
            for j in range(n):
                X[j].append(np.r_[window[:, j], 1.0])
                Y[j].append(q[t, j])
    if len(Y[0]) < lag + 2:
        raise ValueError("too few healthy samples for joint FIR calibration")
    W, residual_columns, ranks, conditions = [], [], [], []
    for j in range(n):
        design, target = np.asarray(X[j]), np.asarray(Y[j])
        weights, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
        W.append(weights)
        residual_columns.append(target - design @ weights)
        ranks.append(int(rank))
        conditions.append(float(singular_values[0] / singular_values[-1])
                          if singular_values[-1] > 0 else float("inf"))
    residual = np.stack(residual_columns, axis=1)
    sample_covariance = np.atleast_2d(np.cov(residual, rowvar=False, ddof=1))
    eigenvalues, vectors = np.linalg.eigh(sample_covariance)
    floor = max(1e-12, float(eigenvalues[-1]) * 1e-9)
    R = (vectors * np.maximum(eigenvalues, floor)) @ vectors.T
    R = (R + R.T) * 0.5
    return dict(W=np.asarray(W), R=R, observation_kind="joint_position_FIR_residual",
                fit_episode_indices=list(fit_indices), samples=int(len(residual)),
                fir_lag=lag, fir_fit="per-channel OLS with intercept",
                fir_ranks=ranks, fir_condition_numbers=conditions,
                residual_mean=residual.mean(axis=0), sample_covariance=sample_covariance,
                eigenvalue_floor=floor, residual_centering_changes_observer_bias=False)


def tracking_grid(model, M, noise, correction_indices, strengths, dt,
                  gamma=0.08, damping=0.0, covariance_steps=200):
    """Choose numerical scale from healthy calibration and test each fixed gain.

    eta = ||dt * rate * P_infinity * D.T * L * D||_2 is a dimensionless
    parameter-feedback strength, before projection. Every declared eta is kept
    in the output, including rejected candidates. Stability screening spans the
    covariance transient and converged covariance; it is not a switching-system
    or nonlinear physical guarantee.
    """
    M = np.asarray(M, dtype=float)
    R = np.asarray(noise["R"], dtype=float)
    if M.ndim != 2 or M.shape != R.shape or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square and match the joint FIR observation covariance")
    n = M.shape[0]
    if np.linalg.matrix_rank(M) != n:
        raise ValueError("matched-gain noise calibration requires full-rank M")
    if not 0 < gamma < 1 or dt <= 0 or damping < 0 or covariance_steps < 1:
        raise ValueError("invalid gamma, dt, damping, or covariance step count")
    mask = np.zeros(n)
    for index in correction_indices:
        if index < 0 or index >= n:
            raise ValueError("correction index is outside estimator coordinates")
        mask[index] = 1
    D = composite.masked_tracking_map(model, mask)
    if model["metric"] is None:
        raise ValueError("a contractive fitted reference is needed for tracking-gain screening")
    L = model["metric"]
    M_inv = np.linalg.inv(M)
    C = M_inv @ R @ M_inv.T
    Q_step = gamma ** 2 / (1 - gamma) * C
    Q_step = (Q_step + Q_step.T) * 0.5
    Q_rate = Q_step / dt
    state, covariance_path = None, []
    previous = None
    converged = False
    for step in range(covariance_steps):
        _, diag = composite.composite_step(
            np.zeros(n), np.zeros(n), M, state=state, Q=Q_rate, R=R,
            dt=dt, damping=damping, tracking_rate=0, clip=None)
        state = diag["estimator_state"]
        P, K = state["covariance"], diag["gain"]
        covariance_path.append((step + 1, P, K))
        if previous is not None:
            relative_change = np.linalg.norm(P - previous, 2) / max(np.linalg.norm(P, 2), 1e-30)
            if relative_change < 1e-10:
                converged = True
                break
        previous = P
    if damping == 0:
        P_inf, K_inf = gamma * C, gamma * M_inv
        covariance_path.append(("analytic_steady_state", P_inf, K_inf))
        converged = True
    else:
        P_inf, K_inf = covariance_path[-1][1:]
    scale = float(np.linalg.norm(dt * P_inf @ D.T @ L @ D, 2))
    candidates = []
    for strength in strengths:
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("tracking strengths must be finite and nonnegative")
        if strength > 0 and scale <= 0:
            candidates.append(dict(strength=strength, tracking_rate=None, allowed=False,
                                   reasons=["zero tracking response in corrected coordinates"]))
            continue
        rate = 0.0 if strength == 0 else float(strength / scale)
        reports = []
        for step, P, K in covariance_path:
            r = composite.augmented_error_report(
                model["A"], D, M, P, K, dt=dt, tracking_rate=rate,
                damping=damping, metric=L)
            reports.append(dict(covariance_step=step, spectral_radius=r["spectral_radius"],
                                schur_stable=r["schur_stable"]))
        worst = max(reports, key=lambda r: r["spectral_radius"])
        allowed = converged and all(r["schur_stable"] for r in reports)
        reasons = []
        if not converged:
            reasons.append("covariance transient did not converge within declared budget")
        if not all(r["schur_stable"] for r in reports):
            reasons.append("non-Schur local augmented dynamics during covariance transient/steady state")
        candidates.append(dict(strength=float(strength), tracking_rate=rate, allowed=allowed,
                               reasons=reasons, maximum_spectral_radius=worst["spectral_radius"],
                               worst_covariance_step=worst["covariance_step"],
                               covariance_path_reports=reports))
    return dict(
        H=M, Q_step=Q_step, Q_rate=Q_rate, R=R, dt=dt, gamma=gamma, damping=damping,
        correction_mask=mask, tracking_map=D, metric=L,
        process_noise_units="parameter covariance per second in Q_rate; per step in Q_step",
        measurement_noise_units="joint-position residual covariance per sample",
        covariance_converged=converged, covariance_steps_evaluated=len(covariance_path),
        steady_covariance=P_inf, steady_gain=K_inf, steady_effective_gain=K_inf @ M,
        scale=scale, scale_definition="spectral norm of dt*P_infinity*D.T*L*D",
        strength_definition="dimensionless spectral norm of tracking parameter feedback",
        candidates=candidates,
        limitation="Fitted local fixed-gain Schur checks do not prove stability of nonlinear or time-varying dynamics",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-indices", type=index_list, required=True)
    parser.add_argument("--command-indices", type=index_list)
    parser.add_argument("--fit-episodes", type=index_list)
    parser.add_argument("--validation-episodes", type=index_list, required=True)
    parser.add_argument("--max-validation-rmse", type=float, required=True,
                        help="predeclared PER-STATE one-step and rollout RMSE tolerance; never tuned here")
    parser.add_argument("--max-condition-number", type=float, default=1e8)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--openloop", type=Path,
                        help="optional M artifact for JOINT-POSITION observation noise/gain calibration")
    parser.add_argument("--correction-indices", type=index_list,
                        help="required with --openloop: parameter/action coordinates actually corrected")
    parser.add_argument("--fir-lag", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--tracking-strengths", type=float_list, default=float_list("0,0.01,0.05,0.1,0.25"))
    parser.add_argument("--covariance-steps", type=int, default=200)
    parser.add_argument("--purpose", choices=("diagnostic", "calibration"), default="diagnostic")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out.resolve() == args.log.resolve() or (
            args.openloop is not None and args.out.resolve() == args.openloop.resolve()):
        parser.error("--out must differ from input files")
    if args.openloop is not None and args.correction_indices is None:
        parser.error("--correction-indices is required with --openloop")
    if not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("--dt must be finite and positive")
    source = json.loads(args.log.read_text())
    episodes = source.get("episodes") if isinstance(source, dict) else source
    if not isinstance(episodes, list) or not episodes:
        parser.error("--log must contain a nonempty episode list, directly or under episodes")
    provenance_fields = ("task", "init", "seed", "episode", "command_donor_episode")
    artifact = dict(
        schema_version=1, purpose=args.purpose,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=dict(path=str(args.log.resolve()), sha256=digest(args.log),
                    episode_provenance=[dict(episode_index=i, **{
                        k: episode[k] for k in provenance_fields if k in episode})
                        for i, episode in enumerate(episodes)],
                    healthy_status="caller-supplied healthy log; not inferred from success labels"),
        code=dict(prepare_sha256=digest(__file__),
                  observer_sha256=digest(composite.__file__), numpy_version=np.__version__),
        settings=vars(args), model=None, report=None,
        qualification=dict(allowed=False, reasons=[]), noise=None, tracking=None, observer=None,
        interpretation=[
            "Position-only healthy model diagnostics do not establish full-state or VLA-loop contraction.",
            "Reference uses live uncorrected policy commands, not a healthy counterfactual rollout.",
            "Validation episodes do not fit the reference, FIR, noise covariance, or gain scale.",
            "Qualification and strength grid are fixed before task-outcome evaluation.",
        ],
    )
    try:
        model, report = composite.fit_joint_reference(
            episodes, args.state_indices, command_indices=args.command_indices,
            fit_episode_indices=args.fit_episodes,
            validation_episode_indices=args.validation_episodes, ridge=args.ridge)
        allowed, reasons = composite.qualify_reference(
            report, max_validation_rmse=args.max_validation_rmse,
            max_condition_number=args.max_condition_number)
        artifact.update(model=model, report=report,
                        qualification=dict(allowed=allowed, reasons=reasons))
        if args.openloop is not None:
            M = np.asarray(json.loads(args.openloop.read_text())["M"], dtype=float)
            artifact["source"]["sensitivity"] = dict(
                path=str(args.openloop.resolve()), sha256=digest(args.openloop))
            noise = fit_joint_fir_noise(episodes, report["training_episode_indices"], args.fir_lag)
            artifact["noise"] = noise
            tracking = tracking_grid(
                model, M, noise, args.correction_indices, args.tracking_strengths,
                args.dt, gamma=args.gamma, damping=args.damping,
                covariance_steps=args.covariance_steps)
            artifact["tracking"] = tracking
            artifact["observer"] = dict(
                W=noise["W"], M=M, Q=tracking["Q_step"], R=noise["R"],
                Q_units="parameter covariance per control STEP; composite_step needs Q/dt",
                R_units="joint-position residual covariance per observation",
                dt=args.dt, gamma=args.gamma,
                fit_episode_indices=report["training_episode_indices"],
                bias=None,
            )
            for candidate in tracking["candidates"]:
                candidate["reference_qualified"] = allowed
                candidate["qualified_for_experiment"] = bool(allowed and candidate["allowed"])
    except (ValueError, np.linalg.LinAlgError) as error:
        artifact["qualification"] = dict(
            allowed=False, reasons=artifact["qualification"]["reasons"] +
            ["%s: %s" % (type(error).__name__, error)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_ready(artifact), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.out), qualification=artifact["qualification"])))
    if artifact["tracking"] is not None:
        for candidate in artifact["tracking"]["candidates"]:
            print(json.dumps({k: candidate.get(k) for k in (
                "strength", "tracking_rate", "maximum_spectral_radius", "qualified_for_experiment")}))
    return 0 if artifact["qualification"]["allowed"] else 2


if __name__ == "__main__":
    sys.exit(main())
