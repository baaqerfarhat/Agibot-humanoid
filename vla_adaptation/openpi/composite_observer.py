"""Position-reference composite adaptation, using only NumPy.

This is an action-interface variant inspired by Neural-Fly, not a reproduction of
its force controller or a claim that its continuous-time theorem applies here.
In particular, the observation matrix H and correction-effect matrix D need not
be equal.  The runner applies ``command = raw_command - mask * theta`` BEFORE
calling ``composite_step`` with the newly observed position error.

For a local healthy servo model q+ = A q + B u + offset, the nominal reference is
q_ref+ = A q_ref + B u_raw + offset.  With e = q - q_ref, D = B @ mask, and a
constant matched fault, its ideal error dynamics are e+ = A e - D (theta-f).
The tracking increment has the positive sign for ACTUAL MINUS REFERENCE error:
``dt * tracking_rate * P_post @ D.T @ metric @ e_next``.  It is preconditioned
descent on next-position error, combined with a prediction-error observer.

Contraction of the FITTED position model is only a screening diagnostic. Hidden
velocity/contact states, model error, saturation, changing references, and the
VLA feedback loop are outside that linear test. A held-out prediction check and
the augmented observer/reference dynamics check are supplied separately.
"""
from __future__ import annotations

import math

import numpy as np


def _array(value, name, ndim=None):
    out = np.asarray(value, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if not np.all(np.isfinite(out)):
        raise ValueError("%s must be finite" % name)
    return out


def _symmetric(value, name, size, positive_definite=False):
    out = _array(value, name, 2)
    if out.shape != (size, size):
        raise ValueError("%s must be %d by %d" % (name, size, size))
    if not np.allclose(out, out.T, rtol=1e-9, atol=1e-12):
        raise ValueError("%s must be symmetric" % name)
    out = (out + out.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(out)
    tolerance = 1e-12 * max(1.0, float(np.linalg.norm(out, 2)))
    if positive_definite:
        if eigenvalues[0] <= 0:
            raise ValueError("%s must be positive definite" % name)
    elif eigenvalues[0] < -tolerance:
        raise ValueError("%s must be positive semidefinite" % name)
    return out


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError("%s must be finite and %s" %
                         (name, "nonnegative" if allow_zero else "positive"))
    return value


def _indices(values, size, name):
    result = list(range(size)) if values is None else list(values)
    if not result or any(not isinstance(v, (int, np.integer)) for v in result):
        raise ValueError("%s must be nonempty integer indices" % name)
    result = [int(v) for v in result]
    if len(set(result)) != len(result) or min(result) < 0 or max(result) >= size:
        raise ValueError("%s contains duplicate or out-of-range indices" % name)
    return result


def _episode_arrays(episode):
    """The existing healthy logs store q AFTER the matching command u."""
    q = _array(episode["q"], "episode q", 2)
    u = _array(episode["u"], "episode u", 2)
    if len(q) != len(u) or len(q) < 2:
        raise ValueError("each episode needs equal q/u lengths and at least two steps")
    return q, u


def _transitions(episode, states, commands):
    q, u = _episode_arrays(episode)
    if "q_before" in episode:
        before = _array(episode["q_before"], "episode q_before", 2)
        if before.shape != q.shape:
            raise ValueError("q_before must have the same shape as post-command q")
        if not np.allclose(before[1:], q[:-1], rtol=1e-9, atol=1e-10):
            raise ValueError("q_before/q rows must form one consecutive trajectory")
        return before[:, states], u[:, commands], q[:, states], u, before[0, states]
    return q[:-1, states], u[1:, commands], q[1:, states], u[1:], q[0, states]


def contraction_report(A, metric=None):
    """Check a fixed discrete model; optionally solve A.T L A - L = -I.

    A nonnormal stable A can expand Euclidean distance while contracting in L.
    Failure returns diagnostics rather than silently modifying A or its poles.
    """
    A = _array(A, "A", 2)
    n = A.shape[0]
    if A.shape != (n, n) or n == 0:
        raise ValueError("A must be nonempty and square")
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    result = dict(spectral_radius=radius, schur_stable=bool(radius < 1.0),
                  euclidean_factor=float(np.linalg.norm(A, 2)),
                  metric=None, contraction_factor=None, contractive=False)
    if metric is None:
        if radius >= 1.0:
            return result
        operator = np.eye(n * n) - np.kron(A.T, A.T)
        flat = np.linalg.solve(operator, np.eye(n).reshape(-1, order="F"))
        metric = flat.reshape((n, n), order="F")
        metric = (metric + metric.T) * 0.5
    metric = _symmetric(metric, "metric", n, positive_definite=True)
    eigenvalues, vectors = np.linalg.eigh(metric)
    sqrt_metric = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    inverse_sqrt = (vectors * (1.0 / np.sqrt(eigenvalues))) @ vectors.T
    factor = float(np.linalg.norm(sqrt_metric @ A @ inverse_sqrt, 2))
    result.update(metric=metric, contraction_factor=factor,
                  contractive=bool(factor < 1.0))
    return result


def reference_step(model, reference_position, raw_action):
    """Advance the healthy reference with live RAW policy commands, never truth."""
    A = _array(model["A"], "A", 2)
    B = _array(model["B"], "B", 2)
    q_ref = _array(reference_position, "reference position", 1)
    raw_action = _array(raw_action, "raw action", 1)
    offset = _array(model["offset"], "offset", 1)
    command_indices = _indices(model["command_indices"], len(raw_action), "command indices")
    if A.shape != (len(q_ref), len(q_ref)) or B.shape != (len(q_ref), len(command_indices)):
        raise ValueError("reference dimensions do not match model")
    if offset.shape != q_ref.shape:
        raise ValueError("reference offset has wrong shape")
    return A @ q_ref + B @ raw_action[command_indices] + offset


def masked_tracking_map(model, correction_mask):
    """Embed B's command columns in estimator coordinates, then apply its mask.

    Parameter j must represent an additive correction on command channel j.
    The function cannot infer this coordinate correspondence for another actuator
    interface (e.g. a joint-torque parameter applied through Cartesian targets).
    """
    B = _array(model["B"], "B", 2)
    mask = _array(correction_mask, "correction mask", 1)
    indices = _indices(model["command_indices"], len(mask), "command indices")
    if B.shape[1] != len(indices):
        raise ValueError("B columns do not match command indices")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("correction mask must contain only zero and one")
    result = np.zeros((B.shape[0], len(mask)))
    result[:, indices] = B
    return result * mask[None, :]


def score_joint_reference(model, episodes, episode_indices):
    """Report both teacher-forced and independent-reference rollout error.

    Only the first logged position initializes each reference rollout. Subsequent
    reference steps use its OWN previous position and the logged raw commands.
    """
    if not episode_indices:
        return None
    indices = _indices(episode_indices, len(episodes), "score episode indices")
    states = model["state_indices"]
    commands = model["command_indices"]
    one_step, rollout, actual = [], [], []
    for index in indices:
        previous, command, target, raw_commands, initial = _transitions(
            episodes[index], states, commands)
        prediction = previous @ model["A"].T + command @ model["B"].T + model["offset"]
        one_step.append(target - prediction)
        q_ref = initial.copy()
        for actual_position, command in zip(target, raw_commands):
            q_ref = reference_step(model, q_ref, command)
            rollout.append(actual_position - q_ref)
        actual.append(target)
    one_step = np.concatenate(one_step)
    rollout = np.asarray(rollout)
    actual = np.concatenate(actual)
    return dict(episode_indices=indices, samples=int(len(one_step)),
                one_step_rmse=float(np.sqrt(np.mean(one_step ** 2))),
                one_step_rmse_per_state=np.sqrt(np.mean(one_step ** 2, axis=0)),
                rollout_rmse=float(np.sqrt(np.mean(rollout ** 2))),
                rollout_rmse_per_state=np.sqrt(np.mean(rollout ** 2, axis=0)),
                rollout_max_error_norm=float(np.linalg.norm(rollout, axis=1).max()),
                target_standard_deviation=np.std(actual, axis=0))


def fit_joint_reference(episodes, state_indices, command_indices=None,
                        fit_episode_indices=None, validation_episode_indices=(), ridge=1e-6):
    """Fit an affine position-only servo model from healthy logs with q/u.

    q[t] is the position AFTER u[t]. With q_before, every transition is used;
    otherwise training uses (q[t-1],u[t])->q[t] and drops the first sample,
    whose pre-command state was not logged. Splits
    are by complete episode. The caller is responsible for healthy provenance.
    No validation data enter the fit, contraction metric, or regularization.
    """
    if not episodes:
        raise ValueError("at least one healthy episode is required")
    ridge = _positive(ridge, "ridge", allow_zero=True)
    q0, u0 = _episode_arrays(episodes[0])
    states = _indices(state_indices, q0.shape[1], "state indices")
    commands = _indices(command_indices, u0.shape[1], "command indices")
    validation = [] if not validation_episode_indices else _indices(
        validation_episode_indices, len(episodes), "validation episode indices")
    training = ([i for i in range(len(episodes)) if i not in validation]
                if fit_episode_indices is None else list(fit_episode_indices))
    training = _indices(training, len(episodes), "fit episode indices")
    if set(training) & set(validation):
        raise ValueError("reference training and validation episodes must be disjoint")
    X, Y = [], []
    for index in training:
        q, u = _episode_arrays(episodes[index])
        if q.shape[1] != q0.shape[1] or u.shape[1] != u0.shape[1]:
            raise ValueError("all episodes must share state and command dimensions")
        previous, command, target, _, _ = _transitions(episodes[index], states, commands)
        X.append(np.concatenate([previous, command], axis=1))
        Y.append(target)
    X, Y = np.concatenate(X), np.concatenate(Y)
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean
    gram = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    coefficients = np.linalg.solve(gram, Xc.T @ Yc)
    offset = y_mean - x_mean @ coefficients
    A = coefficients[:len(states)].T
    B = coefficients[len(states):].T
    contraction = contraction_report(A)
    model = dict(A=A, B=B, offset=offset, metric=contraction["metric"],
                 state_indices=states, command_indices=commands)
    singular_values = np.linalg.svd(Xc, compute_uv=False)
    report = dict(
        description="Healthy fitted position-only servo; not full-state/VLA contraction",
        training_episode_indices=training, validation_episode_indices=validation,
        training_samples=int(len(X)), features=int(X.shape[1]), ridge=ridge,
        design_rank=int(np.linalg.matrix_rank(Xc)),
        design_condition_number=float(np.linalg.cond(Xc)),
        regularized_gram_condition_number=float(np.linalg.cond(gram)),
        design_singular_values=singular_values, contraction=contraction,
        train=score_joint_reference(model, episodes, training),
        validation=score_joint_reference(model, episodes, validation),
        first_unlogged_precommand_sample_dropped=any(
            "q_before" not in episodes[i] for i in training),
    )
    return model, report


def qualify_reference(report, max_validation_rmse, max_condition_number=1e8,
                      min_samples_per_feature=2.0, max_contraction_factor=0.999999):
    """Conservative empirical gate; caller must predeclare its error tolerance.

    This gate authorizes a candidate experiment, not a safety/stability theorem.
    Test data must not be used to relax these limits after a rejection.
    """
    threshold = _positive(max_validation_rmse, "validation RMSE threshold")
    condition_limit = _positive(max_condition_number, "condition limit")
    sample_ratio = _positive(min_samples_per_feature, "sample-to-feature ratio")
    contraction_limit = _positive(max_contraction_factor, "contraction limit")
    if contraction_limit >= 1:
        raise ValueError("contraction limit must be below one")
    reasons = []
    contraction = report["contraction"]
    factor = contraction.get("contraction_factor")
    if not contraction["contractive"] or factor is None or factor > contraction_limit:
        reasons.append("fitted position model is not sufficiently contractive")
    if report["design_rank"] < report["features"]:
        reasons.append("healthy position/command regression is rank deficient")
    if report["design_condition_number"] > condition_limit:
        reasons.append("healthy position/command regression is ill conditioned")
    if report["training_samples"] < sample_ratio * report["features"]:
        reasons.append("too few training samples for the fitted reference")
    validation = report.get("validation")
    if validation is None:
        reasons.append("no held-out healthy reference validation")
    elif (max(validation["one_step_rmse_per_state"]) > threshold or
          max(validation["rollout_rmse_per_state"]) > threshold):
        reasons.append("held-out healthy position error exceeds the declared tolerance")
    return not reasons, reasons


def composite_step(theta, observation, H, *, state=None, Q, R, dt,
                   tracking_error=None, tracking_map=None, metric=None,
                   tracking_rate=0.0, damping=0.0, clip=0.3, initial_covariance=1.0):
    """One regularized KF update plus position-reference feedback.

    ``observation`` is bias-corrected residual r-H*b, NOT achieved position.
    ``H`` maps parameters to that observation. ``tracking_map`` instead maps an
    applied additive command to the selected next-position coordinates; include
    the correction mask in it. ``tracking_error`` is actual-reference position
    AFTER the action produced with input theta. No ground-truth fault is used.

    dt is seconds; damping and tracking_rate are per-second gains. Q is process
    covariance PER SECOND, R is measurement covariance PER SAMPLE. To reproduce
    an existing random-walk KF with Q_step, pass Q=Q_step/dt, damping=0, and
    tracking_rate=0. A positive damping value changes the estimator even with
    tracking disabled and should have its own matched ablation.
    """
    theta = _array(theta, "theta", 1)
    observation = _array(observation, "observation", 1)
    H = _array(H, "H", 2)
    p, m = len(theta), len(observation)
    if H.shape != (m, p):
        raise ValueError("H must map parameters to observation coordinates")
    dt = _positive(dt, "dt")
    damping = _positive(damping, "damping", allow_zero=True)
    rate = _positive(tracking_rate, "tracking rate", allow_zero=True)
    Q = _symmetric(Q, "Q", p)
    R = _symmetric(R, "R", m, positive_definite=True)
    if state is None:
        if np.ndim(initial_covariance) == 0:
            P = np.eye(p) * _positive(initial_covariance, "initial covariance")
        else:
            P = _symmetric(initial_covariance, "initial covariance", p, positive_definite=True)
        updates = 0
    else:
        P = _symmetric(state["covariance"], "state covariance", p)
        updates = int(state.get("updates", 0))
    decay = math.exp(-damping * dt)
    process_interval = dt if damping == 0 else -math.expm1(-2 * damping * dt) / (2 * damping)
    predicted_covariance = decay ** 2 * P + process_interval * Q
    predicted_theta = decay * theta
    gain = np.linalg.solve(H @ predicted_covariance @ H.T + R,
                           H @ predicted_covariance).T
    remaining = np.eye(p) - gain @ H
    covariance = remaining @ predicted_covariance @ remaining.T + gain @ R @ gain.T
    covariance = (covariance + covariance.T) * 0.5
    innovation = observation - H @ predicted_theta
    posterior = predicted_theta + gain @ innovation
    tracking_increment = np.zeros(p)
    tracking_gain = None
    if rate > 0:
        if tracking_error is None or tracking_map is None:
            raise ValueError("positive tracking rate requires a position error and tracking map")
        error = _array(tracking_error, "tracking error", 1)
        D = _array(tracking_map, "tracking map", 2)
        if D.shape != (len(error), p):
            raise ValueError("tracking map must map parameters to tracking-error coordinates")
        L = np.eye(len(error)) if metric is None else _symmetric(
            metric, "tracking metric", len(error), positive_definite=True)
        tracking_gain = dt * rate * covariance @ D.T @ L
        tracking_increment = tracking_gain @ error
    unprojected = posterior + tracking_increment
    if clip is None:
        updated = unprojected.copy()
    else:
        bound = _array(clip, "clip")
        if bound.ndim > 1 or (bound.ndim == 1 and bound.shape != theta.shape) or np.any(bound <= 0):
            raise ValueError("clip must be a positive scalar or one bound per parameter")
        updated = np.clip(unprojected, -bound, bound)
    return updated, dict(
        estimator_state=dict(covariance=covariance, updates=updates + 1),
        gain=gain, effective_gain=gain @ H, innovation=innovation,
        prediction_increment=posterior - theta, tracking_increment=tracking_increment,
        tracking_gain=tracking_gain, decay=decay, dt=dt,
        projection_active=bool(np.any(updated != unprojected)),
        update_applied=True,
    )


def augmented_error_report(A, D, H, covariance, gain, *, dt,
                           tracking_rate=0.0, damping=0.0, metric=None):
    """Check the local UNSATURATED frozen-gain (position, parameter) recursion.

    The action uses theta before the observation/update, as in both repo runners.
    e+ = A e - D theta_error; theta_error+ = (I-KH)*decay*theta_error + T*e+.
    Damping of a nonzero true parameter adds an affine forcing term, so even a
    Schur matrix need not yield unbiased cancellation. Model error, moving faults
    and time-varying gains are also not certified by this frozen-gain diagnostic.
    """
    A = _array(A, "A", 2)
    D = _array(D, "D", 2)
    H = _array(H, "H", 2)
    gain = _array(gain, "gain", 2)
    n, p = A.shape[0], D.shape[1]
    if A.shape != (n, n) or D.shape[0] != n or H.shape[1] != p or gain.shape != (p, H.shape[0]):
        raise ValueError("augmented error model dimensions are inconsistent")
    covariance = _symmetric(covariance, "covariance", p)
    L = np.eye(n) if metric is None else _symmetric(metric, "metric", n, positive_definite=True)
    dt = _positive(dt, "dt")
    rate = _positive(tracking_rate, "tracking rate", allow_zero=True)
    damping = _positive(damping, "damping", allow_zero=True)
    decay = math.exp(-damping * dt)
    T = dt * rate * covariance @ D.T @ L
    F = (np.eye(p) - gain @ H) * decay
    matrix = np.block([[A, -D], [T @ A, F - T @ D]])
    eigenvalues = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(eigenvalues)))
    return dict(matrix=matrix, eigenvalues_real=eigenvalues.real,
                eigenvalues_imag=eigenvalues.imag, spectral_radius=radius,
                schur_stable=bool(radius < 1.0),
                scope="Local unsaturated fixed-gain fitted model; not a physical guarantee")
