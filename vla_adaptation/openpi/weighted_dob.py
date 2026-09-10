"""Masked, bounded ridge allocation of a low-pass disturbance residual.

This is a calibrated DOB/allocation variant, not a new stability or safety claim.
The observation residual is filtered first. Only inputs that can actually receive
correction participate in the allocation, whose box bound is part of the solve.
The caller applies minus the returned estimate to the selected action channels.
No disturbance truth, rollout outcome, or simulator is used here.
"""
from __future__ import annotations

import numpy as np


def _finite(value, name, ndim=None):
    value = np.asarray(value, dtype=float)
    if (ndim is not None and value.ndim != ndim) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite" + (f" and {ndim}-dimensional" if ndim is not None else ""))
    return value


def _positive_vector(value, name, dimension):
    value = _finite(value, name)
    if value.ndim == 0:
        value = np.full(dimension, float(value))
    if value.shape != (dimension,) or np.any(value <= 0):
        raise ValueError(f"{name} must be positive, scalar or one value per parameter")
    return value


def _kkt(H, b, x, limits):
    gradient = H @ x - b
    residual = gradient.copy()
    # Only exact attained bounds are classified as active; a near-bound interior
    # point must still satisfy stationarity. The solver sets blocking bounds exactly.
    lower, upper = x == -limits, x == limits
    residual[lower] = np.minimum(gradient[lower], 0.)
    residual[upper] = np.maximum(gradient[upper], 0.)
    scale = 1. + np.abs(b) + np.abs(H) @ np.abs(x)
    return gradient, residual, scale


def _solve_box_ridge(C, y, limits, prior_std, warm_start=None, *,
                     tolerance=1e-10, max_iterations=1000):
    """Feasible active-set minimization of ||Cx-y||² + ||x/prior_std||².

    Newton steps solve the current free-variable subproblem; a blocking bound
    shortens a step. At a stationary face, a bound with an invalid KKT multiplier
    is released. Each move decreases the convex objective up to recorded numeric
    tolerance. Positive ridge precision makes the minimizer unique.
    """
    n = C.shape[1]
    H = C.T @ C + np.diag(1. / prior_std ** 2)
    b = C.T @ y
    def objective(x):
        residual = C @ x - y
        return float(residual @ residual + np.sum((x / prior_std) ** 2))
    zero_objective = float(y @ y)
    x = np.zeros(n)
    history = [zero_objective]
    if warm_start is not None:
        initial = np.clip(np.asarray(warm_start, float), -limits, limits)
        if objective(initial) < zero_objective:
            x = initial.copy()
            history.append(objective(x))
    active = np.zeros(n, dtype=int)
    active[x == -limits] = -1
    active[x == limits] = 1
    for iteration in range(max_iterations):
        gradient, residual, scale = _kkt(H, b, x, limits)
        normalized = float(np.max(np.abs(residual) / scale)) if n else 0.
        if normalized <= tolerance:
            value = objective(x)
            numeric_slack = 128 * np.finfo(float).eps * max(1., zero_objective)
            if value > zero_objective + numeric_slack:
                raise RuntimeError("bounded ridge solve exceeded its zero-feasible objective")
            return x, dict(status="optimal", iterations=iteration, kkt_residual=float(np.max(np.abs(residual))) if n else 0.,
                kkt_scaled_residual=normalized, kkt_tolerance=tolerance,
                objective=value, zero_objective=zero_objective, objective_history=history,
                objective_numeric_tolerance=numeric_slack, active_lower=(x == -limits),
                active_upper=(x == limits), feasibility_error=float(np.max(np.maximum(np.abs(x)-limits, 0.))) if n else 0.,
                weighted_residual_norm=float(np.linalg.norm(C @ x-y)),
                prior_penalty=float(np.sum((x/prior_std)**2)),
                gram_condition_number=float(np.linalg.cond(H)) if n else None)
        free = active == 0
        stationary = not free.any() or np.max(np.abs(gradient[free]) / scale[free]) <= tolerance
        if stationary:
            # At a lower bound g>=0 is valid; at an upper bound g<=0 is valid.
            violation = np.where(active == -1, -gradient,
                                 np.where(active == 1, gradient, -np.inf)) / scale
            release = int(np.argmax(violation))
            if violation[release] <= tolerance:
                raise RuntimeError("bounded ridge active set stalled before satisfying KKT")
            active[release] = 0
            continue
        direction = np.zeros(n)
        try:
            direction[free] = np.linalg.solve(H[np.ix_(free, free)], -gradient[free])
        except np.linalg.LinAlgError as error:
            raise RuntimeError("bounded ridge free-variable system could not be solved") from error
        positive, negative = free & (direction > 0), free & (direction < 0)
        fractions = np.full(n, np.inf)
        fractions[positive] = (limits[positive]-x[positive])/direction[positive]
        fractions[negative] = (-limits[negative]-x[negative])/direction[negative]
        blocker = int(np.argmin(fractions))
        alpha = min(1., float(fractions[blocker]))
        if alpha < 0 or not np.isfinite(direction).all():
            raise RuntimeError("bounded ridge search direction lost numerical feasibility")
        next_x = x + alpha * direction
        # Remove floating-point overshoot only; box feasibility determines alpha.
        overshoot = float(np.max(np.maximum(np.abs(next_x)-limits, 0.)))
        if overshoot > 64*np.finfo(float).eps*max(1., float(np.max(limits))):
            raise RuntimeError("bounded ridge step violated the box")
        next_x = np.clip(next_x, -limits, limits)
        if fractions[blocker] <= 1.:
            side = 1 if direction[blocker] > 0 else -1
            next_x[blocker] = side * limits[blocker]
            active[blocker] = side
        value = objective(next_x)
        slack = 128*np.finfo(float).eps*max(1., zero_objective, history[-1])
        if not np.isfinite(value) or value > history[-1]+slack:
            raise RuntimeError("bounded ridge objective did not decrease numerically")
        x = next_x
        history.append(value)
    _, residual, scale = _kkt(H, b, x, limits)
    raise RuntimeError("bounded ridge did not converge: iterations=%d, scaled KKT=%g" %
                       (max_iterations, np.max(np.abs(residual)/scale)))


def weighted_dob_step(f_hat, r, M, *, mask, R, gamma, clip, state=None,
                      prior_std=.1, bias=None):
    """Filter the residual, then allocate only realizable correction coordinates.

    d_hat+ = (1-gamma)*d_hat + gamma*(r-M*bias), initially d_hat=0.
    f_active = argmin_{|f|<=clip} ||R^-1/2(M_active*f-d_hat+)||²
                                  + ||f/prior_std||².

    R must be positive definite (use the shared, explicitly floored healthy
    covariance); no extra covariance regularization is silently introduced here.
    Prior standard deviations and clip bounds are in parameter/action units.
    f_hat supplies the output dimension and an optional feasible optimizer warm
    start. The persistent filter state is diag['estimator_state']; f_hat alone is
    insufficient to resume the residual filter. Inactive output entries are zero.

    The objective comparison is against zero correction for THIS filtered
    residual and calibrated M. It is not a claim about task error, a changing
    physical system, or global contraction. Bounds are included in the solve.
    Inputs and previous state are never mutated. Nonconvergence raises explicitly.
    """
    previous = _finite(f_hat, "f_hat", 1)
    residual = _finite(r, "residual", 1)
    M = _finite(M, "M", 2)
    p, m = len(previous), len(residual)
    if not p or not m or M.shape != (m, p):
        raise ValueError("M must map parameter coordinates to residual coordinates")
    mask = _finite(mask, "mask", 1)
    if mask.shape != (p,) or np.any((mask != 0) & (mask != 1)):
        raise ValueError("mask must be a binary vector matching the parameters")
    selected = mask.astype(bool)
    if selected.sum() > 12:
        raise ValueError("this small dense allocator supports at most twelve active inputs")
    limits = _positive_vector(clip, "clip", p)
    prior = _positive_vector(prior_std, "prior_std", p)
    gamma = float(gamma)
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must lie in (0, 1]")
    R = _finite(R, "R", 2)
    if R.shape != (m, m) or not np.allclose(R, R.T, rtol=1e-10, atol=1e-12):
        raise ValueError("R must be symmetric and match the residual dimension")
    try:
        factor = np.linalg.cholesky((R+R.T)*.5)
    except np.linalg.LinAlgError as error:
        raise ValueError("R must be positive definite; apply the declared healthy covariance floor upstream") from error
    measured = residual.copy()
    if bias is not None:
        bias = _finite(bias, "bias", 1)
        if bias.shape != (p,):
            raise ValueError("bias must have one entry per parameter")
        measured -= M @ bias
    if state is None:
        filtered_before = np.zeros(m)
        updates = 0
    else:
        filtered_before = _finite(state["residual_estimate"], "state residual_estimate", 1)
        if filtered_before.shape != (m,):
            raise ValueError("state residual estimate has the wrong dimension")
        updates = state.get("updates", 0)
        if type(updates) not in (int, np.int32, np.int64) or updates < 0:
            raise ValueError("state updates must be a nonnegative integer")
    filtered = (1.-gamma)*filtered_before + gamma*measured
    C = np.linalg.solve(factor, M[:, selected])
    y = np.linalg.solve(factor, filtered)
    allocated, optimizer = _solve_box_ridge(C, y, limits[selected], prior[selected], previous[selected])
    updated = np.zeros(p)
    updated[selected] = allocated
    return updated, dict(estimator_state=dict(residual_estimate=filtered, updates=int(updates)+1),
        update_applied=True, filtered_residual_before=filtered_before.copy(),
        bias_corrected_residual=measured, residual_estimate=filtered.copy(),
        predicted_compensated_filtered_residual=filtered-M @ updated,
        active_input_indices=np.flatnonzero(selected), prior_std=prior.copy(),
        optimizer=optimizer,
        estimate_increment=updated-previous,
        nr=None, attenuation=None, deadzone_fired=None,
        interpretation="Bounded allocation of a filtered calibrated residual; not physical stability or task recovery")
