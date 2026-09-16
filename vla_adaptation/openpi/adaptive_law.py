"""Adaptive correction of an action fault from the robot's own motion -- no search.

The pieces, all measured rather than assumed:

  error      e_t = achieved motion - motion the NOMINAL plant predicts for the command we
             actually sent. Observable every step from robot0_eef_pos; needs no notion of
             task success.
  plant      P: identified by FIR regression on fault-free rollouts. Translation fits at
             R^2 = 0.98; rotation does not (0.1-0.5), which is a known weakness here.
  map        M = d(motion)/d(fault), measured open loop one axis at a time. Condition
             number 3.6, so it inverts -- but 59% of its mass is OFF-DIAGONAL, and the
             rotation axes are a swapped, sign-flipped pair (dry <- drz at +0.383,
             drz <- dry at -0.424, own diagonals ~0.01). A per-axis law would drive dry
             from drz's error with the wrong sign and diverge. This is why M^-1 is needed
             and a diagonal gain is not enough.

Law:  with executed = a_cmd + c_t + f,   r_t = y_t - P(a_cmd + c_t) ~ M f
      f_hat <- f_hat + gamma (M^-1 r_t - f_hat)         (exponential, gamma small)
      c_t   =  -f_hat
`r_t` does not depend on c_t, so the estimate is not chasing its own correction.
"""
from __future__ import annotations

import argparse, collections, contextlib, json, pathlib, sys
import numpy as np

OUT = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
K_FIR = 6
WARMUP_STEPS = 10
IMAGE_SIZE = 224
MATCHED_BASELINES = ("dob", "integral_calibrated", "rls", "kalman")


def correction_mask(corr_dims=None):
    mask = np.ones(6)
    if corr_dims is not None:
        mask = np.zeros(6)
        mask[list(corr_dims)] = 1.0
    return mask


def applied_correction(f_hat, *, adapt, apply_corr=True, static_c=None, mask=None):
    """The six Cartesian corrections; the gripper is never corrected.

    SIGN, because the two runners disagree and it is easy to get backwards. Here the
    estimator path applies `c = -f_hat` while the static path applies `c = static_c`
    UNNEGATED, and the caller then does `a_corr += c`. So `--static-corr` must be given the
    NEGATED fault: to cancel a +0.10 fault, pass -0.10. `aloha_adapt.py` negates internally
    (`c = -static_corr`) and is given the POSITIVE fault, which is why the stored ALOHA runs
    record `static_corr=0.019` against a `fault_vec=0.02`. Passing the positive fault here
    doubles it instead of cancelling it, and silently turns an oracle arm into the worst arm
    in a comparison table.
    """
    if static_c is not None:
        c = static_c.copy() if adapt else np.zeros(6)
    else:
        c = -f_hat if (adapt and apply_corr) else np.zeros(6)
    # Keep the original arithmetic: with no subset there was no multiplication.
    return c if mask is None else c * mask


def estimator_step(f_hat, res, M_inv, *, gamma, dead, norm_r, clip, bias=None,
                   mask=None, norm_channels="all", deadzone_mode="zero",
                   law="legacy", M=None, baseline="none", ki=0.05, raw_error=None,
                   state=None, rls_lambda=0.99, rls_p0=1.0, kf_q=None, kf_r=None):
    """Pure update, preserving the legacy operation order under default flags.

    For the original laws the mask selects norm channels, not the estimate.
    ``weighted_dob`` instead allocates only the selected input columns, with
    the bounds inside its weighted ridge solve. For ``innov`` the
    norm is of its existing innovation (after bias), not of the FIR residual.
    Its historical ``zero`` gate zeros the *step*, so it already holds an
    in-box estimate; ``hold`` also skips projection. Integral action has no
    deadzone/normaliser, as do the matched baselines. All estimates use the
    shared clip; correction_mask is applied by applied_correction, including
    for the matched baselines (unselected faults are still estimated).
    RLS/Kalman return new covariance state in diag['estimator_state']; callers
    must pass it into the next update and reset it at episode boundaries.
    ``attenuation`` is the multiplier corresponding to the division actually
    performed; None means there was no division. No input is mutated.
    """
    if norm_channels not in ("all", "corrected"):
        raise ValueError("norm_channels must be all or corrected")
    if deadzone_mode not in ("zero", "hold"):
        raise ValueError("deadzone_mode must be zero or hold")
    if law not in ("legacy", "innov") or baseline not in ("none", "integral", "weighted_dob") + MATCHED_BASELINES:
        raise ValueError("unknown estimator law or baseline")
    if baseline in MATCHED_BASELINES + ("weighted_dob",) and law != "legacy":
        raise ValueError("matched baselines cannot be combined with --law innov")
    f_hat, res = np.asarray(f_hat), np.asarray(res)
    dimension = len(f_hat)
    selected = np.ones(res.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != res.shape:
        raise ValueError("mask must have the residual's shape")
    diag = dict(nr=None, attenuation=None, deadzone_fired=None,
                update_applied=True, norm_vector=None)
    if baseline == "weighted_dob":
        from weighted_dob import weighted_dob_step
        if M is None or kf_r is None:
            raise ValueError("weighted_dob requires forward M and healthy residual covariance")
        updated, allocation = weighted_dob_step(f_hat, res, M, mask=selected, R=kf_r,
            gamma=gamma, clip=clip, state=state, prior_std=0.1, bias=bias)
        diag.update(allocation)
        return updated, diag
    if baseline in MATCHED_BASELINES:
        est = M_inv @ res
        if bias is not None:
            est = est - bias
        if baseline == "dob":
            if not np.isfinite(gamma) or not 0 < gamma <= 1:
                raise ValueError("dob requires 0 < gamma <= 1 (its alpha)")
            updated = (1.0 - gamma) * f_hat + gamma * est
        elif baseline == "integral_calibrated":
            # Literal requested comparator. Since r is independent of correction,
            # a constant nonzero fault causes drift to the clip, NOT convergence
            # to f. Integrating (est - f_hat) instead would just reproduce DOB.
            if not np.isfinite(ki) or ki <= 0:
                raise ValueError("integral_calibrated requires ki > 0")
            updated = f_hat + ki * est
        else:
            H = np.linalg.inv(M_inv) if M is None else np.asarray(M, dtype=float)
            if H.shape != (dimension, dimension) or not np.all(np.isfinite(H)):
                raise ValueError("M must be a finite square matrix matching the estimate")
            if baseline == "rls":
                if not np.isfinite(rls_lambda) or not 0 < rls_lambda <= 1:
                    raise ValueError("rls_lambda must be in (0, 1]")
                if not np.isfinite(rls_p0) or rls_p0 <= 0:
                    raise ValueError("rls_p0 must be positive")
                P = np.eye(dimension) * rls_p0 if state is None else state["covariance"]
                predicted = P / rls_lambda
                noise = np.eye(dimension)  # Standard block RLS: unweighted residual units.
            else:
                if kf_q is None or kf_r is None:
                    raise ValueError("kalman requires Q and healthy-residual R from kalman_noise")
                if np.shape(kf_q) != (dimension, dimension) or np.shape(kf_r) != (dimension, dimension):
                    raise ValueError("kalman Q and R must match the estimate dimension")
                P = np.eye(dimension) if state is None else state["covariance"]
                predicted = P + kf_q
                noise = kf_r
            gain, covariance = covariance_update(predicted, H, noise)
            measured = res if bias is None else res - H @ bias
            updated = f_hat + gain @ (measured - H @ f_hat)
            diag.update(gain=gain, effective_gain=gain @ H,
                        estimator_state=dict(covariance=covariance,
                            updates=1 if state is None else state["updates"] + 1))
        return np.clip(updated, -clip, clip), diag
    # Preserve precedence: --law innov historically overrides --baseline integral.
    if law == "innov":
        e = res - (M @ f_hat if M is not None else np.linalg.solve(M_inv, f_hat))
        if bias is not None:
            e = e - (M @ bias if M is not None else np.linalg.solve(M_inv, bias))
        nr = float(np.linalg.norm(e if norm_channels == "all" else e[selected]))
        fired = nr < dead
        diag.update(nr=nr, deadzone_fired=fired, norm_vector="innovation")
        if fired and deadzone_mode == "hold":
            diag["update_applied"] = False
            return f_hat.copy(), diag
        if fired:
            step = np.zeros(6)
        else:
            divisor = 1.0 + (nr / norm_r) ** 2
            step = (M_inv @ e) / divisor
            diag["attenuation"] = 1.0 / divisor
        return np.clip(f_hat + gamma * step, -clip, clip), diag
    if baseline == "integral":
        if raw_error is None:
            raise ValueError("integral action requires raw_error")
        return np.clip(f_hat + ki * raw_error, -clip, clip), diag
    est = M_inv @ res
    if bias is not None:
        est = est - bias
    nr = float(np.linalg.norm(res if norm_channels == "all" else res[selected]))
    fired = nr < dead
    diag.update(nr=nr, deadzone_fired=fired, norm_vector="residual")
    if fired and deadzone_mode == "hold":
        diag["update_applied"] = False
        return f_hat.copy(), diag
    if fired:
        est = np.zeros(6)
    divisor = 1.0 + (nr / norm_r) ** 2
    est = est / divisor
    diag["attenuation"] = 1.0 / divisor
    return np.clip(f_hat + gamma * (est - f_hat), -clip, clip), diag


def covariance_update(predicted, M, noise):
    """One block observation; Joseph form keeps the covariance symmetric/PSD."""
    gain = np.linalg.solve(M @ predicted @ M.T + noise, M @ predicted).T
    remaining = np.eye(predicted.shape[0]) - gain @ M
    covariance = remaining @ predicted @ remaining.T + gain @ noise @ gain.T
    return gain, (covariance + covariance.T) * 0.5


def healthy_residual_covariance(data, W, episodes=None):
    """Centered sample covariance in run()'s normalized FIR-residual units.

    Use only the healthy episodes selected for fitting, and only complete FIR
    windows (never cross an episode boundary). No faulted/evaluation samples.
    The supplied healthy bias remains shared by all laws; centering here only
    estimates noise, it does not silently change the estimator's bias setting.
    """
    records = data["records"] if isinstance(data, dict) else data
    A = np.asarray(records[0]["raw_a"], dtype=float)
    D = np.asarray(records[0]["raw_d"], dtype=float)
    lengths = records[0]["ep_len"]
    keep = set(range(len(lengths))) if episodes is None else set(episodes)
    if not keep or not keep.issubset(range(len(lengths))):
        raise ValueError("healthy covariance needs valid calibration episode indices")
    residuals, offset = [], 0
    for episode, length in enumerate(lengths):
        actions, measured = A[offset:offset + length], D[offset:offset + length] / OUT
        offset += length
        if episode not in keep:
            continue
        for t in range(K_FIR, length):
            H = actions[t - K_FIR:t + 1][::-1]
            feat = H.reshape(-1) if W.shape[1] > K_FIR + 2 else None
            pred = (W[:, :-1] @ feat + W[:, -1]) if feat is not None else np.array(
                [W[i, :K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])
            residuals.append(measured[t] - pred)
    if len(residuals) < 2:
        raise ValueError("healthy covariance needs at least two complete FIR residuals")
    residuals = np.asarray(residuals)
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(1e-12, float(eigenvalues[-1]) * 1e-9)
    regularized = (vectors * np.maximum(eigenvalues, floor)) @ vectors.T
    return regularized, dict(samples=len(residuals), episodes=sorted(keep),
        mean=residuals.mean(axis=0), sample_covariance=covariance,
        eigenvalue_floor=floor, covariance=regularized)


def kalman_noise(M_inv, healthy_covariance, *, gamma, q=None, r_scale=1.0):
    """R = r_scale * healthy covariance; Q is in fault/action units per update.

    Default Q = gamma^2/(1-gamma) * M_inv R M_inv.T makes the limiting
    effective gain K M = gamma I for full-rank M, directly matching DOB.
    Explicit --kf-q instead specifies isotropic Q = q I. Initial P = I.
    """
    if not np.isfinite(r_scale) or r_scale <= 0:
        raise ValueError("kf_r must be a positive healthy-covariance multiplier")
    R = np.asarray(healthy_covariance, dtype=float) * r_scale
    dimension = np.shape(M_inv)[0]
    if R.shape != (dimension, dimension) or not np.all(np.isfinite(R)) or not np.allclose(R, R.T):
        raise ValueError("healthy residual covariance must be finite, symmetric, and match M_inv")
    np.linalg.cholesky(R)
    if q is None:
        if not np.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError("automatic kf_q requires 0 < gamma < 1")
        Q = (gamma ** 2 / (1.0 - gamma)) * (M_inv @ R @ M_inv.T)
    else:
        if not np.isfinite(q) or q < 0:
            raise ValueError("kf_q must be nonnegative")
        Q = np.eye(dimension) * q
    return Q, R


def kalman_steady_state(M, Q, R):
    """Report residual gain K and dimensionless K M for a constant model."""
    if np.shape(M) != (6, 6) or not np.all(np.isfinite(M)) or np.linalg.matrix_rank(M) != 6:
        raise ValueError("Kalman steady-state reporting requires a full-rank 6 by 6 M")
    if not np.any(Q):
        return dict(gain=np.zeros((6, 6)), effective_gain=np.zeros((6, 6)),
                    covariance=np.zeros((6, 6)), iterations=0)
    P = np.eye(6)
    previous = None
    for iteration in range(1, 100001):
        gain, P = covariance_update(P + Q, M, R)
        effective = gain @ M
        if previous is not None and np.max(np.abs(effective - previous)) < 1e-12:
            return dict(gain=gain, effective_gain=effective, covariance=P,
                        iterations=iteration)
        previous = effective
    raise ValueError("Kalman steady-state gain did not converge in 100000 updates")


def _json_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_telemetry(stream, record):
    if stream is not None:
        stream.write(json.dumps(record, default=_json_default) + "\n")
        stream.flush()


@contextlib.contextmanager
def telemetry_stream(path, header):
    if path is None:
        yield None
        return
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        write_telemetry(stream, dict(type="header", **header))
        yield stream


def _module_constants(module):
    """Snapshot JSON-compatible public constants from the imported environment helpers."""
    constants = {}
    for key, value in vars(module).items():
        if key.isupper() and not key.startswith("_"):
            try:
                constants[key] = json.loads(json.dumps(value, default=_json_default))
            except TypeError:
                pass
    return constants


CALIB_EPISODES = None   # set by fit_plant; consumed by the held-out check in main()


AR_ORDER = 0   # set by --ar: number of past measured increments (own channel) in the plant regressor


def fit_plant(log_path, lam=1e-2, mimo=False, episodes=None, ar=None, dc=None):
    """MIMO FIR plant, identified on the NOMINAL episodes only.

    Each output dim is regressed on the last K_FIR+1 commands of ALL SIX inputs, not just its
    own. The plant is coupled -- wrist rotation is driven substantially by the translation
    commands, since moving the arm drags the wrist -- and a per-axis model cannot represent
    that. Leave-one-episode-out cross-validation, held-out R^2:

        dry  0.107 (per-axis) -> 0.615 (coupled + ridge)
        drx  0.214            -> 0.376
        translation unchanged at ~0.97

    Ridge because 43 parameters against ~270 samples overfits without it.

    DEFAULT OFF, because the better fit does not survive contact with the closed loop. With
    mimo=True the offset law scores 13/15 and estimates dy at -0.043 against a true +0.050 --
    a sign error the per-axis model never makes -- versus 14/15 and +0.038 with mimo=False.
    The coupled model has 43 parameters fitted on three nominal episodes, and the adaptive law
    runs precisely OUT of that distribution: the correction it applies moves the executed
    command away from the data the plant was identified on. The simpler model fits worse and
    extrapolates better, and extrapolation is what the law depends on.
    """
    d = json.loads(pathlib.Path(log_path).read_text())
    # error_signal.py used to emit a bare list. It now emits a dict carrying the same list
    # under "records" plus the (task, init) pairs the calibration consumed, so that
    # evaluation can be checked for overlap with it. Accept both shapes: every stored
    # artifact predates the change.
    global CALIB_EPISODES
    if isinstance(d, dict):
        CALIB_EPISODES = d.get("calib_episodes")
        d = d["records"]
    A = np.array(d[0]["raw_a"]); D = np.array(d[0]["raw_d"]); lens = d[0]["ep_len"]
    # episodes: optional subset of healthy-episode indices to fit on (calibration-size
    # ablation: how many healthy rollouts does the plant need?). None = all.
    keep = set(range(len(lens))) if episodes is None else set(episodes)
    ar = AR_ORDER if ar is None else int(ar)
    W = []
    for i in range(6):
        X, Y, o = [], [], 0
        for k, L in enumerate(lens):
            a, y = A[o:o+L], D[o:o+L, i] / OUT[i]; o += L
            if k not in keep:
                continue
            for t in range(max(K_FIR, ar), L):
                win = a[t-K_FIR:t+1][::-1]
                feat = win.reshape(-1) if mimo else win[:, i]
                # ARX: the channel's own past measured increments follow the command taps
                # (re4 theory Part 6: the r_y channel's slow pole is not representable by commands alone)
                arf = y[t-ar:t][::-1] if ar else np.zeros(0)
                X.append(np.concatenate([feat, arf, [1.0]]))
                Y.append(y[t])
        X, Y = np.array(X), np.array(Y)
        if dc is not None and i in dc and not mimo:
            # re4 theory record 50: every plant fitted on the closed-loop healthy log recovers an r_y DC
            # gain a third of the probed sensitivity, and the estimate settles at (fitted gain)/(probed
            # gain). Constrained ridge: minimise |Xw - Y|^2 + lam|w|^2 subject to sum(command taps) = dc[i]
            # (the probed M entry), solved through its KKT system; the AR terms and the intercept are free.
            n = X.shape[1]; c = np.zeros(n); c[:K_FIR + 1] = 1.0
            kkt = np.zeros((n + 1, n + 1)); kkt[:n, :n] = X.T @ X + lam * np.eye(n); kkt[:n, n] = c; kkt[n, :n] = c
            b = np.concatenate([X.T @ Y, [float(dc[i])]])
            W.append(np.linalg.solve(kkt, b)[:n])
            continue
        W.append(np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y))
    return np.array(W)                       # (6, 6*(K_FIR+1)+1)


def clip_report(vals, clip, name):
    """Warn when estimates sit on the projection bound: a value AT the clip is not a
    measurement, it is saturation, and four times so far it was nearly reported as one."""
    v = np.abs(np.asarray(vals, float))
    if v.size == 0:
        return
    rail = (np.abs(v - clip) < 1e-6).mean(axis=0)
    if np.any(rail > 0.2):
        ch = ["x", "y", "z", "rx", "ry", "rz"]
        hit = ", ".join(f"{ch[i]} {100*rail[i]:.0f}%" for i in range(len(rail)) if rail[i] > 0.2)
        print(f"  !! {name}: at the clip (+-{clip}) in >20% of episodes on [{hit}] -- "
              f"those channels are SATURATED, not estimated. Raise --clip above the expected fault.")


# Opt-in additions: a pose-referenced tracking term and a policy-free replay mode. Every flag
# defaults to off, and with all of them at these defaults run() and main() take exactly the code
# paths they took before the flags existed (no extra arithmetic, RNG draws, output fields or prints).
OPT_IN_DEFAULTS = dict(track_kappa=0.0, track_ref="dc", track_anchor=None, track_dims=None,
                       replay_log=None, replay_episodes=None)
# The tracking observation's mode (IMP2); recorded only when not at these defaults, so that rate-mode
# runs stay byte-identical to the runner before --track-mode existed.
TRACK_MODE_DEFAULTS = dict(track_mode="rate", track_leak=0.0)
# The tracking observation's map (IMP3); likewise recorded only when not at its default.
TRACK_OBS_DEFAULTS = dict(track_obs="full")


def fir_increment(W, H):
    """Increment an FIR plant without AR terms predicts for a (K_FIR+1, 6) newest-first command
    history -- run()'s own arithmetic, per-axis or MIMO by the width of W."""
    if W.shape[1] > K_FIR + 2:
        return W[:, :-1] @ H.reshape(-1) + W[:, -1]
    return np.array([W[i, :K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])


class PoseTracker:
    """The pose-referenced tracking term (--track-kappa > 0); run() builds one per episode.

    Everything is in run()'s normalised units (motion divided by OUT). At each anchor step -- the
    first rollout step, then every `anchor` steps (0: the first only) -- the reference pose is set
    to the measured pose before that step's command executes. Each step then advances the
    reference by W_ref's predicted increment for the NOMINAL policy command u_nom (not the sent
    command) and the pose by the measured motion: translation from robot0_eef_pos, rotation by
    accumulating the runner's rot_delta increments (its `measured` y[3:]). With e_p = pose -
    reference and n = steps since the anchor (1..anchor), z_T = M_inv @ (e_p / max(n, 1)), and
    after the base estimator update f_hat[d] <- clip(f_hat[d] + kappa z_T[d], -clip, +clip) on the
    tracked dims only.

    Fixed point with tracking alone on all six dims (all corrected), a constant fault f and a
    constant nominal command u, true DC map G and reference DC map G_ref:
    f_hat = f + (I - G^-1 G_ref) u - G^-1 b_ref -- the true offset only if the reference is
    DC-consistent on the directions the commands excite and its intercept b_ref matches the plant's;
    this assumes a stationary innovation. M_inv sets the convergence, not the limit. In position mode
    with uncorrected, untracked channels T (e.g. translation under a rotation-only mask), the full
    M_inv couples their pose error into the tracked channels R, and the fixed point retains
    e_R* = -inv(M_inv[R,R]) M_inv[R,T] e_T; the leak bounds e_T (|e_T| ~ d_T / leak). Note: when W_ref
    is the estimator's own plant (--track-ref dc on top of --dc-constrain), z_T equals the window mean
    of M_inv (r + W*c), i.e. an unattenuated, deadzone-free integrator of the runner's own residual,
    not an independent pose signal.

    mode="position" (--track-mode position) feeds back the retained pose offset itself: z_T = M_inv @
    e_p, no division by n, and after each step the reference leaks toward the measured pose,
    reference <- reference + leak (pose - reference), so e_p becomes a leaky sum of the per-step pose
    errors (sum, not mean, of M_inv (r + W*c) in the --track-ref dc on --dc-constrain case). The
    anchor rule is unchanged; e_p and z_T are those before this step's leak. A constant per-step
    model error d gives e_p = d (1 - (1-leak)^n) / leak <= d / leak, bounded, where leak = 0 integrates
    it without bound. With leak = 0 a converged loop has zero pose rate, so it removes the velocity
    error that a base estimator settling below the fault leaves, retaining only the bounded offset
    that balances that estimator; with leak > 0 the DC gain is kappa / leak, and against a base law
    pulling toward f_base at rate gamma the fixed point is (gamma f_base + (kappa/leak) f_track) /
    (gamma + kappa/leak), f_track being the tracking-alone limit above (the same in both modes).

    obs="tracked" (--track-obs tracked) restricts the observation to the tracked subspace R: with
    v = e_p (position) or e_p / max(n, 1) (rate), z_T[R] = solve(M[R,R], v[R]) and z_T[not R] = 0,
    M = inv(M_inv) being the probed sensitivity. The untracked channels' pose error then no longer
    enters z_T at all, where obs="full" feeds it in through M_inv[R, not R] -- e.g. an uncorrected
    translation error forcing r_x. The price is a DC bias: if the untracked channels carry a residual
    fault f_u, the tracked estimate settles at f_R + inv(M[R,R]) M[R,not R] f_u rather than f_R,
    because the physical coupling M[R,not R] f_u in e_R is now attributed to R.
    """

    def __init__(self, W_ref, M_inv, *, kappa, anchor, dims, clip, mode="rate", leak=0.0, obs="full"):
        W_ref, M_inv = np.asarray(W_ref, dtype=float), np.asarray(M_inv, dtype=float)
        if W_ref.ndim != 2 or W_ref.shape[0] != 6 or W_ref.shape[1] not in (K_FIR + 2, 6 * (K_FIR + 1) + 1):
            raise ValueError("the tracking reference must be a per-axis or MIMO FIR plant without AR terms")
        if M_inv.shape != (6, 6):
            raise ValueError("M_inv must be 6 by 6")
        dims = sorted({int(d) for d in dims})
        if not dims or dims[0] < 0 or dims[-1] > 5:
            raise ValueError("tracked dims must be a nonempty subset of 0..5")
        if not np.isfinite(kappa) or kappa < 0 or int(anchor) < 0:
            raise ValueError("kappa must be finite and nonnegative and anchor nonnegative")
        if mode not in ("rate", "position") or not np.isfinite(leak) or not 0 <= leak <= 1:
            raise ValueError("mode must be rate or position and leak lie in [0, 1]")
        if leak and mode != "position":
            raise ValueError("the reference leak belongs to position mode")
        if obs not in ("full", "tracked"):
            raise ValueError("obs must be full or tracked")
        self.W_ref, self.M_inv, self.kappa = W_ref, M_inv, float(kappa)
        self.anchor, self.dims, self.clip = int(anchor), np.array(dims), clip
        self.mode, self.leak, self.obs = mode, float(leak), obs
        if obs == "tracked":
            try:
                self.M_tracked = np.linalg.inv(M_inv)[np.ix_(dims, dims)]   # M[R,R], M = inv(M_inv)
            except np.linalg.LinAlgError:
                raise ValueError("--track-obs tracked needs an invertible M_inv") from None
            if not np.isfinite(np.linalg.cond(self.M_tracked)) or np.linalg.cond(self.M_tracked) > 1e8:
                raise ValueError("--track-obs tracked: M[R,R] on the tracked dims is singular or ill-conditioned")
        self.history = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
        self.origin, self.rotation, self.reference, self.n = None, np.zeros(3), np.zeros(6), 0

    def observe(self, step, u_nom, x0, x1, measured):
        """Account one executed step (`step` counts rollout steps from 0); returns (e_p, z_T, n)."""
        if self.origin is None or (self.anchor > 0 and step % self.anchor == 0):
            self.origin = np.array(x0, dtype=float)
            self.rotation, self.reference, self.n = np.zeros(3), np.zeros(6), 0
        self.n += 1
        self.history.appendleft(np.array(u_nom, dtype=float)[:6])
        self.reference = self.reference + fir_increment(self.W_ref, np.array(self.history))
        self.rotation = self.rotation + np.asarray(measured, dtype=float)[3:6]
        pose = np.concatenate([(np.asarray(x1, dtype=float) - self.origin) / OUT[:3], self.rotation])
        error = pose - self.reference
        if self.mode == "position":
            z = self.M_inv @ error if self.obs == "full" else self.tracked_observation(error)
            if self.leak:
                self.reference = self.reference + self.leak * (pose - self.reference)
            return error, z, self.n
        if self.obs == "tracked":
            return error, self.tracked_observation(error / max(self.n, 1)), self.n
        return error, self.M_inv @ (error / max(self.n, 1)), self.n

    def tracked_observation(self, v):
        """z_T[R] = solve(M[R,R], v[R]) on the tracked dims R, zero elsewhere (--track-obs tracked)."""
        z = np.zeros(6)
        z[self.dims] = np.linalg.solve(self.M_tracked, np.asarray(v, dtype=float)[self.dims])
        return z

    def correct(self, f_hat, z):
        """Tracked dims move and are projected onto the box; the others come back bit for bit."""
        corrected = np.array(f_hat, dtype=float)
        corrected[self.dims] = np.clip(corrected[self.dims] + self.kappa * np.asarray(z, dtype=float)[self.dims],
                                       -self.clip, self.clip)
        return corrected


class ReplayPolicy:
    """--replay-log: stands in for the policy client, so no policy server is ever contacted.

    The log is an error_signal.py artifact. As in re4_theory/paired_rollout.py, episode i of its
    first (nominal) record is the scenario episode_keys[i] = (task, init) with command rows
    starts[i] : starts[i] + ep_len[i], starts being the cumulative ep_len. infer() ignores the
    observation and returns the current episode's next replan-steps recorded commands -- raw_cmd,
    i.e. the recorded raw_a on the six arm dims (checked equal) plus the recorded gripper;
    control() is a no-op; start(i) rewinds to the i-th replayed episode.
    """

    def __init__(self, log_path, episodes=None, *, replan_steps):
        import hashlib
        raw = pathlib.Path(log_path).read_bytes()
        data = json.loads(raw)
        records = data.get("records") if isinstance(data, dict) else None
        if not records:
            raise ValueError("expected an error_signal.py log with a 'records' list")
        record = records[0]
        missing = [key for key in ("raw_a", "raw_cmd", "ep_len", "episode_keys") if key not in record]
        if missing:
            raise ValueError(f"record 0 lacks {missing}; replay needs full commands and scenario keys")
        commands = np.asarray(record["raw_cmd"], dtype=float)
        lengths = [int(n) for n in record["ep_len"]]
        keys = [(int(k[0]), int(k[1])) for k in record["episode_keys"]]
        if commands.shape != (sum(lengths), 7) or len(keys) != len(lengths) or min(lengths) < 1:
            raise ValueError("raw_cmd, ep_len and episode_keys disagree")
        if not np.array_equal(commands[:, :6], np.asarray(record["raw_a"], dtype=float)):
            raise ValueError("raw_cmd does not equal raw_a on the six arm dims")
        starts = np.cumsum([0] + lengths[:-1])
        self.episodes = list(range(len(lengths))) if episodes is None else [int(i) for i in episodes]
        if not self.episodes or any(not 0 <= i < len(lengths) for i in self.episodes):
            raise ValueError(f"replay episodes must lie in 0..{len(lengths) - 1}")
        self.scenarios = [keys[i] for i in self.episodes]
        self.lengths = [lengths[i] for i in self.episodes]
        self._commands = [commands[starts[i]:starts[i] + lengths[i]] for i in self.episodes]
        self.replan_steps = int(replan_steps)
        self.meta = dict(log=str(log_path), sha256=hashlib.sha256(raw).hexdigest(),
                         suite=data.get("suite"), reset_protocol=data.get("reset_protocol"),
                         record_label=record.get("label"), record_sev=record.get("sev"),
                         episodes=self.episodes, scenarios=self.scenarios, lengths=self.lengths,
                         commands="raw_cmd (recorded raw_a plus gripper), replan-steps rows per infer()")
        self._episode, self._cursor = None, 0

    def start(self, index):
        self._episode, self._cursor = int(index), 0

    def infer(self, observation):
        if self._episode is None:
            raise RuntimeError("replay: start(episode) must precede infer()")
        rows = self._commands[self._episode][self._cursor:self._cursor + self.replan_steps]
        if not len(rows):
            raise RuntimeError("replay: this episode's recorded commands are exhausted")
        self._cursor += len(rows)
        return {"actions": rows.copy()}

    def control(self, request):
        return None


def replay_probe(pp, a, replay):
    """paired_probe.Probe for --replay-log, built by its own __init__ (suite, per-suite MAXS, env
    cache, atexit close) with the websocket client swapped for the replay before it can connect;
    control() becomes the replay's no-op, so no control/ack file is ever touched."""
    websocket_client = pp._wc.WebsocketClientPolicy
    pp._wc.WebsocketClientPolicy = lambda *args, **kwargs: replay
    try:
        probe = pp.Probe(a)
    finally:
        pp._wc.WebsocketClientPolicy = websocket_client
    probe.control = replay.control
    return probe


def _opt_in_arguments(p, a):
    """Check --track-* and --replay-*; returns (tracked dims, or None when tracking is off; the
    ReplayPolicy, or None). With every opt-in flag at its default this changes nothing in `a`."""
    if not np.isfinite(a.track_kappa) or a.track_kappa < 0:
        p.error("--track-kappa must be finite and >= 0")
    if a.track_anchor is not None and a.track_anchor < 0:
        p.error("--track-anchor must be >= 0 (0 = anchor once at episode start)")
    if not np.isfinite(a.track_leak) or not 0 <= a.track_leak <= 1:
        p.error("--track-leak must lie in [0, 1]")
    if a.track_leak and a.track_mode != "position":
        p.error("--track-leak applies to --track-mode position only")
    dims = None
    if a.track_dims is not None:
        try:
            dims = sorted({int(x) for x in a.track_dims.split(",")})
        except ValueError:
            p.error("--track-dims must be comma-separated integers")
        if not dims or dims[0] < 0 or dims[-1] > 5:
            p.error("--track-dims must name channels in 0..5")
    if a.track_kappa > 0:
        if a.ar:
            p.error("--track-kappa is implemented for the FIR plant without AR terms")
        if a.track_ref == "dc" and a.mimo:
            p.error("--track-ref dc pins the per-axis FIR's DC gains and cannot be combined with --mimo")
        if dims is None:
            dims = [int(x) for x in a.corr_dims.split(",")] if a.corr_dims else list(range(6))
        a.track_anchor = a.replan_steps if a.track_anchor is None else a.track_anchor
        a.track_dims = ",".join(str(d) for d in dims)
    else:
        if a.track_mode != "rate" or a.track_leak:
            p.error("--track-mode position / --track-leak need --track-kappa > 0 (tracking is off)")
        if a.track_obs != "full":
            p.error("--track-obs tracked needs --track-kappa > 0 (tracking is off)")
        dims = None
    if a.replay_episodes is not None and a.replay_log is None:
        p.error("--replay-episodes requires --replay-log")
    replay = None
    if a.replay_log is not None:
        clashes = [flag for flag, used in (("--manifest", a.manifest is not None),
                                           ("--sampler-seed", a.sampler_seed is not None),
                                           ("--pin-rng", a.pin_rng), ("--wrist-shift", bool(a.wrist_shift)),
                                           ("--obs-offset", a.obs_offset is not None)) if used]
        if clashes:
            p.error(f"--replay-log replays recorded commands without a policy; {', '.join(clashes)} cannot apply")
        try:
            episodes = None if a.replay_episodes is None else [int(x) for x in a.replay_episodes.split(",")]
            replay = ReplayPolicy(a.replay_log, episodes, replan_steps=a.replan_steps)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            p.error(f"--replay-log {a.replay_log}: {error}")
        if replay.meta["suite"] is not None and replay.meta["suite"] != a.suite:
            p.error(f"--suite {a.suite} differs from the replay log's suite {replay.meta['suite']}")
        a.scenario_reset = True           # the recording's reset (libero_reset), as paired_rollout.py
        a.episodes = len(replay.episodes)
        a.replay_episodes = ",".join(str(i) for i in replay.episodes)
    return dims, replay


def run(pr, tid, init, sev, M_inv, W, gamma, adapt, max_steps=None, fvec=None, onset=0,
        obs_off=None, wrist_shift=0, static_c=None,
        dead=0.05, norm_r=0.5, clip=0.15, apply_corr=True, bias=None, corr_dims=None,
        law="legacy", M=None, profile="step", prof_p=60.0,
        baseline="none", ki=0.05, joint_fault=None, norm_channels="all",
        deadzone_mode="zero", telemetry=None, episode=0, arm=None,
        rls_lambda=0.99, rls_p0=1.0, kf_q=None, kf_r=None,
        step_observer=None, freeze_after=None, correction_scale=1.0, scenario_reset=False,
        gate=None, timing=None, track=None, adapt_from=0):
    # adapt_from: policy step (t - WARMUP_STEPS) before which the adaptive arm sends the raw policy
    # command and performs no estimator update (estimate held at zero; command history advances).
    # Registered for the FrozenYet Adaptive reacting-policy bridge (a common unadapted prefix, then an
    # imposed adaptation delay). 0 keeps the historical behaviour exactly.
    # Keep simulator/client dependencies out of the pure helpers and --selftest.
    # timing: an open line-buffered file; one JSON line per control step (re4 Part E):
    # policy_ms (replan steps only), adapter_ms (correction + FIR prediction + estimator update),
    # env_ms (simulator step), s2c_ms (observation acquired -> command issued), loop_ms.
    # gate: dict(b, sd, k) -- the healthy-phantom channel gate (prereg_records/PREREG_HEALTHY_GATE.md):
    # channel i is corrected at a step iff |f_hat_i - b_i| > k sd_i. Measured on healthy data only;
    # the estimator still runs on all six channels. Re-evaluated every step, no memory.
    # track: dict(W_ref, kappa, anchor, dims[, mode, leak, obs]) -- the opt-in pose tracking term (PoseTracker);
    # None or kappa 0 = off, and then no tracking code runs.
    import main as lm
    from so3 import rot_delta
    from joint_fault import JointFault
    from openpi_client import image_tools
    from gate_faults import apply_action_fault
    import paired_probe as _pp
    import time as _time
    max_steps = max_steps or _pp.MAXS
    env, desc, inits = pr.env_for(tid)
    if scenario_reset:
        from libero_reset import reset_libero
        obs, _ = reset_libero(env, inits[init], suite=pr.suite_name, task=tid, init=init)
    else:
        env.reset(); obs = env.set_init_state(inits[init])
    # A joint-level fault lives in the MuJoCo model, below the controller; applied after
    # the reset so set_init_state cannot undo it. Always restore on exit, including
    # success and exceptions: otherwise non-torque model faults compound on cached envs.
    jf = JointFault(env, joint_fault) if joint_fault else None
    try:
        if jf is not None:
            jf.apply()
        plan, t = collections.deque(), 0
        hist = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
        yhist = collections.deque([np.zeros(6)] * max(AR_ORDER, 1), maxlen=max(AR_ORDER, 1))
        f_hat = np.zeros(6)
        estimator_state = None
        mask = correction_mask(corr_dims)
        step_meta = dict(type="step", arm=arm or ("adaptive" if adapt else "frozen_faulted"),
                         episode=episode, task=int(tid), init=int(init))
        traj = []
        tracker = (PoseTracker(M_inv=M_inv, clip=clip, **track)
                   if track is not None and track["kappa"] > 0 else None)
        _t_obs = _time.perf_counter()
        while t < max_steps + WARMUP_STEPS:
            if t < WARMUP_STEPS:
                if telemetry is not None:
                    x0 = np.array(obs["robot0_eef_pos"], float)
                    q0 = np.array(obs["robot0_eef_quat"], float)
                obs, _, done, _ = env.step(lm.LIBERO_DUMMY_ACTION); _t_obs = _time.perf_counter()
                if telemetry is not None:
                    x1 = np.array(obs["robot0_eef_pos"], float)
                    q1 = np.array(obs["robot0_eef_quat"], float)
                    write_telemetry(telemetry, dict(step_meta, phase="warmup", t=t,
                        raw_action=None, correction=np.zeros(len(lm.LIBERO_DUMMY_ACTION)),
                        command=lm.LIBERO_DUMMY_ACTION,
                        measured=np.concatenate([x1 - x0, rot_delta(q0, q1)]) / OUT,
                        position=x1, quaternion=q1, r=None, nr=None, attenuation=None,
                        deadzone_fired=None, update_applied=False, norm_vector=None,
                        f_hat_before=f_hat, f_hat=f_hat, f_true=np.zeros(6), done=bool(done)))
                t += 1
                continue
            _tl0 = _time.perf_counter(); _wall0 = _time.time()
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), IMAGE_SIZE, IMAGE_SIZE))
            wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), IMAGE_SIZE, IMAGE_SIZE))
            if wrist_shift:
                # A misaligned wrist camera. This fault damages the POLICY's input while leaving
                # the plant untouched, so the command-motion residual stays clean -- the quadrant
                # the sensor-bias test could not reach, because that one did no damage.
                wr = np.roll(wr, wrist_shift, axis=1)
            _replan = not plan; _tp0 = _time.perf_counter()
            if not plan:
                plan.extend(pr.client.infer({
                    "observation/image": img, "observation/wrist_image": wr,
                    "observation/image_raw": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                    "observation/wrist_image_raw": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                    # obs_off: a constant bias on the POSITION SENSOR. The policy is misled about
                    # where the arm is. Crucially the residual is built from dx, a DIFFERENCE, so
                    # a constant sensor offset cancels exactly and is invisible to it -- this is a
                    # fault the method should be structurally unable to see.
                    "observation/state": np.concatenate((
                        np.array(obs["robot0_eef_pos"], float) + (obs_off if obs_off is not None else 0.0),
                        lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                    "prompt": str(desc)})["actions"][: pr.a.replan_steps])
            _policy_ms = (_time.perf_counter() - _tp0) * 1e3 if _replan else 0.0
            a_cmd = np.asarray(plan.popleft(), float)
            # apply_corr=False estimates but does NOT act. r = M f + eps(a+c): with an
            # imperfect plant the residual carries the model error evaluated at the operating
            # point, so once c moves that point the estimate chases its own correction. Freezing
            # c at zero separates open-loop model bias from that feedback.
            _tc0 = _time.perf_counter()
            gmask = None
            if gate is not None:
                gmask = (np.abs(f_hat - gate["b"]) > gate["k"] * gate["sd"]).astype(float)
            adapt_now = adapt and (t - WARMUP_STEPS) >= adapt_from
            c = applied_correction(f_hat, adapt=adapt_now, apply_corr=apply_corr,
                                   static_c=static_c,
                                   mask=gmask if gmask is not None else (mask if corr_dims is not None else None))
            if correction_scale != 1.0:
                c = c * correction_scale
            a_corr = a_cmd.copy(); a_corr[:6] += c                      # our correction
            _tc1 = _time.perf_counter()
            # onset > 0: the fault appears mid-episode. This is the deployment case -- a robot
            # that degrades while running -- and it tests the estimator as a TRACKER rather than
            # just asking whether it converges from step 1.
            live = (t >= onset)
            # Time-varying faults. A constant fault only asks whether the estimator CONVERGES;
            # a moving one asks whether it TRACKS, which is the deployment question -- hardware
            # degrades gradually (ramp), oscillates with load or temperature (sine), or drops in
            # and out with a loose connection (intermittent).
            scale = 1.0
            if live and profile != "step":
                u = t - onset
                if profile == "ramp":
                    scale = min(1.0, u / max(prof_p, 1e-9))          # linear drift to full
                elif profile == "sine":
                    scale = float(np.sin(2.0 * np.pi * u / max(prof_p, 1e-9)))
                elif profile == "sine_bias":
                    # A zero-mean sine averages to nothing over an episode and does no damage --
                    # frozen scored 20/20 on it (Sec 17.1), so the cell proved nothing. This is
                    # the oscillation a real load or thermal cycle produces: it swings between
                    # zero and full fault rather than symmetrically about zero.
                    scale = 0.5 * (1.0 + float(np.sin(2.0 * np.pi * u / max(prof_p, 1e-9))))
                elif profile == "intermittent":
                    scale = 1.0 if (int(u // max(prof_p, 1)) % 2 == 0) else 0.0
            f_true_now = (np.asarray(fvec, float) if fvec is not None
                          else np.full(6, sev)) * (scale if live else 0.0)
            if fvec is not None or profile != "step":
                # A structured fault: per-dim values instead of one scalar on every axis. The law
                # was built and tuned on a uniform +0.05, so recovering a pattern it has never
                # seen is the real generalisation test.
                a_exec = a_corr.copy()
                a_exec[:6] += f_true_now
            else:
                a_exec = apply_action_fault(a_corr, "offset", sev if live else 0.0, 6)       # then the world's fault
            x0 = np.array(obs["robot0_eef_pos"], float)
            q0 = np.array(obs["robot0_eef_quat"], float)
            joint_before = None
            if telemetry is not None or step_observer is not None:
                robot_env = env.env if hasattr(env, "env") else env
                robot = robot_env.robots[0]
                joint_before = np.asarray(robot._joint_positions, float).copy()
            if step_observer is not None:
                step_observer("before", dict(env=env, t=t, task=tid, init=init,
                    raw_action=a_cmd.copy(), correction=c.copy(), command=a_exec.copy(),
                    joint_position=joint_before, f_hat=f_hat.copy(), joint_fault=joint_fault))
            if jf is not None:
                jf.step(live)
            _t_cmd = _time.perf_counter()
            obs, _, done, _ = env.step(a_exec.tolist())
            _t_obs_prev = _t_obs if "_t_obs" in dir() else _t_cmd
            _t_obs = _time.perf_counter()
            x1 = np.array(obs["robot0_eef_pos"], float)
            q1 = np.array(obs["robot0_eef_quat"], float)
            y = np.concatenate([x1 - x0, rot_delta(q0, q1)]) / OUT

            u = a_corr[:6]                       # what we believe we sent (fault unknown to us)
            hist.appendleft(u)
            H = np.array(hist)                                          # (K+1, 6), newest first
            feat = H.reshape(-1) if W.shape[1] > K_FIR + 2 + AR_ORDER else None
            if AR_ORDER:
                YH = np.array(yhist)                                    # (AR, 6), newest first
                pred = np.array([W[i, :K_FIR + 1] @ H[:, i] + W[i, K_FIR + 1:K_FIR + 1 + AR_ORDER] @ YH[:, i] + W[i, -1] for i in range(6)])
            else:
                pred = (W[:, :-1] @ feat + W[:, -1]) if feat is not None else np.array(
                    [W[i, :K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])
            yhist.appendleft(y.copy())
            r = y - pred
            f_hat_before = f_hat.copy() if telemetry is not None else None
            diag = dict(nr=None, attenuation=None, deadzone_fired=None,
                        update_applied=False, norm_vector=None)
            if adapt_now and (freeze_after is None or t - WARMUP_STEPS < freeze_after):
                f_hat, diag = estimator_step(f_hat, r, M_inv, gamma=gamma, dead=dead,
                    norm_r=norm_r, clip=clip, bias=bias, mask=mask,
                    norm_channels=norm_channels, deadzone_mode=deadzone_mode,
                    law=law, M=M, baseline=baseline, ki=ki,
                    raw_error=y - a_corr[:6] if baseline == "integral" and law != "innov" else None,
                    state=estimator_state, rls_lambda=rls_lambda, rls_p0=rls_p0,
                    kf_q=kf_q, kf_r=kf_r)
                estimator_state = diag.get("estimator_state")
            if tracker is not None:
                # Pose tracking: measured every step, applied after the base update wherever it updates.
                track_e, track_z, track_n = tracker.observe(t - WARMUP_STEPS, a_cmd, x0, x1, y)
                if adapt_now and (freeze_after is None or t - WARMUP_STEPS < freeze_after):
                    f_hat = tracker.correct(f_hat, track_z)
                diag = dict(diag, track_e=track_e, track_z=track_z, track_n=track_n)
            _tad1 = _time.perf_counter()
            if timing is not None:
                timing.write(json.dumps(dict(step_meta, t=int(t - WARMUP_STEPS), wall=_wall0,
                    replan=bool(_replan), policy_ms=_policy_ms,
                    adapter_ms=((_tc1 - _tc0) + (_tad1 - _t_obs)) * 1e3,
                    env_ms=(_t_obs - _t_cmd) * 1e3, s2c_ms=(_t_cmd - _t_obs_prev) * 1e3,
                    loop_ms=(_tad1 - _tl0) * 1e3, live=bool(live),
                    f_true=np.asarray(f_true_now, float).round(6).tolist(),
                    correction=np.asarray(c, float).round(6).tolist(),
                    f_hat=np.asarray(f_hat, float).round(6).tolist())) + "\n")
            joint_after = None
            if telemetry is not None or step_observer is not None:
                joint_after = np.asarray(robot._joint_positions, float).copy()
            if step_observer is not None:
                step_observer("after", dict(env=env, t=t, task=tid, init=init,
                    raw_action=a_cmd.copy(), correction=c.copy(), command=a_exec.copy(),
                    joint_before=joint_before, joint_position=joint_after,
                    measured=y.copy(), r=r.copy(), f_hat=f_hat.copy(), done=bool(done)))
            if telemetry is not None:
                full_correction = np.zeros_like(a_cmd)
                full_correction[:6] = c
                write_telemetry(telemetry, dict(step_meta, phase="rollout", t=t,
                    raw_action=a_cmd, correction=full_correction, command=a_exec,
                    nominal_command=a_corr, measured=y, position=x1, quaternion=q1,
                    r=r, f_hat_before=f_hat_before, f_hat=f_hat,
                    joint_before=joint_before, joint_position=joint_after,
                    joint_velocity=np.asarray(robot._joint_velocities, float),
                    joint_torque=np.asarray(robot.torques, float),
                    controller_goal_position=np.asarray(robot.controller.goal_pos, float),
                    controller_goal_orientation=np.asarray(robot.controller.goal_ori, float),
                    f_true=f_true_now, live=bool(live), done=bool(done), **diag))
            traj.append(dict(t=t, f_hat=f_hat.tolist(), r=r.tolist(), live=bool(live),
                             f_true=f_true_now.tolist()))
            if gmask is not None:
                traj[-1]["gate"] = gmask.tolist()
            if "gain" in diag:
                traj[-1].update(gain=diag["gain"].tolist(),
                                effective_gain=diag["effective_gain"].tolist())
            if done:
                return True, f_hat, traj
            t += 1
        return False, f_hat, traj
    finally:
        if jf is not None:
            jf.restore()


def selftest():
    """Synthetic checks only: stdlib + NumPy, with no rollout/client imports."""
    import io

    def original(f_hat, r, M_inv, gamma, dead, norm_r, clip, bias=None,
                 law="legacy", M=None, baseline="none", ki=0.05, raw_error=None):
        # The pre-refactor arithmetic, intentionally independent of estimator_step.
        if law == "innov":
            e = r - (M @ f_hat if M is not None else np.linalg.solve(M_inv, f_hat))
            if bias is not None:
                e = e - (M @ bias if M is not None else np.linalg.solve(M_inv, bias))
            ne = float(np.linalg.norm(e))
            if ne < dead:
                step = np.zeros(6)
            else:
                step = (M_inv @ e) / (1.0 + (ne / norm_r) ** 2)
            return np.clip(f_hat + gamma * step, -clip, clip)
        if baseline == "integral":
            return np.clip(f_hat + ki * raw_error, -clip, clip)
        est = M_inv @ r
        if bias is not None:
            est = est - bias
        nr = float(np.linalg.norm(r))
        if nr < dead:
            est = np.zeros(6)
        est = est / (1.0 + (nr / norm_r) ** 2)
        return np.clip(f_hat + gamma * (est - f_hat), -clip, clip)

    def identical(actual, expected):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), (actual, expected)

    rng = np.random.default_rng(20260907)
    mask = correction_mask([0, 2, 4])
    count = 0
    for i in range(80):
        M = np.eye(6) + rng.normal(size=(6, 6)) * 0.05
        M_inv = np.linalg.pinv(M)
        f_hat = rng.normal(size=6) * 0.2
        r = rng.normal(size=6) * (0.0001 if i % 2 else 0.4)
        bias = rng.normal(size=6) * 0.01 if i % 3 else None
        opts = dict(gamma=[0.0, 0.05, 0.25, 1.0][i % 4], dead=0.05,
                    norm_r=0.5, clip=0.15, bias=bias)
        for extra in ({}, {"law": "innov", "M": M}, {"law": "innov"},
                      {"baseline": "integral", "raw_error": r, "ki": 0.05},
                      {"law": "innov", "M": M, "baseline": "integral"}):
            before = [v.copy() for v in (f_hat, r, M_inv, mask)]
            expected = original(f_hat, r, M_inv, **opts, **extra)
            actual, diag = estimator_step(f_hat, r, M_inv, mask=mask, **opts, **extra)
            identical(actual, expected)
            for actual_input, snapshot in zip((f_hat, r, M_inv, mask), before):
                identical(actual_input, snapshot)
            if not extra:
                nr = float(np.linalg.norm(r))
                assert diag["nr"] == nr and diag["deadzone_fired"] == (nr < opts["dead"])
                assert diag["attenuation"] == 1.0 / (1.0 + (nr / opts["norm_r"]) ** 2)
            count += 1

    opts = dict(gamma=0.25, dead=0.05, norm_r=0.5, clip=0.15)
    r = np.array([0.001, 9.0, 0.002, 0.0, 0.003, 0.0])
    _, all_diag = estimator_step(np.zeros(6), r, np.eye(6), mask=mask, **opts)
    _, corrected_diag = estimator_step(np.zeros(6), r, np.eye(6), mask=mask,
                                       norm_channels="corrected", **opts)
    assert corrected_diag["nr"] == float(np.linalg.norm(r[mask.astype(bool)]))
    assert corrected_diag["nr"] < all_diag["nr"]
    assert corrected_diag["deadzone_fired"] and not all_diag["deadzone_fired"]
    assert corrected_diag["attenuation"] > all_diag["attenuation"]
    # Strict '<': a residual exactly on the gate threshold is not gated.
    _, boundary = estimator_step(np.zeros(6), np.array([0.125, 0, 0, 0, 0, 0]),
        np.eye(6), **dict(opts, dead=0.125))
    assert not boundary["deadzone_fired"]

    f_hat = np.array([0.25, -0.25, -0.0, 0.03125, -0.0625, 0.125])
    for law in ("legacy", "innov"):
        residual = np.zeros(6) if law == "legacy" else f_hat.copy()
        held, diag = estimator_step(f_hat, residual, np.eye(6), law=law,
            deadzone_mode="hold", mask=mask, **opts)
        identical(held, f_hat)  # Includes negative zero and values outside projection.
        assert diag["deadzone_fired"] and not diag["update_applied"]
        assert diag["attenuation"] is None
    initial = np.array([0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125])
    decayed = initial.copy()
    for firing in range(1, 5):
        expected = decayed + opts["gamma"] * (np.zeros(6) - decayed)
        decayed, diag = estimator_step(decayed, np.zeros(6), np.eye(6), **opts)
        identical(decayed, expected)
        identical(decayed, initial * (1.0 - opts["gamma"]) ** firing)
        assert diag["deadzone_fired"] and diag["update_applied"]

    # The exact helper called by run() masks both adaptive and static corrections.
    values = np.arange(1.0, 7.0)
    for adapt in (False, True):
        for apply_corr in (False, True):
            for static_c in (None, values * 0.01):
                expected = (static_c.copy() if adapt else np.zeros(6)) if static_c is not None \
                    else (-values if adapt and apply_corr else np.zeros(6))
                identical(applied_correction(values, adapt=adapt, apply_corr=apply_corr,
                    static_c=static_c), expected)
                masked = applied_correction(values, adapt=adapt, apply_corr=apply_corr,
                                             static_c=static_c, mask=mask)
                identical(masked, expected * mask)
                assert np.all(masked[mask == 0] == 0)
    a_cmd = np.arange(7.0)
    command = a_cmd.copy()
    command[:6] += applied_correction(values, adapt=True, mask=mask)
    assert command[6] == a_cmd[6]
    assert np.array_equal(command[:6][mask == 0], a_cmd[:6][mask == 0])

    class FlushedStream(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    stream = FlushedStream()
    write_telemetry(stream, dict(type="header", argv=["--selftest"],
                                args=dict(out=pathlib.Path("synthetic.json")), constants=dict(OUT=OUT)))
    write_telemetry(stream, dict(type="step", episode=0, t=np.int64(10), raw_action=a_cmd,
        correction=np.r_[applied_correction(values, adapt=True, mask=mask), 0.0],
        command=command, measured=r, r=r, f_hat=decayed, f_true=np.zeros(6), **diag))
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert stream.flushes == 2 and len(records) == 2
    assert records[0]["args"]["out"] == "synthetic.json"
    assert records[1]["t"] == 10 and records[1]["command"] == command.tolist()
    assert records[1]["correction"][6] == 0.0
    with telemetry_stream(None, None) as disabled:
        assert disabled is None
        write_telemetry(disabled, {"not_serializable": object()})
    matched_selftest()
    print(f"LIBERO selftest passed: {count} exact legacy/innovation/integral comparisons; "
          "masked norms, strict gate, bit-identical hold, exact decay, correction mask, JSONL flush")


def matched_selftest():
    """Exercise the executed helpers, including covariance fitting and persistence.

    Rates use sustained 5% relative Euclidean error, not a first crossing:
    calibrated integral briefly crosses the truth on its way to saturation.
    Steady-state variance means actual estimation error for a CONSTANT fault,
    computed from the gain's Lyapunov equation, not the KF random-walk P.
    """
    rng = np.random.default_rng(812)
    U = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    M = U @ np.diag([0.18, 0.21, 0.23, 0.29, 0.34, 0.4]) @ V.T
    M_inv = np.linalg.inv(M)
    fault = np.array([0.04, -0.03, 0.02, -0.05, 0.035, -0.025])
    bias = np.array([0.01, -0.005, 0.006, 0.002, -0.004, 0.003])

    # Synthetic healthy logs exercise diagonal AND MIMO FIRs, nonzero intercepts,
    # OUT normalization, episode selection, centering and incomplete windows.
    healthy_cov = None
    for mimo in (False, True):
        lengths = [46, 51, 43]
        actions = rng.normal(size=(sum(lengths), 6))
        W = rng.normal(size=(6, 6 * (K_FIR + 1) + 1 if mimo else K_FIR + 2)) * 0.02
        measured = np.zeros_like(actions)
        expected, offset = [], 0
        for ep, length in enumerate(lengths):
            for t in range(length):
                if t < K_FIR:
                    measured[offset + t] = 1e5  # Must not enter covariance.
                    continue
                H = actions[offset + t - K_FIR:offset + t + 1][::-1]
                pred = W[:, :-1] @ H.reshape(-1) + W[:, -1] if mimo else np.array(
                    [W[i, :-1] @ H[:, i] + W[i, -1] for i in range(6)])
                noise = rng.normal(size=6) @ (np.eye(6) + 0.2 * U) * 0.02
                residual = M @ bias + noise + (100.0 if ep == 1 else 0.0)
                measured[offset + t] = (pred + residual) * OUT
                if ep != 1:
                    expected.append(residual)
            offset += length
        data = [dict(raw_a=actions.tolist(), raw_d=measured.tolist(), ep_len=lengths)]
        cov, meta = healthy_residual_covariance(data, W, episodes=[0, 2])
        wrapped, _ = healthy_residual_covariance(dict(records=data), W, episodes=[0, 2])
        assert meta["samples"] == sum(lengths[i] - K_FIR for i in (0, 2))
        assert np.allclose(meta["mean"], np.mean(expected, axis=0), atol=1e-12)
        assert np.allclose(cov, np.cov(expected, rowvar=False), atol=1e-12)
        assert np.array_equal(cov, wrapped)
        healthy_cov = cov
    # A noiseless/short healthy log must still yield a usable, documented R floor.
    zeros = np.zeros((K_FIR + 2, 6))
    cov, meta = healthy_residual_covariance(
        [dict(raw_a=zeros, raw_d=zeros, ep_len=[len(zeros)])], np.zeros((6, K_FIR + 2)))
    assert np.array_equal(cov, np.eye(6) * meta["eigenvalue_floor"])

    alpha, forgetting, steps = 0.01, 0.99, 3000
    Q, R = kalman_noise(M_inv, healthy_cov, gamma=alpha)
    explicit_Q, scaled_R = kalman_noise(M_inv, healthy_cov, gamma=alpha, q=0.002, r_scale=2)
    assert np.array_equal(explicit_Q, np.eye(6) * 0.002)
    assert np.array_equal(scaled_R, healthy_cov * 2)
    steady = kalman_steady_state(M, Q, R)
    assert np.allclose(steady["effective_gain"], alpha * np.eye(6), atol=1e-9)
    assert not np.any(kalman_steady_state(M, Q * 0, R)["gain"])
    try:
        kalman_steady_state(np.zeros((6, 6)), Q * 0, R)
    except ValueError:
        pass
    else:
        raise AssertionError("singular M must not report zero unobservable covariance")
    opts = dict(gamma=alpha, dead=0.0, norm_r=0.03, clip=0.15,
                bias=bias, M=M, rls_lambda=forgetting, rls_p0=1.0, kf_q=Q, kf_r=R)
    residual = M @ (fault + bias)

    def trace(baseline, *, initial_state=None, **overrides):
        estimate, state = np.zeros(6), initial_state
        values = []
        for _ in range(steps):
            estimate, diag = estimator_step(estimate, residual, M_inv,
                baseline=baseline, state=state, **dict(opts, **overrides))
            state = diag.get("estimator_state")
            values.append(estimate)
        return np.asarray(values), diag

    def settling_time(values, target):
        outside = np.flatnonzero(np.linalg.norm(values - target, axis=1)
                                 > 0.05 * np.linalg.norm(target))
        return None if len(outside) and outside[-1] == len(values) - 1 else \
            (int(outside[-1]) + 2 if len(outside) else 1)

    histories, diagnostics, rates = {}, {}, {}
    for name in ("none", *MATCHED_BASELINES):
        histories[name], diagnostics[name] = trace(name)
        rates[name] = settling_time(histories[name], fault)
    attenuation = 1.0 / (1.0 + (np.linalg.norm(residual) / opts["norm_r"]) ** 2)
    assert np.allclose(histories["none"][-1], fault * attenuation, atol=1e-12, rtol=0)
    assert rates["none"] is None
    legacy_rate = settling_time(histories["none"], fault * attenuation)
    assert legacy_rate == rates["dob"]
    # With b=0 this is exactly the specified f/(1+||Mf||^2/rho^2).
    zero_bias_residual = M @ fault
    zero_bias_estimate = np.zeros(6)
    for _ in range(steps):
        zero_bias_estimate, _ = estimator_step(zero_bias_estimate, zero_bias_residual,
            M_inv, **dict(opts, bias=None))
    zero_bias_attenuation = 1.0 / (1.0 + (np.linalg.norm(zero_bias_residual) / opts["norm_r"]) ** 2)
    assert np.allclose(zero_bias_estimate, fault * zero_bias_attenuation, atol=1e-12, rtol=0)
    innov, _ = trace("none", law="innov")
    assert np.allclose(innov[-1], fault, atol=1e-12, rtol=0)
    for name in ("dob", "rls", "kalman"):
        assert np.allclose(histories[name][-1], fault, atol=1e-12, rtol=0), name
        assert rates[name] is not None
    assert rates["dob"] == int(np.ceil(np.log(0.05) / np.log(1.0 - alpha)))
    # RLS with lambda=1 is ordinary least squares with its finite initial prior.
    unforgotten, _ = trace("rls", rls_lambda=1.0)
    information = np.eye(6) + steps * (M.T @ M)
    assert np.allclose(unforgotten[-1], np.linalg.solve(information,
        steps * (M.T @ M) @ fault), atol=1e-12)

    integral = histories["integral_calibrated"]
    expected_integral = np.clip(np.arange(1, steps + 1)[:, None] * 0.05 * fault,
                                -opts["clip"], opts["clip"])
    assert np.allclose(integral, expected_integral, atol=1e-12, rtol=0)
    assert np.array_equal(integral[-1], np.sign(fault) * opts["clip"])
    assert rates["integral_calibrated"] is None
    rail_time = int(np.flatnonzero(np.all(np.abs(integral) == opts["clip"], axis=1))[0]) + 1
    assert abs(rail_time - np.ceil(opts["clip"] / (0.05 * np.min(np.abs(fault))))) <= 1

    # Validate the actual limiting gains against stationary constant-fault
    # variance: C = (I-KM) C (I-KM)' + K R K'. Shared R includes cross-covariance.
    reference_variance = alpha / (2.0 - alpha) * (M_inv @ R @ M_inv.T)
    for name in ("rls", "kalman"):
        gain = diagnostics[name]["gain"]
        assert np.allclose(gain @ M, alpha * np.eye(6), atol=1e-11)
        A = np.eye(6) - gain @ M
        stationary = np.linalg.solve(np.eye(36) - np.kron(A, A),
                                     (gain @ R @ gain.T).reshape(-1)).reshape(6, 6)
        assert np.allclose(stationary, reference_variance, rtol=1e-8, atol=1e-14)
        assert rates[name] < rates["dob"]  # Specific startup case, not a universal claim.
        settled, _ = trace(name, initial_state=diagnostics[name]["estimator_state"])
        assert settling_time(settled, fault) == rates["dob"]

    # Identical noisy observations through the executed update path at steady
    # gains must give the same variance, not merely agree in a Riccati formula.
    noisy_estimates = {name: np.zeros(6) for name in ("dob", "rls", "kalman")}
    states = {name: diagnostics[name].get("estimator_state") for name in noisy_estimates}
    noise_samples = rng.multivariate_normal(np.zeros(6), R, size=300)
    for noise in noise_samples:
        for name in noisy_estimates:
            noisy_estimates[name], diag = estimator_step(noisy_estimates[name],
                residual + noise, M_inv, baseline=name, state=states[name], **opts)
            states[name] = diag.get("estimator_state")
        for name in ("rls", "kalman"):
            assert np.allclose(noisy_estimates[name], noisy_estimates["dob"], atol=1e-11, rtol=0)

    # All new rules ignore the robustness gates, share bias subtraction and
    # clipping, and retain full observations under a correction-only mask.
    for name in MATCHED_BASELINES:
        for mask in (correction_mask(), correction_mask([0, 2, 4]), correction_mask([])):
            initial = np.array([0.1, -0.12, 0.08, -0.06, 0.04, -0.02])
            large_residual = M @ (fault * 1000 + bias)
            state = diagnostics[name].get("estimator_state")
            snapshots = [x.copy() for x in (initial, large_residual, M_inv, bias, mask)]
            state_before = json.dumps(state, default=_json_default)
            estimate, diag = estimator_step(initial, large_residual, M_inv,
                baseline=name, state=state, mask=mask, norm_channels="corrected",
                deadzone_mode="hold", **dict(opts, dead=1e9, norm_r=1e-12))
            assert np.all(np.abs(estimate) <= opts["clip"])
            assert np.any(np.abs(estimate) == opts["clip"])
            correction = applied_correction(estimate, adapt=True, mask=mask)
            assert np.all(correction[mask == 0] == 0)
            assert np.array_equal(correction, -estimate * mask)
            assert diag["nr"] is None and diag["attenuation"] is None
            assert diag["deadzone_fired"] is None and diag["update_applied"]
            unmasked, _ = estimator_step(initial, large_residual, M_inv,
                baseline=name, state=state, **opts)
            assert np.array_equal(estimate, unmasked)
            for value, snapshot in zip((initial, large_residual, M_inv, bias, mask), snapshots):
                assert value.tobytes() == snapshot.tobytes()
            assert state_before == json.dumps(state, default=_json_default)
            zero, _ = estimator_step(np.zeros(6), M @ bias, M_inv,
                baseline=name, **opts)
            assert np.allclose(zero, 0, atol=1e-15)
    # RLS covariance and estimate agree with an independent weighted batch fit.
    estimate, state, information, rhs = np.zeros(6), None, np.eye(6), np.zeros(6)
    for noise in noise_samples[:12]:
        observed = residual + noise
        information = forgetting * information + M.T @ M
        rhs = forgetting * rhs + M.T @ (observed - M @ bias)
        estimate, diag = estimator_step(estimate, observed, M_inv, baseline="rls",
                                        state=state, **dict(opts, clip=10.0))
        state = diag["estimator_state"]
        assert np.allclose(estimate, np.linalg.solve(information, rhs), atol=1e-12)
        assert np.allclose(state["covariance"], np.linalg.inv(information), atol=1e-12)
    # One Kalman update checked independently against the posterior information form.
    predicted = np.eye(6) + Q
    posterior = np.linalg.inv(np.linalg.inv(predicted) + M.T @ np.linalg.solve(R, M))
    expected = posterior @ M.T @ np.linalg.solve(R, residual - M @ bias)
    estimate, diag = estimator_step(np.zeros(6), residual, M_inv, baseline="kalman", **opts)
    assert np.allclose(estimate, expected, atol=1e-12)
    assert np.allclose(diag["estimator_state"]["covariance"], posterior, atol=1e-12)

    print(f"Matched synthetic fault: {fault.tolist()}; sustained 5% error criterion")
    print(f"  legacy fixed point: {np.round(histories['none'][-1], 8).tolist()} "
          f"({attenuation:.8f}*f with supplied bias; {zero_bias_attenuation:.8f}*f at b=0); "
          f"5% to own fixed point={legacy_rate} updates, never 5% to f")
    print(f"  innov fixed point=f; 5% to f={settling_time(innov, fault)} updates")
    for name in ("dob", "rls", "kalman"):
        print(f"  {name}: fixed point=f; 5% to f={rates[name]} updates; "
              f"asymptotic error factor={1-alpha:.2f}")
    print(f"  integral_calibrated: NO unbiased fixed point; drift=ki*f/update; "
          f"fixed point={integral[-1].tolist()}, all rails by update {rail_time}")
    print(f"  matched constant-fault stationary variance trace={np.trace(reference_variance):.9g}; "
          f"EMA alpha={alpha}, RLS lambda={forgetting}/P0=1, KF P0=I, K*M={alpha}*I")
    print(f"  RLS/KF startup faster here; after covariance settles BOTH take {rates['dob']} "
          "updates, equal to EMA (no tracking-speed advantage at matched variance).")
    print("  healthy diagonal/MIMO covariance, bias, mask, clip, immutable state, "
          "weighted batch RLS and Kalman posterior checks passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true", help="run synthetic checks without a simulator")
    p.add_argument("--control", type=pathlib.Path)
    p.add_argument("--ack", type=pathlib.Path)
    p.add_argument("--log", type=pathlib.Path)
    p.add_argument("--openloop", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path)
    p.add_argument("--telemetry", type=pathlib.Path, default=None,
                   help="optional streamed per-step JSONL with a self-describing header")
    p.add_argument("--norm-channels", choices=["all", "corrected"], default="all",
                   help="channels in the residual norm (innovation norm for --law innov)")
    p.add_argument("--deadzone-mode", choices=["zero", "hold"], default="zero",
                   help="zero: legacy zero target (decays f_hat); hold: skip the update. "
                        "innov's existing zero gate zeros its step; integral has no gate.")
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--baseline", choices=["none", "integral", *MATCHED_BASELINES, "weighted_dob"], default="none",
                   help="integral: historical raw-error integral; dob: calibrated EMA "
                        "(alpha=gamma); integral_calibrated: accumulate M_inv*r-bias "
                        "(drifts to clip for a constant fault); rls/kalman: covariance "
                        "estimators. Matched baselines require --law legacy and share "
                        "the FIR residual, bias, clip, correction mask and update rate.")
    p.add_argument("--ki", type=float, default=0.05,
                   help="integral gain for integral or integral_calibrated")
    p.add_argument("--rls-lambda", type=float, default=0.99,
                   help="RLS forgetting factor in (0,1]; limiting K*M=(1-lambda)*I")
    p.add_argument("--rls-p0", type=float, default=1.0,
                   help="RLS initial covariance = p0*I in fault units")
    p.add_argument("--kf-q", type=float, default=None,
                   help="Kalman process covariance Q=q*I in fault units per update; "
                        "default: gamma^2/(1-gamma)*M_inv*R*M_inv.T, matching DOB gain")
    p.add_argument("--kf-r", type=float, default=1.0,
                   help="positive multiplier of the centered healthy FIR-residual "
                        "covariance R (default 1); Kalman initial covariance is I")
    p.add_argument("--profile", choices=["step", "ramp", "sine", "sine_bias", "intermittent"],
                   default="step", help="how the fault varies in time")
    p.add_argument("--prof-p", type=float, default=60.0,
                   help="ramp length / sine period / intermittent half-period, in steps")
    p.add_argument("--dc-constrain", choices=["none", "corrected", "all"], default="none",
                   help="refit the per-axis FIR with its command-tap sum (DC gain) constrained to the probed "
                        "M diagonal on the corrected channels or on all six (re4 theory record 50: the "
                        "closed-loop fit under-reads the r_y offset gain 2.7x and the estimate settles at "
                        "fitted/probed); none = the unconstrained fit")
    p.add_argument("--m-diag", default=None,
                   help="override diagonal entries of the loaded sensitivity M, e.g. '0=0.257,1=0.263,2=0.291' (six-channel "
                        "correction on libero_10: the shipped z entry 0.126 is a single-state artefact, records 45 and E2 probes)")
    p.add_argument("--dc-gain", default=None,
                   help="explicit command-tap-sum constraints per channel, e.g. '4=0.254': the per-axis FIR is refitted with "
                        "those channels' tap sums pinned to the given values (E2: a separately probed, qualified local "
                        "response; PREREG_E2_PROBE_QUALIFICATION.md). Mutually exclusive with --dc-constrain")
    p.add_argument("--law", choices=["legacy", "innov"], default="legacy",
                   help="legacy: normalise the estimate (biased low ~5%%). "
                        "innov: normalise the step, unbiased fixed point.")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--static-corr", default=None,
                   help="fixed 6-vector correction; the no-estimation baseline")
    p.add_argument("--wrist-shift", type=int, default=0,
                   help="pixels to roll the wrist camera -- a fault the plant never sees")
    p.add_argument("--obs-offset", default=None,
                   help="x,y,z bias on the position SENSOR -- a fault the residual cannot see")
    p.add_argument("--adapt-from", type=int, default=0,
                   help="policy step (after the warmup) at which the adaptive arm starts correcting and "
                        "updating; before it the raw policy command is sent and the estimate stays zero")
    p.add_argument("--onset", type=int, default=0,
                   help="control step at which the fault appears (0 = from the start)")
    p.add_argument("--eval-init", type=int, default=45,
                   help="first evaluation initial state (keep fresh per run)")
    p.add_argument("--fault-vec", default=None,
                   help="6 comma-separated per-dim fault values (overrides --sev)")
    p.add_argument("--joint-fault", default=None,
                   help="kind:joint:magnitude, a fault BELOW the controller in the MuJoCo model "
                        "(torque N.m bias, friction, damping, gain scale, lock +-rad); see joint_fault.py")
    p.add_argument("--sampler-seed", type=int, default=None,
                   help="explicit policy-sampler seed (unified plan, protocol item 4): the server draws call i of "
                        "episode e with fold_in(fold_in(key(seed), e), i); the runner re-issues the control at every "
                        "episode start so both arms share the schedule. Recorded per episode. Incompatible with --pin-rng.")
    p.add_argument("--arms", choices=["both", "frozen", "adaptive"], default="both",
                   help="which arms to run (a manifest design runs the frozen arm once and pairs it by key)")
    p.add_argument("--manifest", type=pathlib.Path, default=None,
                   help="JSON scenario manifest: {'scenarios': [{'task': t, 'init': i, 'sampler_seed': s}, ...]} "
                        "replaces --episodes/--eval-init/--task-stride; per-scenario sampler seeds override --sampler-seed")
    p.add_argument("--pin-rng", action="store_true",
                   help="pin the policy server's flow-sampler noise (one key per call) so two runs draw identical "
                        "actions given identical observations (re4 theory Part 8.2)")
    p.add_argument("--ar", type=int, default=0,
                   help="ARX plant: this many past measured increments of the channel's own motion in the "
                        "regressor (re4 theory Part 6); M is rescaled per channel by (1 - sum of AR coefficients)")
    p.add_argument("--fir-k", type=int, default=None,
                   help="FIR order K for the plant (default the module's K_FIR=6); 0 = static observer")
    p.add_argument("--timing", type=pathlib.Path, default=None,
                   help="per-step timing JSONL (re4 Part E)")
    p.add_argument("--scenario-reset", action="store_true",
                   help="reset through libero_reset.reset_libero: clears applied/external forces, "
                        "seeds the cached env per scenario and records the model/state fingerprint, "
                        "so paired arms provably start from the same physical scene (audit protocol)")
    p.add_argument("--gate-stats", type=pathlib.Path, default=None,
                   help="healthy phantom stats (gate_stats.py): correct channel i iff |f_hat_i-b_i| > k sd_i")
    p.add_argument("--gate-k", type=float, default=3.0)
    p.add_argument("--corr-dims", default=None,
                   help="comma-separated dims to correct, e.g. 3,4,5 for rotation only")
    p.add_argument("--bias", default=None,
                   help="6 comma-separated values: the estimator bias measured on healthy runs")
    p.add_argument("--estimate-only", action="store_true",
                   help="update f_hat but never apply it; isolates estimator feedback")
    p.add_argument("--calib-episodes", default=None,
                   help="comma-separated indices of healthy log episodes to fit the plant on (default all)")
    p.add_argument("--mimo", action="store_true",
                   help="coupled plant: fits better, extrapolates worse (see fit_plant)")
    p.add_argument("--dead", type=float, default=0.05, help="residual deadzone")
    p.add_argument("--norm-r", type=float, default=0.5, help="update normalisation")
    p.add_argument("--clip", type=float, default=0.15, help="projection box on f_hat")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--task-stride", type=int, default=1,
                   help="step between task ids; >1 samples a large suite evenly")
    p.add_argument("--sev", type=float, default=0.05)
    # Opt-in (OPT_IN_DEFAULTS): every default below leaves the runner exactly as it was.
    p.add_argument("--track-kappa", type=float, default=0.0,
                   help="pose-referenced tracking gain, 0 = off: after the base update, f_hat[d] <- "
                        "clip(f_hat[d] + kappa (M_inv e_p / n)[d]) on --track-dims (see PoseTracker)")
    p.add_argument("--track-ref", choices=["dc", "fitted"], default="dc",
                   help="plant that turns the nominal command into the reference: dc = the --dc-constrain fit "
                        "pinned to the probed M diagonal on the tracked dims; fitted = the unconstrained FIR")
    p.add_argument("--track-anchor", type=int, default=None,
                   help="re-anchor the reference to the measured pose every this many steps "
                        "(default: --replan-steps; 0 = once at episode start)")
    p.add_argument("--track-dims", default=None,
                   help="comma-separated dims the tracking term updates (default: --corr-dims, else all six)")
    p.add_argument("--replay-log", type=pathlib.Path, default=None,
                   help="no policy server: replay the commands of an error_signal.py log (raw_cmd, "
                        "episode_keys); each episode resets to its recorded scenario (implies "
                        "--scenario-reset) and ends at its recorded length; --control/--ack not needed")
    p.add_argument("--replay-episodes", default=None,
                   help="comma-separated episode indices of --replay-log (default: all); replaces --episodes")
    p.add_argument("--track-mode", choices=["rate", "position"], default="rate",
                   help="tracking observation: rate = z_T = M_inv e_p / n, the mean pose rate since the anchor; "
                        "position = z_T = M_inv e_p, the retained pose offset itself (see PoseTracker)")
    p.add_argument("--track-leak", type=float, default=0.0,
                   help="position mode: after each step the reference leaks toward the measured pose by this "
                        "fraction, bounding drift from a constant model error (0 = no leak)")
    p.add_argument("--track-obs", choices=["full", "tracked"], default="full",
                   help="tracking observation map: full = M_inv over all six pose channels; tracked = "
                        "solve(M[R,R], .) on the tracked dims R only, so untracked channels' pose error "
                        "(e.g. uncorrected translation) does not force them (see PoseTracker)")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if a.baseline in MATCHED_BASELINES + ("weighted_dob",) and a.law != "legacy":
        p.error("matched baselines cannot be combined with --law innov")
    if a.baseline == "dob" and (not np.isfinite(a.gamma) or not 0 < a.gamma <= 1):
        p.error("dob requires 0 < --gamma <= 1")
    if a.baseline == "integral_calibrated" and (not np.isfinite(a.ki) or a.ki <= 0):
        p.error("integral_calibrated requires --ki > 0")
    if a.baseline == "rls":
        if not np.isfinite(a.rls_lambda) or not 0 < a.rls_lambda <= 1:
            p.error("--rls-lambda must be in (0,1]")
        if not np.isfinite(a.rls_p0) or a.rls_p0 <= 0:
            p.error("--rls-p0 must be positive")
    if a.baseline == "kalman":
        if not np.isfinite(a.kf_r) or a.kf_r <= 0:
            p.error("--kf-r must be positive")
        if a.kf_q is not None and (not np.isfinite(a.kf_q) or a.kf_q < 0):
            p.error("--kf-q must be nonnegative")
        if a.kf_q is None and (not np.isfinite(a.gamma) or not 0 < a.gamma < 1):
            p.error("automatic --kf-q requires 0 < --gamma < 1")
    required_paths = ("control", "ack", "log", "openloop", "out")
    if a.replay_log is not None:
        required_paths = ("log", "openloop", "out")   # replay: no policy server, no control/ack handshake
    missing = ["--" + name for name in required_paths if getattr(a, name) is None]
    if missing:
        p.error("the following arguments are required: " + ", ".join(missing))
    if a.telemetry is not None:
        for name in required_paths:
            if a.telemetry.resolve() == getattr(a, name).resolve():
                p.error(f"--telemetry must differ from --{name}")
    track_dims, replay = _opt_in_arguments(p, a)

    import main as lm
    import paired_probe as pp

    calib = [int(x) for x in a.calib_episodes.split(",")] if a.calib_episodes else None
    if a.fir_k is not None:
        globals()["K_FIR"] = int(a.fir_k)
        print(f"FIR order K = {K_FIR} (--fir-k)")
    if a.ar:
        if a.mimo:
            raise SystemExit("--ar is implemented for the per-axis plant only")
        globals()["AR_ORDER"] = int(a.ar)
    W = fit_plant(a.log, mimo=a.mimo, episodes=calib)
    if calib is not None:
        print(f"plant fitted on healthy episodes {calib} only")
    M = np.array(json.loads(a.openloop.read_text())["M"])
    if a.m_diag:
        md = {int(k): float(v) for k, v in (item.split("=") for item in a.m_diag.split(","))}
        before = np.diag(M).copy()
        for i, v in md.items():
            M[i, i] = v
        print(f"M diagonal overridden {md}: {np.round(before, 3)} -> {np.round(np.diag(M), 3)}")
    if a.ar:
        # Under a constant command offset f the measured increment settles at M f, and the ARX
        # residual r = y - (FIR taps . u + AR . y_prev + c) settles at (1 - sum AR) M f: the past
        # measured increments absorb the offset's accumulated part. Rescale M consistently.
        ar_sum = W[:, K_FIR + 1:K_FIR + 1 + AR_ORDER].sum(axis=1)
        M = np.diag(1.0 - ar_sum) @ M
        print(f"ARX plant, AR order {AR_ORDER}: AR coefficients {np.round(ar_sum, 3)}; M rescaled by (1 - AR) per channel")
    if a.dc_gain:
        if a.dc_constrain != "none" or a.ar or a.mimo:
            raise SystemExit("--dc-gain is for the per-axis FIR plant without AR terms and excludes --dc-constrain")
        dcg = {int(k): float(v) for k, v in (item.split("=") for item in a.dc_gain.split(","))}
        before = W[:, :K_FIR + 1].sum(axis=1)
        W = fit_plant(a.log, mimo=a.mimo, episodes=calib, dc=dcg)
        print(f"DC-gain-constrained plant {dcg}: tap sums {np.round(before, 3)} -> {np.round(W[:, :K_FIR + 1].sum(axis=1), 3)}")
    if a.dc_constrain != "none":
        if a.ar or a.mimo:
            raise SystemExit("--dc-constrain is implemented for the per-axis FIR plant without AR terms")
        sel = list(range(6)) if a.dc_constrain == "all" else [int(x) for x in a.corr_dims.split(",")]
        before = W[:, :K_FIR + 1].sum(axis=1)
        W = fit_plant(a.log, mimo=a.mimo, episodes=calib, dc={i: float(M[i, i]) for i in sel})
        after = W[:, :K_FIR + 1].sum(axis=1)
        print(f"DC-constrained plant on channels {sel}: tap sums {np.round(before, 3)} -> {np.round(after, 3)}; "
              f"probed M diagonal {np.round(np.diag(M), 3)}")
    M_inv = np.linalg.pinv(M)
    print(f"plant identified; cond(M) = {np.linalg.cond(M):.1f}, gamma = {a.gamma}\n")
    kf_q, kf_r, estimator_config = None, None, None
    if a.baseline == "weighted_dob":
        kf_r, calibration = healthy_residual_covariance(json.loads(a.log.read_text()), W, episodes=calib)
        estimator_config = dict(healthy_residual=calibration, R=kf_r, allocation_prior_std=0.1,
            allocation="box-constrained weighted ridge on selected input columns")
    if a.baseline == "kalman":
        healthy_cov, calibration = healthy_residual_covariance(
            json.loads(a.log.read_text()), W, episodes=calib)
        kf_q, kf_r = kalman_noise(M_inv, healthy_cov, gamma=a.gamma,
                                 q=a.kf_q, r_scale=a.kf_r)
        steady = kalman_steady_state(M, kf_q, kf_r)
        estimator_config = dict(healthy_residual=calibration, Q=kf_q, R=kf_r,
                               initial_covariance=np.eye(6), steady_state=steady)
        print(f"Kalman: R from {calibration['samples']} centered healthy FIR residuals; "
              f"covariance eigenvalue floor={calibration['eigenvalue_floor']:.3g}")
        print(f"Kalman steady-state K (residual to fault):\n{steady['gain']}")
        print(f"Kalman steady-state K*M (compare with gamma={a.gamma}):\n"
              f"{steady['effective_gain']}")
    if a.baseline == "integral_calibrated":
        print("Calibrated integral uses f_hat += ki*(M_inv*r-bias): constant nonzero "
              "faults accumulate to the clip; this residual is independent of correction.")

    bias = np.array([float(x) for x in a.bias.split(",")]) if a.bias else None
    if bias is not None:
        print(f"estimator zeroed against healthy-run bias {np.round(bias,3)}")
    cdims = [int(x) for x in a.corr_dims.split(",")] if a.corr_dims else None
    gate = None
    if a.gate_stats is not None:
        if cdims is not None:
            raise SystemExit("--gate-stats replaces --corr-dims; give one or the other")
        gs = json.loads(a.gate_stats.read_text())
        gate = dict(b=np.array(gs["b"], float), sd=np.array(gs["sd"], float), k=float(a.gate_k))
        print(f"channel gate from healthy phantom {a.gate_stats}: k={a.gate_k}, b={np.round(gate['b'],4)}, sd={np.round(gate['sd'],4)}")
    static_c = np.array([float(x) for x in a.static_corr.split(",")]) if a.static_corr else None
    if static_c is not None:
        print(f"STATIC baseline: fixed correction {static_c} (no online estimation)")
    obs_off = np.array([float(x) for x in a.obs_offset.split(",")]) if a.obs_offset else None
    if obs_off is not None:
        print(f"SENSOR fault: position observation biased by {obs_off} m "
              f"(invisible to a difference-based residual)")
    fvec = np.array([float(x) for x in a.fault_vec.split(",")]) if a.fault_vec else None
    if fvec is not None:
        print(f"structured fault: {fvec}")
    if cdims is not None:
        print(f"correction applied only on dims {cdims}")
    track = tracking_record = None
    if track_dims is not None:
        # The reference plant, chosen independently of the estimator's own (--dc-constrain): the same
        # constrained fitting path pinned on the tracked dims, or the unconstrained (shipped) fit.
        W_ref = fit_plant(a.log, mimo=a.mimo, episodes=calib,
                          dc={i: float(M[i, i]) for i in track_dims} if a.track_ref == "dc" else None)
        track = dict(W_ref=W_ref, kappa=a.track_kappa, anchor=a.track_anchor, dims=track_dims)
        tracking_record = dict(kappa=a.track_kappa, reference=a.track_ref, anchor=a.track_anchor,
                               dims=track_dims, dc_pinned_dims=track_dims if a.track_ref == "dc" else [],
                               W_ref=W_ref, reference_command="nominal policy command (before correction and fault)",
                               z_map="M_inv (the probed map the estimator uses)")
        taps = (f"tap sums {np.round(W_ref[:, :K_FIR + 1].sum(axis=1), 3)}"
                if W_ref.shape[1] == K_FIR + 2 else "MIMO FIR")
        print(f"POSE TRACKING: kappa {a.track_kappa} on dims {track_dims}; reference '{a.track_ref}' ({taps}); "
              f"anchored {'every %d steps' % a.track_anchor if a.track_anchor else 'once at episode start'}")
        if a.track_mode == "position":
            # Rate mode (the default) adds nothing here, so its outputs stay as they were.
            track.update(mode="position", leak=a.track_leak)
            tracking_record.update(mode="position", leak=a.track_leak,
                                   observation="z_T = M_inv e_p (no division by n); after each step the "
                                               "reference leaks toward the measured pose by `leak`")
            print(f"  position mode: z_T = M_inv e_p (the retained pose offset), reference leak {a.track_leak} per step")
        if a.track_obs == "tracked":
            # The full observation (the default) adds nothing here, so its outputs stay as they were.
            track.update(obs="tracked")
            tracking_record.update(obs="tracked", observation_map="z_T[R] = solve(M[R,R], v[R]), z_T[not R] = 0, "
                                                                  "M = inv(M_inv), R = the tracked dims",
                                   z_map="solve(M[R,R], .) on the tracked dims, M = inv(M_inv); 0 elsewhere")
            print(f"  tracked-subspace observation: z_T[R] = solve(M[R,R], v[R]) on R = {track_dims}, 0 elsewhere")
    pr = pp.Probe(a) if replay is None else replay_probe(pp, a, replay)
    parsed_args = json.loads(json.dumps(vars(a), default=_json_default))
    if all(parsed_args[key] == value for key, value in OPT_IN_DEFAULTS.items()):
        # Nothing opt-in was used: record exactly the argument set of the runner before these
        # flags, so a default run's result JSON and telemetry args stay byte-identical to it.
        for key in OPT_IN_DEFAULTS:
            del parsed_args[key]
    if all(parsed_args[key] == value for key, value in TRACK_MODE_DEFAULTS.items()):
        # Likewise for the tracking mode: a rate-mode run records what it recorded before --track-mode.
        for key in TRACK_MODE_DEFAULTS:
            del parsed_args[key]
    if all(parsed_args[key] == value for key, value in TRACK_OBS_DEFAULTS.items()):
        # And for the observation map: a full-observation run records what it did before --track-obs.
        for key in TRACK_OBS_DEFAULTS:
            del parsed_args[key]
    res = {"gamma": a.gamma, "joint_fault": a.joint_fault, "args": parsed_args, "arms": {}}
    if estimator_config is not None:
        res["estimator_config"] = json.loads(json.dumps(estimator_config, default=_json_default))
    if tracking_record is not None:
        res["tracking"] = json.loads(json.dumps(tracking_record, default=_json_default))
    if replay is not None:
        res["replay"] = json.loads(json.dumps(replay.meta, default=_json_default))
    if a.joint_fault:
        print(f"JOINT-LEVEL fault (below the controller): {a.joint_fault}")
    # Spread episodes across the WHOLE suite. The old form, i % 10, silently confined a
    # libero_90 run to its first ten tasks. With --task-stride 4 on 90 tasks, 20 episodes
    # land on tasks 0, 4, 8, ... 76 -- a sample of the suite rather than a corner of it.
    n_tasks = pr.suite.n_tasks
    eps = [(((i * a.task_stride) % n_tasks), a.eval_init + (i * a.task_stride) // n_tasks)
           for i in range(a.episodes)]
    ep_seeds = [a.sampler_seed] * len(eps)
    if a.manifest is not None:
        man = json.loads(a.manifest.read_text())
        eps = [(int(m["task"]), int(m["init"])) for m in man["scenarios"]]
        ep_seeds = [m.get("sampler_seed", a.sampler_seed) for m in man["scenarios"]]
        print(f"scenario manifest {a.manifest}: {len(eps)} scenarios, sampler seeds "
              f"{sorted(set(x for x in ep_seeds if x is not None)) or 'none'}")
    if replay is not None:
        eps, ep_seeds = list(replay.scenarios), [None] * len(replay.scenarios)
        print(f"REPLAY (no policy server): {a.replay_log} episodes {replay.episodes} -> (task, init) {eps}, "
              f"recorded lengths {replay.lengths}; scenario reset implied; "
              f"--eval-init/--task-stride are ignored (scenarios come from the recording)")
    if a.pin_rng and (a.sampler_seed is not None or any(x is not None for x in ep_seeds)):
        raise SystemExit("--pin-rng and --sampler-seed are different schedules; give one")
    # LIBERO stores 50 initial states per task (0-49). A run that needs more states per task
    # than 50 - eval_init (re4 theory Part 8.2: n = 80 on ten tasks from init 45) wraps DOWNWARD
    # from eval_init - 1 rather than crashing at index 50 or silently reusing a state:
    # 45, 46, ..., 49, 44, 43, ... The first n = 80 attempt crashed at episode 51 on this.
    N_INIT = 50
    if any(init >= N_INIT for _, init in eps):
        eps = [(t, init if init < N_INIT else a.eval_init - 1 - (init - N_INIT)) for t, init in eps]
        assert all(0 <= init < N_INIT for _, init in eps), "init wrap ran out of stored states"
        print(f"initial states wrap below {a.eval_init}: using {sorted(set(i for _, i in eps))}")
    # Held-out check. error_signal.py --init-base and this script's --eval-init BOTH default
    # to 45, so unless one of them is moved the plant is identified on initial states the
    # evaluation then scores on. The published spatial cell evaluates inits {45,46}: half its
    # episodes sit on the state the plant was fitted on. This is stated as a limitation in the
    # paper; the check exists so it cannot recur silently.
    if CALIB_EPISODES:
        overlap = sorted(set(map(tuple, CALIB_EPISODES)) & set(eps))
        if overlap:
            print(f"!! CALIBRATION OVERLAP: {len(overlap)} of {len(eps)} evaluation episodes "
                  f"reuse (task, init) pairs the plant was fitted on: {overlap[:6]}"
                  f"{' ...' if len(overlap) > 6 else ''}")
            print("   Evaluation is NOT held out. Move --eval-init clear of the calibration "
                  "range, or refit with a different --init-base.")
        else:
            print(f"held-out check: {len(eps)} evaluation episodes, none shared with the "
                  f"{len(CALIB_EPISODES)} calibration episodes")
    else:
        print("held-out check: this plant artifact records no calibration provenance "
              "(predates the change); overlap cannot be verified")
    header = None
    if a.telemetry is not None:
        header = dict(runner="adaptive_law.py", schema_version=1, argv=list(sys.argv),
            args=parsed_args, numpy_version=np.__version__,
            constants=dict(OUT=OUT, K_FIR=K_FIR, WARMUP_STEPS=WARMUP_STEPS,
                IMAGE_SIZE=IMAGE_SIZE, fit_ridge=1e-2, arm_dims=6, env_seed=7,
                profile_period_floor=1e-9, intermittent_period_floor=1,
                probe=_module_constants(pp), libero=_module_constants(lm)),
            config=dict(W=W, M=M, M_inv=M_inv, mask=correction_mask(cdims),
                bias=bias, static_correction=static_c, fault_vector=fvec,
                observation_offset=obs_off, calibration_episodes=calib,
                max_steps=pp.MAXS if replay is None else replay.lengths, episodes=eps, n_tasks=n_tasks,
                control_request=dict(site=None, pin_rng=bool(a.pin_rng))),
            fields=dict(t="environment step, including the warmup",
                raw_action="full policy action before correction and action fault",
                correction="full action vector added by this runner before the action fault",
                nominal_command="full corrected action before the action fault",
                command="full action passed to env.step, including the action fault",
                measured="translation/relative rotation increment divided elementwise by OUT",
                r="measured minus FIR prediction from the corrected, pre-fault command",
                f_hat_before="estimate that generated this step's correction",
                f_hat="estimate after this step's update",
                f_true="additive Cartesian action fault; joint faults are specified in args",
                attenuation="reciprocal of the applied divisor; null when no division occurs",
                nr="norm_vector on the configured channels; null when no law norm is computed",
                warmup="no policy action, FIR residual, estimator update, or action fault"),
            # This once-only snapshot also preserves every literal in profiles, fitting,
            # correction and diagnostics alongside the named constants and fitted arrays.
            runner_source=pathlib.Path(__file__).read_text())
        if estimator_config is not None:
            header["config"]["estimator"] = estimator_config
        if tracking_record is not None:
            header["config"]["tracking"] = tracking_record
            header["fields"].update(
                track_e="pose minus reference since the anchor, normalised units (translation from "
                        "robot0_eef_pos, rotation from accumulated rot_delta increments)",
                track_z="M_inv @ (track_e / max(track_n, 1)); kappa times its tracked entries is added "
                        "to f_hat after the base update",
                track_n="steps since the anchor, this step included")
            if a.track_mode == "position":
                header["fields"].update(
                    track_e="pose minus the leaky reference since the anchor, before this step's leak "
                            "(normalised units, same pose conventions)",
                    track_z="M_inv @ track_e (position mode); kappa times its tracked entries is added to "
                            "f_hat after the base update")
            if a.track_obs == "tracked":
                header["fields"].update(
                    track_z="tracked-subspace observation: solve(M[R,R], v[R]) on the tracked dims R and 0 "
                            "elsewhere, v = track_e (position mode) or track_e / max(track_n, 1) (rate mode), "
                            "M = inv(M_inv); kappa times its tracked entries is added to f_hat after the base update")
        if replay is not None:
            header["config"]["replay"] = replay.meta
    timing_fh = None
    if a.timing is not None:
        a.timing.parent.mkdir(parents=True, exist_ok=True)
        timing_fh = open(a.timing, "w", buffering=1)
    with telemetry_stream(a.telemetry, header) as telemetry:
        pr.control(dict(site=None, pin_rng=bool(a.pin_rng)))
        arm_list = [("frozen_faulted", False), ("adaptive", True)]
        arm_list = [x for x in arm_list if a.arms == "both" or x[0].startswith(a.arms)]
        for tag, adapt in arm_list:
            ok, fh, trajs, per_ep = 0, [], [], []
            for episode, (tid, init) in enumerate(eps):
                if ep_seeds[episode] is not None:
                    # per-episode sampler schedule; the handshake's probe calls consume the first few keys
                    # identically in every arm, so the arms stay paired on (task, init, seed, call index)
                    pr.control(dict(site=None, pin_rng=False, sampler_seed=int(ep_seeds[episode]), episode=int(episode)))
                opt_in = {}
                if replay is not None:
                    replay.start(episode)
                    opt_in["max_steps"] = replay.lengths[episode]
                if track is not None:
                    opt_in["track"] = track
                s, f_hat, traj = run(pr, tid, init, a.sev, M_inv, W, a.gamma, adapt,
                                     dead=a.dead, norm_r=a.norm_r, clip=a.clip,
                                     apply_corr=not a.estimate_only, bias=bias,
                                     corr_dims=cdims, fvec=fvec, onset=a.onset,
                                     obs_off=obs_off, wrist_shift=a.wrist_shift,
                                     static_c=static_c, law=a.law, M=M,
                                     profile=a.profile, prof_p=a.prof_p,
                                     baseline=a.baseline, ki=a.ki, joint_fault=a.joint_fault,
                                     norm_channels=a.norm_channels, deadzone_mode=a.deadzone_mode,
                                     telemetry=telemetry, episode=episode, arm=tag,
                                     rls_lambda=a.rls_lambda, rls_p0=a.rls_p0,
                                     kf_q=kf_q, kf_r=kf_r, gate=gate,
                                     scenario_reset=a.scenario_reset, timing=timing_fh,
                                     adapt_from=a.adapt_from, **opt_in)
                ok += int(s); fh.append(f_hat.tolist())
                # Per-episode outcome, keyed by (task, init). The arms run on the SAME episode
                # list, so these pair up -- which is what McNemar needs and what the earlier
                # runs threw away by only accumulating a total. See mcnemar.py.
                per_ep.append(dict(task=int(tid), init=int(init), ok=bool(s),
                                   **({"sampler_seed": int(ep_seeds[episode])} if ep_seeds[episode] is not None else {})))
                if replay is not None:
                    per_ep[-1].update(replay_episode=replay.episodes[episode],
                                      recorded_steps=replay.lengths[episode], steps=len(traj))
                if traj and "gain" in traj[-1]:
                    per_ep[-1].update(last_gain=traj[-1]["gain"],
                                     last_effective_gain=traj[-1]["effective_gain"],
                                     estimator_updates=len(traj))
                trajs.append(traj)
                print(f"  [{tag}] task {tid} init {init}: success={s}  "
                      f"f_hat={np.round(f_hat, 3)}")
            res["arms"][tag] = dict(successes=ok, n=len(eps), f_hat=fh, per_ep=per_ep,
                                    traj=[[st["f_hat"] for st in tr] for tr in trajs],
                                    f_true=[[st["f_true"] for st in tr] for tr in trajs])
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
            clip_report(fh, a.clip, tag)
            print(f"{tag}: {ok}/{len(eps)} = {100*ok/len(eps):.0f}%\n")
        print(f"true fault = {fvec if fvec is not None else [a.sev]*6}")


if __name__ == "__main__":
    main()
