"""The same law on a different manipulator: bimanual ALOHA (ViperX x2) in gym_aloha, driven
by pi0_aloha_sim over openpi's websocket server.

Everything that made LIBERO work is re-derived here in JOINT space, because that is what
this robot's action interface is:
  action   14 absolute targets, 6 arm angles (rad) + 1 normalized gripper per side, at 50 Hz
  state    the same 14 coordinates, measured; gripper coordinates are not radians
  plant    per-joint FIR on POSITION: q_t = sum_k h_k u_{t-k} + c, identified on HEALTHY
           rollouts (mode log). Not on the increment dq as in LIBERO: there the command IS an
           increment, here it is an absolute target that a position servo tracks, so an
           offset in the target lives in the steady-state POSITION error and leaves almost
           nothing in dq -- a dq plant gave M ~ 0.008 and a law that amplified noise 125x.
  fault    additive offset on the commanded targets (an encoder / calibration offset), or a
           gain (loss of effectiveness), on chosen joints
  M        d(motion)/d(fault) per joint by open-loop replay (mode openloop)
  law      legacy (default): normalise M^-1 r, then EMA toward it
           innov: normalise the step M^-1 (r - M f_hat), then add gamma * step
           Both use the same deadzone / normaliser / clip arithmetic as adaptive_law.py.
No SO(3) anywhere: joint space is a vector space, which removes the whole class of bug that
Sec 2.2b of the LIBERO record was about.
"""
from __future__ import annotations
import argparse, atexit, collections, contextlib, io, json, pathlib, sys
import numpy as np

NJ, HORIZON, K_FIR, DT = 14, 10, 6, 0.02
TASK = "gym_aloha/AlohaTransferCube-v0"
PROMPT = "Transfer cube"
MAX_STEPS, IMAGE_SIZE, OPENLOOP_STEPS = 300, 224, 120


def estimator_step(f_hat, res, M_inv, *, gamma, dead, norm_r, clip, mask,
                   bias=None, norm_channels="all", deadzone_mode="zero", law="legacy", M=None):
    """Pure update; the mask selects the norm, never the estimated channels.

    Keep division of est and the original update expression: multiplying by a reciprocal
    or rewriting the decay as (1 - gamma) * f_hat can change floating-point results.
    A fired hold gate skips even projection, preserving every bit of the input estimate.
    Innov gates/normalises the innovation after bias, and its zero gate zeros the step
    (holding an in-box estimate). Pass the original measured M; the solve fallback for
    callers with only M_inv matches adaptive_law.py and requires an invertible M_inv.
    """
    if norm_channels not in ("all", "corrected"):
        raise ValueError("norm_channels must be all or corrected")
    if deadzone_mode not in ("zero", "hold"):
        raise ValueError("deadzone_mode must be zero or hold")
    if law not in ("legacy", "innov"):
        raise ValueError("law must be legacy or innov")
    if law == "innov":
        e = res - (M @ f_hat if M is not None else np.linalg.solve(M_inv, f_hat))
        if bias is not None:
            e = e - (M @ bias if M is not None else np.linalg.solve(M_inv, bias))
        nr = float(np.linalg.norm(e if norm_channels == "all" else e[np.asarray(mask, bool)]))
        fired = nr < dead
        if fired and deadzone_mode == "hold":
            return f_hat.copy(), dict(nr=nr, attenuation=None, deadzone_fired=True,
                                      update_applied=False)
        attenuation = None
        if fired:
            step = np.zeros(NJ)
        else:
            divisor = 1.0 + (nr / norm_r) ** 2
            step = (M_inv @ e) / divisor
            attenuation = 1.0 / divisor
        return np.clip(f_hat + gamma * step, -clip, clip), dict(
            nr=nr, attenuation=attenuation, deadzone_fired=fired, update_applied=True)
    est = M_inv @ res
    if bias is not None:
        est = est - bias
    norm_res = res if norm_channels == "all" else res[np.asarray(mask, bool)]
    nr = float(np.linalg.norm(norm_res))
    fired = nr < dead
    if fired and deadzone_mode == "hold":
        return f_hat.copy(), dict(nr=nr, attenuation=None, deadzone_fired=True,
                                  update_applied=False)
    if fired:
        est = np.zeros(NJ)
    divisor = 1.0 + (nr / norm_r) ** 2
    est = est / divisor
    f_next = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
    return f_next, dict(nr=nr, attenuation=1.0 / divisor, deadzone_fired=fired,
                        update_applied=True)


def applied_correction(f_hat, mask, *, adapt, static_corr=None):
    """Preserve the oracle's historical unmasked correction and mask adaptive control."""
    return ((-np.asarray(static_corr, float)) if static_corr is not None
            else ((-f_hat * mask) if adapt else np.zeros(NJ)))


def json_value(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def write_telemetry(stream, record):
    """Flush complete JSON lines so an interrupted rollout keeps prior steps readable."""
    stream.write(json.dumps(record, default=json_value) + "\n")
    stream.flush()


def telemetry_header(args, config):
    return dict(type="header", schema_version=1, runner="aloha_adapt", argv=list(sys.argv),
                args=vars(args), config=config,
                constants=dict(NJ=NJ, HORIZON=HORIZON, K_FIR=K_FIR, DT=DT, TASK=TASK,
                               PROMPT=PROMPT, MAX_STEPS=MAX_STEPS, IMAGE_SIZE=IMAGE_SIZE,
                               OPENLOOP_STEPS=OPENLOOP_STEPS),
                # Snapshot also captures every literal constant and the exact arithmetic.
                runner_source=pathlib.Path(__file__).read_text(),
                fields=dict(t="zero-based environment step within episode",
                            measured="post-step joint position; radians, grippers in env units",
                            motion="post-step minus pre-step joint position",
                            r="measured minus FIR prediction of nominal_command; null without plant",
                            correction="added to raw_action before the injected fault/gain",
                            command="actual target passed to env.step, after fault/gain",
                            f_hat="post-update estimate; f_hat_before generated this step's correction",
                            f_true="injected additive offset; gain is recorded separately in config",
                            nr="selected-channel norm of r for legacy; r - M f_hat_before for innov",
                            attenuation="reciprocal divisor actually applied; null if no division",
                            deadzone_fired="null when no estimator update was attempted"))


@contextlib.contextmanager
def telemetry_file(path, args, config):
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        write_telemetry(stream, telemetry_header(args, config))
        yield stream


def clip_report(vals, clip, name):
    v = np.abs(np.asarray(vals, float))
    if v.size == 0: return
    rail = (np.abs(v - clip) < 1e-6).mean(axis=0)
    if np.any(rail > 0.2):
        hit = ", ".join(f"j{i} {100*rail[i]:.0f}%" for i in range(len(rail)) if rail[i] > 0.2)
        print(f"  !! {name}: at the clip (+-{clip}) in >20% of episodes on [{hit}] -- SATURATED, not estimated.")


class Aloha:
    def __init__(self, host, port, seed=0):
        import gymnasium as gym
        import gym_aloha  # noqa: F401 (registers the envs)
        from openpi_client import websocket_client_policy as _wc

        self.env = gym.make(TASK, obs_type="pixels_agent_pos", render_mode="rgb_array")
        self.client = _wc.WebsocketClientPolicy(host, port)
        self.seed = seed
        atexit.register(self.env.close)

    def reset(self, ep):
        obs, _ = self.env.reset(seed=self.seed + ep)
        return obs

    def policy_obs(self, obs):
        from openpi_client import image_tools

        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs["pixels"]["top"], IMAGE_SIZE, IMAGE_SIZE))
        return {"state": np.asarray(obs["agent_pos"], np.float64),
                "images": {"cam_high": np.transpose(img, (2, 0, 1))}, "prompt": PROMPT}


def fit_plant(log_path):
    """Per-joint FIR on healthy (u, dq) pairs. Returns W (NJ, K_FIR+2): taps + intercept."""
    d = json.loads(pathlib.Path(log_path).read_text())
    W = np.zeros((NJ, K_FIR + 2)); r2 = np.zeros(NJ)
    for j in range(NJ):
        X, Y = [], []
        for ep in d:
            u = np.array(ep["u"])[:, j]; qq = np.array(ep["q"])[:, j]
            for t in range(K_FIR, len(u)):
                X.append(np.r_[u[t - K_FIR:t + 1][::-1], 1.0]); Y.append(qq[t])
        X, Y = np.array(X), np.array(Y)
        w, *_ = np.linalg.lstsq(X, Y, rcond=None); W[j] = w
        ss = ((Y - X @ w) ** 2).sum(); st = ((Y - Y.mean()) ** 2).sum()
        r2[j] = 1 - ss / max(st, 1e-12)
    return W, r2


def episode(A, ep, W=None, M_inv=None, fvec=None, gain=None, adapt=False, gamma=0.08, dead=0.002,
            norm_r=0.05, clip=0.3, corr=None, profile="step", prof_p=60.0, onset=0, log=None,
            static_corr=None, f_init=None, freeze_after=None, norm_channels="all",
            deadzone_mode="zero", telemetry=None, arm=None, law="legacy", M=None, timing=None,
            baseline="none", kf_q=None, kf_r=None, reference_model=None,
            tracking_rate=0.0, damping=0.0, ki=0.02, rls_lambda=0.95,
            initial_covariance=1.0, rls_p0=1.0):
    obs = A.reset(ep); q = np.asarray(obs["agent_pos"], float)
    # History initialised at the CURRENT joint position (holding), not zeros. Zeros are a
    # valid history for delta commands (LIBERO) but mean target = 0 rad here: for the first
    # K_FIR steps the prediction was wildly wrong (|r| = 1.4), the normaliser zeroed every
    # update, and f_hat decayed toward zero by gamma per step -- a 1.6 cm dip at the start
    # of every episode, cold or warm (Sec 27.7).
    hist = collections.deque([q.copy()] * (K_FIR + 1), maxlen=K_FIR + 1)
    f_hat = np.zeros(NJ) if f_init is None else np.asarray(f_init, float).copy()
    plan = collections.deque(); traj = []; success = False
    fvec = np.zeros(NJ) if fvec is None else np.asarray(fvec, float)
    m = np.zeros(NJ) if corr is None else np.isin(np.arange(NJ), corr).astype(float)
    observer_state = None
    q_ref = tracking_map = None
    if baseline == "composite":
        from composite_observer import reference_step, masked_tracking_map, composite_step
        if reference_model is None:
            raise ValueError("composite adaptation requires a qualified joint reference model")
        q_ref = q[np.asarray(reference_model["state_indices"], int)].copy()
        tracking_map = masked_tracking_map(reference_model, m)
    import time as _time
    _t_obs = _time.perf_counter()
    for t in range(MAX_STEPS):
        _tl0 = _time.perf_counter(); _wall0 = _time.time(); _replan = not plan
        if not plan:
            chunk = np.asarray(A.client.infer(A.policy_obs(obs))["actions"], float)
            plan.extend(chunk[:HORIZON])
        _policy_ms = (_time.perf_counter() - _tl0) * 1e3 if _replan else 0.0
        a_cmd = np.asarray(plan.popleft(), float)
        q_before = q.copy()
        if baseline == "composite":
            q_ref = reference_step(reference_model, q_ref, a_cmd)
        # oracle baseline: a FIXED correction equal to minus a known fault, no estimator.
        # If the task still fails with this, the failure is not the adaptive transient.
        f_hat_before = f_hat.copy() if telemetry is not None else None
        _tc0 = _time.perf_counter()
        c = applied_correction(f_hat, m, adapt=adapt, static_corr=static_corr)
        a_corr = a_cmd + c
        _tc1 = _time.perf_counter()
        live = t >= onset; u = t - onset
        scale = {"step": 1.0, "ramp": min(1.0, u / max(prof_p, 1e-9)),
                 "sine_bias": 0.5 * (1 + np.sin(2 * np.pi * u / max(prof_p, 1e-9))),
                 "intermittent": 1.0 if int(u // max(prof_p, 1)) % 2 == 0 else 0.0}[profile] if live else 0.0
        f_now = fvec * scale
        a_exec = a_corr + f_now
        if gain is not None and live:
            # loss of effectiveness on the commanded MOTION, not the absolute target: the
            # controller receives q + g (target - q), i.e. it only gets a fraction of the way
            a_exec = q + gain * (a_exec - q)
        _t_cmd = _time.perf_counter(); _t_obs_prev = _t_obs
        obs, r, term, trunc, info = A.env.step(a_exec)
        _t_obs = _time.perf_counter()
        q1 = np.asarray(obs["agent_pos"], float); dq = q1 - q; q = q1
        tracking_error = (q1[np.asarray(reference_model["state_indices"], int)] - q_ref
                          if baseline == "composite" else None)
        hist.appendleft(a_corr[:NJ].copy())
        if log is not None:
            log["u"].append(a_corr.tolist()); log["q"].append(q1.tolist())
            log.setdefault("q_before", []).append(q_before.tolist())
        res = None
        diag = dict(nr=None, attenuation=None, deadzone_fired=None, update_applied=False)
        if W is not None:
            H = np.array(hist)
            pred = np.array([W[j, :K_FIR + 1] @ H[:, j] + W[j, -1] for j in range(NJ)])
            res = q1 - pred                       # position residual
            if adapt and (freeze_after is None or t < freeze_after):
                if baseline == "composite":
                    f_hat, extra = composite_step(f_hat, res, M, state=observer_state,
                        Q=kf_q / DT, R=kf_r, dt=DT, tracking_error=tracking_error,
                        tracking_map=tracking_map, metric=reference_model["metric"],
                        tracking_rate=tracking_rate, damping=damping, clip=clip,
                        initial_covariance=initial_covariance)
                    diag.update(extra)
                    observer_state = extra["estimator_state"]
                elif baseline != "none":
                    from adaptive_law import estimator_step as matched_step
                    f_hat, extra = matched_step(f_hat, res, M_inv, gamma=gamma, dead=dead,
                        norm_r=norm_r, clip=clip, mask=m, M=M, baseline=baseline,
                        state=observer_state, kf_q=kf_q, kf_r=kf_r, ki=ki,
                        rls_lambda=rls_lambda, rls_p0=rls_p0)
                    diag.update(extra)
                    observer_state = extra.get("estimator_state")
                else:
                    f_hat, diag = estimator_step(f_hat, res, M_inv, gamma=gamma, dead=dead,
                                                 norm_r=norm_r, clip=clip, mask=m,
                                                 norm_channels=norm_channels, deadzone_mode=deadzone_mode,
                                                 law=law, M=M)
            traj.append(dict(t=t, f_hat=f_hat.tolist(), f_true=f_now.tolist()))
        _tad1 = _time.perf_counter()
        if timing is not None:
            timing.write(json.dumps(dict(type="step", arm=arm, episode=ep, task=0, init=int(ep), t=int(t),
                wall=_wall0, replan=bool(_replan), policy_ms=_policy_ms,
                adapter_ms=((_tc1 - _tc0) + (_tad1 - _t_obs)) * 1e3, env_ms=(_t_obs - _t_cmd) * 1e3,
                s2c_ms=(_t_cmd - _t_obs_prev) * 1e3, loop_ms=(_tad1 - _tl0) * 1e3, live=bool(live),
                f_true=np.asarray(f_now, float).round(6).tolist(), correction=np.asarray(c, float).round(6).tolist(),
                f_hat=np.asarray(f_hat, float).round(6).tolist())) + "\n")
        if telemetry is not None:
            write_telemetry(telemetry, dict(type="step", arm=arm, episode=ep, task=TASK,
                                           init=A.seed + ep, phase="rollout", t=t,
                                           raw_action=a_cmd, correction=c, nominal_command=a_corr,
                                           command=a_exec, measured=q1, motion=dq, r=res,
                                           joint_before=q_before, reference_position=q_ref,
                                           tracking_error=tracking_error,
                                           f_hat_before=f_hat_before, f_hat=f_hat, f_true=f_now,
                                           **diag))
        success = success or bool(info.get("is_success", False)) or (r >= 4)
        if term or trunc: break
    return success, f_hat, traj


def selftest_innov(mask):
    """Quantitative law checks on known 14-channel position maps, without a simulator."""
    selected = mask.astype(bool)
    # Coupling to an uncorrected channel must remain in the inverse and the innovation.
    # Binary fractions make the two selected norms and the gate boundary exact.
    M = 2.0 * np.eye(NJ); M[0, -1] = 0.5
    M_inv = np.linalg.inv(M)
    initial = np.full(NJ, 0.03125)
    bias = np.full(NJ, 0.0078125)
    e = np.zeros(NJ); e[0] = 0.00390625; e[-1] = 0.25
    residual = M @ initial + M @ bias + e
    opts = dict(gamma=0.08, dead=0.002, norm_r=0.05, clip=0.08, mask=mask,
                bias=bias, law="innov", M=M)
    snapshots = [x.tobytes() for x in (initial, residual, M, M_inv, mask, bias)]
    diagnostics = {}
    for channels in ("all", "corrected"):
        nr = float(np.linalg.norm(e if channels == "all" else e[selected]))
        divisor = 1.0 + (nr / opts["norm_r"]) ** 2
        expected_step = M_inv @ e / divisor
        for mode in ("zero", "hold"):
            actual, diag = estimator_step(initial, residual, M_inv, norm_channels=channels,
                                           deadzone_mode=mode, **opts)
            np.testing.assert_allclose(actual, initial + opts["gamma"] * expected_step,
                                       rtol=0, atol=1e-16)
            assert diag["nr"] == nr and diag["attenuation"] == 1.0 / divisor
            assert not diag["deadzone_fired"] and diag["update_applied"]
            assert actual[0] < initial[0]  # inverse coupling reverses this step's sign
            assert actual[-1] > initial[-1]  # estimation is not masked
            correction = applied_correction(actual, mask, adapt=True)
            assert np.array_equal(correction[selected], -actual[selected])
            assert np.array_equal(correction[~selected], np.zeros((~selected).sum()))
            # The LIBERO fallback solves M_inv * x = f_hat (and bias), without pinv.
            fallback, fallback_diag = estimator_step(initial, residual, M_inv,
                norm_channels=channels, deadzone_mode=mode, **(opts | {"M": None}))
            assert fallback.tobytes() == actual.tobytes() and fallback_diag == diag
            diagnostics[channels] = diag
    assert diagnostics["corrected"]["attenuation"] > 25 * diagnostics["all"]["attenuation"]
    assert snapshots == [x.tobytes() for x in (initial, residual, M, M_inv, mask, bias)]

    # Large raw residual but small corrected innovation: a residual norm, omitted bias,
    # or masked f_hat in M @ f_hat would each make the following hold assertion fail.
    small_e = e.copy(); small_e[0] = 0.0009765625
    outside = initial.copy(); outside[0] = -0.0; outside[-1] = 0.25
    residual = M @ outside + M @ bias + small_e
    for test_mask in (mask, np.zeros(NJ)):
        for mode in ("zero", "hold"):
            actual, diag = estimator_step(outside, residual, M_inv,
                norm_channels="corrected", deadzone_mode=mode, **(opts | {"mask": test_mask}))
            assert diag["nr"] == (small_e[0] if test_mask.any() else 0.0)
            assert diag["deadzone_fired"] and diag["attenuation"] is None
            if mode == "hold":
                assert actual.tobytes() == outside.tobytes() and not diag["update_applied"]
            else:
                expected = np.clip(outside + opts["gamma"] * np.zeros(NJ), -opts["clip"], opts["clip"])
                assert actual.tobytes() == expected.tobytes() and diag["update_applied"]
    _, diag = estimator_step(outside, residual, M_inv, **opts)
    assert not diag["deadzone_fired"]  # the all-channel innovation still exceeds the gate
    _, diag = estimator_step(outside, residual, M_inv, norm_channels="corrected",
                              **(opts | {"dead": small_e[0]}))
    assert not diag["deadzone_fired"]  # strict < at the innovation boundary too
    in_box = initial.copy()
    unchanged, diag = estimator_step(in_box, M @ in_box + M @ bias, M_inv, **opts)
    assert unchanged.tobytes() == in_box.tobytes() and diag["deadzone_fired"]

    # A supplied forward map also supports a singular measured M. Reconstructing M
    # from M_inv would discard small singular values retained in the measured map.
    for last_gain in (0.0, 1e-17):
        singular = np.eye(NJ); singular[-1, -1] = last_gain
        singular_inv = np.linalg.pinv(singular)
        assert singular_inv[-1, -1] == 0.0
        _, diag = estimator_step(initial, singular @ initial, singular_inv,
                                  **(opts | {"M": singular, "bias": None}))
        assert diag["nr"] == 0.0 and diag["deadzone_fired"]

    # Synthetic, well-conditioned coupled map scaled to exactly 5% legacy bias under
    # the preregistered all-channel norm. This is a mechanism fixture, not measured M.
    fault = 0.02 * mask
    M = np.eye(NJ) + 0.15 * np.roll(np.eye(NJ), 1, axis=0)
    M *= 0.4 * np.sqrt(1.0 / 0.95 - 1.0) / np.linalg.norm(M @ fault)
    M_inv = np.linalg.pinv(M)
    residual = M @ fault
    opts = dict(gamma=0.08, dead=0.002, norm_r=0.4, clip=0.08, mask=mask, M=M)
    fixed = {}
    for channels in ("all", "corrected"):
        nr = float(np.linalg.norm(residual if channels == "all" else residual[selected]))
        prediction = fault / (1.0 + (nr / opts["norm_r"]) ** 2)
        for mode in ("zero", "hold"):
            for dead in (0.0, opts["dead"]):
                estimates = {}
                for law in ("legacy", "innov"):
                    estimate = np.zeros(NJ)
                    previous_error = float(np.linalg.norm(residual))
                    for _ in range(1200):
                        estimate, diag = estimator_step(estimate, residual, M_inv, law=law,
                            norm_channels=channels, deadzone_mode=mode, **(opts | {"dead": dead}))
                        error = float(np.linalg.norm(residual - M @ estimate))
                        if law == "innov":
                            assert error <= previous_error + 1e-15
                        previous_error = error
                    estimates[law] = estimate
                    fixed[channels, mode, dead, law] = estimate
                legacy, innov = estimates["legacy"], estimates["innov"]
                np.testing.assert_allclose(legacy, prediction, rtol=0, atol=1e-14)
                assert np.all(legacy[selected] < fault[selected])
                assert 0.049 < float(np.mean(1 - legacy[selected] / fault[selected])) < 0.051
                assert np.linalg.norm(innov - fault) < 0.5 * np.linalg.norm(legacy - fault)
                if dead == 0.0:
                    np.testing.assert_allclose(innov, fault, rtol=0, atol=1e-14)
                    assert np.linalg.norm(residual - M @ innov) < 1e-13
                else:
                    e_final = residual - M @ innov
                    ne = float(np.linalg.norm(e_final if channels == "all" else e_final[selected]))
                    assert ne < dead and diag["deadzone_fired"]
                    assert 0.0195 <= float(innov[selected].mean()) <= 0.0200
                    # From zero, this linear fixture stays on the ray toward f. Its
                    # final step enters the deadband by at most the step factor gamma.
                    fraction_left = (fault[selected] - innov[selected]) / fault[selected]
                    assert np.all(fraction_left < dead / nr + 1e-13)
                    assert np.all(fraction_left >= (1 - opts["gamma"]) * dead / nr - 1e-13)

    # Removal benefit is specific to 'hold': legacy gates on r=0 and freezes; innov
    # gates on -M f_hat and decays to its deadband. Default legacy 'zero' DOES decay.
    removed = {}
    for channels in ("all", "corrected"):
        for mode in ("zero", "hold"):
            for law in ("legacy", "innov"):
                start = fixed[channels, mode, opts["dead"], law]
                estimate = start.copy()
                previous_norm = float(np.linalg.norm(estimate))
                for _ in range(600):
                    estimate, diag = estimator_step(estimate, np.zeros(NJ), M_inv, law=law,
                        norm_channels=channels, deadzone_mode=mode, **opts)
                    assert np.linalg.norm(estimate) <= previous_norm + 1e-15
                    previous_norm = float(np.linalg.norm(estimate))
                if law == "legacy" and mode == "hold":
                    assert estimate.tobytes() == start.tobytes() and not diag["update_applied"]
                elif law == "legacy":
                    assert np.linalg.norm(estimate) < 1e-20
                else:
                    assert np.linalg.norm(estimate) < 0.025 * np.linalg.norm(start)
                    e_final = -M @ estimate
                    assert np.linalg.norm(e_final if channels == "all" else e_final[selected]) < opts["dead"]
                removed[channels, mode, law] = float(estimate[selected].mean())
    legacy = float(fixed["all", "zero", 0.002, "legacy"][selected].mean())
    innov = float(fixed["all", "zero", 0.002, "innov"][selected].mean())
    ungated = float(fixed["all", "zero", 0.0, "innov"][selected].mean())
    print(f"Synthetic fixed points (truth=0.020000000): legacy={legacy:.9f} "
          f"(predicted=0.019000000), innov dead=0.002: {innov:.9f}, innov dead=0: {ungated:.9f}")
    print("Fault removal means: " + "; ".join(
        f"{mode}: legacy={removed['all', mode, 'legacy']:.9g}, innov={removed['all', mode, 'innov']:.9g}"
        for mode in ("zero", "hold")))


def selftest():
    """Only NumPy/stdlib: exact arithmetic, masks, gates and streamed synthetic steps."""
    rng = np.random.default_rng(7)
    mask = np.r_[np.ones(6), np.zeros(NJ - 6)]
    for dtype in (np.float32, np.float64):
        for scale in (0.0, 1e-5, 0.01, 10.0):
            for _ in range(12):
                f_hat = rng.normal(size=NJ).astype(dtype)
                res = (rng.normal(size=NJ) * scale).astype(dtype)
                M_inv = rng.normal(size=(NJ, NJ)).astype(dtype)
                before = tuple(x.tobytes() for x in (f_hat, res, M_inv, mask))
                # Original rollout arithmetic, intentionally independent of the helper.
                est = M_inv @ res
                nr = float(np.linalg.norm(res))
                if nr < 0.002: est = np.zeros(NJ)
                est = est / (1.0 + (nr / 0.05) ** 2)
                expected = np.clip(f_hat + 0.08 * (est - f_hat), -0.3, 0.3)
                actual, diag = estimator_step(f_hat, res, M_inv, gamma=0.08, dead=0.002,
                                              norm_r=0.05, clip=0.3, mask=mask)
                assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
                explicit, explicit_diag = estimator_step(f_hat, res, M_inv, gamma=0.08, dead=0.002,
                    norm_r=0.05, clip=0.3, mask=mask, law="legacy", M=np.eye(NJ))
                assert explicit.tobytes() == actual.tobytes() and explicit_diag == diag
                assert diag["nr"] == nr and diag["deadzone_fired"] == (nr < 0.002)
                assert diag["attenuation"] == 1.0 / (1.0 + (nr / 0.05) ** 2)
                assert before == tuple(x.tobytes() for x in (f_hat, res, M_inv, mask))

    options = dict(gamma=0.25, dead=0.002, norm_r=0.05, clip=0.3, mask=mask)
    residual = np.zeros(NJ); residual[0] = 0.001; residual[-1] = 10.0
    initial = np.linspace(-0.25, 0.25, NJ)
    _, all_diag = estimator_step(initial, residual, np.eye(NJ), **options)
    _, corrected_diag = estimator_step(initial, residual, np.eye(NJ),
                                       norm_channels="corrected", **options)
    assert corrected_diag["nr"] == 0.001 < all_diag["nr"]
    assert corrected_diag["deadzone_fired"] and not all_diag["deadzone_fired"]
    assert corrected_diag["attenuation"] > all_diag["attenuation"]
    empty_options = options | {"mask": np.zeros(NJ)}
    _, empty_diag = estimator_step(initial, residual, np.eye(NJ),
                                   norm_channels="corrected", **empty_options)
    assert empty_diag["nr"] == 0.0 and empty_diag["deadzone_fired"]

    held = initial.copy(); held[0] = -0.0; held[-1] = 1.0  # hold skips projection too
    decayed = np.full(NJ, 0.25)
    for _ in range(8):
        next_held, diag = estimator_step(held, np.zeros(NJ), np.eye(NJ),
                                         deadzone_mode="hold", **options)
        assert next_held.tobytes() == held.tobytes()
        assert diag["deadzone_fired"] and not diag["update_applied"]
        assert diag["attenuation"] is None
        next_decay, _ = estimator_step(decayed, np.zeros(NJ), np.eye(NJ), **options)
        assert next_decay.tobytes() == (decayed + 0.25 * (np.zeros(NJ) - decayed)).tobytes()
        assert np.array_equal(next_decay, decayed * 0.75)
        held, decayed = next_held, next_decay
    boundary = np.zeros(NJ); boundary[0] = 0.125
    _, diag = estimator_step(initial, boundary, np.eye(NJ), **(options | {"dead": 0.125}))
    assert not diag["deadzone_fired"]  # the original test is strictly less-than
    correction = applied_correction(initial, mask, adapt=True)
    assert np.array_equal(correction[:6], -initial[:6])
    assert np.array_equal(correction[6:], np.zeros(NJ - 6))
    assert np.array_equal(applied_correction(initial, mask, adapt=False), np.zeros(NJ))
    assert np.array_equal(applied_correction(initial, mask, adapt=True, static_corr=initial), -initial)

    class MemoryStream(io.StringIO):
        flushes = 0

        def flush(self):
            self.flushes += 1
            super().flush()

    class SyntheticAloha:
        seed = 3

        def __init__(self):
            self.env = self.client = self

        def reset(self, ep):
            self.t = 0
            return {"agent_pos": np.zeros(NJ)}

        def policy_obs(self, obs):
            return obs

        def infer(self, obs):
            return {"actions": np.full((HORIZON, NJ), 0.1)}

        def step(self, command):
            self.t += 1
            return {"agent_pos": command + 0.01}, 0, self.t == 3, False, {}

    M = 2.0 * np.eye(NJ); M[0, -1] = 0.5
    M_inv = np.linalg.inv(M)
    for law in ("legacy", "innov"):
        for channels in ("all", "corrected"):
            for mode in ("zero", "hold"):
                stream = MemoryStream()
                write_telemetry(stream, telemetry_header(argparse.Namespace(
                    telemetry=pathlib.Path("synthetic.jsonl"), law=law),
                    dict(M=M, M_inv=M_inv, correction_mask=mask)))
                ep_options = dict(W=np.zeros((NJ, K_FIR + 2)), M_inv=M_inv, M=M, law=law,
                                  norm_channels=channels, deadzone_mode=mode, adapt=True,
                                  corr=list(range(6)), f_init=initial, fvec=np.full(NJ, 0.02), freeze_after=2)
                logged = episode(SyntheticAloha(), 0, telemetry=stream, arm="adaptive", **ep_options)
                plain = episode(SyntheticAloha(), 0, **ep_options)
                assert logged[0] == plain[0] and np.array_equal(logged[1], plain[1]) and logged[2] == plain[2]
                records = [json.loads(line) for line in stream.getvalue().splitlines()]
                assert len(records) == stream.flushes == 4 and records[0]["type"] == "header"
                assert records[0]["args"] == dict(telemetry="synthetic.jsonl", law=law)
                for t, record in enumerate(records[1:]):
                    assert record["episode"] == 0 and record["t"] == t and record["arm"] == "adaptive"
                    before = np.array(record["f_hat_before"])
                    c = -before * mask
                    assert np.array_equal(record["correction"], c)
                    assert np.array_equal(record["command"], np.array(record["raw_action"]) + c + record["f_true"])
                    assert record["r"] == record["measured"]  # zero synthetic FIR
                    assert record["f_hat"] == logged[2][t]["f_hat"]
                    assert record["update_applied"] == (t < 2)
                    if t < 2:
                        # Replay the update from telemetry to verify episode forwards
                        # the law, forward map and S1 options to the tested pure helper.
                        expected, diag = estimator_step(before, np.array(record["r"]), M_inv,
                            gamma=0.08, dead=0.002, norm_r=0.05, clip=0.3, mask=mask,
                            law=law, M=M, norm_channels=channels, deadzone_mode=mode)
                        assert expected.tobytes() == np.array(record["f_hat"]).tobytes()
                        assert all(record[key] == value for key, value in diag.items())
                    else:
                        assert before.tobytes() == np.array(record["f_hat"]).tobytes()
                assert records[-1]["nr"] is None and records[-1]["attenuation"] is None
                # Identify-then-hold uses freeze_after=0 in subsequent episodes.
                _, held, trajectory = episode(SyntheticAloha(), 1,
                    **(ep_options | {"f_init": logged[1], "freeze_after": 0}))
                assert held.tobytes() == logged[1].tobytes()
                assert all(step["f_hat"] == held.tolist() for step in trajectory)
    selftest_innov(mask)
    print("ALOHA selftest passed: exact default/explicit legacy arithmetic (max abs difference=0.0), "
          "innovation fixed points, norm selection, hold/zero gates, removal, mask, JSONL rollout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["log", "openloop", "run"])
    ap.add_argument("--selftest", action="store_true", help="run synthetic checks without simulator imports")
    ap.add_argument("--telemetry", type=pathlib.Path, help="stream per-step JSONL telemetry (default off)")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--log", type=pathlib.Path); ap.add_argument("--openloop", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--fault-vec", default=None, help="14 comma-separated joint offsets (rad)")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--corr-joints", default=None, help="joints to correct, e.g. 0,1,2,3,4,5")
    ap.add_argument("--profile", default="step", choices=["step", "ramp", "sine_bias", "intermittent"])
    ap.add_argument("--prof-p", type=float, default=60.0); ap.add_argument("--onset", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--dead", type=float, default=0.002)
    ap.add_argument("--norm-r", type=float, default=0.05); ap.add_argument("--clip", type=float, default=0.3)
    ap.add_argument("--law", choices=["legacy", "innov"], default="legacy",
                    help="legacy: normalise the estimate (biased low); innov: normalise the innovation step")
    ap.add_argument("--baseline", choices=["none", "dob", "rls", "kalman", "integral_calibrated", "composite", "weighted_dob"],
                    default="none", help="matched calibrated observer; default retains the existing law")
    ap.add_argument("--reference-artifact", type=pathlib.Path,
                    help="qualified healthy reference and shared W/M/Q/R from prepare_composite_reference.py")
    ap.add_argument("--reference-source-log", type=pathlib.Path,
                    help="composite validation: portable copy of the artifact's hash-matched healthy source")
    ap.add_argument("--reference-source-sensitivity", type=pathlib.Path,
                    help="composite validation: portable copy of the artifact's hash-matched M source")
    ap.add_argument("--tracking-rate", type=float, default=0.0,
                    help="position-error feedback rate; zero is the matched Kalman ablation")
    ap.add_argument("--damping", type=float, default=0.0, help="composite parameter damping per second")
    ap.add_argument("--ki", type=float, default=0.02)
    ap.add_argument("--rls-lambda", type=float, default=0.95)
    ap.add_argument("--norm-channels", choices=["all", "corrected"], default="all",
                    help="channels used by the residual deadzone and normaliser (innovation for --law innov)")
    ap.add_argument("--deadzone-mode", choices=["zero", "hold"], default="zero",
                    help="zero: legacy zeros the target (decay), innov zeros the step; "
                         "hold: skip the entire gated update including projection")
    ap.add_argument("--probe", type=float, default=0.02, help="openloop: per-joint probe (rad)")
    ap.add_argument("--static-corr", default=None, help="oracle: 14 offsets subtracted from every command, no estimator")
    ap.add_argument("--identify-episodes", type=int, default=None,
                    help="adapt during the first N episodes, then HOLD the estimate for the rest. "
                         "The deployable scheme for a persistent fault on a tight-margin task: pay the "
                         "transient once, then apply a stationary correction (Sec 27.12).")
    ap.add_argument("--freeze-after", type=int, default=None,
                    help="stop updating f_hat after this step; the correction stays applied. Tests whether "
                         "mid-episode updating, not the estimate itself, is what breaks a tight-margin task")
    ap.add_argument("--warm-start", action="store_true",
                    help="carry f_hat from one episode into the next (a persistent fault has a persistent estimate)")
    ap.add_argument("--f-init", default=None,
                    help="comma-separated estimate; every adaptive episode starts from it, no carry across "
                         "episodes (re4 Part F: held vs continued from a matched initial estimate)")
    ap.add_argument("--skip-frozen", action="store_true",
                    help="run only the adaptive arm; its paired frozen arm is run elsewhere on the same seeds")
    ap.add_argument("--timing", type=pathlib.Path, default=None,
                    help="per-step timing JSONL (re4 Part E)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.mode is None or a.out is None:
        ap.error("mode and --out are required unless --selftest is used")
    if a.baseline != "none" and a.law != "legacy":
        ap.error("matched baselines cannot be combined with --law innov")
    if a.baseline in ("kalman", "composite", "weighted_dob") and a.reference_artifact is None:
        ap.error("Kalman/composite/weighted_dob requires --reference-artifact with shared calibration")
    if not np.isfinite(a.tracking_rate) or a.tracking_rate < 0 or not np.isfinite(a.damping) or a.damping < 0:
        ap.error("tracking rate and damping must be finite and nonnegative")
    if a.baseline != "composite" and (a.tracking_rate != 0 or a.damping != 0):
        ap.error("tracking rate and damping are only used by the composite variant")
    if a.telemetry is not None and any(a.telemetry.resolve() == p.resolve()
                                       for p in (a.out, a.log, a.openloop) if p is not None):
        ap.error("--telemetry must differ from --out, --log and --openloop")
    W = r2 = M = M_inv = None
    observer = reference_model = reference_validation = None
    if a.mode == "run":
        W, r2 = fit_plant(a.log)
        # As in adaptive_law.py, retain the measured forward map for the innovation;
        # reconstructing it with pinv(M_inv) would add rounding and lose truncated modes.
        M = np.array(json.loads(a.openloop.read_text())["M"])
        M_inv = np.linalg.pinv(M)
        if a.reference_artifact is not None:
            artifact = json.loads(a.reference_artifact.read_text())
            observer = {key: np.asarray(value, float) for key, value in artifact["observer"].items()
                        if key in ("W", "M", "Q", "R")}
            W, M = observer["W"], observer["M"]
            # The artifact uses only its declared training split. The R2 above
            # describes a different all-log fit and must not label this model.
            r2 = None
            M_inv = np.linalg.pinv(M)
            reference_model = artifact["model"]
            if a.baseline == "composite" and not artifact["qualification"]["allowed"]:
                ap.error("reference artifact failed qualification: " + str(artifact["qualification"]["reasons"]))
            if a.baseline == "composite":
                tracking = artifact.get("tracking") or {}
                selected = [int(x) for x in a.corr_joints.split(",")] if a.corr_joints else []
                mask = np.isin(np.arange(NJ), selected).astype(float)
                if not np.array_equal(mask, np.asarray(tracking.get("correction_mask"))):
                    ap.error("correction mask differs from the qualified tracking configuration")
                if not np.isclose(tracking.get("dt", -1), DT) or not np.isclose(tracking.get("damping", -1), a.damping):
                    ap.error("dt/damping differs from the qualified tracking configuration")
                matches = [candidate for candidate in tracking.get("candidates", [])
                           if candidate.get("tracking_rate") is not None and
                           np.isclose(candidate["tracking_rate"], a.tracking_rate, rtol=1e-10, atol=1e-12)]
                if not matches or not matches[0].get("qualified_for_experiment", False):
                    ap.error("tracking rate is not in the qualified calibration grid")
                from validate_composite_reference import validate_composite_artifact
                try:
                    reference_validation = validate_composite_artifact(artifact,
                        artifact_path=a.reference_artifact, source_log=a.reference_source_log,
                        source_sensitivity=a.reference_source_sensitivity)
                except ValueError as error:
                    ap.error(str(error))
    # log/openloop historically ignore most law flags; record their effective settings.
    config = dict(W=W, r2=r2, M=M, M_inv=M_inv, observer=observer,
                  reference_model=reference_model, baseline=a.baseline,
                  tracking_rate=a.tracking_rate, damping=a.damping,
                  reference_validation=reference_validation)
    if a.mode == "log":
        config.update(profile="step", onset=0, gain=None, adapt=False,
                      correction_mask=np.zeros(NJ), plant_available=False)
    elif a.mode == "openloop":
        config.update(probe=a.probe, replay_limit=OPENLOOP_STEPS, reset_episode=0,
                      measurement_window="second half of replay", plant_available=False)
    else:
        corr = [int(x) for x in a.corr_joints.split(",")] if a.corr_joints else []
        config.update(correction_mask=np.isin(np.arange(NJ), corr).astype(float),
                      bias=None, gain=a.gain, plant_available=True,
                      initial_estimate=np.zeros(NJ), static_correction_uses_mask=False)
    with telemetry_file(a.telemetry, a, config) as telemetry:
        run_cli(a, telemetry, W, r2, M_inv, M=M, observer=observer, reference_model=reference_model)


def run_cli(a, telemetry, W=None, r2=None, M_inv=None, M=None, observer=None, reference_model=None):
    A = Aloha(a.host, a.port, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.mode == "log":
        eps = []
        for ep in range(a.episodes):
            # log mode with a fault: records (u, q) under the fault so the anchoring question of
            # Sec 26.2 -- does the policy convert a target offset into a drift? -- is measurable
            fv = [float(x) for x in a.fault_vec.split(",")] if a.fault_vec else None
            L = dict(u=[], q=[])
            s, _, _ = episode(A, ep, fvec=fv, log=L, telemetry=telemetry, arm="log")
            L.update(success=s, task=TASK, init=A.seed + ep); eps.append(L)
            print(f"  healthy ep {ep}: success={s} steps={len(L['u'])}")
        a.out.write_text(json.dumps(eps))
        if not a.fault_vec:
            W, r2 = fit_plant(a.out); print("FIR R2 per joint:", np.round(r2, 3))
        print(f"successes {sum(e['success'] for e in eps)}/{len(eps)}")
        return

    if a.mode == "openloop":
        # replay a healthy command sequence open loop with and without a per-joint probe
        d = json.loads(a.log.read_text()); cmds = np.array(d[0]["u"])[:OPENLOOP_STEPS]
        replay_id = 0
        def replay(f):
            nonlocal replay_id
            obs = A.reset(0); q = np.asarray(obs["agent_pos"], float); D = []
            for t, u_t in enumerate(cmds):
                command = np.asarray(u_t) + f
                obs, *_ = A.env.step(command)
                q1 = np.asarray(obs["agent_pos"], float); D.append(q1)
                if telemetry is not None:
                    write_telemetry(telemetry, dict(type="step", arm="openloop", episode=replay_id,
                                                   task=TASK, init=A.seed, phase="replay", t=t,
                                                   raw_action=u_t, correction=np.zeros(NJ),
                                                   nominal_command=u_t, command=command,
                                                   measured=q1, motion=q1 - q, r=None, nr=None,
                                                   attenuation=None, deadzone_fired=None,
                                                   update_applied=False, f_hat_before=np.zeros(NJ),
                                                   f_hat=np.zeros(NJ), f_true=f))
                q = q1
            replay_id += 1
            return np.array(D)[len(cmds) // 2:]          # steady state only
        base = replay(np.zeros(NJ)); M = np.zeros((NJ, NJ))
        for j in range(NJ):
            acc = []
            for sgn in (1.0, -1.0):
                f = np.zeros(NJ); f[j] = sgn * a.probe
                acc.append((replay(f) - base).mean(0) / (sgn * a.probe))
            M[:, j] = np.mean(acc, axis=0)
        print("M diagonal:", np.round(np.diag(M), 3)); print("cond(M):", round(float(np.linalg.cond(M)), 1))
        a.out.write_text(json.dumps({"M": M.tolist(), "probe": a.probe})); return

    fvec = [float(x) for x in a.fault_vec.split(",")] if a.fault_vec else None
    corr = [int(x) for x in a.corr_joints.split(",")] if a.corr_joints else None
    sc = [float(x) for x in a.static_corr.split(",")] if a.static_corr else None
    res = dict(args=json.loads(json.dumps(vars(a), default=json_value)), arms={})
    timing_fh = open(a.timing, "w", buffering=1) if getattr(a, "timing", None) else None
    f_fixed = np.array([float(x) for x in a.f_init.split(",")]) if a.f_init else None
    assert f_fixed is None or len(f_fixed) == NJ, f"--f-init needs {NJ} values"
    arm_list = (("adaptive", True),) if a.skip_frozen else (("frozen_faulted", False), ("adaptive", True))
    for tag, adapt in arm_list:
        ok, fh, per_ep, trajs = 0, [], [], []
        f_carry = None
        for ep in range(a.episodes):
            s, f_hat, traj = episode(A, ep, W, M_inv, fvec, a.gain, adapt, a.gamma, a.dead, a.norm_r, a.clip,
                                     corr, a.profile, a.prof_p, a.onset,
                                     static_corr=(sc if adapt else None),
                                     f_init=(f_fixed if (adapt and f_fixed is not None) else (f_carry if (adapt and (a.warm_start or a.identify_episodes is not None)) else None)),
                                     freeze_after=(0 if (a.identify_episodes is not None
                                                         and ep >= a.identify_episodes) else a.freeze_after),
                                     norm_channels=a.norm_channels, deadzone_mode=a.deadzone_mode,
                                     telemetry=telemetry, arm=tag, law=a.law, M=M, timing=timing_fh,
                                     baseline=getattr(a, "baseline", "none"),
                                     kf_q=None if observer is None else observer["Q"],
                                     kf_r=None if observer is None else observer["R"],
                                     reference_model=reference_model,
                                     tracking_rate=getattr(a, "tracking_rate", 0.0),
                                     damping=getattr(a, "damping", 0.0),
                                     ki=getattr(a, "ki", 0.02), rls_lambda=getattr(a, "rls_lambda", 0.95))
            f_carry = f_hat
            ok += int(s); fh.append(f_hat.tolist()); per_ep.append(dict(task=0, init=ep, ok=bool(s))); trajs.append(traj)
            print(f"  [{tag}] ep {ep}: success={s}  f_hat[:6]={np.round(f_hat[:6], 3)}")
        res["arms"][tag] = dict(successes=ok, n=a.episodes, f_hat=fh, per_ep=per_ep,
                                traj=[[st["f_hat"] for st in tr] for tr in trajs],
                                f_true=[[st["f_true"] for st in tr] for tr in trajs])
        a.out.write_text(json.dumps(res)); clip_report(fh, a.clip, tag)
        print(f"{tag}: {ok}/{a.episodes} = {100*ok/a.episodes:.0f}%\n")


if __name__ == "__main__":
    main()
