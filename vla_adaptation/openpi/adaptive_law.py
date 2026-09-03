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

import argparse, collections, json, pathlib
import numpy as np

import main as lm
from so3 import rot_delta
from openpi_client import image_tools
from paired_probe import Probe
from gate_faults import apply_action_fault

OUT = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
K_FIR = 6


def fit_plant(log_path, lam=1e-2, mimo=False):
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
    A = np.array(d[0]["raw_a"]); D = np.array(d[0]["raw_d"]); lens = d[0]["ep_len"]
    W = []
    for i in range(6):
        X, Y, o = [], [], 0
        for L in lens:
            a, y = A[o:o+L], D[o:o+L, i] / OUT[i]; o += L
            for t in range(K_FIR, L):
                win = a[t-K_FIR:t+1][::-1]
                feat = win.reshape(-1) if mimo else win[:, i]
                X.append(np.concatenate([feat, [1.0]]))
                Y.append(y[t])
        X, Y = np.array(X), np.array(Y)
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


def run(pr, tid, init, sev, M_inv, W, gamma, adapt, max_steps=None, fvec=None, onset=0,
        obs_off=None, wrist_shift=0, static_c=None,
        dead=0.05, norm_r=0.5, clip=0.15, apply_corr=True, bias=None, corr_dims=None,
        law="legacy", M=None, profile="step", prof_p=60.0,
        baseline="none", ki=0.05):
    import paired_probe as _pp
    max_steps = max_steps or _pp.MAXS
    env, desc, inits = pr.env_for(tid)
    env.reset(); obs = env.set_init_state(inits[init])
    plan, t = collections.deque(), 0
    hist = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
    f_hat = np.zeros(6)
    traj = []
    while t < max_steps + 10:
        if t < 10:
            obs, _, done, _ = env.step(lm.LIBERO_DUMMY_ACTION); t += 1; continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        if wrist_shift:
            # A misaligned wrist camera. This fault damages the POLICY's input while leaving
            # the plant untouched, so the command-motion residual stays clean -- the quadrant
            # the sensor-bias test could not reach, because that one did no damage.
            wr = np.roll(wr, wrist_shift, axis=1)
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
        a_cmd = np.asarray(plan.popleft(), float)
        # apply_corr=False estimates but does NOT act. r = M f + eps(a+c): with an
        # imperfect plant the residual carries the model error evaluated at the operating
        # point, so once c moves that point the estimate chases its own correction. Freezing
        # c at zero separates open-loop model bias from that feedback.
        if static_c is not None:
            # Baseline: a FIXED correction, tuned offline for one known fault and applied
            # always. It needs the fault to be known in advance and to never change -- the
            # two assumptions the online law exists to drop.
            c = static_c.copy() if adapt else np.zeros(6)
        else:
            c = -f_hat if (adapt and apply_corr) else np.zeros(6)
        if corr_dims is not None:
            # Apply the correction on a SUBSET of axes. The separation test says the
            # estimator identifies the fault on rotation (0.047-0.049 against a true 0.050)
            # and not at all on translation (-0.013, -0.005). If the benefit really comes
            # from the channels that are identified, rotation-only should retain it.
            m = np.zeros(6); m[list(corr_dims)] = 1.0
            c = c * m
        a_corr = a_cmd.copy(); a_corr[:6] += c                      # our correction
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
        obs, _, done, _ = env.step(a_exec.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        q1 = np.array(obs["robot0_eef_quat"], float)
        y = np.concatenate([x1 - x0, rot_delta(q0, q1)]) / OUT

        u = a_corr[:6]                       # what we believe we sent (fault unknown to us)
        hist.appendleft(u)
        H = np.array(hist)                                          # (K+1, 6), newest first
        feat = H.reshape(-1) if W.shape[1] > K_FIR + 2 else None
        pred = (W[:, :-1] @ feat + W[:, -1]) if feat is not None else np.array(
            [W[i, :K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])
        r = y - pred
        if adapt:
            # Robustness, because the bare law drifts. P is identified on NOMINAL data, so
            # as the correction grows the executed action goes off-distribution, the plant
            # prediction degrades, the residual grows and f_hat chases it -- measured:
            # dz reached 0.642 against a true 0.05 on 2 of 6 episodes.
            #   normalisation  damps the update when the residual is large (an outlier step,
            #                  a contact, a joint limit) instead of trusting it
            #   deadzone       stops model noise from integrating into drift when the
            #                  residual is already at the plant's own fit error
            #   projection     keeps f_hat inside a physically plausible box
            # --law innov: normalise the STEP, not the target (review, 2026-09-01).
            # The legacy line divides the estimate itself, so with r ~= M f constant the
            # fixed point is f/(1+|r|^2/rho^2) -- biased LOW by 4.9% at the measured
            # |r|=0.034, rho=0.15, which matches the separation test landing at 0.049
            # against a true 0.050. Driving the innovation e = r - M f_hat to zero puts
            # the fixed point back at f exactly, and gives the estimate a way home: if the
            # fault clears, e = -M f_hat is still non-zero, so f_hat decays instead of
            # freezing a correction that is no longer needed.
            if law == "innov":
                e = r - (M @ f_hat if M is not None else np.linalg.solve(M_inv, f_hat))
                if bias is not None:
                    e = e - (M @ bias if M is not None else np.linalg.solve(M_inv, bias))
                ne = float(np.linalg.norm(e))
                if ne < dead:
                    step = np.zeros(6)
                else:
                    step = (M_inv @ e) / (1.0 + (ne / norm_r) ** 2)
                f_hat = np.clip(f_hat + gamma * step, -clip, clip)
                traj.append(dict(t=t, f_hat=f_hat.tolist(), r=r.tolist(), live=bool(live),
                         f_true=f_true_now.tolist()))
                if done:
                    return True, f_hat, traj
                t += 1
                continue
            if baseline == "integral":
                # BASELINE: textbook integral action on the RAW motion error, with no plant
                # model and no M. e = achieved - commanded, in action units; the correction
                # integrates it away. This is what a control engineer would write first, and
                # it is the honest test of whether the FIR plant and the sensitivity matrix
                # earn their place. If this matches the full law, the machinery is decoration.
                e_raw = y - a_corr[:6]
                f_hat = np.clip(f_hat + ki * e_raw, -clip, clip)
                traj.append(dict(t=t, f_hat=f_hat.tolist(), r=r.tolist(), live=bool(live),
                                 f_true=f_true_now.tolist()))
                if done:
                    return True, f_hat, traj
                t += 1
                continue
            est = M_inv @ r
            if bias is not None:
                # Zero the estimator against known-healthy data. On a fault-free run f_hat
                # settles at [+0.022 +0.007 +0.042 -0.006 +0.008 +0.001] instead of zero --
                # plant-model error that M^-1 maps into a phantom fault. It is measurable
                # offline on healthy rollouts, so it is subtracted here, exactly as one
                # zeroes a sensor before trusting its readings.
                est = est - bias
            nr = float(np.linalg.norm(r))
            if nr < dead:
                est = np.zeros(6)
            est = est / (1.0 + (nr / norm_r) ** 2)
            f_hat = np.clip(f_hat + gamma * (est - f_hat), -clip, clip)
        traj.append(dict(t=t, f_hat=f_hat.tolist(), r=r.tolist(), live=bool(live),
                         f_true=f_true_now.tolist()))
        if done:
            return True, f_hat, traj
        t += 1
    return False, f_hat, traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--log", type=pathlib.Path, required=True)
    p.add_argument("--openloop", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--baseline", choices=["none", "integral"], default="none",
                   help="integral: plain integral action on the raw motion error, "
                        "no plant model and no M. The ablation that asks whether "
                        "the identification machinery is doing any work.")
    p.add_argument("--ki", type=float, default=0.05,
                   help="integral gain for --baseline integral")
    p.add_argument("--profile", choices=["step", "ramp", "sine", "sine_bias", "intermittent"],
                   default="step", help="how the fault varies in time")
    p.add_argument("--prof-p", type=float, default=60.0,
                   help="ramp length / sine period / intermittent half-period, in steps")
    p.add_argument("--law", choices=["legacy", "innov"], default="legacy",
                   help="legacy: normalise the estimate (biased low ~5%). "
                        "innov: normalise the step, unbiased fixed point.")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--static-corr", default=None,
                   help="fixed 6-vector correction; the no-estimation baseline")
    p.add_argument("--wrist-shift", type=int, default=0,
                   help="pixels to roll the wrist camera -- a fault the plant never sees")
    p.add_argument("--obs-offset", default=None,
                   help="x,y,z bias on the position SENSOR -- a fault the residual cannot see")
    p.add_argument("--onset", type=int, default=0,
                   help="control step at which the fault appears (0 = from the start)")
    p.add_argument("--eval-init", type=int, default=45,
                   help="first evaluation initial state (keep fresh per run)")
    p.add_argument("--fault-vec", default=None,
                   help="6 comma-separated per-dim fault values (overrides --sev)")
    p.add_argument("--corr-dims", default=None,
                   help="comma-separated dims to correct, e.g. 3,4,5 for rotation only")
    p.add_argument("--bias", default=None,
                   help="6 comma-separated values: the estimator bias measured on healthy runs")
    p.add_argument("--estimate-only", action="store_true",
                   help="update f_hat but never apply it; isolates estimator feedback")
    p.add_argument("--mimo", action="store_true",
                   help="coupled plant: fits better, extrapolates worse (see fit_plant)")
    p.add_argument("--dead", type=float, default=0.05, help="residual deadzone")
    p.add_argument("--norm-r", type=float, default=0.5, help="update normalisation")
    p.add_argument("--clip", type=float, default=0.15, help="projection box on f_hat")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--task-stride", type=int, default=1,
                   help="step between task ids; >1 samples a large suite evenly")
    p.add_argument("--sev", type=float, default=0.05)
    a = p.parse_args()

    W = fit_plant(a.log, mimo=a.mimo)
    M = np.array(json.loads(a.openloop.read_text())["M"])
    M_inv = np.linalg.pinv(M)
    print(f"plant identified; cond(M) = {np.linalg.cond(M):.1f}, gamma = {a.gamma}\n")

    bias = np.array([float(x) for x in a.bias.split(",")]) if a.bias else None
    if bias is not None:
        print(f"estimator zeroed against healthy-run bias {np.round(bias,3)}")
    cdims = [int(x) for x in a.corr_dims.split(",")] if a.corr_dims else None
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
    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=False))
    res = {"gamma": a.gamma, "arms": {}}
    # Spread episodes across the WHOLE suite. The old form, i % 10, silently confined a
    # libero_90 run to its first ten tasks. With --task-stride 4 on 90 tasks, 20 episodes
    # land on tasks 0, 4, 8, ... 76 -- a sample of the suite rather than a corner of it.
    n_tasks = pr.suite.n_tasks
    eps = [(((i * a.task_stride) % n_tasks), a.eval_init + (i * a.task_stride) // n_tasks)
           for i in range(a.episodes)]
    for tag, adapt in (("frozen_faulted", False), ("adaptive", True)):
        ok, fh, trajs, per_ep = 0, [], [], []
        for tid, init in eps:
            s, f_hat, traj = run(pr, tid, init, a.sev, M_inv, W, a.gamma, adapt,
                                 dead=a.dead, norm_r=a.norm_r, clip=a.clip,
                                 apply_corr=not a.estimate_only, bias=bias,
                                 corr_dims=cdims, fvec=fvec, onset=a.onset,
                                 obs_off=obs_off, wrist_shift=a.wrist_shift,
                                 static_c=static_c, law=a.law, M=M,
                                 profile=a.profile, prof_p=a.prof_p,
                                 baseline=a.baseline, ki=a.ki)
            ok += int(s); fh.append(f_hat.tolist())
            # Per-episode outcome, keyed by (task, init). The arms run on the SAME episode
            # list, so these pair up -- which is what McNemar needs and what the earlier
            # runs threw away by only accumulating a total. See mcnemar.py.
            per_ep.append(dict(task=int(tid), init=int(init), ok=bool(s)))
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
