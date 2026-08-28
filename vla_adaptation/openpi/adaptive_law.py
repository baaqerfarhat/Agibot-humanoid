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


def run(pr, tid, init, sev, M_inv, W, gamma, adapt, max_steps=220,
        dead=0.05, norm_r=0.5, clip=0.15, apply_corr=True, bias=None):
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
        if not plan:
            plan.extend(pr.client.infer({
                "observation/image": img, "observation/wrist_image": wr,
                "observation/state": np.concatenate((obs["robot0_eef_pos"],
                    lm._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "prompt": str(desc)})["actions"][: pr.a.replan_steps])
        a_cmd = np.asarray(plan.popleft(), float)
        # apply_corr=False estimates but does NOT act. r = M f + eps(a+c): with an
        # imperfect plant the residual carries the model error evaluated at the operating
        # point, so once c moves that point the estimate chases its own correction. Freezing
        # c at zero separates open-loop model bias from that feedback.
        c = -f_hat if (adapt and apply_corr) else np.zeros(6)
        a_corr = a_cmd.copy(); a_corr[:6] += c                      # our correction
        a_exec = apply_action_fault(a_corr, "offset", sev, 6)       # then the world's fault
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
        traj.append(dict(t=t, f_hat=f_hat.tolist(), r=r.tolist()))
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
    p.add_argument("--sev", type=float, default=0.05)
    a = p.parse_args()

    W = fit_plant(a.log, mimo=a.mimo)
    M = np.array(json.loads(a.openloop.read_text())["M"])
    M_inv = np.linalg.pinv(M)
    print(f"plant identified; cond(M) = {np.linalg.cond(M):.1f}, gamma = {a.gamma}\n")

    bias = np.array([float(x) for x in a.bias.split(",")]) if a.bias else None
    if bias is not None:
        print(f"estimator zeroed against healthy-run bias {np.round(bias,3)}")
    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=False))
    res = {"gamma": a.gamma, "arms": {}}
    eps = [((i % 10), 45 + i // 10) for i in range(a.episodes)]
    for tag, adapt in (("frozen_faulted", False), ("adaptive", True)):
        ok, fh = 0, []
        for tid, init in eps:
            s, f_hat, traj = run(pr, tid, init, a.sev, M_inv, W, a.gamma, adapt,
                                 dead=a.dead, norm_r=a.norm_r, clip=a.clip,
                                 apply_corr=not a.estimate_only, bias=bias)
            ok += int(s); fh.append(f_hat.tolist())
            print(f"  [{tag}] task {tid} init {init}: success={s}  "
                  f"f_hat={np.round(f_hat, 3)}")
        res["arms"][tag] = dict(successes=ok, n=len(eps), f_hat=fh)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"{tag}: {ok}/{len(eps)} = {100*ok/len(eps):.0f}%\n")
    print("true fault = +0.05 on all six arm dims")


if __name__ == "__main__":
    main()
