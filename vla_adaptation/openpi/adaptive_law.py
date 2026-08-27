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


def fit_plant(log_path):
    """FIR plant per output dim, identified on the NOMINAL episodes only."""
    d = json.loads(pathlib.Path(log_path).read_text())
    A = np.array(d[0]["raw_a"]); D = np.array(d[0]["raw_d"]); lens = d[0]["ep_len"]
    W = []
    for i in range(6):
        X, Y, o = [], [], 0
        for L in lens:
            a, y = A[o:o+L, i], D[o:o+L, i] / OUT[i]; o += L
            for t in range(K_FIR, L):
                X.append(np.concatenate([a[t-K_FIR:t+1][::-1], [1.0]])); Y.append(y[t])
        w, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
        W.append(w)
    return np.array(W)                       # (6, K_FIR+2)


def run(pr, tid, init, sev, M_inv, W, gamma, adapt, max_steps=220,
        dead=0.05, norm_r=0.5, clip=0.15):
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
        c = -f_hat if adapt else np.zeros(6)
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
        pred = np.array([W[i, :K_FIR+1] @ H[:, i] + W[i, -1] for i in range(6)])
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
    p.add_argument("--dead", type=float, default=0.05, help="residual deadzone")
    p.add_argument("--norm-r", type=float, default=0.5, help="update normalisation")
    p.add_argument("--clip", type=float, default=0.15, help="projection box on f_hat")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--sev", type=float, default=0.05)
    a = p.parse_args()

    W = fit_plant(a.log)
    M = np.array(json.loads(a.openloop.read_text())["M"])
    M_inv = np.linalg.pinv(M)
    print(f"plant identified; cond(M) = {np.linalg.cond(M):.1f}, gamma = {a.gamma}\n")

    pr = Probe(a)
    pr.control(dict(site=None, pin_rng=False))
    res = {"gamma": a.gamma, "arms": {}}
    eps = [((i % 10), 45 + i // 10) for i in range(a.episodes)]
    for tag, adapt in (("frozen_faulted", False), ("adaptive", True)):
        ok, fh = 0, []
        for tid, init in eps:
            s, f_hat, traj = run(pr, tid, init, a.sev, M_inv, W, a.gamma, adapt,
                                 dead=a.dead, norm_r=a.norm_r, clip=a.clip)
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
