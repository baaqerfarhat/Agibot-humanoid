"""Adaptive correction of a MULTIPLICATIVE (gain) fault -- a state-dependent regressor.

The offset law in adaptive_law.py estimates a CONSTANT vector. That is the special case of
the MRAC gradient form with regressor phi = I, and it structurally cannot correct a fault
whose size depends on the command. A gain fault is exactly that:

    u = g . (a + c)                 g unknown, diagonal, g = 1 nominal

Derivation. With P the fault-free plant and M = dP/du its DC gain,

    r  =  y - P(a + c)  =  P(g.(a+c)) - P(a+c)  ~  M ((g-1) . psi),    psi = a + c
    z  =  M^-1 r        ~  beta . psi,                                 beta = g - 1

so the regressor is phi = diag(psi) rather than I, and each dimension is a scalar
regression z_i = beta_i psi_i. Normalised LMS:

    beta_i <- beta_i + gamma psi_i (z_i - beta_i psi_i) / (eps + psi_i^2)

To cancel we need u = a, i.e. c = a (1/g - 1), so

    c = -a . beta_hat / (1 + beta_hat)

PERSISTENCY OF EXCITATION. This is the new failure mode, absent for a constant offset: where
psi_i ~ 0 the command carries no information about beta_i and the update is pure noise
division. Dimensions with |psi_i| below a threshold are therefore frozen, not updated -- the
estimator waits for the policy to excite that axis.
"""
from __future__ import annotations

import argparse, collections, json, pathlib
import numpy as np

import main as lm
from so3 import rot_delta
from openpi_client import image_tools
from paired_probe import Probe
from gate_faults import apply_action_fault
from adaptive_law import fit_plant, OUT, K_FIR


def run(pr, tid, init, gain, M_inv, W, gamma, adapt, pe_min, clip, max_steps=220,
        rls=True, lam=0.999, dither=0.0, dither_dims=(3, 4, 5), corr_dims=None,
        g_min=0.05, g_max=3.0, intercept=False):
    env, desc, inits = pr.env_for(tid)
    env.reset(); obs = env.set_init_state(inits[init])
    plan, t = collections.deque(), 0
    hist = collections.deque([np.zeros(6)] * (K_FIR + 1), maxlen=K_FIR + 1)
    beta = np.zeros(6)                      # beta = g - 1, nominal 0
    n_upd = np.zeros(6)
    # RLS covariance per dim. LMS with a fixed step over-corrects the strongly excited
    # channels -- dx and dz both saturated the projection bound at 0.8 against a true 0.5 --
    # because the step does not shrink as evidence accumulates. P_i does exactly that.
    P = np.full(6, 1e3)
    theta = np.zeros((6, 2))                # [b_i, beta_i] per channel
    Pm = np.array([np.diag([1e3, 1e3]) for _ in range(6)])
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
        # c = a (1/g - 1) with g_hat = 1 + beta_hat
        # a_corr = a + c = a/(1+beta_hat): the EXACT certainty-equivalent inverse gain,
        # not a first-order approximation. That exactness is the problem. The applied factor
        # 1/(1+beta_hat) has derivative -1/(1+beta)^2 = -4 at the true beta = -0.5, so
        # estimation error is AMPLIFIED fourfold by the compensator. Measured: beta_hat_z =
        # -0.797 applies 4.93x where 2.00x is correct (+146%). An additive fault's
        # compensator is linear in the estimate and forgives that error; a gain fault's
        # inverts it and does not. Hence projection: refuse to invert a gain we do not
        # believe. g_min is a PHYSICAL prior -- below it we decline to repair rather than
        # command a large multiple -- and is not centred on the true value.
        g_hat = np.clip(1.0 + beta, g_min, g_max)
        c = a_cmd[:6] * (1.0 / g_hat - 1.0) if adapt else np.zeros(6)
        if corr_dims is not None:
            # Restrict the correction to nominated channels. For an OFFSET fault the theory
            # says rotation is identifiable; for a GAIN fault, regressor diag(psi), it says
            # TRANSLATION -- loud where the command is large. That is a prediction, and this
            # flag is what tests it rather than fitting the restriction after the fact.
            m = np.zeros(6); m[list(corr_dims)] = 1.0
            c = c * m
        a_corr = a_cmd.copy(); a_corr[:6] += c
        if adapt and dither:
            # Deliberate excitation. A gain is identifiable only where the command moves
            # that axis, and the policy barely rotates the wrist -- drx/dry/drz received
            # 0-6 updates per episode against 100+ for translation. A small alternating
            # probe makes those axes identifiable at the cost of a tiny command perturbation.
            for j in dither_dims:
                a_corr[j] += dither * (1.0 if (t // 4) % 2 == 0 else -1.0)
        psi = a_corr[:6].copy()
        a_exec = apply_action_fault(a_corr, "gain", gain, 6)
        x0 = np.array(obs["robot0_eef_pos"], float)
        q0 = np.array(obs["robot0_eef_quat"], float)
        obs, _, done, _ = env.step(a_exec.tolist())
        x1 = np.array(obs["robot0_eef_pos"], float)
        q1 = np.array(obs["robot0_eef_quat"], float)
        y = np.concatenate([x1 - x0, rot_delta(q0, q1)]) / OUT
        hist.appendleft(psi)
        H = np.array(hist)
        pred = np.array([W[i, :K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])
        z = M_inv @ (y - pred)
        if adapt:
            for i in range(6):
                if abs(psi[i]) < pe_min:        # not excited -> not identifiable -> hold
                    continue
                if intercept:
                    # Regress z_i = b_i + beta_i * psi_i, estimating BOTH.
                    #
                    # Without b_i this fits a line through the origin to data that does not
                    # pass through it: the plant model carries a constant bias (the same
                    # phantom the offset law subtracts via --bias, +0.042 on z). With no
                    # intercept beta must absorb b/psi, which diverges as psi -> 0. Measured:
                    # on a HEALTHY robot the no-intercept law reports beta_x = -0.327,
                    # beta_z = -0.337 and rails the clip in 40% of episodes (Sec 12.2).
                    # The intercept gives that bias somewhere to go.
                    phi = np.array([1.0, psi[i]])
                    th = theta[i]
                    Pi = Pm[i]
                    err = z[i] - phi @ th
                    k = Pi @ phi / (lam + phi @ Pi @ phi)
                    theta[i] = th + k * err
                    Pm[i] = (Pi - np.outer(k, phi @ Pi)) / lam
                    beta[i] = theta[i][1]
                else:
                    err = z[i] - beta[i] * psi[i]
                    if rls:
                        k = P[i] * psi[i] / (lam + P[i] * psi[i] ** 2)   # RLS gain, shrinks
                        beta[i] += k * err
                        P[i] = (P[i] - k * psi[i] * P[i]) / lam
                    else:
                        beta[i] += gamma * psi[i] * err / (1e-6 + psi[i] ** 2)
                n_upd[i] += 1
            beta = np.clip(beta, -clip, clip)
            if intercept:
                theta[:, 1] = beta
        if done:
            return True, beta, n_upd
        t += 1
    return False, beta, n_upd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--log", type=pathlib.Path, required=True)
    p.add_argument("--openloop", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--gain", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--pe-min", type=float, default=0.15, help="excitation floor on |psi_i|")
    p.add_argument("--clip", type=float, default=0.8)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--lms", action="store_true", help="plain LMS instead of RLS")
    p.add_argument("--lam", type=float, default=0.999, help="RLS forgetting")
    p.add_argument("--g-min", type=float, default=0.05,
                   help="floor on the estimated gain before inverting it. 0.05 = "
                        "effectively off (20x authority); 0.35 = decline to repair "
                        "below 35%% remaining authority.")
    p.add_argument("--g-max", type=float, default=3.0)
    p.add_argument("--intercept", action="store_true",
                   help="estimate a per-channel offset alongside the gain. Without it "
                        "beta absorbs the plant-model bias and diverges as the command "
                        "goes to zero -- on a HEALTHY robot it reports a 33%% gain loss.")
    p.add_argument("--corr-dims", default=None,
                   help="dims to correct; 0,1,2 = translation, 3,4,5 = rotation")
    p.add_argument("--dither", type=float, default=0.0,
                   help="excitation amplitude on the rotation axes")
    a = p.parse_args()

    W = fit_plant(a.log)
    M_inv = np.linalg.pinv(np.array(json.loads(a.openloop.read_text())["M"]))
    cdims = [int(x) for x in a.corr_dims.split(',')] if a.corr_dims else None
    if cdims is not None:
        print(f'correction restricted to dims {cdims}')
    pr = Probe(a); pr.control(dict(site=None, pin_rng=False))
    true_beta = a.gain - 1.0
    print(f"gain fault g = {a.gain}  -> true beta = {true_beta:+.2f}; gamma={a.gamma}, "
          f"PE floor={a.pe_min}\n")

    eps = [((i % 10), 45 + i // 10) for i in range(a.episodes)]
    res = {"gain": a.gain, "true_beta": true_beta, "arms": {}}
    for tag, adapt in (("frozen_faulted", False), ("adaptive_gain", True)):
        ok, B, U, per_ep = 0, [], [], []
        for tid, init in eps:
            s, beta, n_upd = run(pr, tid, init, a.gain, M_inv, W, a.gamma, adapt,
                                 a.pe_min, a.clip, rls=not a.lms, lam=a.lam,
                                 dither=a.dither, corr_dims=cdims,
                                 g_min=a.g_min, g_max=a.g_max,
                                 intercept=a.intercept)
            ok += int(s)
            per_ep.append(dict(task=int(tid), init=int(init), ok=bool(s))); B.append(beta.tolist()); U.append(n_upd.tolist())
            print(f"  [{tag}] task {tid}: success={s}  beta={np.round(beta,2)}  "
                  f"updates={n_upd.astype(int)}")
        res["arms"][tag] = dict(successes=ok, per_ep=per_ep, n=len(eps), beta=B, updates=U)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"{tag}: {ok}/{len(eps)} = {100*ok/len(eps):.0f}%\n")
    B = np.array(res["arms"]["adaptive_gain"]["beta"])
    print(f"beta_hat mean = {np.round(B.mean(0),3)}   true = {true_beta:+.2f}")


if __name__ == "__main__":
    main()
