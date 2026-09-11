"""re4 Part G.1: how faithfully does a native decoder-bias edit realise an intended final action?
(docs/RE4_EVIDENCE_PLAN.md item G.1; prereg_records/PREREG_RE4_G1_DECODER.md)

The ACE experiments edit pi0.5's action_out_proj bias in place (ace_server.py `bias_add`) instead
of subtracting a correction from the executed action. At a fixed observation and a pinned
flow-sampler draw (pin_rng), this script queries the served policy with and without bias edits
and measures, on the first executed LIBERO action (7 dims, unnormalised):
  J     small-edit Jacobian w.r.t. the 7 LIBERO bias dims (central difference at the smallest edit)
  D     least-squares linear map over every edit magnitude;  ||D - J||_F / ||J||_F
  Q     even (quadratic) part (a(+m e_j) + a(-m e_j))/2 - a0 = Q_j m^2
  zeta  realisation error: for intended corrections c = +-0.05 e_i (i = 0..5), apply b = J^-1 c and
        measure ||a(b) - a0 - c|| / ||c||
The re4 appendix symbols are not on this machine; these are operational definitions.
MUST NOT overlap any rollout run: the edit lands on the shared server for every client.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys, time
import numpy as np

BIAS_DIM, LIB, ARM = 32, 7, 6
MAGS = (0.02, 0.05, 0.1)
C_MAG = 0.05


def measure(set_edit, query, n_obs):
    """set_edit(vec32) applies an edit; query(i) -> (first_action[7], chunk_mean[7]) for obs i."""
    zero = np.zeros(BIAS_DIM)
    set_edit(zero)
    a0 = np.array([query(i)[0] for i in range(n_obs)])
    a0_rep = np.array([query(i)[0] for i in range(n_obs)])
    m0 = np.array([query(i)[1] for i in range(n_obs)])
    resp, resp_mean = {}, {}
    for j in range(LIB):
        for m in MAGS:
            for s in (1.0, -1.0):
                v = zero.copy(); v[j] = s * m; set_edit(v)
                q = [query(i) for i in range(n_obs)]
                resp[(j, s * m)] = np.array([x[0] for x in q]); resp_mean[(j, s * m)] = np.array([x[1] for x in q])
    mn = MAGS[0]
    Jn = np.zeros((n_obs, LIB, LIB))
    for j in range(LIB):
        Jn[:, :, j] = (resp[(j, mn)] - resp[(j, -mn)]) / (2 * mn)
    J = Jn.mean(axis=0)
    X, Y = [], []
    for (j, d), R in resp.items():
        b = np.zeros(LIB); b[j] = d
        for i in range(n_obs):
            X.append(b); Y.append(R[i] - a0[i])
    D = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0].T
    Q = np.zeros((LIB, LIB)); quad_rel = np.zeros(LIB)
    for j in range(LIB):
        num = np.zeros(LIB); den = 0.0
        for m in MAGS:
            even = ((resp[(j, m)] + resp[(j, -m)]) / 2 - a0).mean(axis=0)
            num += even * m ** 2; den += m ** 4
        Q[:, j] = num / den
        mx = MAGS[-1]
        quad_rel[j] = np.linalg.norm(Q[:, j] * mx ** 2) / max(np.linalg.norm(J[:, j] * mx), 1e-12)
    chunk_ratio = [float(np.mean((resp_mean[(j, MAGS[-1])] - m0)[:, j]) / np.mean((resp[(j, MAGS[-1])] - a0)[:, j]))
                   if abs(np.mean((resp[(j, MAGS[-1])] - a0)[:, j])) > 1e-9 else float("nan") for j in range(LIB)]
    zeta = {}
    for i in range(ARM):
        for s in (1.0, -1.0):
            c = np.zeros(LIB); c[i] = s * C_MAG
            b = np.linalg.solve(J, c); v = np.zeros(BIAS_DIM); v[:LIB] = b; set_edit(v)
            real = np.array([query(k)[0] for k in range(n_obs)]) - a0
            zeta[f"{'+' if s > 0 else '-'}{i}"] = (np.linalg.norm(real - c, axis=1) / C_MAG).tolist()
    set_edit(zero)
    diag = np.abs(np.diag(J))[:ARM]
    off = np.array([np.max(np.abs(np.delete(J[r, :ARM], r))) for r in range(ARM)])
    Jd = np.array([np.diag(Jn[i])[:ARM] for i in range(n_obs)])
    z_all = np.concatenate([np.array(v) for v in zeta.values()])
    return dict(
        n_obs=n_obs, magnitudes=list(MAGS), correction_magnitude=C_MAG,
        determinism_max_abs=float(np.abs(a0 - a0_rep).max()),
        J=J.tolist(), J_diag=np.diag(J).tolist(),
        J_offdiag_over_diag_max_per_row=(off / np.maximum(diag, 1e-12)).tolist(),
        J_diag_across_obs_cv=(Jd.std(axis=0) / np.maximum(np.abs(Jd.mean(axis=0)), 1e-12)).tolist(),
        D=D.tolist(), D_minus_J_rel_fro=float(np.linalg.norm(D - J) / np.linalg.norm(J)),
        Q=Q.tolist(), quadratic_over_linear_at_max_edit=quad_rel.tolist(),
        chunk_mean_over_first_action_diag=chunk_ratio,
        zeta_per_axis=zeta, zeta_median=float(np.median(z_all)), zeta_p95=float(np.percentile(z_all, 95)),
        zeta_max=float(z_all.max()),
        raw=dict(a0=a0.tolist(), responses={f"{j}:{d:+g}": R.tolist() for (j, d), R in resp.items()}))


def selftest():
    rng = np.random.default_rng(0); n = 12
    a0s = rng.normal(size=(n, LIB)); Jt = np.diag(rng.uniform(0.2, 1.0, LIB)) + 0.01 * rng.normal(size=(LIB, LIB))
    Qt = 0.05 * rng.normal(size=(LIB, LIB)); cur = {"b": np.zeros(BIAS_DIM)}
    def set_edit(v): cur["b"] = np.asarray(v, float)
    def query(i):
        b = cur["b"][:LIB]; a = a0s[i] + Jt @ b + Qt @ (b ** 2)
        return a, a
    r = measure(set_edit, query, n)
    assert r["determinism_max_abs"] == 0.0
    assert np.abs(np.array(r["J"]) - Jt).max() < 5e-3, np.abs(np.array(r["J"]) - Jt).max()
    assert np.abs(np.array(r["Q"]) - Qt).max() < 1e-6
    assert r["zeta_median"] < 0.05, r["zeta_median"]
    print(f"g1 selftest passed: J recovered to {np.abs(np.array(r['J']) - Jt).max():.1e}, Q exact, "
          f"zeta median {r['zeta_median']:.4f}, D-J rel {r['D_minus_J_rel_fro']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--control", type=pathlib.Path); ap.add_argument("--ack", type=pathlib.Path)
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--mid-steps", type=int, default=40); ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import paired_probe as PP
    pr = PP.Probe(a); IT = PP.image_tools; LM = PP.libero_main

    def el(obs, desc):
        img = IT.convert_to_uint8(IT.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wr = IT.convert_to_uint8(IT.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        return {"observation/image": img, "observation/wrist_image": wr,
                "observation/image_raw": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                "observation/wrist_image_raw": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                "observation/state": np.concatenate((obs["robot0_eef_pos"], LM._quat2axisangle(obs["robot0_eef_quat"]),
                                                     obs["robot0_gripper_qpos"])), "prompt": str(desc)}

    acks = []
    def set_edit(v):
        ack = pr.control(dict(bias_add=np.asarray(v, float).tolist(), pin_rng=True, draw=f"g1:{np.round(v[:LIB], 4).tolist()}"))
        if not ack.get("ok"):
            raise RuntimeError(f"bias edit did not land as requested: {ack}")
        acks.append(dict(requested_l2=ack.get("requested_l2"), applied_l2=ack.get("applied_l2")))

    set_edit(np.zeros(BIAS_DIM))             # baseline weights, pinned sampler, before capturing
    obs_list, meta = [], []
    for tid in range(10):
        for init in (45, 46):                 # first policy step of each headline scenario
            env, desc, inits = pr.env_for(tid); env.reset(); obs = env.set_init_state(inits[init])
            for _ in range(10):
                obs, _, _, _ = env.step(LM.LIBERO_DUMMY_ACTION)
            obs_list.append(el(obs, desc)); meta.append(dict(task=tid, init=init, step=0))
    for tid in range(10):                     # mid-episode states under the healthy policy
        env, desc, inits = pr.env_for(tid); env.reset(); obs = env.set_init_state(inits[45])
        for _ in range(10):
            obs, _, _, _ = env.step(LM.LIBERO_DUMMY_ACTION)
        plan = collections.deque(); k = 0
        while k < a.mid_steps:
            if not plan:
                plan.extend(np.asarray(pr.client.infer(el(obs, desc))["actions"], float)[: a.replan_steps])
            obs, _, done, _ = env.step(list(plan.popleft())); k += 1
            if done:
                break
        obs_list.append(el(obs, desc)); meta.append(dict(task=tid, init=45, step=k))

    def query(i):
        ch = np.asarray(pr.client.infer(obs_list[i])["actions"], float)
        return ch[0, :LIB], ch[:, :LIB].mean(axis=0)

    t0 = time.time()
    res = measure(set_edit, query, len(obs_list))
    pr.control(dict(site=None, pin_rng=False))     # leave the server clean
    res.update(observations=meta, server_acks=acks, seconds=time.time() - t0,
               definitions=__doc__, note="first executed action of the chunk, LIBERO action space after unnormalisation")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}: determinism {res['determinism_max_abs']:.2e}, D-J rel {res['D_minus_J_rel_fro']:.3f}, "
          f"zeta median {res['zeta_median']:.3f} p95 {res['zeta_p95']:.3f} max {res['zeta_max']:.3f}")


if __name__ == "__main__":
    main()
