"""LAYER-FAULT CORRESPONDENCE, phase 2: the 2x2 search + ACE arm.

Pre-registered in theory_2026-08-15/PREREG_LAYER_CORRESPONDENCE.md BEFORE this run.
For ONE fault cell (--fault obs_bias | joint_offset, vectors loaded exactly from
outputs/layer_fault_gate.json), run:

  b6 search : CEM over a constant 20-d action residual (= mlp.6 bias), sigma0 0.04
  b0 search : CEM over 20 obs-channel coefficients a, db0 = sum_i a_i * u_i with
              u_i = -W0[:, sl_i]/(obs_std_i + eps)  (white-box, no gradients),
              applied as the double-forward action residual, sigma0 0.15 rad
  ACE arm   : 8 random draws per class at the class's own prior scale, scored on the
              train seeds -> ACE_hat = mean effect vs frozen (the ACC-2026 estimator)

Everything is clipped to the SAME deployed envelope u_max = 0.10, so a b0-vs-b6 difference
is attributable to the function class, not the budget. Settled fix = best-ever by train
score (const_adapt convention). Held-out seeds are disjoint from train.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from x2_ttcl.backends.mjlab_backend import MjlabBackend
from x2_ttcl.backends.rom_command import forward_command
from x2_ttcl.policy.actor import RomFomActor

OUT = Path.home() / "theory_ws/x2_ttcl/outputs"
CLEAR = {"action_delay_steps": 0, "joint_offset": None, "joint_delay": None,
         "joint_gain": None, "obs_bias": None, "joint_friction": None,
         "payload": None, "action_lag": None, "dof_friction": None,
         "dof_damping_scale": None, "ground_friction": None,
         "inertia_scale": None, "gravity_tilt": None, "ext_force": None,
         "motor_kp_scale": None, "motor_kd_scale": None, "gear_scale": None,
         "torque_limit_scale": None, "armature_scale": None}
U_MAX = 0.10
N_PRE, N_POST = 100, 300
POP, ITERS, ELITE = 10, 8, 3
SIGMA0 = {"b6": 0.04, "b0": 0.15, "w6": 0.15}
TRAIN = [2000, 2001]
HELDOUT = [3000, 3001, 3002, 3003, 3004, 3005]
ACE_DRAWS = 8


def episode(be, seed, cond_extra, res_fn):
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(N_PRE):
        be.step(residual=None)
    be.set_condition({**CLEAR, **cond_extra})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    for k in range(N_POST):
        x = float(be.observe()["base_pos"][0])
        be.step(residual=None if res_fn is None else res_fn(be))
        if be.fallen():
            return k + 1, x - x0
    return N_POST, x - x0


def score(steps, dist):
    return dist + 0.002 * steps


def evaluate(be, cond, res_fn, seeds):
    st, di = zip(*[episode(be, s, cond, res_fn) for s in seeds])
    return (float(np.mean([score(a, b) for a, b in zip(st, di)])),
            float(np.mean(st)), float(np.mean(di)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=["obs_bias", "joint_offset", "tq05", "joint_gain"],
                    required=True)
    ap.add_argument("--mag", type=float, required=True,
                    help="gate magnitude identifying the cell (tq05: the torque scale, 0.5)")
    a = ap.parse_args()
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"

    gate = json.loads((OUT / "layer_fault_gate.json").read_text())
    s0, s1 = gate["jp_slice"]
    cellrec = None
    if a.fault == "joint_gain":
        jg = json.loads((OUT / "jg_gate.json").read_text())
        cellrec = next(c for c in jg["cells"] if abs(c["mag"] - a.mag) < 1e-9)
    elif a.fault != "tq05":
        cellrec = next(c for c in gate["cells"]
                       if c["fault"] == a.fault and abs(c["mag"] - a.mag) < 1e-9)

    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        actor = RomFomActor.from_checkpoint().eval()
        actor_mod = RomFomActor.from_checkpoint().eval()
        b0_base = actor.mlp[0].bias.detach().clone()

        W0 = actor.mlp[0].weight.detach().numpy()
        sig = actor.obs_std.detach().numpy() + actor.eps
        U = np.stack([-W0[:, s0 + i] / sig[s0 + i] for i in range(s1 - s0)], axis=1)  # 512 x 20

        if a.fault == "obs_bias":
            vec = np.asarray(cellrec["delta"], float)
            cond = {"obs_bias": vec, "obs_bias_slice": (s0, s1)}
            alpha_star = vec.copy()                      # a* = Delta in the u_i basis
            db0_star = U @ alpha_star
        elif a.fault == "joint_offset":
            vec = np.asarray(cellrec["offset"], float)
            cond = {"joint_offset": vec}
            alpha_star, db0_star = None, None
        elif a.fault == "joint_gain":                    # fault C (follow-up, 00:45 Aug 19)
            vec = np.asarray(cellrec["gain"], float)
            cond = {"joint_gain": vec}
            alpha_star, db0_star = None, None
            theta_star = (1.0 / vec) - 1.0               # exact multiplicative inverse
        else:                                            # tq05 (prereg amendment, 17:08)
            vec = None
            cond = {"torque_limit_scale": a.mag}
            alpha_star, db0_star = None, None
        if a.fault != "joint_gain":
            theta_star = None

        def res_b0(alpha):
            db0 = torch.as_tensor(U @ alpha, dtype=torch.float32)
            def fn(be_):
                o = torch.as_tensor(be_.policy_obs())[None]
                with torch.no_grad():
                    actor_mod.mlp[0].bias.copy_(b0_base + db0)
                    d = (actor_mod(o) - actor(o))[0].numpy()
                return np.clip(d, -U_MAX, U_MAX)
            return fn

        def res_b6(r):
            rr = np.clip(np.asarray(r, float), -U_MAX, U_MAX)
            return lambda be_: rr

        def res_w6(theta):
            th = np.asarray(theta, float)
            def fn(be_):
                o = torch.as_tensor(be_.policy_obs())[None]
                with torch.no_grad():
                    an = actor(o)[0].numpy()
                return np.clip(th * an, -U_MAX, U_MAX)
            return fn

        make = {"b0": res_b0, "b6": res_b6, "w6": res_w6}
        classes = ("b6", "b0", "w6") if a.fault == "joint_gain" else ("b6", "b0")
        rng_search = {"b6": 0, "b0": 1, "w6": 2}
        rng_ace = {"b6": 10, "b0": 11, "w6": 12}

        base, bs, bd = evaluate(be, cond, None, TRAIN)
        print(f"[train] frozen: score {base:+.3f}  steps {bs:.1f}  dist {bd:+.2f}", flush=True)

        results = {"fault": a.fault, "mag": a.mag,
                   "vec": None if vec is None else vec.tolist(),
                   "config": {"pop": POP, "iters": ITERS, "elite": ELITE,
                              "sigma0": SIGMA0, "u_max": U_MAX, "train": TRAIN,
                              "heldout": HELDOUT, "settled": "best_ever_train"}}

        found = {}
        for cls in classes:
            rng = np.random.default_rng(rng_search[cls])
            mu, sg = np.zeros(20), np.full(20, SIGMA0[cls])
            best = (base, np.zeros(20))
            print(f"== SEARCH {cls} (sigma0 {SIGMA0[cls]})", flush=True)
            for it in range(ITERS):
                pool = rng.normal(mu, sg, size=(POP, 20))
                if cls == "b6":
                    pool = np.clip(pool, -U_MAX, U_MAX)
                vals = []
                for th in pool:
                    v, _, _ = evaluate(be, cond, make[cls](th), TRAIN)
                    vals.append(v)
                    if v > best[0]:
                        best = (v, th.copy())
                idx = np.argsort(vals)[::-1][:ELITE]
                mu, sg = pool[idx].mean(0), pool[idx].std(0) + 1e-3
                print(f"    it{it+1}/{ITERS} best {best[0]:+.3f}  "
                      f"elite {np.mean([vals[i] for i in idx]):+.3f}  "
                      f"||mu|| {np.linalg.norm(mu):.3f}", flush=True)
            found[cls] = best[1]
            results[f"{cls}_found"] = best[1].tolist()
            results[f"{cls}_train_best"] = best[0]

        print("== ACE arm (8 draws per class at the class prior scale, train seeds)",
              flush=True)
        ace = {}
        for cls in classes:
            rng = np.random.default_rng(rng_ace[cls])
            effs = []
            target = SIGMA0[cls] * np.sqrt(20)
            for j in range(ACE_DRAWS):
                th = rng.normal(0.0, 1.0, 20); th *= target / np.linalg.norm(th)
                v, _, _ = evaluate(be, cond, make[cls](th), TRAIN)
                effs.append(v - base)
            ace[cls] = {"mean": float(np.mean(effs)), "sd": float(np.std(effs)),
                        "effects": effs}
            print(f"    ACE_hat[{cls}] = {np.mean(effs):+.3f}  (sd {np.std(effs):.3f})",
                  flush=True)
        results["ace"] = ace

        print("== HELD-OUT, seeds", HELDOUT, flush=True)
        nom_v, nom_s, nom_d = evaluate(be, {}, None, HELDOUT)
        res = {"nominal": {"steps": nom_s, "dist": nom_d}}
        _, fs, fd = evaluate(be, cond, None, HELDOUT)
        res["frozen"] = {"steps": fs, "dist": fd}
        print(f"  nominal  {nom_s:7.1f} steps  {nom_d:+.2f} m", flush=True)
        print(f"  frozen   {fs:7.1f} steps  {fd:+.2f} m", flush=True)
        hr = nom_d - fd
        for cls in classes:
            _, ss, dd = evaluate(be, cond, make[cls](found[cls]), HELDOUT)
            rec = (dd - fd) / hr if hr > 0.05 else float("nan")
            res[cls] = {"steps": ss, "dist": dd, "recovery": rec}
            print(f"  {cls} fix   {ss:7.1f} steps  {dd:+.2f} m   recovery {rec:+.1%}",
                  flush=True)
        results["heldout"] = res
        results["headroom_heldout"] = hr

        if theta_star is not None:
            ft = found["w6"]
            cosv = float(ft @ theta_star /
                         ((np.linalg.norm(ft) + 1e-12) * np.linalg.norm(theta_star)))
            results["cos_w6_found_vs_analytic"] = cosv
            print(f"\n  cos(found theta, analytic 1/g-1) = {cosv:+.3f}   "
                  f"||theta|| found {np.linalg.norm(ft):.3f} vs star "
                  f"{np.linalg.norm(theta_star):.3f}", flush=True)

        if db0_star is not None:
            fb0 = U @ found["b0"]
            cosv = float(fb0 @ db0_star /
                         ((np.linalg.norm(fb0) + 1e-12) * np.linalg.norm(db0_star)))
            results["cos_b0_found_vs_analytic"] = cosv
            results["alpha_norms"] = {"found": float(np.linalg.norm(found["b0"])),
                                      "star": float(np.linalg.norm(alpha_star))}
            print(f"\n  cos(found db0, analytic db0*) = {cosv:+.3f}   "
                  f"||a|| found {np.linalg.norm(found['b0']):.3f} vs a* "
                  f"{np.linalg.norm(alpha_star):.3f}", flush=True)

        p = OUT / f"b0b6_{a.fault}_{a.mag}.json"
        p.write_text(json.dumps(results, indent=1))
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
