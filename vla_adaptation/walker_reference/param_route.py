"""PARAMETER-ROUTE class-shape test on joint_gain (the experiment the envelope blocked).

Under the deployed residual contract (clip u_max=0.10) the fault-C search was killed at the
gate: the multiplicative inverse needs ||r|| ~0.33, 3.4x the cap. This experiment CHANGES
THE BOUND CONTRACT — direct single-layer parameter edits, expressed exactly through the
residual channel WITHOUT the clip:

    r_k = f_edited(o_k) - f_nominal(o_k)        (unclipped)

Three dim-20 classes, all literal one-layer edits of the deployed actor:
    w6  : per-joint output scale, a' = (1+theta) . a   == row-rescale of W6 AND b6
    b6  : constant action shift               == b6 += theta
    b0  : obs-channel first-layer bias        == b0 += U alpha
Exact inverse of joint_gain g: theta* = 1/g - 1 on the faulted joints (w6 class).

--phase gate   : nominal / frozen / exact-parameter-oracle on g in {0.5, 0.7} (fast)
--phase search : 3-class CEM + ACE arm + held-out, on the cell fixed by the prereg

Rule-6 analog under the new contract: per arm, report mean ||r_k|| (induced action
displacement) — classes are compared at matched dim and budget, displacement REPORTED.
"""
from __future__ import annotations
import argparse
import json
import re
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
N_PRE, N_POST = 100, 300
POP, ITERS, ELITE = 10, 8, 3
SIGMA0 = {"w6": 0.30, "b6": 0.15, "b0": 0.30}
TRAIN = [2000, 2001]
HELDOUT = [3000, 3001, 3002, 3003, 3004, 3005]
ACE_DRAWS = 8


def episode(be, seed, cond_extra, res_fn):
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(N_PRE):
        be.step(residual=None)
    be.set_condition({**CLEAR, **cond_extra})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    norms = []
    for k in range(N_POST):
        x = float(be.observe()["base_pos"][0])
        r = None
        if res_fn is not None:
            r = res_fn(be)
            norms.append(float(np.linalg.norm(r)))
        be.step(residual=r)
        if be.fallen():
            return k + 1, x - x0, (float(np.mean(norms)) if norms else 0.0)
    return N_POST, x - x0, (float(np.mean(norms)) if norms else 0.0)


def score(steps, dist):
    return dist + 0.002 * steps


def evaluate(be, cond, res_fn, seeds):
    st, di, nm = zip(*[episode(be, s, cond, res_fn) for s in seeds])
    return (float(np.mean([score(a, b) for a, b in zip(st, di)])),
            float(np.mean(st)), float(np.mean(di)), float(np.mean(nm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gate", "search"], required=True)
    ap.add_argument("--gain", type=float, default=0.5)
    a = ap.parse_args()
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"

    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        actor = RomFomActor.from_checkpoint().eval()
        names = be.actuated_joint_names()
        legs = [i for i, n in enumerate(names)
                if re.search(r"hip|knee|ankle", n, re.IGNORECASE)]
        gate = json.loads((OUT / "layer_fault_gate.json").read_text())
        s0, s1 = gate["jp_slice"]
        W0 = actor.mlp[0].weight.detach().numpy()
        sig = actor.obs_std.detach().numpy() + actor.eps
        U = np.stack([-W0[:, s0 + i] / sig[s0 + i] for i in range(s1 - s0)], axis=1)
        actor_mod = RomFomActor.from_checkpoint().eval()
        b0_base = actor.mlp[0].bias.detach().clone()

        def anom(be_):
            o = torch.as_tensor(be_.policy_obs())[None]
            with torch.no_grad():
                return actor(o)[0].numpy(), o

        def res_w6(theta):
            th = np.asarray(theta, float)
            def fn(be_):
                an, _ = anom(be_)
                return th * an                       # UNCLIPPED
            return fn

        def res_b6(theta):
            th = np.asarray(theta, float)
            return lambda be_: th                    # UNCLIPPED

        def res_b0(alpha):
            db0 = torch.as_tensor(U @ np.asarray(alpha, float), dtype=torch.float32)
            def fn(be_):
                an, o = anom(be_)
                with torch.no_grad():
                    actor_mod.mlp[0].bias.copy_(b0_base + db0)
                    return (actor_mod(o)[0].numpy() - an)   # UNCLIPPED
            return fn

        make = {"w6": res_w6, "b6": res_b6, "b0": res_b0}

        if a.phase == "gate":
            print("== nominal", flush=True)
            _, ns, nd, _ = evaluate(be, {}, None, HELDOUT[:4])
            print(f"    nominal   {ns:7.1f} steps  {nd:+.2f} m", flush=True)
            results = {"nominal": [ns, nd], "cells": []}
            for g in (0.5, 0.7):
                gain = np.ones(be.num_actions); gain[legs] = g
                theta_star = (1.0 / gain) - 1.0
                cond = {"joint_gain": gain}
                _, fs, fd, _ = evaluate(be, cond, None, HELDOUT[:4])
                _, os_, od, on = evaluate(be, cond, make["w6"](theta_star), HELDOUT[:4])
                hr = nd - fd
                rec = (od - fd) / hr if hr > 0.05 else float("nan")
                print(f"  g={g}: frozen {fs:.1f}/{fd:+.2f}  param-oracle {os_:.1f}/{od:+.2f}"
                      f" (||r|| {on:.3f})  headroom {hr:+.2f}  recovery {rec:+.1%}",
                      flush=True)
                results["cells"].append({"g": g, "gain": gain.tolist(),
                                         "frozen": [fs, fd], "oracle": [os_, od, on],
                                         "headroom": hr, "recovery": rec})
            (OUT / "param_gate.json").write_text(json.dumps(results, indent=1))
            print("wrote", OUT / "param_gate.json", flush=True)
            return

        # ---- search phase ----------------------------------------------------------
        gain = np.ones(be.num_actions); gain[legs] = a.gain
        theta_star = (1.0 / gain) - 1.0
        cond = {"joint_gain": gain}
        base, bs, bd, _ = evaluate(be, cond, None, TRAIN)
        print(f"[train] frozen: score {base:+.3f}  steps {bs:.1f}  dist {bd:+.2f}",
              flush=True)
        results = {"gain": a.gain, "config": {"pop": POP, "iters": ITERS, "elite": ELITE,
                   "sigma0": SIGMA0, "train": TRAIN, "heldout": HELDOUT,
                   "contract": "parameter route, UNCLIPPED",
                   "settled": "best_ever_train"}}
        found = {}
        for cls in ("w6", "b6", "b0"):
            rng = np.random.default_rng({"w6": 0, "b6": 1, "b0": 2}[cls])
            mu, sg = np.zeros(20), np.full(20, SIGMA0[cls])
            best = (base, np.zeros(20))
            print(f"== SEARCH {cls} (sigma0 {SIGMA0[cls]})", flush=True)
            for it in range(ITERS):
                pool = rng.normal(mu, sg, size=(POP, 20))
                vals = []
                for th in pool:
                    v, _, _, _ = evaluate(be, cond, make[cls](th), TRAIN)
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

        print("== ACE arm", flush=True)
        ace = {}
        for cls in ("w6", "b6", "b0"):
            rng = np.random.default_rng({"w6": 10, "b6": 11, "b0": 12}[cls])
            effs = []
            target = SIGMA0[cls] * np.sqrt(20)
            for _ in range(ACE_DRAWS):
                th = rng.normal(0.0, 1.0, 20); th *= target / np.linalg.norm(th)
                v, _, _, _ = evaluate(be, cond, make[cls](th), TRAIN)
                effs.append(v - base)
            ace[cls] = {"mean": float(np.mean(effs)), "sd": float(np.std(effs))}
            print(f"    ACE_hat[{cls}] = {np.mean(effs):+.3f}  (sd {np.std(effs):.3f})",
                  flush=True)
        results["ace"] = ace

        print("== HELD-OUT, seeds", HELDOUT, flush=True)
        _, ns, nd, _ = evaluate(be, {}, None, HELDOUT)
        _, fs, fd, _ = evaluate(be, cond, None, HELDOUT)
        print(f"  nominal  {ns:7.1f} steps  {nd:+.2f} m", flush=True)
        print(f"  frozen   {fs:7.1f} steps  {fd:+.2f} m", flush=True)
        hr = nd - fd
        res = {"nominal": [ns, nd], "frozen": [fs, fd]}
        for cls in ("w6", "b6", "b0"):
            _, ss, dd, nn = evaluate(be, cond, make[cls](found[cls]), HELDOUT)
            rec = (dd - fd) / hr if hr > 0.05 else float("nan")
            res[cls] = {"steps": ss, "dist": dd, "recovery": rec, "mean_dr": nn}
            print(f"  {cls} fix   {ss:7.1f} steps  {dd:+.2f} m   recovery {rec:+.1%}"
                  f"   mean||r|| {nn:.3f}", flush=True)
        results["heldout"] = res
        results["headroom_heldout"] = hr

        ft = found["w6"]
        cosv = float(ft @ theta_star /
                     ((np.linalg.norm(ft) + 1e-12) * np.linalg.norm(theta_star)))
        results["cos_w6_found_vs_analytic"] = cosv
        print(f"\n  cos(found theta, 1/g-1) = {cosv:+.3f}   ||theta|| found "
              f"{np.linalg.norm(ft):.3f} vs star {np.linalg.norm(theta_star):.3f}",
              flush=True)

        p = OUT / f"param_route_g{a.gain}.json"
        p.write_text(json.dumps(results, indent=1))
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
