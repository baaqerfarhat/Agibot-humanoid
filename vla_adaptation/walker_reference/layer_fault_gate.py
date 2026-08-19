"""LAYER-FAULT CORRESPONDENCE, phase 1: headroom screens + analytic-fix ceilings.

The backend was BUILT for this test and says so (mjlab_backend.py `_shift_obs` docstring):
an `obs_bias` is exactly cancellable by the FIRST layer's bias and not downstream (the error
passes through ELUs); a `joint_offset` is exactly cancellable by the LAST layer's bias (a
constant action residual) and not upstream. "Together they test whether layer selection
tracks where a shift is actually repairable."

This script measures, per fault x magnitude (4 seeds, fault onset at n_pre=100):
    frozen damage        (condition A: is there headroom?)
    analytic-fix ceiling (condition C: is the repair inside the deployed envelope?)
both against the no-fault nominal on the same seeds. The analytic fixes USE the true fault
vector, so they are ceilings/validation, not deployable methods -- the deployable search over
the same coordinates is pre-registered separately and runs only if this gate passes.

Analytic fixes:
    obs_bias Delta on obs slice sl:  delta_b0* = -W0[:, sl] @ (Delta / (obs_std[sl] + eps))
      applied via double-forward action residual r_k = clip(f_{b0+d}(o) - f_{b0}(o), u_max),
      where o is the SHIFTED observation the deployed policy sees (policy_obs()).
    joint_offset c (action units):   r = -c   (constant residual, |c|_inf < u_max = interior)

Gate 0 (plumbing): RomFomActor.from_checkpoint must reproduce the deployed policy's action on
live observations (max |diff| < 1e-4 over 30 steps) BEFORE anything else is believed.
"""
from __future__ import annotations
import copy
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
SEEDS = [3000, 3001, 3002, 3003]
OBS_MAGS = [0.05, 0.10, 0.20]        # rad, sd of per-channel bias on the joint_pos slice
OFF_MAGS = [0.03, 0.05, 0.08]        # action units, scaled to this linf; < u_max so fix interior


def episode(be, seed, cond_extra, res_fn):
    """res_fn: None (frozen) | callable(be) -> residual for the current step."""
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(N_PRE):
        be.step(residual=None)
    be.set_condition({**CLEAR, **cond_extra})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    clipped, norms = 0, []
    for k in range(N_POST):
        x = float(be.observe()["base_pos"][0])
        r = None
        if res_fn is not None:
            r = res_fn(be)
            norms.append(float(np.linalg.norm(r)))
            clipped += int(np.any(np.abs(r) >= U_MAX - 1e-9))
        be.step(residual=r)
        if be.fallen():
            return k + 1, x - x0, clipped, (float(np.mean(norms)) if norms else 0.0)
    return N_POST, x - x0, clipped, (float(np.mean(norms)) if norms else 0.0)


def cell(be, cond_extra, res_fn, tag):
    st, di, cl, nm = zip(*[episode(be, s, cond_extra, res_fn) for s in SEEDS])
    r = {"steps": float(np.mean(st)), "dist": float(np.mean(di)),
         "full": int(sum(s == N_POST for s in st)),
         "clip_steps": float(np.mean(cl)), "mean_res_norm": float(np.mean(nm))}
    print(f"    {tag:<26} {r['steps']:7.1f} steps  {r['dist']:+.2f} m   "
          f"full {r['full']}/{len(SEEDS)}"
          + (f"   ||r|| {r['mean_res_norm']:.3f}  clip@{r['clip_steps']:.0f} steps"
             if res_fn is not None else ""), flush=True)
    return r


def main():
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"
    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        actor = RomFomActor.from_checkpoint().eval()

        # ---- gate 0: the torch rebuild reproduces the deployed policy -------------
        be.reset(seed=1234)
        diffs = []
        for _ in range(30):
            o = be.policy_obs()
            with torch.no_grad():
                a = actor(torch.as_tensor(o)[None])[0].numpy()
            diffs.append(np.max(np.abs(a - be.nominal_action())))
            be.step(residual=None)
        g0 = float(np.max(diffs))
        print(f"GATE 0  max|RomFomActor - deployed policy| over 30 steps = {g0:.2e}  "
              f"{'PASS' if g0 < 1e-4 else 'FAIL -- STOP'}", flush=True)
        assert g0 < 1e-4, "actor rebuild does not match deployed policy"

        # ---- obs layout ------------------------------------------------------------
        om = be.env.unwrapped.observation_manager
        print("actor obs terms:", list(zip(om.active_terms["actor"],
                                           [int(np.prod(d)) for d in
                                            om.group_obs_term_dim["actor"]])), flush=True)
        sl = None
        for cand in ("joint_pos", "joint_pos_rel", "joint_positions"):
            try:
                sl = be.obs_term_slice(cand); jp_term = cand; break
            except KeyError:
                continue
        assert sl is not None, "no joint_pos-like obs term found"
        width = sl.stop - sl.start
        print(f"joint_pos term '{jp_term}' slice {sl.start}:{sl.stop} (width {width})",
              flush=True)

        W0 = actor.mlp[0].weight.detach().numpy()          # 512 x 160
        sig = actor.obs_std.detach().numpy() + actor.eps   # normaliser denominator

        results = {"gate0": g0, "jp_slice": [sl.start, sl.stop], "cells": []}

        # ---- nominal reference -----------------------------------------------------
        print("== nominal (no fault), seeds", SEEDS, flush=True)
        nom = cell(be, {}, None, "nominal")
        results["nominal"] = nom

        # ---- obs_bias cells --------------------------------------------------------
        rng = np.random.default_rng(7)
        for m in OBS_MAGS:
            delta = rng.normal(0.0, m, width)
            db0 = -W0[:, sl.start:sl.stop] @ (delta / sig[sl.start:sl.stop])
            actor_mod = copy.deepcopy(actor)
            with torch.no_grad():
                actor_mod.mlp[0].bias += torch.as_tensor(db0, dtype=torch.float32)

            def res_b0(be_, _am=actor_mod):
                o = torch.as_tensor(be_.policy_obs())[None]
                with torch.no_grad():
                    d = (_am(o) - actor(o))[0].numpy()
                return np.clip(d, -U_MAX, U_MAX)

            cond = {"obs_bias": delta, "obs_bias_slice": (sl.start, sl.stop)}
            print(f"== obs_bias sd={m} on {jp_term} (||Delta|| {np.linalg.norm(delta):.3f})",
                  flush=True)
            frozen = cell(be, cond, None, "frozen")
            fix = cell(be, cond, res_b0, "analytic b0 fix")
            hr = nom["dist"] - frozen["dist"]
            rec = (fix["dist"] - frozen["dist"]) / hr if hr > 0.05 else float("nan")
            print(f"    headroom {hr:+.2f} m   recovery {rec:+.1%}", flush=True)
            results["cells"].append({"fault": "obs_bias", "mag": m,
                                     "delta": delta.tolist(), "frozen": frozen,
                                     "fix": fix, "headroom": hr, "recovery": rec})

        # ---- joint_offset cells ----------------------------------------------------
        rng = np.random.default_rng(11)
        for m in OFF_MAGS:
            off = rng.normal(0.0, 1.0, be.num_actions)
            off *= m / np.max(np.abs(off))                 # linf = m < u_max
            fix_r = np.clip(-off, -U_MAX, U_MAX)

            cond = {"joint_offset": off}
            print(f"== joint_offset linf={m} (||c|| {np.linalg.norm(off):.3f})", flush=True)
            frozen = cell(be, cond, None, "frozen")
            fix = cell(be, cond, (lambda be_, _r=fix_r: _r), "exact b6 fix (-c)")
            hr = nom["dist"] - frozen["dist"]
            rec = (fix["dist"] - frozen["dist"]) / hr if hr > 0.05 else float("nan")
            print(f"    headroom {hr:+.2f} m   recovery {rec:+.1%}", flush=True)
            results["cells"].append({"fault": "joint_offset", "mag": m,
                                     "offset": off.tolist(), "frozen": frozen,
                                     "fix": fix, "headroom": hr, "recovery": rec})

        p = OUT / "layer_fault_gate.json"
        p.write_text(json.dumps(results, indent=1))
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
