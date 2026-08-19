"""joint_gain gate screen (fault C for PREREG_LAYER_CORRESPONDENCE follow-up).

A multiplicative actuator-gain fault has a STATE-DEPENDENT inverse (a' = a/g), which is the
regime the 2x2 falsification licensed: the backend comment (mjlab_backend.py ~line 316)
predicts it is "exactly compensable by rescaling the LAST layer's weights by 1/alpha, and
not by any input-side change."

Screens legs-only gain g in {0.7, 0.5}: nominal / frozen / clipped multiplicative oracle
r_k = clip((1/g - 1) * a_nominal(o_k), u_max) -- the exact inverse pushed through the
deployed residual envelope. Actions are large, so the clip is expected to bind hard; the
oracle number IS the envelope-limited ceiling of the w6-scale class, measured before any
search is run.
"""
from __future__ import annotations
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
U_MAX = 0.10
N_PRE, N_POST = 100, 300
SEEDS = [3000, 3001, 3002, 3003]
GAINS = [0.7, 0.5]


def episode(be, seed, cond_extra, res_fn):
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(N_PRE):
        be.step(residual=None)
    be.set_condition({**CLEAR, **cond_extra})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    clip_hits, norms = 0, []
    for k in range(N_POST):
        x = float(be.observe()["base_pos"][0])
        r = None
        if res_fn is not None:
            r = res_fn(be)
            norms.append(float(np.linalg.norm(r)))
            clip_hits += int(np.any(np.abs(r) >= U_MAX - 1e-9))
        be.step(residual=r)
        if be.fallen():
            return k + 1, x - x0, clip_hits, (float(np.mean(norms)) if norms else 0.0)
    return N_POST, x - x0, clip_hits, (float(np.mean(norms)) if norms else 0.0)


def cell(be, cond, res_fn, tag):
    st, di, cl, nm = zip(*[episode(be, s, cond, res_fn) for s in SEEDS])
    out = {"steps": float(np.mean(st)), "dist": float(np.mean(di)),
           "full": int(sum(s == N_POST for s in st)),
           "clip_steps": float(np.mean(cl)), "mean_res_norm": float(np.mean(nm))}
    print(f"    {tag:<26} {out['steps']:7.1f} steps  {out['dist']:+.2f} m   "
          f"full {out['full']}/{len(SEEDS)}"
          + (f"   ||r|| {out['mean_res_norm']:.3f}  clip@{out['clip_steps']:.0f}"
             if res_fn is not None else ""), flush=True)
    return out


def main():
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"
    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        actor = RomFomActor.from_checkpoint().eval()
        names = be.actuated_joint_names()
        legs = [i for i, n in enumerate(names)
                if re.search(r"hip|knee|ankle", n, re.IGNORECASE)]
        print("leg indices:", legs, flush=True)

        print("== nominal", flush=True)
        nom = cell(be, {}, None, "nominal")
        results = {"legs": legs, "nominal": nom, "cells": []}

        for g in GAINS:
            gain = np.ones(be.num_actions)
            gain[legs] = g
            inv = (1.0 / gain) - 1.0            # exact multiplicative inverse coefficients

            def res_oracle(be_, _inv=inv):
                o = torch.as_tensor(be_.policy_obs())[None]
                with torch.no_grad():
                    a = actor(o)[0].numpy()
                return np.clip(_inv * a, -U_MAX, U_MAX)

            cond = {"joint_gain": gain}
            print(f"== joint_gain LEGS g={g}", flush=True)
            frozen = cell(be, cond, None, "frozen")
            fix = cell(be, cond, res_oracle, "clipped mult. oracle")
            hr = nom["dist"] - frozen["dist"]
            rec = (fix["dist"] - frozen["dist"]) / hr if hr > 0.05 else float("nan")
            print(f"    headroom {hr:+.2f} m   envelope-limited ceiling {rec:+.1%}",
                  flush=True)
            results["cells"].append({"fault": "joint_gain", "mag": g,
                                     "gain": gain.tolist(), "frozen": frozen,
                                     "fix": fix, "headroom": hr, "recovery": rec})

        p = OUT / "jg_gate.json"
        p.write_text(json.dumps(results, indent=1))
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
