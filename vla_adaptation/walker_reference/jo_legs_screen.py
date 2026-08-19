"""joint_offset fallback screen: legs-concentrated draws (PREREG_LAYER_CORRESPONDENCE cell-B
criteria). The spread-over-20-joints draws were inert at every interior-fixable magnitude
(headroom +0.05/+0.05/+0.03 m). This screens draws concentrated on the leg joints at
linf {0.06, 0.095} (< u_max = 0.10, so the exact fix r = -c stays interior).
Pass criterion (stated in the prereg): frozen damage >= 1.0 m on 4 seeds.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from x2_ttcl.backends.mjlab_backend import MjlabBackend
from x2_ttcl.backends.rom_command import forward_command

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
MAGS = [0.06, 0.095]


def episode(be, seed, cond_extra, r):
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(N_PRE):
        be.step(residual=None)
    be.set_condition({**CLEAR, **cond_extra})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    rr = None if r is None else np.clip(np.asarray(r, float), -U_MAX, U_MAX)
    for k in range(N_POST):
        x = float(be.observe()["base_pos"][0])
        be.step(residual=rr)
        if be.fallen():
            return k + 1, x - x0
    return N_POST, x - x0


def cell(be, cond, r, tag):
    st, di = zip(*[episode(be, s, cond, r) for s in SEEDS])
    out = {"steps": float(np.mean(st)), "dist": float(np.mean(di)),
           "full": int(sum(s == N_POST for s in st))}
    print(f"    {tag:<22} {out['steps']:7.1f} steps  {out['dist']:+.2f} m   "
          f"full {out['full']}/{len(SEEDS)}", flush=True)
    return out


def main():
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"
    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        names = be.actuated_joint_names()
        legs = [i for i, n in enumerate(names)
                if re.search(r"hip|knee|ankle", n, re.IGNORECASE)]
        print("actuated joints:", names, flush=True)
        print("leg indices:", legs, [names[i] for i in legs], flush=True)

        print("== nominal", flush=True)
        nom = cell(be, {}, None, "nominal")

        rng = np.random.default_rng(13)
        results = {"legs": legs, "nominal": nom, "cells": []}
        for m in MAGS:
            off = np.zeros(be.num_actions)
            off[legs] = rng.normal(0.0, 1.0, len(legs))
            off[legs] *= m / np.max(np.abs(off[legs]))
            print(f"== joint_offset LEGS linf={m} (||c|| {np.linalg.norm(off):.3f})",
                  flush=True)
            frozen = cell(be, {"joint_offset": off}, None, "frozen")
            fix = cell(be, {"joint_offset": off}, -off, "exact b6 fix (-c)")
            hr = nom["dist"] - frozen["dist"]
            rec = (fix["dist"] - frozen["dist"]) / hr if hr > 0.05 else float("nan")
            print(f"    headroom {hr:+.2f} m   recovery {rec:+.1%}", flush=True)
            results["cells"].append({"mag": m, "offset": off.tolist(),
                                     "frozen": frozen, "fix": fix,
                                     "headroom": hr, "recovery": rec})

        p = OUT / "jo_legs_screen.json"
        p.write_text(json.dumps(results, indent=1))
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
