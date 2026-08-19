#!/usr/bin/env python3
"""Which error directions does the BASE POLICY reject, and which persist? (the loop-gain test)

THEORY. Every analysis in this project has treated the residual as acting on an open-loop
plant, `edot = G r + d`. That is wrong: the frozen policy is itself a feedback controller
already regulating tracking error with a 160-input network, so the residual acts on the CLOSED
loop,

    de/dr = (I + G K)^-1 G = S G,     S = closed-loop sensitivity.

`S` is not isotropic. It is SMALL in the directions the base policy regulates hard and ~I in
directions nothing regulates. Three predictions follow, and all three match what was already
measured independently:

  1. a residual aimed at a TRACKED direction is rejected by the base policy's own loop -- not
     merely redundant but fighting a stronger controller. Hence the velocity row being
     ANTI-informative in both directions (too slow under torque limit, too fast downhill);
  2. UNTRACKED directions keep their authority -- hence e_ideal putting +0.245 on height while
     the shipped objective puts -0.002;
  3. the repair is a small INTERIOR CONSTANT: a bias in an unregulated subspace persists, one
     in a regulated subspace is integrated away. Measured r*: |r*| 0.149, strictly interior,
     constant, harmonics gave nothing.

THE TEST. Measure `B_(k,h) = de_(k+h)/dr_k` by central differences for a spread of h. Then per
error row compare the response at horizon h against the open-loop extrapolation of the h=1
response. A row the loop REJECTS decays; a row nothing regulates grows roughly linearly with h,
because an un-rejected constant residual integrates.

    persistence(row, h) = |B_h[row]| / (h * |B_1[row]|)

    ~1  -> unregulated: the effect accumulates, the residual owns this direction
    <<1 -> the base policy is cancelling it; adaptation here fights the policy

This says WHICH rows an objective should be built from, and it is a prediction of the theory
rather than a search over weightings -- the previous 48-cell screen sat at chance precisely
because it reweighted rows without asking which ones the residual can still move.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

from x2_ttcl.adaptation.acc_gain import ACCGain                      # noqa: E402
from x2_ttcl.backends.mjlab_backend import MjlabBackend              # noqa: E402
from x2_ttcl.backends.rom_command import forward_command             # noqa: E402
from x2_ttcl.tasks.velocity.task_error import (TaskErrorConfig,      # noqa: E402
                                               task_error)

OUT = Path(__file__).resolve().parents[1] / "outputs"
ROWS = ["vx", "vy", "wz", "roll", "pitch", "height"]
# What the frozen policy is trained to track, i.e. where loop gain should be HIGH.
TRACKED = {"vx": "commanded", "vy": "commanded", "wz": "commanded",
           "roll": "implicit (balance)", "pitch": "implicit (balance)",
           "height": "NOT tracked"}

CLEAR = {"action_delay_steps": 0, "joint_offset": None, "joint_delay": None,
         "joint_gain": None, "obs_bias": None, "joint_friction": None,
         "payload": None, "action_lag": None, "dof_friction": None,
         "dof_damping_scale": None, "ground_friction": None,
         "inertia_scale": None, "gravity_tilt": None, "ext_force": None,
         "motor_kp_scale": None, "motor_kd_scale": None, "gear_scale": None,
         "torque_limit_scale": None, "armature_scale": None}


def b_at(be, snap, eps, h, n_a, tcfg):
    """B_(k,h) = de_(k+h)/dr_k, central differences, residual HELD for the horizon.

    Held rather than impulsive because that is what an adapted layer produces: a persistent
    state-dependent offset, not a one-step kick. An impulse response would measure a different
    experiment and would not show the loop rejecting a sustained bias.
    """
    B = np.zeros((6, n_a))
    for j in range(n_a):
        u = np.zeros(n_a)
        u[j] = eps
        out = []
        for sgn in (+1.0, -1.0):
            be.restore(snap)
            for _ in range(h):
                be.step(residual=sgn * u)
                if be.fallen():
                    break
            out.append(task_error(be.observe(), tcfg))
        B[:, j] = (out[0] - out[1]) / (2.0 * eps)
    be.restore(snap)
    return B


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", default="tq05")
    ap.add_argument("--vx", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001])
    ap.add_argument("--n-pre", type=int, default=100)
    ap.add_argument("--n-post-shift", type=int, default=40)
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 27, 54])
    ap.add_argument("--eps", type=float, default=0.01)
    a = ap.parse_args()

    tcfg = TaskErrorConfig()
    rows = []

    with forward_command(a.vx):
        be = MjlabBackend(device="cpu", render=False)
        acc = ACCGain(be, fidelity="L3", objective="task", contacts="active")
        n_a = be.num_actions
        from x2_ttcl.experiments.online_adapt import shift_conditions
        cond = shift_conditions(be)[a.shift] if a.shift else {}
        print(f"condition {a.shift}: {cond}\n", flush=True)

        for seed in a.seeds:
            be.reset(seed=seed)
            be.set_command()
            be.set_condition({**CLEAR})
            for _ in range(a.n_pre):
                be.step(residual=None)
            be.set_condition({**CLEAR, **cond})
            for _ in range(a.n_post_shift):
                be.step(residual=None)
                if be.fallen():
                    break
            for i in range(a.n_states):
                for _ in range(a.stride):
                    be.step(residual=None)
                if be.fallen():
                    break
                snap = be.snapshot()
                mode = acc.contact_mode()
                Bs = {h: b_at(be, snap, a.eps, h, n_a, tcfg) for h in a.horizons}
                b1 = np.linalg.norm(Bs[1], axis=1)          # per-row response at h=1
                rec = {"seed": seed, "i": i, "mode": mode, "rows": {}}
                for k, nm in enumerate(ROWS):
                    rec["rows"][nm] = {
                        str(h): {"norm": float(np.linalg.norm(Bs[h][k])),
                                 "persistence": float(np.linalg.norm(Bs[h][k])
                                                      / max(h * b1[k], 1e-12))}
                        for h in a.horizons}
                rows.append(rec)
                print(f"  seed {seed} s{i} [{mode:6s}] " + "  ".join(
                    f"{nm}:{rec['rows'][nm][str(a.horizons[-1])]['persistence']:.2f}"
                    for nm in ROWS), flush=True)
                be.restore(snap)
        be.close()

    if not rows:
        raise SystemExit("no usable states")

    print(f"\n{'=' * 92}")
    print(f"PERSISTENCE  |B_h[row]| / (h |B_1[row]|)   -- {len(rows)} states, {a.shift}")
    print("  ~1 means the effect accumulates (the residual owns this direction)")
    print("  <<1 means the base policy is cancelling it (adaptation here fights the policy)\n")
    print(f"{'row':<9}" + "".join(f"{'h=' + str(h):>9}" for h in a.horizons)
          + "   base policy tracks?")
    print("-" * (9 + 9 * len(a.horizons) + 24))
    persist = {}
    for nm in ROWS:
        cells = [float(np.median([r["rows"][nm][str(h)]["persistence"] for r in rows]))
                 for h in a.horizons]
        persist[nm] = cells
        print(f"{nm:<9}" + "".join(f"{c:>9.2f}" for c in cells)
              + f"   {TRACKED[nm]}")

    hlast = str(a.horizons[-1])
    final = {nm: persist[nm][-1] for nm in ROWS}
    order = sorted(ROWS, key=lambda n: -final[n])
    print(f"\n  Ranked by surviving authority at h={hlast}:")
    for nm in order:
        print(f"    {nm:<9}{final[nm]:>7.2f}   {TRACKED[nm]}")
    print(f"\n  PREDICTED objective: build it from the TOP rows -- those are the directions a")
    print(f"  residual can still move at the horizon the repair acts on. The shipped objective")
    print(f"  weights vx/vy hardest, which this says is where the base policy rejects it.")

    OUT.mkdir(exist_ok=True)
    p = OUT / f"loop_gain_{a.shift}_vx{a.vx}.json"
    p.write_text(json.dumps({"config": vars(a), "states": rows,
                             "persistence": persist}, indent=2, default=str))
    print(f"\nwrote {p.relative_to(OUT.parent)}")
