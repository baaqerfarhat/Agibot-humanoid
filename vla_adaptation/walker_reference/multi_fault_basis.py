"""MULTI-FAULT BASIS: what rank does the fault space actually need?

`const_adapt.py` confirmed on `tq05` that CEM over a constant 20-d residual, scored on the REALISED
metric, finds essentially the horizon oracle's fix (||r|| 0.141 vs 0.149) and transfers to held-out
seeds (+62.3 steps, +1.70 m, beating a norm-matched random by +2.62 m).

That is one fault. The LoRA-style question (LORA_ONLINE_DESIGN.md) is whether a LOW-RANK basis
spans the corrections for MANY faults, which decides whether an online adapter can carry K << 20
coefficients instead of a fresh 20-vector per fault.

Per fault this records:
    nominal   no-fault reference on the SAME seeds  -> recovery becomes a FRACTION, not a raw delta
    off       fault, frozen policy
    r_f*      the CEM constant, from realised returns only
    held-out  r_f* and a norm-matched random control, on disjoint seeds

Then, once the vectors exist:
    SVD over {r_f*}                 -> how many directions carry the energy
    leave-one-fault-out projection  -> does the span of the OTHERS contain a held-out fix
    replay of the projected fix     -> realised recovery, not merely cosine

Fault set chosen for measured headroom, plus one deliberate negative control:
    tq05, tq04, lag08, dofric6   -- headroom established
    s2r_moderate                 -- the conjunction cell (250 -> 92), the sim-to-real headline
    kp05                         -- NEGATIVE CONTROL: the horizon oracle says only +16% is
                                    available, so a null there is uninformative and must not be
                                    reported as a failure of the method
"""
from __future__ import annotations

import argparse
import json
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

FAULTS = [
    ("tq05",         {"torque_limit_scale": 0.5}),
    ("tq04",         {"torque_limit_scale": 0.4}),
    ("lag08",        {"action_lag": 0.8}),
    ("dofric6",      {"dof_friction": 6.0, "fault_joints": "legs"}),
    ("s2r_moderate", {"armature_scale": 3.0, "inertia_scale": 1.35, "dof_friction": 0.15,
                      "dof_damping_scale": 3.0, "gear_scale": 0.85, "action_lag": 0.8,
                      "action_delay_steps": 1}),
    ("kp05",         {"motor_kp_scale": 0.5, "fault_joints": "legs"}),   # negative control
]


def episode(be, seed, cond, r, n_pre, n_post):
    be.reset(seed=seed)
    be.set_condition({**CLEAR})
    for _ in range(n_pre):
        be.step(residual=None)
    be.set_condition({**CLEAR, **cond})
    x0 = float(be.observe()["base_pos"][0])
    x = x0
    rr = None if r is None else np.clip(np.asarray(r, float), -U_MAX, U_MAX)
    for k in range(n_post):
        x = float(be.observe()["base_pos"][0])
        be.step(residual=rr)
        if be.fallen():
            return k + 1, x - x0
    return n_post, x - x0


def score(s, d):
    """TRUE metric: distance dominates, survival breaks ties. Never s^T Q s."""
    return d + 0.002 * s


def ev(be, cond, r, seeds, n_pre, n_post):
    st, di = zip(*[episode(be, s, cond, r, n_pre, n_post) for s in seeds])
    return (float(np.mean([score(a, b) for a, b in zip(st, di)])),
            float(np.mean(st)), float(np.mean(di)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--test", type=int, nargs="+", default=[3000, 3001, 3002, 3003, 3004, 3005])
    ap.add_argument("--n-pre", type=int, default=100)
    ap.add_argument("--n-post", type=int, default=300)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--sigma0", type=float, default=0.04)
    a = ap.parse_args()

    import x2_ttcl
    assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"

    rows = []
    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        n_a = be.num_actions
        for name, cond in FAULTS:
            rng = np.random.default_rng(0)
            print(f"\n{'=' * 82}\n{name}\n{'=' * 82}", flush=True)
            nom = ev(be, {}, None, a.test, a.n_pre, a.n_post)
            off = ev(be, cond, None, a.test, a.n_pre, a.n_post)
            head = nom[2] - off[2]
            print(f"  nominal {nom[1]:6.1f} steps {nom[2]:+6.2f} m | "
                  f"off {off[1]:6.1f} {off[2]:+6.2f} m | headroom {head:+.2f} m", flush=True)

            base, _, _ = ev(be, cond, None, a.train, a.n_pre, a.n_post)
            mu, sigma = np.zeros(n_a), np.full(n_a, a.sigma0)
            best = (base, np.zeros(n_a))
            for it in range(a.iters):
                pool = np.clip(rng.normal(mu, sigma, size=(a.pop, n_a)), -U_MAX, U_MAX)
                vals = []
                for th in pool:
                    v, _, _ = ev(be, cond, th, a.train, a.n_pre, a.n_post)
                    vals.append(v)
                    if v > best[0]:
                        best = (v, th.copy())
                idx = np.argsort(vals)[::-1][:a.elite]
                mu = pool[idx].mean(0)
                sigma = pool[idx].std(0) + 1e-3
                print(f"    it{it + 1}/{a.iters} best {best[0]:+.3f} "
                      f"||mu|| {np.linalg.norm(mu):.3f}", flush=True)

            r = best[1]
            rn = float(np.linalg.norm(r))
            rand = rng.normal(size=n_a)
            rand *= rn / (np.linalg.norm(rand) + 1e-12)
            cem = ev(be, cond, r, a.test, a.n_pre, a.n_post)
            rnd = ev(be, cond, rand, a.test, a.n_pre, a.n_post)
            rec = (cem[2] - off[2]) / head if abs(head) > 1e-6 else float("nan")
            print(f"  CEM {cem[1]:6.1f} {cem[2]:+6.2f} m | rand {rnd[1]:6.1f} {rnd[2]:+6.2f} m",
                  flush=True)
            print(f"  ** CEM-off {cem[2] - off[2]:+.2f} m   CEM-rand {cem[2] - rnd[2]:+.2f} m   "
                  f"RECOVERY {100 * rec:5.1f}% of headroom **", flush=True)
            rows.append({"fault": name, "cond": cond, "r": r.tolist(), "norm": rn,
                         "nominal": nom, "off": off, "cem": cem, "rand": rnd,
                         "headroom_dist": head, "recovery_frac": rec})
            (OUT / "multi_fault_basis.json").write_text(json.dumps(rows, indent=2))

    print(f"\n\n{'=' * 82}\nSUMMARY\n{'=' * 82}")
    print(f"{'fault':<14}{'headroom':>10}{'CEM-off':>10}{'recovery':>10}"
          f"{'CEM-rand':>10}{'||r||':>8}")
    print("-" * 82)
    for q in rows:
        print(f"{q['fault']:<14}{q['headroom_dist']:>+10.2f}"
              f"{q['cem'][2] - q['off'][2]:>+10.2f}{100 * q['recovery_frac']:>9.1f}%"
              f"{q['cem'][2] - q['rand'][2]:>+10.2f}{q['norm']:>8.3f}")

    R = np.array([q["r"] for q in rows])
    s = np.linalg.svd(R, compute_uv=False)
    print(f"\nSVD over the {len(rows)} correction vectors")
    print("  singular values :", np.round(s, 4))
    print("  cumulative energy:", np.round(np.cumsum(s ** 2) / np.sum(s ** 2), 3))
    C = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    G = C @ C.T
    print("  pairwise cosines:")
    for i, q in enumerate(rows):
        print(f"    {q['fault']:<14}" + " ".join(f"{c:+.2f}" for c in G[i]))
    print("\nHigh cumulative energy at small rank => a fixed low-rank basis can carry these faults")
    print("and an online adapter needs only K coefficients. Flat spectrum => each fault needs its")
    print("own direction and LoRA-style adaptation is structurally dead here.")


if __name__ == "__main__":
    main()
