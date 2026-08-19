"""CAN A DEPLOYABLE SEARCH FIND THE FIX? tq05, constant residual, TRUE objective.

LESSONS proves, on `tq05` (torque_limit_scale 0.5), that a bounded CONSTANT residual recovers
+100% of the fault (250/250 steps, +6.45 m, further than nominal), and that the fix is strictly
INTERIOR (||r*|| = 0.149, linf 0.068 < u_max 0.10, 0/20 components pinned). Everything except the
objective is eliminated: gradient certified (cos 0.954), authority +100%, bound non-binding, eta
swept 1600x, conditioning fixed. The surrogate's gradient agrees with the fix 50% of the time --
exactly chance.

So the fix EXISTS, is TINY, is REACHABLE, and the surrogate cannot see it. This asks the only
remaining question:

    can a search that uses ONLY the realised task metric find it, and does it TRANSFER?

Two differences from every previous attempt, and they are the framework's own prescription:
  * dim(theta) = 20 (a constant), not 2,580. The ES learning curve failed at layer scale with
    held-out ~0 at every training-set size; 20 is in the regime where every durable gain in this
    project lives (<= 12 params).
  * the objective is the REALISED METRIC (distance, with survival), never s^T Q s. Condition D is
    satisfied by construction rather than hoped for.

Also unlike `horizon_oracle.py`, this scores WHOLE EPISODES FROM RESET rather than rollouts from a
privileged snapshot, so nothing here needs information a deployed robot lacks -- and it measures
HELD-OUT seeds, which is what layer-scale search failed.

Controls: frozen `off`, and a norm-matched RANDOM constant (rule 6). Reported together.
"""
from __future__ import annotations
import argparse, json
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
COND = {"torque_limit_scale": 0.5}
U_MAX = 0.10

def episode(be, seed, r, n_pre, n_post):
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(n_pre):
        be.step(residual=None)
    be.set_condition({**CLEAR, **COND})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    rr = None if r is None else np.clip(np.asarray(r, float), -U_MAX, U_MAX)
    for k in range(n_post):
        x = float(be.observe()["base_pos"][0])
        be.step(residual=rr)
        if be.fallen():
            return k + 1, x - x0
    return n_post, x - x0

def score(steps, dist, n_post):
    """TRUE metric: forward distance, with survival as the tie-break. Never s^T Q s."""
    return dist + 0.002 * steps          # 0.002 m/step: distance dominates, survival breaks ties

def evaluate(be, r, seeds, n_pre, n_post):
    st, di = zip(*[episode(be, s, r, n_pre, n_post) for s in seeds])
    return (float(np.mean([score(a, b, n_post) for a, b in zip(st, di)])),
            float(np.mean(st)), float(np.mean(di)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs="+", default=[2000, 2001])
    ap.add_argument("--test", type=int, nargs="+", default=[3000,3001,3002,3003,3004,3005])
    ap.add_argument("--n-pre", type=int, default=100)
    ap.add_argument("--n-post", type=int, default=300)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--sigma0", type=float, default=0.04)
    a = ap.parse_args()
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"
    rng = np.random.default_rng(0)

    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)
        n_a = be.num_actions
        base, bs, bd = evaluate(be, None, a.train, a.n_pre, a.n_post)
        print(f"[train] frozen: score {base:+.3f}  steps {bs:.1f}  dist {bd:+.2f}", flush=True)

        mu, sigma = np.zeros(n_a), np.full(n_a, a.sigma0)
        best = (base, np.zeros(n_a))
        for it in range(a.iters):
            pool = np.clip(rng.normal(mu, sigma, size=(a.pop, n_a)), -U_MAX, U_MAX)
            vals = []
            for th in pool:
                v, _, _ = evaluate(be, th, a.train, a.n_pre, a.n_post)
                vals.append(v)
                if v > best[0]:
                    best = (v, th.copy())
            idx = np.argsort(vals)[::-1][:a.elite]
            mu = pool[idx].mean(0); sigma = pool[idx].std(0) + 1e-3
            print(f"  cem it{it+1}/{a.iters}  best {best[0]:+.3f}  "
                  f"elite {np.mean([vals[i] for i in idx]):+.3f}  "
                  f"||mu|| {np.linalg.norm(mu):.3f}", flush=True)

        r = best[1]
        rnorm = float(np.linalg.norm(r))
        rand = rng.normal(size=n_a); rand *= rnorm / (np.linalg.norm(rand) + 1e-12)

        print(f"\nfound constant: ||r|| {rnorm:.3f}  linf {np.abs(r).max():.3f}  "
              f"(oracle: 0.149 / 0.068)")
        print(f"\n{'arm':<16}{'score':>9}{'steps':>9}{'dist':>9}   HELD-OUT seeds {a.test}")
        print("-"*72)
        res = {}
        for name, rr in (("off (frozen)", None), ("CEM constant", r),
                         ("norm-matched rand", rand)):
            v, s_, d_ = evaluate(be, rr, a.test, a.n_pre, a.n_post)
            res[name] = {"score": v, "steps": s_, "dist": d_}
            print(f"{name:<16}{v:>9.3f}{s_:>9.1f}{d_:>+9.2f}")
        o = res["off (frozen)"]
        c = res["CEM constant"]
        print("-"*72)
        print(f"CEM - off : {c['steps']-o['steps']:+.1f} steps, {c['dist']-o['dist']:+.2f} m")
        print(f"CEM - rand: {c['steps']-res['norm-matched rand']['steps']:+.1f} steps, "
              f"{c['dist']-res['norm-matched rand']['dist']:+.2f} m")
        (OUT/"const_adapt.json").write_text(json.dumps(
            {"r": r.tolist(), "norm": rnorm, "train": a.train, "test": a.test,
             "heldout": res}, indent=2))

if __name__ == "__main__":
    main()
