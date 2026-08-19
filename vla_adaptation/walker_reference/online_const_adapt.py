"""ONLINE across-episode adaptation of the mlp.6 bias (constant residual) on tq05.

Pre-registered in theory_2026-08-15/PREREG_ONLINE_TQ05.md BEFORE this run. Differences from
const_adapt.py, and they are the entire point:

  * every episode uses a FRESH seed in a fixed deployment order (5000+gen); no seed is ever
    replayed, so nothing here uses information a deployed robot lacks;
  * all candidates of one generation share that generation's single seed (common random
    numbers WITHIN a generation only), so selection compares candidates under identical
    conditions while remaining honest across generations;
  * the settled estimate for EVERY arm is the elite-mean (top-3) of the final generation --
    the same estimator for treatment and control, so the only difference between
    `cem_online` and `random_search` is whether the sampling distribution was refit.

Arms: cem_online (refit), random_search (never refit; rule-11 control), frozen.
Held-out: settled thetas + frozen on seeds 3000-3005 (disjoint).
"""
from __future__ import annotations
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
COND = {"torque_limit_scale": 0.5}
U_MAX = 0.10
N_PRE, N_POST = 100, 300
POP, ITERS, ELITE, SIGMA0 = 10, 6, 3, 0.04
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--seeds-per-gen", type=int, default=1)
_ap.add_argument("--seed-base", type=int, default=5000)
_ap.add_argument("--tag", type=str, default="")
ARGS = _ap.parse_args()
# V1: one fresh seed per generation (seeds-per-gen 1, base 5000).
# V2 (PREREG_ONLINE_TQ05_V2): each candidate scored on the MEAN of 2 fresh seeds.
GEN_SEEDS = [[ARGS.seed_base + g * ARGS.seeds_per_gen + j
              for j in range(ARGS.seeds_per_gen)] for g in range(ITERS)]
HELDOUT = [3000, 3001, 3002, 3003, 3004, 3005]


def episode(be, seed, r):
    be.reset(seed=seed); be.set_condition({**CLEAR})
    for _ in range(N_PRE):
        be.step(residual=None)
    be.set_condition({**CLEAR, **COND})
    x0 = float(be.observe()["base_pos"][0]); x = x0
    rr = None if r is None else np.clip(np.asarray(r, float), -U_MAX, U_MAX)
    for k in range(N_POST):
        x = float(be.observe()["base_pos"][0])
        be.step(residual=rr)
        if be.fallen():
            return k + 1, x - x0
    return N_POST, x - x0


def score(steps, dist):
    return dist + 0.002 * steps


def run_arm(be, refit: bool, rng: np.random.Generator):
    """One online arm: 6 generations x 10 candidates, one fresh seed per generation."""
    mu, sigma = np.zeros(20), np.full(20, SIGMA0)
    history, best = [], (-1e9, np.zeros(20))
    for gen, seeds in enumerate(GEN_SEEDS):
        pool = np.clip(rng.normal(mu, sigma, size=(POP, 20)), -U_MAX, U_MAX)
        vals = []
        for th in pool:
            evs = [episode(be, s, th) for s in seeds]
            v = float(np.mean([score(st, di) for st, di in evs]))
            vals.append(v)
            if v > best[0]:
                best = (v, th.copy())
        idx = np.argsort(vals)[::-1][:ELITE]
        settled = pool[idx].mean(0)                    # elite-mean, same estimator every arm
        if refit:
            mu, sigma = settled, pool[idx].std(0) + 1e-3
        history.append({"gen": gen, "seeds": seeds, "mean": float(np.mean(vals)),
                        "elite": float(np.mean([vals[i] for i in idx])),
                        "norm_settled": float(np.linalg.norm(settled))})
        print(f"    gen {gen+1}/{ITERS} seeds {seeds}  mean {np.mean(vals):+.3f}  "
              f"elite {history[-1]['elite']:+.3f}  ||settled|| {history[-1]['norm_settled']:.3f}",
              flush=True)
    return settled, best, history


def evaluate(be, r, seeds):
    st, di = zip(*[episode(be, s, r) for s in seeds])
    return float(np.mean(st)), float(np.mean(di))


def main():
    import x2_ttcl; assert "theory_ws" in x2_ttcl.__file__, "WRONG TREE"
    with forward_command(1.0):
        be = MjlabBackend(device="cpu", render=False)

        flat_seeds = [s for grp in GEN_SEEDS for s in grp]
        print("== frozen on the deployment stream", flush=True)
        froz_stream = [episode(be, s, None) for s in flat_seeds]
        for s, (st, di) in zip(flat_seeds, froz_stream):
            print(f"    seed {s}: {st} steps  {di:+.2f} m", flush=True)

        print("== ARM cem_online (refit each generation)", flush=True)
        th_cem, best_cem, hist_cem = run_arm(be, refit=True,
                                             rng=np.random.default_rng(0))
        print("== ARM random_search (never refit; rule-11 control)", flush=True)
        th_rnd, best_rnd, hist_rnd = run_arm(be, refit=False,
                                             rng=np.random.default_rng(1))

        print("== HELD-OUT, seeds", HELDOUT, flush=True)
        res = {}
        for name, th in (("frozen", None), ("cem_online", th_cem),
                         ("random_search", th_rnd),
                         ("cem_best_ever", best_cem[1]),
                         ("random_best_ever", best_rnd[1])):
            st, di = evaluate(be, th, HELDOUT)
            res[name] = {"steps": st, "dist": di,
                         "norm": None if th is None else float(np.linalg.norm(th))}
            print(f"  {name:<18} {st:7.1f} steps  {di:+.2f} m"
                  + ("" if th is None else f"   ||r|| {np.linalg.norm(th):.3f}"), flush=True)

        d = res["cem_online"]["dist"] - res["frozen"]["dist"]
        print(f"\nPRIMARY  cem_online settled - frozen = {d:+.2f} m  "
              f"({'PASS' if d > 1.0 else 'FAIL'} at +1.0 m)", flush=True)
        print(f"SECONDARY cem settled - random settled = "
              f"{res['cem_online']['dist'] - res['random_search']['dist']:+.2f} m", flush=True)

        out = {"config": {"cond": COND, "u_max": U_MAX, "n_pre": N_PRE, "n_post": N_POST,
                          "pop": POP, "iters": ITERS, "elite": ELITE, "sigma0": SIGMA0,
                          "gen_seeds": GEN_SEEDS, "heldout": HELDOUT,
                          "seeds_per_gen": ARGS.seeds_per_gen, "tag": ARGS.tag,
                          "estimator": "elite_mean_final_gen"},
               "frozen_stream": froz_stream,
               "cem_online": {"settled": th_cem.tolist(), "history": hist_cem,
                              "best_ever_score": best_cem[0]},
               "random_search": {"settled": th_rnd.tolist(), "history": hist_rnd,
                                 "best_ever_score": best_rnd[0]},
               "heldout_results": res}
        p = OUT / f"online_const_tq05{('_' + ARGS.tag) if ARGS.tag else ''}.json"
        p.write_text(json.dumps(out, indent=1))
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
