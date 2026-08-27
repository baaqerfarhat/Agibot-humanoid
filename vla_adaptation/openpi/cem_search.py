"""Phase 0.5 -- does a gradient-free search actually recover the ceiling?

The oracle established that the fault is fully repairable: a computed 6-vector on
action_out_proj/bias takes the faulted policy from 46.7% to 100%. That edit was CALCULATED.
This asks the question the method is actually about: starting from nothing, does across-
episode CEM on the realised task metric find it?

Two things this run does differently from the naive version, both measured rather than
assumed:

  ENV-ACTION UNITS. The search vector v is a correction in env action units, mapped to the
  bias by beta_i = v_i / (scale_i * 0.34). In env units the target is isotropic (+0.05 on all
  six arm dims); in raw bias units the same target spans 7.3x, and an isotropic CEM is then
  ill-conditioned exactly where the fault is largest.

  SIGMA FROM THE MEASURED BASIN. 1.5x the correct edit scores 20% -- worse than no repair --
  so the useful range is roughly k in [0.4, 1.2]. sigma0 = 0.15 * v_oracle keeps the initial
  population inside it instead of spending the budget below its own starting point.

Arms: cem_online (the method), never_refit (selection without learning -- same sampling, the
mean never updates), and frozen_faulted. Settled estimate for each adaptive arm is the
elite-mean of the final generation, evaluated on held-out initial states no generation saw.
"""
from __future__ import annotations

import argparse, json, pathlib, time
import numpy as np

from paired_probe import Probe
from reachability_score import NORM

ATTEN, FAULT_ENV, ARM = 0.34, 0.05, 6


def scales():
    n = json.load(open(NORM))
    n = n.get("norm_stats", n)["actions"]
    return (np.array(n["q99"][:7]) - np.array(n["q01"][:7])) / 2.0


class Search:
    def __init__(self, a):
        self.a = a
        self.pr = Probe(a)
        self.sc = scales()
        self.v_oracle = np.full(ARM, FAULT_ENV)          # the known solution, for sigma only
        self.n_ep = 0

    def to_bias(self, v):
        """env-unit correction -> bias vector (7 dims; the gripper channel is untouched)."""
        b = np.zeros(7)
        b[:ARM] = np.asarray(v)[:ARM] / (self.sc[:ARM] * ATTEN)
        return b

    def evaluate(self, v, eps):
        self.pr.control(dict(bias_add=list(map(float, self.to_bias(v)))))
        s = sum(self.pr.rollout(t, i)[0] for t, i in eps)
        self.n_ep += len(eps)
        return s / len(eps)

    def episodes(self, gen, k):
        """Fresh (task, initial-state) pairs every generation -- no replay, ever."""
        base = self.a.init_base + gen * self.a.per_cand
        return [((gen * 7 + k * 3 + j) % 10, base + j) for j in range(self.a.per_cand)]


def run_arm(S, a, refit: bool, tag: str, log: list):
    mean = np.zeros(ARM)
    sigma = np.full(ARM, a.sigma0 * FAULT_ENV)
    settled = mean.copy()
    for g in range(a.gens):
        eps = S.episodes(g, 0)                     # common random numbers within a generation
        pop = [np.random.default_rng(1000 * g + i + (0 if refit else 500)).normal(mean, sigma)
               for i in range(a.pop)]
        sc = [S.evaluate(v, eps) for v in pop]
        elite = [pop[i] for i in np.argsort(sc)[-a.elites:]]
        settled = np.mean(elite, axis=0)
        if refit:
            mean = settled
            sigma = np.maximum(np.std(elite, axis=0), 0.2 * a.sigma0 * FAULT_ENV)
        log.append(dict(arm=tag, gen=g, scores=sc, best=float(max(sc)),
                        mean_score=float(np.mean(sc)), settled=settled.tolist(),
                        sigma=sigma.tolist(), episodes=S.n_ep))
        print(f"  [{tag}] gen {g}  best {max(sc):.2f}  mean {np.mean(sc):.2f}  "
              f"settled/oracle {np.mean(settled)/FAULT_ENV:+.2f}  (n={S.n_ep})")
    return settled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--pop", type=int, default=10)
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--elites", type=int, default=3)
    p.add_argument("--per-cand", type=int, default=2)
    p.add_argument("--sigma0", type=float, default=0.15)
    p.add_argument("--init-base", type=int, default=20)
    p.add_argument("--holdout-base", type=int, default=38)
    p.add_argument("--holdout-n", type=int, default=20)
    a = p.parse_args()

    S = Search(a)
    out = {"log": [], "settled": {}, "holdout": {}}
    if a.out.exists():
        out = json.loads(a.out.read_text())

    for tag, refit in (("cem_online", True), ("never_refit", False)):
        if tag in out["settled"]:
            print(f"{tag}: already done"); continue
        print(f"=== {tag}")
        v = run_arm(S, a, refit, tag, out["log"])
        out["settled"][tag] = v.tolist()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))

    hold = [((i % 10), a.holdout_base + i // 10) for i in range(a.holdout_n)]
    for tag, v in (("frozen_faulted", np.zeros(ARM)),
                   ("cem_online", np.array(out["settled"]["cem_online"])),
                   ("never_refit", np.array(out["settled"]["never_refit"])),
                   ("oracle", np.full(ARM, FAULT_ENV))):
        if tag in out["holdout"]:
            continue
        t0 = time.time()
        r = S.evaluate(v, hold)
        out["holdout"][tag] = r
        a.out.write_text(json.dumps(out, indent=1))
        print(f"HELD-OUT {tag:<16} {r*100:5.1f}%  ({time.time()-t0:.0f}s)")
    f = out["holdout"]["frozen_faulted"]
    print(f"\nPRIMARY: cem_online - frozen = {100*(out['holdout']['cem_online']-f):+.1f} pts "
          f"(needs > +15) ; vs never_refit {100*(out['holdout']['cem_online']-out['holdout']['never_refit']):+.1f} pts (needs > 0)")
    print("=== CEM SEARCH DONE")


if __name__ == "__main__":
    main()
