"""Antithetic ACE: rank sites by the FIRST-ORDER response, which is what adaptation uses.

An isotropic ACE_hat(l) = E[M(W+Delta)] - M(W) with symmetric Delta has no first-order
term -- E[grad(M).Delta] = 0 -- so to leading order it measures 0.5*rho^2*tr(H), the
average CURVATURE. Adaptability is a first-order quantity: the best edit in a neighbourhood
gains ~rho*||grad(M)||. The two need not correlate, which is the structural reason an
isotropic screen keeps failing to predict searchability, and why a site can be nearly inert
under random perturbation (action_out_proj/bias: ACE ~ 0) while admitting a 100% repair
along one specific direction.

Antithetic pairs fix the estimand. For direction Delta:

    D = M(W + Delta) - M(W - Delta)  =  2*grad(M).Delta + O(rho^3)

The curvature terms are identical in both arms and cancel exactly. E|D| over random Delta
estimates rho*||grad(M)|| up to a dimensional constant, so mean|D| ranks sites by
first-order sensitivity. Both arms are scored paired against the same deterministic
baseline, so every episode is noise-free.
"""
from __future__ import annotations

import argparse, json, pathlib, time
import numpy as np
from paired_probe import Probe
from ace_screen_v2 import SITES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--matched-c", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8001)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--sites", default=None, help="comma-separated subset")
    p.add_argument("--init-base", type=int, default=8,
                   help="first LIBERO initial-state index (10 = confirmatory held-out set)")
    p.add_argument("--dims", type=int, default=None,
                   help="restrict the perturbation to the first N task dims")
    a = p.parse_args()

    matched = json.loads(a.matched_c.read_text())
    sites = [s for s in SITES if s in matched]
    if a.sites:
        want = [x.strip() for x in a.sites.split(",")]
        sites = [s for s in sites if s in want]
    pr = Probe(a)
    # 10 tasks at initial state 8, then wrap to state 9 -- disjoint from every earlier run
    eps = [((i % 10), a.init_base + i // 10) for i in range(a.episodes)]

    state = {"episodes": a.episodes, "draws": a.draws, "baseline": None, "blocks": []}
    if a.out.exists():
        state = json.loads(a.out.read_text())
        print(f"resuming: {len(state['blocks'])} blocks")

    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(state, indent=1))

    if state["baseline"] is None:
        pr.control(dict(site=None, pin_rng=True))
        state["baseline"] = [bool(pr.rollout(t, i)[0]) for t, i in eps]
        save()
    base = state["baseline"]
    print(f"deterministic baseline {sum(base)}/{len(base)}")

    done = {(b["site"], b["draw"]) for b in state["blocks"]}
    for si, site in enumerate(sites):
        c = matched[site]["c"]
        for d in range(a.draws):
            if (site, d) in done:
                continue
            seed = 9000 + si * 100 + d
            arms = {}
            t0 = time.time()
            for sign in (+1.0, -1.0):
                req = dict(site=site, seed=seed, c_rel=c, sign=sign, pin_rng=True)
                if a.dims:
                    req["dims"] = a.dims
                pr.control(req)
                res = [bool(pr.rollout(t, i)[0]) for t, i in eps]
                arms[sign] = sum(res) / len(res)
            D = arms[+1.0] - arms[-1.0]
            mean = (arms[+1.0] + arms[-1.0]) / 2 - sum(base) / len(base)
            blk = dict(site=site, draw=d, seed=seed, c_rel=c, dims=a.dims,
                       m_plus=arms[+1.0], m_minus=arms[-1.0],
                       D=D, abs_D=abs(D), curvature_term=mean,
                       wall_s=round(time.time() - t0, 1))
            state["blocks"].append(blk)
            save()
            print(f"{site:<38} d{d}  M+ {arms[+1.0]:.3f}  M- {arms[-1.0]:.3f}  "
                  f"D {D:+.3f}  curv {mean:+.3f}  ({blk['wall_s']}s)")
    print("=== ANTITHETIC DONE")


if __name__ == "__main__":
    main()
