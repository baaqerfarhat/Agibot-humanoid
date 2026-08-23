"""ACE screen v2 — same estimand, measured with common random numbers.

The pre-registered screen returned p = 0.974, eta^2 = 0.036. Two causes were measured
rather than guessed:

  scale  at c = 0.02 a perturbation flips 0 of 6 outcomes. Trajectories diverge across the
         full action range and the policy re-converges to the same result. There was
         nothing at that scale to detect.
  noise  draws were scored UNPAIRED, and 96% of a 5-episode success rate is sampler noise.

v2 changes only the measurement, not the estimand. ACE_hat is still
E[M | do(W+Delta)] - E[M | W]; it is now estimated as a PAIRED difference with the sampler
RNG pinned, so an episode is deterministic (verified: repeating one gives max|dAction| =
0.0 exactly) and baseline and perturbed runs differ only by the weights. One deterministic
baseline serves every site, and each paired episode contributes a clean -1 / 0 / +1.

Scale is swept rather than assumed, and reported as a scale-dependent object: ACE on this
model is only informative where the perturbation actually moves outcomes.
"""
from __future__ import annotations

import argparse, json, pathlib, time
import numpy as np

from paired_probe import Probe

SITES = ["action_out_proj/bias", "action_out_proj/kernel", "action_in_proj/kernel",
         "time_mlp_out/kernel", "expert/mlp_1/linear/L0", "expert/mlp_1/linear/L8",
         "expert/mlp_1/linear/L17", "llm/mlp/linear/L0", "llm/mlp/linear/L17",
         "img/MlpBlock_0/Dense_1/kernel/B26"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=pathlib.Path, required=True)
    p.add_argument("--ack", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8000)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--c-rel", type=float, default=0.5)
    p.add_argument("--matched-c", type=pathlib.Path, default=None,
                   help="per-site c from calibrate_sites.py, for OUTPUT-matched probing")
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--episodes", type=int, default=6)
    a = p.parse_args()

    pr = Probe(a)
    eps = [(t, 8) for t in range(a.episodes)]
    matched = json.loads(a.matched_c.read_text()) if a.matched_c else None
    sites = [s for s in SITES if (matched is None or s in matched)]
    if matched:
        print("output-matched mode: per-site c chosen for a common |dAction|")

    state = {"c_rel": a.c_rel, "draws": a.draws, "episodes": a.episodes,
             "matched": bool(matched), "baseline": None, "blocks": []}
    if a.out.exists():
        state = json.loads(a.out.read_text())
        print(f"resuming: {len(state['blocks'])} blocks")

    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(state, indent=1))

    if state["baseline"] is None:
        pr.control(dict(site=None, pin_rng=True))
        base = [bool(pr.rollout(t, i)[0]) for t, i in eps]
        state["baseline"] = base
        save()
        print(f"deterministic baseline: {sum(base)}/{len(base)}")
    base = state["baseline"]

    done = {(b["site"], b["draw"]) for b in state["blocks"]}
    for si, site in enumerate(sites):
        c_site = matched[site]["c"] if matched else a.c_rel
        for d in range(a.draws):
            if (site, d) in done:
                continue
            seed = 7000 + si * 100 + d
            ack = pr.control(dict(site=site, seed=seed, c_rel=c_site, pin_rng=True))
            t0 = time.time()
            res = [bool(pr.rollout(t, i)[0]) for t, i in eps]
            diff = [int(r) - int(b) for r, b in zip(res, base)]
            blk = dict(site=site, draw=d, seed=seed, c_rel=c_site,
                       applied_rel=ack.get("applied_rel"),
                       result=res, paired_diff=diff, ace=float(np.mean(diff)),
                       wall_s=round(time.time() - t0, 1))
            state["blocks"].append(blk)
            save()
            print(f"{site:<38} d{d} {sum(res)}/{len(res)} vs {sum(base)}  "
                  f"ACE {blk['ace']:+.3f}  ({blk['wall_s']}s)")
    print("=== V2 SCREEN COMPLETE")


if __name__ == "__main__":
    main()
