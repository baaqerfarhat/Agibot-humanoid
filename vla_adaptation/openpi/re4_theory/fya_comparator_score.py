#!/usr/bin/env python3
"""Named-arm paired comparator scorer (closed-loop EXPERIMENT_PLAN.md §5 'Endpoints and scorer still needed').

Inputs: a manifest (expected full keys task/init/sampler_seed) and named arm result files (adaptive_law.py JSON).
Checks: every expected key present exactly once in every arm, no extra keys, sampler seed per key equals the manifest,
configuration fields equal across arms except the registered differences (baseline, gamma) which are printed.
Primary: paired success difference between two named arms (default NT minus DOB, faulted), with wins/losses, every
discordant key, per-task differences, task-clustered bootstrap 95 % interval (10,000 draws, seed 20260916),
leave-one-task-out means, exact two-sided McNemar (descriptive). Secondary contrasts are listed and labelled.
"""
import argparse, json, pathlib
import numpy as np
from math import comb


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    d = json.load(open(p)); arm = next(iter(d["arms"].values()))
    keys = [(e["task"], e["init"], e.get("sampler_seed")) for e in arm["per_ep"]]
    assert len(keys) == len(set(keys)), f"duplicate keys in {p}"
    return {k: bool(e["ok"]) for k, e in zip(keys, arm["per_ep"])}, d.get("args", {})


def paired(a, b, rng, n=10000):
    keys = sorted(a); wins = [k for k in keys if a[k] and not b[k]]; losses = [k for k in keys if b[k] and not a[k]]
    diff = np.array([int(a[k]) - int(b[k]) for k in keys], float); tasks = np.array([k[0] for k in keys]); ut = np.unique(tasks)
    per_task = {str(t): float(diff[tasks == t].mean()) for t in ut}; loo = {str(t): float(diff[tasks != t].mean()) for t in ut}
    means = [np.concatenate([diff[tasks == ut[i]] for i in rng.integers(0, len(ut), len(ut))]).mean() for _ in range(n)]
    return dict(n=len(keys), a_successes=int(sum(a.values())), b_successes=int(sum(b.values())), wins=len(wins), losses=len(losses), net=len(wins) - len(losses),
                rate_difference=float(diff.mean()), task_clustered_ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                per_task=per_task, leave_one_task_out=loo, mcnemar_exact_p_descriptive=mcnemar(len(wins), len(losses)), win_keys=[list(k) for k in wins], loss_keys=[list(k) for k in losses])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=path, repeatable (e.g. fault_nt=results/.../eval_fault_nt.json)")
    ap.add_argument("--primary", default="fault_nt-fault_dob", help="named contrast a-b for the registered primary")
    ap.add_argument("--secondary", default="", help="comma list of a-b contrasts, exploratory")
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    man = json.loads(a.manifest.read_text()); expected = [(s["task"], s["init"], s["sampler_seed"]) for s in man["scenarios"]]
    assert len(expected) == len(set(expected)), "manifest has duplicate keys"
    arms, args = {}, {}
    for spec in a.arm:
        name, path = spec.split("=", 1); arms[name], args[name] = load(path)
        missing = [k for k in expected if k not in arms[name]]; extra = [k for k in arms[name] if k not in expected]
        assert not missing and not extra, f"{name}: missing {missing[:3]} extra {extra[:3]}"
    cfg_keys = ("gamma", "dead", "norm_r", "clip", "corr_dims", "norm_channels", "law", "onset", "adapt_from", "fault_vec", "baseline", "scenario_reset")
    cfg = {n: {k: args[n].get(k) for k in cfg_keys} for n in arms}
    rng = np.random.default_rng(20260916)
    pa, pb = a.primary.split("-"); out = dict(manifest=str(a.manifest), n_keys=len(expected), arms={n: f"{sum(v.values())}/{len(v)}" for n, v in arms.items()}, configuration=cfg,
                                            primary=dict(contrast=a.primary, **paired(arms[pa], arms[pb], rng)), secondary={})
    for c in [x for x in a.secondary.split(",") if x]:
        x, y = c.split("-"); out["secondary"][c] = paired(arms[x], arms[y], rng)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    p = out["primary"]; print(json.dumps(dict(arms=out["arms"], primary={k: p[k] for k in ("contrast", "a_successes", "b_successes", "wins", "losses", "net", "rate_difference", "task_clustered_ci95", "mcnemar_exact_p_descriptive")},
                                            secondary={k: dict(net=v["net"], ci=v["task_clustered_ci95"]) for k, v in out["secondary"].items()}), indent=1))


if __name__ == "__main__":
    main()
