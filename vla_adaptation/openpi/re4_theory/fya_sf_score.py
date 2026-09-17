#!/usr/bin/env python3
"""Scorer for the stronger-fault reacting-policy campaign (PREREG_FYA_STRONGER_FAULT_V1.md).

Development: per candidate fault the faulted-off success on the development keys and the registered selection.
Evaluation: paired comparisons on the 40 untouched keys: faulted NT - faulted off (primary; exact McNemar descriptive,
task-clustered bootstrap of the rate difference, 10,000 draws, seed 20260916; registered minimum effect +5 net keys),
healthy NT - healthy off (individual losses listed), plus estimator transients from telemetry where present.
"""
import argparse, json, pathlib, gzip
import numpy as np
from math import comb


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(p):
    d = json.load(open(p)); arm = next(iter(d["arms"].values()))
    return {(e["task"], e["init"], e.get("sampler_seed")): bool(e["ok"]) for e in arm["per_ep"]}, arm["successes"], arm["n"]


def paired(a, b, rng, n=10000):
    keys = sorted(set(a) & set(b)); wins = [k for k in keys if a[k] and not b[k]]; losses = [k for k in keys if b[k] and not a[k]]
    diff = np.array([int(a[k]) - int(b[k]) for k in keys], float); tasks = np.array([k[0] for k in keys]); ut = np.unique(tasks)
    means = [np.concatenate([diff[tasks == ut[i]] for i in rng.integers(0, len(ut), len(ut))]).mean() for _ in range(n)]
    return dict(n=len(keys), a_successes=int(sum(a[k] for k in keys)), b_successes=int(sum(b[k] for k in keys)), wins=len(wins), losses=len(losses), net=len(wins) - len(losses),
                win_keys=[list(k) for k in wins], loss_keys=[list(k) for k in losses], rate_difference=float(diff.mean()),
                task_clustered_ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], mcnemar_exact_p_descriptive=mcnemar(len(wins), len(losses)))


def telemetry_summary(path, adapt_from=30):
    op = gzip.open if str(path).endswith(".gz") else open; eps = {}
    with op(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("type") == "step" and r.get("phase") == "rollout":
                eps.setdefault(r["episode"], []).append((r["t"] - 10, float(np.asarray(r["f_hat"])[4]), float(np.asarray(r["f_true"])[4])))
    finals = [v[-1][1] for v in eps.values()]; lengths = [len(v) for v in eps.values()]
    return dict(n=len(eps), final_fhat_ry_median=float(np.median(finals)), final_fhat_ry_iqr=[float(np.percentile(finals, 25)), float(np.percentile(finals, 75))],
                episode_steps_median=float(np.median(lengths)), unexposed_keys=int(sum(l <= adapt_from for l in lengths)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/fya_stronger_fault_v1"))
    ap.add_argument("--out", type=pathlib.Path, default=None); a = ap.parse_args()
    R = a.root; out = dict(development={}, evaluation={}); rng = np.random.default_rng(20260916)
    D = R / "development"
    for f in sorted(D.glob("dev_*.json")):
        _, s, n = load(f); out["development"][f.stem] = f"{s}/{n}"
    sel = (D / "SELECTED_FAULT.txt").read_text().strip() if (D / "SELECTED_FAULT.txt").exists() else None; out["development"]["selected"] = sel
    if sel and sel.split()[0] != "NONE":
        name = sel.split()[0]
        if (D / f"dev_{name}_off.json").exists() and (D / f"dev_{name}_nt.json").exists():
            out["development"]["nt_vs_off_dev"] = paired(load(D / f"dev_{name}_nt.json")[0], load(D / f"dev_{name}_off.json")[0], rng)
    E = R / "evaluation"; arms = {}
    for name in ("eval_healthy_off", "eval_healthy_nt", "eval_fault_off", "eval_fault_nt"):
        if (E / f"{name}.json").exists():
            arms[name] = load(E / f"{name}.json")[0]; out["evaluation"][name] = f"{sum(arms[name].values())}/{len(arms[name])}"
    if "eval_fault_nt" in arms and "eval_fault_off" in arms:
        c = paired(arms["eval_fault_nt"], arms["eval_fault_off"], rng); out["evaluation"]["primary_fault_nt_minus_off"] = c
        out["evaluation"]["T1_min_effect_5_and_ci_excludes_zero"] = bool(c["net"] >= 5 and c["task_clustered_ci95"][0] > 0)
    if "eval_healthy_nt" in arms and "eval_healthy_off" in arms:
        c = paired(arms["eval_healthy_nt"], arms["eval_healthy_off"], rng); out["evaluation"]["healthy_nt_minus_off"] = c; out["evaluation"]["T2_healthy_losses_le_3"] = bool(c["losses"] <= 3)
    for name in ("eval_fault_nt", "eval_healthy_nt"):
        for ext in (".jsonl", ".jsonl.gz"):
            p = E / f"{name}_telemetry{ext}"
            if p.exists():
                out["evaluation"][f"{name}_telemetry"] = telemetry_summary(p); break
    js = json.dumps(out, indent=1); print(js)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(js)


if __name__ == "__main__":
    main()
