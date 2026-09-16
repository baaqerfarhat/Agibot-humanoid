#!/usr/bin/env python3
"""Stage 2 reacting-policy bridge scorer (PREREG_FYA_RECOVERY_DEADLINE_V1.md section 6).

Arms (adaptive_law.py runs, one JSON each): healthy_off, healthy_nt, healthy_innovation, fault_off, fault_nt,
fault_innovation, delay_nt, delay_innovation. The faulted off arm is aliased for the delayed condition.
Pairing on (task, init, sampler_seed). Reports successes, paired wins/losses, exact two-sided McNemar (descriptive),
task-clustered bootstrap of the rate difference, individual healthy regressions, and the registered expectations
S1 (faulted nt/innovation > faulted off), S2 (delayed not better than immediate), S3 (<= 3 healthy losses per law).
"""
import argparse, json, pathlib
import numpy as np
from math import comb


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c); p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def load(root, name):
    d = json.load(open(root / f"{name}.json")); arm = next(iter(d["arms"].values()))
    return {(e["task"], e["init"], e.get("sampler_seed")): bool(e["ok"]) for e in arm["per_ep"]}


def paired(a, b, rng, n=10000):
    keys = sorted(set(a) & set(b)); wins = [k for k in keys if a[k] and not b[k]]; losses = [k for k in keys if b[k] and not a[k]]
    diff = np.array([int(a[k]) - int(b[k]) for k in keys], float); tasks = np.array([k[0] for k in keys])
    ut = np.unique(tasks); means = []
    for _ in range(n):
        pick = rng.integers(0, len(ut), len(ut)); means.append(np.concatenate([diff[tasks == ut[i]] for i in pick]).mean())
    return dict(n=len(keys), a_successes=int(sum(a[k] for k in keys)), b_successes=int(sum(b[k] for k in keys)), wins=len(wins), losses=len(losses),
                win_keys=[list(k) for k in wins], loss_keys=[list(k) for k in losses], rate_difference=float(diff.mean()),
                task_clustered_ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], mcnemar_exact_p_descriptive=mcnemar(len(wins), len(losses)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1/reacting_policy"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1/analysis/stage2_summary.json")); a = ap.parse_args()
    names = ["healthy_off", "healthy_nt", "healthy_innovation", "fault_off", "fault_nt", "fault_innovation", "delay_nt", "delay_innovation"]
    arms = {n: load(a.root, n) for n in names if (a.root / f"{n}.json").exists()}
    rng = np.random.default_rng(1909); out = dict(arms={n: dict(successes=sum(v.values()), n=len(v)) for n, v in arms.items()}, alias="fault_off serves as the off arm of the delayed condition", comparisons={})
    def cmp(x, y):
        if x in arms and y in arms:
            out["comparisons"][f"{x} - {y}"] = paired(arms[x], arms[y], rng)
    for law in ("nt", "innovation"):
        cmp(f"fault_{law}", "fault_off"); cmp(f"delay_{law}", "fault_off"); cmp(f"delay_{law}", f"fault_{law}"); cmp(f"healthy_{law}", "healthy_off")
    cmp("fault_nt", "fault_innovation"); cmp("delay_nt", "delay_innovation")
    c = out["comparisons"]
    out["registered"] = dict(
        S1_faulted_law_exceeds_off={law: (c[f"fault_{law} - fault_off"]["rate_difference"] > 0 and c[f"fault_{law} - fault_off"]["task_clustered_ci95"][0] > 0) if f"fault_{law} - fault_off" in c else None for law in ("nt", "innovation")},
        S2_delayed_not_better={law: (c[f"delay_{law} - fault_{law}"]["rate_difference"] <= 0) if f"delay_{law} - fault_{law}" in c else None for law in ("nt", "innovation")},
        S3_healthy_losses_le_3={law: (c[f"healthy_{law} - healthy_off"]["losses"] <= 3) if f"healthy_{law} - healthy_off" in c else None for law in ("nt", "innovation")},
        note="S1 uses the task-clustered interval excluding zero; S2 is directional; S3 counts individually lost healthy keys")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(arms=out["arms"], comparisons={k: dict(diff=v["rate_difference"], ci=v["task_clustered_ci95"], wins=v["wins"], losses=v["losses"], p=v["mcnemar_exact_p_descriptive"]) for k, v in c.items()}, registered=out["registered"]), indent=1))


if __name__ == "__main__":
    main()
