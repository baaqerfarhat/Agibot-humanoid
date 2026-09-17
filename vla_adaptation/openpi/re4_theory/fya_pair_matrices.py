#!/usr/bin/env python3
"""Paired comparison of two crossed matrices on their common fidelity-valid keys (recovery study P3).

Reads two source_effects.csv files (e.g. the NT matrix and the DOB matrix built on the same M0 stream and healthy
reference), pairs rows on (task, init, seed), and reports paired differences A - B of the contrasts D0, R1, T, I and of
the adapted cells J10, J11 with the equal-task estimand, task-clustered percentile bootstrap (10,000 draws, frozen seed),
per-task and leave-one-task-out sensitivities, and a decision against a practical margin for the registered primary.
"""
import argparse, csv, json, pathlib
import numpy as np


def load(p):
    return {(int(r["task"]), int(r["init"]), int(r["seed"])): r for r in csv.DictReader(open(p))}


def boot(vals, tasks, n, rng):
    vals = np.asarray(vals, float); tasks = np.asarray(tasks); ut = np.unique(tasks); pt = np.array([vals[tasks == t].mean() for t in ut])
    b = [pt[rng.integers(0, len(ut), len(ut))].mean() for _ in range(n)]
    return dict(estimate=float(pt.mean()), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], n_keys=int(len(vals)), n_tasks=int(len(ut)),
                positive=int((vals > 0).sum()), negative=int((vals < 0).sum()), per_task={str(t): float(vals[tasks == t].mean()) for t in ut},
                leave_one_task_out={str(t): float(np.mean([vals[tasks == s].mean() for s in ut if s != t])) for t in ut}, source_weighted=float(vals.mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", type=pathlib.Path, required=True); ap.add_argument("--b", type=pathlib.Path, required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B"); ap.add_argument("--primary", default="T")
    ap.add_argument("--delta", type=float, default=1e-5); ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", type=pathlib.Path, required=True); a = ap.parse_args()
    A, B = load(a.a), load(a.b); keys = sorted(set(A) & set(B)); only_a = sorted(set(A) - set(B)); only_b = sorted(set(B) - set(A))
    rng = np.random.default_rng(a.seed); tasks = [k[0] for k in keys]; out = dict(label_a=a.label_a, label_b=a.label_b, n_common=len(keys), only_a=[list(k) for k in only_a], only_b=[list(k) for k in only_b], contrasts={})
    for q in ("D0", "R1", "T", "I", "J10", "J11", "J01"):
        d = [float(A[k][q]) - float(B[k][q]) for k in keys]; out["contrasts"][q] = boot(d, tasks, a.boot, rng)
    p = out["contrasts"][a.primary]; lo, hi = p["ci95"]
    out["primary"] = dict(contrast=f"{a.primary}({a.label_a}) - {a.primary}({a.label_b})", estimate=p["estimate"], ci95=p["ci95"],
                          decision=("resolved_positive" if lo > a.delta else "resolved_negative" if hi < -a.delta else "small_within_margin" if (lo >= -a.delta and hi <= a.delta) else "unresolved"), delta=a.delta)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(n_common=len(keys), primary=out["primary"], contrasts={q: dict(est=v["estimate"], ci=v["ci95"], pos=v["positive"], neg=v["negative"]) for q, v in out["contrasts"].items()}), indent=1))


if __name__ == "__main__":
    main()
