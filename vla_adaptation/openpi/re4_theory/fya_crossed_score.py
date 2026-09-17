#!/usr/bin/env python3
"""Scorer for the crossed command-stream replay (PREREG_FYA_CROSSED_REPLAY_V1.md §5-6).

Per valid key, with the common healthy reference path `ref` of the same key:
  J_ab = .05 * sum_k ||p_k(cell) - p_k(ref)||^2  over the window (m^2 s), cells J00, J10, J01, J11
  D0 = J00 - J10, D1 = J01 - J11, R0 = J00 - J01, R1 = J10 - J11, T = J00 - J11, I = D1 - D0 (= R1 - R0)
Also per cell: angular energy (SO(3) rotation vector of R_cell R_ref^T, rad^2 s), endpoint translation (m) and
rotation (rad), done step, clipped-input steps, final r_y estimate for adapted cells.

Aggregation: I is the single primary contrast; task-clustered percentile bootstrap (resample tasks, each carrying its
states and all cells; 10,000 draws, seed 20260916) with the registered equal-task estimand (mean over tasks of the
within-task mean); a source-weighted mean is a sensitivity; leave-one-task-out means; per-task effects; the interaction
decision against the registered margin delta_I. D0, R1, T, R0, D1 are descriptive. Keys failing any fidelity check are
excluded and listed. Outputs: <out>/source_costs.csv, source_effects.csv, source_intervals.json.
"""
from __future__ import annotations
import argparse, csv, gzip, json, pathlib
import numpy as np

DT = 0.05; CELLS = ("J00", "J10", "J01", "J11")


def read_json(path):
    path = pathlib.Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def rotvec(R):
    R = np.asarray(R, float).reshape(3, 3); c = float(np.clip((np.trace(R) - 1) / 2, -1, 1)); th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th)) * th


def costs(cell, ref):
    n = min(len(cell), len(ref))
    P = np.array([np.array(c["ee_pos"]) - np.array(r["ee_pos"]) for c, r in zip(cell[:n], ref[:n])])
    Rr = np.array([rotvec(np.array(c["ee_mat"]).reshape(3, 3) @ np.array(r["ee_mat"]).reshape(3, 3).T) for c, r in zip(cell[:n], ref[:n])])
    return dict(J=float(DT * np.sum(P ** 2)), J_r=float(DT * np.sum(Rr ** 2)), endpoint_p=float(np.linalg.norm(P[-1])), endpoint_r=float(np.linalg.norm(Rr[-1])), n=n)


def task_boot(vals, tasks, n, rng, equal_task=True):
    vals = np.asarray(vals, float); tasks = np.asarray(tasks); ut = np.unique(tasks)
    per_task = np.array([vals[tasks == t].mean() for t in ut])
    est = float(per_task.mean()) if equal_task else float(vals.mean())
    boots = []
    for _ in range(n):
        pick = rng.integers(0, len(ut), len(ut))
        boots.append(per_task[pick].mean() if equal_task else np.concatenate([vals[tasks == ut[i]] for i in pick]).mean())
    return dict(estimate=est, ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], n_sources=int(len(vals)), n_tasks=int(len(ut)),
                per_task={str(t): float(vals[tasks == t].mean()) for t in ut}, positive=int((vals > 0).sum()), negative=int((vals < 0).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=pathlib.Path); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260916); ap.add_argument("--delta-I", type=float, default=1e-5)
    ap.add_argument("--primary", default="I", choices=["I", "R1", "D0", "T"], help="registered primary contrast (I for the archived matrices; R1 for the coupled healthy replication)"); a = ap.parse_args()
    d = read_json(a.run); rng = np.random.default_rng(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    valid = [k for k in d["keys"] if k["fidelity"]["valid"]]; invalid = [dict(key=k["key"], checks=k["fidelity"]["checks"], n_window=k["n_window_steps"]) for k in d["keys"] if not k["fidelity"]["valid"]]
    cost_rows, eff_rows = [], []
    for k in valid:
        ref = k["branches"]["ref"]; C = {}
        for cell in CELLS:
            b = k["branches"][cell]; c = costs(b, ref)
            c.update(done_step=k["done_steps"].get(cell), clipped_steps=int(sum(x["clipped_input"] for x in b)), final_fhat_ry=float(b[-1]["f_hat_after"][4]), updates=int(sum(x["update_applied"] for x in b)))
            C[cell] = c; cost_rows.append(dict(task=k["key"]["task"], init=k["key"]["init"], seed=k["key"]["sampler_seed"], cell=cell, **c))
        J = {c: C[c]["J"] for c in CELLS}; Jr = {c: C[c]["J_r"] for c in CELLS}
        eff = dict(task=k["key"]["task"], init=k["key"]["init"], seed=k["key"]["sampler_seed"], **{f"{c}": J[c] for c in CELLS},
                   D0=J["J00"] - J["J10"], D1=J["J01"] - J["J11"], R0=J["J00"] - J["J01"], R1=J["J10"] - J["J11"], T=J["J00"] - J["J11"], I=(J["J01"] - J["J11"]) - (J["J00"] - J["J10"]),
                   I_r=(Jr["J01"] - Jr["J11"]) - (Jr["J00"] - Jr["J10"]), D0_r=Jr["J00"] - Jr["J10"], R1_r=Jr["J10"] - Jr["J11"], T_r=Jr["J00"] - Jr["J11"],
                   cell_order="/".join(k["cell_order"]))
        eff_rows.append(eff)
    if not eff_rows:
        (a.out / "source_intervals.json").write_text(json.dumps(dict(run=str(a.run), n_keys_valid=0, invalid=invalid), indent=1))
        raise SystemExit(f"no valid keys ({len(invalid)} invalid): nothing to score")
    for name, rows in (("source_costs.csv", cost_rows), ("source_effects.csv", eff_rows)):
        with open(a.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    tasks = [r["task"] for r in eff_rows]; agg = {}
    for q in ("I", "D0", "D1", "R0", "R1", "T", "I_r", "D0_r", "R1_r", "T_r"):
        agg[q] = dict(equal_task=task_boot([r[q] for r in eff_rows], tasks, a.boot, rng, True), source_weighted=task_boot([r[q] for r in eff_rows], tasks, a.boot, rng, False))
    Ivals = np.array([r["I"] for r in eff_rows]); ut = sorted(set(tasks))
    loo = {str(t): float(np.mean([np.mean(Ivals[np.array(tasks) == s]) for s in ut if s != t])) for t in ut}
    Pvals = np.array([r[a.primary] for r in eff_rows]); loo_primary = {str(t): float(np.mean([np.mean(Pvals[np.array(tasks) == s]) for s in ut if s != t])) for t in ut}
    lo, hi = agg["I"]["equal_task"]["ci95"]
    decision = ("resolved_positive" if lo > a.delta_I else "resolved_negative" if hi < -a.delta_I else "small_within_margin" if (lo >= -a.delta_I and hi <= a.delta_I) else "unresolved")
    plo, phi = agg[a.primary]["equal_task"]["ci95"]
    primary_decision = dict(contrast=a.primary, estimate=agg[a.primary]["equal_task"]["estimate"], ci95=[plo, phi],
                            decision=("resolved_positive" if plo > a.delta_I else "resolved_negative" if phi < -a.delta_I else "small_within_margin" if (plo >= -a.delta_I and phi <= a.delta_I) else "unresolved"),
                            leave_one_task_out=loo_primary, per_task=agg[a.primary]["equal_task"]["per_task"])
    cells_med = {c: dict(J_median=float(np.median([r[c] for r in eff_rows])), J_q25=float(np.percentile([r[c] for r in eff_rows], 25)), J_q75=float(np.percentile([r[c] for r in eff_rows], 75)),
                         endpoint_p_median=float(np.median([x["endpoint_p"] for x in cost_rows if x["cell"] == c])), done_any=int(sum(x["done_step"] is not None for x in cost_rows if x["cell"] == c)),
                         clipped_total=int(sum(x["clipped_steps"] for x in cost_rows if x["cell"] == c))) for c in CELLS}
    out = dict(run=str(a.run), label=d["meta"].get("label"), n_keys_valid=len(valid), n_tasks=len(ut), invalid=invalid, delta_I=a.delta_I, boot=a.boot, seed=a.seed,
               units=dict(J="m^2 s", J_r="rad^2 s", endpoint_p="m", endpoint_r="rad"), primary_I=agg["I"], interaction_decision=decision, registered_primary=primary_decision, leave_one_task_out_I=loo,
               descriptive=agg, cells=cells_med, secondary_ratio_R1_over_T=dict(note="unstable near T = 0; reported per source only", values=[(r["R1"] / r["T"]) if abs(r["T"]) > 1e-9 else None for r in eff_rows]),
               fidelity=[dict(key=k["key"], **{kk: vv for kk, vv in k["fidelity"].items() if kk != "checks"}) for k in valid], driver_sha256=d["meta"]["driver_sha256"], configuration_sha256=d["meta"]["configuration_sha256"])
    (a.out / "source_intervals.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(n=len(valid), invalid=len(invalid), decision=decision, I=agg["I"]["equal_task"], D0=agg["D0"]["equal_task"]["estimate"], R1=agg["R1"]["equal_task"]["estimate"], T=agg["T"]["equal_task"]["estimate"], cells={c: cells_med[c]["J_median"] for c in CELLS}), indent=1))


if __name__ == "__main__":
    main()
