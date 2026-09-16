#!/usr/bin/env python3
"""Source-level physical costs and benefits for a fya_continuations.py run (EXPERIMENT_PLAN.md section 4,
'Locked test outputs and decision rules'). Measured quantities only; predictions are handled by fya_forecast.py.

Per source (one checkpoint) and branch, relative to the matched healthy_off continuation of the same source:
  E_p[k] = ee_pos_branch[k] - ee_pos_healthy[k]                       (m, 3-vector)
  E_r[k] = log(R_branch[k] R_healthy[k]^T)                          (rad, SO(3) rotation vector)
  J_p = dt * sum_k ||E_p[k]||^2  (m^2 s, PRIMARY, dt = .05 s)          J_r = dt * sum_k ||E_r[k]||^2  (rad^2 s)
  endpoint translation ||E_p[H]|| (m) and rotation ||E_r[H]|| (rad)
  remaining-disturbance norm at the endpoint, mean |fhat - f| on the corrected channels, mean residual norm.
Benefits per scenario and law A in {nt, innovation, reference}:  B = J_off - J_A  (positive = benefit), plus the
relative reduction when J_off is not negligible. Healthy harm: J_p of healthy_nt / healthy_innovation (deviation
from healthy_off, which is zero for itself), and the duplicate gap.

Aggregation: the whole source is the unit. Source-level percentile bootstrap and (primary) task-clustered bootstrap
(resample the ten tasks, each carrying its two states) with a fixed seed; descriptive, unadjusted. Sources excluded
by the length rule or the fidelity tolerance are listed, never silently dropped.

Outputs: <out>/source_metrics.csv (one row per source x branch), <out>/source_benefits.csv (one row per source x
scenario x law), <out>/source_intervals.json (aggregates, contrasts, exclusions, counts).
"""
from __future__ import annotations
import argparse, csv, json, pathlib
import numpy as np

DT = 0.05


def read_json(path):
    """json or gzip-compressed json (campaign runs are stored .json.gz in the repository)."""
    import gzip
    path = pathlib.Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())
LAWS = ("nt", "innovation", "reference")


def rotvec(R):
    """log map of a rotation matrix -> rotation vector (rad)."""
    R = np.asarray(R, float).reshape(3, 3)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)); th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


def branch_errors(br, hb):
    P = np.array([np.array(s["ee_pos"]) - np.array(h["ee_pos"]) for s, h in zip(br, hb)])
    Rr = np.array([rotvec(np.array(s["ee_mat"]).reshape(3, 3) @ np.array(h["ee_mat"]).reshape(3, 3).T) for s, h in zip(br, hb)])
    return P, Rr


def source_rows(src, corr_dims):
    hb = src["branches"]["healthy_off"]; rows = {}
    for name, br in src["branches"].items():
        P, Rr = branch_errors(br, hb)
        f = np.array(br[0]["injected"]); cd = list(corr_dims)
        est_err = np.mean([np.abs(np.array(s["fhat_before"])[cd] - f[cd]).mean() for s in br])
        rows[name] = dict(source=src["episode"], task=src["task"], init=src["init"], branch=name,
                          J_p=float(DT * np.sum(np.sum(P ** 2, 1))), J_r=float(DT * np.sum(np.sum(Rr ** 2, 1))),
                          endpoint_p=float(np.linalg.norm(P[-1])), endpoint_r=float(np.linalg.norm(Rr[-1])),
                          sum_abs_p=float(np.sum(np.linalg.norm(P, axis=1))),
                          remaining_end=float(np.linalg.norm(np.array(br[-1]["remaining_disturbance"]))),
                          mean_est_err=float(est_err), mean_residual_norm=float(np.mean([np.linalg.norm(s["residual"]) for s in br])),
                          updates_applied=int(sum(bool(s["update_applied"]) for s in br)),
                          saturated_steps=int(sum(bool(s["saturated"]) for s in br)), done_any=bool(any(s["done"] for s in br)),
                          E_p=P, E_r=Rr)
    return rows


def boot(vals, keys, n, rng):
    """percentile bootstrap of the mean; keys = cluster labels (task-clustered) or None (source-level)."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    if keys is None:
        idx = lambda: rng.integers(0, len(vals), len(vals))
        means = [vals[idx()].mean() for _ in range(n)]
    else:
        keys = np.asarray(keys); uk = np.unique(keys); groups = [vals[keys == k] for k in uk]
        means = [np.concatenate([groups[i] for i in rng.integers(0, len(uk), len(uk))]).mean() for _ in range(n)]
    return dict(n=int(len(vals)), mean=float(vals.mean()), median=float(np.median(vals)), ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                positive=int((vals > 0).sum()), negative=int((vals < 0).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=pathlib.Path); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=1909)
    ap.add_argument("--zero-tol", type=float, default=None, help="tolerance around zero benefit (m^2 s); default: max healthy_duplicate J_p + 1e-9")
    a = ap.parse_args()
    d = read_json(a.run); meta = d["meta"]; cd = meta["corr_dims"]; scen = [s["id"] for s in meta["scenarios"]]
    rng = np.random.default_rng(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    valid = [s for s in d["episodes"] if s.get("status") == "ok"]
    excluded = [dict(episode=s["episode"], task=s["task"], init=s["init"], status=s["status"]) for s in d["episodes"] if s.get("status") != "ok"]
    per = {s["episode"]: source_rows(s, cd) for s in valid}
    mcols = ["source", "task", "init", "branch", "J_p", "J_r", "endpoint_p", "endpoint_r", "sum_abs_p", "remaining_end", "mean_est_err", "mean_residual_norm", "updates_applied", "saturated_steps", "done_any"]
    with open(a.out / "source_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mcols); w.writeheader()
        for e in sorted(per):
            for name in meta["branches"]:
                w.writerow({k: per[e][name][k] for k in mcols})
    dup = [per[e]["healthy_duplicate"]["J_p"] for e in per]
    tol = a.zero_tol if a.zero_tol is not None else (max(dup) if dup else 0.0) + 1e-9
    ben_rows = []
    for e in sorted(per):
        for sc in scen:
            off = per[e][f"{sc}__off"]
            for law in LAWS:
                A = per[e][f"{sc}__{law}"]
                ben_rows.append(dict(source=e, task=per[e]["healthy_off"]["task"], init=per[e]["healthy_off"]["init"], scenario=sc, law=law,
                                     J_off=off["J_p"], J_A=A["J_p"], B=off["J_p"] - A["J_p"], rel=((off["J_p"] - A["J_p"]) / off["J_p"] if off["J_p"] > tol else None),
                                     B_r=off["J_r"] - A["J_r"], endpoint_off=off["endpoint_p"], endpoint_A=A["endpoint_p"], sign=("benefit" if off["J_p"] - A["J_p"] > tol else ("harm" if off["J_p"] - A["J_p"] < -tol else "within_tol"))))
    with open(a.out / "source_benefits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ben_rows[0].keys())); w.writeheader(); w.writerows(ben_rows)
    agg = {}
    for sc in scen:
        agg[sc] = {}
        for law in LAWS:
            rows = [r for r in ben_rows if r["scenario"] == sc and r["law"] == law]
            B = [r["B"] for r in rows]; T = [r["task"] for r in rows]
            agg[sc][law] = dict(B_task_clustered=boot(B, T, a.boot, rng), B_source_level=boot(B, None, a.boot, rng),
                                J_off_median=float(np.median([r["J_off"] for r in rows])), J_A_median=float(np.median([r["J_A"] for r in rows])),
                                rel_median=(float(np.median([r["rel"] for r in rows if r["rel"] is not None])) if any(r["rel"] is not None for r in rows) else None),
                                endpoint_off_median=float(np.median([r["endpoint_off"] for r in rows])), endpoint_A_median=float(np.median([r["endpoint_A"] for r in rows])),
                                sources_benefit=int(sum(r["sign"] == "benefit" for r in rows)), sources_harm=int(sum(r["sign"] == "harm" for r in rows)),
                                B_r_task_clustered=boot([r["B_r"] for r in rows], T, a.boot, rng))
    # delay / cap / sign contrasts against the reference scenario, per law, paired within source
    contrasts = {}
    for law in LAWS:
        for sc in scen:
            if sc == "reference":
                continue
            pairs = []
            for e in sorted(per):
                r0 = per[e][f"reference__{law}"]["J_p"] - per[e]["reference__off"]["J_p"]
                r1 = per[e][f"{sc}__{law}"]["J_p"] - per[e][f"{sc}__off"]["J_p"]
                pairs.append((r1 - r0, per[e]["healthy_off"]["task"]))
            contrasts[f"B({sc},{law}) - B(reference,{law})"] = boot([p[0] for p in pairs], [p[1] for p in pairs], a.boot, rng)
        # scenario endpoints of the law itself (delay/cap effect on the adapted cost, not only on the benefit)
        for sc in scen:
            if sc == "reference":
                continue
            pairs = [(per[e][f"{sc}__{law}"]["J_p"] - per[e][f"reference__{law}"]["J_p"], per[e]["healthy_off"]["task"]) for e in sorted(per)]
            contrasts[f"J({sc},{law}) - J(reference,{law})"] = boot([p[0] for p in pairs], [p[1] for p in pairs], a.boot, rng)
    healthy = {}
    for name in ("healthy_nt", "healthy_innovation", "healthy_duplicate"):
        J = [per[e][name]["J_p"] for e in sorted(per)]; T = [per[e]["healthy_off"]["task"] for e in sorted(per)]
        healthy[name] = dict(J_p=boot(J, T, a.boot, rng), J_p_max=float(max(J)), endpoint_p_median=float(np.median([per[e][name]["endpoint_p"] for e in per])),
                             updates_applied_mean=float(np.mean([per[e][name]["updates_applied"] for e in per])))
    fid = [dict(episode=s["episode"], **{k: v for k, v in s["fidelity"].items() if k != "fingerprint_mismatched_fields"}) for s in valid]
    out = dict(run=str(a.run), split_label=meta.get("split_label"), n_sources_valid=len(valid), excluded=excluded, zero_tolerance_m2s=tol,
               units=dict(J_p="m^2 s", J_r="rad^2 s", endpoint_p="m", endpoint_r="rad"), boot=a.boot, seed=a.seed,
               benefits=agg, contrasts=contrasts, healthy=healthy, fidelity=fid,
               saturation=dict(total_saturated_steps=int(sum(per[e][n]["saturated_steps"] for e in per for n in per[e])), branches_with_saturation=sorted({n for e in per for n in per[e] if per[e][n]["saturated_steps"]})),
               done_flags=dict(branches_with_done=sorted({n for e in per for n in per[e] if per[e][n]["done_any"]})))
    (a.out / "source_intervals.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(n_sources=len(valid), excluded=len(excluded), benefits={sc: {law: agg[sc][law]["B_task_clustered"] for law in LAWS} for sc in scen},
                          healthy={n: healthy[n]["J_p"] for n in healthy}), indent=1))


if __name__ == "__main__":
    main()
