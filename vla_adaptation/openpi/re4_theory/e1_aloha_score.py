#!/usr/bin/env python3
"""Score an e1_aloha_continuations.py output: joint-position deviation of every branch from the healthy
branch of the same checkpoint (rad; the position-servo interface's physical quantity), integrated (rad·step
and rad·s at 50 Hz) and at the endpoint; the cube's env_state deviation; the remaining injected
disturbance and the applied-estimate error on the corrected joints; source-level aggregation and paired
contrasts against faulted/off with a source bootstrap interval."""
import argparse, json, pathlib
import numpy as np
DT = 0.02


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=pathlib.Path); ap.add_argument("--out", type=pathlib.Path); ap.add_argument("--boot", type=int, default=5000); a = ap.parse_args()
    d = json.loads(a.run.read_text()); cj = d["meta"]["fault"]["joints"]; rows = []
    for ep in d["episodes"]:
        for cp in ep["checkpoints"]:
            if cp.get("status") != "ok":
                continue
            h = cp["branches"]["healthy"]
            for name, br in cp["branches"].items():
                L = min(len(br), len(h)); dq = [float(np.linalg.norm(np.array(s["q"]) - np.array(hh["q"]))) for s, hh in zip(br[:L], h[:L])]
                dcube = [float(np.linalg.norm(np.array(s["env_state"])[:3] - np.array(hh["env_state"])[:3])) for s, hh in zip(br[:L], h[:L])]
                last = br[L - 1]; est = np.array(last["estimate_applied"])[cj]; inj = np.array(last["injected"])[cj]
                rows.append(dict(episode=ep["episode"], checkpoint=cp["checkpoint"], branch=name, steps=L, integrated_joint_rad_step=float(np.sum(dq)), integrated_joint_rad_s=float(DT * np.sum(dq)),
                                 endpoint_joint_rad=dq[-1], max_joint_rad=float(np.max(dq)), endpoint_cube_m=dcube[-1], final_remaining_disturbance=float(np.linalg.norm(np.array(last["remaining_disturbance"])[cj])),
                                 final_estimate_error=float(np.linalg.norm(est - inj)), duplicate_gap=cp["duplicate_max_joint_gap"], step20_joint_rad=(dq[19] if L > 19 else None)))
    summary = {}
    for name in sorted({r["branch"] for r in rows}):
        by = {}
        for r in rows:
            if r["branch"] == name:
                by.setdefault(r["episode"], []).append(r)
        agg = {k: [float(np.mean([r[k] for r in v])) for v in by.values()] for k in ("integrated_joint_rad_step", "endpoint_joint_rad", "endpoint_cube_m", "final_remaining_disturbance", "final_estimate_error")}
        summary[name] = dict(n_sources=len(by), **{k: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75))) for k, v in agg.items()})
    rng = np.random.default_rng(0); contrasts = {}
    def per_source(name, key):
        by = {}
        for r in rows:
            if r["branch"] == name:
                by.setdefault(r["episode"], []).append(r[key])
        return {k: float(np.mean(v)) for k, v in by.items()}
    ref = per_source("faulted_off", "integrated_joint_rad_step")
    for name in ("legacy_from_zero", "innovation_from_zero", "exact_cancellation", "hold_supplied"):
        cur = per_source(name, "integrated_joint_rad_step"); ks = sorted(ref.keys() & cur.keys())
        if ks:
            diff = np.array([cur[k] - ref[k] for k in ks]); boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(a.boot)]
            contrasts[f"{name}_minus_faulted_off"] = dict(n_sources=len(ks), mean=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], excludes_zero=bool(np.percentile(boots, 97.5) < 0))
    growth = [r["endpoint_joint_rad"] / max(r["step20_joint_rad"], 1e-9) for r in rows if r["branch"] == "faulted_off" and r["step20_joint_rad"]]
    out = dict(run=str(a.run), split_label=d["meta"].get("split_label"), n_rows=len(rows), duplicate_gap_max=float(max((r["duplicate_gap"] for r in rows), default=0)),
               faulted_growth_ratio_endpoint_over_step20_median=(float(np.median(growth)) if growth else None), summary_by_branch=summary, source_level_contrasts_integrated_joint=contrasts, rows=rows)
    js = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(js)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)[:3500])


if __name__ == "__main__":
    main()
