#!/usr/bin/env python3
"""Score an e1_continuations.py output (units: *_m / *_rad sums are per-step sums, i.e. m·step; *_m_s / *_rad_s are
time integrals at 20 Hz, DT = 0.05 s; the estimate error is on the corrected rotation channels of the APPLIED estimate): PHYSICAL error of every branch against the healthy branch of
the same checkpoint (joint-position norm in rad, end-effector translation in m, orientation angle in
rad, augmented joint metric with velocity x control period), integrated over the continuation and at
the endpoint; the remaining injected disturbance and the observation-side estimate error alongside,
never mixed. Aggregation is by source episode (the plan's inference unit), with a source-level
bootstrap interval on paired branch differences.
"""
import argparse, json, pathlib
import numpy as np
DT = 0.05   # 20 Hz control period


def rot_angle(Ra, Rb):
    R = np.asarray(Ra).reshape(3, 3).T @ np.asarray(Rb).reshape(3, 3)
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def errors(branch, healthy):
    L = min(len(branch), len(healthy)); out = dict(joint=[], ee=[], angle=[], joint_aug=[], remaining=[], est_err=[], contact=[])
    for s, h in zip(branch[:L], healthy[:L]):
        dq = np.array(s["q"]) - np.array(h["q"]); dv = np.array(s["v"]) - np.array(h["v"])
        out["joint"].append(float(np.linalg.norm(dq))); out["joint_aug"].append(float(np.sqrt(np.sum(dq ** 2) + np.sum((dv * DT) ** 2))))
        out["ee"].append(float(np.linalg.norm(np.array(s["ee_pos"]) - np.array(h["ee_pos"])))); out["angle"].append(rot_angle(h["ee_mat"], s["ee_mat"]))
        out["remaining"].append(float(np.linalg.norm(s["remaining_disturbance"])))
        est = np.array(s.get("estimate_applied", s["f_hat"]))[3:6]; out["est_err"].append(float(np.linalg.norm(est - np.array(s["injected"])[3:6])))
        out["contact"].append(bool(s["contact"]) or bool(h["contact"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run", type=pathlib.Path); ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--boot", type=int, default=5000); a = ap.parse_args()
    d = json.loads(a.run.read_text()); rows = []
    for ep in d["episodes"]:
        for cp in ep["checkpoints"]:
            if cp.get("status") != "ok":
                continue
            h = cp["branches"]["healthy"]
            for name, br in cp["branches"].items():
                e = errors(br, h)
                rows.append(dict(episode=ep["episode"], task=ep["task"], checkpoint=cp["checkpoint"], branch=name, steps=len(e["ee"]),
                                 integrated_ee_m=float(np.sum(e["ee"])), integrated_joint_rad=float(np.sum(e["joint"])), integrated_angle_rad=float(np.sum(e["angle"])),
                                 integrated_ee_m_s=float(DT * np.sum(e["ee"])), integrated_joint_rad_s=float(DT * np.sum(e["joint"])), integrated_angle_rad_s=float(DT * np.sum(e["angle"])),
                                 endpoint_ee_m=e["ee"][-1], endpoint_joint_rad=e["joint"][-1], endpoint_angle_rad=e["angle"][-1],
                                 max_ee_m=float(np.max(e["ee"])), contact_fraction=float(np.mean(e["contact"])),
                                 final_remaining_disturbance=e["remaining"][-1], final_estimate_error=e["est_err"][-1],
                                 duplicate_gap=cp["duplicate_max_joint_gap"]))
    # per-branch summary by source episode: mean over checkpoints within an episode, then median/IQR over episodes
    summary = {}
    for name in sorted({r["branch"] for r in rows}):
        by_ep = {}
        for r in rows:
            if r["branch"] == name:
                by_ep.setdefault(r["episode"], []).append(r)
        agg = {k: [float(np.mean([r[k] for r in v])) for v in by_ep.values()] for k in ("integrated_ee_m", "endpoint_ee_m", "endpoint_angle_rad", "integrated_joint_rad", "final_remaining_disturbance", "final_estimate_error")}
        summary[name] = dict(n_sources=len(by_ep), **{k: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75))) for k, v in agg.items()})
    # paired source-level contrasts on integrated ee error: each adaptive branch minus faulted_off (does the correction reduce physical deviation?)
    rng = np.random.default_rng(0); contrasts = {}
    def per_source(name, key):
        by = {}
        for r in rows:
            if r["branch"] == name:
                by.setdefault(r["episode"], []).append(r[key])
        return {k: float(np.mean(v)) for k, v in by.items()}
    ref = per_source("faulted_off", "integrated_ee_m")
    for name in ("legacy_from_zero", "innovation_from_zero", "exact_cancellation", "hold_supplied"):
        cur = per_source(name, "integrated_ee_m"); ks = sorted(ref.keys() & cur.keys())
        if not ks:
            continue
        diff = np.array([cur[k] - ref[k] for k in ks]); boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(a.boot)]
        contrasts[f"{name}_minus_faulted_off"] = dict(n_sources=len(ks), mean=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                     excludes_zero=bool(np.percentile(boots, 97.5) < 0 or np.percentile(boots, 2.5) > 0))
    out = dict(run=str(a.run), split_label=d["meta"].get("split_label"), n_rows=len(rows), duplicate_gap_max=float(max((r["duplicate_gap"] for r in rows), default=0)),
               summary_by_branch=summary, source_level_contrasts_integrated_ee=contrasts, rows=rows)
    js = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(js)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
