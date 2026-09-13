"""re4 theory plan, Part 1 analysis: sampled contraction rate and input sensitivity from the paired
replays of paired_rollout.py, scored against prereg_records/PREREG_RE4T_1_2_CONTRACTION.md.
Metrics (the registered list): (a) arm joint positions; (b) joint positions + velocities/20 Hz;
(c) end-effector position; (d) end-effector position weighted by the Part 2.2 metric P (x, y, z).
"""
import argparse, json, pathlib
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("replay", type=pathlib.Path)
    ap.add_argument("--cert", type=pathlib.Path, default=pathlib.Path("results/re4_theory/2_metric/metric_certificate_panda.json"))
    ap.add_argument("--out", type=pathlib.Path); a = ap.parse_args()
    d = json.loads(a.replay.read_text()); cert = json.loads(a.cert.read_text())
    Pxyz = np.array([p["P"] for p in cert["per_axis"][:3]], float)
    det = []; rows = []
    for r in d["results"]:
        for p in r["perturbations"]:
            gq = np.array(p["gap_q"]); gqv = np.array(p["gap_qv"]); gee = np.array(p["gap_ee"]); con = np.array(p["contact"], bool)
            gee_n = np.linalg.norm(gee, axis=1); gee_p = np.sqrt((gee ** 2 * Pxyz).sum(axis=1))
            if p["kind"] == "none":
                det.append(float(gq.max())); continue
            for name, g in (("q", gq), ("qv", gqv), ("ee", gee_n), ("eeP", gee_p)):
                lam = g[1:] / np.maximum(g[:-1], 1e-12)
                for i, l in enumerate(lam):
                    rows.append(dict(kind=p["kind"], mag=p["mag"], metric=name, step=i + 1, lam=float(l), contact=bool(con[i + 1] or con[i]),
                                     episode=r["episode"], save_step=r["save_step"], g0=float(g[0]), g1=float(g[1]) if len(g) > 1 else None))
    R = np.array([(x["metric"], x["kind"], x["mag"], x["lam"], x["contact"]) for x in rows], dtype=object)
    out = dict(replay=str(a.replay), determinism_max_gap=max(det), n_perturbations=sum(len(r["perturbations"]) - 1 for r in d["results"]), metrics={})
    print(f"determinism: max restored-replay gap {max(det):.2e} over {len(det)} checks")
    print(f"{'metric':6s} {'kind':5s} {'mag':>6} {'regime':8s} {'n':>5} {'median':>7} {'p90':>7} {'frac>=1':>8}")
    for m in ("q", "qv", "ee", "eeP"):
        out["metrics"][m] = {}
        for kind in ("joint", "cmd"):
            mags = sorted({x["mag"] for x in rows if x["kind"] == kind})
            for mag in mags:
                for regime, flag in (("free", False), ("contact", True)):
                    L = np.array([x["lam"] for x in rows if x["metric"] == m and x["kind"] == kind and x["mag"] == mag and x["contact"] == flag])
                    if L.size == 0:
                        continue
                    st = dict(n=int(L.size), median=float(np.median(L)), p90=float(np.percentile(L, 90)), frac_ge_1=float((L >= 1).mean()))
                    out["metrics"][m][f"{kind}:{mag}:{regime}"] = st
                    print(f"{m:6s} {kind:5s} {mag:6g} {regime:8s} {st['n']:5d} {st['median']:7.3f} {st['p90']:7.3f} {st['frac_ge_1']:8.2f}")
    # input sensitivity: gap at the first step after a command perturbation, per unit |du|
    Lhat = {}
    for r in d["results"]:
        for p in r["perturbations"]:
            if p["kind"] != "cmd":
                continue
            ax = int(np.argmax(np.abs(p["dcmd"]))); g1 = np.array(p["gap_ee"][0]); Lhat.setdefault(ax, []).append(float(np.abs(g1[ax] if ax < 3 else np.linalg.norm(g1))) / abs(p["dcmd"][ax]))
    out["input_sensitivity_ee_m_per_unit"] = {ax: dict(median=float(np.median(v)), n=len(v)) for ax, v in Lhat.items()}
    print("L_hat (end-effector metres per unit command, first step):", {ax: round(np.median(v), 4) for ax, v in Lhat.items()})
    # registered decision: free-space median < 1 and p90 < 1.05 in at least one metric, all three joint magnitudes
    verdict = {}
    for m in ("q", "qv", "ee", "eeP"):
        ok = all((out["metrics"][m].get(f"joint:{mag}:free", {}).get("median", 9) < 1) and (out["metrics"][m].get(f"joint:{mag}:free", {}).get("p90", 9) < 1.05) for mag in (0.005, 0.01, 0.02))
        verdict[m] = bool(ok)
    out["prediction1_free_space_contraction"] = verdict
    print("prediction 1 (median<1 and p90<1.05 in free space at all magnitudes):", verdict)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1)); print("wrote", a.out)


if __name__ == "__main__":
    main()
