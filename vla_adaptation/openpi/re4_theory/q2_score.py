#!/usr/bin/env python3
"""Q2 scorer (PREREG_Q2_POSE_TRACKING.md): the primary endpoint is recomputed OFFLINE from telemetry,
identically for C (no tracker) and D (tracker), exactly as the collaborator's PoseTracker defines it
in position mode: reference pose advanced by the DC-constrained plant's predicted increment for the
NOMINAL policy command, pose from measured motion, e_p = pose - reference, then reference leaks
toward the pose (leak 0.02), anchored at the first rollout step. Late-window mean |e_p| on r_x and
r_z (steps >= 100, or the last half of shorter episodes) per episode, averaged; R = D / C per
channel with a whole-episode (paired-key) bootstrap interval. Also: healthy cost (HD vs HC), r_y
specificity, success contrasts (McNemar descriptive + task-clustered interval).
"""
import argparse, csv, gzip, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import adaptive_law as AL
from mcnemar import mcnemar_exact
try:
    from adaptive_law import fir_increment
except ImportError:                                   # main before the merge: the per-axis FIR prediction
    def fir_increment(W, H):
        return np.array([W[i, :AL.K_FIR + 1] @ H[:, i] + W[i, -1] for i in range(6)])


def open_log(d):
    p = d / "telemetry.jsonl"; g = d / "telemetry.jsonl.gz"
    return open(p) if p.exists() else gzip.open(g, "rt")


def pose_excess(d, W_ref, leak=0.02, dims=(3, 5), late=100):
    per = {}
    for line in open_log(d):
        s = json.loads(line)
        if s.get("type") != "step" or s.get("arm") != "adaptive":
            continue
        per.setdefault(int(s["episode"]), []).append(s)
    out = {}
    for ep, steps in per.items():
        steps.sort(key=lambda s: (s["phase"] != "warmup", s["t"]))
        roll = [s for s in steps if s["phase"] == "rollout"]; warm = [s for s in steps if s["phase"] == "warmup"]
        if not roll:
            continue
        x_prev = np.array((warm[-1] if warm else roll[0])["position"], float)
        hist = [np.zeros(6)] * (AL.K_FIR + 1); origin = None; rotation = np.zeros(3); reference = np.zeros(6); E = []
        for s in roll:
            x1 = np.array(s["position"], float); u_nom = np.asarray(s["raw_action"], float)[:6]; y = np.asarray(s["measured"], float)
            if origin is None:
                origin = x_prev.copy()
            hist.insert(0, u_nom); hist = hist[:AL.K_FIR + 1]
            reference = reference + fir_increment(W_ref, np.array(hist)); rotation = rotation + y[3:6]
            pose = np.concatenate([(x1 - origin) / AL.OUT[:3], rotation]); e = pose - reference
            reference = reference + leak * (pose - reference); E.append(e); x_prev = x1
        E = np.array(E); n = len(E); w = E[late:] if n > late else E[n // 2:]
        out[(str(roll[0]["task"]), str(roll[0]["init"]), ep)] = dict(n=n, late_abs=dict(zip([str(i) for i in dims], [float(np.mean(np.abs(w[:, i]))) for i in dims])),
                                                              ry_settle=float(np.mean([st["f_hat"][4] for st in roll[-50:]])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=pathlib.Path); ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--log", default=str(HERE.parent / "results/phase05/error_signal_so3.json")); ap.add_argument("--openloop", default=str(HERE.parent / "results/phase05/openloop_so3.json"))
    ap.add_argument("--out", type=pathlib.Path); a = ap.parse_args()
    M = np.array(json.loads(pathlib.Path(a.openloop).read_text())["M"]); W_ref = AL.fit_plant(a.log, dc={i: float(M[i, i]) for i in (3, 4, 5)})
    arms = {n: a.root / f"q2_{n}_{a.suite}" for n in ("C", "D", "HC", "HD")}; have = {n: d for n, d in arms.items() if (d / "episodes.csv").exists()}
    pe = {n: pose_excess(d, W_ref) for n, d in have.items()}
    # pair C/D by episode ordinal (same manifest, same order); the key check is task/init
    out = dict(suite=a.suite, arms_present=sorted(have))
    if "C" in pe and "D" in pe:
        kc = {k[2]: k for k in pe["C"]}; kd = {k[2]: k for k in pe["D"]}; common = sorted(set(kc) & set(kd))
        assert all(kc[i][:2] == kd[i][:2] for i in common), "C/D episode keys differ"
        rng = np.random.default_rng(0); res = {}
        for ch in ("3", "5"):
            c = np.array([pe["C"][kc[i]]["late_abs"][ch] for i in common]); d = np.array([pe["D"][kd[i]]["late_abs"][ch] for i in common])
            R = float(d.mean() / c.mean()); boots = [d[idx].mean() / c[idx].mean() for idx in (rng.choice(len(c), len(c), replace=True) for _ in range(10000))]
            res[{"3": "r_x", "5": "r_z"}[ch]] = dict(C_mean=float(c.mean()), D_mean=float(d.mean()), ratio=R, ratio_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
        ry_c = np.median([pe["C"][kc[i]]["ry_settle"] for i in common]); ry_d = np.median([pe["D"][kd[i]]["ry_settle"] for i in common])
        out["primary"] = dict(n_pairs=len(common), pose_excess=res,
                              prediction_1=dict(rz_ratio_le_0_80=bool(res["r_z"]["ratio"] <= 0.80), rx_ratio_le_0_80=bool(res["r_x"]["ratio"] <= 0.80), refuted_rz_ge_0_95=bool(res["r_z"]["ratio"] >= 0.95)),
                              specificity=dict(ry_settle_C=float(ry_c), ry_settle_D=float(ry_d), within_0_10=bool(abs(ry_c - ry_d) / 0.05 <= 0.10)))
        oc = {n: {(r["task"], r["init"], r["policy_seed"]): int(r["outcome"]) for r in csv.DictReader(open(have[n] / "episodes.csv"))} for n in ("C", "D")}
        keys = sorted(set(oc["C"]) & set(oc["D"])); diff = np.array([oc["D"][k] - oc["C"][k] for k in keys], float); tasks = [k[0] for k in keys]
        tb = [np.mean(diff[[i for t in rng.choice(sorted(set(tasks)), len(set(tasks)), replace=True) for i, tt in enumerate(tasks) if tt == t]]) for _ in range(10000)]
        xo = sum(oc["D"][k] and not oc["C"][k] for k in keys); yo = sum(oc["C"][k] and not oc["D"][k] for k in keys)
        out["success"] = dict(C=sum(oc["C"][k] for k in keys), D=sum(oc["D"][k] for k in keys), n=len(keys), D_minus_C=int(sum(diff)), D_only=xo, C_only=yo,
                              mcnemar_p=mcnemar_exact(yo, xo), task_clustered_ci95_points=[float(np.percentile(tb, 2.5) * 100), float(np.percentile(tb, 97.5) * 100)],
                              prediction_4_within_minus2_plus4=bool(-2 <= sum(diff) <= 4))
    if "HC" in pe and "HD" in pe:
        oc = {n: {(r["task"], r["init"], r["policy_seed"]): int(r["outcome"]) for r in csv.DictReader(open(have[n] / "episodes.csv"))} for n in ("HC", "HD")}
        keys = sorted(set(oc["HC"]) & set(oc["HD"])); lost = sum(oc["HC"][k] and not oc["HD"][k] for k in keys)
        ec = {ch: np.mean([v["late_abs"][ch] for v in pe["HC"].values()]) for ch in ("3", "5")}; ed = {ch: np.mean([v["late_abs"][ch] for v in pe["HD"].values()]) for ch in ("3", "5")}
        out["healthy"] = dict(HC=sum(oc["HC"][k] for k in keys), HD=sum(oc["HD"][k] for k in keys), n=len(keys), HD_lost_vs_HC=lost,
                              late_abs_excess=dict(HC={"r_x": float(ec["3"]), "r_z": float(ec["5"])}, HD={"r_x": float(ed["3"]), "r_z": float(ed["5"])}),
                              prediction_2=dict(lost_le_2=bool(lost <= 2), excess_increase_le_0_02=bool(max(ed["3"] - ec["3"], ed["5"] - ec["5"]) <= 0.02)))
    js = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(js)
    print(js[:4000])


if __name__ == "__main__":
    main()
