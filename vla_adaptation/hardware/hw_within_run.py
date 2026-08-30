"""Within-run plant identification: calibrate at startup, watch for drift thereafter.

The offline recipe does not port to this robot -- the plant is run-varying on the
load-bearing joints (hips: within-run R^2 0.89-0.92, across-run -0.64 to -0.98). The fix is
to identify the plant on the opening seconds of each rollout and monitor the residual after.

That imposes a design constraint worth stating: the calibration window must be HEALTHY. A
plant fitted on a window that already contains the fault absorbs it, and the fault becomes
invisible. So this only detects degradation that ARISES during operation -- which is the
deployment case, and the same structure as the mid-episode onset experiment in simulation.

This measures whether the residual stays flat across a healthy rollout after a startup fit.
Any systematic drift is a false-positive floor: the estimator would read it as a fault.
"""
from __future__ import annotations
import argparse, csv, glob, json, pathlib
import numpy as np
from collections import Counter

K = 4


def load(p):
    rows = list(csv.reader(open(p))); hdr = rows[0]
    D = [r for r in rows[1:] if len(r) == len(hdr)]
    idx = {h: i for i, h in enumerate(hdr)}
    def col(n):
        i = idx[n]
        return np.array([float(r[i]) if r[i] not in ("", "nan") else np.nan for r in D])
    return col, idx


def des(t, p):
    X = np.vstack([*[np.roll(t, k) for k in range(K + 1)], np.ones_like(t)]).T[K:]
    return X, p[K:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/home/mtaheri/ws_AgibotX2/Agibot-humanoid/run_logs")
    ap.add_argument("--calib", type=int, default=150, help="startup steps used to identify")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    recs = []
    for m in glob.glob(f"{a.logs}/*.meta.json"):
        try: d = json.load(open(m))
        except Exception: continue
        c = m.replace(".meta.json", ".csv")
        if pathlib.Path(c).exists() and d.get("gain_scale") == 1.0:
            recs.append((d.get("created", "")[:8], d.get("run_name", ""), c, d.get("joint_names", [])))
    sess = Counter((r[0], r[1]) for r in recs).most_common(1)[0][0]
    files = [r[2] for r in recs if (r[0], r[1]) == sess]
    joints = [r[3] for r in recs if (r[0], r[1]) == sess][0]
    print(f"session {sess[0]}  {len(files)} healthy runs   calibration window {a.calib} steps\n")
    print(f"{'joint':<24} {'post-calib R^2':>15} {'resid sd':>10} {'drift over run':>15} {'min detect':>11}")

    out = {}
    for j in joints:
        r2s, sds, drifts = [], [], []
        for f in files:
            try:
                col, idx = load(f)
                if f"{j}__tgt" not in idx: continue
                t, p = col(f"{j}__tgt"), col(f"{j}__pos_meas")
            except Exception: continue
            m = ~(np.isnan(t) | np.isnan(p))
            t, p = t[m], p[m]
            if len(t) < a.calib + 120 or t[:a.calib].std() < 0.05: continue
            X, Y = des(t, p)
            c = a.calib - K
            w = np.linalg.lstsq(X[:c], Y[:c], rcond=None)[0]
            e = Y[c:] - X[c:] @ w
            sst = ((Y[c:] - Y[c:].mean()) ** 2).sum()
            if sst < 1e-6: continue
            r2s.append(1 - (e ** 2).sum() / sst); sds.append(e.std())
            h = len(e) // 2
            drifts.append(abs(e[h:].mean() - e[:h].mean()))     # false-positive floor
        if len(r2s) < 4: continue
        r2, sd, dr = np.median(r2s), np.median(sds), np.median(drifts)
        det = 3 * sd / np.sqrt(200)
        out[j] = dict(r2=float(r2), sd=float(sd), drift=float(dr), min_detect=float(det))
        print(f"{j:<24} {r2:>15.3f} {sd:>10.5f} {dr:>15.5f} {np.degrees(det):>8.3f} deg")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    good = {k: v for k, v in out.items() if v["r2"] > 0.8}
    print(f"\n{len(good)} joints with post-calibration R^2 > 0.8")
    if good:
        print("  " + ", ".join(sorted(good)))
        dr = np.median([v["drift"] for v in good.values()])
        det = np.median([v["min_detect"] for v in good.values()])
        print(f"\nmedian drift on a HEALTHY run {dr:.5f} rad = {np.degrees(dr):.3f} deg "
              f"-- this is the false-positive floor")
        print(f"median noise-limited detection  {det:.5f} rad = {np.degrees(det):.3f} deg")
        print(f"a fault must exceed the LARGER of the two: {np.degrees(max(dr, det)):.3f} deg")


if __name__ == "__main__":
    main()
