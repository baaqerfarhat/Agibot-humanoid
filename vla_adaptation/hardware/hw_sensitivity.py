"""How large a joint-offset fault must be to be detectable on the real X2.

An encoder/calibration bias adds a constant to the measured joint position. Unlike the
stiffness fault it is NOT absorbed by the PD loop -- the loop drives the BIASED position to
the target, so the true position ends up offset permanently, and the residual
r = pos_meas - P(tgt) carries it directly.

Detectability is then set by the residual noise on healthy hardware runs. This measures that
noise per joint and converts it into a minimum detectable offset, so the hardware experiment
can be specified before any robot time is spent.
"""
from __future__ import annotations
import csv, glob, json, pathlib, argparse
import numpy as np

K = 4


def load(p):
    rows = list(csv.reader(open(p))); hdr = rows[0]
    D = [r for r in rows[1:] if len(r) == len(hdr)]
    idx = {h: i for i, h in enumerate(hdr)}
    def col(n):
        i = idx[n]
        return np.array([float(r[i]) if r[i] not in ("", "nan") else np.nan for r in D])
    return col, idx


def design(t, p):
    X = np.vstack([*[np.roll(t, k) for k in range(K + 1)], np.ones_like(t)]).T[K:]
    return X, p[K:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/home/mtaheri/ws_AgibotX2/Agibot-humanoid/run_logs")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    # healthy runs from ONE session, so the plant and the noise floor are not confounded
    recs = []
    for m in glob.glob(f"{a.logs}/*.meta.json"):
        try: d = json.load(open(m))
        except Exception: continue
        c = m.replace(".meta.json", ".csv")
        if pathlib.Path(c).exists() and d.get("gain_scale") == 1.0:
            recs.append((d.get("created", "")[:8], d.get("run_name", ""), c, d.get("joint_names", [])))
    from collections import Counter
    sess = Counter((r[0], r[1]) for r in recs).most_common(1)[0][0]
    files = [r[2] for r in recs if (r[0], r[1]) == sess]
    joints = [r[3] for r in recs if (r[0], r[1]) == sess][0]
    print(f"session {sess[0]}  {sess[1][:44]}   {len(files)} healthy runs\n")

    half = max(1, len(files) // 2)
    fit_f, held_f = files[:half], files[half:] or files[:half]
    out = {}
    print(f"{'joint':<26} {'R^2':>7} {'resid sd':>10} {'min detectable':>15} {'':>8}")
    for j in joints:
        Xs, Ys, moved = [], [], []
        for f in fit_f:
            try:
                col, idx = load(f)
                if f"{j}__tgt" not in idx: continue
                t, p = col(f"{j}__tgt"), col(f"{j}__pos_meas")
            except Exception: continue
            m = ~(np.isnan(t) | np.isnan(p))
            if m.sum() < 60: continue
            X, Y = design(t[m], p[m]); Xs.append(X); Ys.append(Y)
            moved.append(float(np.std(t[m])))        # WITHIN-run motion, not pooled
        if not Xs: continue
        Xa, Ya = np.vstack(Xs), np.concatenate(Ys)
        # Joints the policy barely moves WITHIN a run carry no information: R^2 then divides
        # by ~0 and the fit cannot be validated. Pooled variance is misleading here because a
        # static joint still parks at a different constant each run, which looks like
        # variance and also defeats a single-constant plant term. On box-pickup this is the
        # whole upper body -- the legs are what the policy actually drives.
        if np.median(moved) < 0.05:
            continue
        w = np.linalg.lstsq(Xa, Ya, rcond=None)[0]
        res, r2s = [], []
        for f in held_f:
            try:
                col, idx = load(f)
                t, p = col(f"{j}__tgt"), col(f"{j}__pos_meas")
            except Exception: continue
            m = ~(np.isnan(t) | np.isnan(p))
            if m.sum() < 60: continue
            X, Y = design(t[m], p[m]); e = Y - X @ w
            sst = ((Y - Y.mean()) ** 2).sum()
            # A run where this joint happens to sit still has sst ~ 0 and yields an R^2 of
            # -1e14, which destroys a MEAN over runs. Skip those and take the median.
            if sst < 1e-6 or Y.std() < 0.02:
                continue
            res.append(e)
            r2s.append(1 - (e ** 2).sum() / sst)
        if not res: continue
        e = np.concatenate(res); sd = float(e.std()); r2 = float(np.median(r2s))
        # a constant offset shifts the residual mean; with N steps the SE is sd/sqrt(N)
        n200 = 3 * sd / np.sqrt(200)             # 3-sigma detection over a 200-step episode
        out[j] = dict(r2=r2, sd=sd, min_offset_rad=float(n200))
        print(f"{j:<26} {r2:>7.3f} {sd:>10.5f} {n200:>12.5f} rad "
              f"{np.degrees(n200):>8.3f} deg")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    good = {k: v for k, v in out.items() if v["r2"] > 0.9}
    if good:
        mn = np.median([v["min_offset_rad"] for v in good.values()])
        print(f"\n{len(good)} joints with R^2 > 0.9; median minimum detectable offset "
              f"{mn:.5f} rad = {np.degrees(mn):.3f} deg over a 200-step episode")
        print("this is the fault magnitude the hardware experiment must exceed")


if __name__ == "__main__":
    main()
