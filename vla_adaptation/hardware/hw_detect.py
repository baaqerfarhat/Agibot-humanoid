"""Detect a real actuator-authority fault on the X2 humanoid, from logs already recorded.

The same law used on the frozen VLA, applied to hardware and to a completely different
policy. It never touches the policy -- it needs only the commanded joint target, the measured
joint position, and a plant identified from healthy runs.

The fault here is real rather than injected by us: `gain_scale` in deploy_x2_box_pickup.py
multiplies stiffness, damping and feedforward torque together, so gain_scale = 0.7 is a 30%
loss of actuator authority -- what a worn drive, a derated motor or a flat battery does. The
run_logs directory already spans 0.1 to 1.3 across 134 rollouts.

Plant: pos_meas_t = sum_k h_k tgt_(t-k) + c, identified on gain_scale = 1.0 runs ONLY.
Residual: r_t = pos_meas_t - P(tgt_t). Healthy runs set the noise floor; faulted runs are
scored against it, which is the same separation test the simulation work required.
"""
from __future__ import annotations

import argparse, csv, glob, json, pathlib
import numpy as np

K = 4


def load(csv_path):
    rows = list(csv.reader(open(csv_path)))
    hdr, D = rows[0], rows[1:]
    idx = {h: i for i, h in enumerate(hdr)}
    def col(name):
        i = idx[name]
        return np.array([float(r[i]) if r[i] not in ("", "nan") else np.nan for r in D])
    return col, idx


def design(t, p, K=K):
    X = np.vstack([*[np.roll(t, k) for k in range(K + 1)], np.ones_like(t)]).T[K:]
    return X, p[K:]


def fit_plant(files, joints):
    """FIR per joint on healthy runs only."""
    W = {}
    for j in joints:
        Xs, Ys = [], []
        for f in files:
            try:
                col, idx = load(f)
                if f"{j}__tgt" not in idx:
                    continue
                t, p = col(f"{j}__tgt"), col(f"{j}__pos_meas")
            except Exception:
                continue
            m = ~(np.isnan(t) | np.isnan(p))
            if m.sum() < 60:
                continue
            X, Y = design(t[m], p[m])
            Xs.append(X); Ys.append(Y)
        if Xs:
            X, Y = np.vstack(Xs), np.concatenate(Ys)
            W[j] = np.linalg.lstsq(X, Y, rcond=None)[0]
    return W


def residual(f, W, joints):
    """Mean signed residual per joint: how far the achieved position falls short."""
    try:
        col, idx = load(f)
    except Exception:
        return None
    out = {}
    for j in joints:
        if j not in W or f"{j}__tgt" not in idx:
            continue
        t, p = col(f"{j}__tgt"), col(f"{j}__pos_meas")
        m = ~(np.isnan(t) | np.isnan(p))
        if m.sum() < 60:
            continue
        X, Y = design(t[m], p[m])
        out[j] = float(np.mean(Y - X @ W[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/home/mtaheri/ws_AgibotX2/Agibot-humanoid/run_logs")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--joints", default="left_hip_pitch_joint,right_hip_pitch_joint,"
                                        "left_knee_joint,right_knee_joint,left_hip_yaw_joint")
    a = ap.parse_args()
    joints = a.joints.split(",")

    meta = {}
    for m in glob.glob(f"{a.logs}/*.meta.json"):
        try:
            d = json.load(open(m))
            meta[m.replace(".meta.json", ".csv")] = d.get("gain_scale")
        except Exception:
            pass
    by_gain = {}
    for f, g in meta.items():
        if g is not None and pathlib.Path(f).exists():
            by_gain.setdefault(float(g), []).append(f)

    healthy = by_gain.get(1.0, [])
    print(f"identifying the plant on {len(healthy)} healthy runs (gain_scale = 1.0)\n")
    # hold out half the healthy runs so the noise floor is measured out-of-sample
    fit_files, held = healthy[: len(healthy) // 2], healthy[len(healthy) // 2:]
    W = fit_plant(fit_files, joints)
    print(f"{'gain_scale':>10} {'runs':>5} " + " ".join(f"{j.split('_joint')[0][:12]:>13}" for j in joints))
    res = {}
    for g in sorted(by_gain):
        files = held if g == 1.0 else by_gain[g]
        vals = [residual(f, W, joints) for f in files]
        vals = [v for v in vals if v]
        if not vals:
            continue
        row = {j: float(np.mean([v[j] for v in vals if j in v])) for j in joints
               if any(j in v for v in vals)}
        res[g] = dict(n=len(vals), mean=row)
        print(f"{g:>10} {len(vals):>5} " + " ".join(f"{row.get(j, float('nan')):>13.5f}" for j in joints))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("\n(residual = achieved minus what the HEALTHY plant predicts; gain_scale 1.0 row is held out)")


if __name__ == "__main__":
    main()
