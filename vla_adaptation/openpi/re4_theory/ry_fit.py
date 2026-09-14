#!/usr/bin/env python3
"""Reproducible scoring of the r_y plant-fit numbers quoted in record 48 / PREREG_RE4T_6_RY.md.

Per-axis increment models (the deployed ridge, lam = 1e-2, intercept included) are fitted on the
healthy logs and scored two ways, both reported:
  * training R^2 on the shipped three-episode log (what the deployed plant was fitted on);
  * leave-one-episode-out R^2 on the POOLED healthy logs (shipped 3 episodes + held-out 10 at
    init 25), the definition behind the 0.41 / 0.95 quoted in record 48.
Models: FIR K=6, FIR K=20, ARX(1) (K=6 + one own-channel past measured increment).
"""
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import adaptive_law as AL


def episodes(log):
    d = json.loads(pathlib.Path(log).read_text()); d = d["records"] if isinstance(d, dict) else d
    A = np.array(d[0]["raw_a"]); D = np.array(d[0]["raw_d"]) / np.array(AL.OUT); o = 0; eps = []
    for L in d[0]["ep_len"]:
        eps.append((A[o:o + L], D[o:o + L])); o += L
    return eps


def design(a, y, i, K, ar):
    X, Y = [], []
    for t in range(max(K, ar), len(a)):
        win = a[t - K:t + 1][::-1][:, i]; arf = y[t - ar:t, i][::-1] if ar else np.zeros(0)
        X.append(np.concatenate([win, arf, [1.0]])); Y.append(y[t, i])
    return np.array(X), np.array(Y)


def fit(X, Y, lam=1e-2):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def r2(Y, P):
    return float(1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2))


def score(eps, i, K, ar):
    Xs = [design(a, y, i, K, ar) for a, y in eps]
    Xall = np.concatenate([x for x, _ in Xs]); Yall = np.concatenate([y for _, y in Xs])
    train = r2(Yall, Xall @ fit(Xall, Yall))
    Yo, Po = [], []
    for k in range(len(Xs)):
        Xtr = np.concatenate([x for j, (x, _) in enumerate(Xs) if j != k]); Ytr = np.concatenate([y for j, (_, y) in enumerate(Xs) if j != k])
        w = fit(Xtr, Ytr); Yo.append(Xs[k][1]); Po.append(Xs[k][0] @ w)
    loo = r2(np.concatenate(Yo), np.concatenate(Po))
    return train, loo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", default=str(HERE.parent / "results/phase05/error_signal_so3.json"))
    ap.add_argument("--heldout", default=str(HERE.parent / "results/heldout/error_signal_init25.json"))
    ap.add_argument("--out", type=pathlib.Path); a = ap.parse_args()
    ship = episodes(a.shipped); held = episodes(a.heldout); pooled = ship + held
    out = dict(definition=__doc__, shipped_episodes=len(ship), heldout_episodes=len(held), models={})
    for name, K, ar in (("fir_k6", 6, 0), ("fir_k20", 20, 0), ("arx1_k6", 6, 1)):
        m = {}
        for i, ch in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            tr_ship, _ = score(ship, i, K, ar); tr_pool, loo_pool = score(pooled, i, K, ar)
            m[ch] = dict(train_r2_shipped_log=round(tr_ship, 4), train_r2_pooled=round(tr_pool, 4), loo_r2_pooled=round(loo_pool, 4))
        out["models"][name] = m
        print(name, {ch: (v["train_r2_shipped_log"], v["loo_r2_pooled"]) for ch, v in m.items()})
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
