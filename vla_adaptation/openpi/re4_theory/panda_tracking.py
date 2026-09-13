"""re4 theory plan, Part 2.2: Panda operational-space tracking poles from the healthy logs
(prereg_records/PREREG_RE4T_1_2_CONTRACTION.md). CPU only.

Per end-effector axis i: d_i[k+1] = lambda_i d_i[k] + g_i u_i[k], with d the measured increment and u
the commanded increment (both in the log's raw units; lambda is scale-free). Leave-one-episode-out
across the pooled episodes of the three healthy logs. Candidate metric P = diag(1/(1-lambda^2)).
"""
import json, hashlib, pathlib, datetime
import numpy as np

LOGS = ["results/phase05/error_signal_so3.json", "results/heldout/error_signal_init25.json",
        "results/re4_evidence/H_third_init/error_signal_init5.json"]
AX = ["x", "y", "z", "rx", "ry", "rz"]


def episodes(path):
    d = json.loads(pathlib.Path(path).read_text())
    rec = d["records"][0] if isinstance(d, dict) else d[0]
    u = np.asarray(rec["raw_a"], float); y = np.asarray(rec["raw_d"], float); lens = rec["ep_len"]
    out, i = [], 0
    for L in lens:
        out.append((u[i:i + L], y[i:i + L])); i += L
    return out


def fit(eps, i):
    X, Y = [], []
    for u, y in eps:
        X.append(np.c_[y[:-1, i], u[:-1, i]]); Y.append(y[1:, i])
    X, Y = np.concatenate(X), np.concatenate(Y)
    w = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(w[0]), float(w[1])


def r2(u, y, i, lam, g):
    pred = lam * y[:-1, i] + g * u[:-1, i]; res = y[1:, i] - pred
    st = float(((y[1:, i] - y[1:, i].mean()) ** 2).sum())
    return 1 - float((res ** 2).sum()) / st if st > 1e-15 else float("nan")


def main():
    eps = [e for p in LOGS for e in episodes(p)]
    per = []
    for i, name in enumerate(AX):
        lams, gs, r2s = [], [], []
        for h in range(len(eps)):
            lam, g = fit([e for k, e in enumerate(eps) if k != h], i); lams.append(lam); gs.append(g); r2s.append(r2(*eps[h], i, lam, g))
        lam_all, g_all = fit(eps, i)
        ok = all(0 < l < 1 for l in lams) and all(r >= 0.3 for r in r2s if r == r)
        per.append(dict(axis=name, lam=lam_all, g=g_all, lam_fold_min=min(lams), lam_fold_max=max(lams),
                        heldout_r2_min=min(r2s), heldout_r2_median=float(np.median(r2s)), in_domain=ok,
                        P=1.0 / (1.0 - lam_all ** 2) if 0 < lam_all < 1 else None))
    cert = dict(schema_version="re4t-v1", robot="panda_libero_osc", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                model="d[k+1] = lam d[k] + g u[k] per end-effector axis; P = diag(1/(1-lam^2)) candidate metric",
                episodes=len(eps), folds="leave-one-episode-out over the pooled logs", r2_min=0.3,
                domain=[p["axis"] for p in per if p["in_domain"]], per_axis=per,
                sources={p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in LOGS})
    out = pathlib.Path("results/re4_theory/2_metric/metric_certificate_panda.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=1))
    print(f"{len(eps)} healthy episodes pooled from 3 logs")
    print(f"{'axis':4s} {'lam':>7} {'fold':>15} {'g':>8} {'R2 min':>7} {'R2 med':>7}  domain")
    for p in per:
        print(f"{p['axis']:4s} {p['lam']:7.3f} {p['lam_fold_min']:7.3f}-{p['lam_fold_max']:6.3f} {p['g']:8.4f} {p['heldout_r2_min']:7.3f} {p['heldout_r2_median']:7.3f}  {'yes' if p['in_domain'] else 'NO'}")
    print("wrote", out)


if __name__ == "__main__":
    main()
