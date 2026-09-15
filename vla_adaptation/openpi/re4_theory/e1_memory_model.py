#!/usr/bin/env python3
"""E1 prediction 4 (PREREG_E1_PHYSICAL_CONTINUATIONS.md): a signed finite-memory linear response
model of the PHYSICAL end-effector deviation driven by the remaining injected disturbance, against
two comparators of the same fit budget, with a remainder envelope selected on the qualification
sources and scored prospectively on the locked sources. Registered here, before the locked pass.

Quantities (per checkpoint, per non-healthy branch, step k = 1..H):
  e_k  = end-effector position deviation from the healthy branch, 3-vector (m)
  d_k  = remaining injected command disturbance on the corrected channels, 6-vector, normalised
         units (the executed command minus the nominal one, i.e. f + correction)
Models (all linear, fitted by least squares on the fit sources' checkpoints, per output axis):
  memory   : e_k = sum_{j=0}^{L-1} G_j d_{k-j}                       (finite memory, L = 20, signed)
  geometric: e_k = lam * e_{k-1} + B d_k, lam fitted, |lam| < 1 enforced by the fit range
  neutral  : e_k = e_{k-1} + B d_k                                    (pure accumulation, lam = 1)
Envelope: on the qualification sources the memory model's absolute residual |e_k - e^_k| per step
gives a per-step 95th percentile r_k; the bound is R_k = |e^_k| + r_k (zero-error convention: a
step with |e_k| = 0 is covered iff R_k >= 0, trivially). Scored on the locked sources:
whole-continuation coverage (all H steps inside), per-step coverage, endpoint bound/error ratio,
and the paired source-level contrast of integrated absolute prediction error, memory minus each
comparator, with a source-level bootstrap 95 % interval.
"""
import argparse, json, pathlib
import numpy as np
L = 20


def series(cp):
    h = cp["branches"]["healthy"]; out = {}
    for name, br in cp["branches"].items():
        if name in ("healthy", "healthy_duplicate"):
            continue
        e = np.array([np.array(s["ee_pos"]) - np.array(hh["ee_pos"]) for s, hh in zip(br, h)])
        d = np.array([s["remaining_disturbance"] for s in br])
        out[name] = (e, d)
    return out


def design_memory(d, k):
    rows = []
    for j in range(L):
        rows.append(d[k - j] if k - j >= 0 else np.zeros(d.shape[1]))
    return np.concatenate(rows)


def fit_memory(data):
    X, Y = [], []
    for e, d in data:
        for k in range(len(e)):
            X.append(design_memory(d, k)); Y.append(e[k])
    X, Y = np.array(X), np.array(Y); lam = 1e-6
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)          # (L*6, 3)


def predict_memory(G, d):
    return np.array([design_memory(d, k) @ G for k in range(len(d))])


def fit_geometric(data, neutral=False):
    best = None
    lams = [1.0] if neutral else np.linspace(0.5, 0.999, 50)
    for lam in lams:
        X, Y = [], []
        for e, d in data:
            prev = np.zeros(3)
            for k in range(len(e)):
                X.append(d[k]); Y.append(e[k] - lam * prev); prev = e[k]
        X, Y = np.array(X), np.array(Y); B = np.linalg.lstsq(X, Y, rcond=None)[0]
        err = float(np.mean((Y - X @ B) ** 2))
        if best is None or err < best[0]:
            best = (err, lam, B)
    return best[1], best[2]


def predict_geometric(lam, B, d):
    out = []; prev = np.zeros(3)
    for k in range(len(d)):
        prev = lam * prev + d[k] @ B; out.append(prev.copy())
    return np.array(out)


def collect(run, task_filter=None):
    d = json.loads(pathlib.Path(run).read_text()); data = []
    for ep in d["episodes"]:
        if task_filter is not None and not task_filter(ep["task"]):
            continue
        for cp in ep["checkpoints"]:
            if cp.get("status") == "ok":
                for name, (e, dd) in series(cp).items():
                    data.append(dict(episode=ep["episode"], task=ep["task"], checkpoint=cp["checkpoint"], branch=name, e=e, d=dd))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", type=pathlib.Path, required=True, help="fit + qualification sources (state 39: tasks 0-5 fit, 6-9 qualification)")
    ap.add_argument("--pass2", type=pathlib.Path, required=True, help="locked test sources")
    ap.add_argument("--out", type=pathlib.Path); ap.add_argument("--boot", type=int, default=5000); a = ap.parse_args()
    fit = collect(a.pass1, lambda t: t < 6); qual = collect(a.pass1, lambda t: t >= 6); test = collect(a.pass2)
    pairs = lambda rows: [(r["e"], r["d"]) for r in rows]
    G = fit_memory(pairs(fit)); lam, Bg = fit_geometric(pairs(fit)); _, Bn = fit_geometric(pairs(fit), neutral=True)
    # envelope from qualification residuals of the memory model
    res = [np.linalg.norm(r["e"] - predict_memory(G, r["d"]), axis=1) for r in qual]
    H = max(len(x) for x in res); r95 = np.array([np.percentile([x[k] for x in res if len(x) > k], 95) for k in range(H)])
    frozen = dict(L=L, lam_geometric=float(lam), envelope_r95_by_step=r95.tolist(), fit_rows=len(fit), qual_rows=len(qual))
    # locked scoring
    rows = []; cov_all = []; ratios = []
    for r in test:
        pm = predict_memory(G, r["d"]); pg = predict_geometric(lam, Bg, r["d"]); pn = predict_geometric(1.0, Bn, r["d"])
        err = lambda p: float(np.sum(np.linalg.norm(r["e"] - p, axis=1)))
        en = np.linalg.norm(r["e"], axis=1); bound = np.linalg.norm(pm, axis=1) + r95[:len(en)]
        inside = en <= bound; cov_all.append(bool(inside.all()))
        if en[-1] > 0:
            ratios.append(float(bound[-1] / en[-1]))
        rows.append(dict(episode=r["episode"], task=r["task"], checkpoint=r["checkpoint"], branch=r["branch"], int_err_memory=err(pm), int_err_geometric=err(pg), int_err_neutral=err(pn),
                         whole_coverage=bool(inside.all()), step_coverage=float(inside.mean()), endpoint_bound_over_error=(float(bound[-1] / en[-1]) if en[-1] > 0 else None),
                         endpoint_error_m=float(en[-1]), endpoint_pred_memory_m=float(np.linalg.norm(pm[-1]))))
    rng = np.random.default_rng(0)
    def contrast(key_a, key_b):
        by = {}
        for r in rows:
            by.setdefault(r["episode"], []).append(r[key_a] - r[key_b])
        diff = np.array([np.mean(v) for v in by.values()]); boots = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(a.boot)]
        return dict(n_sources=len(diff), mean=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    memory_better_excluding_zero=bool(np.percentile(boots, 97.5) < 0))
    out = dict(frozen_model=frozen, n_locked_rows=len(rows), whole_continuation_coverage=float(np.mean(cov_all)) if cov_all else None,
               median_endpoint_bound_over_error=float(np.median(ratios)) if ratios else None,
               prediction_4=dict(coverage_ge_0_9=bool(cov_all and np.mean(cov_all) >= 0.9), median_ratio_le_10=bool(ratios and np.median(ratios) <= 10)),
               contrast_memory_minus_geometric=contrast("int_err_memory", "int_err_geometric"), contrast_memory_minus_neutral=contrast("int_err_memory", "int_err_neutral"), rows=rows)
    js = json.dumps(out, indent=1)
    if a.out:
        a.out.write_text(js)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
