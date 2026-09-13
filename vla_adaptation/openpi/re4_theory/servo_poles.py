"""re4 theory plan, Part 2.1: per-joint first-order servo poles from the healthy logs
(prereg_records/PREREG_RE4T_2_1_SERVO_POLES.md). CPU only, stored data only.

Model per joint j:  q[k+1] - q[k] = a_j * dt * (u[k] - q[k]);  pole  lambda_j = 1 - a_j*dt;  M_c = I.
Leave-one-episode-out: fit on the other episodes, score the held-out one (one-step increment R^2 and
free-running 10-step prediction error from the true start state). Emits metric_certificate_<robot>.json.
"""
import argparse, hashlib, json, pathlib, datetime
import numpy as np

ROBOTS = {
    "gr1":   dict(log="results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json", dt=1/20, nj=29,
                  arm=list(range(7, 14)), other_arm=list(range(0, 7)), grippers=list(range(14, 26)), waist=[26, 27, 28],
                  names=None),
    "aloha": dict(log="results/aloha/healthy_log.json", dt=1/50, nj=14,
                  arm=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], other_arm=[], grippers=[6, 13], waist=[], names=None),
}
R2_MIN, KFREE = 0.5, 10


def fit_a(eps, j, dt):
    X, Y = [], []
    for e in eps:
        u, q = np.asarray(e["u"], float)[:, j], np.asarray(e["q"], float)[:, j]
        X.append(dt * (u[:-1] - q[:-1])); Y.append(q[1:] - q[:-1])
    X, Y = np.concatenate(X), np.concatenate(Y)
    a = float(X @ Y / max(X @ X, 1e-12))
    return a


def score(e, j, a, dt):
    u, q = np.asarray(e["u"], float)[:, j], np.asarray(e["q"], float)[:, j]
    dq = q[1:] - q[:-1]; pred = a * dt * (u[:-1] - q[:-1])
    ss = float(((dq - pred) ** 2).sum()); st = float(((dq - dq.mean()) ** 2).sum())
    r2 = 1 - ss / st if st > 1e-15 else float("nan")
    one_step = float(np.sqrt(np.mean((dq - pred) ** 2)))
    # free-running k-step prediction from the true state at each start
    errs = []
    for k0 in range(0, len(q) - KFREE - 1, KFREE):
        qh = q[k0]
        for k in range(k0, k0 + KFREE):
            qh = qh + a * dt * (u[k] - qh)
        errs.append(qh - q[k0 + KFREE])
    kfree = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return r2, one_step, kfree


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("robot", choices=list(ROBOTS)); ap.add_argument("--root", default=".")
    ns = ap.parse_args(); R = ROBOTS[ns.robot]; root = pathlib.Path(ns.root)
    eps = json.loads((root / R["log"]).read_text()); dt = R["dt"]; nj = R["nj"]
    per = []
    for j in range(nj):
        folds = []
        for h in range(len(eps)):
            train = [e for i, e in enumerate(eps) if i != h]; a = fit_a(train, j, dt)
            r2, os_, kf = score(eps[h], j, a, dt); folds.append(dict(fold=h, a=a, pole=1 - a * dt, r2=r2, one_step=os_, k_free=kf))
        a_all = fit_a(eps, j, dt); poles = [f["pole"] for f in folds]; r2s = [f["r2"] for f in folds]
        group = ("arm" if j in R["arm"] else "other_arm" if j in R["other_arm"] else "gripper" if j in R["grippers"] else "waist")
        in_dom = all(0 < p < 1 for p in poles) and all(r >= R2_MIN for r in r2s if r == r)
        why = ("" if in_dom else ("pole outside (0,1) on a fold" if not all(0 < p < 1 for p in poles) else f"held-out R^2 < {R2_MIN} on a fold"))
        per.append(dict(joint=j, group=group, a=a_all, pole=1 - a_all * dt, pole_fold_min=min(poles), pole_fold_max=max(poles),
                        heldout_r2_min=min(r2s), heldout_r2_median=float(np.median(r2s)),
                        one_step_rmse=float(np.median([f["one_step"] for f in folds])), k10_rmse=float(np.median([f["k_free"] for f in folds])),
                        in_domain=in_dom, reason=why))
    arm = [p for p in per if p["group"] == "arm"]
    cert = dict(schema_version="re4t-v1", robot=ns.robot, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                model="q[k+1]-q[k] = a*dt*(u[k]-q[k]) per joint; pole = 1-a*dt; metric M_c = I on the certified joints",
                dt=dt, episodes=len(eps), folds="leave-one-episode-out", r2_min=R2_MIN, k_free=KFREE,
                domain=[p["joint"] for p in per if p["in_domain"]], outside=[dict(joint=p["joint"], reason=p["reason"]) for p in per if not p["in_domain"]],
                arm_pole_range=[min(p["pole"] for p in arm), max(p["pole"] for p in arm)],
                per_joint=per, source=str(R["log"]), source_sha256=hashlib.sha256((root / R["log"]).read_bytes()).hexdigest())
    out = root / "results/re4_theory/2_metric" / f"metric_certificate_{ns.robot}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=1))
    print(f"{ns.robot}: {len(eps)} episodes, dt={dt}")
    print(f"{'j':>3} {'group':10s} {'pole':>7} {'fold min-max':>15} {'R2 min':>7} {'1-step':>8} {'10-step':>8}  in-domain")
    for p in per:
        print(f"{p['joint']:3d} {p['group']:10s} {p['pole']:7.3f} {p['pole_fold_min']:7.3f}-{p['pole_fold_max']:6.3f} {p['heldout_r2_min']:7.3f} {p['one_step_rmse']:8.5f} {p['k10_rmse']:8.5f}  {'yes' if p['in_domain'] else 'NO: ' + p['reason']}")
    print(f"arm poles {cert['arm_pole_range'][0]:.3f}-{cert['arm_pole_range'][1]:.3f}; certified joints: {cert['domain']}; wrote {out}")


if __name__ == "__main__":
    main()
