"""Premise test: is the torque-fault disturbance state-dependent within an episode?
Adaptive arm uses a CONSTANT basis Phi=E, so f_hat(t) is the estimator's running read of an
action-equivalent disturbance. If the true disturbance d(q) moves with configuration, a
converged f_hat keeps moving; if d is constant, f_hat goes flat (up to estimator noise)."""
import gzip, json, numpy as np, sys, glob, os, pathlib
TEL = pathlib.Path(__file__).resolve().parents[3] / "results" / "sweep" / "telemetry"
T0 = 60            # steps allowed for convergence
rows = []
for fn in sorted(glob.glob(str(TEL / "spatial_j*.jsonl.gz"))):
    j = os.path.basename(fn).split("_j")[1].split(".")[0]
    ep = {}
    with gzip.open(fn, "rt") as f:
        for l in f:
            r = json.loads(l)
            if r.get("type") != "step" or r.get("phase") == "warmup": continue
            k = (r["arm"], r["episode"])
            ep.setdefault(k, []).append((r["t"], r["f_hat"][:6], r["r"][:6], r.get("done")))
    within, between_means, lag_frac, frozen_lvl = [], [], [], []
    for (arm, e), seq in ep.items():
        seq.sort(); F = np.array([s[1] for s in seq]); R = np.array([s[2] for s in seq])
        if len(seq) < T0 + 30: continue
        if arm == "adaptive":
            Fs = F[T0:]
            m = Fs.mean(0); sd = Fs.std(0)
            within.append(sd); between_means.append(m)
        else:
            # frozen: low-passed residual level = fault signal per step, 10-step boxcar
            k = 10; lp = np.array([R[i:i+k].mean(0) for i in range(T0, len(R)-k)])
            frozen_lvl.append(np.abs(lp.mean(0)))
    W = np.array(within); Bm = np.array(between_means)
    amp = np.abs(Bm).mean(0)                  # typical |converged estimate| per channel
    rows.append((j, len(W), W.mean(0), amp, Bm.std(0), np.array(frozen_lvl).mean(0) if frozen_lvl else None))
np.set_printoptions(precision=3, suppress=True)
print("channels: x y z rx ry rz  (action-equivalent units; correction cap .25)\n")
for j, n, w, amp, btw, fz in rows:
    dom = int(np.argmax(amp))
    print(f"joint {j}  ({n} adaptive episodes long enough)")
    print(f"   |mean f_hat| per ch        {amp}")
    print(f"   within-episode std        {w}")
    print(f"   between-episode std       {btw}")
    print(f"   dominant ch {dom}: within/|mean| = {w[dom]/amp[dom]:.2f}   between/|mean| = {btw[dom]/amp[dom]:.2f}")
