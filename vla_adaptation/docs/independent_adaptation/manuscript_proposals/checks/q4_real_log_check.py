"""Q4 on the stored Panda calibration log: refit the deployed FIR (K=6) and ARX(1) exactly as
openpi/adaptive_law.fit_plant does (per axis, ridge 1e-2 incl. intercept, motion in controller units),
report (1) fitted DC gains vs the probed M (record 50: 0.103 / 0.099 on r_y vs 0.276),
(2) the predicted fixed points in the two regimes, against the measured settles (T1 NT 41 %, T4 innovation 45 %, ARX NT 32 %),
(3) how far the two FITTED residuals are from the exact-model filter relation r_A,k+1 = r_F,k+1 - a_hat r_F,k
    (the misspecification part), per channel.
CPU only; reads results/phase05/error_signal_so3.json and results/phase05/openloop_so3.json.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path("/home/fengze/vla-adaptation")
OUT_FILE = pathlib.Path(__file__).with_suffix(".json")
OUT = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]); K = 6; LAM = 1e-2
CH = ["x", "y", "z", "rx", "ry", "rz"]

d = json.loads((ROOT / "results/phase05/error_signal_so3.json").read_text())
recs = d["records"] if isinstance(d, dict) else d
A = np.array(recs[0]["raw_a"]); D = np.array(recs[0]["raw_d"]); lens = recs[0]["ep_len"]
M = np.array(json.loads((ROOT / "results/phase05/openloop_so3.json").read_text())["M"])


def fit(ar):
    W = []
    for i in range(6):
        X, Y, o = [], [], 0
        for L in lens:
            a, y = A[o:o + L], D[o:o + L, i] / OUT[i]; o += L
            for t in range(max(K, ar), L):
                win = a[t - K:t + 1][::-1][:, i]
                arf = y[t - ar:t][::-1] if ar else np.zeros(0)
                X.append(np.concatenate([win, arf, [1.0]])); Y.append(y[t])
        X, Y = np.array(X), np.array(Y)
        W.append(np.linalg.solve(X.T @ X + LAM * np.eye(X.shape[1]), X.T @ Y))
    return np.array(W)


W_f = fit(0); W_a = fit(1)
dc_f = W_f[:, :K + 1].sum(1); a_hat = W_a[:, K + 1]; dc_a = W_a[:, :K + 1].sum(1) / (1 - a_hat)
rho_f = dc_f / np.diag(M); rho_a = dc_a / np.diag(M)
res = dict(channels=CH, M_diag=np.diag(M).tolist(), fir_dc=dc_f.tolist(), arx_a_hat=a_hat.tolist(), arx_dc=dc_a.tolist(), rho_fir=rho_f.tolist(), rho_arx=rho_a.tolist())
print("channel   M      FIR DC  rho_F   ARX a^   ARX DC  rho_A   | regime(a) 1/(2-rho): F     A   | regime(b) rho: F     A")
for i in range(6):
    print(f"{CH[i]:4s}  {M[i,i]:6.3f}  {dc_f[i]:6.3f}  {rho_f[i]:5.3f}  {a_hat[i]:6.3f}  {dc_a[i]:6.3f}  {rho_a[i]:5.3f}   |  {1/(2-rho_f[i]):5.3f} {1/(2-rho_a[i]):5.3f} |  {rho_f[i]:5.3f} {rho_a[i]:5.3f}")
res["predicted_fixed_points"] = {CH[i]: dict(fir_a=1 / (2 - rho_f[i]), fir_b=rho_f[i], arx_a=1 / (2 - rho_a[i]), arx_b=rho_a[i]) for i in range(6)}
res["measured_settles_rotation"] = dict(T1_NT_fir=[0.87, 0.41, 0.88], T4_innov_fir=[0.95, 0.45, 0.98], ARX_NT=[0.94, 0.32, 0.97], source="record 49/50; part_4.3.json")

# residual filter relation on the fitted models (misspecified): compare r_A,k+1 with r_F,k+1 - a_hat r_F,k per channel, per episode
rel = {}
o = 0
rF_all = {i: [] for i in range(6)}; rA_all = {i: [] for i in range(6)}; rpred_all = {i: [] for i in range(6)}
for L in lens:
    a, y = A[o:o + L], D[o:o + L] / OUT; o += L
    for i in range(6):
        rF = np.full(L, np.nan); rA = np.full(L, np.nan)
        for t in range(K, L):
            win = a[t - K:t + 1][::-1][:, i]
            rF[t] = y[t, i] - (W_f[i, :K + 1] @ win + W_f[i, -1])
            if t >= max(K, 1):
                rA[t] = y[t, i] - (W_a[i, :K + 1] @ win + W_a[i, K + 1] * y[t - 1, i] + W_a[i, -1])
        for t in range(K + 1, L):
            rF_all[i].append(rF[t]); rA_all[i].append(rA[t]); rpred_all[i].append(rF[t] - a_hat[i] * rF[t - 1])
for i in range(6):
    rA = np.array(rA_all[i]); rp = np.array(rpred_all[i]); rF = np.array(rF_all[i])
    mis = rA - rp
    rel[CH[i]] = dict(std_rA=float(rA.std()), std_filtered_rF=float(rp.std()), std_misspec_part=float(mis.std()), corr=float(np.corrcoef(rA, rp)[0, 1]),
                      frac_var_explained=float(1 - mis.var() / rA.var()), std_rF=float(rF.std()), lag1_autocorr_rF=float(np.corrcoef(rF[1:], rF[:-1])[0, 1]), lag1_autocorr_rA=float(np.corrcoef(rA[1:], rA[:-1])[0, 1]))
    print(f"{CH[i]:3s}: std r_A {rA.std():.4f}, std (r_F,k+1 - a^ r_F,k) {rp.std():.4f}, corr {rel[CH[i]]['corr']:.3f}, misspec part std {mis.std():.4f} ({100*rel[CH[i]]['frac_var_explained']:.0f}% of var(r_A) explained by the filtered FIR residual); lag-1 autocorr r_F {rel[CH[i]]['lag1_autocorr_rF']:.2f}, r_A {rel[CH[i]]['lag1_autocorr_rA']:.2f}")
res["fitted_residual_filter_relation"] = rel
OUT_FILE.write_text(json.dumps(res, indent=1)); print("wrote", OUT_FILE)
