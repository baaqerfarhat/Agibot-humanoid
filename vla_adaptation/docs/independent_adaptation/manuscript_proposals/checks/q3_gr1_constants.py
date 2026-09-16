"""Q3: instantiate the small-gain corollary (iclr2027/theory_main.tex, uni:smallgain) on GR1 from stored data.

Comparison matrix B = [[sigma, beta],[t, q]] with the innovation law: t = alpha*l, q = 1 - alpha + alpha*m.
Same-command joint-position metric (M_c = I on the corrected joints 7..13; units rad):
  sigma = max certified pole (fold max), beta = (1 - min pole) * ||D L|| with D = L = I on the corrected joints (offset fault on the same joints).
Measured here from stored logs (no simulator):
  * the deployed GR1 predictor (per-joint FIR on POSITION, K = 6, plain LS with intercept, as openpi/gr1_adapt.fit_plant)
    fitted on the 6-episode screen log -> DC gain per joint vs the probed M (openloop_arms_clean.json) -> rho_j, m = ||I - M^-1 D||_2
  * epsilon: the observation mismatch w = M^-1 r - f on joints 7..13, on held-out healthy episodes (f = 0) and on the
    frozen faulted logs (f = 0.05, 0.15 on 7..13; u and q logged, no correction), after the FIR's onset transient
  * the onset transient of z = M^-1 r under the fault against (1 - lam^k) f
  * l in the SAME-COMMAND sense only: regress ||w_{k+1}|| on X^sc_k = ||q_k - q^sc_k||, q^sc = certified first-order replay of u + f
  * the innovation law's gain range alpha_k = gamma s(||e||), from the faulted logs' residual norms (gamma = .08, rho_n = .11, dead = .013)
  * from results/gr1/p2p_right010_cont.json (innovation law, +0.10 on 7..13): E_k = ||f^_k - f|| per step, settle and time-to-80 %
Then: the corollary's condition, Perron root, weights p, lambda, and the threshold on l that the missing healthy-policy coupling must satisfy.
"""
import json, pathlib, collections
import numpy as np

ROOT = pathlib.Path("/home/fengze/vla-adaptation")
OUT_FILE = pathlib.Path(__file__).with_suffix(".json")
NJ, K, DT = 29, 6, 0.05
J = list(range(7, 14)); GAMMA, DEAD, RHO_N, CLIP = 0.08, 0.013, 0.11, 0.2

cert = json.loads((ROOT / "results/re4_theory/2_metric/metric_certificate_gr1.json").read_text())
per = {p["joint"]: p for p in cert["per_joint"]}
poles = np.array([per[j]["pole"] for j in J]); pole_min_fold = min(per[j]["pole_fold_min"] for j in J); pole_max_fold = max(per[j]["pole_fold_max"] for j in J)
a_srv = np.array([per[j]["a"] for j in J])
M = np.array(json.loads((ROOT / "results/gr1/openloop_arms_clean.json").read_text())["M"]); Mb = M[np.ix_(J, J)]; Mb_inv = np.linalg.inv(Mb)
healthy = json.loads((ROOT / "results/gr1/screen_PosttrainPnPNovelFromPlateToPlateSplitA.json").read_text())


def fit_plant(eps):
    W = np.zeros((NJ, K + 2)); r2 = np.zeros(NJ)
    for j in range(NJ):
        X, Y = [], []
        for ep in eps:
            u = np.array(ep["u"])[:, j]; qq = np.array(ep["q"])[:, j]
            for t in range(K, len(u)):
                X.append(np.r_[u[t - K:t + 1][::-1], 1.0]); Y.append(qq[t])
        X, Y = np.array(X), np.array(Y)
        w, *_ = np.linalg.lstsq(X, Y, rcond=None); W[j] = w
        ss = ((Y - X @ w) ** 2).sum(); st = ((Y - Y.mean()) ** 2).sum(); r2[j] = 1 - ss / max(st, 1e-12)
    return W, r2


def residuals(ep, W):
    """Replicate gr1_adapt.episode: hist seeded with q0 (K+1 copies), appendleft(a_corr) after each step, residual on q1."""
    u = np.array(ep["u"], float); q = np.array(ep["q"], float); T = len(u)
    q0 = q[0]  # the log stores q AFTER each step; the pre-step initial q is not stored -> use q[0] as the seed (one-step approximation for the first K steps only)
    hist = collections.deque([q0.copy()] * (K + 1), maxlen=K + 1)
    R = np.zeros((T, NJ))
    for t in range(T):
        hist.appendleft(u[t].copy())
        H = np.array(hist)
        pred = np.array([W[j, :K + 1] @ H[:, j] + W[j, -1] for j in range(NJ)])
        R[t] = q[t] - pred
    return R


res = {}
W_all, r2_all = fit_plant(healthy)
dc = W_all[:, :K + 1].sum(1)
rho = dc[J] / np.diag(Mb); Rmat = Mb_inv @ np.diag(dc[J]); m_spec = float(np.linalg.norm(np.eye(7) - Rmat, 2)); m_max = float(np.abs(1 - rho).max())
res["predictor"] = dict(fitted_on_episodes=len(healthy), r2_train_7_13=r2_all[J].tolist(), dc_gain_7_13=dc[J].tolist(), M_diag_7_13=np.diag(Mb).tolist(), rho_7_13=rho.tolist(), m_specnorm=m_spec, m_max_abs=m_max, first_tap_7_13=W_all[J, 0].tolist())
print("GR1 FIR-on-position, joints 7..13: train R2", np.round(r2_all[J], 3)); print("  DC gain", np.round(dc[J], 4), " M diag", np.round(np.diag(Mb), 4)); print("  rho = DC/M", np.round(rho, 4), " -> m = ||I - M^-1 D||_2 =", round(m_spec, 4), " max|1-rho| =", round(m_max, 4))
# LOO fits: DC spread and held-out healthy mismatch
loo_dc = []; eps_h = []
for h in range(len(healthy)):
    W_h, _ = fit_plant([e for i, e in enumerate(healthy) if i != h]); loo_dc.append(W_h[J, :K + 1].sum(1))
    R = residuals(healthy[h], W_h); w = (Mb_inv @ R[:, J].T).T
    eps_h.append(np.linalg.norm(w[K + 1:], axis=1))
loo_dc = np.array(loo_dc); eps_h = np.concatenate(eps_h)
res["predictor"]["loo_dc_min_7_13"] = loo_dc.min(0).tolist(); res["predictor"]["loo_dc_max_7_13"] = loo_dc.max(0).tolist()
res["epsilon_healthy_heldout"] = dict(n=int(len(eps_h)), median=float(np.median(eps_h)), p90=float(np.percentile(eps_h, 90)), max=float(eps_h.max()))
print("  LOO DC gain range per joint:", np.round(loo_dc.min(0), 3), "-", np.round(loo_dc.max(0), 3))
print(f"epsilon (healthy, held-out, ||M^-1 r|| over 7..13, rad): median {np.median(eps_h):.4f}, p90 {np.percentile(eps_h,90):.4f}, max {eps_h.max():.4f}")

# faulted frozen logs
fl = {}
for name, fval in (("faulted_log_right005.json", 0.05), ("faulted_log_right015.json", 0.15)):
    eps = json.loads((ROOT / "results/gr1" / name).read_text()); f = fval * np.ones(7)
    settle, wn_all, wn_late, X_all, w_next, onset = [], [], [], [], [], []
    s_vals, e_norms = [], []
    for ep in eps:
        R = residuals(ep, W_all); z = (Mb_inv @ R[:, J].T).T; w = z - f
        T = len(z); u = np.array(ep["u"], float)[:, J]; q = np.array(ep["q"], float)[:, J]
        settle.append(z[50:].mean(0) / fval if T > 100 else z[20:].mean(0) / fval)
        wn = np.linalg.norm(w, axis=1); wn_all.append(wn[K + 1:]); wn_late.append(wn[20:])
        onset.append(z[:40] / fval)
        # same-command certified replay of u + f from q[0]
        qs = np.zeros_like(q); qs[0] = q[0]
        for k in range(T - 1):
            qs[k + 1] = qs[k] + a_srv * DT * (u[k + 1] + f - qs[k])   # log stores q after step t and u at step t; q[t+1] responds to u[t+1] ... see note in report
        X = np.linalg.norm(q - qs, axis=1)
        X_all.append(X[20:-1]); w_next.append(wn[21:])
        # innovation-law gain along the frozen log (what alpha would be with f_hat = 0): e = r - M*0 = r on the selected joints
        rn = np.linalg.norm(R[:, J], axis=1); s = np.where(rn < DEAD, 0.0, 1.0 / (1.0 + (rn / RHO_N) ** 2)); s_vals.append(s); e_norms.append(rn)
    settle = np.array(settle); wn_all = np.concatenate(wn_all); wn_late = np.concatenate(wn_late); X_all = np.concatenate(X_all); w_next = np.concatenate(w_next)
    Areg = np.c_[np.ones_like(X_all), X_all]; coef, *_ = np.linalg.lstsq(Areg, w_next, rcond=None)
    ss_res = ((w_next - Areg @ coef) ** 2).sum(); ss_tot = ((w_next - w_next.mean()) ** 2).sum()
    on = np.mean([o for o in onset if len(o) == 40], axis=0) if all(len(o) == 40 for o in onset) else np.mean([o[:40] for o in onset], axis=0)
    lam_bar = float(poles.mean()); pred_onset = (1 - lam_bar ** np.arange(1, 41))
    partial = np.cumsum(W_all[J, :K + 1], axis=1) / dc[J][:, None]
    s_vals = np.concatenate(s_vals); e_norms = np.concatenate(e_norms)
    fl[name] = dict(f=fval, episodes=len(eps), settle_ratio_per_joint_median=np.median(settle, 0).tolist(), settle_ratio_mean=float(np.median(settle)),
                    eps_fault_all=dict(median=float(np.median(wn_all)), p90=float(np.percentile(wn_all, 90))), eps_fault_after20=dict(median=float(np.median(wn_late)), p90=float(np.percentile(wn_late, 90)), max=float(wn_late.max())),
                    onset_mean_z_over_f_first10=on[:10].mean(1).tolist(), onset_pred_1_minus_lam_k_first10=pred_onset[:10].tolist(), fitted_partial_sums_first7_mean=partial.mean(0).tolist(),
                    l_sc=dict(intercept=float(coef[0]), slope=float(coef[1]), r2=float(1 - ss_res / ss_tot), X_median=float(np.median(X_all)), X_p90=float(np.percentile(X_all, 90)), n=int(len(X_all))),
                    alpha_from_frozen_residual=dict(alpha_median=float(GAMMA * np.median(s_vals)), alpha_min=float(GAMMA * s_vals[s_vals > 0].min()) if np.any(s_vals > 0) else 0.0, frac_gate_closed=float((s_vals == 0).mean()), e_norm_median=float(np.median(e_norms))))
    print(f"\n{name} (f = {fval} rad on 7..13, {len(eps)} episodes, frozen):")
    print("  settle z/f per joint (median over eps, steps>=50):", np.round(np.median(settle, 0), 3))
    print(f"  eps under fault: all steps median {np.median(wn_all):.4f} p90 {np.percentile(wn_all,90):.4f}; after step 20 median {np.median(wn_late):.4f} p90 {np.percentile(wn_late,90):.4f} max {wn_late.max():.4f}")
    print("  onset mean z/f, k=1..10:", np.round(on[:10].mean(1), 3)); print("  (1 - lam^k), k=1..10  :", np.round(pred_onset[:10], 3)); print("  fitted FIR partial sums (mean over joints), j=0..6:", np.round(partial.mean(0), 3))
    print(f"  same-command coupling: ||w_k+1|| = {coef[0]:.4f} + {coef[1]:.4f} X^sc_k (R2 {1-ss_res/ss_tot:.3f}; X^sc median {np.median(X_all):.4f}, p90 {np.percentile(X_all,90):.4f})")
    print(f"  innovation gain along the frozen log (f^=0): alpha median {GAMMA*np.median(s_vals):.4f}, min {fl[name]['alpha_from_frozen_residual']['alpha_min']:.4f}, gate closed {100*(s_vals==0).mean():.0f}% of steps")
res["faulted_frozen"] = fl

# p2p innovation run: E_k
p2p = json.loads((ROOT / "results/gr1/p2p_right010_cont.json").read_text()); tr = p2p["arms"]["adaptive"]["traj"]; f = 0.10 * np.ones(7)
E = [np.linalg.norm(np.array(t, float)[:, J] - f, axis=1) for t in tr]
Lmin = min(len(e) for e in E); Emat = np.array([e[:Lmin] for e in E])
t80 = [int(np.argmax(e <= 0.2 * np.linalg.norm(f))) if np.any(e <= 0.2 * np.linalg.norm(f)) else None for e in E]
settle_p2p = [np.array(t, float)[100:200, J].mean(0) / 0.10 for t in tr if len(t) >= 200]
res["p2p_innov"] = dict(episodes=len(tr), lens=[len(t) for t in tr], E0=float(np.median(Emat[:, 0])), E_median_at=dict({str(k): float(np.median(Emat[:, k])) for k in (10, 20, 50, 100, 150) if k < Lmin}), time_to_80pct=t80,
                        settle_100_200_per_joint_median=np.median(settle_p2p, 0).tolist() if settle_p2p else None, geometric_rate_alpha_08_at_k=dict({str(k): float((1 - 0.08) ** k) for k in (10, 20, 50)}))
print(f"\np2p innovation (+0.10 on 7..13): E_0 median {np.median(Emat[:,0]):.3f}; E median at k=10,20,50,100:", [round(float(np.median(Emat[:, k])), 4) for k in (10, 20, 50, 100) if k < Lmin], "; time to 80%:", t80)
print("  settle f^/f per joint (steps 100-200, median over eps):", np.round(np.median(settle_p2p, 0), 3) if settle_p2p else None)

# the corollary on GR1
sigma = float(pole_max_fold); beta = float(1 - pole_min_fold)
res["corollary"] = dict(sigma=sigma, beta=beta, sigma_source="max fold pole over joints 7..13", beta_source="(1 - min fold pole) * ||D L||, D = L = I on the corrected joints", lam16=float(sigma ** 16))
print(f"\nCorollary constants (same-command, joint-position metric, joints 7..13): sigma = {sigma:.4f}, beta = {beta:.4f}; sigma^16 (one 16-step chunk) = {sigma**16:.4f}")


def perron(B):
    s, b_, t, q = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    return (s + q + np.sqrt((s - q) ** 2 + 4 * b_ * t)) / 2


for m_used, m_label in ((m_spec, "m = ||I - M^-1 D||_2"), (0.0, "m = 0 (exact DC)")):
    thr = (1 - sigma) * (1 - m_used) / beta
    print(f"  with {m_label} = {m_used:.4f}: condition beta*t < (1-sigma)(1-q) with t = alpha*l, q = 1 - alpha(1-m) <=> l < {thr:.4f} (any constant alpha)")
    for l_used in (0.0, 0.1, 0.3, thr * 0.999):
        for alpha in (0.08, 0.03):
            B = np.array([[sigma, beta], [alpha * l_used, 1 - alpha * (1 - m_used)]])
            r = perron(B); ok = r < 1
            p = np.linalg.solve(np.eye(2) - B.T, np.ones(2)) if ok else None
            lam_st = float(np.max((B.T @ p) / p)) if ok else None
            res["corollary"].setdefault("table", []).append(dict(m=m_used, l=float(l_used), alpha=alpha, rho_B=float(r), holds=bool(ok), p=(p.tolist() if ok else None), lambda_storage=lam_st))
            print(f"     l = {l_used:.3f}, alpha = {alpha}: rho(B) = {r:.4f} {'holds' if ok else 'FAILS'}" + (f", p = ({p[0]:.2f}, {p[1]:.2f}), lambda = {lam_st:.4f}" if ok else ""))
    res["corollary"][f"l_threshold_{'m_spec' if m_used > 0 else 'm0'}"] = float(thr)
# uniform (common weight) version over the gate's range alpha in [alpha_min, gamma]
for a_min in (0.012, 0.033, 0.079):
    thr_u = (1 - sigma) * (1 - m_spec) / beta * a_min / GAMMA
    res["corollary"].setdefault("uniform_threshold", {})[str(a_min)] = float(thr_u)
    print(f"  common storage function over alpha in [{a_min}, {GAMMA}]: needs l < {thr_u:.4f}")
# steady state of the comparison recursion b = B b + d (constant coefficients), d = (chi, alpha*eps + nu), nu = 0 (constant fault):
# chi = servo model remainder (norm of the certified one-step RMSE over 7..13) + 0 unmatched (offset fault on the corrected joints, projection inactive)
chi = float(np.linalg.norm([per[j]["one_step_rmse"] for j in J]))
eps_med, eps_p90 = res["epsilon_healthy_heldout"]["median"], res["epsilon_healthy_heldout"]["p90"]
E_late = [float(np.median(np.linalg.norm(np.array(t, float)[100:200, J] - f, axis=1))) for t in tr if len(t) >= 200]
res["steady_state"] = dict(chi=chi, eps_median=eps_med, eps_p90=eps_p90, measured_E_steps_100_200_median_over_eps=float(np.median(E_late)), measured_E_steps_100_200_per_ep=E_late)
print(f"\nSteady state of b = B b + d (nu = 0, chi = {chi:.4f} rad = certified servo one-step RMSE over 7..13):")
for l_used in (0.0, 0.2):
    for eps_used, lab in ((eps_med, "eps median"), (eps_p90, "eps p90")):
        alpha = 0.08; B = np.array([[sigma, beta], [alpha * l_used, 1 - alpha * (1 - m_spec)]]); dvec = np.array([chi, alpha * eps_used])
        bstar = np.linalg.solve(np.eye(2) - B, dvec)
        res["steady_state"][f"l_{l_used}_{lab.replace(' ', '_')}"] = dict(X_star=float(bstar[0]), E_star=float(bstar[1]))
        print(f"  l = {l_used}, {lab} = {eps_used:.4f}: X* = {bstar[0]:.4f} rad, E* = {bstar[1]:.4f} rad (alpha cancels in E* when l = 0: E* = eps/(1-m))")
print(f"  measured ||f^ - f|| over steps 100-200 (innovation, +0.10): median over episodes {np.median(E_late):.4f} rad; per episode {np.round(E_late, 4).tolist()}")
OUT_FILE.write_text(json.dumps(res, indent=1))
OUT_FILE.write_text(json.dumps(res, indent=1)); print("wrote", OUT_FILE)
