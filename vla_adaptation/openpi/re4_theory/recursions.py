"""re4 theory plan, Parts 3, 4 and 5 on LIBERO telemetry (prereg_records/PREREG_RE4T_3_4_5_TELEMETRY.md).
Reads a run's telemetry.jsonl (per-step raw_action, correction, command, measured, r, f_hat, f_true,
attenuation, ...) plus the plant (fit_plant on the healthy log), M, and the Part 1/2.2 constants.

4.1 onset:   residual after onset vs the FIR partial-sum prediction  r_k = (sum_{l<=k-k0} W_l) f   (K-step transient)
4.2 envelope: measured estimate error |f_hat - f_true| on corrected channels vs the propagated bound
             E_{k+1} <= (1 - g s_k) E_k + g s_k ||M^-1|| ||eps_k||,  eps_k = r_k - M (f_true - f_hat_k)   (measured noise)
4.3 fixed pts: attenuated settle predicted from the logged residual distribution, mean_k s_rho(r_k) M^-1 r_k, for rho grid
4.4 drift:   ramp cell: lag (f_true - f_hat) after the transient vs nu / alpha_min, alpha = gamma s_k
3   tube:    R_{k+1} = lam R_k + L |f_true - c_k| + eta, lam = Part 1 (ee metric), L = Part 1 L_hat, eta = healthy residual p90;
             X_k = |r_k| in the Part 2.2 metric (the nominal-replay residual); coverage and violation classes
5   small gain: regress |M dfhat_k| on (E_k, X_k) -> (eps0, kE, kX); test a c > b kX with a = 1 - lam, b = L, c = gamma
"""
import argparse, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import adaptive_law as AL


def load_tel(path):
    eps = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("type") != "step" or d.get("phase") != "rollout":
            continue
        eps.setdefault((d["arm"], d["episode"]), []).append(d)
    return {k: sorted(v, key=lambda x: x["t"]) for k, v in eps.items()}


def arr(ep, key):
    return np.array([np.asarray(s[key], float) if s.get(key) is not None else np.full(6, np.nan) for s in ep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path); ap.add_argument("--part", required=True, choices=["4.1", "4.2", "4.3", "4.4", "3", "5"])
    ap.add_argument("--log", default=str(HERE.parent / "results/phase05/error_signal_so3.json"))
    ap.add_argument("--openloop", default=str(HERE.parent / "results/phase05/openloop_so3.json"))
    ap.add_argument("--contraction", default=str(HERE.parent / "results/re4_theory/1_contraction/summary.json"))
    ap.add_argument("--cert", default=str(HERE.parent / "results/re4_theory/2_metric/metric_certificate_panda.json"))
    ap.add_argument("--gamma", type=float, default=0.08); ap.add_argument("--corr", default="3,4,5"); ap.add_argument("--onset", type=int, default=40)
    ap.add_argument("--out", type=pathlib.Path); a = ap.parse_args()
    tel = load_tel(a.run_dir / "telemetry.jsonl"); ad = {k: v for k, v in tel.items() if k[0] == "adaptive"}
    W = AL.fit_plant(a.log); K = AL.K_FIR; M = np.array(json.loads(pathlib.Path(a.openloop).read_text())["M"]); Mi = np.linalg.pinv(M)
    corr = [int(x) for x in a.corr.split(",")]; out = dict(run=str(a.run_dir), part=a.part, episodes=len(ad))
    if a.part == "4.1":
        # residual step after onset vs FIR partial sums of the fault step
        rows = []
        for (arm, e), ep in ad.items():
            R = arr(ep, "r"); F = arr(ep, "f_true"); t = np.array([s["t"] for s in ep])
            k0 = int(np.argmax(np.linalg.norm(F, axis=1) > 0)) if np.any(np.linalg.norm(F, axis=1) > 0) else None
            if k0 is None or k0 + K + 4 >= len(ep):
                continue
            f = F[k0]; base = R[max(0, k0 - 10):k0].mean(axis=0)   # pre-onset residual level
            pred = np.array([[W[i, :min(j, K) + 1].sum() * f[i] for i in range(6)] for j in range(K + 4)])
            meas = R[k0:k0 + K + 4] - base
            rows.append(dict(episode=e, onset=k0, meas=meas.tolist(), pred=pred.tolist()))
        if rows:
            meas = np.array([r["meas"] for r in rows]); pred = np.array([r["pred"] for r in rows])
            peak = np.abs(pred[:, -1, corr]).mean(); err = np.sqrt(np.mean((meas[:, :, corr] - pred[:, :, corr]) ** 2, axis=(0, 2)))
            # secondary (not registered): the episode-MEAN response per channel against the FIR partial sums, which
            # removes the per-step plant noise the registered per-episode RMS is dominated by
            mm = meas.mean(axis=0); pm = pred.mean(axis=0); per_ch = {}
            for i in corr:
                pk = max(abs(pm[-1, i]), 1e-9)
                per_ch[str(i)] = dict(mean_meas=mm[:, i].tolist(), pred=pm[:, i].tolist(),
                                      rms_err_first_K_over_peak=float(np.sqrt(np.mean((mm[:K, i] - pm[:K, i]) ** 2)) / pk),
                                      rms_err_after_K_over_peak=float(np.sqrt(np.mean((mm[K:, i] - pm[K:, i]) ** 2)) / pk),
                                      plateau_meas_over_pred=float(mm[-3:, i].mean() / pm[-1, i]) if abs(pm[-1, i]) > 1e-9 else None,
                                      fir_dc_gain=float(W[i].sum()), M_diag=float(M[i, i]))
            out.update(mean_response_by_channel=per_ch)
            out.update(n_onsets=len(rows), rms_err_over_peak_by_step=(err / max(peak, 1e-9)).tolist(),
                       rms_err_first_K_over_peak=float(err[:K].mean() / max(peak, 1e-9)), rms_err_after_K_over_peak=float(err[K:].mean() / max(peak, 1e-9)),
                       prediction_4_1=dict(within_25pct_first_K=bool(err[:K].mean() / max(peak, 1e-9) <= 0.25), within_10pct_after_K=bool(err[K:].mean() / max(peak, 1e-9) <= 0.10)))
    elif a.part == "4.2":
        cov, cov15, ball = [], [], []
        for (arm, e), ep in ad.items():
            R = arr(ep, "r"); F = arr(ep, "f_true"); Fh = arr(ep, "f_hat"); Fb = arr(ep, "f_hat_before")
            att = np.array([s.get("attenuation") if s.get("attenuation") is not None else 0.0 for s in ep], float)
            E = np.linalg.norm((Fh - F)[:, corr], axis=1); eps = np.linalg.norm((R - (M @ (F - Fb).T).T)[:, corr], axis=1)
            B = np.zeros(len(ep)); B[0] = np.linalg.norm(F[0, corr])
            for k in range(len(ep) - 1):
                gs = a.gamma * att[k]; B[k + 1] = (1 - gs) * B[k] + gs * np.linalg.norm(Mi[np.ix_(corr, corr)], 2) * eps[k]
            ok = E <= B * 1.0001 + 1e-9; cov.append(ok.mean()); cov15.append(ok[15:].mean() if len(ok) > 15 else np.nan)
        out.update(coverage_all=float(np.mean(cov)), coverage_after_15=float(np.nanmean(cov15)), prediction_4_2=dict(coverage_after_15_ge_0_9=bool(np.nanmean(cov15) >= 0.9)))
    elif a.part == "4.3":
        R = np.concatenate([arr(ep, "r") for ep in ad.values()]); Z = (Mi @ R.T).T
        settle = {}
        for rho in (0.05, 0.15, 0.50):
            s = 1.0 / (1.0 + np.linalg.norm(R, axis=1) ** 2 / rho ** 2); settle[f"{rho:.2f}"] = (s[:, None] * Z).mean(axis=0)[corr].tolist()
        stored = {"0.05": [0.027, 0.011, 0.027], "0.15": [0.044, 0.020, 0.044], "0.50": [0.048, 0.022, 0.049]}
        meas_settle = np.median(np.array([arr(ep, "f_hat")[-50:].mean(axis=0)[corr] for ep in ad.values() if len(ep) > 50]), axis=0).tolist()
        truth = np.median(np.concatenate([arr(ep, "f_true") for ep in ad.values()]), axis=0)[corr]
        ratio = [float(m / t) if abs(t) > 1e-12 else None for m, t in zip(meas_settle, truth)]
        out.update(predicted_settle_from_logged_residuals=settle, stored_ablation_settles=stored, measured_settle_this_run=meas_settle,
                   within_20pct={r: [bool(abs(p - s0) <= 0.2 * abs(s0)) for p, s0 in zip(settle[r], stored[r])] for r in settle},
                   truth_on_corrected_channels=truth.tolist(), measured_settle_over_truth=ratio,
                   # T4 (innovation law): the settle should sit within 10 % of the truth on rx and rz (channels 0 and 2 of corr)
                   prediction_4_3_innovation=dict(within_10pct_of_truth_rx_rz=bool(all(r is not None and abs(r - 1) <= 0.10 for r in (ratio[0], ratio[2])))))
    elif a.part == "4.4":
        lags, alphas = [], []
        for (arm, e), ep in ad.items():
            F = arr(ep, "f_true"); Fh = arr(ep, "f_hat"); att = np.array([s.get("attenuation") or 0.0 for s in ep], float)
            live = np.linalg.norm(F, axis=1) > 0
            if live.sum() < 40:
                continue
            idx = np.where(live)[0][20:]; lag = np.abs((F - Fh)[idx][:, corr]).mean(); lags.append(lag); alphas.append(a.gamma * att[idx].min())
        nu = 0.05 / 60.0
        out.update(measured_lag_median=float(np.median(lags)) if lags else None, alpha_min_median=float(np.median(alphas)) if alphas else None,
                   nu_over_alpha=float(nu / np.median(alphas)) if alphas else None,
                   prediction_4_4=dict(within_2x=bool(lags and 0.5 <= np.median(lags) / (nu / np.median(alphas)) <= 2.0)))
    elif a.part == "3":
        con = json.loads(pathlib.Path(a.contraction).read_text()); cert = json.loads(pathlib.Path(a.cert).read_text())
        lam = np.median([con["metrics"]["eeP"][f"joint:{m}:free"]["median"] for m in (0.005, 0.01, 0.02)])
        L = float(np.mean([v["median"] for k, v in con["input_sensitivity_ee_m_per_unit"].items() if int(k) < 3])) / 0.05   # m per unit -> normalised (5 cm per unit)
        P = np.array([p["P"] for p in cert["per_axis"]], float)
        eta = np.percentile(np.linalg.norm(np.concatenate([arr(ep, "r") for ep in tel.values() if True])[:, :3] * np.sqrt(P[:3]), axis=1), 90)
        covs, viol, ratio_end, ratio_all, Xmax = [], [], [], [], []
        for (arm, e), ep in ad.items():
            R = arr(ep, "r"); F = arr(ep, "f_true"); C = np.array([np.asarray(s["correction"], float)[:6] for s in ep])
            X = np.linalg.norm(R[:, :3] * np.sqrt(P[:3]), axis=1); Rb = np.zeros(len(ep)); Rb[0] = eta
            for k in range(len(ep) - 1):
                Rb[k + 1] = lam * Rb[k] + L * np.linalg.norm((F + C)[k]) + eta
            ok = X <= Rb; covs.append(ok.mean()); viol += [dict(episode=e, t=int(ep[k]["t"]), X=float(X[k]), R=float(Rb[k])) for k in np.where(~ok)[0]]
            ratio_end.append(Rb[-1] / max(X.max(), 1e-9)); ratio_all += list(Rb[1:] / np.maximum(X[1:], 1e-9)); Xmax.append(X.max())
        # tightness: with lam ~ 1 the bound grows without limit (eta per step), so coverage alone says nothing;
        # report the bound-to-measurement ratio and the bound's growth against the episode's largest X.
        out.update(lam=float(lam), L=float(L), eta=float(eta), coverage=float(np.mean(covs)), violations=len(viol), violation_examples=viol[:10],
                   tightness=dict(median_bound_over_X=float(np.median(ratio_all)), p10_bound_over_X=float(np.percentile(ratio_all, 10)),
                                  median_final_bound_over_episode_max_X=float(np.median(ratio_end)), median_episode_max_X=float(np.median(Xmax)),
                                  bound_per_step_growth_at_zero_error=float(eta), steps_until_bound_exceeds_10x_eta=int(np.ceil(9 / (1 - lam))) if lam < 1 else None,
                                  vacuous=bool(np.median(ratio_end) > 10)),
                   prediction_3=dict(coverage_ge_0_9=bool(np.mean(covs) >= 0.9)))
    elif a.part == "5":
        con = json.loads(pathlib.Path(a.contraction).read_text())
        lam = np.median([con["metrics"]["eeP"][f"joint:{m}:free"]["median"] for m in (0.005, 0.01, 0.02)]); b = float(np.mean([v["median"] for k, v in con["input_sensitivity_ee_m_per_unit"].items() if int(k) < 3])) / 0.05
        Xs, Es, Ws = [], [], []
        for (arm, e), ep in ad.items():
            R = arr(ep, "r"); F = arr(ep, "f_true"); Fh = arr(ep, "f_hat"); Fb = arr(ep, "f_hat_before")
            Xs.append(np.linalg.norm(R, axis=1)); Es.append(np.linalg.norm((Fb - F)[:, corr], axis=1)); Ws.append(np.linalg.norm((M @ (Fh - Fb).T).T, axis=1))
        X, E, Wn = map(np.concatenate, (Xs, Es, Ws)); A = np.c_[np.ones_like(X), E, X]; coef = np.linalg.lstsq(A, Wn, rcond=None)[0]
        eps0, kE, kX = map(float, coef); aa = 1 - lam; c = a.gamma
        out.update(eps0=eps0, kE=kE, kX=kX, a=float(aa), b=b, c=c, small_gain_holds=bool(aa * c > b * kX), margin=float(aa * c - b * kX))
    txt = json.dumps(out, indent=1); print(txt)
    if a.out:
        a.out.write_text(txt)


if __name__ == "__main__":
    main()
