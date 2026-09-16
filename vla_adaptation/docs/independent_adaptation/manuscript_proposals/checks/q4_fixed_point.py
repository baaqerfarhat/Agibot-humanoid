"""Q4, part (ii): the estimator fixed point under a misspecified predictor depends on the model class
only through its fitted-to-probed DC-gain ratio rho = D_fit / M, in every command regime.

Plant (one channel, the Panda increment channel r_y as measured): y_{k+1} = lam y_k + (1-lam) M (u_k + f) + n_{k+1},
lam = 0.93 (Part 2.2 pole), M = 0.276 (probe), constant fault f. The pose is the integral of y.
Predictors (misspecified on purpose, DC gain D = rho M):
  FIR-K:  y_hat = sum_{j<=K} W_j u_{k-j}, W_j = the truncated exact taps rescaled to DC gain D; z = r / M
  ARX(1): y_hat = a_hat y_k + b_hat u_k with b_hat/(1-a_hat) = D, a_hat possibly != lam; z = r / ((1-a_hat) M)  (the --ar 1 rescaling)
Laws: innovation f^_{k+1} = f^_k + alpha (z - f^_k); NT f^_{k+1} = (1-g) f^_k + g s(|r|) z.
Regimes for the nominal command a_k:
  (a) fixed a (no policy feedback)                       -> closed form f^ = [f + (1-rho) a] / (2 - rho)
  (b) pose-restoring feedback: a_k = a0 + Kp (p0_k - p_k)/M with p the integrated increment (any Kp > 0)
                                                          -> closed form f^ = rho f + (1-rho) a0
  (theta) partial absorption a = a0 - theta (f - f^)       -> f^/f = [1 - (1-rho) theta] / [1 + (1-rho)(1-theta)]   (a0 = 0)
Also: ARX first-step overshoot (1-lam)/(1-a_hat) when a_hat != lam; the quasi-static stability condition |1 - alpha (2 - rho)| < 1.
"""
import json, pathlib
import numpy as np

OUT = pathlib.Path(__file__).with_suffix(".json")
lam, M, f, alpha, gam = 0.93, 0.276, 0.05, 0.08, 0.08
K = 6


def taps_fir(rho):
    w = (1 - lam) * lam ** np.arange(K + 1) * M
    return w * (rho * M) / w.sum()          # DC gain rho*M


def simulate(model, rho, regime, a_hat=None, law="innov", Kp=0.15, theta=0.0, a0=0.0, N=3000, rho_nt=0.15, seed=0):
    rng = np.random.default_rng(seed)
    if model == "fir":
        W = taps_fir(rho)
    else:
        a_hat = lam if a_hat is None else a_hat
        b_hat = rho * M * (1 - a_hat)
    y = 0.0; p = 0.0; p0 = 0.0; y0 = 0.0; fh = 0.0
    uh = np.zeros(K + 1)
    fh_hist = []
    for k in range(N):
        if regime == "a":
            a = a0
        elif regime == "b":
            a = a0 + Kp * (p0 - p) / M
        else:  # theta family: the policy absorbs a fraction theta of the remaining disturbance
            a = a0 - theta * (f - fh)
        u = a - fh
        uh = np.r_[u, uh[:-1]]
        n = 0.0  # noiseless for the fixed point; the estimator is a low-pass, noise only adds jitter
        y_new = lam * y + (1 - lam) * M * (u + f) + n
        y0 = lam * y0 + (1 - lam) * M * a0; p0 = p0 + y0   # healthy execution of the nominal command a0 (same plant, no fault): its pose is the reference
        if model == "fir":
            yhat = W @ uh
            r = y_new - yhat; z = r / M
        else:
            yhat = a_hat * y + b_hat * u
            r = y_new - yhat; z = r / ((1 - a_hat) * M)
        if law == "innov":
            fh = fh + alpha * (z - fh)
        else:
            s = 1.0 / (1 + (abs(r) / rho_nt) ** 2)
            fh = (1 - gam) * fh + gam * s * z
        y = y_new; p = p + y_new
        fh_hist.append(fh)
    return np.array(fh_hist)


res = {}
rows = []
for rho in (1.0, 0.9, 0.5, 0.373, 0.359):
    for model in ("fir", "arx"):
        fa = simulate(model, rho, "a")[-1] / f
        fb = simulate(model, rho, "b", Kp=0.15)[-1] / f
        fb2 = simulate(model, rho, "b", Kp=0.5)[-1] / f
        rows.append(dict(model=model, rho=rho, sim_regime_a=float(fa), closed_a=1 / (2 - rho), sim_regime_b_Kp015=float(fb), sim_regime_b_Kp05=float(fb2), closed_b=rho))
        print(f"{model:3s} rho={rho:5.3f}  regime (a): sim {fa:.4f} closed {1/(2-rho):.4f} | regime (b): sim {fb:.4f} (Kp=.15) {fb2:.4f} (Kp=.5) closed {rho:.4f}")
res["regimes"] = rows
# theta family
th_rows = []
rho = 0.373
for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
    fs = simulate("fir", rho, "theta", theta=theta)[-1] / f
    closed = (1 - (1 - rho) * theta) / (1 + (1 - rho) * (1 - theta))
    th_rows.append(dict(theta=theta, sim=float(fs), closed=float(closed)))
    print(f"theta={theta:.2f}: sim {fs:.4f} closed {closed:.4f}")
res["theta_family_rho_0.373"] = th_rows
# nonzero nominal command a0: regime (a) f^ = [f + (1-rho) a0]/(2-rho); regime (b) f^ = rho f + (1-rho) a0
a0 = 0.2; rho = 0.5
fa = simulate("fir", rho, "a", a0=a0)[-1]; fb = simulate("arx", rho, "b", a0=a0)[-1]
res["nonzero_a0"] = dict(a0=a0, rho=rho, sim_a=float(fa), closed_a=(f + (1 - rho) * a0) / (2 - rho), sim_b=float(fb), closed_b=rho * f + (1 - rho) * a0)
print(f"a0={a0}: regime (a) sim {fa:.4f} closed {(f+(1-rho)*a0)/(2-rho):.4f}; regime (b) sim {fb:.4f} closed {rho*f+(1-rho)*a0:.4f}")
# NT law: fixed point s(|r|) * (innovation fixed point) -- check at rho = 0.373, regime (b)
fh_nt = simulate("fir", 0.373, "b", law="nt")[-1]
# stationary residual under NT: r = M f + (M - D) u with u = a - fh and a = a0 - (f - fh) (regime b) -> r = D f (a0 = 0) -> s = 1/(1+(D f/rho_nt)^2)
D = 0.373 * M; s_bar = 1 / (1 + (D * f / 0.15) ** 2)
res["nt_regime_b"] = dict(sim_over_f=float(fh_nt / f), s_bar_times_rho=float(s_bar * 0.373), s_bar=float(s_bar))
print(f"NT law regime (b): sim {fh_nt/f:.4f}, s_bar*rho = {s_bar*0.373:.4f} (s_bar = {s_bar:.4f})")
# ARX transient overshoot: a_hat != lam, exact DC (rho = 1), regime (a), first steps of z
for a_hat in (0.71, 0.93, 0.95):
    fh_tr = simulate("arx", 1.0, "a", a_hat=a_hat, N=60)
    # first-step innovation z_1 = r_1/((1-a_hat) M) with r_1 = (1-lam) M f (y_0 = 0, u = 0 before the estimator moves; f^_0 = 0)
    z1 = (1 - lam) / (1 - a_hat)
    res.setdefault("arx_first_step_gain", {})[str(a_hat)] = dict(z1_over_f=float(z1), fhat_after_5_steps_over_f=float(fh_tr[4] / f), fhat_after_60_over_f=float(fh_tr[-1] / f))
    print(f"ARX a_hat={a_hat}: first-step z/f = (1-lam)/(1-a_hat) = {z1:.2f}; f^/f after 5 steps {fh_tr[4]/f:.3f}, after 60 {fh_tr[-1]/f:.3f}")
fir_tr = simulate("fir", 1.0, "a", N=60)
print(f"FIR exact DC: f^/f after 5 steps {fir_tr[4]/f:.3f}, after 60 {fir_tr[-1]/f:.3f}")
res["fir_exact_transient"] = dict(fhat_after_5_steps_over_f=float(fir_tr[4] / f), after_60=float(fir_tr[-1] / f))
# quasi-static stability: f^_{k+1} = (1 - alpha(2-rho)) f^_k + ... ; with plant dynamics, simulate rho grid
stab = {}
for rho in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
    fh_hist = simulate("fir", rho, "a", N=2000)
    stab[str(rho)] = dict(quasi_static_root=float(1 - alpha * (2 - rho)), sim_bounded=bool(np.isfinite(fh_hist[-1]) and abs(fh_hist[-1]) < 1e3), sim_final_over_f=float(fh_hist[-1] / f) if np.isfinite(fh_hist[-1]) else None)
    print(f"rho={rho:5.2f}: quasi-static root {1-alpha*(2-rho):.3f}; sim bounded {stab[str(rho)]['sim_bounded']}; final/f {stab[str(rho)]['sim_final_over_f']}")
res["stability_regime_a"] = stab
OUT.write_text(json.dumps(res, indent=1)); print("wrote", OUT)
