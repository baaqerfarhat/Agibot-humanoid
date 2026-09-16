"""Q4, part (i): the input-only (impulse-response / FIR) residual and the state (ARX) innovation are
related by the monic filter A(q^-1) of the ARX factorisation, for ANY noise and ANY fault sequence.

Checks
  S1  symbolic scalar case (sympy): first-order plant y_{k+1} = lam y_k + b (u_k + f_k) + n_{k+1}
      r_in_{k+1} = y_{k+1} - sum_j b lam^j u_{k-j}     (exact impulse response, zero initial conditions)
      r_st_{k+1} = y_{k+1} - lam y_k - b u_k            (exact ARX(1))
      identity  r_st_{k+1} = r_in_{k+1} - lam r_in_k .
  N1  numeric MIMO case: random 2-channel ARX(p=2, K=3) plant with a stable A, a time-varying fault,
      and a noise that depends on the state (n_k = 0.3 tanh(y_k) + white): the identity holds to
      machine precision (it is algebraic; no assumption on n or f).
  N2  the map r_in_{1:N} -> (r_in_{1:p}, r_st_{p+1:N}) is unit lower triangular (invertible):
      condition number reported; Fisher information about a constant fault under Gaussian noise is
      identical for the two residual histories (equal information).
  N3  frequency domain: the ratio of fault-signal to noise spectral density is the same at every
      frequency for both residuals (both are multiplied by |A(e^{jw})|^2).
"""
import json, pathlib
import numpy as np

OUT = pathlib.Path(__file__).with_suffix(".json")
res = {}

# ---------- S1 symbolic ----------
try:
    import sympy as sp
    lam, b, f0, f1, f2, u0, u1, u2, n1, n2, n3, y0 = sp.symbols("lam b f0 f1 f2 u0 u1 u2 n1 n2 n3 y0")
    # plant from zero initial condition y0 = 0, inputs u_k + f_k, k = 0,1,2
    y1 = b * (u0 + f0) + n1
    y2 = lam * y1 + b * (u1 + f1) + n2
    y3 = lam * y2 + b * (u2 + f2) + n3
    # exact impulse-response (input-only) residuals: subtract the response to u alone
    r_in1 = sp.expand(y1 - b * u0)
    r_in2 = sp.expand(y2 - (b * u1 + lam * b * u0))
    r_in3 = sp.expand(y3 - (b * u2 + lam * b * u1 + lam**2 * b * u0))
    # exact ARX residuals
    r_st2 = sp.expand(y2 - lam * y1 - b * u1)
    r_st3 = sp.expand(y3 - lam * y2 - b * u2)
    ok1 = sp.simplify(r_st2 - (r_in2 - lam * r_in1)) == 0
    ok2 = sp.simplify(r_st3 - (r_in3 - lam * r_in2)) == 0
    # what each residual is in terms of the fault and noise
    res["S1"] = dict(identity_holds=bool(ok1 and ok2),
                     r_in3=str(sp.collect(r_in3, [f0, f1, f2])), r_st3=str(sp.collect(r_st3, [f0, f1, f2])))
    print("S1 symbolic identity r_st = (1 - lam q^-1) r_in :", ok1 and ok2)
    print("   r_in_3 =", res["S1"]["r_in3"]); print("   r_st_3 =", res["S1"]["r_st3"])
except ImportError:
    res["S1"] = "sympy unavailable"; print("S1 skipped (no sympy)")

# ---------- N1 numeric MIMO ----------
rng = np.random.default_rng(0)
d, p, K, N = 2, 2, 3, 400
# random stable monic A(q^-1) = I - A1 q^-1 - A2 q^-2 (companion spectral radius < 1), random B_0..B_K
while True:
    A1 = 0.5 * rng.standard_normal((d, d)); A2 = 0.2 * rng.standard_normal((d, d))
    comp = np.block([[A1, A2], [np.eye(d), np.zeros((d, d))]])
    if np.max(np.abs(np.linalg.eigvals(comp))) < 0.9:
        break
B = [rng.standard_normal((d, d)) for _ in range(K + 1)]
u = rng.standard_normal((N, d)); f = 0.3 * np.sin(np.arange(N)[:, None] / 37.0 + np.arange(d)[None, :])  # time-varying fault
y = np.zeros((N + 1, d)); n = np.zeros((N + 1, d))
for k in range(N):
    v = u[k] + f[k]
    yk = np.zeros(d)
    for i, Ai in enumerate([A1, A2], start=1):
        if k + 1 - i >= 0: yk += Ai @ y[k + 1 - i]
    for j, Bj in enumerate(B):
        if k - j >= 0: yk += Bj @ (u[k - j] + f[k - j])
    n[k + 1] = 0.3 * np.tanh(y[k]) + 0.1 * rng.standard_normal(d)      # state-dependent + white noise
    y[k + 1] = yk + n[k + 1]
# impulse response G_j of A^-1 B (long truncation, geometric decay makes it exact to ~1e-16 by j=200)
J = 300; G = np.zeros((J, d, d))
for j in range(J):
    Gj = B[j] if j <= K else np.zeros((d, d))
    for i, Ai in enumerate([A1, A2], start=1):
        if j - i >= 0: Gj = Gj + Ai @ G[j - i]
    G[j] = Gj
r_in = np.zeros((N + 1, d)); r_st = np.zeros((N + 1, d))
for k in range(N):
    pred_in = sum(G[j] @ u[k - j] for j in range(min(J, k + 1)))
    r_in[k + 1] = y[k + 1] - pred_in
    pred_st = sum(B[j] @ u[k - j] for j in range(min(K, k) + 1))
    for i, Ai in enumerate([A1, A2], start=1):
        if k + 1 - i >= 0: pred_st += Ai @ y[k + 1 - i]
    r_st[k + 1] = y[k + 1] - pred_st
# identity r_st_{k+1} = r_in_{k+1} - A1 r_in_k - A2 r_in_{k-1}
lhs = r_st[3:]; rhs = r_in[3:] - (A1 @ r_in[2:-1].T).T - (A2 @ r_in[1:-2].T).T
err = float(np.abs(lhs - rhs).max()); scale = float(np.abs(lhs).max())
# The simulation adds n inside the difference equation (equation-error form): y = A^-1 (B (u+f) + n).
# Then r_st = B f + n and r_in = G f + A^-1 n. In the output-error form y = G(u+f) + n_out one has
# r_in = G f + n_out and r_st = B f + A n_out; the two are the same statement with n_out = A^-1 n.
chk_st = np.array([sum(B[j] @ f[k - j] for j in range(min(K, k) + 1)) + n[k + 1] for k in range(N)])
n_out = np.zeros_like(n)   # A^-1 n by recursion
for k in range(1, N + 1):
    n_out[k] = n[k] + sum(Ai @ n_out[k - i] for i, Ai in enumerate([A1, A2], start=1) if k - i >= 0)
chk_in = np.array([sum(G[j] @ f[k - j] for j in range(min(J, k + 1))) + n_out[k + 1] for k in range(N)])
res["N1"] = dict(max_abs_identity_error=err, residual_scale=scale, rel_error=err / scale,
                 r_st_equals_Bf_plus_n=float(np.abs(r_st[1:] - chk_st).max()), r_in_equals_Gf_plus_Ainv_n=float(np.abs(r_in[1:] - chk_in).max()))
print(f"N1 MIMO identity: max |r_st - A r_in| = {err:.2e} (residual scale {scale:.2f}); r_st = Bf + n: {res['N1']['r_st_equals_Bf_plus_n']:.1e}; r_in = Gf + A^-1 n: {res['N1']['r_in_equals_Gf_plus_Ainv_n']:.1e}")

# ---------- N2 invertibility and Fisher information ----------
# stacked map T: r_in_{1:N} -> (r_in_{1:p}, r_st_{p+1:N}); block unit lower triangular
Nn = 60; T = np.eye(Nn * d)
for k in range(p, Nn):
    for i, Ai in enumerate([A1, A2], start=1):
        T[k * d:(k + 1) * d, (k - i) * d:(k - i + 1) * d] = -Ai
detT = float(np.linalg.det(T)); condT = float(np.linalg.cond(T))
# Fisher information about a constant fault f (d-vector) with i.i.d. Gaussian noise cov S:
# r_in = H f + n_stack, H_k = sum_{j<=k} G_j ; r_st = T r_in
S = np.array([[0.04, 0.01], [0.01, 0.09]]); Sig = np.kron(np.eye(Nn), S)
H = np.vstack([sum(G[j] for j in range(k + 1)) for k in range(Nn)])
I_in = H.T @ np.linalg.solve(Sig, H)
TH = T @ H; TS = T @ Sig @ T.T
I_st = TH.T @ np.linalg.solve(TS, TH)
res["N2"] = dict(det_T=detT, cond_T=condT, fisher_rel_diff=float(np.abs(I_in - I_st).max() / np.abs(I_in).max()))
print(f"N2 stacked map: det = {detT:.6f} (unit lower triangular), cond = {condT:.1f}; Fisher information rel. difference = {res['N2']['fisher_rel_diff']:.1e}")

# ---------- N3 spectral SNR ratio (scalar, first order) ----------
lam_s, b_s = 0.93, 0.276
w = np.linspace(1e-3, np.pi, 2000); A_w = np.abs(1 - lam_s * np.exp(-1j * w)) ** 2
G_w = (b_s ** 2) / A_w                       # |G|^2 = |B|^2 / |A|^2 with B = b
S_f = 1.0 / (1 + (w / 0.02) ** 2)             # any fault spectrum
S_n = 0.6 + 0.5 * np.cos(w)                   # any (coloured, strictly positive) noise spectrum
snr_in = G_w * S_f / S_n; snr_st = (A_w * G_w * S_f) / (A_w * S_n)
res["N3"] = dict(max_rel_diff=float(np.abs(snr_in - snr_st).max() / snr_in.max()),
                 arx_over_fir_noise_var_white=float((1 + lam_s ** 2) / (1 - lam_s) ** 2),
                 arx_over_fir_noise_density_ratio_at_w=dict(w0=0.0, w_0_08=float(((1 - lam_s) ** 2 + 2 * lam_s * (1 - np.cos(0.08))) / (1 - lam_s) ** 2)))
print(f"N3 spectral SNR ratio identical: max rel diff {res['N3']['max_rel_diff']:.1e}; per-step white-noise variance of the DC-normalised ARX innovation over the FIR residual at lam=0.93: {res['N3']['arx_over_fir_noise_var_white']:.0f}x, but the density ratio at w = 0.08 rad/step (the estimator bandwidth) is {res['N3']['arx_over_fir_noise_density_ratio_at_w']['w_0_08']:.2f}x and 1x at DC")
OUT.write_text(json.dumps(res, indent=1)); print("wrote", OUT)
