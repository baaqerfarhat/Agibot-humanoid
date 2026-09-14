"""Numerical checks of the synthesis's constructed mathematical examples.

These are not robot experiments, proof substitutes, or fitted certificates.
Run with the existing LIBERO Python. Optionally write --out verification.json
and --figure memory_example.pdf (requires matplotlib for the figure only).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--figure", type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    errors = {}

    # Finite-horizon bound: varying metrics and matrices, including expansion.
    worst = -np.inf
    max_growth = 0.
    for _ in range(128):
        n, horizon = 4, 40
        mats = [rng.normal(size=(n, n)) for _ in range(horizon+1)]
        metrics = [m.T@m + np.eye(n) for m in mats]
        z = rng.normal(size=n)
        radius = np.sqrt(z @ metrics[0] @ z)
        for k in range(horizon):
            a = .8*np.eye(n) + .22*rng.normal(size=(n, n))
            w = .02*rng.normal(size=n)
            # C.T C=P; factor form avoids choosing a matrix square root.
            c0 = np.linalg.cholesky(metrics[k]).T
            c1 = np.linalg.cholesky(metrics[k+1]).T
            gain = np.linalg.norm(c1 @ a @ np.linalg.inv(c0), 2)
            max_growth = max(max_growth, gain)
            radius = gain*radius + np.linalg.norm(c1 @ w)
            z = a @ z + w
            worst = max(worst, float(np.linalg.norm(c1 @ z)-radius))
    assert worst <= 1e-10 and max_growth > 1
    errors["finite_horizon_max_bound_violation"] = max(0., worst)

    # Exact scalar histories, with both neutral and contracting execution.
    horizon, f, gamma, gain, attenuation = 160, .05, .08, 1., .6
    trajectories = {}
    worst = 0.
    for lam in (.8, 1-gamma, 1.):
        p = 0.
        history = [p]
        for k in range(horizon):
            p = lam*p + gain*f*(1-gamma)**k
            n = k+1
            closed = (gain*f*n*lam**(n-1) if abs(lam-(1-gamma)) < 1e-12 else
                      gain*f*(lam**n-(1-gamma)**n)/(lam-(1-gamma)))
            worst = max(worst, abs(p-closed))
            history.append(p)
        trajectories[f"lambda={lam:.2f}, innovation"] = history
    p = 0.
    history = [p]
    for k in range(horizon):
        remaining = f*((1-attenuation)+attenuation*(1-gamma)**k)
        p += gain*remaining
        n = k+1
        closed = gain*f*((1-attenuation)*n+
                         attenuation/gamma*(1-(1-gamma)**n))
        worst = max(worst, abs(p-closed))
        history.append(p)
    trajectories["lambda=1.00, legacy s=0.6"] = history
    assert worst < 1e-12
    errors["scalar_memory_formula_max_error"] = worst

    # Jury region, away from boundaries, checked against eigenvalues.
    disagreements = 0
    for alpha in np.linspace(.025, 2.2, 49):
        for kappa in np.linspace(-.17, 4.3, 53):
            if min(abs(kappa), abs(4-2*alpha-kappa), abs(alpha-2)) < 1e-8:
                continue
            a = np.array([[1., -1.], [kappa, 1-alpha-kappa]])
            criterion = 0 < alpha < 2 and 0 < kappa < 4-2*alpha
            spectral = max(abs(np.linalg.eigvals(a))) < 1-1e-10
            disagreements += bool(criterion) != bool(spectral)
    assert disagreements == 0
    errors["jury_grid_disagreements"] = disagreements

    alpha, kappa = .2, .5
    a = np.array([[1., -1.], [kappa, 1-alpha-kappa]])
    p = solve_discrete_lyapunov(a.T, np.eye(2))
    lyapunov_error = np.max(abs(a.T @ p @ a-p+np.eye(2)))
    assert np.linalg.eigvalsh(p)[0] > 0 and lyapunov_error < 1e-12
    coefficient = np.linalg.norm(p, 2)+2*np.linalg.norm(a.T @ p, 2)**2
    worst = -np.inf
    for _ in range(1000):
        z, w = rng.normal(size=(2, 2))
        after = a @ z+w
        dv = after @ p @ after-z @ p @ z
        bound = -.5*np.dot(z, z)+coefficient*np.dot(w, w)
        worst = max(worst, dv-bound)
    assert worst <= 1e-10
    errors["joint_lyapunov_equation_error"] = float(lyapunov_error)
    errors["joint_forced_bound_max_violation"] = max(0., float(worst))

    # Static calibration counterexample: gain ratio is not the fixed point.
    g, ghat, calibration = .276, .10212, .276
    estimate = 0.
    for _ in range(500):
        residual = g*f-(g-ghat)*estimate
        estimate += .08*(residual/calibration-estimate)
    expected = g*f/(calibration+g-ghat)
    assert abs(estimate-expected) < 1e-12
    assert abs(estimate/f-ghat/g) > .2

    report = {
        "status": "constructed mathematical checks only; no robot certificate",
        "seed": 20260914,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": errors,
        "finite_horizon_cases": 128,
        "joint_example": {"alpha": alpha, "kappa": kappa, "P": p.tolist(),
                          "spectral_radius": float(max(abs(np.linalg.eigvals(a))))},
        "integrator_unbiased_limiting_displacement": gain*f/gamma,
        "mismatch_fixed_point_ratio": estimate/f,
        "fitted_to_probed_ratio": ghat/g,
    }
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for key in ("lambda=0.80, innovation", "lambda=1.00, innovation",
                    "lambda=1.00, legacy s=0.6"):
            axes[0].plot(trajectories[key], label=key)
        axes[0].set(xlabel="Control step", ylabel="Displacement (illustrative units)",
                    title="The same observer meets different execution memory")
        axes[0].legend(fontsize=7)
        for tracking in (0., .5, 4.):
            transition = np.array([[1., -1.], [tracking, 1-alpha-tracking]])
            z = np.array([1., 0.])
            values = [np.linalg.norm(z)]
            for _ in range(50):
                z = transition @ z
                values.append(np.linalg.norm(z))
            axes[1].semilogy(values, label=f"tracking gain {tracking:g}")
        axes[1].set(xlabel="Control step", ylabel="Joint state norm",
                    title="Joint feedback can stabilize a neutral mode")
        axes[1].legend(fontsize=7)
        fig.suptitle("Constructed examples — not fitted robot results", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.figure, bbox_inches="tight")
        plt.close(fig)
    text = json.dumps(report, indent=2)+"\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
