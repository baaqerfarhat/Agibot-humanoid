#!/usr/bin/env python3
"""Fixed-seed numerical sanity checks for the two theory appendices.

NumPy and the actual composite observer only; no simulator or policy rollouts.
Finite numerical checks do not prove the propositions or certify a robot.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openpi"))
import composite_observer as composite  # noqa: E402

SEED = 71


def composite_recursion(rng):
    maximum, cases = 0.0, []
    for damping in (0.0, 0.7):
        for clip in (None, 0.08):
            A = np.array([[0.8, 0.1], [0.0, 0.7]])
            D = rng.normal(size=(2, 3))
            G = D + 0.1 * rng.normal(size=D.shape)
            H = rng.normal(size=(4, 3))
            L = np.array([[2.0, 0.2], [0.2, 1.0]])
            e, f, theta = rng.normal(size=2), rng.normal(size=3), rng.normal(size=3)
            state, dt, rate, clipped = None, 0.02, 3.0, 0
            for _ in range(20):
                w, v = 0.01 * rng.normal(size=2), 0.01 * rng.normal(size=4)
                f_next = f + 0.01 * rng.normal(size=3)
                e_next = A @ e + G @ f - D @ theta + w
                actual, info = composite.composite_step(
                    theta, H @ f + v, H, state=state,
                    Q=0.2 * np.eye(3), R=np.eye(4), dt=dt,
                    tracking_error=e_next, tracking_map=D, metric=L,
                    tracking_rate=rate, damping=damping, clip=clip)
                K, P = info["gain"], info["estimator_state"]["covariance"]
                T, a = dt * rate * P @ D.T @ L, np.exp(-damping * dt)
                C = np.eye(3) - K @ H
                matrix = np.block([[A, -D], [T @ A, a * C - T @ D]])
                b = (G - D) @ f + w
                unprojected = a * theta + K @ (H @ f + v - H @ (a * theta)) + T @ e_next
                projection = actual - unprojected
                forcing = np.r_[b, T @ b + (a - 1) * C @ f
                                - (f_next - f) + K @ v + projection]
                derived = matrix @ np.r_[e, theta - f] + forcing
                observed = np.r_[e_next, actual - f_next]
                maximum = max(maximum, float(np.max(np.abs(derived - observed))))
                np.testing.assert_allclose(derived, observed, rtol=1e-12, atol=1e-12)
                clipped += int(info["projection_active"])
                e, theta, f, state = e_next, actual, f_next, info["estimator_state"]
            cases.append(dict(damping=damping, clip=clip, steps=20, clipped_updates=clipped))
    return dict(steps=80, cases=cases, maximum_absolute_recursion_discrepancy=maximum)


def scalar_tracking():
    # Exact rational Jury boundary: 1 + trace + determinant = 0.
    A, F, posterior = Fraction(4, 5), Fraction(4, 5), Fraction(1, 5)
    critical = (1 + A + F + A * F) / posterior
    assert critical == Fraction(81, 5)
    records = []
    for rate in (0.0, float(critical), 20.0):
        _, info = composite.composite_step(
            np.zeros(1), np.zeros(1), np.eye(1),
            state=dict(covariance=np.array([[0.2]]), updates=0),
            Q=np.array([[0.05]]), R=np.eye(1), dt=1,
            tracking_error=np.zeros(1), tracking_map=np.eye(1), metric=np.eye(1),
            tracking_rate=rate, damping=0, clip=None)
        np.testing.assert_allclose(info["gain"], [[0.2]], atol=1e-15)
        np.testing.assert_allclose(info["estimator_state"]["covariance"], [[0.2]], atol=1e-15)
        report = composite.augmented_error_report(
            np.array([[0.8]]), np.eye(1), np.eye(1),
            info["estimator_state"]["covariance"], info["gain"],
            dt=1, tracking_rate=rate, damping=0, metric=np.eye(1))
        eigenvalues = np.sort(np.linalg.eigvals(report["matrix"]))
        expected = ([0.8, 0.8] if rate == 0 else [-1, -0.64] if rate == float(critical)
                    else [-1.2 - np.sqrt(0.8), -1.2 + np.sqrt(0.8)])
        np.testing.assert_allclose(eigenvalues, expected, rtol=1e-12, atol=1e-12)
        records.append(dict(rate=rate, eigenvalues=eigenvalues.tolist()))
    return dict(critical_rate_exact=str(critical), dt=1, cases=records,
                scope="Scalar constructed example; not the ALOHA rate or active-strength threshold.")


def authority_counterexamples():
    M, d = np.array([[1.0, 1.0], [0.0, 0.1]]), np.array([0.0, 0.02])
    full = np.linalg.solve(M, d)
    masked = d - M[:, 0] * full[0]
    ratio1 = np.linalg.norm(masked) / np.linalg.norm(d)
    np.testing.assert_allclose(ratio1, np.sqrt(101), atol=1e-13)

    alpha, t, tau = 0.01, 0.2, 0.01
    ridge, Z = tau * alpha**2, 1 + 2 * tau + tau * (1 + tau) * alpha**2
    nominal = np.array([[1.0, 1.0], [0.0, alpha]])
    physical = np.array([[1.0, 1 + 2 * alpha], [0.0, alpha]])
    d = np.array([0.0, t * alpha])
    candidate = t / Z * np.array([-1.0, 1 + tau * alpha**2])
    solved = np.linalg.solve(nominal.T @ nominal + ridge * np.eye(2), nominal.T @ d)
    np.testing.assert_allclose(candidate, solved, rtol=1e-10, atol=1e-12)
    assert np.max(np.abs(candidate)) < 0.3
    objective_ratio = (np.linalg.norm(d - nominal @ candidate)**2
                       + ridge * np.linalg.norm(candidate)**2) / np.linalg.norm(d)**2
    physical_ratio = np.linalg.norm(d - physical @ candidate) / np.linalg.norm(d)
    np.testing.assert_allclose(objective_ratio, (2 * tau + tau**2 * alpha**2) / Z, atol=1e-13)
    formula = np.sqrt((2 * (1 + tau * alpha**2) + tau * alpha)**2
                      + (2 * tau + tau**2 * alpha**2)**2) / Z
    np.testing.assert_allclose(physical_ratio, formula, atol=1e-13)
    exact = np.array([-(1 + 2 * alpha) * t, t])
    np.testing.assert_allclose(physical @ exact, d, atol=1e-14)
    assert np.max(np.abs(exact)) < 0.3 and objective_ratio < 0.02 and physical_ratio > 1.96
    return dict(
        inverse_then_mask=dict(estimate=full.tolist(), physical_error_ratio=ratio1,
                               authority_floor=0.02, optimal_permitted_correction=0.0),
        perturbed_ridge_map=dict(candidate=candidate.tolist(), true_exact_correction=exact.tolist(),
                                nominal_objective_ratio=objective_ratio, physical_error_ratio=physical_ratio,
                                relative_map_error=float(np.linalg.norm(physical - nominal, 2)
                                                         / np.linalg.norm(nominal, 2))))


def robust_authority_bounds(rng, count=128):
    minimum_gap, maximum_sharp_error = float("inf"), 0.0
    for i in range(count):
        X, nominal, v = rng.normal(size=(3, 3)), rng.normal(size=(3, 2)), rng.normal(size=3)
        R = X @ X.T + np.eye(3)
        W = np.linalg.inv(np.linalg.cholesky(R))
        c = np.zeros(2) if i % 16 == 0 else rng.uniform(-0.3, 0.3, size=2)
        delta, epsilon = (0.0, 0.01, 0.2)[i % 3], (0.0, 0.01, 0.2)[(i // 3) % 3]
        u, E = rng.normal(size=3), rng.normal(size=(3, 2))
        u *= delta * rng.random() / np.linalg.norm(u)
        E *= epsilon * rng.random() / np.linalg.norm(E, 2)
        d, physical = v + np.linalg.solve(W, u), nominal + np.linalg.solve(W, E)
        a, b, eta = W @ v, W @ nominal @ c, epsilon * np.linalg.norm(c)
        residual = W @ (d - physical @ c)
        improvement = np.linalg.norm(W @ d)**2 - np.linalg.norm(residual)**2
        lower = (np.linalg.norm(a)**2 - np.linalg.norm(a - b)**2
                 - 2 * eta * np.linalg.norm(a - b) - eta**2
                 - 2 * delta * (np.linalg.norm(b) + eta))
        gap = float(improvement - lower)
        minimum_gap = min(minimum_gap, gap)
        assert gap >= -1e-11
        envelope = np.linalg.norm(a - b) + delta + eta
        assert np.linalg.norm(residual) <= envelope + 1e-11
        # The unstructured uncertainty ball also attains this envelope.
        direction = (a - b) / np.linalg.norm(a - b)
        E_sharp = (-epsilon * np.outer(direction, c) / np.linalg.norm(c)
                   if np.linalg.norm(c) else np.zeros((3, 2)))
        sharp = a - b + delta * direction - E_sharp @ c
        discrepancy = abs(float(np.linalg.norm(sharp) - envelope))
        maximum_sharp_error = max(maximum_sharp_error, discrepancy)
        assert discrepancy <= 1e-11
    return dict(cases=count, minimum_observed_improvement_minus_bound=minimum_gap,
                maximum_sharp_envelope_discrepancy=maximum_sharp_error,
                includes_zero_uncertainty_and_zero_correction=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "paper/interface_theory_receipt.json")
    args = parser.parse_args()
    paths = [Path(__file__).resolve(), Path(composite.__file__).resolve(),
             ROOT / "paper/theory_appendix.tex", ROOT / "paper/authority_appendix.tex"]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    rng = np.random.default_rng(SEED)
    checks = dict(composite=composite_recursion(rng), scalar_tracking=scalar_tracking(),
                  authority_counterexamples=authority_counterexamples(),
                  robust_authority=robust_authority_bounds(rng))
    if hashes() != before:
        raise RuntimeError("A checked source changed during the calculation")
    receipt = dict(
        schema_version=1, status="passed", seed=SEED,
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        sources_sha256=before,
        reproduce="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python paper/verify_interface_theory.py",
        interpretation="Numerical sanity checks of finite constructed cases, not proofs, robot qualification, or policy evaluation.",
        checks=checks)
    args.out.resolve().write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(receipt=str(args.out.resolve()), status="passed",
                         recurrence_error=checks["composite"]["maximum_absolute_recursion_discrepancy"],
                         scalar_critical_rate=checks["scalar_tracking"]["critical_rate_exact"],
                         robust_cases=checks["robust_authority"]["cases"]), indent=2))


if __name__ == "__main__":
    main()
