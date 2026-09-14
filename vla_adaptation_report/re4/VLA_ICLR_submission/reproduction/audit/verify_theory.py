#!/usr/bin/env python3
"""Deterministic numerical consistency checks for the manuscript's analysis.

These finite-dimensional constructed examples are neither robot experiments nor
empirical VLA validation. Symbols in this retained numerical suite predate the
contraction revision; it does not parse or certify the manuscript. They do not replace the analytical proofs or verify
their assumptions on the reported systems. Run: python verify_theory.py
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


TOL = 2e-11
RESULTS: list[dict] = []


def record(name: str, excesses, **details) -> None:
    values = np.asarray(excesses, dtype=float)
    largest = float(np.max(values))
    passed = bool(np.all(np.isfinite(values)) and largest <= TOL)
    RESULTS.append({"name": name, "passed": passed,
                    "maximum_bound_excess": largest, **details})
    if not passed:
        raise AssertionError(f"{name}: inequality exceeded tolerance by {largest}")


def projected_innovation() -> None:
    """Active projection, variable normalized steps, drift and bounded noise."""
    count = 150
    ts = np.arange(count + 1)
    faults = np.column_stack((.85 + .08*np.sin(.11*ts),
                              -.90 + .06*np.cos(.09*ts),
                              .70 + .05*np.sin(.06*ts)))
    M = np.array([[.5, .07, 0], [0, .8, .03], [.02, 0, 1.2]])
    noises = np.column_stack((.013*np.cos(.2*ts[:-1]),
                              .014*np.sin(.3*ts[:-1]),
                              .01*np.cos(.17*ts[:-1])))
    noises[0] = [1.2, -1.1, .9]  # Deliberately triggers box projection.
    gamma, rho = .95, 5.0
    estimate = np.zeros(3)
    errors = [float(np.linalg.norm(faults[0] - estimate))]
    alphas, epsilons, drifts, one_step_excess = [], [], [], []
    projection_count = 0
    for t in range(count):
        observation = faults[t] + noises[t]
        innovation = M @ (observation - estimate)
        alpha = gamma / (1 + np.dot(innovation, innovation)/rho**2)
        candidate = estimate + alpha*(observation-estimate)
        estimate = np.clip(candidate, -1, 1)
        projection_count += int(np.any(np.abs(candidate) > 1))
        epsilon = float(np.linalg.norm(noises[t]))
        drift = float(np.linalg.norm(faults[t+1]-faults[t]))
        error = float(np.linalg.norm(faults[t+1]-estimate))
        one_step_excess.append(error-((1-alpha)*errors[-1]+alpha*epsilon+drift))
        errors.append(error)
        alphas.append(alpha)
        epsilons.append(epsilon)
        drifts.append(drift)
    assert projection_count > 0 and np.ptp(alphas) > .1
    # Independently evaluate the product form rather than iterate its recursion.
    product_excess = []
    for t in range(1, count+1):
        bound = np.prod(1-np.array(alphas[:t])) * errors[0]
        for j in range(t):
            bound += np.prod(1-np.array(alphas[j+1:t])) * (
                alphas[j]*epsilons[j]+drifts[j])
        product_excess.append(errors[t]-bound)
    diameter = 2*np.sqrt(3)
    epsilon_max = max(epsilons)
    alpha_min = gamma/(1+np.linalg.norm(M, 2)**2*(diameter+epsilon_max)**2/rho**2)
    uniform_excess = [errors[t]-((1-alpha_min)**t*errors[0]+epsilon_max+
                                  max(drifts)/alpha_min)
                      for t in range(1, count+1)]
    record("innovation_projection_drift_and_product_bound",
           one_step_excess+product_excess+uniform_excess,
           steps=count, projected_steps=projection_count,
           observed_alpha_range=[float(min(alphas)), float(max(alphas))],
           theoretical_alpha_lower_bound=float(alpha_min),
           final_estimation_error=errors[-1],
           checked="One-step, independently expanded product, and uniform bounds")


def fir_history() -> None:
    """A perfect FIR still sees a filtered fault immediately after onset."""
    K, onset, horizon = 6, 5, 25
    base = np.array([[.35, .07], [-.02, .28]])
    G = np.array([(.55**ell)*base for ell in range(K+1)])
    M = G.sum(axis=0)
    inverse = np.linalg.inv(M)
    fault = np.array([.20, -.10])
    def f(t):
        return fault if t >= onset else np.zeros(2)
    def u(t):
        return np.array([.12*np.sin(.3*t), .09*np.cos(.2*t)]) if t >= 0 else np.zeros(2)
    excesses, transient, settled = [], [], []
    for t in range(horizon):
        measured = sum(G[ell] @ (u(t-ell)+f(t-ell)) for ell in range(K+1))
        predicted = sum(G[ell] @ u(t-ell) for ell in range(K+1))
        residual = measured-predicted
        z = inverse @ residual
        history_vector = sum(G[ell] @ (f(t-ell)-f(t)) for ell in range(K+1))
        exact_error = z-f(t)
        excesses.append(np.linalg.norm(exact_error-inverse@history_vector))
        bound = np.linalg.norm(inverse, 2)*sum(
            np.linalg.norm(G[ell], 2)*np.linalg.norm(f(t-ell)-f(t))
            for ell in range(K+1))
        excesses.append(np.linalg.norm(exact_error)-bound)
        if onset <= t < onset+K:
            transient.append(float(np.linalg.norm(exact_error)))
        if t >= onset+K:
            settled.append(float(np.linalg.norm(exact_error)))
    assert transient[0] > .05
    record("fir_step_onset_and_settled_history", excesses,
           memory_K=K, onset=onset, first_post_onset_error=transient[0],
           largest_settled_error=max(settled),
           checked="Direct plant/predictor convolution and exact history decomposition")


def legacy_fixed_point() -> None:
    fault = np.array([1.0, -.7, .3])
    M = np.diag([.4, .6, .2])
    gamma, rho = .08, .15
    residual = M@fault
    scale = 1/(1+np.dot(residual, residual)/rho**2)
    target = scale*fault
    initial = np.array([-1.0, 1.0, -.4])
    estimate = initial.copy()
    excesses = []
    for t in range(1, 701):
        previous = np.linalg.norm(estimate-fault)
        estimate = np.clip((1-gamma)*estimate+gamma*scale*fault, -1, 1)
        closed_form = target+(1-gamma)**t*(initial-target)
        excesses.append(np.linalg.norm(estimate-closed_form))
        excesses.append(np.linalg.norm(estimate-fault)-(
            (1-gamma)*previous+gamma*(1-scale)*np.linalg.norm(fault)))
    fixed_error = np.linalg.norm(estimate-target)
    excesses.append(fixed_error)
    assert np.linalg.norm(estimate-fault) > .5
    record("legacy_attenuation_and_exact_fixed_point", excesses,
           attenuation=float(scale), convergence_error=float(fixed_error),
           remaining_true_fault_error=float(np.linalg.norm(estimate-fault)),
           checked="Closed-form iteration and error recursion at a boundary-feasible fault")


def trajectory_tube() -> None:
    """Variable incremental factors, a mask, and a nonzero adapter discrepancy."""
    count = 60
    D = np.diag([1.0, 0.0])
    identity = np.eye(2)
    x = np.zeros(2)
    R = 0.0
    factors, forcings = [], []
    excesses = []
    discrepancy_required = False
    for t in range(count):
        fault = np.array([.30, .025*np.sin(.2*t)])
        estimate = fault-np.array([.01*np.cos(.17*t), .02*np.sin(.11*t)])
        zeta = np.array([.02, .003*np.sin(.3*t)])
        q = fault-D@estimate+zeta
        E = np.linalg.norm(fault-estimate)
        unmatched = np.linalg.norm((identity-D)@fault)
        U = unmatched+np.linalg.norm(D, 2)*E+np.linalg.norm(zeta)
        discrepancy_required |= np.linalg.norm(q) > unmatched+E+1e-8
        lam = [.8, 1.1, .65][t % 3]
        L = .08+.02*np.cos(.1*t)
        angle = .13*t
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        x = lam*rotation@x+L*q
        R = lam*R+L*U
        factors.append(lam)
        forcings.append(L*U)
        product_bound = sum(np.prod(factors[j+1:])*forcings[j]
                            for j in range(t+1))
        excesses += [np.linalg.norm(q)-U, np.linalg.norm(x)-R,
                     abs(R-product_bound)]
    assert discrepancy_required
    record("masked_finite_horizon_tube_with_adapter_discrepancy", excesses,
           steps=count, final_distance=float(np.linalg.norm(x)), final_radius=float(R),
           includes_expansive_steps=True,
           adapter_discrepancy_needed_in_at_least_one_step=True,
           checked="Explicit linear trajectories and independent tube-product expansion")


def small_gain() -> None:
    configs = [
        dict(lam=.8, b=.12, kE=.15, kX=.2, amin=.12),
        dict(lam=.95, b=.2, kE=.4, kX=.14, amin=.05),
        dict(lam=.6, b=.3, kE=.2, kX=0., amin=.2),
    ]
    all_excesses, summaries = [], []
    for p in configs:
        lam, b, kE, kX, amin = (p[k] for k in ("lam", "b", "kE", "kX", "amin"))
        amax, epsilon0, nu, Uperp, zeta, eta = 1.0, .004, .001, .01, .005, .0002
        assert kX*b < (1-kE)*(1-lam)
        lower = b/(1-lam)
        s = np.sqrt(lower*(1-kE)/kX) if kX else lower+1
        chi = max(lam+b/s, 1-amin*(1-kE-s*kX))
        d0 = b*(Uperp+zeta)+eta  # L=b, ||D||=1 in this constructed example.
        dstar = max(d0, s*(amax*epsilon0+nu))
        assert 0 < chi < 1
        X, E = .8, .4
        W0 = max(X, s*E)
        largest_spectral_radius = 0.
        for t in range(1, 401):
            alpha = [amin, amax, .5*(amin+amax), .75][(t-1) % 4]
            A = np.array([[lam, b], [alpha*kX, 1-alpha*(1-kE)]])
            oldW = max(X, s*E)
            # Equality in each positive comparison recursion is a worst-case envelope.
            X, E = A@np.array([X, E])+np.array([d0, alpha*epsilon0+nu])
            W = max(X, s*E)
            bound = chi**t*W0+dstar*(1-chi**t)/(1-chi)
            all_excesses += [W-(chi*oldW+dstar), W-bound]
            largest_spectral_radius = max(largest_spectral_radius,
                                          float(max(abs(np.linalg.eigvals(A)))))
        summaries.append({**p, "weight_s":float(s), "chi":float(chi),
                          "forcing_dstar":float(dstar),
                          "largest_constant_matrix_spectral_radius":largest_spectral_radius,
                          "final_weighted_error":float(W), "final_bound":float(bound)})
    record("coupled_small_gain_variable_step_worst_case", all_excesses,
           configurations=summaries, steps_per_configuration=400,
           checked="Common weighted bound, forced recursion, near-boundary and kX=0 cases")




def main() -> None:
    for check in (projected_innovation, fir_history, legacy_fixed_point,
                  trajectory_tube, small_gain):
        check()
    report = {
        "purpose":"Analytical consistency checks on constructed numerical examples",
        "limitations":["Not robot experiments or empirical VLA validation",
                       "Not a substitute for proofs",
                       "Does not certify contraction, calibration, or task margins on the reported systems"],
        "numpy_version":np.__version__, "absolute_tolerance":TOL,
        "all_passed":all(result["passed"] for result in RESULTS), "checks":RESULTS,
    }
    destination = Path(__file__).with_name("theory_checks.json")
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(RESULTS)} analytical consistency checks passed; wrote {destination.name}")


if __name__ == "__main__":
    main()
