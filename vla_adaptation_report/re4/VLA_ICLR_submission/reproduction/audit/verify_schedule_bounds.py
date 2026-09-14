#!/usr/bin/env python3
"""Deterministic checks of sampled update/hold and application-delay bounds.

These constructed sequences test mathematical statements, not robot outcomes,
measured latencies, or the completeness of any experimental configuration.
Run from any working directory; writes adjacent JSON and a short audit memo.
"""
from pathlib import Path
import json
import numpy as np

RNG = np.random.default_rng(20260909)
ROOT = Path(__file__).resolve().parent
TOL = 5e-11
RESULTS = {}


def record(name, values, count):
    vals = np.asarray(values, dtype=float)
    violation = max(0.0, float(np.max(vals)))
    assert violation < TOL, (name, violation)
    RESULTS[name] = {"checks": int(count), "maximum_positive_violation": violation}


def projection_checks():
    """Full vector projection, both laws, arbitrary gates and moving targets."""
    errors = {"legacy": [], "innovation": []}
    for _ in range(4000):
        dim = int(RNG.integers(1, 9))
        radius = RNG.uniform(0.1, 2.0, dim)
        f = RNG.uniform(-radius, radius)
        fn = np.clip(f + RNG.normal(0, 0.15, dim), -radius, radius)
        hat = RNG.uniform(-radius, radius)
        w = RNG.normal(0, 0.2, dim)
        z = f + w
        chi = int(RNG.integers(0, 2))
        s = float(RNG.choice([0.0, 1.0, RNG.uniform()]))
        gamma = RNG.uniform(0.01, 1.0)
        alpha = RNG.uniform(0.0, 1.0)
        E, eps, nu = map(np.linalg.norm, [hat - f, w, fn - f])
        h_legacy = np.clip((1-chi*gamma)*hat + chi*gamma*s*z, -radius, radius)
        bound_l = (1-chi*gamma)*E + chi*gamma*((1-s)*np.linalg.norm(f)+s*eps)+nu
        errors["legacy"].append(np.linalg.norm(h_legacy-fn)-bound_l)
        h_innov = np.clip(hat + chi*alpha*(z-hat), -radius, radius)
        bound_i = (1-chi*alpha)*E + chi*alpha*eps+nu
        errors["innovation"].append(np.linalg.norm(h_innov-fn)-bound_i)
    for key, val in errors.items():
        record("projected_scheduled_" + key, val, len(val))


def scheduled_sequences():
    """Fault onset/removal, holds/leakage, chunk delay, noncontracting phases."""
    T, dim = 96, 3
    t = np.arange(T+1)
    f = np.column_stack([0.32+0.0008*t, -0.18+0.0003*t, 0.12-0.0005*t])
    f[17:] += np.array([0.25, -0.16, 0.1])
    f[59:] -= np.array([0.4, -0.1, 0.18])
    w = 0.035*np.column_stack([np.sin(0.7*t), np.cos(0.31*t), np.sin(0.19*t+.2)])
    nu = np.linalg.norm(np.diff(f, axis=0), axis=1)
    eps = np.linalg.norm(w, axis=1)
    chi = ((t % 3) != 1).astype(int)
    chi[27:41] = 0
    chi[77:] = 0
    s = 0.3+0.5*(1+np.sin(.3*t))/2
    s[::11] = 0  # Target-only gating on update steps retains leakage.
    alpha = .08/(1+np.linalg.norm(w, axis=1)**2/.1**2)
    tau = (t//5)*5  # New application snapshot only at chunk boundaries.
    tau[70:] = 70   # Later hold is deliberately not a bounded-delay regime.
    D = np.diag([0.0, 1.0, 1.0])
    zeta = .012*np.column_stack([np.sin(.23*t), np.cos(.61*t), np.sin(.43*t)])
    for law in ["legacy", "innovation"]:
        hat, B = np.zeros((T+1, dim)), np.zeros(T+1)
        B[0] = np.linalg.norm(f[0]-hat[0])
        beta = chi*.08 if law == "legacy" else chi*alpha
        omega = (1-s)*np.linalg.norm(f, axis=1)+s*eps if law == "legacy" else eps
        for k in range(T):
            target = s[k]*(f[k]+w[k]) if law == "legacy" else f[k]+w[k]
            hat[k+1] = np.clip((1-beta[k])*hat[k]+beta[k]*target, -.7, .7)
            B[k+1] = (1-beta[k])*B[k]+beta[k]*omega[k]+nu[k]
        E = np.linalg.norm(hat-f, axis=1)
        record(law+"_envelope", E-B, T+1)
        # Independent explicit product expansion.
        expanded = []
        for k in range(T+1):
            val = np.prod(1-beta[:k])*B[0]
            for j in range(k):
                val += np.prod(1-beta[j+1:k])*(beta[j]*omega[j]+nu[j])
            expanded.append(abs(val-B[k]))
        record(law+"_product_identity", expanded, T+1)
        increments = np.linalg.norm(np.diff(hat, axis=0), axis=1)
        Bapp, Eapp, first, second = [], [], [], []
        for k in range(T+1):
            j = int(tau[k])
            eapp = np.linalg.norm(f[k]-hat[j])
            inc_bound = B[k]+np.sum(increments[j:k])
            age_bound = B[j]+np.sum(nu[j:k])
            Bapp.append(min(inc_bound, age_bound))
            Eapp.append(eapp)
            first.append(eapp-inc_bound)
            second.append(eapp-age_bound)
        Bapp, Eapp = np.array(Bapp), np.array(Eapp)
        record(law+"_delay_increment_bound", first, T+1)
        record(law+"_delay_age_bound", second, T+1)
        q = f-hat[tau]@D.T+zeta
        decomposed = f@(np.eye(dim)-D).T+(f-hat[tau])@D.T+zeta
        record(law+"_mismatch_identity", np.linalg.norm(q-decomposed, axis=1), T+1)
        U = np.linalg.norm(f@(np.eye(dim)-D).T, axis=1)+np.linalg.norm(D,2)*Bapp+np.linalg.norm(zeta,axis=1)
        record(law+"_mismatch_bound", np.linalg.norm(q,axis=1)-U, T+1)
        # Constructed scalar system: exact Lipschitz constants, includes lambda>1.
        lam = np.resize(np.array([1.15, 1.0, .75, .64, .85]), T)
        L = .15+.025*np.sin(.27*t[:T])
        disturbance = .005*np.cos(.9*t[:T])
        x, R = np.zeros(T+1), np.zeros(T+1)
        x[0], R[0] = .04, .04
        direction = np.array([1., -2., .5]); direction /= np.linalg.norm(direction)
        for k in range(T):
            x[k+1] = lam[k]*x[k]+L[k]*float(direction@q[k])+disturbance[k]
            R[k+1] = lam[k]*R[k]+L[k]*U[k]+abs(disturbance[k])
        record(law+"_noncontracting_phase_tube", np.abs(x)-R, T+1)
        prod_errors = []
        for k in range(T+1):
            val = np.prod(lam[:k])*R[0]
            for j in range(k):
                val += np.prod(lam[j+1:k])*(L[j]*U[j]+abs(disturbance[j]))
            prod_errors.append(abs(val-R[k]))
        record(law+"_tube_product_identity", prod_errors, T+1)
    RESULTS["constructed_schedule"] = {
        "physical_steps": T,
        "active_updates": int(chi[:T].sum()),
        "last_application_snapshot": int(tau[-1]),
        "maximum_snapshot_age": int(np.max(t-tau)),
        "maximum_local_gain": float(lam.max()),
        "measured_robot_schedule": False,
    }


def rectangle_checks():
    """Worst-case comparison maps on 3000 independent legacy rectangles."""
    corners, mapped = [], []
    for _ in range(3000):
        a, c = RNG.uniform(.1,.9,2)
        b = RNG.uniform(.01,.5)
        kx = RNG.uniform(.05,.9)*a*c/b
        gamma = RNG.uniform(.01,1)
        eps, attenuation, nu, d0 = RNG.uniform(0,.1,4)
        v = eps+attenuation+nu/gamma
        delta = a*c-b*kx
        Xstar, Estar = (c*d0+b*v)/delta, (kx*d0+a*v)/delta
        scale = RNG.uniform(1,3)
        Xbar, Ebar = scale*Xstar, scale*Estar
        corners.extend([b*Ebar+d0-a*Xbar, kx*Xbar+v-c*Ebar])
        # Include the upper corner and random interior points.
        X = np.r_[Xbar, RNG.uniform(0,Xbar,19)]
        E = np.r_[Ebar, RNG.uniform(0,Ebar,19)]
        Xnext = (1-a)*X+b*E+d0
        Enext = (1-gamma*c)*E+gamma*(eps+attenuation+kx*X)+nu
        mapped.extend(np.r_[Xnext-Xbar, Enext-Ebar])
    record("legacy_rectangle_corner_feasibility", corners, len(corners))
    record("legacy_rectangle_invariance", mapped, len(mapped))


def mechanisms():
    # These simple exact cases show why each term cannot just be omitted.
    f, h, gamma, s = .6, .6, .08, .4
    full_hold = h
    target_gate_leak = (1-gamma)*h
    assert abs(full_hold-f) == 0 and abs(target_gate_leak-f) > .04
    h = 0.
    for _ in range(1000):
        h = (1-gamma)*h+gamma*s*f
    assert abs(h-s*f) < 1e-12
    assert abs(abs(h-f)-(1-s)*f) < 1e-12
    # A perfect current estimate does not cancel an old applied snapshot.
    current_E = abs(f-f)
    delayed_E = abs(f-0.)
    assert current_E == 0 and delayed_E == f
    RESULTS["necessary_terms_examples"] = {
        "full_hold_error_at_initially_exact_estimate": abs(full_hold-f),
        "target_only_gate_error": abs(target_gate_leak-f),
        "legacy_constant_target_error": abs(h-f),
        "legacy_attenuation_floor": (1-s)*f,
        "current_error_with_stale_zero_snapshot": current_E,
        "applied_error_with_stale_zero_snapshot": delayed_E,
        "active_updates_for_95pct_legacy_transient": int(np.ceil(np.log(.05)/np.log(1-gamma))),
    }


if __name__ == "__main__":
    projection_checks()
    scheduled_sequences()
    rectangle_checks()
    mechanisms()
    RESULTS["scope"] = "Constructed deterministic algebra and bound checks; no robot reruns or latency measurements."
    RESULTS["seed"] = 20260909
    RESULTS["status"] = "passed"
    (ROOT/"schedule_bounds_checks.json").write_text(json.dumps(RESULTS, indent=2)+"\n")
    checks = [v for v in RESULTS.values() if isinstance(v,dict) and "checks" in v]
    count = sum(v["checks"] for v in checks)
    worst = max(v["maximum_positive_violation"] for v in checks)
    memo = f"""# Schedule and attenuation verification\n\nAll {count:,} deterministic checks passed (seed 20260909; maximum positive\nroundoff residual {worst:.3g}; tolerance {TOL:g}). These are constructed numerical\nchecks of the formulas, not a substitute for the accompanying proofs.\n\n- 4,000 vector box-projection transitions per law test moving targets, full holds,\n  target-only gates, attenuation, and innovation updates.\n- Two 96-step sequences test onset, removal, drift, long holds, snapshot delays,\n  estimator and tube product identities, and an exact mismatch decomposition.\n  Their local transition factors include 1.15, so the finite-horizon check does\n  not impose strict contraction at every physical step.\n- 3,000 independently generated legacy invariant rectangles test the corner\n  conditions and 20 points per rectangle against the attenuation-aware comparison.\n- Exact illustrative cases show that target-only leakage differs from holding,\n  observation attenuation retains a steady bias, and a perfect current estimate\n  can coexist with a nonzero applied error when the correction snapshot is old.\n\nThe JSON records each check separately. All schedules, faults, and dynamics in\nthis script are synthetic. It supplies no measured robot update schedule,\nlatency, recovery time, task-success count, or empirical robustness certificate.\n"""
    (ROOT/"schedule_bounds_memo.md").write_text(memo)
    print(json.dumps({"status":"passed", "checks":count, "maximum_positive_violation":worst}))
