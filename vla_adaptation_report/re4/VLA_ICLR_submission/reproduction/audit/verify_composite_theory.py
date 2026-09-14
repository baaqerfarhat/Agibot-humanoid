"""Bounded numerical stress checks of the proved composite inequalities.

These checks do not validate a VLA controller, excitation, or a task margin.
Run with Python + NumPy; writes a neighboring JSON audit summary.
"""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(902026)
max_dissipation_violation = -float("inf")
max_projection_violation = -float("inf")
max_jump_violation = -float("inf")
trials = 5000

for _ in range(trials):
    p = 3
    R = rng.normal(size=(p, p))
    Gamma = R @ R.T + 0.2 * np.eye(p)
    Ginv = np.linalg.inv(Gamma)
    R = rng.normal(size=(p, p))
    H = R @ R.T + 0.1 * np.eye(p)
    mu = np.linalg.eigvalsh(H)[0]
    gain_eigs = np.linalg.eigvalsh(Gamma)
    kc = float(rng.uniform(0.05, 5))
    lam = float(rng.uniform(0.05, 3))
    Vx = float(rng.uniform(0.0, 10))
    tilde = rng.normal(size=p)
    # The estimate is on the active face n^T theta <= 0, theta_hat=0.
    # Choosing n^T tilde >= 0 puts theta_star=-tilde in the set.
    normal = rng.normal(size=p)
    if normal @ tilde < 0:
        normal = -normal
    sigma = rng.normal(size=p)
    b = rng.normal(size=p)
    drift = rng.normal(size=p)
    a = float(rng.uniform(0, 2))
    velocity = Gamma @ (sigma - kc * (H @ tilde - b))
    projected = velocity - Gamma @ normal * max(0., normal @ velocity) / (normal @ Gamma @ normal)
    projection_value = tilde @ Ginv @ (projected - velocity)
    max_projection_violation = max(max_projection_violation, float(projection_value))
    Vtheta = 0.5 * tilde @ Ginv @ tilde
    V = Vx + Vtheta
    actual = -2 * lam * Vx - sigma @ tilde + a * np.sqrt(2 * Vx) + tilde @ Ginv @ (projected - drift)
    beta = min(lam, kc * mu * gain_eigs[0])
    Q = np.sqrt(a*a + gain_eigs[-1] * (kc * np.linalg.norm(b) + np.linalg.norm(Ginv, 2) * np.linalg.norm(drift))**2)
    upper = -2 * beta * V + Q * np.sqrt(2 * V)
    max_dissipation_violation = max(max_dissipation_violation, float(actual - upper))
    jump = rng.normal(size=p)
    Wbefore = np.sqrt(2 * V)
    Wafter = np.sqrt(2 * Vx + (tilde - jump) @ Ginv @ (tilde - jump))
    J = np.sqrt(jump @ Ginv @ jump)
    max_jump_violation = max(max_jump_violation, float(Wafter - Wbefore - J))

# An exact scalar Euler example tests both sides of the step-size boundary.
beta_scalar = 2.0
h_stable, h_unstable = 0.3, 1.25
q_stable = abs(1 - beta_scalar * h_stable)
q_unstable = abs(1 - beta_scalar * h_unstable)
assert q_stable < 1 < q_unstable

# Full-rank memory from an old regime can identify the wrong parameter.
H_old, theta_old, theta_new = 4.0, 1.0, -1.0
h_old = H_old * theta_old
stale_bias = h_old - H_old * theta_new
stale_equilibrium_error = abs(h_old / H_old - theta_new)
assert stale_equilibrium_error == abs(stale_bias) / H_old == 2.0

assert max_projection_violation < 1e-10
assert max_dissipation_violation < 1e-10
assert max_jump_violation < 1e-10
summary = {
    "purpose": "Numerical checks of derived inequalities, not experimental controller validation",
    "seed": 902026,
    "randomized_trials": trials,
    "max_projection_inequality_excess": max_projection_violation,
    "max_composite_dissipation_bound_excess": max_dissipation_violation,
    "max_parameter_jump_bound_excess": max_jump_violation,
    "scalar_euler_stable_factor": q_stable,
    "scalar_euler_unstable_factor": q_unstable,
    "stale_memory_full_rank_mu": H_old,
    "stale_memory_bias": stale_bias,
    "stale_memory_equilibrium_parameter_error": stale_equilibrium_error,
    "result": "passed",
}
target = Path(__file__).with_name("composite_theory_checks.json")
target.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
