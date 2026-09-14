"""Constructed checks for the execution-scope clarification; no VLA data.

The PD metric has an exact rational Lyapunov identity. The feedback example
compares an unstable closed loop with its same-command nominal response.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
A_exact = [[F(0), F(1)], [F(-4), F(-3)]]
P_exact = [[F(29, 24), F(1, 8)], [F(1, 8), F(5, 24)]]
identity = [[sum(A_exact[k][i]*P_exact[k][j] + P_exact[i][k]*A_exact[k][j]
                  for k in range(2)) for j in range(2)] for i in range(2)]
assert identity == [[F(-1), F(0)], [F(0), F(-1)]]
A = np.array(A_exact, dtype=float)
P = np.array(P_exact, dtype=float)
w, U = np.linalg.eigh(P)
assert np.all(w > 0)
assert np.all(np.real(np.linalg.eigvals(A)) < 0)
invroot = (U / np.sqrt(w)) @ U.T
rate = float(np.linalg.eigvalsh(invroot @ invroot)[0] / 2)
metric_defect = A.T @ P + P @ A + 2 * rate * P
assert np.linalg.eigvalsh(metric_defect)[-1] < 1e-12

# Actual feedback r=2*z gives z=e^t. The nominal physical response to the
# SAME supplied r(t)=2*e^t, initialized at zero, is z_r=e^t-e^{-t}.
t = np.linspace(0.0, 4.0, 401)
z = np.exp(t)
z_r = np.exp(t) - np.exp(-t)
r = 2 * z
reference_derivative = np.exp(t) + np.exp(-t)
reference_ode_error = float(np.max(np.abs(reference_derivative - (-z_r + r))))
tracking_error = float(np.max(np.abs((z - z_r) - np.exp(-t))))
assert reference_ode_error < 2e-14
assert tracking_error < 4e-15
assert z[-1] > z[0] and abs(z[-1] - z_r[-1]) < abs(z[0] - z_r[0])
result = {
    'scope': 'Constructed PD metric and shared-command counterexample; no VLA validation',
    'pd': {
        'Kp': 4, 'Kd': 3, 'Q_PD': [[1, 0], [0, 1]],
        'P_exact': [[str(v) for v in row] for row in P_exact],
        'exact_lyapunov_identity': True,
        'A_cl_eigenvalues': [{'real': float(z.real), 'imag': float(z.imag)} for z in np.linalg.eigvals(A)],
        'metric_eigenvalues': w.tolist(),
        'contraction_rate': rate,
        'max_metric_inequality_eigenvalue': float(np.linalg.eigvalsh(metric_defect)[-1]),
    },
    'common_command_counterexample': {
        'actual_state': 'z(t)=exp(t), r(t)=2*exp(t)',
        'nominal_response': 'z_r(t)=exp(t)-exp(-t)',
        'execution_error': 'z(t)-z_r(t)=exp(-t)',
        'full_feedback_growth_rate': 1,
        'execution_contraction_rate': 1,
        'max_reference_ode_error': reference_ode_error,
        'max_execution_error_formula_defect': tracking_error,
    },
    'all_checks_pass': True,
}
(HERE/'execution_scope_checks.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
