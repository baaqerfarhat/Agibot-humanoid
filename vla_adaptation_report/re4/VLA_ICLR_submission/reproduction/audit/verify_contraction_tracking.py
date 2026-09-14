"""Reproduce the globally certified *constructed servo* example and figure.

Run from any directory: python audit/verify_contraction_tracking.py
Requires NumPy and Matplotlib. No VLA/robot outcome is generated or changed.
The metric certificate is the symbolic identity A.T P A = a**2 P.
Numerical assertions check the implementation of the proved recursions;
sampled tests are not used to infer a metric or uniform system constants.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, DT, K = 600, 0.02, 6
A_RATE, PHI = 0.92, 0.04
GAMMA, RHO, DEADZONE = 0.08, 0.015, 0.0008
P = np.diag([1.0, 4.0])
P_HALF = np.diag([1.0, 2.0])
ROT = np.array([[np.cos(PHI), -np.sin(PHI)],
                [np.sin(PHI), np.cos(PHI)]])
A = A_RATE * np.diag([1.0, 0.5]) @ ROT @ P_HALF
S = np.diag([1.0, 0.7])
WEIGHTS = np.array([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02])
BIAS = np.array([0.0003, -0.0002])
FAULT = np.array([0.006, 0.0025])
BOX, COMMAND_LIMIT = 0.015, 0.025
L_INPUT, TOL = 2.0, 1e-11


def norm(x):
    return np.linalg.norm(x, axis=-1)


def fault_at(k):
    return FAULT.copy() if k >= 50 else np.zeros(2)


def snapshot_at(k):
    # A new chunk at steps 0,5,... uses the estimate available two steps ago.
    return max(0, 5 * (k // 5) - 2)


def updates_at(k):
    # Clock-defined outage and periodic missed sample, independent of truth.
    return not (65 <= k < 100 or k % 10 == 9)


def simulate(name, family, mask):
    D = np.asarray(mask, dtype=float)
    f = np.array([fault_at(k) for k in range(N + 1)])
    x, xr, h = (np.zeros((N + 1, 2)) for _ in range(3))
    B, radius = np.zeros(N + 1), np.zeros(N + 1)
    u, v, z, residual, q, zeta = (np.zeros((N, 2)) for _ in range(6))
    Eapp, Bapp, eps, age = (np.zeros(N) for _ in range(4))
    tau = np.array([snapshot_at(k) for k in range(N)], dtype=int)
    active = np.array([updates_at(k) and family != 'frozen'
                       for k in range(N)], dtype=bool)
    gains, atten, projections = (np.zeros(N) for _ in range(3))
    violations = {key: -np.inf for key in
                  ('observation_identity', 'observation_bound',
                   'selected_estimator_bound', 'applied_snapshot_bound',
                   'mismatch_bound', 'one_step_metric_bound',
                   'trajectory_envelope')}
    for k in range(N):
        time = DT * k
        target = np.array([0.06 * np.sin(0.6 * time),
                           0.03 * np.cos(0.4 * time)])
        nominal = (np.eye(2) - A) @ target
        intended = nominal - D * h[tau[k]]
        u[k] = np.clip(intended, -COMMAND_LIMIT, COMMAND_LIMIT)
        zeta[k] = u[k] - intended
        q[k] = f[k] - D * h[tau[k]] + zeta[k]
        x[k + 1] = A @ x[k] + u[k] + f[k]
        xr[k + 1] = A @ xr[k] + nominal
        # Measured after response: v_k=x_{k+1}-A x_k=u_k+f_k.
        v[k] = x[k + 1] - A @ x[k]
        y, predicted, filtered_fault = S @ BIAS, np.zeros(2), np.zeros(2)
        for ell, weight in enumerate(WEIGHTS):
            j = k - ell
            if j >= 0:
                y = y + weight * S @ v[j]
                predicted = predicted + weight * S @ u[j]
                filtered_fault += weight * f[j]
        residual[k] = y - predicted
        z[k] = np.linalg.solve(S, residual[k])
        history = filtered_fault - f[k]
        w = z[k] - f[k]
        # A priori model bound, with no regression fit to observed errors.
        eps[k] = norm(D * history) + norm(D * BIAS)
        delta_f = norm(D * (f[k + 1] - f[k]))
        if family == 'attenuated':
            atten[k] = float(norm(residual[k]) > DEADZONE) / (
                1.0 + norm(residual[k]) ** 2 / RHO ** 2)
            gains[k] = GAMMA
            raw = (1.0 - GAMMA) * h[k] + GAMMA * atten[k] * z[k]
            forcing = (1.0 - atten[k]) * norm(D * f[k]) + atten[k] * eps[k]
        elif family == 'innovation':
            # This is the actual estimate-dependent normalizer, not a fixed gain.
            gains[k] = GAMMA / (1.0 + norm(S @ (z[k] - h[k])) ** 2 / RHO ** 2)
            raw = h[k] + gains[k] * (z[k] - h[k])
            forcing = eps[k]
        else:
            raw = h[k]
            forcing = 0.0
        projected = np.clip(raw, -BOX, BOX)
        projections[k] = norm(projected - raw) if active[k] else 0.0
        h[k + 1] = projected if active[k] else h[k]
        g = float(active[k]) * gains[k]
        B[k + 1] = (1.0 - g) * B[k] + g * forcing + delta_f
        j0 = tau[k]
        increments = np.sum(norm(np.diff(h[j0:k + 1], axis=0) * D))
        drift = np.sum(norm(np.diff(f[j0:k + 1], axis=0) * D))
        Bapp[k] = min(B[k] + increments, B[j0] + drift)
        Eapp[k] = norm(D * (h[j0] - f[k]))
        age[k] = k - j0
        q_bound = norm((1.0 - D) * f[k]) + Bapp[k] + norm(zeta[k])
        radius[k + 1] = A_RATE * radius[k] + L_INPUT * q_bound
        state_error = norm(P_HALF @ (x[k + 1] - xr[k + 1]))
        previous_error = norm(P_HALF @ (x[k] - xr[k]))
        excess = dict(
            observation_identity=norm(z[k] - (filtered_fault + BIAS)),
            observation_bound=norm(D * w) - eps[k],
            selected_estimator_bound=norm(D * (h[k + 1] - f[k + 1])) - B[k + 1],
            applied_snapshot_bound=Eapp[k] - Bapp[k],
            mismatch_bound=norm(q[k]) - q_bound,
            one_step_metric_bound=state_error - A_RATE * previous_error - L_INPUT * norm(q[k]),
            trajectory_envelope=state_error - radius[k + 1])
        for key, value in excess.items():
            violations[key] = max(violations[key], float(value))
    actual = norm((x - xr) @ P_HALF)
    hlim = np.zeros(2)
    slim = 0.0
    if family == 'attenuated':
        slim = 1.0 / (1.0 + norm(S @ (FAULT + BIAS)) ** 2 / RHO ** 2)
        hlim = np.clip(slim * (FAULT + BIAS), -BOX, BOX)
        selected_floor = ((1.0 - slim) * norm(D * FAULT)
                          + slim * norm(D * BIAS))
    elif family == 'innovation':
        hlim = np.clip(FAULT + BIAS, -BOX, BOX)
        selected_floor = norm(D * BIAS)
    else:
        selected_floor = 0.0
    qlim = FAULT - D * hlim
    actual_floor = norm(P_HALF @ np.linalg.solve(np.eye(2) - A, qlim))
    certified_floor = L_INPUT * (norm((1.0 - D) * FAULT) + selected_floor) / (1.0 - A_RATE)
    assert all(v <= TOL for v in violations.values()), (name, violations)
    assert np.max(age) == 6
    assert np.max(norm(zeta)) == 0.0
    assert np.max(projections) == 0.0
    assert np.max(np.abs(f)) <= BOX and np.max(np.abs(h)) <= BOX
    recovery = None
    # Error below 12 mm continuously through the remaining horizon;
    # this exceeds a 1 s dwell, with the time indexed at state acquisition.
    threshold, dwell = 0.012, 50
    for k in range(50, N + 1 - dwell):
        if np.all(actual[k:] <= threshold):
            recovery = DT * (k - 50)
            break
    certified_recovery = next((DT * (k - 50) for k in range(50, N + 1 - dwell)
                               if np.all(radius[k:] <= threshold)), None)
    result = dict(
        name=name, family=family, correction_mask=D.tolist(),
        total_commands=N, active_updates=int(active.sum()),
        zero_observation_gates_during_updates=int(np.sum(active & (atten == 0))) if family == 'attenuated' else None,
        maximum_snapshot_age_steps=int(age.max()),
        maximum_snapshot_age_seconds=float(DT * age.max()),
        projection_events=int(np.sum(projections > 0)),
        command_clipping_events=int(np.sum(norm(zeta) > 0)),
        minimum_realized_gain_on_active_steps=float(gains[active].min()) if active.any() else None,
        steady_attenuation=slim if family == 'attenuated' else None,
        analytic_estimate_limit_m=hlim.tolist(),
        analytic_metric_tracking_floor_m=float(actual_floor),
        analytic_propagated_envelope_floor_m=float(certified_floor),
        final_metric_error_m=float(actual[-1]), final_envelope_m=float(radius[-1]),
        post_fault_integrated_squared_metric_error_m2_s=float(DT * np.sum(actual[50:] ** 2)),
        recovery_below_12mm_seconds_after_fault=recovery,
        envelope_below_12mm_seconds_after_fault=certified_recovery,
        maximum_positive_assertion_excess={key: max(0.0, value) for key, value in violations.items()})
    arrays = dict(x=x, reference=xr, estimate=h, true_fault=f, commands=u,
                  response_increment=v, observation=z, residual=residual,
                  mismatch=q, realization_error=zeta, E_D=norm((h-f)*D),
                  B_D=B, E_D_applied=Eapp, B_D_applied=Bapp,
                  epsilon_D=eps, metric_error=actual, envelope=radius,
                  snapshot_index=tau, snapshot_age_steps=age, update_active=active,
                  realized_gain=gains, observation_attenuation=atten,
                  projection_displacement=projections)
    return result, arrays


def make_figure(runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                         'axes.titlesize': 9, 'legend.fontsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.35))
    colors = {'frozen': '#62666c', 'attenuated': '#bf6433',
              'innovation': '#2465a4', 'restricted': '#6c4b9a'}
    t, tc = DT * np.arange(N + 1), DT * np.arange(N)
    ax = axs[0, 0]
    ax.step(t, runs['innovation']['true_fault'][:, 0]*1000, where='post', color='black', lw=1.0, label='Fault')
    # Response k is acquired at t_{k+1}; it cannot be plotted at command start t_k.
    ax.step(tc + DT, runs['innovation']['observation'][:, 0]*1000, where='post', color='#718579', lw=1.0, ls=':', label='FIR observation')
    for name, label in [('attenuated', 'Attenuated'), ('innovation', 'Innovation')]:
        ax.step(t, runs[name]['estimate'][:, 0]*1000, where='post', color=colors[name], lw=1.3, label=label)
    ax.set(title='(a) Causal identification and scheduled hold', ylabel='First coordinate (mm)', xlim=(0.75, 4.0))
    ax.axvspan(65*DT, 100*DT, color='#888888', alpha=.13, lw=0)
    ax.text(1.64, 1.1, 'Full hold', ha='center', fontsize=7, color='#555555')
    ax.legend(loc='lower right', frameon=False, ncol=2, columnspacing=.7)
    ax = axs[0, 1]
    for name, label in [('frozen', 'Frozen'), ('attenuated', 'Attenuated'), ('innovation', 'Innovation')]:
        ax.plot(t, runs[name]['metric_error']*1000, color=colors[name], lw=1.3, label=label)
        ax.plot(t, runs[name]['envelope']*1000, color=colors[name], lw=1.0, ls='--')
    ax.set(title='(b) Certified execution-tracking tubes', ylabel='Metric tracking error (mm)', xlim=(0, 6))
    ax.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, .71))
    ax.text(.03, .92, 'Solid: actual\nDashed: bound', transform=ax.transAxes, va='top', fontsize=7)
    ax = axs[1, 0]
    for name, label in [('attenuated', 'Attenuated'), ('innovation', 'Innovation')]:
        ax.plot(tc, runs[name]['E_D_applied']*1000, color=colors[name], lw=1.3, label=label)
        ax.plot(tc, runs[name]['B_D_applied']*1000, color=colors[name], lw=1.0, ls='--')
    ax.axvspan(65*DT, 100*DT, color='#888888', alpha=.13, lw=0)
    ax.set(title='(c) Error in the applied estimate snapshot', ylabel='Selected error (mm)', xlim=(.75, 4))
    ax.legend(frameon=False, loc='upper right')
    ax = axs[1, 1]
    for name, label in [('restricted', 'Only coordinate 1 corrected'), ('innovation', 'Both coordinates corrected')]:
        ax.plot(t, runs[name]['metric_error']*1000, color=colors[name], lw=1.3, label=label)
        ax.plot(t, runs[name]['envelope']*1000, color=colors[name], lw=1.0, ls='--')
    ax.set(title='(d) Insufficient support leaves a larger floor', ylabel='Metric tracking error (mm)', xlim=(0, 6))
    ax.legend(frameon=False, loc='upper right')
    for ax in axs.flat:
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=.16, linewidth=.6)
        ax.axvline(1.0, color='#999999', lw=.7, ls=':')
    fig.tight_layout(pad=.7, w_pad=1.25, h_pad=1.2)
    fig.savefig(ROOT / 'paper' / 'fig_contraction_tracking.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'audit' / 'fig_contraction_tracking_preview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    results, arrays = {}, {}
    for name, family, mask in [('frozen', 'frozen', [0, 0]),
                                ('attenuated', 'attenuated', [1, 1]),
                                ('innovation', 'innovation', [1, 1]),
                                ('restricted', 'innovation', [1, 0])]:
        results[name], arrays[name] = simulate(name, family, mask)
    metric_residual = A.T @ P @ A - A_RATE**2 * P
    assert np.linalg.norm(metric_residual, 2) < TOL
    assert abs(WEIGHTS.sum() - 1.0) < TOL
    all_time_intended_command_bound = np.abs(np.eye(2) - A) @ np.array([0.06, 0.03]) + BOX
    assert np.all(all_time_intended_command_bound < COMMAND_LIMIT)
    all_time_gain_lower_bound = GAMMA / (1 + (2*np.sqrt(2)*BOX/RHO)**2)
    assert all_time_gain_lower_bound > 0
    # Exact onset identity tested separately from simulation-generated sensors.
    onset_excess = []
    for lag in range(K + 1):
        history = sum(WEIGHTS[j] * fault_at(50+lag-j) for j in range(K+1)) - FAULT
        exact = -sum(WEIGHTS[j] for j in range(lag+1, K+1)) * FAULT
        onset_excess.append(float(norm(history - exact)))
    assert max(onset_excess) <= TOL
    report = dict(
        scope='Constructed globally certified execution servo; not a VLA evaluation or full adaptive-system contraction certificate.',
        proof='A=a P^{-1/2} Rot(phi) P^{1/2}; orthogonality yields A^T P A=a^2 P for every x.',
        dt_seconds=DT, horizon_seconds=N*DT, number_commands=N,
        nominal_contraction_factor=A_RATE,
        equivalent_nominal_rate_per_second=float(-np.log(A_RATE)/DT),
        metric=P.tolist(), nominal_matrix=A.tolist(), input_metric_gain=L_INPUT,
        sensitivity=S.tolist(), residual_selector='I_2', fir_weights=WEIGHTS.tolist(),
        post_fault_offset_m=FAULT.tolist(), onset_command_step=50,
        fixed_observation_bias_m=BIAS.tolist(), adaptation_gain=GAMMA,
        normalizer_scale_m=RHO, observation_deadzone_m=DEADZONE,
        estimate_box_m=[-BOX, BOX], command_limits_m=[-COMMAND_LIMIT, COMMAND_LIMIT],
        all_time_absolute_intended_command_bound_m=all_time_intended_command_bound.tolist(),
        analytical_innovation_gain_lower_bound=all_time_gain_lower_bound,
        initial_state_m=[0.0, 0.0], initial_reference_m=[0.0, 0.0], initial_estimate_m=[0.0, 0.0],
        prehistory='u_k=f_k=v_k=0 for k<0; sensing bias enters after the FIR.',
        update_schedule='chi_k=0 for 65<=k<100 or k mod 10=9; otherwise 1, except frozen always 0.',
        application_schedule='tau_k=max(0,5 floor(k/5)-2)',
        observation_acquisition_times='Observation z_k is acquired at t_{k+1}=DT*(k+1); plot uses this timestamp.',
        numerical_assertion_tolerance=TOL,
        numerical_metric_identity_spectral_error=float(np.linalg.norm(metric_residual, 2)),
        numerical_fault_onset_identity_error=max(onset_excess),
        bound_inputs='Analytical fault-history and bias magnitude, true drift, declared gains, and logged estimate increments. These are constructed-model bounds, not fitted VLA quantities or a deployable online monitor.',
        runs=results)
    (ROOT / 'audit' / 'contraction_tracking_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'audit' / 'contraction_tracking_trajectories.npz',
                        **{name+'__'+key: value for name, data in arrays.items() for key, value in data.items()})
    make_figure(arrays)
    print(json.dumps({'metric_identity_error': report['numerical_metric_identity_spectral_error'],
                      'all_assertions_passed': True,
                      'floors_mm': {name: [1000*v['analytic_metric_tracking_floor_m'],
                                           1000*v['analytic_propagated_envelope_floor_m']]
                                    for name, v in results.items()}}, indent=2))


if __name__ == '__main__':
    main()
