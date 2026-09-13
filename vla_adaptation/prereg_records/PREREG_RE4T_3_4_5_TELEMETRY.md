# Preregistration: tube propagation, estimator recursions and small-gain constants on logged
# LIBERO episodes (re4 theory plan, Parts 3, 4, 5; 2026-09-13)

**Written before the runs.** All three parts need per-step logs the earlier runs did not keep
(measured motion, residual, command, correction, joint state). One GPU batch on the pi0.5 LIBERO
server, every run with `--scenario-reset --telemetry --timing` and the section-0 records, headline
constants unless stated, `libero_spatial`, the headline 20 scenarios:

| run | purpose | fault |
|---|---|---|
| T1 headline, telemetry | 3, 4.2, 4.3, 5 | uniform +0.05, rotation corrected |
| T2 onset at step 40 | 4.1 (K-step transient) | uniform +0.05 from step 40 |
| T3 ramp over 60 steps | 4.4 (drift floor) | uniform +0.05 ramp, `--profile ramp --prof-p 60` |
| T4 innovation law, telemetry | 4.3 (innovation settle vs truth) | uniform +0.05 |

Truth (f_true) is logged for evaluation only; the estimator never sees it. Nominal replay: the
recorded corrected command u_k replayed through the healthy FIR (the paper's ξ^nom) gives the
predicted nominal increment; X_k = measured increment − predicted, in the Part 2.2 metric.

**Part 4 predictions.**
- 4.1 Onset transient: after onset, the residual's deviation from the constant-fault response
  is predicted by the K = 6-tap history of the FIR applied to the step in f; the measured
  residual matches that prediction with RMS error ≤ 25 % of the step's peak on the rotation
  channels, and the transient is gone (error ≤ 10 % of peak) by K = 6 responses after onset.
- 4.2 Recursion envelope: with the logged residual and gate, the propagated bound on the
  estimate error covers the measured estimate error on ≥ 90 % of steps after step 15; the
  deadzone ball ε + ‖S⁻¹‖δ covers the settled error on the gated cohort steps.
- 4.3 Fixed points: the attenuated settle predicted from the logged residual distribution,
  E[s_k z_k] with s_k = 1/(1 + ‖r_k‖²/ρ²), for ρ = 0.05, 0.15, 0.50, matches the stored ablation
  settles (0.027/0.011/0.027, 0.044/0.020/0.044, 0.048/0.022/0.049 on rx, ry, rz) within 20 %
  per channel; the innovation settle (T4) is within 10 % of the truth on rx and rz.
- 4.4 Drift floor: on the ramp, the measured estimate lag behind the ramp, after the initial
  transient, is within a factor of two of ν/α̲ with ν the ramp slope per step and α̲ the
  minimum logged effective gain γ·s_k.

**Part 3 prediction.** Propagating R_{k+1} = λ R_k + L·(applied-estimate error) + η with λ and
L from Part 1 and the Part 2.2 metric, and η the healthy residual's 90th percentile, the measured
X_k satisfies X_k ≤ R_k on ≥ 90 % of free-space steps, and every violation is at a contact step,
a domain exit, or a step whose calibration error exceeds the held-out band. Violations are
listed with their class; a violation outside those classes refutes the tube on this plant.

**Part 5 prediction.** Regressing the applied-correction error norm on (estimate error, X_k) over
the corrected T1 episodes gives (ε₀, k_E, k_X) with k_X small enough that a·c > b·k_X holds with
margin, a = 1 − λ from Part 1 on the certified metric, b = L̂. If a·c ≤ b·k_X on the deployed
cohort it is reported as-is: the coupling condition, not just the constants, is load-bearing.

**Refutation handling.** Each numbered prediction is scored separately; a failed one is
reported as primary with the measured value that failed it.
