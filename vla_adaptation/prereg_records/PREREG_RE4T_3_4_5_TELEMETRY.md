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

---

## Outcome, T1 headline telemetry (2026-09-13; `results/re4_theory/telemetry/T1_headline/`)

T1 reproduced the headline cell (frozen 9/20, adaptive 16/20; the shipped result is 9/20 →
16/20). Scored with `openpi/re4_theory/recursions.py` (`part_3.json`, `part_4.2.json`,
`part_4.3.json`, `part_5.json`):

- **Part 3 (tube): letter passed, substance refuted.** Coverage 1.00, no violations to
  classify. But with λ = 1.00 from Part 1 the recursion never forgets: the bound grows by η =
  0.18 (the healthy residual's 90th percentile, 2.2 metric) at every step even at zero applied
  error, and by the end of an episode the median bound is **98×** the episode's largest measured
  residual (median bound/measurement over all steps 223×, tenth percentile 51×). The tube is
  vacuous on this plant. It is the same fact Part 1 registered — no contraction rate on the
  Panda under OSC_POSE — read from the other end; reported as primary.
- **Part 4.2 (estimate-error envelope): passed.** With the logged residual, gate and
  attenuation the propagated bound covers the measured estimate error on 100 % of steps, before
  and after step 15.
- **Part 4.3 (attenuated fixed points): passed.** The settle predicted from the logged residual
  distribution, E[s_k z_k], matches the stored ablation settles within 20 % on all nine
  (ρ, channel) pairs: predicted 0.027/0.013/0.028 vs stored 0.027/0.011/0.027 at ρ = 0.05,
  0.042/0.022/0.044 vs 0.044/0.020/0.044 at ρ = 0.15, 0.046/0.026/0.050 vs 0.048/0.022/0.049
  at ρ = 0.50 (rx/ry/rz). The run's own measured settle at ρ = 0.15 is 0.043/0.020/0.044, i.e.
  87 %/41 %/88 % of the +0.05 truth. The innovation-law half of 4.3 waits on T4.
- **Part 5 (small-gain): refuted.** Regression over the corrected T1 episodes gives ε₀ =
  0.0008, k_E = 0.012, k_X = 0.0086; with a = 1 − λ = 9×10⁻⁵, b = L̂ = 0.111, c = γ = 0.08 the
  condition a·c > b·k_X fails (7×10⁻⁶ vs 9.6×10⁻⁴, margin −9.5×10⁻⁴). The coupling constants
  are small; the condition fails because the plant term a is essentially zero, again λ = 1.00.
  The certificate that makes the composed loop safe cannot be issued on this plant from these
  constants; the empirical repair stands on its paired counts alone.

Deviation noted: the telemetry's `correction` vector carries the gripper channel (7 entries);
the scorer uses the six task channels (fixed in the scorer, not the logs).

## Outcome, T2 onset at step 40 (2026-09-13; `results/re4_theory/telemetry/T2_onset40/`)

Cell: frozen 18/20, adaptive 20/20 (a fault that starts at step 40 damages less than one present
from the start). Scored with `recursions.py --part 4.1 --onset 40` (`part_4.1.json`, 20 onsets).

- **Part 4.1 (onset transient): the registered statistic fails.** Per-episode RMS error of the
  measured residual against the FIR partial-sum prediction is **0.74** of the step's peak over
  the first K = 6 responses (registered ≤ 0.25) and **1.09** after (registered ≤ 0.10). Reported
  as primary. Two causes, both visible in the logs:
  1. the registered statistic is a per-episode, per-step RMS, and the healthy plant's step
     noise (residual p90 = 0.18 in the certified metric) is of the order of the 0.01 step it is
     asked to resolve, so the statistic cannot pass on any single episode; and
  2. on **r_y** the measured residual keeps rising to **3.0×** the FIR's predicted plateau,
     because the FIR's r_y DC gain (0.10) is a third of the probed sensitivity (M's r_y entry
     0.28); the same model error Part 6 diagnosed, now seen in the transient.
- **Secondary, not registered (episode-mean response per channel):** on r_x and r_z the mean
  response follows the FIR partial sums — plateau ratio measured/predicted 1.01 and 0.92, RMS
  error after K = 6 responses 2 % and 9 % of the peak (first six responses 19 % and 10 %; the
  first measured response leads the prediction by one step). On r_y the plateau ratio is 3.02
  and the error after K is 199 % of the predicted peak. The K-step transient picture is right on
  the two channels the FIR models and wrong on the one it does not; the FIR's DC gains
  (0.23/0.10/0.25) against M's (0.25/0.28/0.24) say which is which in advance.

## Outcome, T3 ramp over 60 steps (2026-09-13; `results/re4_theory/telemetry/T3_ramp60/`)

Cell: frozen 18/20, adaptive 19/20. Scored with `recursions.py --part 4.4` (`part_4.4.json`).

- **Part 4.4 (drift floor): passed.** With ν = 0.05/60 per step and α̲ the minimum logged
  effective gain γ·s_k (median over episodes 0.040), ν/α̲ = 0.021; the measured estimate lag
  after the initial transient is 0.016 (median over episodes of the mean over r_x, r_y, r_z),
  a ratio of 0.76, inside the registered factor of two. Read as an upper bound, as the recursion
  states it: during the ramp itself the per-channel lag is 0.013 / 0.022 / 0.012 (r_x / r_y /
  r_z), i.e. 0.62× and 0.58× the bound on the two well-modelled channels and 1.06× on r_y.
  Once the ramp has finished the remaining error is the attenuated fixed point's shortfall
  (0.008 / 0.029 / 0.007), so the drift component proper on r_x, r_z is ≈ 0.005 per channel —
  a quarter of ν/α̲. The bound holds and is loose by that factor on this cohort.

## Outcome, T4 innovation law (2026-09-13; `results/re4_theory/telemetry/T4_innov/`)

Cell: frozen 9/20, adaptive 19/20. Scored with `recursions.py --part 4.3` (`part_4.3.json`).

- **Part 4.3, innovation half: passed on r_x and r_z, and r_y is the model-error floor.** The
  innovation law's settle (last 50 steps, median over 20 episodes) is 0.048 / 0.023 / 0.049
  against the +0.05 truth: **95 % and 98 %** on r_x and r_z (registered: within 10 %), 45 % on
  r_y. The attenuated law's settle on the same scenarios (T1) was 87 % / 41 % / 88 %; removing
  the attenuation removes the r_x, r_z shortfall and leaves r_y where it was, so r_y's deficit
  is not the law's fixed point but the plant model's (Part 6, T2). Time to 80 % of the truth on
  r_x: median 29 steps (range 15–51).
- Not registered, noted: the innovation cell scored 19/20 against the attenuated 16/20 on the
  same 20 scenarios (3 gained, 0 lost), but the policy sampler was unpinned in these runs (the
  frozen arms differ per episode between T1 and T4), so this is a between-run comparison at
  n = 20, consistent with the earlier "indistinguishable" reading (0.12σ) and not evidence
  against it. Part 8.2 is the pinned design for this question.

**Summary of Parts 3, 4, 5 across T1–T4.** Confirmed: 4.2 (error envelope, 100 %), 4.3
(attenuated fixed points within 20 % on all nine pairs; innovation within 10 % of truth on r_x,
r_z), 4.4 (drift floor, ratio 0.76). Refuted or vacuous: 3 (tube covers by being 98× the
measurement), 5 (small-gain fails, a = 1 − λ = 9×10⁻⁵), 4.1's registered per-episode statistic
(0.74 and 1.09 of peak; the episode-mean transient matches the FIR on r_x, r_z and is 3× on
r_y). Every estimator-side prediction held; every plant-side certificate that needs a
contraction rate did not, because the Panda under OSC_POSE has none (Part 1).
