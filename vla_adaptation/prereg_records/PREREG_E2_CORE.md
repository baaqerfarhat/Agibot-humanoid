# Preregistration: E2 core confirmation — model/finite-horizon consistency intervention with
# matched data and correction budgets (2026-09-14, before the runs)

Per `iclr2027/EXPERIMENT_PLAN.md` E2, `PREREG_UNIFIED_PARTITIONS.md`, `PREREG_E2_PROBE_QUALIFICATION.md`.

**Two hypotheses, separately.** H1: enforcing the separately measured finite-horizon r_y
response in the predictor reduces the r_y observation/estimate bias. H2: that reduction improves
long-horizon task outcome. Better prediction fit alone satisfies neither.

**Predictors.** U: the deployed per-axis FIR (K = 6, ridge 0.01, intercept) fitted on the 30 E2
fit episodes (`sources/e2_fit_qual_spatial_init40_43.json`, indices 0–29 = states 40–42).
C: identical, with the rotation-y command-tap sum constrained to **0.254** (`--dc-gain 4=0.254`,
the qualified finite-horizon value; other channels' taps are refitted-identical, verified).
Sensitivity M for the estimator: the historical open-loop probe (`phase05/openloop_so3.json`) in
both. Law (frozen, same in U and C): innovation, normaliser over the corrected channels,
γ = 0.08, δ = 0.008, ρ = 0.15, κ = 0.15, correction mask {3, 4, 5}, `--scenario-reset`. This
intervention uses the innovation law and does not isolate the historical legacy law.

**Scenarios.** `results/iclr_unified_v1/manifests/{libero_spatial,libero_10}_E2_core.json`:
ten tasks × two unused stored states (spatial 49, 44; libero_10 41, 40) × three sampler seeds
(explicit server schedule `fold_in(fold_in(key(seed), episode), call)`, re-issued per episode)
= **120 paired keys**. Suite horizons and five-step chunks unchanged.

**Seven arms per key (840 rollouts; frozen arms run once per condition, `--arms`).** Healthy:
off, U, C. Uniform +0.05 on six channels: off, U, C, same-mask oracle (`--static-corr` −0.05 on
all six, masked to {3, 4, 5}; the translation disturbance remains: not a full-recovery bound).
Telemetry on every adaptive arm.

**Primary metrics and decisions (fixed now).**
1. *Observation consistency (H1):* per episode, the signed r_y error of M⁻¹r and of the estimate
   (both normalised action units) over the last 50 valid steps (episodes shorter than 50 valid
   steps reported separately, not dropped). **C reduces the absolute signed r_y observation bias
   by ≥ 25 % relative to U, with a task-clustered (10 tasks per suite, all keys retained) 95 %
   bootstrap interval excluding zero.** If U's aggregate bias is below 0.005 the relative test is
   uninformative and absolute errors are reported.
2. *Task consequence (H2):* **C − U success on libero_10** (60 keys) is the primary task contrast;
   improvement requires a positive task-clustered 95 % interval, a positive count alone is not
   improvement. Spatial is the no-harm cell. Healthy: a C-minus-off or U-minus-off change below
   −5 percentage points fails the practical no-harm target (intervals reported; meeting the point
   target does not prove noninferiority). Also reported: both-success / fixed / broken / neither
   per contrast, oracle gaps, and the exact McNemar as a descriptive matched check only.
3. *Mechanism:* signed remaining command disturbance, correction realisation and clipping, and
   the full stationary/residual equation of the E0 note rather than a DC-gain ratio prediction.

**Refutations.** A qualified constraint that does not improve the r_y observation bias refutes
the consistency intervention. Better bias without demonstrated task improvement leaves the
recovery mechanism unsupported; a precise zero or negative task effect is evidence against an
identification-only explanation, a wide interval is inconclusive. Worse healthy behaviour is
reported even if faulted success rises. C is not retuned inside this table.

**Order and budget.** spatial arms a–g then libero_10 arms a–g (≈ 24 GPU h). Pilot for this
design: `PREREG_DC_CONSTRAINED_PLANT.md` (legacy law, historical probe, shipped scenarios).

*Note added before any confirmation rollout (22:17):* on the 30-episode E2 fit partition U's
rotation tap sums come out 0.220 / **0.146** / 0.242 (r_x / r_y / r_z), so the gap C closes on r_y
is 0.146 → 0.254 (1.7×), smaller than the 0.103 → 0.276 (2.7×) of the shipped three-episode fit.
The other five channels' taps are identical between U and C (verified numerically).
