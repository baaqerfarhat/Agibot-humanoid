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

---

## Outcome, libero_spatial block (2026-09-15, 02:40; `results/iclr_unified_v1/E2_core/libero_spatial_*`, `score_libero_spatial.json`; scorer v2, all seven arms complete on the 60 manifest keys, no duplicates)

| arm | success / 60 |
|---|---|
| healthy off / U / C | 60 / 58 / 60 |
| faulted off / U / C / same-mask oracle | 30 / 58 / 55 / 54 |

- **H1 (observation consistency): holds.** Mean |signed r_y observation error| over the last 50
  valid steps: U 0.0268, C 0.0173 — a **35.5 % reduction**, task-clustered ratio-bootstrap 95 %
  interval **[0.22, 0.48]** (both registered clauses; U's bias is above the 0.005 informativeness
  floor; no short or empty windows). Estimate error 0.0257 → 0.0164. Signed remaining r_y
  disturbance U +0.0235 (half the fault left), C +0.0029 (unbiased on average).
- **Task, spatial = the no-harm cell.** C − U −5.0 points, interval [−13.3, 0.0] (1 C-only, 4
  U-only successes); U − off +46.7 [+23.3, +70.0]; C − off +41.7 [+20.0, +63.3]. The same-mask
  oracle (exact rotation cancellation, translation disturbance left in) scores **54/60**, below
  both adaptive arms (oracle − U −6.7 [−13.3, 0]; oracle − C −1.7 [−11.7, +8.3]): on this suite an
  exact rotation correction with the translation offset still applied is not better than the
  estimator's partial one — the matched-authority oracle is not an upper bound here.
- **Healthy.** C − off 0.0 [0, 0] (60/60); U − off −3.3 points [−8.3, 0] (2 lost; the healthy
  estimate stays at zero under the corrected-channel deadzone on 58 episodes, max |f̂_ry| 0.021),
  neither below the −5-point target. Under C the healthy estimate reached 0.039 on one episode
  without a lost task.

Reading, before libero_10: the finite-horizon consistency constraint does what H1 asked — it
removes the r_y under-correction — and on the easy suite that buys nothing in task terms (a
five-point deficit whose interval reaches zero). Spatial was registered as the no-harm cell;
the primary task contrast is libero_10, running.

## Outcome, libero_10 block and the E2 decision (2026-09-15, 14:45; `libero_10_*`, `score_libero_10.json`; scorer v2, all seven arms complete on the 60 manifest keys, no duplicates)

| arm | success / 60 |
|---|---|
| healthy off / U / C | 56 / 53 / 52 |
| faulted off / U / C / same-mask oracle | 0 / 18 / 25 / **36** |

- **H1 (observation consistency): fails by the registered statistic on libero_10.** Mean |signed
  r_y observation error| over the last 50 valid steps: U 0.0252, C **0.0336** — a 33 % *increase*,
  ratio-bootstrap interval [−1.36, +0.40]. The signed remaining r_y disturbance nevertheless drops
  from +0.0255 (U, half the fault left) to +0.0056 (C): C is unbiased on average and scatters
  more per episode on the long-horizon suite (clipped fraction 0.05 vs 0.02). H1 therefore holds
  on spatial and fails on libero_10 as registered; the signed bias is removed on both.
- **H2 (task, primary): a positive count with an unresolved interval.** C − U = **+11.7 points**
  (25 vs 18; 12 C-only, 5 U-only; descriptive McNemar p = 0.14), task-clustered 95 % interval
  **[0.0, +25.0]** — the lower bound sits at zero, so by the registered rule this is not a
  demonstrated improvement. U − off +30.0 [+16.7, +46.7]; C − off +41.7 [+23.3, +61.7]. The same-mask
  oracle reaches 36/60: oracle − C +18.3 [+1.7, +35.0], oracle − U +30.0 [+10.0, +51.7]. On this
  suite the exact rotation correction *is* an upper reference, and a gap of 11 episodes remains
  above C.
- **Healthy: both adaptive arms fail the −5-point target's letter or sit on it.** U − off −5.0
  [−11.7, 0] (3 lost, 1 gained; on the target, not below); C − off **−6.7** [−16.7, 0] (5 lost, 1
  gained; below the target). Healthy estimates reach |f̂_ry| 0.045 (U) and 0.058 (C) at most; on
  the long-horizon suite the corrected-channel deadzone does not keep the estimate at zero as it
  did on spatial (healthy |obs bias| 0.0096 on both).

**E2 decision, as registered.** The consistency intervention removes the r_y under-correction on
both suites (signed remaining disturbance to ~0); it reduces the registered absolute-error
statistic on spatial (−35 %, interval excluding zero) and increases it on libero_10 (+33 %,
interval including zero). Its task effect on libero_10 is +7 of 60 with an interval touching
zero: **the identification improvement is supported, its task benefit is not demonstrated**,
which the prereg's interpretation clause anticipated. Healthy behaviour is worse than the
registered target on libero_10 for C and at the target for U; both are reported. No retuning.
Spatial oracle 54 < U 58 and libero_10 oracle 36 > C 25 say the two suites are limited by
different things: on spatial the estimator's partial correction already suffices; on libero_10
eleven episodes lie between the estimator and exact same-mask cancellation — the Q5
six-channel oracle (queued) asks whether the mask or something else holds the rest.
