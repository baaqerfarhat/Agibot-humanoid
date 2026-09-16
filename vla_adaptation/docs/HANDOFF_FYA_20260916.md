# Hand-off to the writing session: FrozenYet Adaptive recovery campaign (16 Sep 2026)

For: whoever revises `papers/frozen_yet_adaptive/` before the 18 Sep abstract and 25 Sep paper deadlines.
Everything below is scored and committed; nothing is running. Registration with outcomes:
`prereg_records/PREREG_FYA_RECOVERY_DEADLINE_V1.md` (§§9–11). Record: `docs/ADAPTIVE_CONTROL_VLA.md` §§63–65.
Data root: `results/frozen_yet_adaptive_deadline_v1/` (runs stored `.json.gz`; scorers read them directly).

## What was run (EXPERIMENT_PLAN.md, all stages)

| Stage | Done | Files |
|---|---|---|
| 0: bundle, floor check, driver checks, dev replays, leave-one-state-out forecast feasibility | yes | `configuration.json`, `stage0/` |
| 1: 40 fresh Spatial sources (states 9–12, pinned seeds), 20-branch replays, calibration (17), locked test (19) | yes | `sources/`, `qualification/`, `physical_test/`, `predictions/`, `analysis/test_*` |
| 2: reacting-policy bridge, 160 episodes (states 13, 14, seed 83001) | yes | `reacting_policy/`, `analysis/stage2_*` |
| figures | yes | `figures/fya_{benefits,delay_cap,forecast,healthy,stage2}.pdf` |

Two sources (qualification task 3 state 10; test task 7 state 12) failed the fresh-prefix fidelity tolerance
(duplicate gap 0, fingerprints identical, warm-start divergence 2e-6 and 1.5e-4 rad) and are excluded per the
rule; two qualification sources were too short. Report as missingness.

## Results to carry into the paper

**1. No prospective certificate (say so).** The benefit-interval procedure (§5 of the prereg) gives **0 decisive
intervals on the locked test** under both width rules (coverage 19/19 affine, 16/19 max). The frozen point
forecast signs the benefit "correctly" on 88 % (NT) / 87 % (innovation) of cells, but that is the base rate
of positive benefit: the forecast barely varies across sources (3.0–4.6e-5 vs measured sd 1.5e-4,
correlation 0.19). The estimator-error control does the same. Write: "neither the forecast nor a scalar
estimator-error summary predicts which source benefits." Keep the failed criteria visible (§9, §10 caveat).
Also note the correction: the forecast code double-counted the healthy residual mean until review; fixed
before the test forecast, qualification forecast regenerated before its run file existed.

**2. Measured delay / cap / sign structure (19 untouched sources, task-clustered 95 % intervals, m² s).**

| Scenario | B_NT | B_innov | B_reference | rel. reduction NT / innov / ref |
|---|---|---|---|---|
| reference (+.05 r_y, cap .05) | 8.8e-5 [4.7e-5, 1.3e-4] | 9.0e-5 [4.8e-5, 1.3e-4] | 1.47e-4 [1.0e-4, 1.9e-4] | .61 / .62 / 1.00 |
| delay 10 steps | 6.1e-5 [2.9e-5, 9.2e-5] | 6.2e-5 [3.1e-5, 9.2e-5] | 1.07e-4 [5.4e-5, 1.5e-4] | .45 / .41 / .78 |
| cap .025 | 7.7e-5 [4.4e-5, 1.1e-4] | 7.5e-5 [3.8e-5, 1.1e-4] | 9.8e-5 [5.3e-5, 1.4e-4] | .55 / .58 / .75 |
| sign −.05 | 1.62e-4 [4.8e-5, 3.3e-4] | 1.63e-4 [4.6e-5, 3.3e-4] | 2.47e-4 [1.2e-4, 4.4e-4] | .59 / .64 / 1.00 |

J_off median 1.19e-4 (endpoint 13.7 mm → NT/innovation 7.8 mm, reference 0.0, delayed reference 2.7,
capped reference 5.7). Paired contrasts: delay raises the adapted cost +2.7e-5 [9.5e-6, 4.4e-5] (NT, 18/19
sources) and the reference's +4.0e-5 [2.2e-5, 6.9e-5] (19/19): this is the integrator-memory prediction
measured prospectively. Halving the cap raises the adapted cost +1.1e-5 [1.8e-6, 1.9e-5] (14/19); the capped
reference keeps a quarter of the energy (quadratic in the remaining half). Sign reversal: off cost 30 %
larger (1.57e-4 vs 1.19e-4), forecast error rises, all three max-rule coverage misses are sign cells: the
response is not sign-symmetric; the linear model's coverage claim (E3) failed. Healthy false-update cost
1.2e-5 (NT) / 1.4e-5 (innovation) per source, ≈ 10 % of the faulted energy. Reference beats both online laws
on every source in the reference and sign scenarios (E4).

**3. Stage 2 is at the ceiling.** Every arm 20/20 except healthy NT 19/20: a +.05 r_y bias from step 30
does not break the reacting policy on Spatial. No task-level evidence on delay (S1 fails at the ceiling).
Telemetry: half the fault identified 9 steps after enablement, settle at 72 %; **healthy phantom estimate
.015 under replanning** (vs 5e-4 in fixed-command replays), the mechanism behind healthy regressions. If a
task-level delay result is wanted, it needs a stronger mid-episode fault under a new registration.

**4. Theory pieces verified in code.** Scalar floor `g [F S_N(λ) − C S_{N−τ}(λ)]_+` equals the exact LP
minimax (3e-11); driver timing (delay, cap, reference) asserted on a fake plant. Say "checked numerically",
not "proved by experiment".

## Wording guards

- The forecast simulates the deployed update function on a *reduced* residual model; it is not a simulation
  of the deployed FIR loop.
- Fixed-command bounds do not transfer to the reacting policy; the bridge shows the healthy phantom, nothing
  about delay.
- Do not present the base-rate sign agreement as predictive skill.
- Units: J_p in m² s (dt .05 s × Σ‖e‖²); endpoints in mm; angular energy J_r reported separately
  (`analysis/test_score/source_intervals.json`, `B_r_task_clustered`).

## Reproduce

```bash
python3 openpi/re4_theory/fya_score.py results/frozen_yet_adaptive_deadline_v1/physical_test/run.json.gz --out /tmp/fya_test
python3 openpi/re4_theory/fya_forecast.py evaluate --predictions results/frozen_yet_adaptive_deadline_v1/predictions/test_predictions.csv \
  --calibration results/frozen_yet_adaptive_deadline_v1/qualification/calibration.json --run results/frozen_yet_adaptive_deadline_v1/physical_test/run.json.gz --out /tmp/fya_eval
python3 openpi/re4_theory/fya_stage2_score.py; python3 openpi/re4_theory/fya_stage2_telemetry.py
```
