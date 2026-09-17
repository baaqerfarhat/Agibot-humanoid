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

## Addendum (16 Sep, evening): crossed command-stream replay for `papers/frozen_yet_adaptive_closed_loop/`

Registration with outcomes: `prereg_records/PREREG_FYA_CROSSED_REPLAY_V1.md` (§§8, 10). Data
`results/fya_crossed_replay_v1/` (`analysis/`, `analysis_{delay,innovation,healthy}/`, `figures/crossed_*.pdf`).
Record §§66–67.

**Fidelity.** Exact reproduction of the archived live paths (gaps 0.0 on joints, positions, corrections) on 15
of 18 eligible keys; three state-14 keys (tasks 2, 5, 7) differ from the live run by 5e-8 to 9e-5 rad although
fresh and restored replay routes agree exactly (the live env carried the previous episode's state). Excluded by
the registered tolerance; say so. A first collection was invalidated by an angle-check formula; disclosed.

**Primary matrix (immediate NT, 15 keys / 9 tasks, equal-task, m² s).** J00 8.3e-4, J10 7.9e-4, J01 6.7e-4,
J11 4.0e-4 (medians; endpoints 31 / 26 / 26 / 26 mm from the reacting healthy reference).
D0 = 8.9e-5 [2.5e-5, 1.5e-4] (12/15 keys), T = 5.0e-4 [9.4e-6, 1.0e-3] (14/15), R1 = 4.1e-4 [−4.7e-5, 9.3e-4],
**I = 2.3e-5 [−2.7e-5, 7.6e-5], unresolved** at δ_I = 1e-5. The correction term equals the fixed-command
benefit (8.8e-5) almost exactly; most of T is the stream term, heavy-tailed. No mechanistic headline on I.

**Optional matrices.** Delayed NT: D0 5.2e-5, T 1.5e-4 (a third of immediate). Innovation: D0 8.9e-5,
R1 5.0e-4 [1.1e-4, 1.0e-3]. **Healthy control (no fault):** J10 = 8.5e-6 (2.2 mm; the direct cost of false
updates, 16/16 keys) but J01 = 8.0e-5 (12.7 mm): R1 = −3.6e-4 [−5.6e-4, −1.7e-4] on every key. The policy's
reaction to small false corrections costs ten times the corrections themselves. Wording: mechanism on archived
keys; not a causal account of the one healthy task loss; not a task-success statement.

**Stronger-fault campaign (running, `PREREG_FYA_STRONGER_FAULT_V1.md`).** Development selection among F1
(uniform six-channel +.05 from step 30), F2 (r_y +.10), F3 (r_y +.15) on states 15, 17; evaluation of four arms
on 40 untouched keys (states 18, 19, 21, 22). Outcome appended to that registration when done.

**Stronger-fault campaign, done (`PREREG_FYA_STRONGER_FAULT_V1.md` §6, `results/fya_stronger_fault_v1/`).**
Development: the policy tolerates mid-episode r_y biases up to +.15 and uniform +.05; uniform **+.10 on all six
channels from step 30** breaks it (5/20) and was selected by the registered rule. Evaluation on 40 untouched keys:
frozen 6/40 → NT 14/40 (9 wins / 1 loss, +20 points [10, 30], exact McNemar p = .021); healthy 40/40 → 40/40, no
discordant key. Both registered expectations hold. Wording: partial repair by construction (rotation-only mask,
cap .05 against a .10 fault); task-level effect of the fixed adapter under a registered command fault on untouched
keys; not a certificate. This is the result the previous bridge could not give.

**Delayed-NT increment (stronger fault, prereg §8):** 14/40, identical to immediate NT (3 wins / 3 losses, [−10, +10] points). Wording: at task level the ten-step delay is neither better nor measurably worse on this fault; the delay cost is a physical-deviation result, not a task-success result.

**Delayed-NT repair (`PREREG_FYA_DELAY_REPAIR_V1.md`):** the split-manifest increment was invalid (sampler ordinals changed on the second shard). Repaired on one server with the full manifest, coupled 40/40: delayed NT 13/40 vs immediate 14/40 (2W/3L, [−10, +5] points), vs off +7 net [7.5, 30]. Use these numbers, not the 14/40 of the first increment, for any controlled delay statement; the statement remains 'not better, not measurably worse'.

**Delayed-NT repair (replaces the sharded increment for any delay statement; `PREREG_FYA_DELAY_REPAIR_V1.md`):** coupled 40/40 with immediate NT; delayed 13/40 vs immediate 14/40 (2 wins / 3 losses, [−10, +5] points, p = 1.0) and vs off 6/40 (+7 net, [7.5, 30], p = .016). Wording: no task-level delay penalty resolved at this n; equal totals are not equivalence.

**Independent healthy replication (`PREREG_FYA_HEALTHY_COUPLED_V1.md` §7):** on 34 fresh coupled keys (9 tasks, seed 86001) R1 = −1.71e-4 [−2.5e-4, −9.8e-5] m² s, negative on every key; D0 = −1.9e-5 (0/34 positive); I unresolved. Replicates the archived healthy control (−3.6e-4, 16 keys) at about half the magnitude. Coupling caveats to state: arms must share one server process; duplicate exact on 36/40; one key excluded by fidelity, four task-0 keys too short. Figure `results/fya_healthy_coupled_v1/figures/healthy_R1_two_cohorts.pdf`.

**Matched DOB comparator (`PREREG_FYA_DOB_COMPARATOR_V1.md`):** in-process trios on the 40 evaluation keys: faulted off 6/40, NT 14/40, DOB (α .08, development-selected and gain-matched) 14/40, 1 win / 1 loss, [−7.5, +7.5] points; healthy off 39/40, NT 40/40, DOB 40/40. Wording: the calibrated execution interface carries the task benefit; NT's gate and normaliser add nothing resolvable; nonsignificance is not equivalence. The fresh trio reproduces the archived 6 → 14.

**Closing status (23:45).** Everything in the closed-loop EXPERIMENT_PLAN.md is now complete: the independent healthy replication (R1 replicated on 34 fresh keys), the delayed-NT repair (13/40 vs 14/40, coupled), and the DOB comparator (tie). Registrations: `PREREG_FYA_HEALTHY_COUPLED_V1.md` §7, `PREREG_FYA_DELAY_REPAIR_V1.md`, `PREREG_FYA_DOB_COMPARATOR_V1.md`. Record §§66–70. Both cards released.

**Recovery study P1 (`PREREG_FYA_CROSSED_STRONG_V1.md`):** four-cell matrix under the +.10 six-channel fault on 30 exact archived keys: R1 1.2e-3 [−2.7e-3, 4.5e-3], D0 5e-5 [−1.7e-4, 3.0e-4], T and I unresolved; costs ~10× the benign matrices, paths ~10 cm off within the window. Wording: the fifty-step decomposition is uninformative in the task-breaking regime and does not explain the 6/40 → 14/40 repair; do not extrapolate the benign-fault mechanism. Keep the full-cohort task table adjacent, with the 30-key denominator explained.

**Recovery study P3 (`PREREG_FYA_CROSSED_DOB_V1.md`):** DOB replay verified by an exact pilot; on 27 common keys the paired NT − DOB total-benefit difference is −1.3e-3 [−3.5e-3, +6e-5] (unresolved), DOB's stream term marginally larger, NT's direct term marginally larger. Wording: no resolved mechanism difference between the laws; report both matrices; task totals remain 14/40 each.

**Recovery study P2 (`PREREG_FYA_HEALTHY_LIBERO10_V1.md`):** fresh LIBERO-10 keys, 36 exact; R1 = −8.5e-5 [−1.6e-4, −2.4e-5] (replicated on a second suite), S = +7.2e-5 [1.1e-5, 1.5e-4] prospective; D0 < 0 on all; I unresolved. Healthy task outcomes off 38/40, NT 37/40, dup 37/40 (list the NT regressions 8/30, 6/33 and the gain 4/31). Coupling 36/40 prefix, duplicate 31/40. Figure `results/fya_recovery_study_v1/healthy_libero10/figures/healthy_R1_three_cohorts.pdf`.

**Recovery-study plan closed (17 Sep 05:35):** P1 negative (decomposition uninformative under the strong fault), P2 replicated on LIBERO-10, P3 unresolved NT–DOB physical difference. Records §§71–73.

**17 Sep afternoon, recovery-study plan §§1–3.** Offline: reference-sensitivity envelopes and all-key tables in `results/fya_recovery_study_v1/analysis_offline/` (record §74; identities checked; healthy R1 sign survives 3.8–7.0 mm/step reference displacement depending on cohort; duplicate-off alternative references are identical on all but 2–3 keys, so that comparison is uninformative; discordant-key lists and the lexicographic case-selection rule are in `inclusion_audit.json` per campaign). GR00T: server seeded per call with provenance acks (`PREREG_FYA_GROOT_HEALTHY_CROSSED_V1.md`), pilot passed exactly (record §75), confirmatory 40-key healthy matrix running (states 23, 24, 29, 31 reused from the pi0.5 cohort, disclosed; seed 89001; scorer seed 20260920). Outcome to be appended as §E.

**Offline verification correction, September 17.** The paper's
`evidence/offline_analyses_verified.json` supersedes the preceding duplicate-reference interpretation:
source `position` is not the primary replay `ee_pos` metric, so no empirical duplicate reference qualifies.
Corrected prefix qualification is 34/34 Spatial and 34/36 LIBERO-10. Analytical bounds remain unchanged,
but point-mean sign radii are not confidence guarantees; LIBERO-10 S loses a resolved lower-envelope
sign already at 1 mm. Corrected outputs are in `analysis_offline_corrected_v1/`; historical files remain.
All fifteen task-discordant cases are retained, and P3 physical fields explicitly describe the NT matrix.
The new trajectory analysis separates cost I from reference-independent Z; the paired NT-minus-DOB
RMS Z difference is .049 [−.389,.558] mm, unresolved.

**GR00T healthy crossed replication (`PREREG_FYA_GROOT_HEALTHY_CROSSED_V1.md` §E):** 36 exact keys (states 23, 24, 29, 31 reused, seed 89001), R1 = −1.81e-4 [−4.2e-4, −3.6e-5] negative on every key, S = +1.64e-4 [2.8e-5, 4.0e-4], D0 < 0 on all, I unresolved; healthy off 38/40, NT 39/40 (rescue 2/29, no regression); coupling 38/40 prefix, duplicate 35/40; cross-device control 35/40 exact. Figure `results/fya_groot_healthy_crossed_v1/figures/healthy_R1_four_cohorts.pdf`. Wording: transfer of the healthy stream effect to a second backbone on inspected physical states; not a task-harm claim.
