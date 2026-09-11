# Pre-registration — the ALOHA residual norm, its channels, and the deadzone

**Written 2026-09-07, BEFORE the runs.** The environment (openpi + gym_aloha + `pi0_aloha_sim`
on lambda.arcl) was still installing when this was committed; no arm below has been executed.
Predictions are stated with numbers so they can be wrong.

Motivated by `report/DUAL_TRACK_AUDIT.md` Addendum 2 and the instrument
`openpi/residual_channels.py`, which is committed and whose outputs are reproduced below.

---

## 1. Established before this experiment (measurement, not hypothesis)

From `openpi/residual_channels.py` on the stored logs, leave-one-episode-out, 8 healthy
rollouts, 2205 scored steps:

| quantity | value |
|---|---|
| right gripper (ch 13) share of squared FIR residual | **93.805%** |
| both grippers | 96.028% |
| the six corrected left-arm joints | **0.089%** |
| gripper / corrected energy ratio | **1076×** |

And, at the settings the stored runs actually use (`norm_r=0.4` in 19/22, `dead=0.002` in 22/22):

| quantity | value |
|---|---|
| deadzone firings, `nr` over all 14 channels | **0 / 2205 steps** |
| deadzone firings, `nr` over corrected channels only | **1436 / 2205 = 65.1%** |
| attenuation penalty from the gripper at `norm_r=0.4` | 1.10× (mild) |
| attenuation penalty at `norm_r=0.05` (3 early runs) | 7.26× |

Structural facts from the source, not inference:

- `aloha_adapt.py` computes `nr = norm(res)` over **all 14** channels while correcting **6**.
- `adaptive_law.py`'s default `legacy` path is the **same law**, but takes the norm over exactly
  the 6 channels it corrects. This is the only structural difference between the two robots.
- When the gate fires, the code sets `est = 0` and then applies `f_hat += gamma*(est − f_hat)`.
  **That decays `f_hat`; it does not hold it.**
- `SETUP.md:173` records that the 14-joint norm was measured at ~0.19 rad and `norm_r` was set
  to 0.4 to match — the paper's own "constants are ratios of measured scales" rule, applied to a
  scale measured over channels that are not corrected.

## 2. Hypothesis

**H1.** The ALOHA correction's within-episode wander (§27.6: 0.019 rad median swing,
"phase-locked, not noise", 2.0 cm closed-loop against 0.4–0.7 cm in open-loop replay) is caused
by the estimator integrating noise on the corrected channels with no functioning deadzone,
because the gate is held open by uncorrected gripper channels.

**H0 (what would refute it).** Arm C below leaves the wander statistically unchanged from Arm A.

## 3. Arms

`gym_aloha` transfer-cube, `pi0_aloha_sim`, +0.02 rad on left-arm joints 0–5, continuous
adaptation (NOT identify-then-hold), `--corr-joints 0,1,2,3,4,5`, `--norm-r 0.4`,
`--dead 0.002`, `--gamma 0.08`, `--clip 0.08`. **n = 20 paired episodes per arm**, seeds fixed
in advance, identical `(task, init)` across arms.

| arm | `--norm-channels` | `--deadzone-mode` | role |
|---|---|---|---|
| **A** | `all` (default) | `zero` (default) | reproduce the published 0/20 |
| **B** | `corrected` | `zero` | the *naive* fix |
| **C** | `corrected` | `hold` | the fix the analysis actually implies |
| **D** | `all` | `zero`, healthy (no fault) | harm control, as published |

Arms B and C are the treatment; A is the control; D checks the change does no harm on a healthy
arm. A fifth reference point already exists and is not rerun: the published static-0.019
correction restoring 5/20.

## 4. Predictions, registered in advance

1. **Arm A reproduces 0/20** (or within sampling noise of it, ≤ 2/20). If A does not reproduce,
   the experiment is void and the discrepancy is the finding.
2. **Arm B is no better than A, and its estimate visibly bleeds.** Specifically: mean `|f̂|` on
   the corrected joints over the final 50 steps will be **below 50% of the 0.02 rad fault**,
   because the gate will fire on ~65% of steps and each firing decays `f̂` by `gamma = 0.08`.
   A 300-step episode firing at 65% retains `(1−0.08)^195 ≈ 0` of the estimate. **This is the
   prediction most likely to embarrass a naive reading of the mechanism, and it is why the
   two-arm version of this experiment would have produced a misleading null.**
3. **Arm C reduces the within-episode wander by at least 2× relative to Arm A**, measured as the
   median peak-to-peak swing of the applied correction on joints 0–5. This is the primary
   endpoint.
4. **Task success in Arm C is secondary and not predicted with confidence.** Recovering the
   healthy rate (≈5/20) would be a strong result; no recovery with reduced wander would still
   confirm the mechanism and leave the margin story intact with the confound removed.
5. **Arm D remains a null**, within the ±10-point free variation the paper documents at n=20.

## 5. Analysis, fixed in advance

- Primary endpoint: within-episode peak-to-peak of the applied correction on joints 0–5,
  compared A vs C by a paired test over matched `(task, init)`.
- Secondary: task success, exact McNemar on matched pairs, via `openpi/mcnemar_crosscheck.py`
  as well as `openpi/mcnemar.py`.
- Estimate accuracy: mean `|f̂|` over the last 50 steps against the true 0.02.
- Telemetry (`--telemetry`) is recorded for every arm so residuals, `nr`, the attenuation
  actually applied and the gate state are available without a rerun.
- **No arm is dropped after seeing its result**, and no threshold is re-tuned. If a constant has
  to change, that is a new preregistration.

## 6. Stated limits

- One robot, one task. This cannot establish generality; it can establish a mechanism.
- The paired design matches on `(task, init)` only. `pin_rng=False`, so policy sampling noise is
  **not** matched across arms — see `report/DUAL_TRACK_AUDIT.md` §4.10. At n=20 with the ±10
  point free variation the paper documents, only large effects are detectable.
- A confirmed mechanism does **not** by itself overturn the margin story of §27; it removes a
  confound from it. Both outcomes are reportable and neither is the "good" one.

---

# RESULT — Arm A, and the refutation of prediction 3

**Added 2026-09-07, after Arm A ran.** Recorded here rather than in a separate document
because the registered prediction was wrong and the record should show that where the
prediction was made.

## Arm A reproduces

`0/20`, exactly as registered (prediction 1 satisfied; artifact
`results/norm_channels/armA_baseline.json`). The experiment is therefore interpretable.

Two observations from the run, neither predicted:

- **The estimate is essentially perfect and perfectly stable.** `f̂` on the corrected joints
  lands at 0.019–0.022 against a true 0.020 on *all twenty* episodes. Repair failure here is
  definitively not an identification failure.
- **The runner's own saturation guard fires on the gripper, every episode:**
  `!! adaptive: at the clip (±0.08) in >20% of episodes on [j13 100%] -- SATURATED, not
  estimated.` Joint 13 is railed against the projection bound in 100% of episodes.

## Prediction 3 is REFUTED

Telemetry, 6000 adaptive-arm steps under the +0.02 rad fault:

| quantity | healthy logs (the basis of the prediction) | faulted closed loop (actual) |
|---|---|---|
| `nr` over all 14 | 0.121 | 0.085 |
| `nr` over the 6 corrected | 0.0014 | **0.049** |
| ratio | 46× | **1.7×** |
| deadzone fires as implemented | 0% | 0% |
| **deadzone would fire if norm restricted** | **65.1%** | **0.00%** |
| attenuation actually applied | — | 0.957 |

Restricting the norm to the corrected channels does **not** activate the gate under fault:
the corrected-channel residual sits at 0.049, twenty-five times the 0.002 threshold.
**Arms B and C are therefore expected to be indistinguishable from Arm A**, and the mechanism
registered in §2 does not operate in the faulted regime.

## Why the prediction failed, and it is structural

`r = y − ŷ` is computed against the command the law *believes* it sent. Because `u` already
contains the correction while the world executes `u + f`, the residual is `≈ Mf`
**independently of how well the correction is working** — which is the paper's own stated
design (`§sec:method`: "the residual estimates the *total* fault"). The residual does not
shrink as `f̂` converges. Under a persistent fault the deadzone therefore cannot fire in
either norm configuration, by construction.

The error was generalising from healthy calibration logs, where there is no fault and the
corrected-channel residual really is 0.0014, to the faulted condition, where it is 35× larger.
The faulted log had already shown the gripper share falling from 93.6% to 43.8%; that number
was recorded and its implication under-weighted.

## What survives

- The 93.8% gripper share and the 1076× energy ratio are correct **as properties of healthy
  data**, and `openpi/residual_channels.py` reports them honestly with the fit/eval split.
- Under fault the norm inflation is 1.7×, worth ~3% of attenuation. Not a mechanism.
- The one regime where the contamination could still matter is the **healthy** arm, where
  `nr_cor ≈ 0.0014` and the gate would fire on ~65% of steps. That bears on whether the law
  does harm with no fault present — the control that lost 8 of 40 episodes — and **not** on
  repair. That is a narrower, still-testable claim and it is now the only live one.
- `--norm-channels` and `--deadzone-mode` remain useful instrumentation and are bit-identical
  to the previous behaviour under their defaults.

## Consequence for the paper

This does **not** rescue the ALOHA section and must not be reported as if it did. The margin
story of §27 stands, with one confound examined and eliminated rather than confirmed. Arms B,
C and D will still be run, because a measured null on B/C is the honest record of this
prediction, and D tests the one claim that survives.
