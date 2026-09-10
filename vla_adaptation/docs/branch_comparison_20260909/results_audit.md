# Evidence audit for the branch comparison

Audit date: 2026-09-09. Comparison endpoints supplied to this audit: historical
`main` at `7ca2f26`, `review/dual-track-audit` at `cda6c92`; common ancestor
`96555f3`. Paths below are relative to the repository root. This is a read-only
reanalysis of stored outcomes and configurations, not a new robot experiment.

## What the review branch establishes

- It exposes important failures of the inherited controller that an elbow-only
  physical-fault evaluation misses, and supplies substantially broader tests.
- Its weighted allocator improves a prespecified Panda joint-5 comparison against
  the legacy controller, but also has a confirmed Object joint-3 counterexample.
  It is an experimental option, not a universal repair.
- It implements and fairly tunes several calibrated estimators. The completed
  ALOHA confirmation does **not** establish composite superiority over Kalman.
- A later, distinct descriptor controller has favorable fresh paired outcomes
  across all seven Panda joints in Spatial plus one Object condition. Those
  outcomes evaluate a supplied constant basis and the integrated controller;
  they do not isolate composite feedback, learned features, or geometry.
- Existing historical data must remain attributed to their actual runner and
  method. The descriptor study does not retroactively validate the predecessor,
  and the predecessor's thousands of trials do not validate learned bases.

## 1. Original seven-joint fault map: reproduced

Source: `results/joint_map/cell_torque_0.json` through `cell_torque_6.json`.
All rows are LIBERO Spatial, 20 matched `(task, init)` pairs, inits 45/46,
continuous correction with `static_corr: null` and `corr_dims: "0,1,2"`.

| Physical joint | Faulted, correction off | Faulted, correction on | Observed fixed | Observed broken | Exact two-sided McNemar p |
|---|---:|---:|---:|---:|---:|
| 0 | 19/20 | 19/20 | 0 | 0 | 1 |
| 1 | 19/20 | 19/20 | 1 | 1 | 1 |
| 2 | 18/20 | 19/20 | 2 | 1 | 1 |
| 3 | 11/20 | 18/20 | 8 | 1 | 0.0390625 |
| 4 | 17/20 | 14/20 | 1 | 4 | 0.375 |
| 5 | 12/20 | 3/20 | 0 | 9 | 0.00390625 |
| 6 | 7/20 | 11/20 | 8 | 4 | 0.3876953 |
| Descriptive total | 103/140 | 103/140 | 20 | 20 | — |

Seven-comparison Bonferroni leaves joint-5 harm resolved at **0.02734375**;
the elbow's improvement is **0.2734375** after that correction. Joint 6 improves
numerically, so “nothing except the elbow repairs” is incorrect. “A constant
correction became wrong” misstates the continuously updated experiment.
Configuration-dependent torque-to-command behavior is a possible mechanism,
not itself proof of unmatched disturbance or a causal explanation of harm.

The same scenarios recur under different faults, so the 140 counts are a
descriptive sum, not 140 independent scenarios for unrestricted pooled inference.
Sampling of policy actions is not pinned between paired arms. A fixed/broken pair
is an observed outcome difference, not a deterministic counterfactual.

## 2. Independent weighted-allocation confirmation

Sources: `report/FOLLOWUP_STATUS.md`,
`results/joint_followup/confirmation_plan/compact/`,
`results/joint_followup/confirmation_plan/analysis/summary.md`, and
`results/composite_followup/aloha_weighted_confirmation_plan/compact/`.

### Panda: 1,440 assigned rollouts

Three LIBERO suites × (seven joints + healthy) × 20 scenarios × three arms.
The arms are off, legacy translation-only, and weighted full correction.
Fresh healthy calibration uses different inits from the confirmation, and
the candidate/configuration was fixed after a separate development screen.
Complete reset, artifact, and telemetry hashes are retained in the compact data.

- Prespecified joint-5 comparison over three suites:
  **legacy 15/60 → weighted 41/60**, 29 fixed / 3 broken,
  exact **p = 2.556014806e-6**.
- The corresponding off score is **36/60**. The increase to 41/60 is a different,
  descriptive secondary comparison, not the same 26-episode treatment effect.
- Object joint 3: **legacy 16/20 → weighted 2/20**, 1 fixed / 15 broken;
  24-comparison Bonferroni **p = 0.01245117**. Off itself is 8/20.
- Goal healthy: **off 19/20, legacy 3/20, weighted 19/20**. The corresponding
  resolved contrasts have adjusted **p = 0.00073242**.
- Direct recount over all 24 cells: off **344/480**, legacy **231/480**, weighted
  **397/480**; these are descriptive totals, not a pooled independent test.

The candidate changes selection, weighting, regularization, and allocation
together. This confirmation does not identify which component caused a benefit.
The same absolute +5 N·m is 6.25% of installed limits on Panda joints 0–4 and
41.7% on joints 5–6; it is not uniform fractional fault severity.

### ALOHA: 780 assigned rollouts

Transfer Cube × (12 arm joints + healthy) × 20 scenarios × three arms.
Faulted descriptive totals reproduce as **off 60/240, legacy 70/240,
weighted 66/240**; healthy is **11/20, 8/20, 6/20** respectively.
No per-cell contrast resolves after its declared 13-comparison Bonferroni family.
Weighted versus off loses five healthy successes and rescues none; raw
**p = 0.0625**, adjusted **p = 0.8125**. Failure to resolve is not equivalence
or a no-harm certificate. Transfer of the Panda candidate is unestablished.

Physical ALOHA arm numbering excludes the grippers: left 0–5, right 6–11.
Command/state vectors also contain fingers, so physical and command indices
must not be interchanged. The torque schedule is per-joint `kp * 0.02 rad`.

### Local physical diagnostics are not task outcomes

The retained authority-grid summaries distinguish local response probes from
success trials. Among 168 nonzero-correction probes, legacy worsens 150 and
weighted worsens 92. One nominal objective improves to 10.5% of the zero-action
value while actual local physical error increases 2.69-fold. The ALOHA privileged
torque-cancellation probes do not demonstrate a realizable VLA task controller.
These observations motivate uncertainty and authority checks; they do not close
the mechanism of every failed episode.

## 3. Estimator comparisons: two separate studies

### Small matched-observer comparison

Recounted `results/observers/eval/*.json` adaptive successes:

| Proposed | DOB | RLS | Kalman | Calibrated integral | Oracle |
|---:|---:|---:|---:|---:|---:|
| 19/20 | 17/20 | 17/20 | 18/20 | 15/20 | 20/20 |

The frozen controls in those files vary between 0/20 and 1/20. Do not turn
this small separately executed comparison into an established total ranking or
equivalence result. Shared clipping remains present across matched estimators;
DOB removes deadzone/normalization, not projection. Several estimators repair
the calibrated interface, but the respective contributions of calibration,
FIR modeling, channel choice, and the update law are not separately identified.

### Completed ALOHA tuning and untouched-seed confirmation

Sources: `results/composite_tuning/STATUS.md`, `PREREGISTRATION.md`,
`confirmation/analysis/analysis.json`,
`audit/INDEPENDENT_STATISTICS_CROSSCHECK.md`, and
`confirmation/diagnosis_v1/DIAGNOSIS.md` under that directory.

Direct recount of every compact `conditions.*.arms.*.per_ep` confirms:

- Search: **2,608** outcomes in 16 studies, seeds 3200–3203.
- Validation: **896** outcomes in 32 studies, seeds 3300–3307.
- Confirmation: **1,920** outcomes in 48 studies, seeds 3400–3429.
- Total: **5,424**, not 5,424 independent confirmation trials. Healthy fitting
  uses seeds 3100–3129; 24 additional smoke rollouts are separate.

| Selected family | Held-out balanced score |
|---|---:|
| Original proposed/legacy | 50.00% |
| RLS | 43.33% |
| DOB | 42.14% |
| Kalman | 36.90% |
| Composite | 35.95% |
| Calibrated integral | 31.90% |

These are numerical scores, not an established ordering. Composite minus Kalman
is **−0.95 percentage points**, whole-seed bootstrap 95% CI **[−9.52, +7.62]**,
exact whole-seed sign-flip **p = 0.86261**. The independent verification accepts
all 1,920 outcomes with zero arithmetic/provenance mismatches; four secondary
comparisons use Holm correction and per-condition comparisons use their declared
eight-condition families. The suite preserves repeated-condition dependence by
resampling/flipping whole seeds.

The selected composite has lower final-window fitted reference error in all
eight confirmation conditions. Yet **99.3916%–99.9678%** of active tracking-update
squared energy concentrates on left joint 1, and late parameter errors are mixed.
For the all-left command-offset condition, reference RMS falls **9.50 → 7.47 mrad**
while parameter RMS rises **5.11 → 5.76 mrad**. This is not a sign-bug diagnosis.
Selected Kalman and composite also differ in gain, damping, and clipping; matched
parameter ablations are needed to isolate tracking feedback.

The finite search is substantial but not globally optimal tuning. In particular,
the `Q=0` static-parameter Kalman case is excluded by the preparation/runner
eligibility screen, despite being a valid convergent limit in appropriate models.
See `results/composite_tuning/audit/TUNING_COVERAGE.md`.

## 4. New descriptor results: favorable, with narrower attribution

Sources: `learned_adaptation/libero_adapter.py`, `results/descriptor/*.json`,
`results/sweep/spatial_j*.json`, and
`prereg_records/PREREG_DESCRIPTOR_FIRST_RUN.md`.

The actual runner fits a healthy ARX2 predictor with 12 states, uses its input
matrix consistently, and supplies a **constant `Phi = E`** basis with six
action-equivalent additive coefficients. There is no learned feature network,
physical wrench estimator, online effectiveness estimate, or geometric controller
in these rollouts. The gripper command is passed through; these are not gripper
adaptation experiments. The finite-motion state is not seven joint positions.

The embedded model metadata records healthy training inits **35/36**, evaluation
uses **37/38**, and policy RNG is explicitly **unpinned**. The Spatial model is
shared across its seven cells; the Object cell uses a separate Object model.
The healthy records originated in the predecessor's confirmation and were reused
for this later method's fitting. This is a legitimate disclosed reuse with
different descriptor evaluation inits, not a newly collected training corpus.

The runner uses an Euler reference embedding `dt=1 policy interval`, `B=I`,
`A=F-I`, `E=G` through **`AdaptationLoop`**, with a left-state drift and completed
state difference in its residual. This exactly represents the fitted ARX
recurrence. The coefficient update **does use Van Loan forgetting propagation
and a Joseph-form covariance update**, followed by endpoint tracking injection.
It does not use the separate `DescriptorLoop`'s continuous ZOH state reference;
this split implementation is not a proved exact discretization of the entire
continuous adaptive system. The fitted matrices identify a
conditional closed-loop predictive map, not separate physical `A,B` or a certified
intervention map. Metadata explicitly retains this caveat and fold input-map
spread; Schur stability of the fitted recurrence does not prove physical/global
closed-loop contraction.

### Directly reproduced outcomes

All rows compare the descriptor against its own frozen-faulted arm, 20 pairs,
`max_steps=280`, `replan_steps=5`, constant +5 N·m joint torque. Except the first
development row, the coefficient/correction bound is 0.25.

| Condition | Frozen-faulted | Descriptor | Fixed | Broken | Exact raw p |
|---|---:|---:|---:|---:|---:|
| Object j3, initial limit 0.15 | 7/20 | 12/20 | 8 | 3 | 0.2265625 |
| Object j3, revised limit 0.25 | 7/20 | 18/20 | 11 | 0 | 0.0009765625 |
| Spatial j5, earlier than sweep | 14/20 | 19/20 | 6 | 1 | 0.125 |
| Spatial j0 | 20/20 | 19/20 | 0 | 1 | 1 |
| Spatial j1 | 19/20 | 20/20 | 1 | 0 | 1 |
| Spatial j2 | 17/20 | 19/20 | 3 | 1 | 0.625 |
| Spatial j3 | 15/20 | 19/20 | 5 | 1 | 0.21875 |
| Spatial j4 | 16/20 | 19/20 | 4 | 1 | 0.375 |
| Spatial j6 | 11/20 | 20/20 | 9 | 0 | 0.00390625 |

- The six newly swept cells total **98/120 → 116/120**, 22 fixed / 4 broken.
- All seven Spatial cells total **112/140 → 135/140**, 28 fixed / 5 broken.
- Seven Spatial + revised Object j3 total **119/160 → 153/160**, 39 fixed /
  5 broken. This is the eight-cell total in `SWEEP_RESULT.md`.
- Counting the initial Object development run as well gives **360 rollouts**
  in nine cells; it must not silently enter the selected eight-cell denominator.

### Corrections needed before importing these claims into main

1. **Coverage:** the eight reported cells span **seven physical joints**, with
   **seven** cells sharing Spatial. Existing prose saying six joints or six of
   eight cells in Spatial is wrong.
2. **Pooled inference:** the raw **1.40516e−7** McNemar calculation is reproduced
   from 39/5 discordances, but its 160 rows are only **40 distinct suite/task/init
   scenarios**, with Spatial scenarios repeated across seven faults. Sharing the
   same pairs does not justify treating those repeated rows as independent. Use
   a prespecified cluster-aware estimand and inference. Do not blindly transfer
   the pooled p-value as validated confirmatory evidence.
   A post-hoc sensitivity check grouping all joint contrasts by
   `(suite, task, init)` gives **p = 4.768371582e−6** from an exact two-sided
   sign-flip distribution of the 40 cluster sums (26 nonzero). This assumes
   independent scenario clusters and sign exchangeability under the null;
   it remains post-hoc, does not correct adaptive development/selection, and
   is not a replacement preregistered claim. The favorable direction therefore
   is not solely an artifact of the naive independent-row arithmetic.
3. **Multiplicity and selection:** raw per-cell p-values are not adjusted. For
   reference, a declared seven-Spatial-cell Bonferroni family would give j6
   **0.02734375**; an eight-cell family would give j6 **0.03125**, Object j3
   **0.0078125**. These retrospective calculations do not undo selection of the
   0.25 bound after the initial Object result or repeat use of the same scenarios.
4. **Baseline identity:** the old joint-5 map uses inits 45/46 and the inherited
   Spatial horizon, whereas this study uses 37/38 and 280 steps. The preregistration
   additionally warns that deployment `joint_fault.py` differs from the branch
   snapshot. “12→3 reversed to14→19 under the same conditions” is therefore not
   a controlled head-to-head comparison. Each run's internal frozen arm is valid
   comparison evidence; the old run supplies context.
5. **What failed:** without a same-scenario healthy arm, the 41 baseline failures
   are **frozen-faulted failures**, not proven episodes that the fault itself
   broke. “Returns to the unfaulted rate” is not measured here.
6. **Saturation:** the preregistration reports 71.5% saturation at limit 0.15 and
   **7.4%** at 0.25. The latter misses its registered **below-5%** manipulation
   check. Subsequent below-10% checks do not establish that the bound had no
   causal influence. “Unclipped” is a filename/label, not literally no clipping.
   Exact step fractions were not independently reconstructed in this audit.
7. **Noise:** an observed three-episode change across earlier runs is not an
   estimated universal ±3 noise floor, confidence interval, or permission to
   disregard a regression. Independent policy sampling remains a limitation.
8. **Attribution:** no descriptor-vs-Kalman, prediction-only, consistent-map-only,
   state-term-only, learned-vs-constant basis, or geometry ablation exists in
   these result files. The positive integrated-controller outcomes cannot identify
   the contribution of the composite term or close the old failure mechanism.
9. **Geometry:** exact transformation tests on synthetic/robot kinematics are
   implementation evidence. The nominal fitted model is itself not certified
   equivariant, no nontrivial symmetry of the frozen policy/deployed experiment
   is established, and the wrench-to-motion response bridge remains unidentified.
10. **Historical provenance:** `docs/independent_adaptation/README.md` still says
    the independent implementation is not connected to a runner. It must be
    updated to distinguish the tested supplied-basis descriptor adapter from
    the still-untrained learned/geometric successor. `paper/independent_method.tex`
    is not part of the nine-page predecessor draft and has additional mathematical
    and implementation claims requiring reconciliation before use.

These limitations do not negate the observed improvements. They determine what
the branch has demonstrated and what needs a frozen, independently evaluated
confirmation protocol before a stronger method claim.

## Reproduction and next checks

The standalone [reproduce_descriptor.py](reproduce_descriptor.py) reads all eight
explicitly named descriptor inputs with `git show` at the exact pinned review
commit `cda6c92357e884ef6e2b1195abda673fc18853ac`, so subsequent working-tree
changes cannot silently alter this comparison. It checks completeness, 20 unique
and equal paired keys per cell, boolean outcomes, and stored success totals.
The executed [descriptor_recount.json](descriptor_recount.json) receipt includes
each input's Git blob ID and SHA-256, the script SHA-256, all counts and exact
probabilities, and all 40 scenario weights and inferential assumptions.

From the repository root, reproduce it with:

```sh
python docs/branch_comparison_20260909/reproduce_descriptor.py \
  --output /tmp/descriptor_recount.json
```

The counts above were reconstructed from `per_ep`, asserting unique paired keys,
equal key sets, and agreement with the stored `successes` and `n` fields. Exact
McNemar arithmetic was computed independently with Python's `math.comb`:

```python
n = fixed + broken
p = min(1, 2 * sum(math.comb(n, k)
                   for k in range(min(fixed, broken) + 1)) / 2**n)
```

The exploratory cluster check forms each weight by summing `int(on)-int(off)`
across the tested fault cells sharing a scenario, then counts all sign choices
by an exact integer dynamic program. With `weights` the 40 signed sums:

```python
from collections import Counter
distribution = Counter({0: 1})
nonzero = [abs(w) for w in weights if w]
for weight in nonzero:
    updated = Counter()
    for value, count in distribution.items():
        updated[value - weight] += count
        updated[value + weight] += count
    distribution = updated
observed = abs(sum(weights))
p = sum(count for value, count in distribution.items()
        if abs(value) >= observed) / 2**len(nonzero)
```

For the descriptor count, use six `results/sweep/spatial_j*.json` files plus
`results/descriptor/g14_spatial_j5.json` and
`results/descriptor/g11_object_j3_unclipped.json`; do not include the earlier
`g11_object_j3.json` in the eight-cell selected total. Key episodes by
`(suite, task, init)` when checking cross-cell dependence, and include the fault
identity for the distinct rollout-assignment key.

Established reproducible project commands, writing regenerated reports to `/tmp`:

```sh
python3 openpi/score_joint_followup.py \
  results/joint_followup/confirmation_plan/score_manifest.json --json
python3 openpi/summarize_followup_confirmation.py \
  results/joint_followup/confirmation_plan/score_manifest.json \
  --score-report results/joint_followup/confirmation_plan/scores.json \
  --out-dir /tmp/panda_branch_comparison_report
python3 results/composite_tuning/operations/crosscheck_confirmation_statistics.py \
  --report results/composite_tuning/confirmation/analysis/analysis.json \
  --manifest results/composite_tuning/confirmation/analysis_manifest.json \
  --collection-dir results/composite_tuning/confirmation/collected \
  --out-dir /tmp/composite_branch_comparison_check
```

Those larger archival commands are provided for the importing agent; this bounded
audit directly recounted the compact data rather than re-running the existing
full hash/telemetry/50,000-resample pipeline. New experiments should first freeze
the actual deployed source hashes, policy settings, horizons, calibration,
selection process, joint/fault panel, and inference families. Then evaluate
healthy harm and matched legacy/prediction-only/composite comparisons on fresh
scenarios before expanding trained geometry claims.
