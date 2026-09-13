# Preregistration: task-clustered bootstrap for the headline (re4 theory plan, Part 8.1; 2026-09-13)

**Written before computing.** The headline's exact McNemar treats the 120 paired episodes as
independent; they are 10 tasks × 2–4 initial states per suite, so episodes sharing a task are
correlated. This recomputes the headline effect with the task as the resampling unit.

**Data.** The four headline files (`results/suites/libero_{spatial,object}_rotonly_paired.json`,
`libero_{goal,10}_rotonly_n40.json`), 120 (task, init) pairs, and the held-out counterparts
(`results/heldout/*_heldout.json`). Statistic: the paired difference in successes
(corrected − frozen) per suite and pooled; and the regression count.

**Method.** Cluster = (suite, task): 40 clusters of 2–4 pairs. Percentile bootstrap over
clusters, B = 20,000, seed 0; two-sided cluster-permutation (sign-flip of each cluster's paired
differences) p-value for "corrected − frozen > 0", pooled and per suite. Also reported: the
episode-iid bootstrap interval for comparison.

**Predictions.** (1) The pooled 95 % cluster-bootstrap interval for corrected − frozen excludes
zero (shipped and held-out). (2) Every suite's cluster sign-flip p < 0.05 under the shipped
calibration; under the held-out calibration `libero_10` may not (its effect is 7/40). (3) The
cluster interval is wider than the iid interval by at least 20 % (the correlation is real).

**Refutation.** A pooled cluster interval including zero withdraws "every suite individually
significant" in favour of the cluster-corrected statement.

---

## Outcome (appended 2026-09-13; nothing above was edited)

B = 20,000, seed 0; cluster = (suite, task), 10 clusters per suite, 40 pooled.

| calibration | suite | effect | cluster 95 % CI | iid 95 % CI | width ratio | cluster sign-flip p |
|---|---|---|---|---|---|---|
| shipped | spatial | +10 | [+5, +16] | [+6, +14] | 1.38 | 0.031 |
| shipped | object | +11 | [+6, +16] | [+7, +15] | 1.25 | 0.016 |
| shipped | goal | +14 | [+5, +24] | [+7, +20] | 1.46 | **0.061** |
| shipped | libero_10 | +15 | [+8, +23] | [+9, +21] | 1.25 | 0.007 |
| shipped | **pooled** | **+50** | **[+36, +65]** | [+39, +61] | 1.32 | < 0.0001 |
| held-out | spatial | +9 | [+6, +12] | [+5, +13] | 0.75 | 0.009 |
| held-out | object | +8 | [+2, +14] | [+2, +14] | 1.00 | 0.081 |
| held-out | goal | +10 | [0, +21] | [+4, +16] | 1.75 | 0.19 |
| held-out | libero_10 | +7 | [+1, +14] | [+3, +12] | 1.44 | 0.13 |
| held-out | **pooled** | **+34** | **[+20, +49]** | [+23, +45] | 1.32 | < 0.0001 |

- **Prediction 1 (pooled cluster interval excludes zero, both calibrations): confirmed.**
- **Prediction 2 (every suite p < 0.05 under the shipped calibration): refuted on `goal`**,
  p = 0.061 at the cluster level (episode-level exact McNemar 0.00052). With ten clusters the
  sign-flip test's resolution floor is 0.002 and a suite whose gain concentrates in a few tasks
  loses significance; under the held-out calibration only spatial stays below 0.05. The paper's
  "every suite individually significant" is an episode-level statement and is now qualified:
  at the task-cluster level, three of four suites under the shipped calibration and one of four
  under the held-out, with the pooled effect unaffected.
- **Prediction 3 (cluster interval ≥ 20 % wider): confirmed on the shipped set (1.25–1.46) and
  pooled (1.32); refuted on held-out spatial (0.75) and object (1.00)**, where with ten clusters of
  two pairs the cluster resample is not systematically wider. The correlation is real where the
  per-task effects are heterogeneous (goal, libero_10) and negligible where they are uniform.

Artifact: `results/re4_theory/8_statistics/clustered_bootstrap.json`.
