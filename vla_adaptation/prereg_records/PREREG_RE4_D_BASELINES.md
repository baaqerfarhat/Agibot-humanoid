# Preregistration: the decisive missing baselines (re4 Part D, 2026-09-11)

Written before any run. Cohorts: `libero_spatial` n = 20 (tasks 0–9 × inits 45, 46) and
`libero_10` n = 40 (inits 45–48). Fault: the headline uniform +0.05 on all six channels.
Headline constants unless stated (legacy, γ 0.08, dead 0.008, ρ 0.15, clip 0.15, rotation
channels 3,4,5, K = 6, shipped calibration). Protocol: `--scenario-reset`, `--timing`,
section-0 records. Every run pairs its arm against its own frozen-faulted arm; baselines are
compared with the method across runs by (task, init) key under the same protocol.

- **D.0 method reference.** The headline law rerun under this protocol with full logs, so every
  baseline has a same-protocol comparator and the headline has an independent rerun. Prediction:
  corrected within ±3 of the headline cells (18/20, 15/40), each significant.
- **D.1 calibrated static observer, K = 0** (`--fir-k 0`): the same healthy log fitted with a
  static gain and bias, everything else identical. Prediction: D.1 ≤ D.0 on both cohorts. "FIR
  memory earns its place" is supported if D.0 − D.1 ≥ 3 on at least one cohort with the pooled
  paired D.0-vs-D.1 McNemar p < 0.05; refuted if D.1 ≥ D.0 on both cohorts.
- **D.2 matched innovation law** (`--law innov`), identical numeric constants. The ratio rule is
  satisfied by identical constants: both laws compare their residual against the same healthy
  residual scale in the same units. Prediction: on LIBERO the legacy attenuation factor
  1/(1+‖r‖²/ρ²) is close to 1 (‖r‖ ≪ ρ = 0.15), so |D.2 − D.0| ≤ 3 on each cohort with no
  significant paired difference. Refutation: a significant difference in either direction.
- **D.3 known-fault correction through the same interface.** The exact −f applied externally with
  no estimator (`--static-corr -0.05,…`; the runner's static path is unnegated, so the negated
  fault is passed). D.3a on the method's channels (`--corr-dims 3,4,5`), both cohorts: the
  denominator for "fraction of the oracle's recovery achieved", (D.0 − frozen)/(D.3a − frozen).
  Prediction: D.3a ≥ D.0 − 1 on both cohorts and the fraction ≥ 0.8 on spatial. D.3b, all six
  channels, spatial only: the ceiling with translation corrected too. Prediction: D.3b within 2 of
  the spatial healthy rate from Part C.
- **D.4 (optional, run only if time remains).** The best integral gain (ki 0.005) on the
  translation 0.15 cell, paired by key with the method on the same list.

Nothing is reported only if positive; each refutation is reported as the primary result for its
baseline.

---

## Outcome (appended 2026-09-12, after all nine runs; nothing above was edited)

| run | spatial (n=20) | libero_10 (n=40) |
|---|---|---|
| D.0 method, corrected protocol | 6 → 18 (12/0) | 0 → 15 (15/0) |
| D.1 static observer, K = 0 | 11 → 18 (7/0) | 0 → 9 (9/0) |
| D.2 innovation law, same constants | 9 → 18 (9/0) | 1 → 12 (12/1) |
| D.3a known −f on the rotation channels | 8 → 19 (12/1) | 1 → 22 (22/1) |
| D.3b known −f on all six channels | 9 → 20 (11/0) | — |

- **D.0:** 18/20 and 15/40, exactly the headline counts; the same-protocol comparator exists.
- **D.1 (FIR memory earns its place): not established at the registered threshold.** Paired by
  scenario against D.0: spatial 18 vs 18 (1 each way); libero_10 15 vs 9 (D.0-only 7, D.1-only 1,
  p = 0.070); pooled D.0-only 8, D.1-only 2, p = 0.109. The gap of six on libero_10 meets the
  "≥ 3 on at least one cohort" clause, the pooled p < 0.05 clause fails. The refutation clause
  (D.1 ≥ D.0 on both) also fails. Reported as: on the short suite a static gain does as well; on
  the long-horizon suite FIR memory is worth six of forty episodes, short of significance at this n.
- **D.2 (innovation law within 3, no significant difference): confirmed** on both cohorts
  (spatial 18 vs 18; libero_10 12 vs 15, D.0-only 7, D.2-only 4, p = 0.55). Final rotation estimates:
  innovation 0.049 on rx, rz against legacy 0.044 (the attenuation bias); ry at 0.018–0.022 under
  both, which is the under-identified channel, not the law.
- **D.3a (oracle ≥ D.0 − 1 on both; fraction ≥ 0.8 on spatial): confirmed.** Spatial: the method
  achieves (18−6)/(19−8) = 1.09 of the oracle's recovery; libero_10: (15−0)/(22−1) = 0.71. With
  the exact correction on the rotation channels the long-horizon ceiling is 22/40; the estimator
  reaches 15 of those 22.
- **D.3b (within 2 of spatial health): confirmed.** 20/20 against the Part C healthy 19/20.
- The oracle arms still run the estimator internally and never apply it (same as ALOHA's static
  path); `run_configuration.json` says so.
- Correction to the D.3a spatial commit message (a011dc9): the per-episode record is 12 fixed /
  1 broken, not 11 / 0.
