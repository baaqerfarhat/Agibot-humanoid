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
