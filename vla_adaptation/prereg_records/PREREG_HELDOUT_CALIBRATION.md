# Preregistration: the headline cells under a held-out calibration (2026-09-10)

**Written before the held-out calibration exists.** Audit finding 1.5: the shipped sensitivity
matrix `M` (`results/phase05/openloop_so3.json`) was identified by replaying task 0 at initial
state 45, and every headline cell evaluates initial states 45/46 (n=20) or 45–48 (n=40); the
healthy FIR log was also collected at `--init-base 45`. Calibration and evaluation overlap.

## Procedure
1. Healthy FIR log: `error_signal.py --healthy-only --episodes 10 --init-base 25` on
   `libero_spatial` (initial states from 25 upward; disjoint from 45–48).
2. `M`: `openloop_id.py --probe 0.02 --probe-init 25` on that log (same probe magnitude and
   steps as the shipped file).
3. The four headline cells, unchanged in every other respect (π0.5, uniform +0.05, rotation
   corrected, γ 0.08, dead 0.008, ρ 0.15, clip 0.15, replan 5; spatial n=20, object n=20,
   goal n=40, libero_10 n=40, inits 45/46 or 45–48), with `--log`/`--openloop` pointing at the
   held-out files. Paired within each run.

## Predictions, registered
- Each cell's corrected count is within 3 of the original (spatial 18, object 16, goal 29,
  libero_10 15 of n); every cell stays individually significant (exact McNemar p < 0.05); the
  pooled fixed/broken pattern stays at ≥ 45 fixed and ≤ 3 broken over the 120 scenarios.
- Refutation: any cell drops by more than 5, or loses significance, or the pooled broken
  count exceeds 5. Then the calibration overlap was load-bearing and the paper says so.
- The held-out plant and `M` themselves: FIR DC gain and `M` diagonal within 15 % of the shipped
  values (the audit's magnitude-independence finding predicts this).

No constants change. The original cells stay as historical evidence; the held-out cells become
the reported ones if the prediction holds.

---

## Outcome (appended 2026-09-11, after all four cells; nothing above was edited)

| suite | original | held-out | delta corrected |
|---|---|---|---|
| spatial n=20 | 8→18, 10/0, p=0.0020 | 9→18, 9/0, p=0.0039 | 0 |
| object n=20 | 5→16, 11/0, p=0.00098 | 7→15, 10/2, p=0.039 | −1 |
| goal n=40 | 15→29, 15/1, p=0.00052 | 18→28, 11/1, p=0.0063 | −1 |
| libero_10 n=40 | 0→15, 15/0, p=6.1e−5 | 0→7, 7/0, p=0.016 | **−8** |
| pooled n=120 | 28→78, 51 fixed, 1 broken | 34→68, 37 fixed, 3 broken | −10 |

- **Prediction 1 (each cell within 3): refuted on `libero_10`** (−8, threshold was −5); holds on
  the other three. Every cell remains individually significant; pooled broken 3 ≤ 5.
- **Prediction 3 (M within 15 %): refuted on y and z** (19–21 % lower at init 25); rotation
  within 8 %.
- Conclusion registered in advance for this case: "the calibration overlap was load-bearing and
  the paper says so." It says so, per suite: unchanged on three, 15→7 on the long-horizon suite.
