# Preregistration: matched healthy controls on the exact primary pairs (re4 Part C, 2026-09-11)

Written before any run. Plan: `docs/RE4_EVIDENCE_PLAN.md` (branch VLA_Adaptation), Part C.

**Cells.** The exact (task, init) pairs of the headline table: `libero_spatial` and `libero_object`
n = 20 (tasks 0–9 × inits 45, 46); `libero_goal` and `libero_10` n = 40 (inits 45–48). No fault
(`--sev 0`). The headline law and constants (legacy, γ 0.08, dead 0.008, ρ 0.15, clip 0.15,
rotation channels 3,4,5, replan 5). Each run has two arms on the same pairs: **C.1 healthy,
frozen** (no adaptation) and **C.2 healthy, adaptation active**. Run twice: with the shipped
calibration (`results/phase05/error_signal_so3.json`, `openloop_so3.json`) and with the held-out
calibration of record 39 (`results/heldout/error_signal_init25.json`, `openloop_init25.json`).
Protocol for every new re4 run: `--scenario-reset`, `--timing`, and the section-0 records
(`openpi/re4_record.py`). Policy sampling is not pinned; pairing is on (task, init).

**Statistics.** C.1: healthy success per suite and pooled over 120. C.2: paired fixed/broken of
healthy-adapted against healthy-frozen, exact McNemar, per suite and pooled.

**Predictions.**
1. C.1: every suite's healthy rate is at least its headline corrected rate (18/20, 16/20, 29/40,
   15/40), and `libero_10` healthy ≥ 25/40.
2. C.2, each calibration: healthy-adapted within ±3 of healthy-frozen on every suite, ≤ 3 broken
   per suite, no suite significant (p ≥ 0.05), pooled broken ≤ 6 of 120.

**Refutation.** Any suite with healthy-adapted significantly below healthy-frozen, or pooled
broken > 6, is harm on a healthy robot at the primary scale and is reported as the primary
result for that suite. A suite whose healthy rate falls below its headline corrected rate means
the headline arm is at or above health on that sample, and is reported as such.

---

## Outcome (appended 2026-09-11, after all eight runs; nothing above was edited)

| calibration | suite | healthy, no law | healthy, law running | fixed / broken |
|---|---|---|---|---|
| shipped | spatial | 19/20 | 20/20 | 1 / 0 |
| shipped | object | 20/20 | 20/20 | 0 / 0 |
| shipped | goal | 39/40 | 40/40 | 1 / 0 |
| shipped | libero_10 | 38/40 | 37/40 | 2 / 3 |
| shipped | **pooled** | **116/120** | **117/120** | **4 / 3**, p = 1.0 |
| held-out | spatial | 20/20 | 20/20 | 0 / 0 |
| held-out | object | 20/20 | 20/20 | 0 / 0 |
| held-out | goal | 40/40 | 39/40 | 0 / 1 |
| held-out | libero_10 | 38/40 | 39/40 | 2 / 1 |
| held-out | **pooled** | **118/120** | **118/120** | **2 / 2**, p = 1.0 |

- **Prediction 1 (healthy ≥ headline corrected on every suite; libero_10 ≥ 25/40): confirmed.**
  Every suite's healthy rate is above its corrected rate; libero_10 is 38/40 under both calibrations.
- **Prediction 2 (law-running within ±3 of no-law per suite, ≤ 3 broken per suite, no suite
  significant, pooled broken ≤ 6): confirmed** under both calibrations. The weakest cell is the
  shipped-calibration libero_10 (37/40 against 38/40, 3 broken), at the registered bound on
  broken episodes; reported as the weakest, not folded away.
- Healthy competence on the exact primary samples is 116–118 of 120, so the headline corrected
  rate of 78/120 (shipped) or 68/120 (held-out) sits at 67 % and 58 % of health on these samples.
- Adapter compute over all eight runs: 0.17 ms median per step, 0.46 ms at the 99th percentile,
  0.77 ms maximum (Part E; `timing_summary.json` in each run folder).
