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
