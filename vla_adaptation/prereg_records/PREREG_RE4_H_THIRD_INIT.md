# Preregistration: the sensitivity matrix at a third initial state (re4 Part H, 2026-09-11)

Written before the probe. Record 39 compared M probed at initial state 45 (shipped) with M
probed at initial state 25 (held-out): rotation within 8 %, y and z 19–21 % lower, condition
number 3.0 against 3.1. Two points cannot say whether that is a trend or scatter.

**Procedure.** Healthy FIR log `error_signal.py --healthy-only --episodes 10 --init-base 5`
(initial states 5–14, disjoint from 25–34 and 45–48), then `openloop_id.py --probe 0.02
--probe-init 5` on it. Nothing else changes.

**Predictions (the plan's own).** Translation diagonal of M differs from the shipped one by
roughly 20 % (registered band: 10–30 % on at least one of x, y, z); rotation diagonal within
10 % of the shipped one on all three; condition number between 2.5 and 3.5.

**Refutation.** Rotation moving by more than 10 %, or the condition number leaving [2.5, 3.5],
means the configuration dependence is not confined to the translation block, and record 39's
reading ("the rotation-corrected cells depend on the rotation block, which held") is weakened.

---

## Outcome (appended 2026-09-12; nothing above was edited)

| probe state | diag x, y, z | diag rx, ry, rz | cond |
|---|---|---|---|
| 45 (shipped) | 0.297, 0.272, 0.126 | 0.253, 0.276, 0.244 | 2.95 |
| 25 (held-out) | 0.268, 0.221, 0.100 | 0.233, 0.259, 0.244 | 3.06 |
| 5 (this run) | 0.253, 0.259, **0.262** | 0.252, 0.246, 0.243 | **1.29** |

- Translation 10–30 % off on at least one axis: **met on x (−15 %)**, but z is **+107 %**, far
  outside the band the plan wrote down: the shipped z sensitivity (0.126) is half of the other two
  translation axes, and at initial state 5 it is their equal.
- Rotation within 10 % on all three: **refuted**, ry is −11 %.
- Condition number in [2.5, 3.5]: **refuted**, 1.29.

Reading, as the refutation it is: the configuration dependence of M is not confined to a 20 %
band on the translation block. Between states 45 and 25 the rotation block held to 8 % and z
fell 21 %; at state 5, z doubles, ry moves 11 %, and M becomes nearly isotropic. The shipped
matrix's small z entry looks like a property of initial state 45 rather than of the plant.
Record 39's sentence "the rotation-corrected headline cells depend on the rotation block, which
held" is true of the 45/25 pair only; three points do not make a monotone trend, they show that
the entry that moves depends on the state. `results/re4_evidence/calibration/third_init5.json`.
