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
