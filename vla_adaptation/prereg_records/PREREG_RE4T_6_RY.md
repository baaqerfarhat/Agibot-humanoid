# Preregistration: the r_y question as a theory test (re4 theory plan, Part 6; 2026-09-13)

**Written before any run.** On every backbone the r_y fault estimate settles at 40–50 % of the
fault (0.020–0.022 against 0.05) while r_x and r_z reach 90 %; on libero_10 the known-fault
oracle reaches 22/40 where the method reaches 15/40, and the paper reads the gap as "the price
of identification", mostly r_y. This part diagnoses r_y and then tests that reading with one
registered intervention.

**Diagnosis (CPU, stored data; no prediction, three measured quantities).**
(a) Sensitivity: the r_y column of M in the three probed states (45, 25, 5) and its condition
against the other columns. (b) Excitation: per-channel command energy in the healthy logs
(mean |u_i| and the fraction of steps with |u_i| above the deadzone-equivalent), r_y against
r_x, r_z. (c) Model: the FIR's held-out increment R² per channel on the pooled healthy logs
(Part 2.2 fitted a first-order pole of 0.93 on r_y, the slowest channel). The diagnosis names
which of sensitivity, excitation or model error dominates, by the ordering of these three.

**Intervention (GPU, one registered run).** Add r_y excitation to the calibration: ten healthy
episodes at initial states 25–34 with a zero-mean ±0.03 square wave (period 8 steps) added to
the r_y command only, logged with the executed command so the FIR fit sees the excitation
(`error_signal.py` will need a `--excite ry:0.03:8` option; the episodes are healthy-policy
episodes with a known excitation, not faulted ones). Refit the FIR on these ten plus the ten
init-25 episodes; recompute M by the same 0.02 probe at state 25. Then rerun libero_10 n=40
(headline fault, rotation corrected) with the new calibration.

**Predictions.**
1. The r_y plant improves: held-out increment R² on r_y rises by ≥ 0.05 and the r_y estimate on
   the rerun settles at ≥ 70 % of the fault (median over episodes of the last-50-step mean).
2. If 1 holds, libero_10's corrected count moves toward the oracle's 22/40: ≥ 18/40 (from 15,
   held-out 7). If 1 holds and the count stays ≤ 16/40, the identification-price reading is
   refuted: r_y identification is not what limits the long-horizon suite.
3. If 1 fails (the estimate stays below 60 %), r_y is limited by sensitivity or the plant, not
   by excitation; reported as such and prediction 2 is not evaluated.

**Refutation handling.** Prediction 2's second clause is the registered refutation of the
paper's reading; it is reported as primary if it occurs.

---

## Diagnosis and amendment (appended 2026-09-13, before the GPU run; nothing above was edited)

**Diagnosis (stored data).** (a) Sensitivity: the r_y column of M is as large and as well
conditioned as r_x and r_z at all three probed states (diagonal 0.246–0.276 against 0.233–0.253
for r_x and 0.243–0.244 for r_z). (b) Excitation: r_y commands are as large as r_z's (mean |u|
0.039–0.045 against 0.030–0.033; above 0.02 on 52–57 % of steps against 44–49 %). (c) Model: the
six-tap FIR's held-out increment R² is 0.41 on r_y against 0.71 on r_x and 0.96 on r_z, and a
longer window does not help (K = 10, 14, 20: 0.41, 0.39, 0.35). **Model error dominates.** One
autoregressive term (the previous measured increment) lifts r_y to 0.95 and r_x to 0.98: the
r_y channel's motion has a slow tracking pole (0.93, Part 2.2) that a command-only FIR cannot
represent, which is the same "absent AR structure" the collaborator's audit named.

**Amendment: the registered intervention is changed before any run**, because the excitation
intervention above targets a cause the diagnosis rules out. Intervention B: an ARX plant,
ŷ_k = Σ_l W_l u_{k−l} + a·y_{k−1} + c per channel, fitted on the same healthy log, with the
sensitivity used by the estimator rescaled consistently: under a constant command offset f the
ARX residual's steady signature is (1 − a) M f (the measured y_{k−1} absorbs the offset's
accumulated part), so M_arx = diag(1 − a)·M, no new probe. Runner flag `--ar 1`. Predictions 1–3
above stand with "the r_y estimate settles at ≥ 70 % of the fault" as the test of 1 and the
libero_10 count as the test of 2; a spatial n = 20 cell is added as a no-harm check (within 3 of
18/20). Prediction 1's plant clause (R² up by ≥ 0.05) is already met on CPU (+0.54).
