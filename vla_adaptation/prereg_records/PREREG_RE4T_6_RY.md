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

---

## Outcome, intervention B on `libero_spatial` n = 20 (2026-09-13; `results/re4_theory/6_ry/arx_spatial/`)

ARX(1) plant fitted on the same healthy log (AR coefficients per channel 0.82, 0.73, 0.88, 0.31,
0.63, 0.06; M rescaled by 1 − a), headline fault, rotation corrected, headline 20 scenarios.

- **Prediction 1 fails (primary).** The r_y estimate settles at **32 %** of the fault (median over
  20 episodes of the last-50-step mean, 0.016 of 0.050; IQR 0.010–0.020), below the 60 % floor of
  prediction 3 and below the FIR plant's 41 % on the same scenarios (T1) and the innovation law's
  45 % (T4). The plant clause of prediction 1 (held-out R² up by ≥ 0.05) was met on CPU and is
  irrelevant to the estimate. r_x and r_z settle at 94 % and 97 % (FIR: 87 %, 88 %).
- **No-harm check passes:** 9/20 → **20/20** (registered: within 3 of 18/20; between-run against
  T1's 16/20 with an unpinned sampler, n = 20).
- **Prediction 2 is not evaluated** by the registered branch (1 failed); the `libero_10` n = 40
  run was already queued and is reported below as unregistered when it lands.

**Why the ARX did not move r_y — the lever was never the model order.** On the healthy
calibration log every fitted plant identifies the same r_y DC gain: FIR K = 6, 0.103; FIR K = 20,
0.118; ARX(1), 0.099 (direct term 0.037 over 1 − 0.627). The open-loop probe gives 0.276, and the
onset transient in T2 (record 49) shows the residual plateau at 0.30 per unit of offset — the
probe is right for an offset. The estimator's steady ratio follows (fitted DC gain)/(probed M)
on every rotation channel: r_x 0.90 predicted against 87–95 % measured, r_y 0.36–0.43 against
32–45 %, r_z 1.02 against 88–98 % (the deficit on r_x, r_z is the attenuation). The healthy
closed-loop regression under-identifies the r_y offset gain by 2.7× and any plant fitted to that
log inherits the number, whatever its order; the correction itself is then executed at 0.28 per
unit and predicted at 0.10, so the applied estimate feeds back into the residual with the
difference (r ≈ M f − (M − G_fit) f̂ on that channel) and the fixed point sits below the fault.
The excitation reading and the AR-structure reading are both refuted; the diagnosis that stands
is a DC-gain mismatch between closed-loop identification and the open-loop probe, specific to
the channel with the slowest tracking pole (0.93, Part 2.2). The intervention that follows from
it — constrain the plant's DC gain to the probed M (or identify from open-loop segments) — is not
registered here and was not run in this pass.

Not used: a per-episode regression of the residual on the applied correction (T1) gives
negative slopes on all three rotation channels including r_z, where the model predicts none —
the two co-evolve through the law, so that check is confounded and is not evidence either way.

## Outcome, intervention B on `libero_10` n = 40 (2026-09-13; `results/re4_theory/6_ry/arx_libero_10/`), unregistered

Prediction 2 is not evaluated under the registered branch (prediction 1 failed on the spatial
cell). For the record: ARX plant, headline fault, rotation corrected, the 40 shipped scenarios:
frozen 0/40, adaptive **11/40** (11 fixed, 0 broken), r_y settle **37 %** (r_x 92 %, r_z 95 %).
Against the FIR plant's 15/40 (shipped calibration) and 7/40 (held-out) it sits between the
two; had the branch been live, a count ≤ 16/40 would have refuted the identification-price
reading. It is consistent with §49–50: the r_y estimate is not what an ARX plant changes.

## Scoring definition of the plant-fit R² (added 2026-09-14, after the consistency review)

The R² values quoted above were computed in-session without a stored script. They are now
reproduced by `openpi/re4_theory/ry_fit.py` (output `results/re4_theory/6_ry/fit_r2.json`):
per-axis ridge (λ = 0.01, intercept), **leave-one-episode-out on the pooled healthy logs** (the
shipped three episodes plus the held-out ten at init 25). Reproduced values, r_x / r_y / r_z:
FIR K = 6 **0.76 / 0.39 / 0.97** (quoted 0.71 / 0.41 / 0.96); FIR K = 20 r_y **0.35** (quoted 0.35);
ARX(1) **0.94 / 0.96 / 0.98** (quoted 0.98 / 0.95). In-sample on the deployed three-episode log the
same models score 0.52 / 0.34 / 0.97 (FIR) and 0.76 / 0.88 / 0.97 (ARX). The quoted numbers were
off by up to 0.05 on r_x; the diagnosis (r_y a third of the other rotation channels under the FIR,
lifted to the others' level by one AR term) is unchanged. The record and the paper now carry the
reproduced values.
