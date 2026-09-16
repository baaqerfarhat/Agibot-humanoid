# Preregistration: six-channel correction on libero_10 — does the estimator realise the oracle's
# headroom? (2026-09-16, before the runs)

**Why.** Q5 (`PREREG_Q5_SIXCHANNEL_ORACLE.md`, record 58): the six-channel oracle reaches 38/40 on
libero_10 against 22/40 rotation-only; every adaptive libero_10 count so far (15–25 of 40/60) was
measured under the rotation mask. This registration asks whether the online estimator, correcting
all six channels, gets there.

**Configuration.** The E2 libero_10 design (`PREREG_E2_CORE.md`): the 60 manifest keys with the
explicit sampler schedule, innovation law, γ = 0.08, δ = 0.008, ρ = 0.15, κ = 0.15, K = 6, the E2
fit partition (30 episodes), **correction mask {0,1,2,3,4,5}** (the corrected-channel normaliser is
then the all-channel one). Predictors U6 (unconstrained) and C6 (`--dc-gain 4=0.254`, as E2's C).
**Sensitivity:** the shipped M with its translation diagonal replaced by the E2 fit-checkpoint
finite-horizon own-axis gains (`--m-diag 0=0.257,1=0.263,2=0.291`; the shipped z entry 0.126 is the
state-45 artefact of record 45; the E2 probes gave 0.22–0.29 on six states). The two hazards
named in record 58 are thereby declared: the translation phantom on healthy episodes is measured
by the healthy arms; the z entry is the probed one. Frozen arms are E2's (`libero_10_healthy_off`
56/60, `libero_10_faulted_off` 0/60, same keys and schedule). New arms: healthy U6, healthy C6,
faulted U6, faulted C6, six-channel oracle on the E2 keys (`--static-corr −0.05 ×6`, mask all six).
300 rollouts; U6/C6 faulted and the oracle on GPU 1, the healthy pair on GPU 0 shared with
another user's training (rendering on GPU 1). Outputs `results/six_channel/`.

**Predictions.**
1. Six-channel oracle on the E2 keys ≥ 50/60 (Q5's rate on 40 keys was 0.95).
2. **Faulted C6 ≥ 36/60** (from 25/60 under the rotation mask). **Refutation: C6 ≤ 25/60** —
   translation estimation does not realise the oracle's headroom online. U6 reported alongside;
   C6 − U6 with a task-clustered interval (H2 of E2 repeated on six channels).
3. Healthy: U6 and C6 each lose ≤ 3 of 60 against healthy off (56/60). **Refutation: ≥ 6 lost** —
   the translation phantom makes six-channel correction unsafe on a healthy robot (record 37's
   gate would then be the registered next step).
4. Translation estimate: last-50-step median on x, y within ±30 % of the fault; z reported
   against the probed entry.
**Decision rule.** If 2 and 3 hold, the six-channel configuration becomes the paper's libero_10
result and the mask sentence changes; if 2 fails, the ceiling is reported as not reached online
with the estimate quality of 4 as the reason offered; if 3 fails, the result is reported with its
healthy cost and not adopted. Scorer: `openpi/re4_theory/e2_score.py` with the arm names below
(`--allow-partial` is not used for the decision).

---

## Outcome, four of five new arms (2026-09-16, 04:35; `results/six_channel/`; the oracle arm on these keys is running)

| arm | success / 60 | paired against | settle, % of fault (x / y / z / r_x / r_y / r_z) |
|---|---|---|---|
| healthy off (E2) | 56 | — | — |
| healthy U6 | **47** | off: 10 lost, 1 gained | phantom median z +0.019 (translation reaches the 0.15 clip on some episodes) |
| healthy C6 | **53** | off: 6 lost, 3 gained | phantom median z +0.019; 10 episodes with |z| > 0.05 |
| faulted off (E2) | 0 | — | — |
| faulted U6 | **35** | rotation-mask U 18: 18 won / 1 lost (p < 0.001) | 118 / 79 / 124 / 88 / 55 / 94 |
| faulted C6 | **41** | rotation-mask C 25: 22 won / 6 lost (p = 0.004); U6: 11 / 5 (p = 0.21), task-clustered [−1.7, +21.7] points | 82 / 72 / 115 / 89 / 74 / 93 |

- **Prediction 2 (C6 ≥ 36/60): holds** — 41/60, from 25 under the rotation mask on the same keys.
  Translation correction realises most of what the six-channel oracle promised (Q5: 38/40 ≈ 57/60
  on this suite's rate; the oracle on these exact keys is pending as prediction 1). C6 − U6 is
  +6 with an interval that includes zero.
- **Prediction 3 (healthy ≤ 3 lost): refuted for U6 (10 lost) and at the refutation line for C6
  (6 lost).** The translation phantom named in record 58 is the cause: the healthy estimate on z
  settles at 0.019 (two fifths of the fault size) with episodes driven to the clip, while the
  rotation phantoms stay near zero. The constrained predictor halves the damage and does not
  remove it.
- **Prediction 4 (x, y within ±30 %): holds at the letter** (C6: x 82 %, y 72 %); z over-corrects
  (115–124 %) with the probed entry 0.291, so the probed value is itself high for this suite's
  states or the translation phantom adds to it — reported, not resolved.

**Decision, as registered.** Prediction 2 holds and prediction 3 fails: the six-channel
configuration is **reported with its healthy cost and not adopted**. The paper's libero_10
sentence becomes: the rotation mask caps the suite (Q5); correcting all six channels online
lifts it from 25 to 41 of 60 (oracle pending) at the price of six to ten healthy episodes in sixty,
the translation phantom the healthy-only gate of record 37 was built to hold — that gate on the
six-channel configuration is the registered next step (`PREREG_HEALTHY_GATE.md`'s design on the
E2 keys, 240 rollouts).

## Oracle arm (2026-09-16, 05:44; `libero_10_faulted_oracle`, the E2 keys)

Six-channel exact cancellation on these 60 keys: **57/60** (prediction 1, ≥ 50: holds). Oracle − C6 +16
episodes = **+26.7 points** (task-clustered [+15.0, +38.3], 16 oracle-only, 0 C6-only); oracle − U6 +22 episodes =
+36.7 points [+20.0, +51.7], 22 / 0. (Numbers from `score_libero_10_six.json`; the line first written here
quoted figures that were not the scorer's and was corrected at 05:50.) Healthy, from the same scorer:
U6 − off −15.0 points [−26.7, −3.3], C6 − off −5.0 [−16.7, +3.3]; C6 − U6 faulted +10.0 [−1.7, +21.7].
The whole block, faulted: off 0 → U6 35 → C6 41 → oracle 57 of 60 against the rotation mask's
18 / 25 / 36. The estimator with six channels closes about two thirds of the distance from the
rotation mask to exact cancellation; the healthy cost is the price (record 61). Scorer output
`score_libero_10_six.json`.

