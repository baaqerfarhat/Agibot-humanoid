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
