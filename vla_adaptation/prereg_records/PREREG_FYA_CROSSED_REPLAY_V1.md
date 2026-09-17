# PREREG: crossed command-stream replay, v1 (Panda / LIBERO-Spatial, pi0.5, archived Stage 2 cohort)

Registered 2026-09-16 (16:05), after the read-only extraction and a two-key technical pilot, and **before any
crossed cell (J10, J01) was collected on the eligible keys**. Implements
`papers/frozen_yet_adaptive_closed_loop/EXPERIMENT_PLAN.md` §§2–6. The previous registration
(`PREREG_FYA_RECOVERY_DEADLINE_V1.md`) is unchanged. Campaign root `results/fya_crossed_replay_v1/`.

## 1. Question and estimand

On the archived Stage 2 keys (states 13, 14, seed 83001; fault +.05 on r_y from policy step 30; NT, cap .05,
mask {3,4,5}), how much of the physical benefit T = J00 − J11 is the correction under the Off-generated stream
(D0 = J00 − J10) and how much is the changed nominal stream produced when the reacting policy saw corrected
motion (R1 = J10 − J11), and do they interact (I = D1 − D0 = R1 − R0)? **I is the single primary contrast.** The
estimand is restricted to the strictly eligible archived cohort (18 keys, 9 tasks × 2 states): the archived
outcomes were inspected, so this is an extension of an inspected cohort with prospective scoring, not an
untouched test.

## 2. Frozen configuration and cells

Deployed arrays from the fault_nt telemetry header (`configuration.json`, sha `c92c273cc2e278a9`): W (r_y tap sum
.10281), M (r_y .2759), mask {3,4,5}, γ .08, δ .008, ρ .15, cap .05, all-channel normalisation, law legacy (NT),
deadzone mode zero. Fault vector (0,0,0,0,.05,0) on every window step. Replayed input = the archived
seven-dimensional `raw_action` (never `nominal_command`). Window: 50 steps from env step 40 (policy step 30)
after a 10-step warm-up with the archived warm-up command and the 30-step common prefix. Cells per key from one
restored snapshot: ref (healthy_off stream, no fault, no correction), J00, J10, J01, J11, plus J00_fresh and
J11_fresh (reset + prefix + continuation without restore). NT is rerun causally in J10 and J11 from a zero
estimate with the FIR history of the prefix's sent commands; a step's correction uses the pre-update estimate.
Cell order rotated by key index (J00,J10,J01,J11 shifted by index mod 4), recorded. 7 continuations per key,
126 for 18 keys.

## 3. Eligibility and fidelity (locked)

Eligible: the 18 keys in `source_keys.csv` / `extraction_audit.json` (sha `50ffa97aee1ebce2`): every arm ≥ 80 policy
steps and exact (tolerance 0) prefix agreement of raw actions, positions and joints across healthy_off, fault_off
and fault_nt. Excluded before registration: task 0 state 14 (72–75 steps) and task 0 state 13 (healthy_off prefix
differs: raw action 2.7e-3, joints 5.1e-4 rad). No replacement, no reconstruction of the task-0/13 reference.

Fidelity, all required per key, tolerances 1e-8 rad (joints), 1e-8 m (position), 1e-8 rad (SO(3) angle),
exact raw actions: prefix vs archive; ref vs archived healthy_off; J00 vs archived fault_off; J11 vs archived
fault_nt including corrections (≤ 1e-8); J00_fresh/J11_fresh vs restored J00/J11; complete 50-step window. A
key failing any check is excluded from the analysis and listed. Continuation after a wrapper done: the
terminated-episode guard is cleared and physics continues (registered semantics); done steps are reported.

## 4. Pilot disclosure

Technical pilot on the two **excluded** task-0 keys (`pilot/pilot_run.json`, cells ref/J00/J11/fresh only,
no crossed cell): prefix gap 0.0; J00 vs fault_off 0.0; J11 vs fault_nt 0.0 (corrections 0.0); fresh vs
restored 0.0 on both keys; ref fails on 0/13 as expected (mismatched archived healthy prefix, 1.2e-3 rad) and
the window is 42/50 on 0/14. The pilot chose no endpoint, setting or margin; exact reproduction is feasible.

## 5. Endpoints, margin, statistics

Primary J = .05 Σ_k ‖p_k − p_k^ref‖² over the window (m² s), one immutable ref path per key. Angular energy
(rad² s), endpoints (m, rad), done steps, clipped-input steps and final estimates are secondary/descriptive.
Margin **δ_I = 1e-5 m² s** (≈ 2 mm constant error over 2.5 s; chosen from the observed cost scale of the
closed campaign, J_off ≈ 1.2e-4). Estimand: equal-task mean (9 tasks × 2 states). Task-clustered percentile
bootstrap, 10,000 draws, seed 20260916; per-task effects; leave-one-task-out; source-weighted sensitivity.
Decision on I (equal-task 95 % interval): above +δ_I or below −δ_I → practically resolved interaction (either
sign informative); within [−δ_I, +δ_I] → small at this margin; otherwise unresolved. D0, R1, T, R0, D1 are
descriptive. R1/T reported per source only, flagged near T = 0. No task-success claim (all arms 20/20).

## 6. Expectations (directional; refutation reported as primary)

X1 D0 > 0 (correction under the Off stream reduces cost; the fixed-command result). X2 T > 0 (the archived
adapted path is closer to healthy than the archived off path). X3 no prior on the sign of R1 or I: the
reacting policy may partly undo (R1 < 0) or reinforce (R1 > 0) the correction; this is what the matrix measures.

## 7. Stop rules and provenance

No key added, no law switched, no window or margin changed after any crossed cell is read. Code: driver
`fya_crossed_replay.py` sha `225921ed9a077806`, extractor `1d172243129b9971`, scorer `c266f8d112a0df6f`, `adaptive_law.py` `f116c33e909bb5db`.
Chain: `scripts/re4/fya_crossed_chain.sh` (GPU 1 EGL, CPU replay, no policy server). Outcome in §8.

## 8. Outcome, primary matrix (collected 16:13–16:19 and re-collected 16:25–16:31; `runs/crossed_run.json.gz`, `analysis/`)

**Collection note.** The first collection (`runs/crossed_run_v1_anglecheck_bug.json.gz`) marked every key
invalid because the SO(3) angle check used arccos(|q_a·q_b|), which cannot resolve below ≈ 1e-5 rad even for
identical quaternions. The formula was replaced by 4·asin(min‖q_a ∓ q_b‖/2) (driver sha now `9753b94f6e73be3a`) and the
deterministic collection re-run; every cost, action and gap of the second run equals the first (the crossed
cells were not inspected between the runs; only the fidelity lines were read). Tolerances unchanged.

**Fidelity.** 15 of 18 keys pass every check with gaps exactly 0.0 (prefix, ref, J00, J11 incl. corrections,
fresh route). Three state-14 keys fail the registered tolerances and are excluded: task 2 (J00 vs live 2.0e-5
rad), task 5 (ref vs live 4.9e-8 rad), task 7 (J00 8.7e-5 rad; J11 3.3e-5, corrections 1.1e-5). In all three the
fresh and restored replay routes agree exactly with each other, so the replay is self-consistent but the live
run's simulator state differed slightly at these keys (the live env had executed the state-13 episode of the same
task immediately before; replay resets without that history): a hidden-state effect, reported, not repaired.
No window terminated early except one done flag each in J00, J01, J11 (task success inside the window; physics
continued under the registered semantics); no clipped input.

**Matrix, 15 keys / 9 tasks, equal-task estimand, task-clustered 95 % intervals (m² s).**

| Cell | J median | endpoint (mm) |
|---|---:|---:|
| J00 (M0, off) | 8.26e-4 | 31.4 |
| J10 (M0, NT) | 7.85e-4 | 25.9 |
| J01 (M1, off) | 6.74e-4 | 25.5 |
| J11 (M1, NT) | 3.97e-4 | 25.9 |

| Contrast | estimate | 95 % CI | sources +/− |
|---|---:|---|---|
| **I = D1 − D0 (primary)** | **2.3e-5** | **[−2.7e-5, 7.6e-5]** | 8 / 7 |
| D0 = J00 − J10 | 8.9e-5 | [2.5e-5, 1.5e-4] | 12 / 3 |
| D1 = J01 − J11 | 1.1e-4 | [6.4e-5, 1.6e-4] | 13 / 2 |
| R0 = J00 − J01 | 3.8e-4 | [−9.4e-5, 9.1e-4] | 12 / 3 |
| R1 = J10 − J11 | 4.1e-4 | [−4.7e-5, 9.3e-4] | 12 / 3 |
| T = J00 − J11 | 5.0e-4 | [9.4e-6, 1.0e-3] | 14 / 1 |

Source-weighted sensitivity: I 1.2e-5, D0 1.1e-4, R1 4.9e-4, T 6.0e-4. Leave-one-task-out I ranges 8e-6 to
3.7e-5. Per-task I from −8.7e-5 (task 6) to +1.4e-4 (tasks 7, 8).

**Decision: interaction unresolved** at δ_I = 1e-5 (the interval spans both meaningful signs). X1 holds
(D0 > 0, 12/15). X2 holds (T > 0, 14/15). R1 is large in point estimate (≈ 80 % of T) but its interval includes
zero: the changed stream generated under adaptation moves the path toward the healthy reference on 12/15 keys,
with heavy tails. Angular energy: D0_r 2.9e-2, R1_r 1.3e-3, I_r 2.1e-3 rad² s (descriptive).

**Reading.** Costs here are dominated by policy-path divergence (J00 median 8.3e-4 versus 1.2e-4 in the
fixed-command study: the reacting healthy reference re-plans differently, and even the adapted live path J11 ends
26 mm from it). Within that, the correction term D0 (8.9e-5) matches the fixed-command benefit (8.8e-5) almost
exactly, and the decomposition attributes most of T to the stream term R1 without resolving the interaction.
The matrix is a descriptive result on 15 archived keys; no task-success claim.

## 9. Registration addendum: optional matrices (committed 16:28 machine time, before any optional crossed cell was collected)

Three incremental matrices from EXPERIMENT_PLAN.md §5 priorities 2–4, same keys, same configuration, same
fidelity rules and tolerances, same cost, margin and statistics; each reported separately (no selection between
laws or conditions; all three are reported whatever their signs). Extraction of the extra archived arms:
`optional/` (audit sha `5eaf3493ef735113`; all 18 primary-eligible keys are also eligible for every optional arm: exact
prefix agreement, ≥ 80 steps, zero corrections before enablement). Driver sha `2fdf74bced3ea729` (`--matrix`).

| Matrix | M0 | M1 | adapter in J10 / J11 | fault | diagonal archives |
|---|---|---|---|---|---|
| delayed NT | fault_off | delay_nt | NT causal, corrections and updates suppressed for window steps 0–9 | on | fault_off / delay_nt |
| innovation | fault_off | fault_innovation | innovation law causal | on | fault_off / fault_innovation |
| healthy control | healthy_off | healthy_nt | NT causal | **off** | healthy_off / healthy_nt (J00 ≡ ref, cost 0) |

Endpoints as in §5 (I primary per matrix, δ_I = 1e-5, equal-task bootstrap seed 20260916). For the healthy
control, D0 = J00 − J10 = −J10 is the cost of false updates on the off-generated healthy stream (expected ≤ 0),
R1 = J10 − J11 the stream term, and I the interaction; this matrix is the controlled counterpart of the healthy
phantom observation and is descriptive of mechanism, not a proof that a spurious estimate caused the one task loss.
Runs `runs/crossed_{delay,innovation,healthy}.json.gz`, analyses `analysis_{delay,innovation,healthy}/`.
Expectations: delay matrix D0 smaller than the immediate D0 (memory); innovation matrix ≈ NT matrix; healthy
D0 < 0 with |D0| small next to the faulted D0. Outcome in §10.

## 10. Outcome, optional matrices (collected 16:28–16:37 in parallel; `runs/crossed_{delay,innovation,healthy}.json.gz`, `analysis_*/`)

Fidelity: the same three state-14 keys fail as in the primary matrix (tasks 2, 5, 7; the healthy matrix loses
only tasks 5 and 7): 15 valid keys for the delayed and innovation matrices, 16 for the healthy control. All
diagonals of the valid keys reproduce the archived delay_nt, fault_innovation and healthy_nt paths exactly.

| Matrix | D0 = J00 − J10 | R1 = J10 − J11 | T = J00 − J11 | I (primary) | decision |
|---|---|---|---|---|---|
| immediate NT (§8) | 8.9e-5 [2.5e-5, 1.5e-4] | 4.1e-4 [−4.7e-5, 9.3e-4] | 5.0e-4 [9.4e-6, 1.0e-3] | 2.3e-5 [−2.7e-5, 7.6e-5] | unresolved |
| delayed NT (10 steps) | 5.2e-5 [4.9e-6, 1.0e-4] | 9.7e-5 [1.0e-5, 1.9e-4] | 1.5e-4 [4.5e-5, 2.6e-4] | 7.5e-6 [−1.7e-5, 3.2e-5] | unresolved |
| innovation | 8.9e-5 [2.2e-5, 1.6e-4] | 5.0e-4 [1.1e-4, 1.0e-3] | 5.9e-4 [1.7e-4, 1.1e-3] | 1.2e-5 [−3.8e-5, 6.9e-5] | unresolved |
| healthy control (fault off) | −2.4e-5 [−4.2e-5, −7.9e-6], 0/16 + | −3.6e-4 [−5.6e-4, −1.7e-4], 0/16 + | −3.9e-4 [−5.9e-4, −1.9e-4] | 2.6e-5 [−1.3e-5, 7.5e-5] | unresolved |

Cell medians (m² s; endpoint mm): delayed J00 8.3e-4 / J10 8.1e-4 / J01 7.4e-4 / J11 6.6e-4 (31 / 29 / 33 / 32);
innovation 8.3e-4 / 7.8e-4 / 5.4e-4 / 3.7e-4 (31 / 25 / 25 / 22); healthy 0 / 8.5e-6 / 8.0e-5 / 7.8e-5
(0 / 2.2 / 12.7 / 10.7).

**Expectations.** Delay: D0 smaller than immediate (5.2e-5 vs 8.9e-5) and T a third of the immediate T:
holds. Innovation ≈ NT: holds (D0 identical, R1 now with an interval excluding zero, 11/15 keys). Healthy:
D0 < 0 on every key with |D0| a quarter of the faulted D0: holds; in addition R1 < 0 on every key and ten times
|D0|: **the reacting policy's response to the small false corrections (J01, 12.7 mm from the healthy reference)
costs an order of magnitude more than the false corrections themselves (J10, 2.2 mm)**. This is a controlled
physical statement about the healthy phantom's mechanism on 16 archived keys; it is not evidence that the
phantom caused the one healthy task loss (19/20), and proximity to one healthy path is not task success.
Interaction unresolved in every matrix at δ_I = 1e-5.
