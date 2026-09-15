# Preregistration: E1 physical continuations on the Panda (2026-09-14, before the runs)

Per `iclr2027/EXPERIMENT_PLAN.md` E1 (Panda half; ALOHA is a separate registration), on the
partitions of `PREREG_UNIFIED_PARTITIONS.md`: fit = state 39 tasks 0–5, qualification = state 39
tasks 6–9, locked test = state 34 tasks 0–9. Tool `openpi/re4_theory/e1_continuations.py`.

**Design.** Checkpoints at steps 30 and 70 of every source (missing if fewer than 100 commands
remain); from a full-state snapshot (MjSimState, solver warm start, actuator state, ctrl,
gripper target) the same next 100 nominal commands run in seven branches: healthy, duplicate
healthy, faulted/off, legacy from zero, innovation from zero, exact same-mask cancellation, and
hold from one common supplied estimate. Fault +0.05 on the normalised rotation-y command,
constant; correction mask {3, 4, 5}; deployed constants (γ 0.08, δ 0.008, ρ 0.15 over six
channels, κ 0.15); the deployed shipped FIR as predictor. Pass 1 runs fit + qualification sources
without the hold branch; the supplied estimate is the median over the qualification checkpoints of
the innovation branch's last-20-step mean estimate, frozen literally; pass 2 runs the locked test
sources with all seven branches. Physical traces (joint position/velocity, end-effector position
and orientation, controller goal, object poses, contact) are logged apart from observation traces
(predictor residual, estimate) and command traces (nominal, correction, executed, remaining
injected disturbance). Scorer `openpi/re4_theory/e1_score.py`: physical error against the healthy
branch of the same checkpoint in rad (joint), m (end-effector), rad (orientation angle), and the
augmented joint metric; integrated and endpoint; aggregated by source episode.

**Predictions (locked test sources, 10; report uncertainty at the source level).**
1. Duplicate healthy replay matches the healthy branch to a joint gap ≤ 1e-9 rad on every
   checkpoint (snapshot fidelity); a failure repairs the tooling and reruns the locked set.
2. Persistent physical offset: the faulted/off branch's end-effector deviation from healthy grows
   through the continuation (endpoint > 2× the value at step 20 on the median source), and the
   exact-cancellation branch's deviation is at the duplicate's level (≤ 1e-6 m) — the correction
   at the interface removes the physical consequence entirely.
3. The adaptive branches from zero lie between: integrated end-effector error of legacy and
   innovation below faulted/off with the paired source-level 95 % interval excluding zero; their
   endpoint deviation is a persistent offset (not decaying to the healthy trajectory) whose size
   scales with the remaining disturbance integrated over the transient — the execution-memory
   signature the theorem relies on; the hold branch's endpoint offset is proportional to the
   supplied estimate's error.
4. Remainder envelope: a signed finite-memory linear response fitted on the six fit sources and
   frozen after the four qualification sources predicts the locked sources' end-effector
   deviation with ≥ 90 % whole-continuation coverage and a median nonzero endpoint bound/error
   ratio ≤ 10 (the plan's operational thresholds); a geometric-decay-only model does worse on the
   paired integrated-error contrast. This prediction's model is registered in its own note before
   the locked pass is scored.

**Refutation handling.** Each prediction scored separately; failures primary. Prediction 2's
first clause failing (no growth) says the OSC controller does not integrate a rotation-y offset
into persistent pose error over 100 steps; prediction 3 failing on the interval says 10 sources
do not resolve it.

---

## Amendment before the runs (2026-09-14, 22:10): horizon 50, checkpoints 20 and 40

The fit/qualification sources came in at 83–154 commands (healthy spatial episodes end at
success); with checkpoints at 30 and 70 and a 100-step continuation only one source would carry
one checkpoint. The plan caps continuations at 100 steps; this registration sets the horizon to
**50 steps** (2.5 s at 20 Hz) and the checkpoints to **20 and 40**, so every source of ≥ 70
commands carries the first checkpoint and every source of ≥ 90 the second (7 of the 10
fit/qualification sources). The same rule applies to the locked test sources unseen. Prediction
2's growth clause is read at step 20 against the endpoint at step 50; nothing else changes. No
branch outcome had been produced when this was written (the E1 passes had not started).

---

## Outcome, locked test sources (2026-09-14, 22:22; `results/iclr_unified_v1/E1/pass2_test.json`, `pass2_score.json`, `memory_model_score.json`)

Ten locked sources (state 34, tasks 0–9), 17 of 20 checkpoints (three sources too short for
step 40), seven branches each, 50-step continuations. Supplied hold estimate (frozen from the
qualification innovation branches, pass 1): 0.034 on r_y.

| branch | integrated ee error (cm·step, median over sources) | endpoint ee deviation (cm) [IQR] | endpoint angle (rad) | remaining r_y disturbance |
|---|---|---|---|---|
| healthy duplicate | 0 | 0 | 0 | 0 |
| exact cancellation | 0 | 0 | 0 | 0 |
| faulted / off | 33.9 | **1.58** [1.11, 2.18] | 0.274 | 0.050 |
| legacy from zero | 19.8 | 0.99 [0.85, 1.17] | 0.143 | 0.027 |
| innovation from zero | 19.6 | 0.95 [0.82, 1.25] | 0.141 | 0.025 |
| hold (supplied 0.034) | 12.0 | 0.53 [0.39, 0.68] | 0.091 | 0.017 |

1. **Snapshot fidelity: holds.** Duplicate healthy replay gap 0.0 rad on all 17 checkpoints.
2. **Persistent physical offset: holds.** Faulted/off deviation at the endpoint is 3.0× its
   value at step 20 (median; registered > 2×); exact same-mask cancellation is identical to
   healthy to the last digit (max endpoint deviation 0). A constant +0.05 rotation-y command
   offset is integrated by the controller into a growing pose error, and removing it at the
   interface removes the physical consequence entirely.
3. **Adaptive branches: holds.** Integrated ee error below faulted/off for both laws with the
   paired source-level 95 % interval excluding zero (legacy −0.121 m·step [−0.162, −0.074];
   innovation −0.119 [−0.163, −0.063]); their endpoint offsets persist (≈ 1 cm, not decaying
   toward healthy) with 50–54 % of the disturbance still un-cancelled at step 50; the hold branch
   at a supplied estimate carrying 33 % of the fault ends at 34 % of the faulted deviation
   (0.53 / 1.58 cm) — the offset scales with the remaining disturbance, the execution-memory
   signature.
4. **Remainder envelope: coverage and ratio hold, the memory model's advantage does not.**
   The signed finite-memory model (L = 20, frozen after qualification) covers **91.8 %** of locked
   continuations whole (registered ≥ 90 %) with a median endpoint bound/error ratio of **2.9**
   (registered ≤ 10). But its paired integrated-error contrast against the comparators does not
   exclude zero and is slightly worse (memory − geometric +0.016 m·step [−0.003, +0.035]; memory −
   neutral +0.016 [−0.004, +0.035]). The geometric comparator fitted λ = 0.999 on the fit sources,
   i.e. it chose the neutral limit itself: a pure accumulation of the remaining disturbance
   predicts the Panda's pose deviation as well as a 20-tap signed memory does. This is the
   Part 1 finding (λ̂ = 1.00, no same-command contraction) seen from the disturbance side, and it
   is reported as primary for prediction 4's last clause.

**Reading.** On this plant the physical execution component behaves as an integrator of the
remaining command disturbance over a 50-step horizon: what the estimator has not yet cancelled
accumulates into pose error and stays. The envelope is usable (coverage 0.92, ratio 2.9), the
memory distinction is not resolved against accumulation on ten sources. ALOHA's half of E1 is
not run in this pass.

---

## Supersession and corrected protocol (2026-09-15, 00:40; after `iclr2027/EXPERIMENTS_ASAP.md` §3)

**Defect, confirmed.** In the driver as run (`e1_continuations.py` at f362428/a939d6d), after branching at
checkpoint 20 with H = 50 the healthy prefix was advanced by the whole continuation to step 70;
for the requested checkpoint 40 the prefix slice `cmds[70:40]` was empty, the snapshot was taken
at step 70, and the branches were fed commands 40–89 with a predictor history ending at step 40.
Every **second** checkpoint is therefore mis-indexed: 5 of 11 in fit, 3 of 7 in qualification,
7 of 17 in the locked test (15 of 35; 97 of 227 branch traces). The duplicate replay's zero gap
proved repeatability of that mismatched branch, not correct indexing. First checkpoints (step
20, replayed from a fresh reset) are correctly indexed. The amendment above said seven eligible
second checkpoints in fit/qualification; the raw pass holds eight — corrected here, the original
text left as written.

**Consequences.** The pass-1/pass-2 scores (`pass1_score.json`, `pass2_score.json`,
`memory_model_score.json`) and the hold estimate derived from qualification checkpoints are
**superseded for confirmatory use**; the outcome section above is retained as the record of
what was scored, marked invalid. Record 55 carries the same marking. The first-checkpoint traces
may support a separately labelled post-hoc sensitivity analysis only. The state-34 test outcomes
have been seen and are no longer an unseen test set.

**Corrected driver (v2, committed before any new run).** The healthy prefix is replayed from the
previous checkpoint to the next; the replay index is asserted equal to the requested checkpoint;
prefix and continuation command hashes and the predictor's initial history are recorded; the
applied estimate is logged per step in every branch so the hold branch's error is scored from the
vector actually applied. Snapshot contents unchanged (MjSimState, warm start, actuator state,
ctrl, gripper target); the OSC controller's goal is recomputed from the current pose at every
step and its interpolator state is re-set by `set_goal`, which is why restore-then-step matched
the healthy continuation exactly in the smoke test — this will be verified explicitly in the rerun
by comparing a fresh-reset prefix continuation against a restored branch on the 20/40 overlap.
Units: per-step sums are labelled m·step; time integrals (×0.05 s) are reported alongside. The
memory model saves all coefficients; the geometric comparator's one-step fit objective and grid
boundary are stated; the envelope is labelled empirical.

**Fresh partitions for the rerun (declared now; outcomes unseen).** Fit and qualification:
re-run on the same state-39 sources with v2 (their first-checkpoint outcomes were seen; the
model choice does not use outcomes, and the partition is not the test). Locked test:
**libero_spatial state 33, tasks 0–9** (unused by any stored result or calibration per the
ledger), collected after the E2 core releases the server; the rule of checkpoints 20 and 40 with
H = 50 and the four predictions stand. Missing checkpoints stay missing by the length rule.

**Prospective coupled prediction (new, per §4 of the ASAP note).** Before opening the fresh test
branches, from the known initial observer state, the injected fault, the frozen U predictor and
M, the declared nominal command sequence and the frozen physical-response model, the adaptation
transient (estimate, correction, remaining disturbance) and the physical deviation are forecast
without the realised test correction as an input; the frozen quantitative decision is: **the
innovation branch's forecast integrated ee deviation is within 30 % of the measured value on the
median locked source, and the forecast ordering faulted > innovation ≈ legacy > hold > exact is
observed.** Forecasts are written to `results/iclr_unified_v1/E1/forecast_v2.json` before the
test pass and scored after.
