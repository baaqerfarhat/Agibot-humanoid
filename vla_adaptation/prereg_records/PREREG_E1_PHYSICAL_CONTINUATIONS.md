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
