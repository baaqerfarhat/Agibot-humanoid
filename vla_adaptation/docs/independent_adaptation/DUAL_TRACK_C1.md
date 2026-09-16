# Dual track C1: state-dependent vs input-only adaptive control, both contraction-based

2026-09-14. Two independent tracks: Track 1 (Claude Opus 5) and Track 2 (a separate Claude Fable 5.1
agent standing in for Codex, unseeded, read-only on tracked files, no GPU jobs). Converged here;
every load-bearing claim below was reproduced by the other track or by a second instrument.

## The question had to be split

"State-dependent" can enter this adapter in three places, and they have different answers:

| axis | input-only | state-dependent | run on a robot? |
|---|---|---|---|
| **P** predictor | FIR on past commands | ARX: past measured increments | yes (`--ar 1`; descriptor ARX2) |
| **R** disturbance regressor | constant offset (Φ = I or E) | Φ(s)θ̂ over configuration / velocity / command | **never** |
| **C** error signal ("contraction") | none | tracking a physical reference error (joint-Lyapunov / composite MRAC) | ALOHA only (null); Panda: see below |

"Contraction" cannot come from the plant: the Panda under OSC_POSE measures λ̂ = 1.00 in position
(record 48). It must be supplied by feedback — the adapter's tracking term or the frozen policy's own
replanning loop.

## Converged answers

**Literal question — contraction + state-dependent vs contraction + input-only: no, on this paper's
fault families.** Both simulations put the difference within about ±0.02 for constant offsets,
configuration-dependent faults and healthy runs. The one exception is friction-like faults, where
the sign depends on model accuracy (Track 1: +0.16 with an exact model, −0.05 under the measured gain
error). What can improve performance is the contraction mechanism itself, which the current method
lacks.

- **P, state in the predictor — no.** Under exact models the FIR residual and the ARX innovation are
  related by an invertible filter (r_A,k+1 = r_F,k+1 − λ r_F,k), so neither carries more fault
  information; they differ only in time-filtering and misspecification (Track 1). On the robot, every
  closed-loop-identified model class under-estimates the r_y DC gain, and the settle ratio follows
  DC_fit/M for FIR and ARX alike (record 50; Track 2 refit: FIR 0.48, ARX 0.68, AR1 0.74 of the probe
  on 23 pooled episodes, fold spread < 0.03). DC consistency, not state, is the lever.
- **R, state in the regressor — not for these faults.** Constant offset: only variance (both tracks).
  Torque faults at the adaptation timescale are fixed-direction: Track 1's converged-estimate
  instrument gives within-episode variation 5–19 % of the mean and early/late direction cosine 0.97–0.98
  (joints 0–6); Track 2's same-state probes give cosine 0.97–1.00 at the 5-step horizon, including
  joint 5. Joint 5's direction rotates only at the 1-step horizon (cosine 0.07–0.92, norm 0.03 against
  0.7 at 5 steps) — a fast transient, not something a slower estimator must represent. Two
  constant-basis methods with six-channel authority repair spatial joint 5 (descriptor 19/20, weighted
  allocator 17/20), so the legacy FIR's joint-5 harm (12 → 3) is better attributed to its
  translation-only mask. Where a state basis could pay off: friction-like disturbances with a
  DC-consistent model, and cross-task transfer (the converged estimate varies 13–63 % *between* tasks).
- **C, the tracking term — the only mechanism that can beat the input-only law on a constant fault,
  with magnitude unknown on the robot.** It needs a **pose** reference; an increment reference is
  redundant (the manuscript says so; both tracks agree). Its value depends on the estimate's bias and
  on how much offset the policy already absorbs: Track 1 finds it redundant with exact models and large
  under the measured gain error; Track 2, with a chunked visual policy, finds a 17–30 % transient trim
  and possible late harm when unanchored. Both find it requires a model accurate enough that the
  reference does not drift. Track 1 shows why: a reference built from the adapter's miscalibrated model
  makes the robot emulate that model (≈2.7x slower on r_y). So **tracking and the DC constraint are
  complements** — DC consistency makes the reference equal healthy behaviour, tracking supplies robust
  cancellation.

## Findings about our own descriptor results (Track 2, reproduced by Track 1)

1. **The composite tracking term was numerically inert in every Panda descriptor run.** Prediction gain
   45 → 1.4, tracking gain 4e-6 → 1e-7 over 300 steps; ratio 4e6–1.5e7 (spatial j6), 8.4e6 (object j3).
   Cause: the innovation covariance is 1e-6…7e-5, so R⁻¹ dominates, while `tracking_scale = 0.01`
   scales the metric to trace 0.12. The sweep is therefore a **prediction-only** ARX2 result with one
   input map used twice. Reproduce: `python3 docs/independent_adaptation/c1_simulation/inert_tracking_gain.py`.
2. **Object joint 3 depends on the correction cap:** 7 → 12/20 at cap 0.15 (8 fixed, 3 broken, p = 0.23);
   7 → 18/20 at cap 0.25 (11 fixed, 0 broken, p = 0.001). The 0.25 cap was chosen after Object
   development outcomes; the spatial sweep then used it as a registered constant.
3. Track 2 explains why the measured r_y settle (0.32–0.45) sits near ρ = DC_fit/M rather than
   1/(2 − ρ) = 0.61: with a policy closing the loop, the fixed point moves to ρ (simulation: 0.36 at any
   policy gain > 0, 0.59 at zero policy gain). Simulation-level explanation, not a robot measurement.

## Decisive robot experiment (converged)

Pinned policy RNG, one shared frozen control per scenario key, healthy controls for every new arm.
Cells: uniform command offset 0.05 on libero_10 (non-ceiling) and spatial joint-5 torque.
Arms: (A) the shipped law; (C) DC-constrained input-only law; (D) C + pose tracking with a reference
from the DC-constrained model, chunk-anchored; (E) C + a velocity/command-direction basis.
Predictions: D ≥ C on libero_10; E ≈ C on both cells; healthy harm ≤ 2/40 per arm. Log completion time
and per-channel motion ratio against healthy for every arm — success alone cannot separate "repaired"
from "emulating a miscalibrated model". ≈ 480 rollouts. Code: a pose-reference tracking term and arm
selector in `openpi/adaptive_law.py`, recorder fields in `openpi/re4_record.py`.

## Paper improvements, converged and prioritized (abstract 2026-09-18, paper 2026-09-25)

1. Put the DC-constrained pilot (r_y 41 → 82 %, libero_10 → 18/40) in the manuscript as the causal
   identification test, gated on its healthy control.
2. Run the experiment above; if it cannot run in time, demote the abstract's joint-Lyapunov sentence
   to a constructed example — there is no active-tracking robot evidence.
3. Narrow the small-gain corollary: on the Panda σ = 1 in pose, so it applies to the position-servo
   case (GR1's servos certify first-order, poles 0.72–0.77) — or instantiate it there.
4. Add the FIR/ARX invertible-filter proposition and the settle-regime explanation: they explain
   record 50 and preempt "why not a state model?".
5. Six-channel oracle on libero_10 (40 rollouts): does the rotation mask or the estimator cap the suite
   (rotation oracle 22/40 against healthy 38/40)?
6. Pinned, paired FIR / ARX / DC comparison on libero_10 — the current claims rest on unpinned counts.
7. E1 as planned; a development-versus-confirmation selection ledger; drop the "delay" motivation for
   K = 6 (pinned null at n = 80).
8. Relabel the descriptor appendix paragraph: prediction-only (tracking inert), cap 0.25 chosen after
   Object, 12/20 at cap 0.15.

Scripts: `docs/independent_adaptation/c1_simulation/` (simulation, fair-tuning evaluation, misspecified
evaluation, completion time, premise test, inert-gain reconstruction). Telemetry used by the premise
test: `results/sweep/telemetry/` (copied from the compute host's `/tmp`, gzip-verified).
