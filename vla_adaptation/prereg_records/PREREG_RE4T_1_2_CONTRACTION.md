# Preregistration: Panda tracking metric (Part 2.2) and sampled contraction / input sensitivity (Part 1) — re4 theory plan, 2026-09-13

**Written before any replay.** Everything here is simulator-only: recorded command sequences are
replayed through the LIBERO Panda (OSC_POSE, 20 Hz) from saved and perturbed simulator states;
no policy server is involved, so no result depends on policy sampling.

## Part 2.2 — Panda operational-space tracking metric, from the healthy logs

Per end-effector axis i (x, y, z, rx, ry, rz), first-order tracking on the measured increment
d_i[k] (normalised units, the FIR's y) driven by the commanded increment u_i[k]:
d_i[k+1] = λ_i d_i[k] + g_i u_i[k], least squares on the shipped and held-out healthy logs
(`results/phase05/error_signal_so3.json`, `results/heldout/error_signal_init25.json`,
`results/re4_evidence/H_third_init/error_signal_init5.json`), leave-one-episode-out. λ_i is the
sampled pole; the candidate metric is P = diag(1/(1 − λ_i²)) (the discrete Lyapunov solution of
the diagonal model with Q = I), restricted to the translation block for the contraction test.
Prediction: λ_i ∈ (0, 0.9) on x, y, z on every fold with held-out R² ≥ 0.3 on increments. A pole
outside (0, 1) or R² < 0.3 puts that axis outside the domain.

## Part 1 — Sampled contraction rate λ̂ and input sensitivity L̂, same-command replay

**Driver** (`openpi/re4_theory/paired_rollout.py`): reset the LIBERO scenario with
`libero_reset.reset_libero`, replay a recorded raw command sequence (the 70-step record in
`error_signal_init25.json`, task 0, init 25; and the first 70 steps of two other healthy
records) through `env.step`. At save steps k* ∈ {10, 25, 40, 55} snapshot `sim.get_state()`
(time, qpos, qvel). Baseline: continue the same commands for H = 15 steps. Perturbed: restore
the snapshot, add δ to the seven arm joint positions (magnitudes 0.005, 0.01, 0.02 rad; 6 random
unit directions per magnitude, fixed seed 0) or to the six end-effector command channels
(δu = ±0.02 on one axis, applied at the first replayed step only), `sim.forward()`, replay the
same commands, record the per-step gap between the two runs.

**Gap coordinates and candidate metrics (the complete list; no others will be tried).**
(a) Euclidean on the seven arm joint positions; (b) Euclidean on arm joint positions and
velocities, velocities scaled by Δt; (c) Euclidean on end-effector position (m); (d) the Part 2.2
metric P on the end-effector position gap. λ̂_k = ‖gap_{k+1}‖/‖gap_k‖ per step; L̂ = ‖gap‖ per
unit ‖δu‖ at the step after the command perturbation. Steps are classified as free-space or
contact by the simulator's contact list (any contact involving a robot geom other than the
gripper fingers with the table, or any contact with an object).

**Predictions.** (1) In free space, the median λ̂_k over steps and perturbations is < 1 in at
least one of the candidate metrics, with the 90th percentile < 1.05, for all three magnitudes.
(2) At contact steps the fraction of λ̂_k ≥ 1 is higher than in free space (the paper's hybrid
caveat). (3) L̂ on translation is within a factor of two of the FIR's first-tap gain
(0.25–0.30 normalised units per unit command, i.e. ≈ 1.3–1.5 cm per unit at 5 cm per unit).

**Refutation.** If no candidate metric gives median λ̂ < 1 with 90th percentile < 1.05 over a
free-space domain of at least 20 steps, the paper's servo example stays constructed-only and the
"unverified" sentence stays; no further metric is tried.

**Artifacts.** `results/re4_theory/1_contraction/` (per-perturbation gap traces, summary JSON),
`results/re4_theory/2_metric/metric_certificate_panda.json`.

---

## Outcome, Part 2.2 (appended 2026-09-13, after the fit; nothing above was edited)

23 healthy episodes pooled from the three logs (initial states 45, 25, 5), leave-one-out.

| axis | λ | fold range | held-out R² min / median | in domain |
|---|---|---|---|---|
| x | 0.707 | 0.687–0.715 | 0.961 / 0.992 | yes |
| y | 0.666 | 0.633–0.673 | 0.971 / 0.991 | yes |
| z | 0.818 | 0.792–0.833 | 0.965 / 0.984 | yes |
| rx | 0.829 | 0.816–0.836 | 0.847 / 0.946 | yes |
| ry | 0.928 | 0.923–0.931 | 0.839 / 0.938 | yes |
| rz | 0.609 | 0.584–0.632 | 0.843 / 0.964 | yes |

**Prediction confirmed** (translation poles in (0, 0.9), R² ≥ 0.3): all six axes enter the domain,
with the pole stable to ±0.02 across folds and initial states. Candidate metric
P = diag(1/(1−λ²)) = diag(2.00, 1.80, 3.02, 3.20, 7.20, 1.59). The slowest channel is ry
(λ = 0.93, a 14-step time constant at 20 Hz), the same channel whose fault estimate lags on
every backbone: a slow tracking pole is one candidate cause of that lag, testable in Part 6.
Artifact: `results/re4_theory/2_metric/metric_certificate_panda.json`.

---

## Outcome, Part 1 (appended 2026-09-13, after the replays; nothing above was edited)

Four recorded episodes (tasks 1–4, initial state 25), four save steps each, 31 replays per save
step (1 determinism check, 18 joint perturbations, 12 command perturbations), H = 15. Restored
states replay bit-identically (max gap 8.7e-15 over 16 checks) once the snapshot includes the
solver warm start, actuator state and the gripper's accumulated target; without the gripper
target, grasp-phase replays diverged by 0.07 rad. `results/re4_theory/1_contraction/`.

| metric | regime | per-step ratio λ̂: median / 90th pct / fraction ≥ 1 (joint perturbations, all magnitudes) |
|---|---|---|
| joint positions | free / contact | 0.999 / 1.019 / 0.47 — 1.001 / 1.05 / 0.54 |
| joint pos + vel | free / contact | 0.999 / 1.019 / 0.47 — 1.001 / 1.05 / 0.53 |
| end-effector position | free / contact | 1.000 / 1.005 / 0.50 — 0.998 / 1.007 / 0.32 |
| end-effector, Part 2.2 metric P | free / contact | 1.000 / 1.005 / 0.49 — 0.999 / 1.008 / 0.33 |

- **Prediction 1 (median < 1, 90th pct < 1.05 in free space at all magnitudes): met by the
  letter in three metrics (0.999 < 1; 1.019 < 1.05) and refuted in substance.** λ̂ is 1.00 to
  three decimals with half the steps above 1: a joint-position perturbation neither decays nor
  grows over 15 steps. The mechanism is the controller: OSC_POSE sets each goal as the current
  end-effector pose plus the commanded delta, so a displaced arm simply executes the same
  deltas from the displaced pose and the gap persists. This plant is marginally stable in the
  same-command sense, not contracting; the execution tube cannot be geometrically bounded by a
  contraction rate here and must be bounded by the input sensitivity and the horizon instead.
  The paper's servo example stays constructed-only for the Panda, and its limitation sentence
  stays, with this measured value as the reason.
- **Prediction 2 (more expansion at contact): confirmed in the joint metrics** (fraction ≥ 1
  0.54 vs 0.47; 90th percentile 1.05–1.06 vs 1.02) **and refuted in the end-effector metrics**
  (0.32 vs 0.50): contact with the table or an object constrains the gripper and damps
  end-effector gaps while joint-space gaps grow. The hybrid caveat is about the joint state.
- **Prediction 3 (L̂ within 2× of the FIR first-tap gain): refuted.** Measured first-step
  sensitivity 0.51, 0.55, 0.60 cm per unit command on x, y, z (0.42–1.3 cm per unit on the
  rotation channels' induced translation) against 1.3–1.5 cm expected from the FIR's first tap
  at 5 cm per unit: a factor 2.3–2.9 lower. The FIR's first tap is fitted on the policy's own
  smooth commands; an isolated one-step command perturbation moves the arm less.
- Command perturbations do not contract either: after the perturbed step the gap ratio's median
  is 1.03 in joint space (90th pct 1.29), i.e. a one-step command error keeps propagating.
