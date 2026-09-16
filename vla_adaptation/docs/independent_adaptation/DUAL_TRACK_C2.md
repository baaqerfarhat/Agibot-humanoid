# Dual track C2: pose-referenced tracking as a source of contribution

2026-09-15. P0 of `PRIORITY_QUEUE.md`. Track 1: Claude Opus 5. Track 2: a separate Claude Fable 5.1
agent standing in for Codex, unseeded. Implementation (IMP1, `b01492a`) by a Claude implementer,
verified by me and by an independent Claude verifier. Evidence tiers: A robot ground truth or exact
identity, B stored-data inference, C calibrated surrogate or simulation, D theory.

## Converged findings

1. **The reference must be probe-consistent (A/B, both tracks, two suites).** On the adaptive arms,
   after settling, a reference built from the fitted plant under-reads the remaining r_y fault by
   about half: Track 1 reads 0.47 (spatial) and 0.34 (libero_10) of the true remaining fault; Track 2
   measures pose-excess slopes of 0.50–0.60 of M(f − f̂) across all four spatial telemetry cells. A
   probe-consistent reference reads 1.06 on spatial. Track 2's theory: consistency with the probe,
   not correctness of the true gain, removes the bias.
2. **Translation cannot be addressed this way (A/B, both).** The pose channel on x, y, z is biased
   0.7–2.7x and its per-chunk SNR is 0.2–0.4, consistent with the record's superposition/contact
   finding.
3. **The frozen policy closes some channels and not others (B, Track 2, three handles).** Command
   deviations under the fault show the VLA compensating r_y (φ = 0.45–0.79) and translation, but not
   r_x or r_z (φ ≈ 0–0.1). The adaptive arm therefore keeps an r_x offset of 0.10–0.34 normalised
   units (3–10°). Pose feedback has a job only on channels the policy leaves open.
   Second instrument (Track 1, pinned libero_10 n = 80: frozen-minus-adaptive mean command ≈ −φ·f̂):
   r_x φ = −0.08 and r_z φ = −0.35 agree that those channels are not compensated; r_y gives −0.22,
   which neither confirms nor refutes Track 2's +0.45–0.79 on spatial, because the instrument is
   confounded by trajectory differences of the same size (0.015–0.03 on the translation channels,
   which neither arm corrects). The r_x/r_z part rests on two instruments, the r_y part on one; the
   design below does not track r_y either way.
4. **Task success cannot show it (B/C, both).** The rotation-only oracle caps libero_10 at 22/40 and
   the DC-constrained law already reaches 18/40, so tracking can add at most about 4/40 (Track 2's
   surrogate: central +1). A paired McNemar test has 11 % power at n = 40 and 66 % at n = 200 even at
   +4/40 (Track 1). Per-episode pose error does not predict success within an arm (fixed-horizon AUC
   0.41–0.57; earlier "significant" AUCs were an episode-length artefact). Spatial is at ceiling.
5. **Anchored tracking with the estimator's own plant is a gain change, not a pose signal (A,
   verifier, to 1e-16).** With `--track-ref dc` on `--dc-constrain`, z_T equals the window mean of
   M_inv (r + W*c): an unattenuated, deadzone-free integrator of the runner's residual (loop gain
   0.08 → ≈0.13). With `--track-anchor 0`, e_p/n integrates the episode-mean rate, not the retained
   offset.
6. **The C1 divergence was a comparator mismatch (C, Track 2 re-ran Track 1's simulation).** Both
   simulations find tracking decisive against a DC-biased input-only law and small against a
   DC-correct one.

## Physical pre-test (A, open-loop replay on the Panda simulator, 10 held-out episodes)

Late rotation deviation (r_x, r_y, r_z) from the exact fault-free replay; the DC-constrained legacy law
leaves 0.280, 0.253, 0.248. Chunk-anchored tracking: −11/−4/−10 % (κ .01), −29/−11/−25 % (κ .03),
−60/−17/−54 % (κ .1, healthy x-drift max 0.18 → 0.32). Rate-mode unanchored on r_x, r_z: −16/0/−19 %
at κ .01, no healthy cost. By finding 5 these arms change the estimator's gain and remove the legacy
law's attenuation; they do not isolate pose feedback. Open loop also shows r_y settling at 7 %
(shipped) and 21 % (DC) against 33–41 % and 82 % in closed loop, and the fitted and DC references
coincide — replay commands contain no policy compensation, so emulation bias cannot appear there.

## The isolating test (A, IMP2, same replay protocol)

Base C = innovation law + DC constraint (no attenuation bias to remove); position mode feeds back the
retained pose offset itself on r_x and r_z only, reference leak 0.02. Late deviation from the
fault-free replay, against C's 0.186 / 0.180 / 0.176 (r_x / r_y / r_z), with the healthy arm's drift
in brackets (HC: r_x 0.047, r_z 0.034):

| κ | r_x | r_y | r_z | healthy drift | settle r_x / r_y / r_z |
|---|---|---|---|---|---|
| 0.0005 | −15 % | 0 | −23 % | +0.003 | 1.26 / 0.31 / 0.97 |
| 0.001 | −22 % | 0 | −41 % | +0.006 | 1.29 / 0.29 / 1.00 |
| 0.002 | −31 % | 0 | −60 % | +0.010 | 1.32 / 0.27 / 1.02 |
| 0.001, leak 0.05 | −15 % | 0 | −24 % | +0.005 | 1.25 / 0.31 / 0.96 |
| 0.002, no leak | −44 % | 0 | −66 % | — | 1.41, unstable (below) |

r_y is untracked and does not move, as it should. The base carries no attenuation bias, and the term
feeds back the offset rather than a rate. **r_x is mostly, not wholly, attributable.** Through
M_inv's off-diagonals, M_inv[r_x, xyz] = [0.030, −0.002, 0.072], the uncorrected translation pose
error also drives the r_x update. Share of the integrated r_x forcing (κ Σ_k M_inv[r_x, xyz]·e_xyz(k)
over κ Σ_k z_rx(k), pooled over the 10 episodes; reconciled by the IMP3 verifier from the committed
telemetry): **9.9 % at κ = 0.001** (per episode −0.3 to +18 %) and **12.8 % at κ = 0.002**, with leak
0.02; 15 % without the leak; 23–40 % at κ = 0.005–0.01 (runs not committed). Late-episode per-step
share 22–24 %. On r_z it opposes the reduction (−2.4 % and −3.1 %). So **r_z's −41 % is pose feedback,
and about 90 % of r_x's −22 % is**. (An earlier statement here, "15–40 % at κ = 0.002–0.01", mislabelled
its lower end: that 15 % came from the leak-free arm.) The healthy arm's larger share, 25 %, rests on a
denominator ten times smaller and should not be quoted.
Restricting the observation to the tracked subspace (masking e_p, or inverting M[R,R]) removes the
leak exactly, at a DC estimate bias of −1.9 % (r_x) and +0.8 % (r_z) of the fault; in the verifier's
closed-loop check it changes the estimate by ≤ 0.001. Q2 should use the restricted observation.

**The trade this exposes.** As the pose error falls, the fault *estimate* gets worse: r_x settles at
1.22 of the true fault under C and 1.32 under κ = 0.002. That is coherent — pose feedback cancels
whatever causes the pose error, including the uncorrected translation fault coupling in, not only the
r_x fault. Estimation accuracy and execution fidelity are different objectives here, and the trade is
measurable.

**Stability.** Without a leak the term integrates the uncorrected translation pose error through
M_inv's off-diagonals without bound (translation e_p 1.608 and growing versus 1.125 held by the leak),
and a surrogate limit-cycles at κ = 0.005 with leak 0.02: inside its deadzone the innovation law holds
the estimate, so the leak is the position loop's only damping. On the robot data there is no
oscillation — the gate is open 95 % of the time and the position arms show only κ-proportional noise
gain (detrended sd 0.0047 → 0.0075 at κ .005 → 0.0179 at κ .02). With leak 0.02 the translation error
is bounded at |e_T| ≈ 0.74 ≈ d_T/leak. Open-loop replay is the worst case for all of this, since no
policy absorbs the translation drift.

**Estimates are non-stationary; report windows, not end values.** A single episode's end-of-run r_y
estimate of 0.085 (against 0.05) was a snapshot of a non-stationary FIR model error: over 10 episodes
the base ends at 0.046 ± 0.031, and 67 of 70 arm-episodes end about 0.02 above their own last-30-step
mean. A translation-only fault reads −0.021 on r_y in replay because the uncompensated translation
drift carries the arm 60–80 mm off the recorded path, where the FIR fails; the policy prevents this in
closed loop. Every number in this document uses late-window means.

## Corrections

- Withdrawn: "the robot becomes ≈2.7x slower on r_y" under a model-gain reference. It was an
  inference from the MRAC mechanism, not a simulated or measured result. The measured statement is
  finding 1.
- My pre-test's reductions were first read as a pose-tracking effect; finding 5 shows they are not.

## What this means for the paper

The contribution available is narrow but real, and the isolating test above supports it: estimate-only
adaptation leaves a retained pose offset on exactly the channels the frozen policy does not close
itself; a probe-consistent, leaky pose term removes it; anchoring it to policy chunks turns it into a
gain change; and a fitted reference biases it. This turns the manuscript's constructed
joint-Lyapunov example into a measured, channel-resolved mechanism, explains record 50 from a second
channel (total fault from the sent command, remaining fault from the nominal command), and states a
design rule. It does not move task success measurably on these suites and does not reach translation.

## Queue item Q2, redesigned

Base C = `--law innov --dc-constrain corrected`; D = C + position-mode tracking on dims 3,5 with leak
0.02 and κ = 0.001 (from the isolating test), using the observation restricted to the tracked subspace
(a small opt-in addition, not yet implemented). Primary endpoint: late r_x/r_z pose offset against the healthy
control (Track 2: resolvable at ≥3σ with 60 keys). Secondary: success, reported with its power. Cells
and budget per Track 2's spec: libero_10 uniform 0.05 (60 keys, E2 core manifest) and spatial joint-5
torque (40 keys), healthy controls, pinned sampler, ≈480 rollouts. Register predictions first.
