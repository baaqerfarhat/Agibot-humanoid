# Priority queue — recorded 2026-09-14 at the user's request

Deadlines: ICLR 2027 abstract **2026-09-18**, full paper **2026-09-25** (AoE).
Evidence base for every item: `docs/independent_adaptation/DUAL_TRACK_C1.md`.

## P0 — current priority: comprehensive investigation of pose tracking

The tracking term is the only mechanism found that can beat the input-only law on a constant fault.
It must track **pose**, not motion increments, and its reference must come from a correctly calibrated
model; with the current fitted model the robot copies the model's error (≈2.7x slower on r_y), so it
complements the DC-gain fix rather than replacing it. Its size on the robot is unknown: Track 1's
simulation shows a large effect, Track 2's a 17–30 % transient trim. Investigate it as a possible
source of innovation and contribution. Report: `docs/independent_adaptation/DUAL_TRACK_C2.md`.

## Queued (in the user's order)

Drafts, preregistrations and the corrections list for these items are indexed in
[manuscript_proposals/README.md](manuscript_proposals/README.md).

| # | item | cost | depends on | status |
|---|---|---|---|---|
| Q1 | Put the DC-constrained pilot in the manuscript as the causal test of the gain bias (r_y estimate 41 → 82 %, libero_10 → 18/40). Only its 20-rollout healthy control is missing. | 20 rollouts + writing | healthy control | text drafted; run requested from Mahdi |
| Q2 | Decisive experiment: pose tracking on top of the DC-constrained model, on libero_10 and spatial joint 5, ≈480 rollouts. Log completion time and per-channel motion, not just success. If it cannot run in time, soften the abstract's joint-Lyapunov sentence, since no robot evidence supports it. **Redesigned by P0 (see DUAL_TRACK_C2.md):** base = innovation law + DC constraint; tracking = position mode with a reference leak, on r_x and r_z only; primary endpoint = late r_x/r_z pose offset against the healthy control; success secondary, with its power stated (11 % at n = 40 for the largest plausible effect). | ≈300 rollouts | code and prereg ready | run requested from Mahdi |
| Q3 | Narrow the small-gain corollary: on the Panda the pose does not contract (σ = 1). Restrict it to position servos, or instantiate it on GR1, whose servos certify (poles 0.72–0.77). | writing (+ GR1 constants) | — | proposal drafted |
| Q4 | Add the FIR/ARX equivalence result: it explains why the ARX attempt failed and pre-empts "why not a state model?". | writing only | — | proposal drafted |
| Q5 | Six-channel oracle on libero_10 (40 rollouts): does the rotation mask or the estimator cap that suite? | 80 rollouts | prereg ready | run requested from Mahdi |
| Q6 | Pinned, paired FIR/ARX/DC comparison (current claims rest on unpinned counts); relabel the descriptor paragraph in the empirical appendix; add a log of which settings were chosen on development data. | 160 rollouts + writing | prereg ready | writing drafted; runs requested from Mahdi |

## Runs requested from Mahdi (2026-09-15)

We do not have compute for the GPU items (Q1, Q2, Q5, Q6), so they are requested from Mahdi, if he is
able to run them. Each has a preregistration with the exact arms, flags, keys, predictions and
refutation conditions, written before any data:
`prereg_records/PREREG_DC_CONSTRAINED_PLANT.md` (Q1, prediction 3), `PREREG_Q2_POSE_TRACKING.md`,
`PREREG_Q5_SIXCHANNEL_ORACLE.md`, `PREREG_Q6_PINNED_FIR_ARX_DC.md`. The code they need is committed and
off by default. Q3, Q4 and Q6's writing parts are drafted and need no compute.

## Coordination constraints

- `iclr2027/` is the collaborator's manuscript. Items Q1, Q3, Q4 and Q6's relabel change it: prepare
  them as proposed text, and apply only with the collaborator's agreement.
- `openpi/adaptive_law.py` is the runner the collaborator uses for E2. Any change must be opt-in
  (default off), additive, and must leave every existing flag's behaviour byte-identical.
- GPU items (Q1, Q2, Q5, Q6) use `--pin-rng` and one shared frozen control per scenario key; every new
  arm gets a healthy control; predictions are registered in `prereg_records/` before any run.
