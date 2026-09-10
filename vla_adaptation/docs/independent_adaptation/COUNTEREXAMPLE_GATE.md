# The LIBERO Object joint-3 counterexample, verified before any attempt to reproduce it

2026-09-08, Claude track. The convergence named one scenario as the cheapest discriminator and set
a precondition: if the old harm cannot be reproduced under a controlled protocol, it is not yet a
valid counterexample test. This verifies the scenario family from stored per-episode records first.

## 1. The scenario is confirmed, and it is a family rather than a single case

From `results/joint_followup/confirmation_plan/compact/libero_object/joint3/`, pairing on
`(task, init)` over 20 shared scenarios:

- **task 2 / init 35 is confirmed** as the first sorted case where OFF and legacy both succeed and
  weighted-full fails.
- **There are six such scenarios**, not one: `(2,35) (2,36) (3,36) (4,35) (5,36) (9,36)`.

Six spread across five tasks and both initial states is a family, which makes a reproduction attempt
substantially more informative than the single case the convergence proposed.

## 2. The OFF control is genuinely shared

The two cell files each carry a `frozen_faulted` arm. They **agree on all 20 scenarios**, so OFF is
one consistent control rather than two independent runs, despite policy RNG being unpinned. Nothing
below rests on comparing two different OFF draws.

## 3. The asymmetry is systematic, not noise

| direction | count |
|---|---|
| weighted **destroys** what OFF and legacy both achieved | **6** |
| weighted **rescues** what OFF and legacy both missed | **1** |

Six to one, across five tasks. That is not sampling variation.

## 4. The detail that sharpens the whole counterexample

| arm | LIBERO Object joint 3 |
|---|---|
| OFF, no correction | **8/20** |
| legacy, translation-only, invert-then-mask | **16/20** |
| weighted full, six inputs | **2/20** |

**Legacy doubles success at this cell**, and the richer correction is **four times worse than doing
nothing**, not merely worse than legacy. The harm is not "the fix helped less here"; it is "adding
input authority through a map that is locally wrong destroyed a cell where the restricted,
theoretically-defective method worked well."

That is precisely the shape of the verified counterexample: a well-fitted correction plus an
inaccurate input map produces a confident wrong action, and more authority makes it worse. The
scalar check through `matched_action` showed a perfect disturbance estimate with a wrong `E`
doubling the physical error; here a richer correction quarters the task success relative to OFF.

## 5. What this does and does not support

**Supports:** that the failure mode identified in G4 is real at scale and systematic, and that
`E` accuracy — not disturbance-basis richness — is what gates whether added authority helps or
harms. It also means the successor method, which adds *both* a richer basis and more inputs, is
walking into the cell that already punished exactly that combination.

**Does not support:** that a wrong `E` is the *cause* of these six failures. The weighted candidate
changes several things at once — six inputs instead of three, different residual filtering and
regularisation, bounds inside the solve, and a weighted DOB that bypasses the legacy deadzone and
normaliser. The compact files omit trajectories, so the applied-correction magnitudes that would
discriminate these are not checkable locally; the 46 MB archived telemetry on the compute host has
them.

## 6. Status of the precondition

Not yet met. Reproduction under the controlled protocol has not been attempted, and the four-arm
ablation cannot run at all until a basis is trained. What is now established is that the target is
worth the trouble: a six-scenario family, a consistent shared control, and a 6:1 asymmetry, in a
cell where the restricted method works and the enriched one is worse than nothing.
