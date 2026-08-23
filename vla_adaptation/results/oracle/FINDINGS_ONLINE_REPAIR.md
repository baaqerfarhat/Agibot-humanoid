# A frozen VLA is repairable online by a 6-dimensional edit on one layer

Run 2026-08-22. Frozen π0.5-LIBERO (3.35B, no gradients, no fine-tuning), `libero_spatial`,
fault `offset @ 0.05` on arm dims 0–5 — the only cell to survive the Phase 0.4 gate.
15 episodes per condition, initial states 8–9, disjoint from the gate, screen and baseline.
Raw: `oracle_sweep.json`.

| condition | success | vs floor |
|---|---|---|
| `k = −1.0` wrong sign, faulted | **0.0%** (0/15) | −46.7 |
| `k = 0.0` no edit, faulted (floor) | 46.7% (7/15) | — |
| `k = 0.5`, faulted | 86.7% (13/15) | +40.0 |
| **`k = 1.0` computed oracle, faulted** | **100.0%** (15/15) | **+53.3** |
| `k = 1.5`, faulted | 20.0% (3/15) | −26.7 |
| `k = 1.0`, **no fault** | 6.7% (1/15) | — |

**The repair is complete.** 46.7% → 100%, against a nominal of 99.0% and ~52 points of
headroom: the entire deficit is recovered by adding a fixed 6-vector to
`action_out_proj/bias`. Nothing else is touched, no gradient is taken, and the model is
never fine-tuned.

**It is specific, not a general improvement.** The same edit applied to the *healthy*
policy drops it to 6.7%, and the sign-flipped edit annihilates the faulted one (0%). This
is a compensating inverse, and it only helps against the fault it inverts.

**The basin is narrow.** k = 1.5 scores 20% — *worse than not repairing at all*. Overshoot
by half is more damaging than the original fault. Any search over this class must be scaled
to a useful range of roughly k ∈ [0.4, 1.2]; a search that wanders past ~1.4 spends its
budget in territory worse than its own starting point.

## Why the edit had to be computed per-dimension, and why that is the transferable part

The obvious version of this — "cancel a +0.05 offset with a −0.05-ish edit" — is wrong, and
would have read as evidence that the class is not reachable.

π0.5 emits **normalised** actions; `Unnormalize` runs afterwards. With quantile norm,
`env = (norm+1)/2·(q99−q01) + q01`, so each dim has its own scale `(q99−q01)/2`, and those
scales differ by 7× across the arm channels. The consequence is that a *uniform* env-space
fault is wildly **anisotropic** in the units the model actually works in:

| dim | scale | fault in norm units | % of action range | β needed |
|---|---|---|---|---|
| dx | 0.842 | 0.059 | 3.0% | 0.175 |
| dy | 0.828 | 0.060 | 3.0% | 0.178 |
| dz | 0.937 | 0.053 | 2.7% | 0.157 |
| **drx** | **0.128** | **0.391** | **19.5%** | **1.149** |
| dry | 0.175 | 0.285 | 14.3% | 0.839 |
| drz | 0.253 | 0.198 | 9.9% | 0.581 |

A "+0.05 offset" is 3% of the action range on translation and **19.5% on drx** — 6.6×
larger. That is why this cell survived the gate while brightness did nothing, and why
`offset @ 0.10` was lethal.

So the repair vector is
`β_i = (0.05 / scale_i) / 0.34`, where 0.34 is the trained flow's measured attenuation of a
bias edit (§0 of the ACE prereg). It spans **0.157 … 1.149**, a 7.3× spread, and is 29–193×
the RMS of the bias vector it is added to (‖W‖_F = 0.0336 over 32 dims).

**k = 1.0 being exactly right is a two-way validation.** The scale came entirely from first
principles — quantile norm stats plus an open-loop attenuation measured on a synthetic
observation — and it lands on 15/15 in closed loop. Both the normalisation mapping and the
0.34 attenuation therefore hold in closed loop, where §0 could only check the latter
open-loop.

## What this means for the search

**Parameterise the edit in env-action units, not raw parameter units.** In env units the
target is isotropic (−0.05 on all six dims) and an isotropic CEM with σ₀ ≈ 0.02 is well
conditioned. In raw bias units the same target spans 7.3×, so an isotropic search is
ill-conditioned in exactly the dims that carry most of the fault — it would under-explore
`drx` (which needs 1.149) while over-exploring `dz` (which needs 0.157).

This is the concrete, transferable answer to "how does online adaptation of a frozen VLA
actually help": the class is reachable and the ceiling is 100%, but only if the edit is
parameterised in the space the task is measured in. The normalisation layer between the
model's output and the environment is not a detail — it is what determines whether a
low-dimensional search is well posed.

## Caveats, stated plainly

- **This is the easy geometry.** The fault is a constant action offset and
  `action_out_proj/bias` produces a constant action offset: fault class and edit class
  coincide exactly. The gate never supplied the state-dependent multiplicative cell that
  would test the hard case.
- **The oracle is not a search.** It establishes the ceiling with a computed edit. Whether a
  gradient-free search finds it from scratch, and how much of the 53 points it recovers, is
  Phase 0.5 and is not answered here.
- 15 episodes per condition. The extremes (0%, 6.7%, 100%) are unambiguous at that n; the
  middle of the curve is not resolved finely.
