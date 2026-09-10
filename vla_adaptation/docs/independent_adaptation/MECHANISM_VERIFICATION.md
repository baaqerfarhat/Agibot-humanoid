# The joint-fault harm mechanism, verified — and what the proposed method does about it

2026-09-08, Claude track. The second track is answering the same question independently. This
verifies `report/JOINT_FAULT_MECHANISM.md` against artifacts rather than restating it.

## 1. The mechanism, as identified on the main branch

The legacy observer computes a **full six-coordinate** equivalent fault `M⁻¹ r`, then **masks** it to
the permitted translation channels. Inverting-then-masking is not the same as solving for the best
correction over the permitted input columns. The report's two-input example:

```
M = [[1, 1], [0, 0.1]]      r = [0, 0.02]
full estimate  M⁻¹r = [-0.2, 0.2]
masked estimate      = [-0.2, 0]
residual after masked correction = [0.2, 0.02],  norm 0.201 against an original 0.02
```

The masked correction makes the residual **ten times worse**, and the best available correction is
**zero**. That is a correction actively doing harm when doing nothing was optimal — which is the
joint-5 phenomenon.

**This also invalidates a measurement I made earlier.** I computed the projection of the residual
onto `span(M[:, :3])` and found 90-97% "cancellable" at every joint, and concluded authority was not
the problem. That answered *is a good translation correction available* (yes) — but the estimator
never computes that projection. It inverts, then masks. The right question was whether the
estimator's actual operation is the good one, and it is not.

## 2. Verification of the confirmation, recomputed from `results/joint_followup/confirmation_plan/scores.json`

Pooled over the three LIBERO suites, n = 60 per arm:

| joint | comparison | left | right | rescues | regressions | exact p |
|---|---|---|---|---|---|---|
| 5 | **legacy vs off** | 36/60 | **15/60** | 4 | **25** | **1.04e-04** |
| 5 | candidate vs legacy | 15/60 | **41/60** | 29 | 3 | **2.56e-06** |
| 5 | candidate vs off | 36/60 | 41/60 | 14 | 9 | 0.40 |
| 3 | legacy vs off | 28/60 | 35/60 | 15 | 8 | 0.21 |
| 3 | candidate vs legacy | 35/60 | 32/60 | 14 | 17 | 0.72 |
| 3 | candidate vs off | 28/60 | 32/60 | 12 | 8 | 0.50 |

Every figure the report cites reproduces exactly, including `p = 2.556e-06`, the 29/3 rescue-
regression split, the 36/60 → 41/60 against off, and the LIBERO Object joint-3 catastrophe
(16/20 → 2/20, 1 rescue, 15 regressions, Bonferroni-24 `p = 0.01245`). **No claim was overstated.**

Two things the pooled view makes sharper than the paper's single n=20 cell:

- **The legacy method at joint 5 destroys 25 of 36 working episodes** and is dramatically worse than
  doing nothing at `p = 1.04e-04` over 60 paired episodes. That is a far stronger statement than the
  `12/20 → 3/20` cell I have been quoting.
- **At joint 3 nothing survives pooling.** The `11/20 → 18/20, p = 0.039` spatial cell that the
  paper's joint section rests on becomes `28/60 → 35/60, p = 0.21` across suites.

## 3. What the allocation fix actually bought

It converts a catastrophe into parity. Against legacy it is decisive (`p = 2.6e-06`); against
**doing nothing** it is `36/60 → 41/60` at `p = 0.40` — **not significantly better than off**. And
it creates a new catastrophe at LIBERO Object joint 3.

So allocation was a real defect and fixing it was necessary, and it is not sufficient.

## 4. Why it is insufficient, and what that implies

The report gives the reason: `r = disturbance + (G − F)·corrected_command`. Changing the estimate
changes the model-error contribution to its own next measurement, and the calibrated `M` is an
**averaged** fault-to-residual map, not the correct local inverse for every configuration, command
history, contact state or disturbance type. Evidence: across 168 same-state probes the weighted full
correction improves the pooled scaled physical-error ratio to 0.905 while **worsening 92 individual
probes**, with one case where the model objective falls to 10.5% of zero while physical error rises
**2.69-fold**.

That is a fitted objective improving while the physical quantity worsens — the signature of an
averaged map used as a local inverse.

## 5. Does the proposed method address it?

Two defects, and the method has a component for each:

| defect | evidence it is real | method component |
|---|---|---|
| invert-then-mask allocation | fixing it: 15/60 → 41/60, `p = 2.6e-06` | `matched_action` solves a constrained least-squares over the available input map **and raises when the target is unachievable**, instead of masking an inverse |
| `M` is an averaged map, not a local inverse | 92/168 probes worsened; objective 10.5% while error 2.69× | `Φ_d(ξ) z_d` makes the disturbance representation **state dependent**, so the map varies with configuration, history and contact rather than being one fitted constant |

The second is the binding constraint, and it is the one the existing fix does not touch. That is the
strongest evidence-grounded case for the successor direction produced so far: **not "a state-
dependent basis is one plausible fix among several", but "the allocation half was fixed, measured,
and found necessary-but-insufficient, and the stated reason for the insufficiency is precisely what
a state-dependent basis replaces."**

## 6. What this does not settle

- **The LIBERO Object joint-3 catastrophe is a warning the method may inherit.** A richer correction
  with more input authority went 16/20 → 2/20 there. A learned basis is also a richer correction. No
  component of the proposed method obviously prevents that failure, and `matched_action` raising on
  infeasibility does not help when the target is feasible but wrong.
- **The `(G − F)` feedback term needs an adapted `G`.** `Φ(ξ)` gives a state-dependent *disturbance*
  model; the residual also contains model error scaled by the corrected command. `E(ξ, z_E)` adapts
  the input map, which is part of `G` but not all of it.
- Nothing is trained. Every claim above is about capability, not evidence.
