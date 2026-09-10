# The two cells want opposite corrections — a reframe of everything above

2026-09-08. Found by the second track, verified independently here from paired `(task, init)`
outcomes. It corrects the framing this track has used throughout the session.

## The fact

| cell | OFF | translation only | six inputs (weighted full) |
|---|---|---|---|
| **libero_spatial joint 5** | 14/20 | **6/20** *harmful* | **17/20** *rescued* |
| **libero_object joint 3** | 8/20 | **16/20** *helps* | **2/20** *destroyed* |

Paired discordances, recomputed here:

- Spatial joint 5, translation → full: **11 rescued, 0 broken, p = 0.00098**
- Object joint 3, translation → full: **1 rescued, 15 broken, p = 0.00052**

**The same intervention rescues one cell and destroys the other, both decisively.**

## What this corrects

I have been treating joint 5 as the cell the method cannot fix, and building on that: the state
dependence gate, the input-map analysis, the authority margin, the MIMO test. All of it took
"joint 5 is unfixable" as the premise. **It is not unfixable.** Adding rotation inputs takes it from
6/20 to 17/20 with zero regressions.

The historical cell I anchored on — `12/20 → 3/20` at inits 45–46 — is a *translation-only*
correction. The confirmation ran the six-input arm at inits 35–36 and it works there. I never
looked, because the historical cell had no six-input arm and I did not check whether a later study
had added one.

## The problem, correctly stated

Not "the correction fails at joint 5." It is:

> **The correct correction subspace differs by cell, choosing it wrongly is catastrophic in either
> direction, and nothing currently selects it.**

Translation-only is right for Object joint 3 and harmful for Spatial joint 5. Six inputs is right
for Spatial joint 5 and catastrophic for Object joint 3. Both errors are significant and large:
8 broken one way, 15 broken the other.

## Why this is a better problem than the one I was working

It is well posed, the stakes are measured in both directions, and it is exactly the shape both
tracks independently named as the strongest available contribution: **a prespecified measurement
that selects a scheme before running it.** Unlike the earlier attempts at that shape, here the
selection problem is not hypothetical — choosing wrong has been measured at p < 0.001 in both
directions on real cells.

## What does not solve it

The **authority margin** does not. On the correct calibration every subspace has margin below 0.1,
so it predicts all of them safe and cannot separate the cell that needs translation from the cell
that needs six inputs. That was measured and recorded before this reframe, and it stands.

The **coupled plant model** does not. It leaves joint 5 at zero fixed and 10 broken, and changes
Object joint 3 not at all.

A **state-dependent disturbance basis** is unlikely to, on the evidence: at joint 5 state features
have negative held-out R², and the explainable residual is input-map error.

## What is now the open question

What distinguishes a cell where rotation authority helps from one where it is catastrophic, and can
it be measured before running? Candidate quantities that have *not* been tested: how much of the
fault's effect lies outside the translation subspace at each cell; whether the rotation channels
carry identifiable signal there specifically; and whether the two cells differ in how much the
policy itself relies on orientation for the task.

None of those has been measured. Unlike everything else in this session, the problem is now stated
sharply enough that a wrong answer would be visible.
