# Is this a contribution? — Claude track

2026-09-08. Written in response to a consistency objection that I think is correct. The Codex track
is answering the same question independently and this will be converged against it.

## 1. The objection, and why it lands

The G1 convergence concluded that the composite law, wrench pullback, group averaging and covariance
transport are "occupied or standard," and that applying them to a manipulator or a frozen VLA "is
not by itself a contribution." The objection:

> If that is not a contribution, then the original main branch does not have one either.

**This is correct, and it catches an inconsistency in my own judgements within a single session.**
Earlier I argued — after being pushed on it — that the predecessor study *is* a genuine
contribution: 23% → 65% pooled, replicated across three VLA architectures and three embodiments,
*despite* its method being a textbook disturbance observer that this repository's own audit showed
five estimator families tie with. The specific update law earns nothing there; the setting and the
demonstration are the contribution.

Applying "the components are standard" as a defeater here, while crediting the predecessor under
the opposite standard, cannot both be right. The new direction has **strictly more** method content
than the predecessor: a state-dependent learned basis rather than a constant correction, an
effectiveness map, and correctness conditions that did not previously exist.

**The error was conflating "novel components" with "contribution."** Almost no paper has novel
components. Neural-Fly is meta-learning (standard) plus Kalman-Bucy (1961) plus composite adaptation
(Slotine 1989); its contribution is the combination, the demonstration and the analysis. The right
test is not "is any part unoccupied" but "does a reader learn something they did not know, and is it
load-bearing."

## 2. The claim, stated precisely

Not "we combine Neural-Fly with morphological symmetry." The claim worth defending:

> A frozen policy's execution faults can be repaired online by a **state-dependent** learned
> disturbance representation adapted at the action interface, where a **constant** correction
> measurably fails; and the symmetry of the robot supplies both a weight-sharing prior on that
> representation and a set of consistency conditions the estimator must satisfy for the prior to
> mean anything.

Two load-bearing halves. The first is empirical and rests on the predecessor's own measured failure.
The second is technical and is the part with new content.

## 3. The motivation is a measured failure, not a story — with limits

The predecessor's joint-level cells, recomputed on this branch, are the motivation:

| | constant Cartesian correction |
|---|---|
| elbow (joint 3) | 11/20 → 18/20, **+7** net |
| all six other joints | **−7** net, combined |
| joint 5 | 12/20 → **3/20**, zero fixed, nine broken, **p = 0.0039** |

A constant correction cannot, by construction, cancel a disturbance that varies within an episode.
That step is a tautology, not an empirical claim, and it is what makes a state-dependent basis the
principled response rather than a fashionable one.

**What the stored data does and does not support.** Measuring the residual's directional stability —
mean cosine between `r(t)` and its own episode-mean direction — gives 0.52 to 0.75 across the seven
joints, **never near 1.0**. So there is real within-episode variation that a constant correction
provably cannot capture, everywhere. That much is established.

What is *not* established is that directional wander **ranks** the joints. Among the four joints with
actual repair opportunity the elbow is most stable (0.722) and joint 5, the significant harm, least
(0.522), but with one inversion and n = 4 the Spearman is 0.8 at p ≈ 0.33. It cannot resolve. The
hypothesis was also formed after seeing the outcomes, so it is a motivated story and is labelled one.

The argument does not need the ranking. It needs only that a constant correction is provably
leaving something on the table, and that this coincides with measured net harm.

## 4. What is structurally different from Neural-Fly and MRAC

Not the law. **The reference.**

In MRAC the reference model is a *design choice*, selected for desired closed-loop behaviour. In
Neural-Fly there is a desired trajectory to track. Here there is neither. The policy is frozen and
black-box, and you cannot ask it what it intended. The reference is the **identified healthy plant's
response to the policy's own command stream** (`FORMULATION.md` Eq. 6).

That inverts the objective. MRAC moves the plant to match a model you chose. Here the model is not
chosen but *measured*, and it encodes the operating assumption the frozen policy was trained under.
Since the policy cannot be moved, the plant must be moved back to it. "Restore the policy's
operating assumptions" is a different control objective from "track a reference," and it is forced
by the frozen-policy setting rather than adopted for convenience.

**The honest bound**, which `FORMULATION.md` states and I would have overclaimed past: this
reference is driven by commands the policy actually issued *on the faulted trajectory*. It is **not**
the policy's counterfactual healthy rollout, and tracking it certifies nothing about the vision
loop or task success.

## 5. What is new in the technical half

After removing everything already in the repository (Eq. 21's transport table) or the literature
(Gada et al. 2210.13728 on consistent equivariant filtering):

- **The invariance condition on fixed hyperparameters.** Transport is covariance; every real
  implementation reuses one set of gains across the orbit, and that is symmetric only if `L, R, Q,
  Lambda` lie in the commutant. Both tracks derived this independently. Measured defect 83 versus
  1.1e-13.
- **The stabilizer rank-loss.** At a configuration fixed by a subgroup, an equivariantly-projected
  basis loses rank in exactly the directions that subgroup acts on, so no coefficient can encode a
  one-limb fault there. Rank 1 and residual 0.707 at symmetric contexts, rank 2 and 0.000 at generic
  ones. This is a genuine limitation of the approach and neither the review nor the source papers
  state it for this use.

Both are small. Both are the kind of thing a reader would otherwise get wrong.

## 6. What would have to be true

The claim in section 2 is currently unevidenced on its first half. Minimum to defend it:

1. **The state-dependent basis beats the constant correction on the cells where the constant one
   does net harm.** Joints 4, 5 and 6, paired, against the stored constant-correction baseline which
   already exists. This is the decisive experiment and its baseline is already on disk.
2. **The equivariance prior buys something measurable** — basis sample-efficiency at fixed data, or
   transfer to a limb held out of basis training. Otherwise it is a correctness condition only, and
   should be presented as one.
3. A trained basis at all. Nothing is trained yet.

Experiment 1 is the one that decides whether this is a contribution or a proposal, and it is
directly answerable against artifacts already in the repository.
