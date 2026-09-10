# Dual-track convergence — is this a contribution?

2026-09-08. The two tracks **diverged**, and on the substance the other track is right. This
records the resolution, including three corrections to the Claude track.

## 1. The resolution

Both tracks agree the user's underlying principle is sound: **established components cannot by
themselves defeat a contribution.** G1's framing — an inventory of standard components — was the
wrong test, and Codex concedes it directly: *"G1 should have foregrounded the missing scientific
result rather than the inventory of standard components."*

But the Claude track then over-conceded, moving from "the objection defeats my framing" to
"therefore it is a contribution." Codex supplies the distinction that actually does the work, and
it is symmetric rather than a special pleading:

> **Demonstrated knowledge versus an unevaluated method.** Strip the predecessor's experiments and
> leave a calibrated observer design plus synthetic correctness tests, and it would not earn the
> empirical contribution either. Give the successor comparably informative evidence about a
> previously unresolved problem, and it earns one without needing a new Riccati equation.

Under that test the standard was never inconsistent. The predecessor has 690 paired episodes across
three backbones and three embodiments; the successor has untrained features, no connection to a
robot runner, and algebraic tests only. They are at different stages, not held to different rules.

**So: this is a contribution-in-waiting whose deciding experiment has not been run.** Not "not a
contribution because the parts are standard," which was wrong, and not "a contribution already,"
which was my over-correction.

## 2. Correction: my motivating premise was false

The Claude track argued that a **constant** Cartesian correction measurably fails off the elbow, and
that a state-dependent basis is therefore the principled response. **The premise is false.** Verified
directly against `results/joint_map/cell_torque_*.json`:

```
static_corr: None      estimate_only: False      gamma: 0.08      corr_dims: 0,1,2
correction range within one episode: 0.232, 0.145, 0.300 per translation dim
```

The estimator was **continuously adapting** throughout. Those cells run an online-updated,
translation-only Cartesian correction against a constant joint torque — not a held constant.

**The corrected observation is different, and in one way stronger.** An estimator adapting at
`gamma = 0.08` still produced net harm at five of seven joints, so the failure is not slow adaptation
or a frozen estimate. But it no longer points at configuration dependence specifically.

## 3. The cause is not identified, and a competing one fits better

At least three candidates explain the joint-map harm, and the stored data does not discriminate:

1. a state-independent representation (fixed `M`, no configuration dependence) — my hypothesis;
2. a translation-only correction with no rotation compensation (`corr_dims 0,1,2`);
3. **projection saturation**, which I measured while checking this: the correction sits at the
   `clip = 0.30` bound on **7.9%** of steps at joint 5 against **1.3%** at joint 3 — six times more
   at the joint with significant harm than at the joint that repairs.

Candidate 3 tracks the outcome better than mine does and is cheap to test. My earlier directional
analysis (residual cosine 0.52–0.75, never near 1.0) shows within-episode variation is real
everywhere, but at n = 4 joints with an inversion it cannot rank them (Spearman 0.8, p ≈ 0.33), and
the hypothesis was formed after seeing the outcomes.

So the motivation for a state-dependent basis survives as **one plausible fix among several for a
measured failure**, not as the identified response to a diagnosed cause. That is a materially weaker
claim than the one I committed.

## 4. Correction: "the law earns nothing" overstates the observer study

I have repeated that five estimator families tie and the specific law earns nothing. The stored
evaluation, recomputed:

| arm | corrected | paired vs proposed |
|---|---|---|
| proposed | **19/20** | — |
| Kalman | 18/20 | P-only 2, other-only 1, p = 1.000 |
| DOB | 17/20 | P-only 3, other-only 1, p = 0.625 |
| RLS | 17/20 | P-only 3, other-only 1, p = 0.625 |
| calibrated integral | 15/20 | P-only 5, other-only 1, p = 0.219 |
| oracle | 20/20 | P-only 0, other-only 1, p = 1.000 |

The proposed law is nominally highest among the non-oracle arms and **every** head-to-head favours it
directionally. None resolves at n = 20. The correct statement is *no advantage was resolved*, not
*the law earns nothing* — the second asserts a null the study cannot support, which is the same
error, in the opposite direction, as claiming the advantage.

## 5. What both tracks now agree the successor needs

The deciding experiment is unchanged and both tracks name it: the state-dependent learned basis must
beat the existing correction **on the cells where the existing one does net harm** — joints 4, 5 and
6, paired, against a baseline already on disk. Before that, and cheaply:

- **Test candidate 3 first.** Re-run joint 5 with a larger `clip`. If saturation explains the harm,
  the expensive representational study is aimed at the wrong thing, and this costs one cell.
- Train a basis at all. Nothing is trained.

Gating the expensive study on the cheap refutation is the same discipline the predecessor's audit
line used, and candidate 3 is the cheapest thing that could kill the premise.
