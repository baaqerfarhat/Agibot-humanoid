# Pre-registration — is the joint-5 harm caused by projection saturation?

**Written 2026-09-08, BEFORE any cell runs.** This is a cheap refutation gate placed in front of an
expensive study, in the discipline the predecessor's audit line used: find the measurement that
could kill the premise, and run that first.

## 1. Why

The successor direction is motivated by the predecessor's joint-level failure: an online-adapting,
translation-only Cartesian correction repairs the elbow (+7 net) and does net harm across the other
six joints (−7 net), with joint 5 at 12/20 → 3/20, zero fixed and nine broken, p = 0.0039.

I originally attributed this to the correction's **representation** being state-independent, which
would make a state-dependent learned basis the principled fix. Checking that, I found a competing
explanation that tracks the outcome better: **the projection bound binds.** Measured on stored
trajectories, the correction sits at `clip = 0.30` on

- **7.9%** of steps at joint 5 (significant harm), against
- **1.3%** of steps at joint 3 (repairs),

with `max |correction|` reaching exactly 0.300 on the z channel in both. Six times more saturation
at the joint that is harmed than at the joint that is repaired.

If saturation is the mechanism, the expensive representational study is aimed at the wrong target.
One cell decides it.

## 2. Design

Identical to the stored cells in every respect except `--clip`. `libero_spatial`, π0.5,
`--joint-fault torque:J:5.0`, `--corr-dims 0,1,2`, `--dead 0.008`, `--norm-r 0.15`, `--gamma 0.08`,
n = 20 paired, `--eval-init 45`, published calibration.

| arm | joint | clip | role |
|---|---|---|---|
| **S5** | 5 | **0.60** | treatment |
| **S3** | 3 | **0.60** | validity gate — must not destroy the cell that works |

Baselines are already on disk and are **not** rerun: joint 5 at `clip 0.30` is 12/20 → 3/20; joint 3
at `clip 0.30` is 11/20 → 18/20.

## 3. Predictions, registered in advance

The three outcomes are distinguishable, which is what makes this worth running:

1. **Saturation is the mechanism.** Joint 5 harm falls materially — broken drops from 9 to ≤ 4.
   The representational story loses most of its motivation and the expensive study is redirected.
2. **The correction is wrong, not merely clipped.** Joint 5 gets **worse** — more headroom to apply
   a wrong correction. Corrected ≤ 3/20. This *supports* a representational cause: the direction
   being tracked is not the one that helps.
3. **Neither.** Joint 5 is unchanged within noise (corrected 2–6/20). Saturation is not the
   mechanism, and the cause remains unidentified.

**My prediction, stated so it can be wrong: outcome 2.** The estimator is adapting continuously and
still produced zero fixes in twelve opportunities, which reads more like tracking the wrong quantity
than like being unable to reach the right magnitude. I hold this weakly — outcome 1 is exactly what
the 7.9%-versus-1.3% asymmetry suggests, and that asymmetry is why the cell is being run.

**Validity gate.** If arm S3 does not roughly reproduce joint 3's repair (corrected ≥ 14/20), then
raising the clip has broken something unrelated and neither arm is interpretable.

## 4. Analysis, fixed in advance

- Paired exact McNemar against the stored `clip 0.30` frozen arm for the same joint, via both
  `openpi/mcnemar.py` and `openpi/mcnemar_crosscheck.py`.
- Regression scored against **opportunities**, not all episodes.
- Saturation fraction recomputed on the new trajectories and reported alongside, so the manipulation
  check is visible: if the correction still sits at the new bound, the clip was not raised enough
  and the arm is inconclusive rather than negative.
- No constant retuned, no arm dropped after seeing its result.

## 5. Stated limits

- One joint, one fault kind, one magnitude, n = 20 against ±10 points of documented free variation.
- This can identify saturation as *a* contributing cause; it cannot establish that the remaining
  harm is representational, because the three candidates are not exhaustive — translation-only
  correction is a fourth and is not varied here.
- The frozen baseline is reused from disk rather than rerun, so the comparison inherits whatever
  policy-sampling variation separates the two runs. `pin_rng` is False.

---

# RESULT — saturation is not the mechanism. My prediction was refuted; the second track's held.

**2026-09-08, treatment cell complete.** Artifact: `results/saturation/S5.json`.

| joint 5 | clip 0.30 (stored) | clip 0.60 (new) |
|---|---|---|
| frozen | 12/20 | **12/20** — baseline reproduced exactly |
| corrected | 3/20 | **4/20** |
| fixed / broken | 0 / 9 | 1 / 9 |
| exact McNemar | 0.0039 | 0.0215 |

## The manipulation check passed, which is what makes this interpretable

A null from an inert intervention would mean nothing. It was not inert:

- `max |correction|` on **z** went **0.300 → 0.4607**, well past the old cap;
- **0.00%** of steps pin at the new 0.60 bound, so the projection was genuinely relieved rather
  than merely moved;
- x and y are unchanged at 0.285 and 0.193, exactly as the second track's per-row reachability
  bound `[0.337, 0.289, 0.569, ...]` predicted — only z could ever exceed 0.30.

The estimator had all the authority it wanted, used more of it, and the harm did not move.

## Scoring, as registered

| track | registered | observed | verdict |
|---|---|---|---|
| **Codex** | harm unchanged, 1–5/20, centre 3 | **4/20** | **confirmed** |
| **Claude** | harm worse, ≤ 3/20 | **4/20** | **refuted** |

## The premise of this gate was wrong, and the other track said so beforehand

I registered this cell on a whole-episode saturation asymmetry: bound occupancy 7.9% at joint 5
against 1.3% at joint 3. That statistic is **duration-confounded**. Joint-5 failures run 220 steps
while its successes run 75–117, so a per-episode occupancy rate rewards long failures. Restricted to
the first 60 applied steps the ordering **reverses**: 10.6% in successes against 2.75% in failures.
Reaching the bound does not select failures.

The second track derived this from stored data alone, before the run, and predicted the null
correctly on that basis. My reasoning for the opposite category — that an estimator adapting
continuously with zero fixes in twelve opportunities must be tracking the wrong quantity, so more
headroom would make it worse — was not supported.

## What this does and does not establish

**Establishes:** projection saturation is not the dominant cause of the joint-5 harm. With the
constraint effectively removed and demonstrably exercised, nine episodes break either way.

**Does not establish:** that the cause is representational. This removes one competitor from a set
of at least four — state-independent representation, translation-only correction (`corr_dims 0,1,2`,
never varied), contact or reachability effects, and adaptation dynamics. Success counts alone cannot
separate them; that needs an intervention on the correction subspace, not another observational
statistic.

**Consequence for the successor direction:** its motivation is unchanged in strength, not improved.
A real and significant measured failure (nine broken, p = 0.0039) with an **unidentified** cause, for
which a state-dependent basis is one plausible fix among several. That is weaker than "diagnosed and
addressed" and stronger than nothing, and it should be written that way.

**Validity gate pending.** Joint 3 at `clip 0.60` must reach ≥ 14/20 (my registration) or ≥ 15/20
(the second track's) or neither arm is interpretable.
