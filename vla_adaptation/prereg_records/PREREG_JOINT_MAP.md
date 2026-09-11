# Pre-registration — which joint faults are repairable from the Cartesian action interface?

**Written 2026-09-07, BEFORE any cell runs.** Fourth registered hypothesis in this project's
audit line. The previous three were refuted or tied on their headline prediction
(`PREREG_ALOHA_NORM_CHANNELS.md`, `PREREG_ALOHA_INNOV_LAW.md`, `PREREG_MATCHED_OBSERVER.md`).
That is the honest prior.

---

## 1. What the paper currently has, and why it is thin

§`sec:joint` answers the sharpest objection to this work — *"you inject a fault at exactly the
interface your estimator inverts, then invert it"* — by injecting faults **below** the
operational-space controller, in the MuJoCo model, where the fault-to-motion map is
state-dependent through the Jacobian.

But the evidence is narrower than the section reads. Every paired joint-level cell in the
repository is on **joint 3, the elbow**:

| stored cell | frozen → corrected |
|---|---|
| `torque:3:5.0` | 20/40 → 32/40 (and 11/20 → 17/20) |
| `friction:3:4.0` | 20/20 → 20/20 (ceiling) |
| `friction:3:10.0` | 11/20 → 14/20 (p = 0.51) |
| `friction:3:20.0` | 0/20 → 8/20 |
| `lock:3:0.05` | 0/20 → 0/20 |

`joint_fault.py` supports five kinds; **`damping` and `gain` have never been run as a paired
cell**, and **no joint other than 3 has ever been tested for repair** — joint 1 appears only in
an estimate-only probe. So "we test that class" rests on one joint of seven.

## 2. The claim this experiment would earn

Not "it also works on more joints". The claim is a **map with a mechanism**:

> Which joint-level torque faults a Cartesian action-interface correction can repair is
> determined by how that joint's error projects into the correctable Cartesian subspace, and
> that projection is measurable *before* any repair attempt.

§29.2 established that joint-level faults are identified **on the translation channels and
nowhere else**. So a joint whose torque bias displaces the end-effector mainly in translation
should be repairable; one whose bias appears mainly as rotation, or which saturates, should not.
That is a falsifiable structural prediction, not a survey.

## 3. Design

`libero_spatial`, π0.5, correction on translation (`--corr-dims 0,1,2`) per §29.2,
`--joint-fault torque:J:5.0` for **J = 0 … 6**, n = 20 paired, evaluation inits 45–46,
published constants, healthy phantom subtracted. Seven cells.

Plus a **prior measurement** on each joint, taken first and used to form the prediction before
any repair cell is scored: an estimate-only probe (`--estimate-only`) giving the induced
Cartesian displacement and its split between translation and rotation.

## 4. Predictions, registered in advance

1. **The induced displacement varies substantially across joints.** §29.1 already shows
   5 N·m gives +4.0 cm at joint 1 and +9.1 cm x / +7.8 cm z at joint 3, so a fixed 5 N·m is
   *not* a fixed difficulty. Cells are therefore reported against their measured displacement,
   not as a bare success table.
2. **Repair success is ordered by the translation share of the induced displacement.** Rank the
   seven joints by (translation displacement) / (total displacement) from the probes; the
   Spearman correlation between that rank and repair success will be **positive**, and I
   commit to reporting it whatever it is.
3. **At least one joint fails to repair** despite being identified — extending §`sec:joint`'s
   existing finding that identifiability is necessary and not sufficient (the lock). If every
   joint repairs, prediction 2 is untestable and the result is a weaker "it generalises" claim.
4. **Wrist joints (5, 6) repair worse than shoulder/elbow joints (1, 3)**, because a wrist
   torque bias produces predominantly end-effector *rotation*, which §29.2 says is not where
   joint faults are identified.

**What falsifies the claim:** no relationship between translation share and repair success
(prediction 2 fails), or uniform success/failure across all seven joints.

## 5. Analysis, fixed in advance

- Paired exact McNemar per cell via **both** `openpi/mcnemar.py` and
  `openpi/mcnemar_crosscheck.py`.
- Regression rate against **opportunities**, per `openpi/regression_rate.py` — several cells
  will have a non-zero frozen baseline, so this matters here more than in the observer study.
- Spearman correlation between translation-share rank and repair success, reported with n = 7
  and its p-value, **and** with the explicit caveat that n = 7 cannot resolve a weak
  correlation.
- Each cell reported with its measured Cartesian displacement alongside its success rate.
- No cell dropped after seeing its result. No constant retuned.

## 6. Stated limits, before the fact

- One suite, one backbone, one fault kind (torque), one magnitude. This maps the *joint* axis
  only.
- 5 N·m is not equal difficulty across joints; that is the point of prediction 1 and the reason
  displacement is reported alongside.
- n = 20 per cell, and the project documents ±10 points of free variation at that n. Only
  large per-cell effects are detectable; the *ordering* across seven cells is the claim, not
  any single cell.
- `M` is calibrated at task 0 init 45, inside the evaluation band (commit `a10d5f3`). This
  affects all seven cells equally and does not distort the ordering, but no cell here is scored
  on a fully held-out calibration.

---

# AMENDMENT — the registered discriminator does not discriminate

**Added 2026-09-07 after the estimate-only probes and BEFORE any repair cell runs.** The probes
were run first precisely so this could be caught here rather than rationalised afterwards.

## Prediction 2 is untestable as designed

Registered: *"repair success is ordered by the translation share of the induced displacement."*

Measured, from telemetry `measured` vectors over 200 steps per joint (mean achieved motion,
translation vs rotation):

| joint | \|trans\| | \|rot\| | translation share |
|---|---|---|---|
| 0 | 0.0827 | 0.0047 | 0.998 |
| 1 | 0.0740 | 0.0071 | 0.995 |
| 2 | 0.0809 | 0.0048 | 0.998 |
| 3 | 0.0438 | 0.0034 | 0.997 |
| 4 | 0.0644 | 0.0124 | 0.982 |
| 5 | 0.0629 | 0.0148 | 0.973 |

**The translation share spans 0.973–0.998 — a 2.5% range.** Every joint torque bias appears at
the end-effector as almost pure translation. With seven cells at n = 20 and ±10 points of
documented free variation, no ordering over a 2.5% spread is detectable. Prediction 2 cannot be
tested by this design and **will be reported as untestable, not as refuted**.

I also initially computed this split from the *estimate* `f̂` rather than the measured motion,
which was doubly wrong: §29.2 already established that joint faults are identified on
translation and nowhere else, so that quantity was guaranteed to be ~1.0 before I measured it.
The table above uses achieved motion and is the correct quantity; it still does not
discriminate.

## A replacement discriminator, registered before any repair cell runs

**Total induced displacement varies nearly 2×** across joints — 0.083 (j0) down to 0.044 (j3) —
and that is a usable axis where translation share is not.

**New prediction (5), registered now, before any repair outcome exists:** repair success is
**not monotone** in induced displacement magnitude. Specifically, the largest-displacement
joints (0, 2) will *not* repair best. Two opposing effects are in play and I am predicting the
second dominates:

- larger displacement means more signal, so easier identification;
- larger displacement means more damage and more excursion outside the regime where the plant
  model was identified, which §`sec:severity` already shows is what bounds the Cartesian
  cells ("the boundary is the plant leaving the regime where its motion is informative").

**Falsified if** repair success rises monotonically with displacement magnitude.

This is a post-hoc *motivated* prediction — the axis was chosen after seeing the probe data —
but it is registered **before any repair cell has run**, and the distinction is stated rather
than blurred. It is weaker evidence than prediction 2 would have been had it been testable, and
it will be reported as such.

## What this experiment can still establish regardless

Even with both ordering predictions weak, the seven cells answer a question the paper currently
cannot: **the joint-level section rests on one joint of seven.** A map across all seven — with
each cell's measured displacement reported alongside its success — replaces "we test that
class" with an actual characterisation of where in the kinematic chain the method works. A
uniform result is a legitimate and useful finding; so is a ragged one.
