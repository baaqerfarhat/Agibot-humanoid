# Pre-registration — which measurement of a fault predicts how much it damages a frozen VLA

**Written 2026-09-07 03:45 UTC.** State of the joint-map sweep at the moment of writing, verified
against the remote filesystem and process table and recorded here so the claim can be checked:

| joint | cell status at registration |
|---|---|
| 0, 1, 2, 3 | **complete and observed** — used as the anchor set below |
| 4 | **executing, result not yet written or seen** — `torque:4:5.0` was in the process table |
| **5, 6** | **not started** — these are the clean prospective cells |

Predictions below are prospective for joints 5 and 6 without qualification, and for joint 4 in the
weaker sense that its outcome did not exist anywhere when this was committed. Score them separately.

---

## 1. The correction that produced this

`PREREG_JOINT_MAP.md`'s amendment ranked the seven joints by "induced displacement", computed from
the telemetry field `measured`. **That was the wrong quantity.** `measured` is the *achieved motion
increment* — how far the end-effector actually moved. A joint fault severe enough to stall the
policy produces *less* achieved motion, so `measured` scores the most damaging fault as the
smallest. It ranked joint 3 last (0.0438) when joint 3 is the most damaged joint on the arm.

The right quantity is the residual `r` = measured motion minus the FIR plant model's prediction
from the commanded action, i.e. the part of the motion the healthy plant model cannot explain.
That is the disturbance itself, in action units, and it needs no healthy reference episode.

Recomputed from the estimate-only probes already on disk, `frozen_faulted` arm, warmup excluded:

| joint | mean \|r\| translation | rank by `r` | rank by `measured` (the wrong one) |
|---|---|---|---|
| 0 | 0.0233 | 7 | 1 |
| 1 | 0.0303 | 5 | 3 |
| 2 | 0.0286 | 6 | 2 |
| **3** | **0.1155** | **1** | **6** |
| 4 | 0.0463 | 4 | 4 |
| 5 | 0.0710 | 2 | 5 |
| 6 | 0.0472 | 3 | — |

The two instruments **anti-correlate**. This is the point of the study.

## 2. The anchor set, observed

| joint | \|r\| | frozen | corrected | fixed / broken | p | repair opportunities |
|---|---|---|---|---|---|---|
| 0 | 0.0233 | 19/20 | 19/20 | 0 / 0 | 1.000 | 1 |
| 1 | 0.0303 | 19/20 | 19/20 | 1 / 1 | 1.000 | 1 |
| 2 | 0.0286 | 18/20 | 19/20 | 2 / 1 | 1.000 | 2 |
| 3 | 0.1155 | 11/20 | 18/20 | 8 / 1 | **0.039** | 9 |

A 5 N·m torque bias on joints 0–2 costs 1–2 episodes in 20. The same bias on joint 3 costs 9.
**"5 N·m" is not a fixed difficulty**, which was prediction 1 of `PREREG_JOINT_MAP.md` and is now
confirmed with a 4–5× spread in the disturbance it actually produces.

## 3. Hypothesis

**H1.** The disturbance magnitude measured at the action interface from healthy-plant residuals —
obtainable from a short estimate-only probe, before any repair is attempted — predicts how much a
frozen VLA's task success degrades. Achieved end-effector motion does not, and inverts.

**H0.** `r` does not order the damage across joints, or orders it no better than achieved motion.

## 4. Predictions, registered now

1. **`P-A` (clean).** Joint 5 (`r` = 0.0710) is more damaged than joints 4 (0.0463) and 6 (0.0472):
   frozen success at joint 5 < frozen success at both 4 and 6.
2. **`P-B` (clean for 5 and 6).** Joints 4, 5, 6 are all more damaged than the least-damaged of
   joints 0–2, i.e. frozen success < 18/20 for each.
3. **`P-C`.** Spearman ρ between `r` and frozen success across all seven joints is **negative**;
   between `measured` and frozen success it is **positive or near zero**. Both reported whatever
   they are, with n = 7 and the explicit caveat that n = 7 cannot resolve a weak correlation.
4. **`P-D`.** Joints 4, 5, 6 each have ≥ 3 repair opportunities, so repair is actually scoreable
   there — unlike joints 0–2, where 1, 1 and 2 opportunities make repair unmeasurable at any n.
5. **Quantitative.** Frozen success falls roughly linearly in `r` over this range: joints 0–2 sit
   near 18–19/20 at `r` ≈ 0.025, joint 3 at 11/20 at `r` = 0.116. Interpolating, joint 5 lands near
   **15–17/20** and joints 4 and 6 near **17–18/20**. Stated as a range because n = 20 carries ±10
   points of documented free variation; the *ordering* is the claim, the interpolation is a bonus.

**Falsified if** joint 5 is not more damaged than 4 and 6, or if `measured` orders the damage as
well as `r` does.

## 5. Analysis, fixed in advance

- `r` is read from probe telemetry **already on disk before this was written** and is not recomputed
  or reweighted after seeing any outcome. The numbers in §1 are final.
- Frozen success and repair scored by both `openpi/mcnemar.py` and `openpi/mcnemar_crosscheck.py`.
- Repair reported against **opportunities**, per `openpi/regression_rate.py`.
- Spearman for both instruments, reported together.
- Episode lengths differ across joints (294 steps at joint 0, 660 at joint 3) because failing
  episodes run to timeout. `r` is a **per-step mean**, so it is not inflated by that; this is stated
  because it is exactly the confound that corrupted the `measured` version.

## 6. Stated limits, before the fact

- One suite, one backbone, one fault kind, one magnitude. Seven points.
- This predicts **damage**, not **repair**. Whether `r` also predicts repairability is a separate
  question that joints 0–2 cannot answer, because they have nothing to repair.
- `r` is measured on the faulted arm, so it requires the fault to be present. It is a *diagnostic*
  available before choosing a correction, not a quantity known before the fault occurs.
- The relationship could be near-tautological: a bigger unmodelled disturbance plausibly does more
  damage. The content is not that it holds, but that the *obvious* instrument — how far the
  end-effector actually moved — gets the ordering backwards, and would have sent an engineer
  looking at the wrong joint.

## 7. Relation to the independent track's ranked plan

An independent strategic review ranked "a prespecified physical measurement rule that forecasts
repair and changes estimator selection" as the single highest-value remaining direction, and
specified that any such rule must be compared against simple sensitivity baselines rather than
merely explaining a reversal after the fact. This study is the cheapest instance of exactly that
shape: two candidate instruments, one prespecified ordering, cells that have not run. It uses
rollouts already budgeted, so it costs nothing beyond the analysis.

It is **not** the full design that review proposed. That design holds the aggregate error norms
identical by construction across a permuted error direction, which this does not do — here `r` and
the damage could both be driven by the same underlying severity. This is the screen, not the result.

---

# RESULT — the correlation holds, two of the sharper predictions do not

**Added 2026-09-07 after all seven cells.** Artifacts: `results/joint_map/cell_torque_*.json`,
`results/joint_map/instruments.json`.

| joint | \|r\| disturbance | \|measured\| motion | frozen | corrected | fixed / broken | p | opportunities |
|---|---|---|---|---|---|---|---|
| 0 | 0.0233 | 0.1504 | 19/20 | 19/20 | 0 / 0 | 1.000 | 1 |
| 1 | 0.0303 | 0.1619 | 19/20 | 19/20 | 1 / 1 | 1.000 | 1 |
| 2 | 0.0286 | 0.1545 | 18/20 | 19/20 | 2 / 1 | 1.000 | 2 |
| 3 | 0.1155 | 0.1255 | 11/20 | **18/20** | 8 / 1 | **0.039** | 9 |
| 4 | 0.0463 | 0.1277 | 17/20 | 14/20 | 1 / 4 | 0.375 | 3 |
| 5 | 0.0710 | 0.1126 | 12/20 | **3/20** | 0 / 9 | **0.0039** | 8 |
| 6 | 0.0472 | 0.1236 | 7/20 | 11/20 | 8 / 4 | 0.388 | 13 |

## Scoring, as registered

| prediction | registered | measured | verdict |
|---|---|---|---|
| P-C `r` vs frozen | Spearman **negative** | **−0.829**, p = 0.021 | **confirmed** |
| P-C `measured` vs frozen | **positive or near zero** | **+0.829**, p = 0.021 | **confirmed** |
| P-B joints 4,5,6 all < 18/20 | < 18 | 17, 12, 7 | **confirmed** |
| P-D ≥ 3 opportunities each | ≥ 3 | 3, 8, 13 | **confirmed** |
| P-A joint 5 more damaged than 4 **and** 6 | both | 12 < 17 ✓, 12 > 7 ✗ | **refuted on half** |
| quantitative interpolation | j4 17–18, j5 15–17, j6 17–18 | 17 ✓, 12 ✗, **7** ✗✗ | **refuted** |

**The rank correlation is real and the point estimates are not.** `r` orders the seven joints well,
but it badly mispredicts joint 6, which carries a middling disturbance (0.0472, fourth of seven) and
the **worst** damage of any joint (7/20). Whatever makes a wrist-roll bias destructive is not its
magnitude at the action interface. The registered linear interpolation was wrong for two of the
three held-out joints and should not be repeated.

## The finding that matters more than the one registered

**The correction's entire benefit comes from one joint of seven.** Summing fixed minus broken:

- **joint 3 (elbow) alone: +7**
- **all six other joints together: −7**

Joint 5 is a significant net harm on its own: 12/20 → 3/20, **zero fixed against nine broken**,
p = 0.0039. Joint 4 is 17/20 → 14/20, one fixed against four broken. Only joint 6 among the
non-elbow joints repairs at all (8 fixed / 4 broken), and it does not reach significance.

`\S`\ref{sec:joint} currently rests on joint 3, and joint 3 is now the **only** joint of seven where
this works. The section as written generalises from the one cell that succeeds. That has to change
in the paper, and it is a stronger objection than the one the section was written to answer.

## What this does not establish

- Seven joints, one suite, one backbone, one fault kind at one magnitude.
- 5 N·m is not equal difficulty across joints — that was prediction 1 of `PREREG_JOINT_MAP.md` and
  is confirmed with a 5× spread in `r`. The cells are therefore not equally powered, and joints 0–2
  cannot test repair at all with 1, 1 and 2 opportunities.
- The `r`-versus-`measured` reversal is a statement about two *instruments*, not about a mechanism.
  It says the obvious measurement misleads; it does not say why joint 6 breaks the pattern.
