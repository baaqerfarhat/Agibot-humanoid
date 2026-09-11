# Pre-registration — does the authority margin predict which correction subspaces are safe?

**Written 2026-09-08, BEFORE any cell runs.** Seventh registered hypothesis in this line. Of the
previous six: three refuted on their headline prediction, one tied, one declared untestable before
running, one void by adversarial review. That is the honest prior.

## 1. The quantity, and why it is preregisterable

`authority_margin(E_a, E_b) = ||I - E_a pinv(E_b)||_2` is the worst-case amplification a correction
applies when the true input map is `E_a` and the allocator believes `E_b`. Below 1 the correction
strictly helps for every disturbance; above 1 some disturbance is amplified by that factor.

It is computed from **healthy calibration episodes only** — no fault, no rollout, no outcome. The
stored LIBERO calibration has three healthy episodes, giving seven independent fits of the same
system, and the margin is the worst disagreement among them.

Measured per axis, over all 42 ordered pairs of fits:

| axis | max margin | pairs above 1 |
|---|---|---|
| x | 0.061 | 0% |
| y | 0.071 | 0% |
| z | 0.065 | 0% |
| **rx** | **4.974** | 31% |
| **ry** | **4.268** | 24% |
| **rz** | **0.087** | 0% |

Translation and `rz` are identifiable from this data. `rx` and `ry` are not.

## 2. The contrast this makes available

| subspace | margin | |
|---|---|---|
| `0,1,2` translation | 0.071 | already run: **16/20** |
| `0,1,2,3,4,5` all six | 4.974 | already run: **2/20** |
| **`0,1,2,5` translation + rz** | **0.087** | **not run** |
| **`0,1,2,4` translation + ry** | **4.268** | **not run** |

The last two are the test. **Same channel count, same structural change — add one rotation axis —
and opposite predictions.** Any confound that scales with the number of inputs, or with "adding
rotation" as such, affects both arms equally. Only *which* axis differs.

## 3. The confound, and the control that removes it

The two existing arms differ in **three** of 48 recorded arguments, one of which is a label. The
substantive differences are `corr_dims` (0,1,2 against 0,1,2,3,4,5) **and** `baseline`
(`none` against `weighted_dob`). So the existing 16/20-against-2/20 comparison does not isolate the
subspace; the estimator changed too.

**Arm A removes this.** Running `weighted_dob` with `corr_dims 0,1,2` holds the estimator fixed and
varies only the subspace.

## 4. Arms

`libero_object`, `--joint-fault torque:3:5.0`, `--eval-init 35`, `--episodes 20`, `--dead 0.008`,
`--norm-r 0.15`, `--clip 0.3`, `--gamma 0.08`, `--baseline weighted_dob` throughout. Only
`--corr-dims` varies.

| arm | corr-dims | margin | role |
|---|---|---|---|
| **A** | `0,1,2` | 0.071 | control: isolates the estimator change from the subspace |
| **B** | `0,1,2,5` | 0.087 | predicted **safe** |
| **C** | `0,1,2,4` | 4.268 | predicted **unsafe** |

Stored references, not rerun: legacy `none`+`0,1,2` = 16/20; `weighted_dob`+all six = 2/20;
OFF = 8/20 (the shared frozen control, verified consistent across both stored files).

## 5. Predictions, registered in advance

1. **Arm A scores at or above OFF's 8/20**, and materially above the 2/20 of `weighted_dob` with
   all six. If A instead scores near 2/20, the estimator change explains the existing harm, the
   margin story is confounded for this comparison, and predictions 2-3 are the only remaining test.
2. **Arm B (+rz, margin 0.087) scores at or above OFF**, and comparable to arm A within the
   documented ±10 points at n = 20.
3. **Arm C (+ry, margin 4.268) scores materially below arm B.** This is the headline. A margin
   computed from healthy data with no knowledge of any outcome predicts which of two structurally
   identical interventions is harmful.
4. **Ordering: B > C.** Stated separately because it survives even if the absolute levels shift.

**Falsification.** If B and C score within noise of each other, the margin does not discriminate
and the instrument is worthless for subspace selection — which is the claim being tested. If C
scores at or above B, it is refuted outright.

## 6. Analysis, fixed in advance

- Paired exact McNemar on matched `(task, init)` via both `openpi/mcnemar.py` and
  `openpi/mcnemar_crosscheck.py`, each arm against the shared OFF control and against arm A.
- Regressions scored against **opportunities**, per `openpi/regression_rate.py`.
- The margins in §1 are computed from healthy episodes **before** these cells run and are frozen;
  they are not recomputed or reweighted afterwards.
- No arm dropped after seeing its result, no constant retuned.

## 7. Stated limits, before the fact

- One suite, one joint, one fault magnitude, n = 20 against ±10 points of documented free variation.
  The **ordering** B > C is the claim; absolute levels are not.
- The margin uses the FIR's implicit diagonal gain, which is itself one of two mutually inconsistent
  input models in this codebase (`||I - M pinv(FIR_gain)|| = 1.896`). A margin computed against `M`
  would differ, and this study does not establish which is the better reference.
- Three healthy episodes is a small basis for an identifiability claim. The margin could be large
  because the map genuinely varies, or because three episodes underdetermine it. **This study
  cannot distinguish those**, and either way the consequence for a fixed calibration is the same.
- A confirmed prediction supports the margin as a **subspace admission criterion**. It would not
  establish that a learned state-dependent basis works, which remains untrained and untested.

---

# VOID — the margins were computed on the wrong calibration, and the claim is refuted

**2026-09-08, before any cell produced a result.** The three cells failed to launch (the deployed
runner predates `weighted_dob`), and tracing the correct runner exposed the defect.

## What went wrong

The cells use a **`libero_object`-specific calibration**
(`followup_20260908/runs/panda_confirmation/calibration/libero_object/fir.json`), not the
`libero_spatial` file (`results/phase05/error_signal_so3.json`) every margin in §1 was computed on.
The registered predictions were therefore for a different robot calibration than the experiment.

## Recomputed on the correct calibration, which has ten healthy episodes rather than three

| axis | min over 10 leave-one-out fits | max | max/min |
|---|---|---|---|
| x | 0.2222 | 0.2283 | 1.03 |
| y | 0.2627 | 0.2633 | 1.00 |
| z | 0.2407 | 0.2452 | 1.02 |
| rx | 0.2532 | 0.2613 | 1.03 |
| **ry** | 0.2039 | 0.2232 | **1.09** |
| rz | 0.2367 | 0.2467 | 1.04 |

Margins over all 90 ordered pairs: translation **0.027**, rotation **0.094**, all six **0.094**.
**Every axis is identifiable. Zero pairs above 1.**

## The claim is refuted, by the alternative this preregistration named

§7 stated: *"the margin could be large because the map genuinely varies, or because three episodes
underdetermine it. This study cannot distinguish those."* Ten episodes distinguishes them, and the
answer is **three episodes underdetermined it**. The `ry` gain varying 5.3-fold across single
spatial episodes was sampling noise, not state dependence.

Consequently:

- **The rotation-is-unidentifiable claim is refuted.**
- **The margin does not explain the LIBERO Object counterexample.** All six inputs have margin 0.094
  on the calibration that cell actually used, so the margin predicts *safe* where the observed
  result is 2/20. It gets the counterexample wrong.
- **This preregistration is void.** Arms B and C both have margin below 0.1 on the correct
  calibration, so there is no contrast to test and no prediction to score. No cell was run.

## What survives, and is strengthened

The **FIR-versus-M inconsistency**, recomputed on the same correct calibration:

| | x | y | z | rx | ry | rz |
|---|---|---|---|---|---|---|
| FIR gain | .225 | .263 | .242 | .259 | .215 | .239 |
| diag(M) | **.053** | .169 | .359 | .261 | .293 | .258 |
| ratio | **0.24** | 0.64 | 1.48 | 1.01 | 1.36 | 1.08 |

`M` carries **71.6%** of its energy off-diagonal, which the per-axis FIR cannot represent at all,
and `||I - M pinv(FIR)|| = 2.789`, `||I - FIR pinv(M)|| = 4.368`.

This is now a **stronger** result than before, because the leave-one-out analysis rules out the
obvious objection. Each model is individually well determined — every leave-one-out fit agrees to
within 1.09x — and they disagree **with each other** by a factor of 2.8 to 4.4. Two stable,
precisely identified models of the same physical quantity, mutually inconsistent. That is
structural, not sampling.

So the user's hypothesis that the FIR form is implicated stands, on better evidence than before,
while my extension of it into an identifiability claim does not.

## Cost

Zero GPU. The launch failure forced tracing the actual runner, which exposed the calibration
mismatch before any cell produced a number that would have been interpreted against the wrong
baseline.

## A second error, found by the independent track

My reasoning had a logical defect independent of the calibration mismatch. I inferred from
"pairwise margins between honest fits exceed 1" that "no fixed calibration can be safe." **That does
not follow.** The margin is not symmetric and not a metric: an intermediate calibration can serve
both candidates.

Verified: for candidate true maps 1 and 3, `m(3,1) = 2.00` — mutually far apart — yet calibrating at
2.0 gives `m(1,2) = m(3,2) = 0.50`, both safe. Calibrating at 2.5 gives 0.60 and 0.20.

So even had the margins been computed on the right calibration, the conclusion I drew from them
would not have followed. The correct statement is the narrower one the independent track reached:
the artifacts do not *establish* a qualified six-channel fixed map, which makes it an **unfulfilled
requirement** rather than a refuted possibility. "No fixed calibration can work" was never supported.
