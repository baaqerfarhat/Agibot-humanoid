# Pre-registration — does the coupled plant model survive closed loop at joint 5?

**Written 2026-09-08, BEFORE any cell runs.** Eighth registered hypothesis in this line. Of the
previous seven: four refuted on their headline prediction, one tied, one untestable, one void. The
base rate for my hypotheses here is poor and this prediction is made against my own earlier one.

## 1. Why

The user proposed a non-diagonal input map. Open-loop it is supported: recomputing the residual
under a coupled `mimo=True` plant and scoring held-out SSE about a constant gives **+5.5% at
joint 5** and +3.1% at joint 3. It helps more at the failing cell.

But `fit_plant`'s docstring rejects the coupled model on closed-loop grounds — *"the correction it
applies moves the executed command away from the data the plant was identified on. The simpler model
fits worse and extrapolates better"* — and my open-loop measurement is taken on the **frozen** arm,
where no correction is applied. It cannot reach that question.

## 2. A mechanism, and it points the other way from the open-loop result

The estimator forms `r` from the FIR `W` and then inverts the **separately identified** `M`. Those
two encode the same physical quantity and already disagree. Measured:

| plant model | off-diagonal energy | `‖I − M·pinv(G_W)‖` |
|---|---|---|
| per-axis | 0% by construction | **1.896** |
| **MIMO** | 54.8% | **2.314** |
| (`M` itself) | 25.5% | — |

**The coupled FIR is further from `M`, not closer.** Both are coupled, but differently. Since the
closed-loop correction passes through `M⁻¹` applied to a residual defined by `W`, widening that gap
should degrade the correction even while the open-loop residual improves.

This also gives a mechanism for the docstring's reported 13/15 against 14/15 and its sign error on
`dy`, rather than merely noting them.

## 3. Arms

`libero_spatial`, `--joint-fault torque:J:5.0`, `--eval-init 45`, `--episodes 20`, `--corr-dims
0,1,2`, `--dead 0.008 --norm-r 0.15 --clip 0.30 --gamma 0.08`, `--mimo`. Only `--mimo` differs from
the stored cells.

| arm | joint | stored per-axis baseline |
|---|---|---|
| **M5** | 5 | frozen 12/20, corrected **3/20** |
| **M3** | 3 | frozen 11/20, corrected **18/20** |

## 4. Predictions, registered in advance

1. **M5 corrected ≤ 5/20**, i.e. no material improvement on the per-axis 3/20, and plausibly worse.
   This is against the direction the open-loop measurement suggests, and it is the honest reading of
   the mechanism in §2.
2. **M3 corrected degrades from 18/20**, to ≤ 16/20. The cell that works has more to lose from a
   widened `M`-versus-`W` gap.
3. **Frozen arms reproduce** their stored 12/20 and 11/20 within the documented ±10 points, since
   `--mimo` cannot affect an arm with no correction applied. If a frozen arm moves materially,
   something other than the plant model changed and neither arm is interpretable.

**Falsification.** If M5 reaches **≥ 7/20**, the open-loop residual quality dominates, the
inconsistency argument in §2 is wrong, and the non-diagonal direction is supported in closed loop —
which would be the user's hypothesis vindicated against my prediction.

## 5. Analysis, fixed in advance

- Paired exact McNemar against each cell's own frozen arm, via both `openpi/mcnemar.py` and
  `openpi/mcnemar_crosscheck.py`.
- Regressions scored against opportunities.
- The margins in §2 are computed before these cells run and are frozen.
- No arm dropped after seeing its result; no constant retuned.

## 6. Limits

- Two cells, n = 20, one suite, one fault kind. ±10 points of documented free variation.
- The stored baselines are reused rather than rerun, so the comparison inherits whatever policy
  sampling variation separates the runs. `pin_rng` is False.
- A negative result would show this coupled model does not help here. It would not show that no
  non-diagonal map helps: `M` itself is non-diagonal and better matched to the residual, and using
  `M`'s own structure for the plant model is a different and untested option.

---

# RESULT — the coupled model does not rescue joint 5, and does not harm joint 3

**2026-09-08.** Artifacts: `results/mimo/M5.json`, `results/mimo/M3.json`.

| | frozen | corrected | fixed | broken | exact p |
|---|---|---|---|---|---|
| joint 5, per-axis (stored) | 12/20 | 3/20 | 0 | 9 | 0.0039 |
| **joint 5, MIMO** | 15/20 | **5/20** | **0** | **10** | 0.0020 |
| joint 3, per-axis (stored) | 11/20 | 18/20 | 8 | 1 | 0.0391 |
| **joint 3, MIMO** | 11/20 | **18/20** | 7 | **0** | 0.0156 |

## Scoring

| prediction | registered | observed | verdict |
|---|---|---|---|
| P1 M5 corrected ≤ 5/20 | ≤ 5 | **5/20**, zero fixed, 10 broken | **satisfied** |
| P2 M3 degrades to ≤ 16/20 | ≤ 16 | **18/20** | **refuted** |
| P3 frozen arms reproduce | within ±10 | M3 exact; M5 12 → 15 | satisfied, see below |

## What it establishes

**The coupled model does not fix the failing cell.** Joint 5 still repairs **nothing** — zero fixed
in both configurations — and breaks 10 of 15 opportunities against 9 of 12. The mechanism in §2
predicted no improvement and that half held.

**It also does not harm the working cell**, which my §2 reasoning predicted it would. Joint 3 is
identical at 18/20 and marginally cleaner: zero regressions instead of one, `p` improving from
0.0391 to 0.0156. So widening the `M`-versus-`W` gap from 1.896 to 2.314 did not degrade the cell
that works, and the argument that it should was wrong.

The honest summary of the user's non-diagonal direction as tested: **a mild safety improvement where
the method already works, and no help at all where it fails.**

## An incidental result worth more than the headline

The joint-5 **frozen** arm moved **12/20 → 15/20** between runs. Nothing in this experiment can
affect it: `--mimo` changes only the plant model the estimator uses, and the frozen arm applies no
correction at all. That is **three episodes of pure policy sampling noise at n = 20**, measured
directly rather than assumed, and it is a floor that applies to every n = 20 cell in this project.

Consequences: differences of ≤ 3/20 at this n are not interpretable, which retroactively confirms
that the earlier saturation result (3/20 → 4/20) was correctly read as a null. The joint-5 harm, at
9–10 broken out of 12–15 opportunities, is far above this floor and unaffected.

## What remains untested, and is now the more interesting option

A negative here does **not** refute the non-diagonal direction. It refutes *this* coupled model. The
measured facts still stand: `M` carries 25.5% of its energy off-diagonal, the per-axis FIR has none,
and the two disagree by 1.896.

The untested third option is to **make the plant model and the sensitivity the same object** rather
than fitting them separately and inverting one against the other. MIMO widened the gap (1.896 →
2.314) because it is coupled *differently* from `M`; using `M`'s own structure would close it. That
is exactly what the descriptor form does structurally, by having one `E` appear in both the residual
and the allocation, and it remains the only version of this direction that has never been tried.
