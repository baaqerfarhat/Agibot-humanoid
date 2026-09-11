# Pre-registration — measuring the GR1's task margin, to make Proposition 2 prospective

**Written 2026-09-07, BEFORE any margin cell runs.** Fifth registered hypothesis in this audit
line. The previous four were refuted, tied, or had their discriminator declared untestable.
That is the honest prior.

---

## 1. Why this matters more than anything else outstanding

Proposition 2 is the paper's only candidate for a genuine scientific contribution. Everything
else has been narrowed by measurement this session: the update law is not what earns the result
(five estimators tie it, `PREREG_MATCHED_OBSERVER.md`), and three attempts at a methodological
claim failed.

Prop 2 says repair succeeds when the correction's within-episode variation `σ` fits inside the
task's spatial margin `μ`, **both measurable before any repair attempt**. Its evidential status
today:

| embodiment | `σ` measured? | `μ` measured? | outcome |
|---|---|---|---|
| LIBERO / Panda | yes (0.019 cm/step) | **no** — asserted as "centimetres" | continuous works |
| ALOHA | yes (0.43 cm) | **yes** — static sweep, §27.8, `< 0.5` cm | continuous fails, hold works |
| GR1 humanoid | yes (0.013–0.058 rad wander) | **NO** | continuous fails (8/30), hold works (21/30) |

So `μ` has been measured on **one** of three embodiments. On the GR1 the margin is *inferred
from the failure it is supposed to explain*, which is circular. The paper's sentence
"Proposition 2 decides the scheme on the third manipulator as on the second" is therefore a
post-hoc label, not a prediction — a distinction I initially got wrong and am correcting here.

**Measuring `μ` on the GR1 independently converts Prop 2 from a description into a criterion
whose two quantities were measured separately on a third robot.** That is the difference
between "consistent with" and "predicted by", and it is the cheapest remaining path to a real
contribution.

## 2. Design

RoboCasa GR1, GR00T N1.5, plate-to-plate, `+0.10` rad offset on the seven right-arm joints —
the same cell as §32. Apply a **static** correction (no estimator) at a fixed fraction of the
true fault and measure success. `--static-corr` already exists; note the sign convention
differs between runners (commit `021eba2`).

| arm | static correction | residual offset |
|---|---|---|
| 1.00 | 0.100 rad | 0 |
| 0.95 | 0.095 rad | 0.005 rad |
| 0.90 | 0.090 rad | 0.010 rad |
| 0.85 | 0.085 rad | 0.015 rad |
| 0.80 | 0.080 rad | 0.020 rad |
| 0.70 | 0.070 rad | 0.030 rad |

n = 20 paired per arm, seeds 110–129 (the band §32.9 already used), healthy arm in the same
process. **`μ` is the largest residual offset at which success is still indistinguishable from
the healthy rate.**

## 3. Predictions, registered in advance

1. **`μ` < 0.058 rad.** The measured continuous-adaptation wander spans 0.013–0.058 rad and
   continuous fails, so under Prop 2 the margin must be below the wander. If `μ` comes out
   **above** 0.058 rad, Prop 2 does **not** explain the GR1 failure and something else does —
   this is the falsification condition and it is the whole point of the experiment.
2. **`μ` ≤ 0.015 rad**, i.e. success degrades by the 0.85 arm. Stated more sharply than
   prediction 1 because continuous fails *consistently* (8/30, not marginally), which under
   Prop 2 requires the margin to sit below most of the wander distribution rather than at its
   top edge.
3. **The 1.00 arm recovers the healthy rate** (≈21/30 scaled to n=20, so ≈14/20). If an exact
   static correction does not repair, the margin framing is inapplicable to this task and the
   study is void — the same gate Arm A served in the observer study.
4. **Success is monotone non-increasing in residual offset.** A non-monotone result means
   something other than the margin is driving the outcome.

## 4. Analysis, fixed in advance

- Paired exact McNemar per arm against frozen **and** against healthy, via both
  `openpi/mcnemar.py` and `openpi/mcnemar_crosscheck.py`.
- Regression rate against **opportunities**, per `openpi/regression_rate.py`.
- `μ` read off as the largest residual offset whose arm is indistinguishable from healthy at
  p > 0.05 — reported with the explicit caveat that "indistinguishable at n = 20" is a weak
  criterion, not equivalence.
- The comparison to `σ` uses the **already-measured** 0.013–0.058 rad from §32.7; that number
  is not recomputed for this study and is not adjustable by it.
- No arm dropped after seeing its result; no constant retuned.

## 5. Stated limits, before the fact

- One task on one humanoid. This makes Prop 2 prospective on a third embodiment; it does not
  make it a law.
- n = 20 per arm. The *shape* of the degradation curve is the claim, not any single arm.
- Converting radians at the joint to centimetres at the hand needs the GR1's Jacobian and is
  not attempted; `μ` is reported in radians of residual joint offset, the same units as `σ`,
  which is what the comparison actually requires.
- Prop 2 would remain unmeasured on LIBERO, where the margin is still only asserted. A complete
  test would measure it there too, and that is not attempted here.

---

## 6. Two corrections incorporated before this runs

An independent strategic review landed after this preregistration was drafted and corrected two
things I had been asserting. Both are recorded here because they bear on why *this* experiment
rather than another.

**There are not three accuracy–repair reversals.** I had claimed three independent appearances.
Checked individually:

| case | accuracy | success | verdict |
|---|---|---|---|
| ALOHA 0.019 → 0.020 static | improves | 5/20 → 8/20, improves | **not a reversal** — both move together (p = 0.45) |
| innovation E → F | MAE −34%, RMSE +0.6%, worst +28% | 6/20 → 2/20 | mixed accuracy, one donor, p = 0.219 |
| matched P → D | better on MAE, RMSE **and** worst-coord | 19/20 → 17/20 | the only clean reversal, p = 0.625 |

So: **one unresolved descriptive reversal**, not three demonstrations. The ALOHA case points
the opposite way and I was counting it in the wrong direction.

**The accuracy–repair angle is occupied prior art.** "Estimation accuracy is the wrong
downstream objective" is identification-for-control (Hjalmarsson & Gevers 1996) and value-aware
model learning (Donti 2017; Farahmand 2017). More damagingly, **J-PARC already reports an oracle
constrained-IK baseline with lower local error and no consistent task benefit** — the closest
competitor has published the dissociation on this exact problem.

**Consequence for this study.** The contribution is not "accuracy is the wrong objective". It is
narrower and, if it holds, actually useful: **a criterion measurable in advance that decides
which correction scheme to deploy, validated prospectively on a robot it was not built from.**
That is a decision rule, not an observation about objectives — and it is what the independent
review independently named as the strongest remaining opportunity.

This also sharpens the falsification condition. If `μ` comes out above the measured wander, the
criterion does not decide the GR1 and Prop 2 is a post-hoc label on all three embodiments. That
would be a genuine negative result about the paper's only candidate contribution, and it should
be reported as one.

---

## 7. Amendment — the arms are unpaired, so the test changes

**Added 2026-09-07 after merging `8c06a40`, still before any margin cell runs.**

Mahdi Taheri established that RoboCasa re-randomises the manipulated object, its placement and the
task language at **every** reset, including within a process: 45 of 86 bodies move between two
resets with the same seed. The GR1 arms were therefore never paired, and the fixed/broken counts
and paired p-values this preregistration cited in §1 were invalid.

**Corrected inputs**, unpaired at n = 30 per arm, Fisher exact:

| scheme | healthy | frozen | corrected | vs frozen | vs healthy |
|---|---|---|---|---|---|
| continuous | 23/30 | 1/30 | 8/30 | p = 0.08 (**not** significant) | p = 2.3×10⁻⁴ |
| identify then hold | 21/30 | 2/30 | **21/30** | p = 5.5×10⁻⁷ | p = 1 |

**The premise of this study survives.** Proposition 2 needs continuous adaptation to *fail to
repair* on the GR1, and it still does: continuous sits significantly below healthy at p = 2.3×10⁻⁴.
What weakened is the separate claim that continuous is better than doing nothing, which is now
unresolved. If anything this sharpens the question the margin measurement answers.

**Three changes to the design in §2–§4, all made before any cell runs:**

1. **Unpaired Fisher exact replaces paired McNemar** for every arm comparison. `openpi/mcnemar.py`
   and `mcnemar_crosscheck.py` are the wrong instruments here and are not used for this study.
2. **`n = 30` per arm, not 20**, matching the existing GR1 cells. Unpaired tests are less efficient
   than paired ones, so the n that sufficed under the old analysis does not carry over.
3. **"Seeds 110–129" is dropped as meaningless** for this embodiment. Matching seeds does not match
   scenes here, which is the whole content of the correction. Arms are independent samples.

`μ` is still read off as the largest residual offset whose arm is indistinguishable from healthy,
now by Fisher at p > 0.05, and still with the caveat that "indistinguishable at n = 30" is a weak
criterion rather than equivalence — a caveat that binds *harder* unpaired than paired.

The falsification condition in §3 and §6 is unchanged.

---

# WITHDRAWN as written — the design cannot reach its own falsification condition

**2026-09-07, before any cell ran.** Found by independent review, not by me. I had recommended this
study twice as the highest-value remaining experiment. It is defective.

**Defect 1, fatal as registered.** The registered sweep runs fractions 1.00 / 0.95 / 0.90 / 0.85 /
0.80 / 0.70 of the 0.10 rad fault, so the **largest residual tested is 0.030 rad**. The primary
falsification condition in §3 is `μ > 0.058 rad`. If every registered arm survives, `μ ≥ 0.030` and
the falsification question is **still open**. The design has no arm that can trigger its own
falsification. Reaching 0.058 rad needs fractions down to about 0.40, which was never registered.

**Defect 2, not repairable by extending the range.** The continuous and held outcomes on the GR1
were **already known** when this measurement was proposed. Measuring `μ` now tests an explanatory
*consequence* of Proposition 2; it cannot retroactively make the scheme comparison prospective. My
framing — "converts Prop 2 from a description into a criterion" — overstated what a post-hoc margin
measurement can do, and the amendment in §6 repeated that overstatement.

**Defect 3.** Defining `μ` as the largest dose with p > 0.05 measures **detectability at the chosen
n**, not equivalence with healthy success. Section 4 acknowledged this as a caveat; it is closer to
a design flaw, and it binds harder unpaired than paired.

**Defect 4.** A constant uniform positive offset on all seven joints does not measure tolerance to
the **heterogeneous, time-varying** estimation error that continuous adaptation actually produces.
The quantity measured and the quantity `σ` describes are not the same kind of object.

**Also corrected here:** §7's amendment recorded continuous versus frozen as `p = 0.08`. That is
wrong. It compares 8/30 against **2/30**, which is the *hold* scheme's frozen arm. Against its own
frozen arm at 1/30, Fisher gives **p = 0.0257** — continuous *is* nominally better than frozen at
5%, while remaining significantly below healthy at p = 2.3×10⁻⁴. Fixed in the paper and README.
This arithmetic correction does not rescue the margin explanation.

**Status: withdrawn, not queued.** Not "run it with a wider sweep" — defect 2 is not a range
problem. If Proposition 2 is to be tested at all it needs a design where the predicted outcome is
not already in hand, and that is a new study rather than an amendment to this one.
