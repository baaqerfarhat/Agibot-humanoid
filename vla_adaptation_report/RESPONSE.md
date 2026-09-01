# Response to the 2026-09-01 review

> **The PDF in this folder is stale.** `vla_adaptation_report.tex` has been edited (bias
> subsection, statistics caveat, fault scoping, corrected estimate claim) but no LaTeX
> toolchain is installed on this machine, so `vla_adaptation_report.pdf` still shows the
> reviewed version. Read the `.tex`, and rebuild the PDF before circulating it.

Written 2026-09-01, after checking each claim against the code and the stored results rather
than accepting or rejecting it on reading. The review is strong and most of it is correct.
Two claims are wrong, and one of those is wrong in an interesting way that produced a new
result.

Status key: **verified** (checked, correct, actioned) · **accepted** (correct, not yet
actioned) · **corrected** (claim does not hold) · **scoped** (correct, but a budget decision).

---

## Verified and actioned

### 1. Equation (4) has a biased fixed point — correct, and it costs 4.9%

The code builds the FIR history from `u = a + c`, the command we believe we sent, while the
world executes `a + c + f`. So `r ≈ Mf`, independent of `f̂`, the normaliser is a constant
attenuation rather than a step size, and the fixed point is `f/(1+‖Mf‖²/ρ²)` — biased low,
exactly as the review says.

What the review does not do is size it. At the measured `‖r‖ = 0.034` with `ρ = 0.15` the
attenuation is 1/1.051, i.e. **4.9% low**; the separation test lands at 0.049 against a true
0.050, **2% low**. Same sign, same order of magnitude. So: a real defect, worth about five
percent, and **it does not explain the gain-fault result** — §8.6 does.

*Actioned:* `--law innov` drives the innovation `e = r − M f̂` to zero and normalises the
step. Fixed point is `f` exactly, and if the fault clears, `e = −M f̂ ≠ 0` so the estimate
decays instead of freezing a stale correction — the second benefit the review notes. The
legacy law remains the default so existing numbers stay reproducible; the two get compared
head-to-head rather than swapped on the strength of algebra.

### 2. Fisher's exact test is the wrong test — correct, and worse than stated

The arms run the same `(task, init)` episodes with the same policy and seeds. That is a
matched-pairs design and Fisher discards the pairing. Exact McNemar and a paired permutation
test are right; both are implemented in `openpi/mcnemar.py` with self-tests.

The review understates the damage. The runs recorded only per-arm totals, so **the pairing
cannot be recovered from any stored file** and no paired p-value can be computed
retrospectively. Per-episode logging is now in place (`per_ep`), but every headline
condition must be **rerun** before a paired p-value is claimed. Conditioning on pairs
usually raises power, so the current p-values are likely conservative — an expectation, not
a result, and it is not evidence until the rerun.

### 3. "Four of six carry the right sign and a sensible magnitude" — correct, the claim was loose

It conflated two criteria. With `β̂ = (−0.62, −0.49, −0.80, −0.20, +0.08, −0.38)` against a
true `−0.50`: **five of six have the right sign**, but **only one (dy) is within ±20%**.
Ratios: 1.24, 0.98, 1.60, 0.40, −0.16, 0.76. Fixed in the report. (The review's own count of
three, at 0.96/0.82/0.94, does not match this table — it was reading a different one.)

### 4. Call it a Cartesian action-interface bias — correct

The fault is added to the six Cartesian dimensions upstream of the OSC. It faithfully models
a miscalibrated tool frame, wrist-mount offset, or stale hand–eye calibration, and the
recovery result is genuine for that class. It is **not** a joint-level actuator fault, which
would enter below the OSC where the controller partially absorbs it and the fault-to-motion
map is state-dependent. This report's own hardware section shows that gap directly: a joint
stiffness fault is invisible in position and recoverable only in effort. Language changed
throughout; new section "What this fault is, and what it is not."

### 5. The flow-matching mapping is assumed, not measured — correct

Editing `action_out_proj/bias` is not identically an addition of `c` to the final action,
because π0.5 integrates a flow. The estimates behave as if the mapping is near-affine, but
that has not been measured. Recorded as a limitation; an external adapter after
unnormalisation would make it exact and architecture-independent, and is the recommended
construction going forward.

---

## Corrected — claims that do not hold

### 6. "The gain compensator is a first-order subtraction that undercompensates" — wrong

The implemented correction is `c = a(1/ĝ − 1)` with `ĝ = 1 + β̂`, so `a_corr = a/ĝ`. That is
the **exact** certainty-equivalent inverse gain the review recommends, already present: at
`β̂ = −0.5` it commands 2.00× and the plant executes `0.5 × 2.00a = a` exactly. Verified
symbolically and numerically.

The real failure is the **opposite** of undercompensation, and the review's framing would
have sent the work the wrong way. `1/(1+β̂)` has derivative `−4` at `β = −0.5`, so the
compensator amplifies estimation error fourfold and blows up as `ĝ → 0`. Measured: `β̂_z =
−0.797` applies **4.93×** where 2.00× is correct, a +146% overcommand on the vertical axis.

This generalises: an additive fault's compensator is *linear* in the estimate, so error
passes at unity gain; a multiplicative fault's *inverts* it, so error passes at `1/ĝ²`.
**Identification accuracy sufficient for an additive fault is not sufficient for a
multiplicative one, and the gap widens as authority is lost — precisely when repair matters
most.** Identifiability is necessary but not sufficient; compensator conditioning is a
second, independent requirement. See §8.6. The implied fix is projection (`--g-min`), not a
better regressor.

### 7. "Rotation commands do not provide enough excitation" — right conclusion, and now measured

Not a correction so much as a convergence: the review's excitation argument, Proposition 2
derived independently, and the run all agree. §8 was committed **while the test was still
computing**, so the prediction is on record ahead of its result. Outcome: translation
recovers the gain fault (−0.46 to −0.80 against −0.50) and rotation returns essentially zero
(3–30% of truth) — the exact mirror of the offset fault. The prediction transfers across
fault classes.

But identification is where it stops. Success went 13/20 frozen → 12/20 correcting the
identifiable channels → 14/20 correcting the non-identifiable ones. None distinguishable at
n = 20. **Identification confirmed, repair not.** §8.6 explains why.

---

## Scoped — correct, but budget decisions rather than fixes

### 8. The contraction spine

Three theorems (representability, estimation, contraction-to-recovery). The review is right
that this would make the paper theoretically serious, and right that the current §6 is an
empirically tuned disturbance estimator rather than contraction-based adaptive control.

The honest constraint: the recovery bound is stated for a composite VLA–OSC–robot map, and
the review itself notes contraction must hold for that composite, not the OSC alone, while
LIBERO is contact-rich and hybrid. A local rollout-tube result is achievable; a global claim
is not. Worth attempting the estimation theorem (which the excitation result already
supports empirically) before the recovery bound.

### 9. The minimum experimental program

Second backbone (OpenVLA-OFT), 50 rollouts per condition, a J-PARC baseline, a LoRA
online-adaptation baseline. Several of these require exactly the fine-tuning compute this
work claims not to need, on one shared GPU. This is a menu, not a checklist, and the
selection is a project decision. The J-PARC comparison in particular needs verifying that
the cited work exists and says what the review reports.

---

## What changed in the repository

| | |
|---|---|
| `openpi/adaptive_law.py` | `--law innov` (unbiased fixed point); `per_ep` per-episode outcomes |
| `openpi/mcnemar.py` | exact McNemar + paired permutation + Wilson intervals, self-tested |
| `openpi/adaptive_gain.py` | `--corr-dims`, `--g-min`/`--g-max` projection on the inverted gain |
| `docs/ADAPTIVE_CONTROL_VLA.md` | §8.5 prediction resolved, §8.6 compensator error amplification |
| `vla_adaptation_report.tex` | bias subsection, statistics caveat, fault scoping, corrected claim |

## What is still owed

1. Rerun the headline conditions with `per_ep` and report paired p-values. Until then no
   p-value in the report is the one the design earned.
2. `--law innov` head-to-head against legacy on a fixed condition set.
3. ~~Projection result for the gain fault~~ — **done, and it is a null.** `--g-min 0.35`
   capped the vertical overcommand at 2.86x (from 4.93x, correct 2.00x), and success went
   15/20 -> 17/20. That +10 is **not evidence**: the identical frozen condition scored
   13/20, 13/20, 15/20 across three runs, so +-10 points is free variation. The gain fault
   remains unrepaired. Two real findings fell out: `beta_hat_z = -0.797` reproduces to three
   decimals across all three runs, so the vertical estimate is *systematically* 60% too
   large and is diagnosable offline; and n=20 cannot resolve this question at all, which is
   the strongest argument yet for the paired record. See ADAPTIVE_CONTROL_VLA.md 8.7.
4. Decide the scope items above.
