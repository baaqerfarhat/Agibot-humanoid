# Pre-registration — which *directions* of action-interface error does a frozen VLA tolerate?

**Written 2026-09-07, BEFORE any cell runs.** Sixth registered hypothesis in this audit line. Of the
previous five: three refuted on their headline prediction, one tied, one had its discriminator
declared untestable before it ran. That is the honest prior, and it is why this design is built so
that the *headline contrast cannot be produced by the confound that killed the others*.

---

## 1. What changed, and why this is a different kind of question

Every prior study in this line asked about the **estimator**. All of them narrowed rather than
established: five observers tie (`PREREG_MATCHED_OBSERVER.md`), removing the estimator bias exactly
as designed did not help (`PREREG_ALOHA_INNOV_LAW.md`), the normaliser hypothesis was retracted.

Two independent tracks then converged on the same diagnosis: the accuracy-versus-repair framing is
**occupied prior art** — identification-for-control since Hjalmarsson & Gevers 1996, value-aware
model learning, and most directly J-PARC, which already reports an oracle constrained-IK baseline
with lower local error and no consistent task benefit. And I had been over-claiming the evidence:
there is **one unresolved descriptive reversal in this project, not three**.

Both tracks then independently named the same replacement shape: a **prespecified measurement that
predicts an outcome on conditions it was not fit to**. This is the cheapest instance of it.

**The question is about the frozen policy, not about my method.** With no correction applied, the
executed action is `a + f`, so an injected `--fault-vec` **is** the command error. No estimator, no
`M`, no calibration, no correction, no sign convention. What is measured is a property of π0.5:

> Which directions of action-interface error does a frozen VLA tolerate, and can that be predicted
> from a measurement made on other directions?

That is a question about VLAs. Nothing in it depends on the adaptive law being any good.

## 2. The design that makes the confound structurally impossible

The defect in every "accuracy does not predict task success" result — including this project's own —
is that the compared conditions differ in *how much* error they carry as well as *where* it points,
so an aggregate norm and a direction are confounded.

**Permutations of a non-uniform vector remove that confound by construction.** Take the multiset
{3, 1, 1} over the three translation axes:

| direction | L1 | L2 | L∞ |
|---|---|---|---|
| `(3,1,1)·u` | 5u | 3.3166u | 3u |
| `(1,3,1)·u` | 5u | 3.3166u | 3u |
| `(1,1,3)·u` | 5u | 3.3166u | 3u |

Identical under **every** norm — mean absolute error, RMSE and worst-coordinate error are the same
number for all three. Verified numerically before writing this. Any difference in task success
between them is therefore attributable to direction alone, with no statistical adjustment required.
Rotation and gripper channels are held at zero throughout: only coordinates of the same physical
kind are permuted.

## 3. Two stages, so the prediction is out of sample

**Stage 1 — measure the per-axis margin.** Single-axis errors only: `fault-vec = s·e_k` for
k ∈ {x, y, z}. Frozen policy, no correction. `m_k` is the largest `s` on axis k whose success is
still indistinguishable from healthy. Nine cells (3 axes × 3 magnitudes), n = 20.

**Stage 2 — predict, then test on mixed directions.** From stage 1 alone, compute for each
permutation

  R(e) = max_k |e_k| / m_k

and **rank the three permutations before running them**. Stage 1 contains no mixed-direction cell,
so the ranking is out of sample in direction. Four cells, n = 24: the three permutations at radius
ρ, plus the predicted-most-tolerant permutation at **2ρ**.

**The inversion contrast is `sensitive at ρ` versus `tolerant at 2ρ`.** The ρ arm carries exactly
half the error of the 2ρ arm under L1, L2 *and* L∞. If it nonetheless scores worse, then no
aggregate error norm orders these conditions, and the demonstration needs no significance test to
be interpretable — it is a sign, not a size.

## 4. Predictions, registered in advance

1. **The three axes have materially different margins**, `max_k m_k / min_k m_k ≥ 1.5`. If the
   margins come out near-equal the permutations are near-equivalent, prediction 2 is untestable, and
   **it will be reported as untestable, not as refuted** — the same call made in
   `PREREG_JOINT_MAP.md` when its discriminator collapsed to a 2.5% spread.
2. **R orders the three permutations correctly.** Spearman between predicted R rank and observed
   success rank is +1 or, with one inversion, +0.5. Reported whatever it is, with n = 3 stated as
   the severe limitation it is.
3. **The inversion occurs**: `sensitive at ρ` scores strictly worse than `tolerant at 2ρ`, despite
   being twice as accurate under all three norms.
4. **Radial monotonicity holds within an axis**: on each single axis, success is non-increasing in
   `s`. If it is not, something other than error magnitude is driving stage 1 and the margins are
   not margins.

**Falsified if** R misorders the permutations, or the inversion fails to invert while the margins
are genuinely unequal. Prediction 3 failing while 2 holds is the interesting middle case: direction
matters and orders the conditions, but not strongly enough to beat a 2× magnitude difference. That
is a *quantitative* result — direction is worth less than a factor of two here — and is worth
reporting as such rather than spun either way.

## 5. Analysis, fixed in advance

- Frozen arm only for the geometry claim. Each cell also yields an adaptive arm; it is **not** part
  of this hypothesis and will not be used to argue it.
- Paired exact McNemar via **both** `openpi/mcnemar.py` and `openpi/mcnemar_crosscheck.py`.
- Every arm reported against the shared healthy control and against opportunities.
- ρ, the multiset {3,1,1}, the axis set and the R formula are all fixed here and not revisited.
- **One declared amplitude adjustment is permitted** if the pilot puts every arm at floor or
  ceiling, and both attempts are kept and reported. This is a resource decision, not a result.
- Verification that L1, L2 and L∞ are numerically identical across arms is printed with the results,
  so a reader can check the central design claim without trusting the text.

## 6. Stated limits, before the fact

- One suite, one backbone, one fault family. This characterises π0.5 on `libero_spatial`, not VLAs.
- `m_k` is measured with fault truth. It is a **privileged diagnostic**, not a deployable selector
  for unknown faults — that stronger claim needs an independently validated way to estimate the
  residual from accessible observations and is explicitly not attempted.
- n = 20–24 per cell against ±10 points of documented free variation. The *ordering* is the claim.
- "Indistinguishable from healthy at n = 20" is a weak criterion and is not equivalence.
- Three permutations is n = 3 for prediction 2. A correct ordering of three items happens by chance
  one time in six. **Prediction 3, the inversion, is the load-bearing one**; prediction 2 alone
  would not carry a claim.
- This does not show that direction matters *more* than magnitude in general, only whether it
  outweighs a specific factor of two on this task.

## 7. Cost

Pilot 3 cells, stage 1 nine cells, stage 2 four cells, one healthy control — about 320 episodes.
LIBERO runs 20 episodes in roughly five minutes on one RTX 6000 Ada, so this is under two GPU-hours,
on hardware already in use for the joint-map sweep.

---

## 8. Amendment, before any cell runs — episode yield per cell

`--estimate-only` sets `apply_corr=False` (`openpi/adaptive_law.py:1104`), so the correction is
never applied and **both** arms execute `a + f`. Each cell therefore yields `2 × episodes`
independent frozen realisations of the same condition, not one arm's worth.

The two arms are **reported separately and pooled only if they agree**. A systematic difference
between them would mean `--estimate-only` is not clean — the estimator would be influencing the
executed action through some path I have not accounted for — and that would be the finding, not a
nuisance to average away. Recorded here before running so the pooling decision is not made after
seeing which way it helps.

## 9. Amendment, before any cell runs — competing predictors, all fixed now

The independent review's standing objection to any measurement rule is that it must beat a *simple*
baseline, not merely beat chance. Three competitors are therefore registered here, with their
numeric inputs already measured from near-healthy telemetry on disk and frozen:

**Design precondition, verified.** `OUT = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`
(`openpi/adaptive_law.py:27`). The three translation dimensions carry an **identical** scale, so a
permutation within them is clean in action units and in measurement units alike. Had they differed,
the L1/L2/L∞ identity of §2 would have been an artifact of raw coordinates rather than a physical
statement, and the design would have been void. It is not.

| predictor | per-axis denominator (x, y, z) | predicted worst permutation |
|---|---|---|
| **N — aggregate norm** | none; L1, L2, L∞ identical across arms | **no difference** (the null) |
| **U-ep — whole-episode usage** | mean\|a_k\| = 0.360, 0.279, 0.462 | 3u on **y** |
| **U-app — approach-window usage** | mean\|a_k\| over the 15 steps before gripper closure = 0.325, 0.132, 0.781 | 3u on **y** |
| **M — measured margin** | `m_x, m_y, m_z` from stage 1 | computed after stage 1, before stage 2 |

Note the usage measure is **1.66× anisotropic over a whole episode and 5.9× during final approach** —
the same quantity, phase-aware or not, differs by more than three-fold in how sharply it
discriminates. That is why a phase-aware measure is worth separating from a naive one.

**U-ep and U-app order the axes identically** (y < x < z), so this design can separate *usage* from
*margin*, and both from the *null*, but it **cannot** separate phase-aware usage from whole-episode
usage on ordering alone. Stated now rather than discovered later.

**How each outcome reads:**

- All three permutations score alike → **N survives**, direction does not matter at this radius, and
  the whole line is finished. This is a real possible outcome and is not spun as anything else.
- They differ and **M** orders them but **U** does not → the margin measurement carries information
  that the policy's own action statistics do not. This is the result worth having.
- They differ and **U** orders them → direction matters, but a baseline anyone would try first is
  sufficient, and no new instrument is justified. Reported as such.
- They differ and neither orders them → direction matters and I cannot predict it. An honest
  negative on the predictive claim, and still a falsification of **N**.

The margin predictor M is not permitted to be reformulated after seeing stage 2. Its formula
`R(e) = max_k |e_k| / m_k` is fixed in §3 and the only free input is `m` from stage 1.

---

# VOID — killed by adversarial review before any cell ran

**2026-09-07.** The design was dispatched to an independent adversarial review specifically to be
attacked before it consumed GPU time. Verdict: **VOID as registered**. The batch was not launched.
No rollout was spent. Recorded in full because the reasons are more useful than the design was.

## 1. Controller clipping destroys the norm-matching — this alone is fatal

The design's entire justification is that permutations of `{3,1,1}` carry identical L1, L2 and L∞,
so direction is separated from magnitude *by construction*. **It is not.**

robosuite 1.4.1's OSC controller calls `scale_action`, which **clips each input to [−1, 1] before
scaling**. The perturbation that actually reaches the controller target is

  δ(a, f) = 0.05 × [clip(a + f, −1, 1) − clip(a, −1, 1)],  not  0.05 × f.

The permutation is applied to `f`. It is **not** applied to `a`, and the policy's commands are
strongly asymmetric across axes — many sit near saturation. So the three arms have different
effective errors.

Stored counterexample, from this repository's own calibration data
(`results/phase05/error_signal_so3.json`, healthy command `(0.977, −0.184, −0.083)`, illustrative
`u = 0.05`):

| injected | effective residual after clipping | effective L2 |
|---|---|---|
| `(0.15, 0.05, 0.05)` | `(0.0229, 0.05, 0.05)` | **0.0743** |
| `(0.05, 0.15, 0.05)` | `(0.0229, 0.15, 0.05)` | **0.1598** |

Both were supposed to be L2 = 0.1658. One is **less than half** the other. Across all 285 healthy
commands the three permutations are altered on 26, 5 and 20 steps respectively.

**My §8 "verified precondition" checked the wrong thing.** I checked `OUT`, which normalises
*measured motion*, and concluded the design was scale-clean. `OUT` has nothing to do with
downstream saturation. Worse, §5 promised to print the injected vector's norms "so a reader can
check the central design claim without trusting the text" — and printing the *injected* norms
cannot detect this defect at all. The verification I built in would have certified a broken design.

## 2. The self-contradiction

`R(e) = ‖diag(1/m) e‖∞` **is itself an aggregate weighted norm.** So the headline — "no aggregate
error norm orders these conditions" — is contradicted by the paper's own proposed predictor. What
permutations match is *permutation-invariant* norms. That is a much narrower and much less
interesting statement, and it is the only one any outcome could have supported.

## 3. "Sign, not size" was a dodge, and the arithmetic says so

§3 claimed the inversion "needs no significance test to be interpretable." Under a true null of
equal success at 0.5, two independent n = 40 counts show the requested strict ordering **45.55% of
the time**. Exact McNemar power at n = 40 is **9.8%** for a 10-point difference and 35.4% for 20
points. Reaching 80% power needs a 22–39 point difference depending on discordance. The design
could not have resolved anything it claimed to.

Stage 1 fails the same way in the other direction: 40/40 versus 35/40 gives p = 0.0625, so a
**12.5-point loss** would have been recorded as "indistinguishable from healthy" and folded into
`m_k`. The margin would have been a noisy function of power, not a tolerance.

## 4. An internal contradiction that makes falsification impossible

With smallest and largest margins `a` and `c`, `R_sensitive / R_tolerant = min(3, c/a)`. So the
model predicts inversion against the 2ρ arm **only when `c/a > 2`**. But §4's precondition for
proceeding was `c/a ≥ 1.5`. For any ratio in `[1.5, 2)` the precondition passes while the predictor
itself ranks the 2ρ arm as *riskier* — so a failure to invert could not falsify the model, contrary
to §4's falsification clause.

## 5. Prior art is far heavier than §1 claimed, including from the predecessor

- **Guo et al., ICLR 2026** — constant additive action bias on π₀/LIBERO, `Ât = At + 0.03·1`,
  success 96% → 23%; and searches for harmful directions within an action-noise norm budget.
- **Lee et al., AAAI 2020** — "Action Dimension Decomposition" explicitly finds greater
  vulnerability to *vertical* than *horizontal* action perturbations under L2-constrained attacks.
- **J-PARC (Jo et al., 2026)** — already explains heterogeneous joint vulnerability in VLAs using
  **usage-weighted end-effector sensitivity**, combining joint usage with translational Jacobian
  sensitivity. That is essentially the `U-app` baseline registered in §9 as a competitor: the
  closest prior work already publishes the explanation this design registered as its own baseline.

"Direction matters" and "a smaller perturbation can be worse than a larger one" cannot carry
novelty. §1 asserted the accuracy framing was occupied but treated the direction framing as open.
It is not.

## 6. Provenance failure in my own registration

§9 registered whole-episode usage denominators as `(0.360, 0.279, 0.462)` and named the artifact
they came from. Recomputed from that artifact's 285 nominal actions: **`(0.351, 0.247, 0.476)`**.
The registered numbers cannot be reconstructed from the source I cited, because I computed them
from a different file (near-healthy joint-fault telemetry) than the one the registration names.
Registering a number whose provenance does not check out is precisely the defect this audit line
exists to catch, and I introduced it.

Also in §9: the Spearman convention has the wrong sign (risk and success should be *negatively*
associated), and "perfect or one-adjacent-swap" accepts **3 of 6** permutations, not the 1-in-6
chance rate §6 claimed.

## 7. It was never a property of π₀.₅

§1's framing — "a question about VLAs, not about my method" — does not survive. Success alone
cannot separate task geometry from controller clipping and actuator saturation, and the stored
sensitivity matrix already shows the translation axes are not equivalent in the plant
(`openloop_so3.json` diagonal ≈ 0.297, 0.272, 0.126, with x responding −0.112 to a z fault). Equal
`OUT` does not imply equal realised displacement. The design measures a policy-controller-task
combination. Only positive-axis faults were registered, so even "axis tolerance" would have meant
positive-direction tolerance, with sign symmetry untested.

## 8. Two concrete defects that were fixed rather than just recorded

- **`mcnemar.py` and `mcnemar_crosscheck.py` key rows by `(task, init)` only**, so pooling the two
  `--estimate-only` blocks as §8 proposed would have silently dropped one replicate per key —
  20 pairs scored while `successes` and `n` still reported 40. **No stored result is affected**
  (315 arms scanned, every one has a single episode per key), so this was latent, not active. Both
  tools now raise instead. The crosscheck shared the input-key convention of the tool it checks,
  so the defect would have survived the crosscheck — which is the more useful lesson.
- **`run_geometry.sh` skips an existing cell** on filename, arm count and episode count alone. It
  does not verify the fault vector, so reusing a cell name after the permitted amplitude adjustment
  would silently reuse the old intervention.
- The stated cost was wrong: at the runner's 40 episodes per cell the budget is **680** episodes,
  not "about 320".

## What survives

`--estimate-only` was checked and is clean: no estimator-to-action, estimator-to-RNG or
estimator-to-termination path, verified across all 5,872 stored probe rows. That was the one
premise of this design that held.

Nothing else does. The correct conclusion is that this line — action-error direction versus
magnitude — is **occupied prior art reached through a broken instrument**, and no amount of
redesign inside this repository's controller stack makes it a contribution.

**Correction to the paragraph above, added after independent review.** The verdict was VOID **as
registered**. Saying no redesign could ever contribute goes beyond what invalidating this design
established, and I overstated it. What is established is narrower and still decisive for the near
term: adaptive augmentation of a pretrained policy, directional action vulnerability, and another
local-error versus task-success mismatch are each insufficient novelty **on their own**, because
Cheng et al. and J-PARC already occupy them. A better quantitative diagnostic is not ruled out. It
is simply not reachable by a short extension of this implementation, and any successor must define
its estimand *after* the controller's clipping and scaling rather than in raw action units.
