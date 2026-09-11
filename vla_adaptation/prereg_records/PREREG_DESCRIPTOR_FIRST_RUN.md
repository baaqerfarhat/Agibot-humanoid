# Pre-registration — the first robot run of the descriptor form

**Written 2026-09-08, BEFORE the cell runs.** Ninth registered hypothesis. Of the previous eight,
five were refuted on their headline prediction, one tied, one untestable, one void. That base rate
is the reason this prediction is deliberately unambitious.

## What is being run

One paired cell: `libero_object`, `--joint-fault torque:3:5`, inits 37/38, 20 scenarios, two arms
(frozen and descriptor-corrected), 280 steps max. This is the **first time** the descriptor form
runs on a robot; every prior result in this project belongs to the predecessor's FIR estimator.

Model fitted on healthy episodes only, held out by task:

| | held-out RMSE |
|---|---|
| legacy per-axis FIR | 0.01272 |
| full MIMO FIR | 0.01111 |
| **descriptor ARX2 (12 states)** | **0.00469** |

`E_rank = 6` (full), condition 11.1, spectral radius 0.659, continuous embedding Hurwitz at −0.345.
So the model is well conditioned and the contraction assumption holds on healthy data.

## Prediction

**The corrected arm lands within ±3 of the frozen arm** — no significant effect in either
direction.

The reasoning is the session's own central finding, applied against my own preference: **prediction
quality and task outcome are dissociated here.** A 2.7× improvement in held-out residual RMSE is
exactly the kind of gain that has repeatedly failed to transfer — 168 same-state probes improved the
pooled error ratio to 0.905 while worsening 92 individually, and one case improved the model
objective to 10.5% of zero while physical error rose 2.69-fold. I have no reason to expect this
time differs, and predicting a rescue would ignore everything measured.

±3 is not arbitrary: it is the **measured noise floor**, from a frozen arm moving 12/20 → 15/20
across runs with nothing that could affect it.

**Falsified if** the corrected arm differs from frozen by more than 3 episodes in either direction.
A significant repair would be the first evidence the reformulation works. A significant harm would
place the descriptor form in the same failure mode as the predecessor despite fixing both of its
identified structural defects.

## What a null would and would not mean

A null means the structural fixes — the `A x` term and one consistent `E` — are **not sufficient**
for task repair on this cell. It would **not** mean they are wrong: G11 states plainly that
*"the fitted input map is not sufficiently robust to call that cell a validated efficacy
experiment"*, and that input authority remains the unresolved blocker. A null is consistent with the
input map being the limiting factor, which is what G4, G5 and the counterexample all independently
concluded.

## Limits, before the fact

- One cell, n = 20, one suite, one joint, one fault magnitude.
- Inits 37/38 differ from the stored baselines' 35/36 and 45/46, so the comparison is the **paired
  frozen arm inside this run**, not any stored number.
- `pin_rng` is False; policy sampling is not matched between arms.
- Nothing is trained. The basis is not learned; this exercises the descriptor model and composite
  law, not the geometric or learned-basis parts.

---

# RESULT — the descriptor form runs, the point estimate is favourable, and the correction saturates

**2026-09-08.** Artifact: `results/descriptor/g11_object_j3.json`. This is the **first time the
descriptor form has run on a robot** in this project.

| | |
|---|---|
| frozen | **7/20** |
| corrected | **12/20** |
| fixed / broken | **8 / 3** |
| exact McNemar | **p = 0.2266** |
| opportunities to break | 7, of which **3 broken (43%)** |
| pairing | 20 of 20 episodes matched on `(task, init)` |

## Scoring, as registered

I registered **"corrected within ±3 of frozen"**. Observed **+5**. **My prediction is refuted as
registered**, and I am not going to rescue it by appealing to the p-value, which I did not register.

But the two things must be reported together: the registered criterion says refuted, and the
**statistical test does not resolve the difference** (`p = 0.2266`). The point estimate is
favourable; the evidence is not.

## The manipulation check, which is the important part

The correction was applied — **95.2%** of steps carry a non-zero correction, so this is not an inert
intervention. But:

> **71.5% of steps sit AT the correction bound of 0.15.**

Max per dimension: `[0.150, 0.118, 0.150, 0.086, 0.150, 0.058]` — three of six channels pinned at
the limit.

**So the estimator's value is largely not what is being applied.** For nearly three quarters of the
episode the applied correction is the *bound*, not the estimate. This result therefore does **not**
isolate the descriptor form's estimation quality; it is closer to "a bounded push in roughly the
right direction helped, on this cell."

That is an order of magnitude more saturation than the predecessor showed when I investigated it
earlier: 7.9% of steps at the bound at spatial joint 5, against 71.5% here.

## What this does and does not establish

**Establishes:** the descriptor form runs end to end on a robot — identification, reference,
residual from the sent command, composite update, allocation, paired scoring. The integration works.
That was the blocker for the entire session and it is cleared.

**Does not establish:** that the reformulation repairs the fault. The difference is unresolved, the
regression rate against opportunities is 43%, and the intervention is dominated by saturation rather
than by the estimate.

## Not comparable to the stored cells

Inits 37/38 rather than 35/36 or 45/46, and `joint_fault.py` differs between the branch and the
deployment, so the injected fault may not be identical to the stored cells'. The only valid
comparison is the paired frozen arm inside this run, which is what is reported.

## The obvious next cell

Raise `--correction-limit` until saturation is no longer dominant, and re-run. Until the bound stops
binding on 71.5% of steps, this cell cannot say whether the descriptor form's *estimate* is any
good — which is the question it was built to answer.

---

# Pre-registration — the unclipped cell

**2026-09-09, before it runs.** Identical cell with `--correction-limit .25`, chosen because the
estimate's maximum is `0.196` across all dimensions, so the bound becomes inert and the applied
correction becomes the estimate. This is the first test of the descriptor form's **estimate**.

**Prediction: the unclipped cell does NOT beat the clipped 12/20, and I expect 7–12/20.**

Reasoning, and it runs against the direction I would prefer. Every prior increase of correction
authority in this project has been neutral or harmful: the predecessor's Object joint 3 went
16/20 → 2/20 given six inputs instead of three, and the authority-margin result says a correction
through an imperfect input map amplifies error by `||I − E_true pinv(E_hat)||`. Removing a bound
that was suppressing 71.9% of the estimate's magnitude exposes whatever error the estimate carries.
The clipped result may have been favourable *because* it was clipped.

**Manipulation check, registered:** saturation must fall below 5% of steps, or the cell is
inconclusive rather than negative.

**Falsified if** the unclipped cell reaches ≥ 14/20, which would mean the estimate is better than
the bound was allowing and the descriptor form's estimation is doing real work.

**Frozen arm** should reproduce 7/20 within ±3; a larger move means something other than the limit
changed and neither arm is interpretable.

---

# RESULT — the unclipped cell repairs, and BOTH tracks predicted the wrong direction

**2026-09-09.** Artifact: `results/descriptor/g11_object_j3_unclipped.json`.

| cell | frozen | corrected | fixed | broken | exact McNemar | regressions |
|---|---|---|---|---|---|---|
| clipped, limit 0.15 | 7/20 | 12/20 | 8 | 3 | 0.2266 | 3 of 7 |
| **unclipped, limit 0.25** | **7/20** | **18/20** | **11** | **0** | **0.00098** | **0 of 7** |

Frozen reproduced **exactly** at 7/20 in both, so nothing rests on a shifted baseline.

## Manipulation check

Saturation fell from **71.5% to 7.4%** of steps at the bound — a ten-fold reduction, so the applied
correction is the estimate for 92.6% of steps rather than 28.5%.

**Registered honestly: I required below 5%, and 7.4% misses that.** Some residual saturation
remains, so this is not a perfectly unclipped test. The criterion existed to stop me reading a null
as a negative; the result is not null, but the miss is recorded rather than waved through.

## Scoring — both tracks refuted, in the same direction

| track | registered | observed | verdict |
|---|---|---|---|
| Claude | "does not beat 12/20", 7–12/20 | **18/20** | **refuted** |
| Codex | "I commit to: unclipping harms" | **18/20** | **refuted** |

**This is a common-mode failure of the dual track, and it is worth more than the result.** Both
tracks reasoned from the same evidence base — the predecessor's history, in which every increase of
correction authority was neutral or catastrophic (Object joint 3 going 16/20 → 2/20 on six inputs;
the authority-margin amplification result). Independent reasoning does not protect against a shared
prior. The two tracks agreed, and were wrong together, for the same reason.

## What the result establishes

**With the bound largely inert, the descriptor form takes this cell from 7/20 to 18/20 with zero
regressions at p = 0.00098.** More authority made it *better and safer*, which is the opposite of
every result the predecessor produced. That contrast is the substantive finding: for an estimator
whose estimate is wrong, authority is dangerous; for one whose estimate is good, authority is what
lets it act. The predecessor's history taught both tracks the first lesson and neither of us
questioned whether it transferred.

## What it does not establish

- **One cell**, n = 20, one suite, one joint, one fault magnitude.
- **Not comparable to any stored cell**: inits 37/38 rather than 35/36 or 45/46, and `joint_fault.py`
  differs between branch and deployment, so the injected fault may not match. Only the paired frozen
  arm inside each run is a valid comparison, and that is what is reported.
- **The basis is not learned.** This exercises the descriptor model and the composite law. The
  geometric addition — wrench features, equivariance, the commutant constraint — is implemented and
  tested but is **not** in this run.
- `pin_rng` is False, and 7.4% saturation remains.
- A single significant cell is a reason to run more cells, not a validated method.

---

# Pre-registration — spatial joint 5, the predecessor's worst cell

**2026-09-09, before it runs.** `libero_spatial`, `torque:5:5.0`, `--correction-limit .25`,
inits 37/38, n = 20 paired. Predecessor on this cell: **frozen 12/20 → 3/20, 0 fixed, 9 broken,
p = 0.0039** — it repaired nothing and destroyed three quarters of what was working.

**Prediction: it repairs. Corrected ≥ frozen + 4, with ≤ 2 broken.**

This is a *positive* prediction, which breaks my recent pattern of registering nulls, and I am
updating on one cell — the object joint-3 result of 11 fixed and 0 broken. I state that openly
because updating hard on n = 1 is exactly the error that would be easy to make here.

The mechanism reasoning: the descriptor form fixes both structural defects measured in the
predecessor — the missing autoregressive term (44–78% of its residual) and the two mutually
inconsistent input maps (`||I − M pinv(FIR)|| = 1.896`). Neither defect is specific to the object
suite, so if they are what caused the joint-5 harm, the fix should transfer.

**Falsified if** it breaks ≥ 3 episodes, which would mean the object result does not generalise and
the descriptor form shares the predecessor's failure mode on the cell that matters most.

**Manipulation check:** saturation must stay below ~10% of steps, or the cell is testing the bound
rather than the estimate. **Frozen** should land near 12/20 within the measured ±3.

---

# RESULT — spatial joint 5: the predecessor's worst cell, reversed

**2026-09-09.** Artifact: `results/descriptor/g14_spatial_j5.json`.

| | frozen | corrected | fixed | broken | p | regressions |
|---|---|---|---|---|---|---|
| **predecessor** (FIR, `corr-dims 0,1,2`) | 12/20 | **3/20** | **0** | **9** | 0.0039 | **9 of 12 = 75%** |
| **descriptor form** | 14/20 | **19/20** | **6** | **1** | 0.125 | **1 of 14 = 7%** |

Manipulation check: **1.4%** of steps at the bound, well inside the ≤10% I registered, so this
tests the estimate rather than the limit. Frozen at 14/20 against the predecessor's 12/20, inside
the measured ±3 noise floor.

## Scoring

| | registered | observed | verdict |
|---|---|---|---|
| Claude | corrected ≥ frozen + 4, ≤ 2 broken | **+5, 1 broken** | **satisfied** |
| Codex | "I commit to: repairs" | repairs | **satisfied** |

**But this cell alone is not significant: `p = 0.125`.** Six fixed against one broken over seven
discordant pairs does not resolve at n = 20. The point estimate is favourable and the prediction
held; the individual cell does not carry statistical weight on its own.

## The contrast is what carries weight

Same robot, same fault, same joint, same suite:

- predecessor: **0 fixed, 9 broken**, destroying 75% of what was working
- descriptor form: **6 fixed, 1 broken**, 7% regression rate

Across both descriptor cells so far: **17 fixed, 1 broken** over 21 opportunities.

## A common-mode caveat on the agreement

Both tracks predicted "repairs" and both were right, but **we updated on the same datum** — the
object joint-3 cell. The second track explicitly audited its evidence base first, as required after
the shared error on the previous cell, and concluded it was reasoning from different primary
evidence. That is better practice than last time. It is still true that agreement between two tracks
updating on one shared new result is weak corroboration, and the honest reading is that both
predictions rested on the same single prior cell.

## What this does and does not establish

**Establishes:** on the cell that motivated this entire direction, the descriptor form repairs where
the predecessor destroyed, with a regression rate of 7% against 75%. Two cells now, two suites, two
joints, consistent direction.

**Does not establish:** significance on this cell (`p = 0.125`); anything about other joints,
suites, fault magnitudes or fault types; or anything about the geometric addition, which remains
implemented, tested, and **not exercised in either run**. The basis is still not learned.

Two favourable cells are a reason to run a proper sweep, not a validated method.

---

# Pre-registration — the seven-joint sweep

**2026-09-09, before it runs.** All seven `libero_spatial` joints, `torque:J:5.0`,
`--correction-limit .25`, inits 37/38, n = 20 paired, descriptor form. The predecessor's map on the
same joints (different inits) was:

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| predecessor fixed | 0 | 1 | 2 | 8 | 1 | 0 | 8 |
| predecessor broken | 0 | 1 | 1 | 1 | 4 | 9 | 4 |

Net **+7 at the elbow and −7 across the other six combined**, 20 broken in total.

**Predictions.**

1. **Net fixed-minus-broken is positive across the sweep**, in contrast to the predecessor's zero.
2. **Total broken across all seven cells ≤ 8**, against the predecessor's 20.
3. **No cell shows the predecessor's catastrophic pattern** — no cell with zero fixed and ≥ 5 broken.

Registered from two favourable cells (17 fixed, 1 broken over 21 opportunities), which is a thin
base, and stated as such. The predecessor's own map is the reason predictions 2 and 3 are worth
making: it had one cell at 0-fixed/9-broken and another at 1-fixed/4-broken.

**Falsified if** total broken exceeds 12, or any cell reproduces the zero-fixed/≥5-broken pattern.

**Manipulation check per cell:** saturation below 10%, else that cell tests the bound.

**Limits.** Inits 37/38 differ from the predecessor's 45/46, and `joint_fault.py` differs between
branch and deployment, so this is **not** a controlled comparison against the stored map — each
cell's own paired frozen arm is the comparison. The predecessor numbers are context, not a baseline.
