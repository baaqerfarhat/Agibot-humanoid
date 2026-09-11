# Pre-registration — does the proposed update law beat a matched observer?

**Written 2026-09-07, BEFORE the estimators are implemented and BEFORE any run.** The
implementation (`openpi/adaptive_law.py`) does not yet contain `dob`, `integral_calibrated`,
`rls` or `kalman`; the LIBERO stack was still downloading its checkpoint when this was
committed. Nothing below has been executed.

This is the **third** registered hypothesis in this line. The first two
(`PREREG_ALOHA_NORM_CHANNELS.md`, `PREREG_ALOHA_INNOV_LAW.md`) were both refuted on their
headline prediction. That base rate is the honest prior for this one.

---

## 1. The question, and why the paper does not currently answer it

Both independent reviews of this project converged on the same sharpest objection:

> A competent **calibrated** disturbance observer should work with any upstream command
> generator. What did we learn beyond that, and why is *this* observer the relevant one?

The paper's §`sec:baselines` answers a weaker question. Its baseline is integral action on the
**raw** motion error, `f̂ += k_i (y_t − a_t)`, which assumes achieved motion equals commanded
action while the measured plant DC gain is 0.23. It therefore chases a phantom 2.5× the size of
the fault. Beating it (35% vs 95%) demonstrates that **having a plant model** helps. It does
not demonstrate that **this update law** helps, because no comparator in the paper has the
plant model.

## 2. Arms

All arms receive **identical information**: the same FIR plant, the same `M`, the same healthy
bias `b`, the same correction mask, the same clip, the same update rate, the same residual.
Only the update rule differs.

| arm | update | what it isolates |
|---|---|---|
| **frozen** | none | the floor |
| **P** proposed (legacy, default) | `f̂ ← Π(f̂ + γ(𝟙[‖r‖>δ]·(M⁻¹r−b)/(1+‖r‖²/ρ²) − f̂))` | the paper's law |
| **D** dob | `f̂ ← (1−α)f̂ + α(M⁻¹r − b)` | **P minus the robustness triple** — what deadzone + normaliser buy |
| **I** integral_calibrated | `f̂ ← f̂ + k_i(M⁻¹r − b)` | the **fair** version of the paper's own baseline |
| **R** rls | RLS on `r ≈ Mf`, forgetting `λ` | a standard uncertainty-weighted estimator |
| **K** kalman | random-walk KF, `Q`/`R` from the healthy residual | the standard optimal-under-assumptions estimator |
| **O** oracle | `f̂ = f` exactly | the ceiling |

**Cell**: `libero_spatial`, π0.5, rotation `+0.10` (frozen 0/20, large headroom, and one of the
nine transfer cells the paper now leads with). n = 20 paired, matched `(task, init)`.

**Tuning budget is matched and declared in advance.** Each of D, I, R, K gets its single free
constant (`α`, `k_i`, `λ`, `Q/R` ratio) swept over the *same* five-point logarithmic grid used
for the paper's existing integral sweep, on **development** initial states disjoint from the
evaluation set, and is then **frozen**. The proposed law P is evaluated at its published
constants with **no** re-tuning. This deliberately favours the comparators; if P still wins
that is meaningful, and if it does not the comparison is not confounded by tuning effort.

## 3. Predictions, registered in advance

1. **Every calibrated comparator beats the paper's raw-error baseline (35%).** If any of D, I,
   R, K fails to clear 35%, the implementation is wrong and the study is void. This is the
   sanity gate.
2. **D (dob) reaches within 3/20 of P.** i.e. the deadzone and normaliser are worth less than
   three episodes on this cell. Stated because §`app:abl` already shows the gain is flat over a
   16× range and the deadzone is flat until it reaches the residual scale — so the robustness
   triple is not doing much work here.
3. **At least one of R (rls) or K (kalman) equals or beats P.** Registered as the *likely*
   outcome, not the hoped-for one.
4. **No comparator beats O (oracle).** Sanity.

**What each outcome means, fixed in advance so it cannot be reinterpreted afterwards:**

- **If P clearly beats all of D, I, R, K** → the update law earns its place, and that is a
  methodological result the paper does not currently have.
- **If P ties them** → the contribution is the *interface and the calibration*, not the law.
  The paper should say so, drop "the machinery earns its place" framing, and lead with the
  nine-cell transfer result. **This is the outcome I expect.**
- **If P loses to any of them** → report it and adopt the better estimator. The paper's claims
  about repair are unaffected; only the attribution changes.

## 4. Analysis, fixed in advance

- Paired exact McNemar on matched `(task, init)`, via **both** `openpi/mcnemar.py` and
  `openpi/mcnemar_crosscheck.py`.
- Regression rate reported **against opportunities**, per `openpi/regression_rate.py`, not
  against all episodes.
- Estimation error reported as **MAE, RMSE and worst-coordinate** together. Reporting only the
  favourable metric is the error made in `PREREG_ALOHA_INNOV_LAW.md` and corrected there.
- Telemetry recorded for every arm.
- n = 20 **cannot** resolve small differences. A tie will be reported as "not resolved at
  n = 20", never as "equivalent". Excluding a 10-point difference at 80% power needs ~186
  pairs; that is out of budget and the limitation is stated up front rather than discovered
  afterwards.

## 5. Stated limits

- One cell, one suite, one backbone, one fault family. This tests attribution on the cell the
  paper leads with; it does not generalise to the other eight.
- Default-flag arithmetic must be bit-identical to `HEAD` before any arm runs, verified by
  replay against git.
- The comparators are given a tuning advantage (swept and frozen) that P is not. A P win under
  that handicap is strong; a P loss is expected and uninformative about tuning.

---

# AMENDMENT, before any arm runs

**Added 2026-09-07, after the estimators were implemented and their synthetic behaviour
characterised, but BEFORE any policy rollout.** Recorded as an amendment rather than an edit,
so the original predictions stand as written.

## Prediction 1 is partly void, for a structural reason

Registered: *"every calibrated comparator beats the raw-error baseline (35%); if any of D, I,
R, K fails, the implementation is wrong and the study is void."*

The synthetic characterisation shows **arm I (`integral_calibrated`) has no fixed point at
all**. It drifts at `k_i·f` per update and rails at the projection bound by update 150. This is
not an implementation error. It is structural: the residual is `r ≈ Mf` by construction and
does **not** shrink as `f̂` converges (the paper's own design — the residual estimates the
*total* fault, not the remaining error), so a pure integrator never stops integrating.

**Consequence.** Arm I will be run and reported, and it is expected to fail. Its failure does
**not** void the study. It is a finding in its own right: a calibrated integrator is not merely
a weaker estimator on this residual, it is inapplicable to it, which is a second and deeper
reason the paper's raw-error baseline was never a fair comparison. Prediction 1 is amended to
apply to **D, R and K only**.

## Synthetic fixed points and convergence, recorded before the experiment

Matched synthetic fault, sustained-5%-error criterion:

| arm | fixed point | updates to within 5% of `f` |
|---|---|---|
| legacy (P) | `0.61·f` (biased; `0.627·f` at `b=0`) | never reaches `f` |
| innov | `f` | 328 |
| dob (D) | `f` | 299 |
| rls (R) | `f` | 161 |
| kalman (K) | `f` | 1 |
| integral_calibrated (I) | none — rails by update 150 | — |

## A refinement to prediction 3, stated in advance

Registered: *"at least one of R or K equals or beats P."*

The synthetic result adds a caveat that should be read alongside it: **at matched steady-state
variance, RLS and Kalman take 299 updates, exactly equal to the EMA.** Their startup advantage
is bought with variance and disappears once variance is matched. So if R or K beats P on the
real cell, the mechanism is most likely the **bias** (P converges to 0.61·f synthetically, and
was measured at 93.5% of truth on ALOHA), not superior tracking. That distinction should be
checked in the telemetry rather than assumed, and is recorded now so it cannot be constructed
after the fact.

---

# AMENDMENT 2 — the tiebreak rule, fixed mid-sweep

**Added 2026-09-07, with 10 of 20 development points complete and the Kalman and integral
points not yet run.** The preregistration says each comparator's constant is "swept ... and is
then frozen" but does not say how to break a tie, and the sweep has produced several.

Development results so far (10 episodes each, inits 20–21, disjoint from the evaluation band):

| dob α | | rls λ | |
|---|---|---|---|
| 0.02 | 9/10 | 0.90 | 9/10 |
| 0.04 | 9/10 | 0.95 | 9/10 |
| 0.08 | 9/10 | 0.99 | 8/10 |
| 0.16 | 8/10 | 0.995 | 8/10 |
| 0.32 | 9/10 | 0.999 | 9/10 |

**Rule, fixed now and applied to all four comparators including the two not yet swept:** among
points tied at the best development score, take the one **closest to the proposed law's own
setting or to the estimator's conventional default**, in that order of preference. Rationale:
where the sweep cannot distinguish, the comparison should isolate the *update rule* rather than
the *constant*, and a comparator handed an unusual constant purely because it won a coin-flip
on ten episodes would be a worse test, not a better one.

Applied:

- **dob α = 0.08** — tied at 9/10 with 0.02, 0.04 and 0.32; 0.08 is the proposed law's own γ,
  so the arms differ only in the update rule.
- **rls λ = 0.95** — tied at 9/10 with 0.90 and 0.999; 0.95 is the conventional forgetting
  factor and sits inside the plateau rather than at its edge.
- kalman and integral constants to be selected by the same rule once their points complete.

This is recorded before the remaining points are seen so the rule cannot be chosen to suit
them.

**A note on what the sweep already shows.** Ten development points spanning two estimator
families and a 16× range of smoothing constant all land at 8–9/10. Nothing here distinguishes
the update rules. That is consistent with the paper's own constants ablation, whose two best
rows (deadzone 0 → 19/20; ρ = 0.50 → 20/20) are the two settings that effectively disable the
robustness modifications, and whose worst row (ρ = 0.05 → 14/20, 2 broken) is the one that
strengthens them. Registered prediction 2 — that the plain DOB lands within 3/20 of the
proposed law — currently looks likely to hold. The evaluation arms on held-out inits remain
the test.

---

# CORRECTION — the constant selection was development-informed, not advance-fixed

**Added 2026-09-07 after an independent verification pass, BEFORE the evaluation arms ran.**
Amendment 2 above overstates the strength of its own procedure. The corrections below narrow
it; the original text is left standing so the overstatement is visible.

## What amendment 2 claimed, and what is actually true

It said the tiebreak rule was "recorded before the remaining points are seen so the rule
cannot be chosen to suit them." That is true for **Kalman and integral only**. Amendment 2 was
written with all ten DOB and RLS points complete, and it prints their scores in its own table.
**For DOB and RLS the tie rule was devised after the ties were visible.** Choosing a constant
on development data is legitimate tuning; inventing the tie-resolution procedure after seeing
the ties is not an advance-fixed selection, and calling it one was wrong.

## The Kalman selection does not follow the stated rule

Amendment 2's rule is "closest to the proposed law's own setting, else the estimator's
conventional default," and it explicitly **defers** the Kalman choice. The value actually
frozen, `q = 1e-5`, was selected as the **median of the three points tied at 10/10** — a
different criterion, introduced after those points were visible. It is also not the
implementation's default, which is an anisotropic `Q = γ²/(1−γ)·M⁻¹RM⁻ᵀ`, not a scalar.

So `kf q = 1e-5` is a **post-hoc selection under a rule change**, and is labelled as such.

## Two further inaccuracies in the committed record

- The sweep commit message says every point lands at "8-10 of 10". **`kf_1e-2` scores 7/10**,
  and the Kalman plateau is not continuous: `1e-4` sits between the perfect points at 9/10.
- `results/observers/sweep/int_0.001.json` was committed while still mid-run and contains only
  the frozen arm. The committed sweep is **15 complete points plus one partial**, not 16.
- "development inits 20–21" is wrong for n = 10; those runs use **init 20 only**. Inits 20 and
  45–46 are still disjoint, so the episode split itself holds.

## How the evaluation is labelled as a result

The comparator constants are **development-informed**, with the Kalman value additionally
selected under a rule changed after the fact. The evaluation remains worth running and is
still informative — the comparators are tuned and the proposed law is not, which biases the
comparison *against* the paper's law, so a P win would survive this caveat. **But this study
cannot be described as a fully preregistered selection procedure, and it will not be.**

## The larger finding this verification surfaced

`openpi/openloop_id.py` measures `M` at task 0, **init 45**, and every headline
`libero_spatial` cell — including this study's evaluation band — evaluates inits {45, 46}. The
sensitivity matrix is calibrated on one of the twenty episodes it is scored on. This is a
property of the paper's calibration, not of this comparison, and it applies equally to all
arms here, so it does not distort the P-versus-comparator contrast. It is recorded in commit
`a10d5f3`.

---

# RESULT — every matched observer works, and nothing is resolved at n=20

**Added 2026-09-07 after all six arms ran.** Artifacts: `results/observers/eval/*.json`.
Scored by `openpi/score_observers.py`, which was committed before the arms ran and hardened
before they were scored. Both McNemar implementations agree on every p-value.

`libero_spatial`, π0.5, rotation +0.10, n = 20 paired, evaluation inits 45–46. Comparator
constants frozen from the development sweep; the proposed law at its published constants with
no re-tuning.

| arm | frozen → corrected | fixed | broken | vs frozen | MAE | worst-coord MAE |
|---|---|---|---|---|---|---|
| **P** proposed | 0/20 → **19/20** | 19 | 0 | 3.8e-06 | 0.0286 | 0.0503 |
| **D** plain DOB | 0/20 → 17/20 | 17 | 0 | 1.5e-05 | **0.0204** | **0.0411** |
| **R** RLS | 0/20 → 17/20 | 17 | 0 | 1.5e-05 | 0.0230 | 0.0474 |
| **K** Kalman | 1/20 → 18/20 | 17 | 0 | 1.5e-05 | 0.0217 | 0.0441 |
| **I** calibrated integral | 1/20 → 15/20 | 14 | 0 | 1.2e-04 | 0.0411 | 0.0500 |
| **O** oracle | 0/20 → **20/20** | 20 | 0 | 1.9e-06 | n/a (static) | n/a |

**Head-to-head, paired on `(task, init)`:** P vs D 3–1 (p = 0.625); P vs R 3–1 (p = 0.625);
P vs K 2–1 (p = 1.0); P vs I 5–1 (p = 0.219); P vs O 0–1 (p = 1.0).
**Nothing is resolved at n = 20.**

## Scoring the registered predictions

| prediction | outcome | verdict |
|---|---|---|
| 1 (amended to D, R, K) — all clear 35% | 17, 17, 18 / 20 | **satisfied** |
| 2 — D within 3/20 of P | within 2/20 | **satisfied** |
| 3 — R or K equals or beats P | neither does | **not satisfied** |
| 4 — nothing beats the oracle | O highest at 20/20 | **satisfied** |
| amendment — arm I fails/rails | I reaches 15/20 | **wrong** |

The registered interpretation branch that occurred is the one marked *"this is the outcome I
expect"*: **P ties them.**

## What this does and does not establish

**Establishes.** Every estimator given the same plant model, `M`, bias, mask and clip repairs
this cell: 15–19 of 20 against a frozen 0/20, all with zero regressions. The plain DOB — the
proposed law with the deadzone, normaliser and projection removed — reaches 17/20. A
calibrated integrator reaches 15/20. **The information the plant model carries is what
produces repair; the update rule applied to it is worth at most a couple of episodes here.**

**Does not establish.** That the comparators equal the proposed law. P is numerically highest
at 19/20 and the ordering P > K > D = R > I is consistent with a real if small advantage. It
is simply not resolvable here — see below.

**Does not refute the paper.** P survives a materially fairer test than the one in
§`sec:baselines`, and beats it: 19/20 here against 17/20 in the published cell. What changes is
the *attribution*, not the result.

## A design flaw in this study, stated plainly

**With P at 19/20 the comparison is arithmetically one-sided.** Exact McNemar needs six
one-directional discordant pairs for p < 0.05; P fails exactly one episode, so a comparator can
win at most one that P loses. **No comparator could have been shown better at n = 20 no matter
how good it was.** Only a comparator collapse was resolvable. That is my error: n = 20 was
chosen to match the paper's cells without checking what it could resolve against a baseline
this strong. Excluding a 10-point difference at 80% power needs ~186 pairs.

## An unregistered observation, flagged as such

**The most accurate estimator is not the most successful.** D has 29% lower MAE than P
(0.0204 vs 0.0286) and 18% lower worst-coordinate error, and scores two fewer episodes. This is
the third independent appearance of the accuracy/repair dissociation in this project, after the
ALOHA static sweep and the innovation-law arms. It was not predicted here and is not tested by
this design; it is recorded as an observation for a study that would actually test it.

## Consequence for the paper

Per the registered branch: **drop "the machinery earns its place"**, since the machinery that
distinguishes this law from a plain DOB is not what earns the result. Replace the raw-error
integral baseline with this table, which is the fair comparison a reviewer will ask for and
which the paper's law survives. **Lead with the nine-cell transfer matrix.** State the
attribution honestly: the plant model and `M` are the contribution; the update law is a
reasonable choice among several that work.
