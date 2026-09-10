# Lessons: online adaptation of a frozen VLA

Extracted from the project-wide lessons file. Sections 1-8 of that document cover the
walker and X2 humanoid threads and are not reproduced here; this is the VLA chapter.

---

## 9a. The fault gate: one cell survives, and it is the easy one

180 episodes, 9 cells (`PREREG_OPENPI_ACE_SCREEN` §1). Only `offset @ 0.05` on the arm dims
passes: 55% against a 99.0% nominal, a 44-point drop that is not floor-dead.

The gain arm brackets the window without landing in it (0.5 → 75%, 0.3 → 0/20), and no
severity between them was tested because the prereg forbids re-sweeping to manufacture a
cell. Brightness does **nothing** at any severity — verified live rather than assumed: the
fault reaches the model (100% of pixels change, image mean 120 → 215) and the served action
moves 0.018–0.020 against a 0.0168 noise floor. π0.5 is genuinely invariant to a global
brightness bias.

**The survivor is the most favourable geometry available**: a constant action offset, when
`action_out_proj/bias` *produces* a constant action offset. Fault class and edit class
coincide, so everything downstream is the easy case and is reported as such.

## 9b. THE POSITIVE: a frozen VLA is fully repairable online, by six numbers

| condition | success |
|---|---|
| wrong sign (k = −1) | 0.0% (0/15) |
| no edit, faulted | 46.7% (7/15) |
| k = 0.5 | 86.7% (13/15) |
| **k = 1.0, the computed edit** | **100.0% (15/15)** |
| k = 1.5 | 20.0% (3/15) |
| k = 1.0 applied to the **healthy** policy | 6.7% (1/15) |

46.7% → 100%, against a 99.0% nominal. No gradient, no fine-tuning, one layer's bias vector.

Three properties, all of which matter more than the headline:

- **It is specific, not a tonic.** The same edit on the healthy policy gives 6.7%. This is a
  compensating inverse and it only helps against the fault it inverts.
- **The basin is narrow.** k = 1.5 scores 20% — *worse than not repairing at all*. Overshoot
  by half is more damaging than the original fault. Any search here must be scaled to
  k ∈ [0.4, 1.2] or it spends its budget below its own starting point.
- **k = 1.0 was computed, not searched.** The scale came from quantile norm stats plus an
  attenuation measured open-loop on a synthetic observation, and it landed on 15/15 in
  closed loop. That is a two-way validation of both.

## 9c. Normalisation is not a detail — it decides whether the search is well posed

The obvious repair ("cancel +0.05 with ≈ −0.05") is **wrong**, and wrong in the direction
that would have been read as *the class is not reachable*.

π0.5 emits **normalised** actions; `Unnormalize` runs afterwards. Under quantile norm,
`env = (norm+1)/2·(q99−q01) + q01`, so each dim carries its own scale `(q99−q01)/2` — and
those differ **7×** across the arm channels. A *uniform* env-space fault is therefore wildly
**anisotropic** in the units the model works in:

| dim | fault in norm units | % of action range | β needed |
|---|---|---|---|
| dx / dy / dz | 0.053–0.060 | ~3% | 0.157–0.178 |
| **drx** | **0.391** | **19.5%** | **1.149** |
| dry / drz | 0.198–0.285 | 10–14% | 0.581–0.839 |

A "+0.05 offset" is 3% of the action range on translation and **19.5% on drx**. That is why
this cell survived the gate while brightness did nothing, and why `offset @ 0.10` was lethal.

**Rule 26. Parameterise the edit in the units the task is measured in, not the units the
parameter happens to live in.** In env-action units the repair is isotropic (−0.05 on all
six dims) and an isotropic CEM is well conditioned. In raw bias units the same target spans
7.3×, and an isotropic search under-explores `drx` — which carries most of the fault — while
over-exploring `dz`. Same search, same budget, different coordinates, and only one of them
can find the answer.

## 9d. Why the ACE screen measured nothing — four diagnoses, each measured

The pre-registered screen returned **F(9,70) = 0.294, p = 0.974, η² = 0.036**: sites explain
3.6% of draw variance. Four causes, in increasing order of how much they matter.

1. **The baseline was on different initial states than the draws.** All ten sites read
   *positive*, which looks like a class effect and is not one. The same faulted policy scores
   **40.0 / 46.7 / 52.0 / 55.0%** on four disjoint initial-state sets; draws vs baseline is
   +12.0 points at SE 11.2, **z = 1.07**. §3 fixed the baseline's episode *count* and never
   required it to share the draws' initial states. (The ANOVA is unaffected — it compares
   sites on identical states, so the offset cancels.)
2. **Unpaired scoring.** 96% of a 5-episode success rate is sampler noise. Pinning the
   sampler RNG makes an episode exactly deterministic — repeating one gives
   `max|ΔAction| = 0.0e+00` — so baseline and perturbed differ *only* by weights and each
   episode contributes a clean −1/0/+1.
3. **The scale was below the threshold of behaviour.** Paired, at c = 0.02, a perturbation
   flips **0 of 6** outcomes. Trajectories diverge across the full action range and the
   policy *re-converges to the same result*. Flips appear as c grows, and the ordering across
   sites is monotone and stable at every scale — a real ranking was there, buried.
4. **The matching itself is the deep fault.** `ρ = c·‖W‖_F/√numel` equalises *relative
   parameter* displacement. It equalises nothing that matters: measured |ΔAction| per unit c
   spans **30×** across sites, and `llm/mlp/linear/L17` returns **exactly zero** at 50% of
   its own norm — a whole trunk layer with no causal path to the action.

   It fails worst exactly where it matters. `action_out_proj/bias` has ‖W‖_F = 0.034 over 32
   entries, so the screen probed it at **ρ = 0.0006** when the oracle showed it needs
   **0.157–1.149** to do anything — and that is the site with a verified 100% repair. For a
   common output displacement the per-site c spans **0.74 … 21.2**, a 28× range. **No single
   c probes these sites comparably**, which is precisely what §3 assumed.

**Rule 27. Match interventions by their effect on the output, never by relative parameter
norm.** §3 chose relative matching so the ranking "would not be a ranking of layer sizes".
It instead produced a ranking of how directly a layer's parameters reach the output, at
wildly unmatched effective strength — the same failure it was designed to avoid, one level
down.

## 9e. THE STRUCTURAL RESULT: ACE measures curvature, adaptation uses the gradient

This is why ACE has now lost its job six times, and it is not an empirical accident.

For a symmetric perturbation `Δ ~ N(0, ρ²I)` and a locally smooth metric `M`, expand:

    E[M(W+Δ)] − M(W)  =  E[∇M·Δ]  +  ½E[ΔᵀHΔ]  +  O(ρ³)
                      =        0   +  ½ρ²·tr(H) +  O(ρ³)

**The first-order term vanishes identically.** So `ACE_hat` estimates `½ρ²·tr(H)` — the
*average curvature* of the metric at that site. Adaptability is a first-order quantity: the
best edit in a ρ-neighbourhood gains `≈ ρ‖∇M‖`. `tr(H)` and `‖∇M‖` are different objects and
there is no reason for them to correlate.

The measurements say exactly this. `action_out_proj/bias` is nearly **inert under random
perturbation** — mixed ±0.167 at matched scale, mean ≈ 0 — while admitting a **100% repair**
along one specific direction. A random 32-dim perturbation projects onto the 6-dim repair
direction with expected magnitude ~1/√32, so an isotropic estimator averages the signal away
by construction.

**Rule 28. An isotropic average cannot rank sites by a quantity defined as a maximum.**
`ACE_hat` is a *class-effect detector* (does perturbing this class help on average?), and
`ACE_CONNECTION.md` §5's original job for it — a searchability screen — is unreachable for a
reason that is now algebraic rather than empirical. Five prior falsifications plus this one.

## 9f. The fix: antithetic pairs isolate the first-order term

If the problem is that `E[∇M·Δ] = 0` kills the signal, difference the arms instead of
averaging them:

    D  =  M(W+Δ) − M(W−Δ)  =  2∇M·Δ + O(ρ³)

The curvature terms are **identical in both arms and cancel exactly**. `E|D|` over random Δ
estimates `ρ‖∇M‖` up to a dimensional constant, so `mean|D|` ranks sites by the first-order
sensitivity that adaptation actually exploits — while `(M₊+M₋)/2 − M(W)` recovers the old
isotropic estimator from the *same episodes*, giving a perfectly matched comparison of the
two estimands at zero extra cost.

Both arms are scored paired against one deterministic baseline, so every episode is
noise-free. Result appended below when the run completes.

## 9g. 78% of the perturbation went into dimensions the task discards

A third dilution, independent of the other two and the sharpest at the site that matters.

**LIBERO uses 7 of the model's 32 action dims.** `LiberoOutputs` returns `actions[..., :7]`
and the checkpoint's norm stats are length 7; dims 7–31 are padding, discarded before the
action ever reaches the environment. So an isotropic perturbation of `action_out_proj/bias`
(32 entries) puts **25/32 = 78% of its energy into directions that cannot affect anything**.

Measured open-loop at fixed ρ, restricting the draw to dims 0:7:

| site | \|ΔA\| all 32 dims | \|ΔA\| dims 0:7 | retained |
|---|---|---|---|
| `action_out_proj/bias` | 0.0162–0.0243 | 0.0162–0.0238 | **0.98–1.01×** |
| `action_out_proj/kernel` | 0.0134–0.0152 | 0.0124–0.0145 | 0.92–0.95× |
| `action_in_proj/kernel` | 0.0177–0.0240 | 0.0152–0.0205 | 0.80–0.86× |

The restricted draw carries only √(7/32) = 0.47 of the norm and reproduces **essentially all**
of the output-side effect — the discarded dims contribute nothing, as the architecture says
they must. `action_in_proj` retains less (0.80–0.86) because it consumes the 32-dim noise
vector `x_t`, whose padding entries still propagate internally even though their outputs are
thrown away.

**The honest qualification.** This does *not* rescue the screen on its own, and it is worth
being precise about why: once perturbations are matched by measured **output displacement**
(§9d, rule 27), the dilution is already absorbed — the calibration simply assigns the bias
site a larger c to compensate. Restricting to the task subspace makes the intervention 2.1×
more *efficient* per unit norm, not more *effective* at matched |ΔAction|. It is a
statement about where a layer's influence lives, not a fix for the estimator.

**Rule 29. Before perturbing a layer, check which of its coordinates the task can even see.**
A padded action head, a multi-embodiment output projection, a shared trunk serving several
heads — all put most of a layer's parameters outside the task's reach, and an isotropic
draw spends its budget there. Cheap to check: mask the axis and re-measure the output.

## 9h. RETRACTION of rule 28 — ACE *does* discriminate, once the measurement is fixed

**Rule 28 as stated in §9e is wrong, and the data that refutes it is my own.** It is
withdrawn. What follows replaces it.

The run: 4 sites spanning three tiers, 12 antithetic draws each, 8 paired episodes per arm,
sampler RNG pinned, perturbations matched by measured output displacement (rule 27) and
restricted to the task-relevant dims (rule 29). Each block yields BOTH estimators from the
same episodes, so the comparison is exact rather than inferred.

| site | tier | mean\|D\| (first-order) | ACE (isotropic) |
|---|---|---|---|
| `action_out_proj/bias` | interface | 0.177 | **+0.109** |
| `action_in_proj/kernel` | interface | 0.167 | **+0.115** |
| `expert/mlp_1/linear/L8` | action expert | 0.094 | +0.057 |
| `llm/mlp/linear/L0` | VLM trunk | 0.156 | **−0.026** |

| estimator | F(3,44) | p | η² | verdict |
|---|---|---|---|---|
| **ACE, isotropic** | **8.751** | **0.0001** | **0.374** | **DISCRIMINATES** |
| antithetic, first-order | 1.074 | 0.370 | 0.068 | no |

**The isotropic estimator — the paper's own — clears both bars, and the antithetic "fix"
does not.** Pairwise, both interface sites sit above the VLM trunk (p = 0.0007, p = 0.0003);
the two interface sites are indistinguishable from each other (p = 0.86); the trunk site is
the only negative value and sits below the expert site (p = 0.021). The tier ordering is
**interface > action expert > VLM trunk** — which is exactly prediction **P2** of the
original pre-registration, confirmed here for the first time.

**Where the reasoning failed.** The algebra in §9e is correct: for symmetric Δ the
first-order term vanishes and `ACE_hat` estimates `½ρ²·tr(H)`. The *inference* from it was a
non sequitur. "It measures curvature rather than gradient" does not imply "it cannot rank
layers" — average curvature is itself a perfectly good site-dependent quantity, and it
varies across tiers by more than enough to separate them. I conflated *ACE does not measure
adaptability* (still true, and still the reason it failed as a searchability screen five
times) with *ACE measures nothing rankable* (false).

**Why the n = 5 result pointed the other way.** It was draw luck, and the gaps closed with
data: mean|D| for the bias site went 0.200 → 0.177 while the expert site went 0.050 → 0.094.
The antithetic estimator is also the noisier of the two — it differences two noisy arms, so
its sd runs 0.094–0.155 against the isotropic 0.064–0.090. Worse variance and no separation.

**Rule 28 (replacement). A null from an estimator is a claim about the measurement before it
is a claim about the estimator.** The pre-registered screen's p = 0.974 was an artifact of
four measurement faults — unpaired scoring, a scale below the threshold of behaviour,
relative-norm matching, and 78% of the draw spent on dims the task discards. Fix those and
the same estimator, on the same model, separates layers at p = 0.0001. Nothing about ACE
needed changing.

**What this does and does not license.**
- It does **not** rescue the pre-registered §4 primary. That was run as declared and failed;
  this is a *post-hoc* re-measurement on 4 of the 10 sites at a different scale, and it is
  exploratory. Confirmatory status needs a fresh pre-registration over the remaining sites.
- It does **not** show ACE predicts repairability. `action_out_proj/bias` — the site with the
  verified 100% repair — is statistically tied for first, not uniquely identified. What the
  ranking buys is the *tier*: it points at the interface, which is where the repair lives.
  That is useful for selection and is weaker than "it finds the right layer".
- The five earlier falsifications of ACE-as-searchability-screen stand untouched. Ranking
  layers and predicting searchability remain different jobs.

## 9i. THE CONFIRMATORY RESULT: ACE discriminates, under a declared protocol

`PREREG_ACE_CONFIRMATORY.md`, written before the run, held-out initial states 10–17, 9 sites
× 8 draws × 2 arms × 10 episodes = 1440 episodes.

**PRIMARY PASSES: F(8,63) = 35.758, p = 1.33e-20, η² = 0.820.** Against the original screen's
F(9,70) = 0.294, p = 0.974, η² = 0.036 — same estimator, same model, same two bars — **the
null was a measurement artifact.** Four faults produced it (§9d, §9g): unpaired scoring, a
scale below the threshold of behaviour, relative-norm matching, and 78% of each draw spent on
dims the task discards.

Predictions scored: **P1 holds** (passes). **P2 holds** — interface (+0.048) > expert (+0.002)
> VLM trunk (−0.006), the ordering the original prereg could not resolve. **P3 holds** —
`action_out_proj/bias` ranks **5th of 9**; the top site is `action_out_proj/kernel`. **P4
FAILS against me** — I predicted the first-order secondary would not discriminate and it does
(p = 0.0017, η² = 0.313), so §9f's estimator comparison was underpowered rather than decisive.

**The limitation that qualifies all of it.** 54% of the between-site variance in ACE is
explained by how hard each site was actually perturbed (r = −0.738 vs log achieved
displacement, p = 0.037). Matching used one observation; across 20 observations from real
rollouts the achieved displacement spans **0.79×–6.24×** target, and the within-site CV runs
**0.29–0.98** — so **no scalar ρ per layer can equalise it**, because the response depends on
the observation. Relative-norm matching gave 30×; output-matching cut it to 7.9× and stopped.

Read it as: **ACE separates these layers, and about half of what separates them is
intervention strength rather than causal importance.** A strong result about measurement; a
weak one about layer selection.

## 9j. Two modifications that improve ACE, and what they are worth

**Max, not mean.** Adaptability is a *max* over directions; ACE averages over random ones.
Scoring each site by its best draw instead of its average, on the same episodes:

| statistic | rank of the known-repairable site |
|---|---|
| mean (standard ACE) | #5 of 8 |
| max / top-3 / top-5 / p90 | **#2 of 8** |
| consistency (§9k) | **#1 of 8** |

Stable across every upper-tail statistic, so not a knife-edge artifact. It is a one-line
change and it measurably improves the ranking.

**Measure in effect space, not metric space.** The fault here is a *constant* action offset,
and a layer can implement one only if its parameter change shifts the action the same way
regardless of input. Score `consistency = ‖mean_obs ΔA‖ / mean_obs‖ΔA‖`, a ratio, hence
scale-free — the exact defect that made ACE incomparable across layers:

| site | consistency |
|---|---|
| `action_out_proj/bias` | **0.906** |
| `action_in_proj/kernel` | 0.521 |
| `action_out_proj/kernel` / `expert L0` | 0.516 |
| `time_mlp_out`, `expert L8` | 0.43 |
| `img B26`, `llm L0` | 0.33, 0.30 |

It ranks the repairable site #1 with a 1.7× gap, cuts the confound from 54% to 26% of
variance, and costs forward passes rather than 1440 episodes. **Rule 30. Prefer a ratio of
measured effects to a difference of task metrics: ratios are scale-free, and scale is what
does not transfer across layers.**

## 9k. The ground truth that could not be measured — five diagnosed failures

Both claims above are validated against **one** known-repairable site. Selecting an estimator
by how well it ranks a single known answer is overfitting, so the fix was to measure
repairability for every site: probe, least-squares the combination pointing along the repair
direction, apply it, run episodes. It never worked. Each failure was a real defect:

1. Fitted a **unit-norm** target when the required shift is 0.532 — a 1.88× overshoot into
   the regime already measured as catastrophic (k = 1.5 → 20%). 0/10.
2. The combo handler **diffed the whole 3.35B tree per probe**, one model copy per term, and
   exhausted the card.
3. **Unregularised least squares** reached the right action effect by cancelling large probe
   coefficients: ‖δ‖ = 13.6 against the oracle's known-good 1.56. Ridge added.
4. **One-shot rescale assumed linearity.** The ~26× amplification needed sits in the
   sampler's saturating regime, so the achieved effect is not proportional to the multiplier.
   Iterative secant calibration converged (gain 2.301 → 0.881 → 1.018) and still scored 0/10.
5. **Probes rescaled to the repair magnitude** (adaptive c = 135.7 vs 21.2; one probe |ΔA|
   0.215 vs 0.188 target). Converged in two steps, still 0/10 at ‖δ‖ = 3.35 where the oracle
   needs 0.78. The mask is not the culprit — `delta_l2` is measured after masking.

**Rule 31. A random probe basis is the wrong instrument for constructing a targeted edit.**
The analytic repair needs ‖δ‖ = 1.56; the best fit from random probes needs ~4× that for the
same measured action shift, and the excess drives per-dimension saturation that destroys the
policy. This is the same lesson §9e was reaching for by a different route: random directions
are useful for *detecting* that a layer matters, and useless for *building* the edit.

**So the n = 1 limitation stands**, and it bounds §9j: "consistency ranks the repairable site
first" is one site, not a validated selection rule. Getting real ground truth means running
the actual search per site — which is a separate experiment, not a patch on this one.
