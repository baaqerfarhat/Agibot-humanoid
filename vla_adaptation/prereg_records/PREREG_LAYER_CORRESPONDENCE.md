# Pre-registration — layer-fault correspondence: which layer's bias can a realised-metric search repair?

**Written 2026-08-18 ~17:05 PDT, BEFORE any search run.** The gate screens
(`layer_fault_gate.py`, same day) are complete for obs_bias and in flight for joint_offset;
their numbers are quoted as of writing. Results appended below; deviations recorded as
deviations.

## Claim under test

The layer to adapt is the one whose bias coordinates contain the fault's inverse
(mjlab_backend `_shift_obs` docstring, written before any of this): an **observation bias**
is exactly cancellable at the **first layer's bias** and not downstream; an **actuation
offset** is exactly cancellable at the **last layer's bias** (= a constant action residual)
and not upstream. If a search that uses only realised episode returns recovers the diagonal
cells and not the off-diagonal ones, "which layer" has a mechanism, in the paradigm that
actually works here (intervention search, no gradients) — layer-level ACE with structured
interventions (see `ACE_CONNECTION.md`).

Independent convergence, noted 18:05: a parallel session derived the same diagonal from
`S + T = I` (`docs/PLAN_CROSS_EMBODIMENT.md` §1.5) — perception/reference-side shifts are
tracked by the loop (`T ≈ I`), never rejected, hence repairable at the layer where they
enter; dynamics-side disturbances are rejected in-band (`S ≈ 0`) exactly where the residual
has authority. This experiment is the walker-scale empirical test of that principle:
obs_bias is the perception-side cell, tq05 the dynamics-side one.

## Cells

- **Fault A (obs_bias):** the gate's sd=0.20 draw on the `joint_pos` obs slice (49:69),
  ‖Δ‖ = 0.819, exact vector stored in `outputs/layer_fault_gate.json`. Frozen damage
  2.39 m (184.0 steps, 1/4 full); **analytic b0-fix ceiling +72.8%** under the deployed
  u_max = 0.10 double-forward route. (sd=0.10 is EXCLUDED: its required correction has
  ‖r‖ = 0.434, clipped at every step, ceiling 13.7% — the envelope, not the layer, binds.)
- **Fault B (joint_offset):** magnitude to be fixed by the running gate before that half
  launches. Criteria, stated now: frozen damage ≥ 1.0 m distance on 4 seeds AND the exact
  fix `r = −c` interior (linf(c) < 0.10). If the spread-over-20-joints draws are all inert
  (linf 0.03/0.05 already are, headroom +0.05 m), a legs-concentrated draw at linf ≤ 0.095
  is screened the same way and used. If no interior-fixable magnitude damages the robot,
  that half is reported as "condition A unreachable for interior joint offsets" and only
  the ACE arm runs there.

## Arms, per fault (constants fixed now; no rescue runs)

All scored on the realised metric `dist + 0.002·steps`, train seeds 2000–2001, CEM
pop 10 × iters 8, elites 3, sigma floor 1e-3. Held-out seeds 3000–3005. The **settled fix**
of a search arm is its best-ever candidate by train score (the const_adapt convention, so
recovery fractions are comparable to the multi-fault table).

1. **b6 search** — constant 20-d action residual (≡ mlp.6 bias), clip u_max 0.10,
   `sigma0 = 0.04` (action units; the const_adapt value).
2. **b0 search** — δb₀ = Σ αᵢ uᵢ over the 20 obs-channel directions
   `uᵢ = −W₀[:, slᵢ]/(σᵢ+eps)` (deployable: needs only the white-box weights), applied as
   the double-forward action residual clipped to the same u_max 0.10. `sigma0 = 0.15` rad
   (α* for fault A has ‖α*‖ ≈ 0.89, so 8 iterations at 0.15 can reach it; stated so the
   budget is auditable in advance).
3. **ACE arm** (the ACC-2026 estimator, 8 draws per layer): random coefficients at the
   class's own sigma0, norm-matched to it, scored on the train seeds; `ACE_hat` = mean
   effect vs frozen.

## Amendment — declared 17:08 PDT, before any fault-B search ran

The gate and the legs-concentrated fallback screen (`jo_legs_screen.py`) both returned
**inert** for every interior-fixable joint_offset: spread draws linf 0.03/0.05/0.08 →
headroom +0.05/+0.05/+0.03 m; legs-only linf 0.06/0.095 → +0.04/+0.06 m, 4/4 full episodes.
**Condition A is unreachable for interior joint offsets on this policy** — DR absorbs the
whole constant-offset channel up to the deployable envelope. That is itself a reportable
finding (the exactly-cancellable-at-b6 fault class has no room on this robot).

**Fault B is therefore replaced by `tq05`** (`torque_limit_scale 0.5`) — the output-side
fault with proven headroom (2.50 m) and a proven b6-search recovery (44.9% multi-fault /
68% const_adapt). Both arms rerun under THIS document's config (pop 10 × iters 8) so the
2×2 is internally consistent. Predictions for the tq05 column, stated now: **b6 ≥ 25%
recovery** (replication under the new config); **b0 < b6** — tq05's known fix is a
constant (harmonics added nothing), and the b0 parameterization expresses constants only
approximately through the ELU stack. If b0 ≥ b6 there, the correspondence claim weakens to
"b0's class contains constant-like members" and is reported as such. Neither cell is
exactly cancellable — the tq05 column tests class-match, not exact-inverse-match; the
exact-inverse contrast lives entirely in the obs_bias column.

## Primary endpoint

**Held-out recovery fraction per cell** (recovery = (arm − frozen)/(nominal − frozen), distance).
Predictions, in advance:

| | b0 search | b6 search |
|---|---|---|
| obs_bias (A) | **≥ 25% of the 72.8% analytic ceiling (≥ 18% absolute)** | **< 10%** |
| joint_offset (B) | < b6 | **≥ 50%** |

Diagonal ≥ thresholds AND off-diagonal below them ⇒ correspondence confirmed.

## Secondary

- `ACE_hat` sign per (layer, fault): predicted negative on the diagonal, ≥ 0 off-diagonal,
  and predictive of whether that cell's search clears its threshold. A miss here is
  reported against the ACE-screen proposal (ACE_CONNECTION §5), not tuned away.
- Search-vs-class credit: settled fix − mean(ACE draws), per diagonal cell.
- ‖found α‖ and cos(found δb₀, analytic δb₀*) on fault A — the search has an internal
  ground truth; this is the strongest single number the experiment can produce.

## Stated in advance

- The clipped double-forward route bounds BOTH classes by the same deployed envelope, so a
  b0-vs-b6 difference is attributable to the function class, not to a bigger budget.
- Train exposure is 2 seeds; transfer is expected because the fix is a property of the
  fault (as in PREREG_CONST_TQ05, where it held). Divergence is reported, not tuned.
- The obs_bias half launches immediately on this document; the joint_offset half launches
  only after its cell is fixed by the stated criteria — before its search starts, its
  chosen magnitude is appended here.

---

## Results, fault A (obs_bias sd=0.2) — 2026-08-18 22:59 PDT. **CORRESPONDENCE NOT CONFIRMED.**

Raw: `outputs/b0b6_obs_bias_0.2.json`, log `outputs/b0b6_obs_bias.log`.

| arm (held-out 3000–3005) | steps | dist | recovery |
|---|---|---|---|
| nominal | 300.0 | +5.91 m | — |
| frozen | 219.0 | +4.15 m | (headroom 1.76 m) |
| **b6 search (off-diagonal)** | 276.0 | **+5.58 m** | **+80.9%** |
| **b0 search (diagonal)** | 253.7 | +5.35 m | **+68.3%** |

- Prediction "b0 ≥ 18%": **passes** (68.3%). Prediction "b6 < 10%": **fails decisively** —
  the off-diagonal constant recovers MORE than the matched layer. Primary NOT confirmed.
- **cos(found δb₀, analytic δb₀\*) = +0.040**, ‖α‖ 0.56 vs α\* 0.82: the search did NOT find
  the analytic inverse — it found an essentially orthogonal direction worth the same
  recovery (68.3% vs the 72.8% analytic ceiling). Exactness is not where the value is.
- **ACE arm fails on this cell**: ACE_hat[b6] = −1.10 (7/8 draws harmful), ACE_hat[b0] =
  −0.76 (7/8 harmful) — random interventions in BOTH classes hurt, yet both searches
  succeed held-out. Combined with tq05 (random helps, search ≈ random), the mean-of-random
  ACE screen does not predict searchability across cells. Reported against
  `ACE_CONNECTION.md` §5 as pre-registered.
- Search-vs-class credit REVERSES the tq05 finding: here the searched **direction** carries
  the value (+1.43 m held-out for b6 while random draws average −1.1 score on train).

### Reading (mechanism, post-hoc but constrained by the numbers)

A CONSTANT obs bias, pushed through a policy operating on a limit cycle, produces an
approximately constant action error — so its practical inverse is roughly constant in action
space, i.e. inside BOTH classes (b6 directly; b0 through the ELU stack). Both tested faults
have constant-dominated inverses, which is exactly the regime where layer identity should
NOT matter — and it did not. The correspondence hypothesis as stated is dead; a sharper
version would need a fault whose exact inverse is strongly state-dependent. Not launched
tonight; recorded as the open follow-up.

---

## Results, fault B (tq05) + final scoring of the 2×2 — 2026-08-18 23:27 PDT

Raw: `outputs/b0b6_tq05_0.5.json`, log `outputs/b0b6_tq05.log`.

tq05 column predictions: "b6 ≥ 25%" **passes** (67.9%); "b0 < b6" **passes** (42.6% < 67.9%).
Note for honesty: the b6 arm used rng(0) like `const_adapt`, so its first 6 iterations retraced
the same candidate sequence and settled on the same best-ever (+5.11 m held-out — identical
number); iterations 7–8 found nothing better. It is a deterministic re-run plus two null
iterations, not an independent replication.

### The complete table (held-out recovery)

| recovery | b0 search | b6 search | ACE_hat b0 | ACE_hat b6 |
|---|---|---|---|---|
| obs_bias (perception-side) | 68.3% | **80.9%** | −0.76 | −1.10 |
| tq05 (dynamics-side) | 42.6% | **67.9%** | −1.62 | **+0.64** |

### What the 2×2 actually found

1. **No correspondence structure.** b6 ≥ b0 in BOTH rows. The output-layer constant class is
   simply the better search space on every tested fault — most plausibly because it is
   box-bounded, flat (no curvature between θ and the residual), and pays no clip interaction,
   while the b0 route pays ELU curvature and per-step clipping at matched budget.
2. **Search succeeds in all four cells** (42.6–80.9%) — the recovery is a property of
   low-dim realised-metric search, not of layer identity. "Which layer" collapses to a
   tractability answer — the output layer — consistent with the gradient-paradigm answer
   (`actor-jacobian`: mlp.6 the only well-conditioned layer) but for search reasons.
3. **ACE_hat tracks the blind class effect, not searchability**: positive only in (b6, tq05)
   — exactly where random constants help — while search wins everywhere, including three
   cells with negative ACE_hat. As a searchability screen it is dead; as a class-effect
   detector it is fine. The multi-fault "4/4 prediction" was headroom-confounded.

### Standing conclusions for the paper

- The method that works, in every paradigm tested tonight: **bounded low-dimensional
  intervention search on the realised metric, on the output layer's bias** — offline (tq05
  68%, obs_bias 81%), and online with fresh-seed episodes (V2: +1.38 m, ~55%, 120 episodes).
- The exact-inverse story and the layer-correspondence story are both falsified with
  internal ground truth (cos +0.04 at equal recovery). Do not resurrect without a fault
  whose inverse is strongly state-dependent — that is the one open follow-up this
  experiment licenses.

---

## Follow-up, declared 2026-08-19 00:45 PDT, BEFORE its search runs — fault C (joint_gain), three classes

The 2×2 falsification licensed exactly one follow-up: a fault whose inverse is strongly
STATE-DEPENDENT. `joint_gain` (multiplicative on the applied action, legs only) is that
fault, and the backend's own comment (mjlab_backend.py, written long before tonight)
predicts the ordering: *"exactly compensable by rescaling the LAST layer's weights by
1/alpha, and not by any input-side change."*

**This is the third test of the correspondence idea; that multiplicity travels with any
positive.** The exact inverse `r = (1/g − 1)·a` is NOT interior — actions are large, so the
u_max clip binds. The gate (`jg_gate.py`, running now) measures the **clipped multiplicative
oracle** = the envelope-limited ceiling. The experiment therefore compares CLASS SHAPES
under the same deployed envelope, not exactness.

Cell: legs-only g from {0.7, 0.5}, chosen by the standing criteria (frozen damage ≥ 1.0 m,
4 seeds); if both qualify, g = 0.7 (milder; more usable signal).

Arms (same config as the 2×2: pop 10 × iters 8, elites 3, train 2000–2001, held-out
3000–3005, settled = best-ever, all residuals clipped to u_max = 0.10):

1. **b6-const** — sigma0 0.04 (unchanged).
2. **b0** — obs-channel basis, sigma0 0.15 (unchanged).
3. **w6-scale (NEW)** — θ ∈ R²⁰, residual `r_k = clip(θ ⊙ a_nom(o_k), u_max)` — a diagonal
   rescale of the output layer, the fault's own inverse class. sigma0 0.15
   (θ\* = 1/g − 1 = +0.43 on legs at g = 0.7). Internal ground truth: cos(found θ, θ\*).
4. ACE arm — 8 draws per class at its prior scale.

### Predictions, in advance

- **P1 (the backend's comment, operationalised): w6 > b6 > b0** on held-out recovery.
- **P2:** w6 reaches ≥ 60% of the clipped-oracle ceiling from the gate.
- **P3 (the kill rule):** if b6-const ≥ w6, the class-shape version of correspondence dies
  too, the standing conclusion becomes "the bounded output-bias constant is the universally
  best search space on this plant", and **no further correspondence variants are run**.

### Fault C gate verdict — 2026-08-19 01:07. **BLOCKED BY THE ENVELOPE; the search is NOT run.**

`jg_gate.log` / `jg_gate.json`, 4 seeds:

| cell | frozen | headroom | needed ‖r‖ | envelope-limited ceiling |
|---|---|---|---|---|
| g=0.7 legs | 300.0 steps / +5.26 m (4/4 full) | **0.64 m — fails the ≥1.0 m criterion** | 0.322 (clip@every step) | +15.1% |
| g=0.5 legs | 44.8 steps / +0.81 m (0/4) | 5.09 m | 0.338 (clip@every step) | **+5.8%** |

Neither declared cell qualifies: g=0.7 fails condition A (too little damage), g=0.5 fails
condition C (the exact multiplicative inverse needs 3.4× the deployed cap; its clipped
ceiling is 5.8%, and searching classes under a 5.8% ceiling is a rule-3 violation). Per the
standing criteria the three-class search is **not launched** — the gate killed a six-hour
run in twenty minutes.

**The structural reading, and it sharpens the whole "which layer/class" answer:** every
fault with a state-dependent or large inverse now measured (obs_bias sd=0.1 → ‖r‖ 0.434;
joint_gain 0.7/0.5 → 0.32/0.34) exceeds the u_max = 0.10 action-residual envelope, while
every recoverable cell's fix is a small near-constant. **The deployed residual contract can
only express small constant-like corrections — the ENVELOPE selects the class before any
search does.** That is why the output-bias constant keeps winning: it is the only class the
contract leaves intact. Answering the class-shape question would require the parameter
route (direct weight edits under a parameter-space trust region) — a different bound
contract, i.e. a different experiment, not run tonight.

P3's standing conclusion activates in this modified form, and per the kill rule **no
further correspondence variants will be run under the residual contract.** Also replicated
in passing: the one-severity-wide fault band (g=0.7 walks fully; g=0.5 dies at 45 steps).

---

## PARAMETER-ROUTE experiment — declared 2026-08-19 01:15 PDT, BEFORE its search phase

The residual-contract block (above) is itself the motivation: under u_max the class-shape
question is unanswerable. This experiment **changes the bound contract, and says so**: the
residual is UNCLIPPED and computed as `f_edited(o) − f_nominal(o)`, so each arm is a literal
single-layer parameter edit of the deployed actor — `w6` = row-rescale of W₆ and b₆
(a′ = (1+θ)⊙a), `b6` = b₆ shift, `b0` = b₀ shift via the obs-channel basis. No action-space
bound; instead the rule-6 analog: mean induced ‖r_k‖ is REPORTED per arm. This is the
paper's own object (online layer adaptation with a trust region) rather than the bounded
safety residual. **Fourth test of the correspondence family; the multiplicity travels with
any positive.**

- **Gate (running now, `param_route.py --phase gate`):** the exact parameter oracle
  (θ\* = 1/g − 1 on legs) on g ∈ {0.5, 0.7}. **The search runs only if the g=0.5 oracle
  recovers ≥ 50% of headroom; otherwise the parameter route is blocked on this fault too
  and that is the (reported) result.** g=0.5 is the primary cell (5.09 m headroom); g=0.7
  is recorded but not searched (0.64 m headroom fails condition A).
- Search config: 3 classes × pop 10 × iters 8 × train 2000–2001 (2 eps/candidate);
  σ0 = {w6 0.30, b6 0.15, b0 0.30}; settled = best-ever; ACE arm 8 draws/class at prior
  scale; held-out 3000–3005.

### Predictions, in advance

- **P1 (class shape): w6 > b6 AND w6 > b0** on held-out recovery.
- **P2: w6 reaches ≥ 60% of the gate oracle's recovery.**
- **P3 (internal ground truth): cos(found θ, θ\*) > +0.5** — the search actually finds the
  inverse this time, unlike obs_bias's cos +0.04. If w6 wins with cos ≈ 0, the win is
  class-shape-generic, not inverse-finding, and is reported as such.
- **K (kill rule): if b6 ≥ w6 even unclipped, the correspondence idea is dead in every
  form tested, and no fifth variant is run.**

### Parameter-route results — 2026-08-19 13:20 PDT. **P1, P2, P3 ALL PASS; the ranking inverts.**

Raw: `outputs/param_route_g0.5.json`, log `outputs/param_route_search.log`. Gate first:
the exact inverse (θ\* = 1/g − 1) recovers **100.0%** at BOTH magnitudes through the
parameter route (vs 5.8% clipped) — the backend comment confirmed to the centimetre.

| arm (held-out, unclipped, joint_gain g=0.5) | steps | dist | recovery | mean ‖r_k‖ |
|---|---|---|---|---|
| nominal | 300.0 | +5.90 m | — | — |
| frozen | 44.8 | +0.81 m | — | — |
| **w6 rescale (inverse class)** | 231.3 | **+6.31 m** | **+107.7%** | 17.5 |
| b0 (input-side) | 210.3 | +4.44 m | +71.1% | **97.9** |
| **b6 constant** | 46.0 | +0.84 m | **+0.9%** | 0.51 |

- **P1 passes** (w6 > b0 > b6), **P2 passes** (107.7% vs the 100% oracle ceiling),
  **P3 passes** (cos(found θ, θ\*) = **+0.604**, vs obs_bias's 0.04). Kill rule not
  triggered — decisively: the constant that won every envelope-bound cell recovers
  **0.9%** here.
- Honest riders: (a) distance-recovery exceeds 100% while steps are 231/300 — the found
  edit **overspeeds** at some stability cost (rule 4: both reported); the found θ has
  ‖θ‖ 1.34 vs θ\*'s 3.46 — a smaller, differently-shaped edit that out-SCORES the exact
  inverse on the metric. (b) b0's 71.1% comes with mean ‖r_k‖ ≈ 98 — a violent,
  brute-force solution the unclipped contract permits; displacement reporting (rule 6)
  is what keeps this comparable. (c) ACE_hat is ≈ 0/negative for all three classes while
  search recovers up to 107.7% — the ACE-rescope (class-effect detector only) replicates
  a fifth time.

### The completed "which layer/class" answer, both contracts

**The winning edit site is jointly determined by the fault's inverse SHAPE and the
deployment CONTRACT.** Under the shipped action envelope (u_max 0.10), only near-constant
edits survive the clip, so the output bias wins everywhere (2×2 above). Under a parameter
trust-region, the inverse-shaped class wins by two orders (107.7% vs 0.9%) with the found
edit genuinely pointing at the analytic inverse (cos +0.60). Neither "always the last
layer" nor "the layer where the fault enters" is right on its own; the contract decides
which question is being asked. This section closes the correspondence family (four tests,
multiplicity recorded).
