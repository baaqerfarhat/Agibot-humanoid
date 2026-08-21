# Pre-registration — Phase 0.4 on openpi: headroom gate, then the first multi-site ACE screen

**Written 2026-08-19 23:55 PDT, committed 2026-08-20 00:02 PDT — BEFORE any fault or screen run.** Results appended below;
deviations recorded as deviations.

## Established, not assumed

Measured on this machine today, before this document:

- **Gate 0.1 reproduces.** π0.5-LIBERO (`gs://openpi-assets/checkpoints/pi05_libero`),
  `libero_spatial`, 500 episodes: **99.0%** against the published **98.8%**. 5 failures;
  per-task 1.00 ×6, 0.98 ×3, 0.96 ×1.
- **Both tier-1 edit sites are live** on the loaded model: `action_out_proj` is
  `Linear(1024 → 32)`, kernel and bias mutable in place between episodes.
- **The b6 class survives the flow-matching sampler.** The head emits the velocity field
  `v_t`, not the action, and the sampler integrates `x_t += dt·v_t` over 10 steps with
  `Σdt = −1`, so a constant bias `β` should land as `−β` on the action. Measured on
  random init, noise held fixed: **ratio 0.942**, cross-dimension leak 11.9%. The 5.8%
  shortfall is the perturbed `x_t` feeding back into later `v_t` evaluations.
  **This was NOT measured on trained weights — see §0 below; it is a pre-run check, not
  an assumption.**
- **The layer menu**, read from the checkpoint (51 arrays, 3.35B params), has three tiers
  (§2). π0.5 has **no `state_proj`**: proprioception enters as discretised prefix tokens,
  so the input-side continuous projection that the walker's b0 arm used does not exist here.

## What this document is, and the multiplicity it carries

Two phases, pre-registered together because the second is only defined on what the first
returns: a **headroom/reachability gate** over a fault suite (PHASE0_OPENPI 0.3–0.4), then
an **ACE screen over many layers** on the surviving cell.

**ACE has appeared in this project five times already, and lost its original job every
time.** `ACE_CONNECTION.md` §5 proposed `ACE_hat` as a *searchability* screen; the 2×2
(`PREREG_LAYER_CORRESPONDENCE.md`) falsified that — positive in only 1 of 4 cells while
search succeeded in all 4 — and the parameter route replicated the falsification a fifth
time. `core/episodic_search.py` lesson 4 records the rescope: **class-effect detector, not
searchability screen.**

**This experiment does not re-litigate that.** It asks the question the walker was
structurally unable to ask. The ACC-2026 estimator is a *layer selection* stage — it ranks
`ℓ` within a fixed fault (§III-B; the paper selects layer 4 of 6). The walker only ever had
**two** candidate sites, `mlp.0` and `mlp.6`, so "ranking" was a coin flip; and on the
spacecraft the paper's own margins were ~1% (ACE −0.004387 vs −0.004256, nearly tied), so
the published case study cannot establish that the estimator discriminates either.

π0.5 supplies the first genuine menu. The question here is prior to usefulness:
**does `ACE_hat` separate layers at all, beyond its own draw noise?** If it does not, ACE
as a selection stage is dead generally rather than locally, and that is worth knowing at
the cost of one screen. If it does, the ranking is a real object and the follow-up
(search on the top-ranked site) is licensed.

## 0. Pre-run check (not a gate — a number that must be recorded before anything else)

Re-measure the bias-edit fidelity on the **trained** checkpoint, noise and observation held
fixed, β = 0.05 on 8 of 32 dims: report `ratio = mean(Δaction)/(−β)` and cross-dim leak.
Random init gave 0.942 / 11.9%. This is recorded, not gated: the screen perturbs weights
directly and does not depend on the ratio, but every later *search* claim in the b6 class
does, and discovering a trained-weight ratio of, say, 0.5 after the fact would poison it.

## 1. Fault suite and the headroom gate

Client-side wrappers on the LIBERO env, so every fault is exactly repeatable per seed and
no model surgery is involved. Suite: **`libero_spatial`** — the suite whose baseline is
reproduced (99.0%), which is also the cleanest place to measure a fault-induced drop,
since baseline failures do not confound it.

| side | fault | severities |
|---|---|---|
| actuation | per-joint action gain, multiplicative on the 7-DoF arm action | g ∈ {0.7, 0.5, 0.3} |
| actuation | constant action offset, arm dims | c ∈ {0.05, 0.10, 0.20} |
| perception | camera brightness/bias on the agent-view image | b ∈ {0.1, 0.2, 0.4} |

**Budget:** 2 trials/task × 10 tasks = **20 episodes** per cell, 9 cells = 180 episodes.
Frozen policy only. Nominal reference is the 500-episode gate-0.1 run already in hand.

**Gate (the framework's order — condition A before anything is built):** keep a cell iff

- **drop ≥ 30 percentage points** from the 99.0% nominal, **and**
- **not floor-dead: success > 5%** (the walker's tq04 lesson — a dead robot returns no
  search signal, and an ACE draw scored on a dead robot measures nothing).

**The screen runs on exactly one cell: the surviving cell with the largest drop that is
still above the floor.** If two cells tie within 5 points, the actuation-gain cell is
taken, because its inverse is state-dependent and multiplicative — the regime the walker's
residual contract could never express (clipped ceiling 5.8%) and the parameter route
recovered at 107.7%. Recorded now so the choice is not made after seeing the ACE numbers.

**Kill rule:** if no cell passes, Phase 0.4 reports "no usable fault cell on
`libero_spatial` under client-side wrappers" and the screen does not run. Severities are
NOT re-swept to manufacture a cell; a second suite (`libero_10`, published 92.4) may be
screened once under this same document, and that is the only extension permitted.

## 2. The sites

Ten sites, spanning the three tiers. Trunk tiers are sampled at first/middle/last rather
than exhaustively; `_1`-suffixed arrays are the action-expert stream.

| # | tier | site | shape |
|---|---|---|---|
| 1 | interface | `action_out_proj/bias` | (32,) |
| 2 | interface | `action_out_proj/kernel` | (1024, 32) |
| 3 | interface | `action_in_proj/kernel` | (32, 1024) |
| 4 | interface | `time_mlp_out/kernel` | (1024, 1024) |
| 5 | action expert | `mlp_1/linear` layer 0 | (4096, 1024) |
| 6 | action expert | `mlp_1/linear` layer 8 | (4096, 1024) |
| 7 | action expert | `mlp_1/linear` layer 17 | (4096, 1024) |
| 8 | VLM trunk | `llm/layers/mlp/linear` layer 0 | (16384, 2048) |
| 9 | VLM trunk | `llm/layers/mlp/linear` layer 17 | (16384, 2048) |
| 10 | vision | `img/.../MlpBlock_0/Dense_1/kernel` block 26 | (4304, 1152) |

## 3. The estimator, and the one thing that must be matched

`ACE_hat(ℓ) = E[M | do(W_ℓ + Δ_ℓ)] − E[M | W_ℓ]`, `Δ_ℓ ~ N(0, ρ_ℓ² I)`, each draw scored by
**closed-loop rollouts** on the realised metric (LIBERO success rate) — the ACC-2026
estimator unchanged.

**`ρ_ℓ` is scaled per layer so the RELATIVE perturbation is identical across sites:**

    ρ_ℓ  =  c · ‖W_ℓ‖_F / sqrt(numel(W_ℓ))        so that  ‖Δ_ℓ‖_F / ‖W_ℓ‖_F = c

with **c = 0.02**, one level. Without this the ranking is a ranking of layer sizes, not of
causal importance — the same matched-displacement discipline as rule 6, and the same
convention as the existing `AceL0–L3` probes in `adaptation/adapt_experiments_isaac.py`.
`‖W_ℓ‖_F` is computed from the checkpoint and recorded per site before any rollout.

**Budget: 8 draws per site × 5 episodes per draw = 40 episodes per site**, 10 sites =
**400 episodes** (~2 h at the measured 18 s/episode). Baseline `M | W_ℓ` is the frozen
faulted arm, 20 episodes, shared across sites. Episode seeds are fresh per draw and
disjoint from the gate's; the same seed set is used for every site, so sites are compared
under common random numbers.

## 4. Primary endpoint

**Does `ACE_hat` discriminate between sites beyond draw noise?**

One-way ANOVA over the 10 sites × 8 draws (`df = 9, 70`), plus the effect size
`η² = SS_between / SS_total`.

**Primary passes iff `p < 0.05` AND `η² ≥ 0.25`.** Both are required: with 80 draws a
trivial separation can reach significance, and a ranking that explains under a quarter of
the variance is not a selection stage.

## 5. Predictions, stated in advance

1. **Primary passes.** A VLA's layers differ far more in causal role than a 6-layer
   spacecraft net's, where the paper's own margins were ~1%.
2. **Tier ordering: `mean|ACE_hat|` interface > action expert > VLM trunk**, at matched
   relative perturbation. This is prediction **P4** of `PLAN_CROSS_EMBODIMENT.md` §2,
   tested here for the first time and by intervention rather than gradient.
3. **Sign: `ACE_hat < 0` at ≥ 7 of 10 sites.** Random perturbation of a trained model at
   2% relative magnitude should hurt. A positive `ACE_hat` at a site is the class-effect
   signal (random edits of that class help), which on the walker occurred in exactly one
   cell of five.
4. **`action_out_proj/bias` (site 1) is NOT the top-ranked site by `|ACE_hat|`.** It is 32
   parameters against millions; at matched *relative* norm its absolute displacement is
   tiny. If it ranks top anyway, the screen is measuring something other than what the
   matching intends, and that is reported rather than explained away.

## 6. Decision rule

| outcome | reading |
|---|---|
| primary passes AND prediction 2 holds | ACE discriminates AND the ordering is the theory's. The top-ranked site is carried to a pre-registered search; the ACC selection stage is validated on a real menu for the first time. |
| primary passes, prediction 2 fails | ACE discriminates but not along the interface/trunk axis. Report the ranking as an empirical object; the layer story is decided by measurement, not by §1.5. |
| **primary fails** | **ACE does not separate layers beyond draw noise on a 3.35B-parameter model with a genuine menu.** Combined with the five prior class-effect-only results, the estimator is retired as a selection stage in this project — reported, not re-tuned. Phase 0.5 proceeds directly to a tier-1 search, which needs no layer ranking. |

## 7. Stated in advance

- **No rescue runs.** Not at another `c`, another draw count, another episodes-per-draw,
  another site list. One value of `c`; the site list above is frozen.
- **`ACE_hat` is not claimed to predict searchability.** That claim was falsified five
  times and is not revived here. This screen's object is the *ranking*, and §6's kill row
  is what happens if the ranking is noise.
- **Screening dimension is not search dimension.** ACE perturbs a whole layer isotropically,
  so condition E does not bind on the screen. It binds hard on whatever follows: only
  tier 1 has a ≤32-dim parameterisation for free, and a top-ranked trunk site would need a
  basis before any search — which is a separate experiment, not an extension of this one.
- **Multiplicity:** sixth appearance of ACE in this project, first as a multi-site ranker.
  The five prior class-effect results travel with any positive.
- 20 episodes per gate cell is thin for a 30-point threshold; a cell landing within 5
  points of the boundary is reported as borderline and re-run once at 40 episodes, which
  is declared here rather than decided later.

---

## Section 0 result — 2026-08-20 00:30 PDT, BEFORE the gate and screen. **The random-init number was wrong by 3x.**

Raw: `results/openpi_s0_bias_fidelity.json`. Trained `pi05_libero`, bf16, observation and
noise held fixed, beta on 8 of 32 dims.

| | random init | **trained** |
|---|---|---|
| ratio measured/predicted | 0.942 | **0.311** |
| cross-dim leak | 11.9% | **2.9%** |

Stability sweep, 3 noise seeds x beta in {0.02, 0.05, 0.10}:

| beta | seed 7 | seed 11 | seed 23 |
|---|---|---|---|
| 0.02 | 0.310 | 0.326 | 0.373 |
| 0.05 | 0.313 | 0.323 | 0.378 |
| 0.10 | 0.324 | 0.338 | 0.375 |

**ratio = 0.340 +/- 0.026** (min 0.310, max 0.378). Essentially constant in beta across a
5x range, so the map is LINEAR; the spread is driven by the noise draw, not the magnitude.

### What this changes

- **The b6 class still works, at ~1/3 the analytic gain.** A target action offset needs
  `beta ~ 2.9x` the naive value. Any sigma0 carried over from the walker's action-space
  constants would have searched a 3x-too-small region -- the exact units trap of LESSONS
  rule "a scale correction silently retunes every constant downstream of it".
- **The trained flow is CONTRACTIVE, and that is the interesting part.** At random init the
  sampler passes a velocity-field offset through almost intact (0.94). Trained, it rejects
  two thirds of it. The denoiser is a feedback loop that funnels `x_t` toward the data
  manifold, so perturbing its velocity produces a partially cancelling correction on the
  next step -- a flow-matching instance of the `S + T = I` argument in
  `PLAN_CROSS_EMBODIMENT.md` section 1.5, arising INSIDE the action head rather than in the
  robot's servo loop. It also predicts the direction of a useful contrast: an edit applied
  to the emitted action CHUNK (client-side, post-sampler) faces no such rejection and should
  land at ratio 1.0, whereas the same edit at the layer is attenuated 3x. The two routes
  are therefore NOT equivalent here, which is the openpi analogue of the residual-contract
  vs parameter-route distinction the walker closed on.
- Cross-dim leak FELL (11.9% -> 2.9%): the trained head is far cleaner in its channel
  separation than a random one.

### Caveat carried forward

Measured on `cfg.fake_obs()`, a synthetic observation. The ratio is stable across noise
draws but has NOT been measured on real LIBERO observations, which is the operating
regime. Re-measure on a real observation stream before any b6-class search claim rests on
the 0.34 figure; the qualitative finding (trained flow attenuates, ~3x, linearly) is what
this section establishes.

**No part of the gate or screen design changes as a result** -- section 3 perturbs weights
directly and does not depend on this ratio. Recorded here because a later search does.

---

## Results

### §1 fault suite and headroom gate — run 2026-08-20, 18:24–20:49 PDT

Frozen π0.5-LIBERO served from `gs://openpi-assets/checkpoints/pi05_libero`, `libero_spatial`,
2 trials/task × 10 tasks = 20 episodes per cell, identical initial states across cells.
Client: `vla_adaptation/openpi/gate_faults.py`; per-cell JSONs in `results/gate04/`.

**Harness check.** The same client run with no fault: **19/20 = 95%**. Consistent with the
99.0% reference (one failure in 20 has probability 0.18 at p = 0.99); the reference for the
gate stays the 500-episode number, per §1.

| fault | sev | success | drop | verdict |
|---|---|---|---|---|
| gain | 0.7 | 19/20 = 95% | 4 | drop < 30 |
| gain | 0.5 | 15/20 = 75% | 24 | drop < 30 |
| gain | 0.3 | 0/20 = 0% | 99 | floor-dead |
| offset | 0.05 | 11/20 = 55% | 44 | **KEEP** |
| offset | 0.10 | 1/20 = 5% | 94 | floor-dead |
| offset | 0.20 | 0/20 = 0% | 99 | floor-dead |
| brightness | 0.1 | 20/20 = 100% | −1 | drop < 30 |
| brightness | 0.2 | 20/20 = 100% | −1 | drop < 30 |
| brightness | 0.4 | 20/20 = 100% | −1 | drop < 30 |

**Exactly one cell survives: `offset @ 0.05`, 55%, a 44-point drop.** The §1 tie-break
(prefer the actuation-gain cell) never triggers, there being no tie.

**Interpretation recorded with the runs, not after.** §1 says "per-joint action gain … on
the 7-DoF arm action", but LIBERO's 7-vector is 6 OSC deltas plus a gripper channel that is
effectively ±1; scaling that 7th dim models a disabled gripper rather than a weak joint. The
arm faults were therefore applied to dims 0–5. Every cell JSON records `dims: 6`.

**The brightness null is verified, not assumed.** A perception fault reading exactly 100% at
all three severities is the shape of a dead term, so it was checked directly: the fault is
live at the input — 100% of pixels change, image mean 120 → 145.6 → 170.5 → 215.5 — while
the resulting change in the served action, |Δa| ≈ 0.018–0.020, sits **at the sampler's own
noise floor**: querying the identical observation twice gives |Δa| = 0.0168. π0.5-LIBERO is
invariant to a global brightness bias up to its own stochasticity. That noise floor is
itself a number the screen must clear.

**Two cells excluded near a boundary, neither re-run.**
- `gain @ 0.5` drops 24 points, 6 points outside the 30-point threshold — and 6 > 5, so §7's
  declared re-run-at-40-episodes clause does not cover it. No gain severity between 0.5 and
  0.3 was tested and none will be: §1 forbids re-sweeping severities to manufacture a cell.
- `offset @ 0.10` sits at exactly 5.0% — 1 episode of 20 — and the floor is `> 5%`, so it is
  excluded by a single episode. Flagged as fragile and left excluded, for the same reason.

**Consequence for the screen, stated plainly.** The surviving fault is a *constant action
offset*, which is precisely the b6 class: `action_out_proj/bias` produces a constant action
offset, so fault class and tier-1 edit class coincide. That is the most favourable geometry
available and the least informative about the general case. §1 preferred the gain cell
exactly because its inverse is state-dependent and multiplicative — the regime the walker's
residual contract could not express — and that cell did not survive the gate. Any positive
result from a search on this cell carries that caveat.
