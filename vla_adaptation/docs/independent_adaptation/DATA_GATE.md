# Admission gate — is there state structure for a learned basis to capture?

2026-09-08, Claude track, before building any training pipeline. The second track is answering the
same question independently.

## Why this gate

The successor is fully specified and implemented with zero evidence, and the blocker is that no
basis is trained. Before building a training pipeline, the cheap question is whether the data
contains the structure the basis is supposed to represent. This follows the discipline that has
worked in this line: run the measurement that could make the expensive work pointless, first.

## Method

Leave-one-episode-out over the 20 episodes of each stored cell, ridge regression predicting the
residual `r`, scored by **held-out** sum of squared error against a constant baseline. In-sample fit
is not reported; a constant is the honest baseline because if it wins, `Phi_d(xi)` has nothing to
learn.

Features split three ways, because `r` already subtracts an FIR of the **command history**, so
including the current action can pick up input-map error rather than state dependence:

- **state**: end-effector position, quaternion, normalised time
- **action**: the current 6-dim policy command
- **both**

## Result

| features | joint 3 (correction **repairs**, 11/20 → 18/20) | joint 5 (correction **harms**, 12/20 → 3/20) |
|---|---|---|
| state only | **+13.6%** | **−6.9%** |
| action only | +7.5% | **+12.7%** |
| both | +16.9% | +11.3% |

**At the cell where the correction fails, state features have negative held-out R^2** — they overfit,
there is no state structure to capture — **and essentially all of the explainable residual is the
action term.** At the cell where the correction succeeds, state genuinely dominates.

## What this establishes

It is an independent confirmation of G4, reached by a different route. G4 argued algebraically that
`r(a) = d(xi) + (G - F) a`, so a state-dependent additive basis can fit the disturbance and still be
wrong about the response to a changed action; it verified this with a scalar counterexample through
`matched_action`, where a perfect `Phi_d` with a wrong input map doubled the physical error.

This gate reaches the same conclusion from **held-out regression on real telemetry**: at joint 5 the
learnable residual structure is input-map error, not state-dependent disturbance. Two independent
routes, algebraic and empirical, to the same finding — **`E` is the binding component, `Phi_d` is
not.**

## Consequence for the training work

- Training `Phi_d(xi)` is **worth attempting at joint 3**, where state carries 13.6% of held-out
  variance.
- At joint 5 it would learn nothing. **A learned disturbance basis cannot fix the cell that
  motivates the whole direction.** Fixing joint 5 requires the input map.
- So the training pipeline, if built, should target `E(xi, z_E)` first. The effectiveness branch is
  implemented in the second track's `loop_codex.py` and absent from mine.

## Limits

- **A linear probe.** A nonlinear basis might find state structure this misses; a negative linear
  result is weaker evidence than a positive one. It does establish that no *linear* state model
  helps at joint 5, which is what the current `Phi(xi) z` form with a linear-in-coefficients
  structure would exploit.
- Two cells, one suite, one fault kind, ridge at a single regularisation.
- "Action features" conflate genuine input-map error with any instantaneous command dependence the
  FIR history model misses. The decomposition separates state from action, not the sub-causes of the
  action term.
- Held-out by episode, but episodes within a cell share task and initial-state structure, so this is
  not a fully independent generalisation test.

---

# CONVERGENCE — my conclusion was wrong, and the reason is structural

**2026-09-08, after reading the second track's report (`codex_prompt/CODEX_G8.md`), which I had
left unopened while reporting my own result. That was a dual-track failure on my part.**

## What it found that I missed

I tested whether **absolute configuration** (position, quaternion, time) predicts the residual, found
`−6.9%` at joint 5, and concluded "there is nothing for a state-dependent basis to learn."

The second track tested a **nested ladder** of predictors and found the dominant term is one I never
included: **the previous two measured-motion increments.** Its held-out-task numbers, on the correct
suite calibrations:

> History reduces SSE against the full command-only correction by **90.2%, 85.1%, 94.8%** on healthy
> Spatial/Object/Goal and **87.6%, 94.7%** on Object joint 3 / joint 5, improving on **every one of
> the twenty held-out episodes in each cell.**

Verified independently on my own cells, held out by episode:

| | command only | + configuration | **+ past measured motion** |
|---|---|---|---|
| joint 3 | — | +10.1% | **+44.0%** |
| joint 5 | — | **−1.7%** | **+70.7%** |

**My negative result on configuration is confirmed** — the second track independently finds State
versus Phase-control deltas of `+0.18%` to `+6.52%`, all worse. What was wrong was the conclusion
drawn from it.

## Why this is structural, not a missing feature

The FIR is a **pure moving-average model**: `q_j[t] = Σ_k w_jk u_j[t−k] + c_j`. I said so early in
this session — no autoregressive term, no dependence on any state variable, output predicted from
command history alone. I identified the gap correctly and then failed to test it.

Adding past **output** is exactly the autoregressive structure the FIR lacks. So 44–71% of what the
estimator treats as *a fault to be cancelled* is **unmodelled plant dynamics**.

That reframes the method's central quantity. The estimator forms `r`, inverts `M`, and applies the
result as a correction. If the majority of `r` is missing plant memory rather than disturbance, the
correction is cancelling model error as though it were a fault — which is a mechanism for the
dissociation this project has measured all session: better estimates, worse task outcomes.

## What it means for the successor

The descriptor form is `B ẋ = A x + E a_sent + Φ_d z`. **The `A x` term is precisely the state
dependence the FIR omits.** So the successor's model structurally contains the term that explains
44–71% of the predecessor's residual, and it does not need a learned basis to get it — it needs
`A` to be identified, which is ordinary system identification.

This is the strongest argument for the descriptor form produced in this session, and it is
measured rather than argued.

## What the second track's verdict actually was

> "State/history dependence survives held-out testing. A small per-robot basis experiment is
> admissible; expensive nonlinear meta-training is not yet justified."

Also: velocity gives a specific positive (Object joint 3 improves **34.0%** on the task split, 10/10
tasks, 17/20 episodes), while quadratic features "contribute little or hurt." So the admissible next
step is a **small linear** basis, not a neural one — which is a cheaper and better-supported plan
than the Neural-Fly-style meta-training the direction started from.
