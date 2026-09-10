# Dual-track convergence — does the method address the joint-fault harm?

2026-09-08. Both tracks worked the same question from the same artifacts. **Verdict: partly, and
conditionally** — and the second track corrected which component does the work.

## 1. What both tracks verified independently

I aggregated `results/joint_followup/confirmation_plan/scores.json`; the second track reconciled
from `compact/{suite}/joint{j}/*.json` by pairing `per_ep` on `(suite, task, init)`. **Identical
figures by two different routes**, so the agreement is not a shared-instrument artifact:

| joint | comparison | | rescues / regressions | exact p |
|---|---|---|---|---|
| 5 | legacy vs OFF | 36/60 → **15/60** | 4 / **25** | **1.04e-04** |
| 5 | candidate vs legacy | 15/60 → **41/60** | 29 / 3 | **2.556e-06** |
| 5 | candidate vs OFF | 36/60 → 41/60 | 14 / 9 | 0.405 |
| Object 3 | candidate vs legacy | 16/20 → **2/20** | 1 / 15 | 5.19e-04, Bonferroni-24 **0.01245** |

**No claim in `report/JOINT_FAULT_MECHANISM.md` is wrong or overstated.** The second track went
further than I did: initial-state hashes match across arms, runner source hashes match the frozen
snapshots, the scorer validates all 72 contrasts, and 24 × 20 × 3 = **1,440** confirms the rollout
count. Policy RNG is not pinned, so these are paired *scenarios*, not identical policy-randomness
counterfactuals.

## 2. Where the tracks diverged, and the second one is right

I framed this as **two** defects and attributed the binding one to `Φ_d(ξ)`. Both were wrong.

> "M is averaged, not a locally correct inverse" **combines two problems**: predicting the current
> disturbance, and predicting the response to its proposed correction. `Φ_d` can help the former; an
> accurate or adapted **input map** is needed for the latter.

The algebra settles it. With context held fixed, `r(a) = d(ξ) + (G(ξ) − F(ξ))a`, so

```
r(a₂) − r(a₁) = (G − F)(a₂ − a₁)
```

A state-dependent **additive** model can fit `r(a₁)` exactly and predict nothing about the second
expression — which is precisely what applying a correction does. `Φ_d(ξ)` is a forward disturbance
representation in modelled row units; it is not a learned inverse merely by being state dependent.

Confirmed at source: `openloop_id.py:identify` averages plus/minus command-offset replay differences
over a common episode prefix to build each `M` column. It does not identify a command Jacobian at
faulted states. `M` really is an average.

## 3. The counterexample, run through the real allocator

`ẋ = −x + 3a + d`, true effectiveness 3, nominal model believes 1, disturbance `d = 0.1` known
**exactly**. Through `learned_adaptation.core.matched_action`:

| arm | action | resulting ẋ | |
|---|---|---|---|
| OFF | 0.0000 | **+0.100** | |
| perfect `Φ_d`, **wrong** input map | −0.1000 | **−0.200** | **2× the physical error of doing nothing** |
| perfect `Φ_d`, correct `Ê` | −0.0333 | **0.000** | cancels |

The disturbance estimate is exact in every row. State-dependence of `Φ_d` changes nothing in this
same-state calculation, and at the first step `e = 0` so the composite tracking term cannot
intervene. **The binding requirement is the input map, not the disturbance basis.**

## 4. Four defects, four components — with what each does not fix

| defect | component | what remains |
|---|---|---|
| invert-then-mask allocation | `matched_action` constrained solve | cannot make a wrong map correct or create missing physical directions |
| predicting the disturbance `d(ξ)` | `Φ_d(ξ) z_d` | does not predict how the plant responds to a changed command |
| predicting the response `(G − F)` | **`E(ξ, z_E)`** | needs identifiable excitation, adequate context, feasible inputs, singularity handling |
| prediction-vs-tracking mismatch | composite term, Riccati `P` | does not optimise grasp success or contact retention; cannot act when `e = 0` |

**So the method does address the mechanism — but through `E`, not `Φ_d`, which is where I had it.**

## 5. Three scope qualifications that also correct me

1. **The confirmation tests a *combined* candidate**, not allocation alone: xyz becomes all six
   inputs, residual filtering and regularisation change, bounds move inside the solve, and the
   weighted DOB bypasses the legacy deadzone and normaliser. So "fixing allocation moved joint 5
   from 15/60 to 41/60" — which I wrote — is not established. Allocation is one of several
   simultaneous changes.
2. **The 168 same-state probes prove model/physical disagreement, not a fitted feedback loop.** The
   report calls its feedback equation a *possible* mechanism; reading it as the established
   explanation of the Object joint-3 failure overstates it.
3. Those 168 probes share scenarios and checkpoints and are **not** 168 independent task trials.

## 6. Does the method inherit the LIBERO Object failure?

**Yes, it can.** The counterexample in §3 is exactly that failure mode: a well-fitted disturbance
field plus an inaccurate input map allocates a correction that changes motion or contact the wrong
way, and no component checks the task outcome. `matched_action` raising on infeasibility does not
help when the target is feasible but **wrong**. Missing modelled authority raises an exception;
deployment still needs a defined bounded fallback, which does not exist.

## 7. The agreed next experiment

Not another sweep. **One scenario**: LIBERO Object joint 3, **task 2 / init 35** — the first sorted
case where both OFF and legacy succeed and weighted-full fails. Reset-paired continuation through
the decisive grasp/contact transition, comparing OFF, constant basis with fixed `E`, learned
additive basis with fixed `E`, and learned basis with adapted `E`. That four-arm ablation separates
`Φ_d` from `E`, which §3 shows is the crux.

Preconditions the second track specifies: reconstruct full simulator and controller state by replay
and verify it before snapshotting, since a logged endpoint is not a complete state; establish the
failure and contact event from the old weighted replay; freeze the checkpoint and scoring horizon
before inspecting any learned-controller result. **If the old harm cannot be reproduced under that
protocol, it is not yet a valid counterexample test.**

Archived telemetry:
`lambda:/data/fxxie/vla/followup_20260908/runs/panda_confirmation/cells/libero_object/joint3/telemetry.jsonl`
(46,624,070 bytes, SHA-256 `692b7408…8c09ce14`).
