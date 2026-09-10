# Three proposed directions for the joint-5 failure, tested

2026-09-08, Claude track. Second track answering independently. All numbers are held-out,
leave-one-episode-out over the 20 episodes of each stored cell.

## Summary

| direction | joint 3 (correction repairs) | joint 5 (correction harms) | verdict |
|---|---|---|---|
| **non-diagonal coupled plant** | +3.1% | **+5.5%** | **supported at the failing cell** |
| **end-effector body frame** | +3.4% | **−3.6%** | **not supported at the failing cell** |
| SE(3) equivariance | see §3 | see §3 | not directly testable as posed |

## 1. Non-diagonal input map — supported, with a serious caveat

The measurement that motivated this: on the correct `libero_object` calibration, `M` carries
**71.6%** of its energy off-diagonal while the per-axis FIR has none, and the two disagree by a
factor of 2.8-4.4 despite each being individually well determined.

`fit_plant` already has a coupled `mimo=True` option, default off. Recomputing the residual under
each plant model — reconstruction verified exact against stored `r` to `8.2e-14` — and scoring
held-out SSE about a constant, where lower means more of the residual is a clean fault signal
rather than model error:

| | per-axis | MIMO | |
|---|---|---|---|
| joint 3 | 66.389 | 64.336 | +3.1% |
| **joint 5** | 30.467 | **28.788** | **+5.5%** |

It helps, and it helps **more at the failing cell than the working one**, which is the right
direction.

**The caveat, which is the whole difficulty.** `fit_plant`'s docstring rejects the coupled model on
a specific ground: *"the adaptive law runs precisely OUT of that distribution: the correction it
applies moves the executed command away from the data the plant was identified on. The simpler model
fits worse and extrapolates better, and extrapolation is what the law depends on."* It reports
13/15 against 14/15 and a sign error on `dy`.

**My measurement cannot address that.** It scores the residual on the **frozen** arm, where no
correction is applied — precisely the in-distribution regime where the coupled model was never in
dispute. The documented failure is in closed loop, off-distribution. So this result says the coupled
model is a better *open-loop* plant model at joint 5; it says nothing about whether it survives the
correction being applied.

Two further qualifications: the docstring's rejection was measured on the **additive-offset** cell,
not on joint faults, so re-testing it on joint faults is legitimate; and the coupled model has 43
parameters against roughly 270 samples, which is why it needs ridge and why extrapolation is the
concern.

## 2. End-effector body frame — not supported at the failing cell

Verified first that this is a real question rather than a mis-specified one: `rot_delta` computes
`q_rel = q1 (x) conj(q0)`, a **left** multiplication, so the rotation residual is in the world frame,
as is `x1 - x0`. Quaternions are **xyzw** (`so3.py`), confirmed rather than assumed.

Fair test: fit a constant residual in each frame on training episodes, predict held out, score both
in the world frame.

| | world-frame constant | body-frame constant | |
|---|---|---|---|
| joint 3 | 66.531 | 64.261 | +3.4% |
| **joint 5** | 30.675 | **31.791** | **−3.6%** |

The body frame is slightly better where the correction works and slightly **worse** where it fails.
Both effects are small against the +12.7% the action features carry at joint 5. The hypothesis that
apparent state dependence is a world-frame artifact is not supported at the cell that matters.

This does not mean frames are unimportant here — `so3.py` documents that an earlier chart error left
`M`'s rotation block **swapped and sign-flipped** with diagonals near 0.01, and fixing it took the
rotation FIR from R^2 0.11-0.49 toward translation's 0.98. Representation errors have already been
decisive once. This particular frame change is not the remaining one.

## 3. SE(3) equivariance — not testable as posed, for a concrete reason

The regressand is a 6-vector of translation and rotation increments **divided elementwise by**
`OUT = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`. That normalisation mixes metres and radians into one
vector with no common metric, so a rigid-body group action on it is not well defined without first
undoing `OUT` and declaring a frame and origin for each block.

A genuine SE(3) equivariance constraint is also a different object from a frame change: the frame
change in §2 is a single fixed transformation of coordinates, whereas equivariance is a constraint
relating the model at *transformed* inputs to the transformed model output. `learned_adaptation/`
already implements the discrete morphological version of that constraint and its commutant
requirement, and the §2 result suggests the continuous rigid-body version would buy little at the
failing cell — but that is an inference, not a measurement, and it is the second track's to check.

## 4. What this changes

Of the three, only the non-diagonal map is supported at the failing cell, and only open-loop. The
decisive open question is the one my measurement cannot reach: **does a coupled plant model still
help once the correction is applied and the command leaves the identification distribution?** That
is one cell — joint 5 with `--mimo` — against the stored `12/20 -> 3/20` baseline, and it is the
cheapest experiment either of these directions admits.

---

# RE-VERIFICATION — two of my three verdicts were wrong, for the same reason

**2026-09-08, after G8 established that 44–71% of the residual is the FIR's missing autoregressive
term.**

## The error

I tested all three directions on the raw FIR residual `r`. That quantity is **dominated by
unmodelled plant dynamics**, not disturbance: adding the two previous measured-motion increments
removes **54.0% of it at joint 3 and 78.2% at joint 5**.

So I was measuring the frame, coupling and state-dependence **of the model error**, and reporting
conclusions about the disturbance. All three tests were run on the wrong quantity.

## Corrected results, on the AR-corrected residual

| direction | my earlier verdict | corrected |
|---|---|---|
| **body frame**, joint 5 | **−3.6%**, "not supported" | **+0.6%** — sign flipped |
| body frame, joint 3 | +3.4% | +0.2% |
| **coupling**, joint 5 | +5.5% | **+5.4%** — holds |
| coupling, joint 3 | +3.1% | +3.1% — holds |

Coupling decomposes: at joint 5, coupling the command terms gives +3.6% and coupling the
autoregressive terms +2.5%, jointly +5.4%.

## What each verdict now is

**Body frame — my refutation was wrong and is withdrawn.** On the correct quantity it is a small
positive at both cells rather than negative at the failing one. But +0.2% and +0.6% is not support
either; it is "no longer contradicted, and too small to build on."

**Non-diagonal — my verdict holds.** Supported at both cells, and more strongly at the failing one,
and the closed-loop cell I ran confirmed it neither rescues joint 5 nor harms joint 3.

**SE(3) — my dismissal was too quick.** I argued the `OUT = [.05,.05,.05,.5,.5,.5]` normalisation
mixing metres and radians blocks a well-defined rigid-body action. The second track states directly
that *"mixed translation/rotation units do not prevent a well-defined rotation action on these
normalized increments."* That objection of mine does not stand, and the direction remains untested
rather than refuted.

## The finding that subsumes all three

Every one of these effects is **0.2% to 5.4%**. The missing autoregressive term is **54% to 78%**.

They are second-order refinements on top of a first-order structural defect, and I spent the round
measuring the refinements while the defect was still in the residual. That is the honest summary of
what went wrong, and it applies to the body-frame test, the coupling test, and the earlier
state-dependence gate equally.

The first-order fix is the `A x` term, which the descriptor form has and the FIR does not — and
which has still never been run.
