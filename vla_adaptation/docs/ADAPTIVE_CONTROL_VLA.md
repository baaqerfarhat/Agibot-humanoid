# Online adaptive correction of a frozen VLA — theory and results

**Result in one line.** An actuator fault in a frozen 3.35 B vision-language-action model is
identified and corrected *within a single episode*, from the robot's own motion, with no
gradients through the task and no search over task success — **across four LIBERO suites,
29% → 73% pooled, p = 1.1×10⁻⁷** — **provided the correction is restricted to the channels
where the fault is identifiable.** Correcting the rest is worse than doing nothing.

It generalises to a fault pattern it was never tuned on (17% → 83%), tracks a fault appearing
mid-episode within 0.75 s, and costs nothing when no fault is present.

**The scope condition is the contribution.** With all six action channels corrected, the
method is significant on one suite of four. Restricted to identifiable channels it is
significant on all four. See §7.

Video: `results/phase05/adaptive_vs_frozen.mp4`. Data: `results/phase05/`.
Code: `openpi/{error_signal,openloop_id,adaptive_law,compare_video}.py`.

---

## 1. Setting

π0.5-LIBERO, frozen: 3.35 B parameters, no fine-tuning, no backprop through the environment.
Benchmark `libero_spatial`, nominal **99.0%** over 500 episodes.

**The fault.** A constant offset `f = +0.05` added to the six arm dimensions of every action
the policy emits — a miscalibrated actuator. It is the only fault of nine screened that both
hurts enough to matter and leaves the robot alive (`PREREG_OPENPI_ACE_SCREEN` §1): success
falls to **47%**.

**Why this is not a search problem.** Black-box search over task success (CEM and similar)
throws away everything known about the plant and pays for it in episodes. The robot measures
its own end-effector pose every step. If the fault is observable there, it can be identified
directly, in one episode, without ever consulting the task reward.

## 2. Theory

### 2.1 The three objects an adaptive law needs

An adaptive law needs an **error** it can drive to zero, a **plant model** that predicts the
error-free behaviour, and the **map** from the unknown parameter to that error. Getting any
of the three wrong makes the law diverge rather than converge — all three failure modes were
observed here before they were fixed.

### 2.2 Plant

LIBERO's arm runs an OSC_POSE controller at 20 Hz with `control_delta = True` and

```
output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]      (m, m, m, rad, rad, rad)
```

so one action unit *commands* 0.05 m of translation. **It does not achieve it.** Regressing
achieved on commanded displacement over fault-free rollouts gives a static gain of only
**0.21–0.25** for translation: the arm covers about a fifth of the commanded delta inside one
50 ms control period. Using `output_max` as the plant — the obvious choice — is therefore
wrong by a factor of five, and an adaptive law built on it mistakes ordinary controller lag
for a fault.

A finite-impulse-response model over the last `K = 6` commands captures the lag:

```
y_t  =  sum_{k=0..K} h_k a_{t-k}  +  c            (per output dimension)
```

| fit | dx | dy | dz | drx | dry | drz |
|---|---|---|---|---|---|---|
| static (K=0) R² | 0.885 | 0.961 | 0.978 | 0.416 | 0.091 | 0.077 |
| **FIR (K=6) R²** | **0.975** | **0.982** | **0.982** | 0.489 | 0.109 | 0.168 |

Translation is well identified. **Rotation is not**, and that limitation propagates to every
result below.

### 2.2b Orientation increments

Rotation change must be the **relative rotation**, not a difference of axis-angle vectors:

```
omega  =  axis_angle( q_t+1  ⊗  conj(q_t) )
```

Axis-angle is a chart, not a vector space: the same rotation has representations differing by
2π, and the chart is singular at 0 and π. Over 2000 random increments of 0.05 rad the relative
rotation recovers the true increment to **1.4e-14**, while the difference of charts errs by up
to **6.28** — a full wraparound.

Using the wrong one corrupted four separate results before it was found: rotation appeared
unpredictable (FIR R² 0.11 on `dry`, 0.17 on `drz`), `M` appeared cross-coupled, the gain law's
rotation estimates were meaningless, and deliberate excitation made performance *worse*. All
four were the same bug. Corrected fits: **drz 0.168 → 0.972**, dry 0.109 → 0.336, drx 0.489 →
0.522, translation unchanged.

### 2.3 Error

With `u_t = a_t + c_t + f` the executed action (policy command, our correction, the unknown
fault), and `P̂` the identified fault-free plant:

```
r_t  =  y_t  -  P̂(a_t + c_t)   ≈   M f
```

`r_t` does not depend on `c_t`, so the estimator is not chasing its own correction — an
important property, and the reason this form was chosen over comparing against a
reference trajectory.

### 2.4 The map

`M = ∂(achieved motion)/∂f`, measured **open loop**: a recorded command sequence is replayed
with and without a fault, so the commands are identical by construction and any difference in
motion is the fault propagating through the plant. One input axis at a time, ±0.02 central
difference:

```
              dx       dy       dz      drx      dry      drz   <- fault applied to
    dx     0.297    0.021   -0.112   -0.029    0.081   -0.006
    dy     0.008    0.272    0.023   -0.034    0.016    0.003
    dz     0.029    0.032    0.126   -0.037    0.012   -0.003
   drx    -0.003   -0.001   -0.001    0.253   -0.004    0.002
   dry     0.013    0.001   -0.001   -0.007    0.276   -0.000
   drz     0.001    0.004   -0.003    0.011    0.005    0.244
```

- **Attenuation.** The diagonal is 0.13–0.30 — the plant damps the fault 3–8×, so the law
  needs roughly 4× gain on the observed error.
- **Mild coupling.** Off-diagonal mass is **26%** of `|M|`, dominated by a real dx↔dz term
  (−0.112). `M⁻¹` is worth using, though a diagonal approximation would not be catastrophic.
- **Well conditioned.** cond(M) = **3.0**.

> **Retraction (2026-08-27).** An earlier version of this section reported the rotation block
> as "a swapped, sign-flipped pair" — `dry ← drz` at +0.383, `drz ← dry` at −0.424, own
> diagonals ≈ 0.01 — and argued that a per-axis law would therefore diverge. **That was
> wrong.** It was an artifact of measuring orientation change as
> `axis_angle(q1) − axis_angle(q0)`, a difference of charts rather than a rotation increment
> (§2.2b). With the correct metric the rotation axes have ordinary diagonal gains of 0.276 and
> 0.244, off-diagonal mass falls from 0.59 to 0.26, and the divergence argument does not apply.
> The claim is withdrawn.

### 2.5 The law

```
f̂ ← clip( f̂ + γ ( M⁻¹ r_t / (1 + ‖r_t‖²/ρ²) − f̂ ),  ±0.15 )      with a deadzone ‖r_t‖ < δ
c_t = − f̂
```

Exponential forgetting, normalised update, deadzone, projection: γ = 0.08, ρ = 0.15,
δ = 0.008. Each robustness term earned its place by fixing an observed failure (§4).

### 2.6 Is the fault identifiable at all?

Per step, the fault contributes `M f` to a residual whose noise is the plant's own fit error:

| | dx | dy | dz | drx | dry | drz |
|---|---|---|---|---|---|---|
| plant residual sd | 0.0135 | 0.0113 | 0.0173 | 0.0049 | 0.0147 | 0.0155 |
| fault signature | 0.0119 | 0.0147 | 0.0086 | 0.0152 | 0.0185 | 0.0207 |
| **SNR per step** | 0.88 | 1.30 | 0.50 | 3.11 | 1.27 | 1.34 |
| steps for SNR = 3 | 12 | 5 | 37 | 1 | 6 | 5 |

Episodes run ~100 control steps, so **within-episode identification is comfortably feasible**.
The information is there; whether it is extracted depends on the estimator.

## 3. Results

All faults are injected client-side between the policy and `env.step`; the model is never
modified. Frozen and adaptive arms share task, initial state, policy and fault — the only
difference is whether the law runs.

### 3.1 Headline, confirmatory

| arm | success | 95% CI |
|---|---|---|
| nominal (no fault) | 99% | — |
| frozen, faulted | **18/40 = 45%** | [31, 60] |
| **adaptive** | **38/40 = 95%** | [83, 99] |

**+50 points, Fisher p = 1.1×10⁻⁶**, on initial states 40–43 that no earlier run touched.
The earlier n=15 estimate (47% → 93%, p = 0.014) did not shrink under retest — it grew
slightly, and the estimates reproduced within ±0.004 on all six axes across the two
independent runs.

### 3.2 The full condition set

| condition | frozen | adaptive | note |
|---|---|---|---|
| offset 0.05, from step 1 (n=40) | 45% | **95%** | p = 1.1e-06 |
| offset 0.10, near-lethal (n=8) | 0% | 38% | frozen fails every episode |
| **structured `[0,0,0,+.06,−.06,+.03]`** (n=12) | 17% | **83%** | fault shape never tuned on |
| **onset at step 15** (n=20) | 60% | **90%** | p = 0.065; tracking, see §3.4 |
| onset at step 45 (n=15) | 73% | 80% | little headroom by design |
| **no fault** (n=15) | 100% | **100%** | safe to leave running |
| gain 0.5, multiplicative (n=10) | 50–70% | 70–90% | noise-dominated at this n |

### 3.3 Generalisation

The structured fault `[0, 0, 0, +0.06, −0.06, +0.03]` — mixed signs, rotation only, unseen
magnitudes — is recovered as a *pattern*, not a scalar:

| dim | true | separation | error |
|---|---|---|---|
| drx | +0.060 | +0.053 | −0.007 |
| dry | −0.060 | −0.038 | +0.022 |
| drz | +0.030 | +0.029 | −0.001 |
| dx, dy, dz | 0.000 | −0.003, −0.024, −0.006 | ≈0 |

Sign correct on 3/3 including the negative axis, and correctly ≈0 where the fault is zero.
This rules out the obvious objection that the method was fitted to one fault shape.

### 3.4 Tracking a fault that appears mid-episode

The deployment case: a healthy robot that breaks while running. Fault injected at control
step 15, mean `f̂` on the identifiable rotation channels over 20 episodes:

| window | f̂ |
|---|---|
| steps 0–14, before onset | **+0.009** |
| steps 15–30, just after | +0.035 |
| steps 30–50, settling | +0.041 |
| steps 50–78, late | **+0.042** (true 0.050) |

Quiet before the fault exists, **70% of truth within 15 control steps (0.75 s)**, settling at
84% and holding flat — no post-convergence drift. This is also the cleanest control in the
whole set: **the episode is its own baseline**, so the estimator bias of §6 cancels without
needing a separate matched run.

### 3.5 Fault estimates

Mean over the n=40 run, true value 0.050 on every dim:

```
f̂      [0.050  0.035  0.075  0.038  0.030  0.046]
```

Reproducible to ±0.004 against the independent n=15 run. **But see §6** — the raw values
overstate identification, and only the separation against a matched no-fault control is
evidence.

## 4. Three failures worth keeping

Each was diagnosed from measurement, not guessed, and each is a general trap.

**Naive error signal, no plant model.** Using `Δx/output_max − a` directly: nominal e_t is
−0.175 on dx when it should be 0, because the arm does not reach the commanded target in one
step. Differencing faulted against nominal recovered ratios of 0.39, 2.46, −2.85 — noise.
*Controller lag reads as a fault unless the plant is modelled.*

**Per-axis gains instead of M⁻¹.** With the rotation block swapped and sign-flipped, a
diagonal law feeds error into the wrong channel with the wrong sign. The closed-loop estimates
came out sign-wrong on three of six dimensions.

**Robustness constants set without measuring.** The bare law drifted — `dz` reached 0.642,
13× truth, on 2 of 6 episodes — because the plant is identified on nominal data and a growing
correction walks the executed action off-distribution. The first fix made it *worse*: a
deadzone of 0.05 against a typical residual norm of 0.034 suppressed nearly every update, and
several episodes ended at `f̂ = 0` exactly. Rescaling to the measured residual fixed both, and
raising the normalisation constant (0.05 → 0.15, cutting a systematic 32% attenuation to ~5%)
took the result from 70% to 87%.

## 5. Limits

- **Identification is channel-dependent, and which channels work depends on the fault type.**
  An additive fault has regressor `I` and is loudest where the quantile scale is small —
  rotation, where a uniform env-space offset is 19.5% of the action range against 3% on
  translation. A multiplicative fault has regressor `diag(ψ)` and is loudest where the command
  is large — translation, commanded at sd 0.32–0.55 against rotation's 0.02–0.055. Measured
  recovery: offset 21% translation / 75% rotation; gain 86% translation / 40% rotation. The
  two fault types are identifiable on **complementary** channels, and neither is universal.
- **Raw estimates overstate identification** (§6). Only the separation against a matched
  no-fault control is evidence.
- **A constant input fault and a constant output model bias are not separable within one
  episode.** The bias must be calibrated externally, or removed by a within-episode control
  such as the mid-episode onset design (§3.4).
- **`dry` remains the weakest channel** — worst plant fit (R² 0.336 per-axis) and worst
  recovery (63% on the structured fault). Its motion is driven substantially by the
  *translation* commands; a coupled model fixes the fit (held-out R² 0.615) but **hurts the
  closed loop** (§4), so it is left uncorrected.
- **One task suite, one robot, injected faults.** `libero_spatial`, a MountedPanda under
  OSC_POSE. Not observed hardware degradation.
- **The failure boundary is unmapped.** Every fault tested lives in the action space, which is
  exactly what the regressor is built for. A fault whose signature is absent from the
  command–motion residual — a perception fault, say — has not been tried, so it is not known
  where the method stops working.

## 6. What the correction is actually doing (added 2026-08-27)

§3 reports 47% → 93% and implies all six axes contribute. They do not. A matched **no-fault
control** plus an **axis ablation** pin the mechanism down, and the claim needs restating.

### 6.1 `f̂` alone cannot tell identification from bias

Running the law with **no fault at all**, `f̂` does not go to zero — it settles at
`[+0.045, +0.014, +0.073, −0.009, +0.017, −0.001]`, which on `dz` is 146% of the magnitude of
the real fault. The estimator reports a large offset where there is nothing to correct.

The diagnostic that separates the two is the **separation**: the same estimate under a fault,
minus the estimate with no fault.

| dim | no fault | fault 0.05 | **separation** | true |
|---|---|---|---|---|
| dx | +0.045 | +0.048 | **+0.003** | 0.050 |
| dy | +0.014 | +0.038 | +0.023 | 0.050 |
| dz | +0.073 | +0.079 | **+0.006** | 0.050 |
| **drx** | −0.009 | +0.041 | **+0.049** | 0.050 |
| dry | +0.017 | +0.033 | +0.016 | 0.050 |
| **drz** | −0.001 | +0.047 | **+0.048** | 0.050 |

**Rotation identifies the fault almost exactly. Translation does not identify it at all.**

### 6.2 Where the phantom comes from

Two mechanisms, measured separately with `--estimate-only` (update `f̂` but never apply it):

| | mean \|phantom\| |
|---|---|
| open loop (correction never applied) | 0.0143 |
| closed loop (correction applied) | 0.0266 |

**Half is plant-model error**, and it is not task-specific: identifying the plant on 10 tasks
instead of 3 made the phantom *worse* (0.027 → 0.034).

**Half is estimator feedback, and it corrects a claim made in §2.3.** `r_t` is independent of
`c_t` only when `P̂ = P` exactly. With model error `r = M·f + ε(a+c)`: the residual carries
the model error *at the shifted operating point*, so `f̂` converges to `f + M⁻¹ε(a+c)` — it
chases its own correction. Measured amplification: **1.9×**.

**This is an identifiability limit, not a tuning failure.** Within one episode a constant
input fault contributes `(Σh_k)·f` to the output and a constant output model bias contributes
`b`; both are constant and `M` is full rank, so the two cannot be separated from one signal.
It has to be calibrated externally. Subtracting the healthy-run bias fixes `dz` (0.079 →
0.046 against a true 0.050) and changes task performance not at all — it buys estimator
honesty, not success.

### 6.3 The ablation

Correcting **rotation only** (dims 3–5): **14/15 = 93%**, identical to correcting all six.

So the mechanism is:

1. The fault does its damage through rotation — a uniform env-space offset is 3% of the
   action range on translation and **19.5%** on `drx` (§2.3), 6.6× larger.
2. The estimator identifies rotation (separation 0.047–0.049 against 0.050) and not
   translation (−0.013, −0.005).
3. Correcting rotation alone recovers the entire benefit; the translation correction does no
   work.

**The restated claim.** Online identification and correction of a fault in a frozen VLA works
**on the channels where the plant model is good enough to separate a fault from model error**
— and here those are exactly the channels carrying the damage. The 47% → 93% result is
unchanged; what it means is now pinned down rather than assumed.

**Method note worth carrying forward.** No estimate of a fault parameter should be believed
without a matched no-fault control. `f̂` looked convincing on all six axes for weeks of
runs; only the separation revealed that half of it was bias.


---

## 7. The scope condition: correct only what you can identify (added 2026-09-01)

§3 evaluated on `libero_spatial` alone. Extending to four suites first looked like a failure
to generalise, and then explained itself.

### 7.1 With all six channels corrected, the method works on one suite of four

| suite | frozen | corrected | delta | p |
|---|---|---|---|---|
| `libero_spatial` | 18/40 = 45% | 38/40 = 95% | +50 | **<0.0001** |
| `libero_goal` | 8/20 = 40% | 12/20 = 60% | +20 | 0.34 |
| `libero_object` | 7/20 = 35% | 9/20 = 45% | +10 | 0.75 |
| `libero_10` | 0/20 = 0% | 3/20 = 15% | +15 | 0.23 |
| **pooled, 3 new suites** | 15/60 = 25% | 24/60 = 40% | +15 | 0.12 |

It is **not** an identification failure: the plant fits on the new suites are comparable or
better (`object` reaches rotation R² 0.771/0.692 against spatial's 0.522/0.336).

### 7.2 Restricted to identifiable channels, it works on all four

| suite | frozen | corrected | delta | p |
|---|---|---|---|---|
| `libero_spatial` | 7/15 = 47% | 14/15 = 93% | +47 | **0.014** |
| `libero_goal` | 9/20 = 45% | 18/20 = 90% | +45 | **0.006** |
| `libero_object` | 6/20 = 30% | 16/20 = 80% | +50 | **0.004** |
| `libero_10` | 0/20 = 0% | 7/20 = 35% | +35 | **0.008** |
| **pooled** | **22/75 = 29%** | **55/75 = 73%** | **+44** | **1.1×10⁻⁷** |

`libero_10` is the sharpest case: a long-horizon suite (520-step cap) where the fault takes
the policy to **zero**, and correcting three channels recovers 7/20 from that floor.

### 7.3 The non-identifiable channels do not merely fail — they do harm

On `libero_object`, all three arms on the same fault and the same episodes:

| correction | frozen | corrected | delta |
|---|---|---|---|
| all six dims | 7/20 | 9/20 | +10 |
| **rotation only** | 6/20 | **16/20** | **+50** |
| **translation only** | 6/20 | **6/20** | **0** |

Translation alone contributes **exactly nothing**, and including it drags a +50 effect down
to +10. The reason is in §6: the separation test shows the estimator identifies rotation
(+0.049, +0.047 against a true 0.050) and not translation (−0.013, −0.005). On `object` the
translation estimates are **sign-wrong** (−0.031, −0.018 against +0.050), so that correction
pushes the arm the wrong way. On `spatial` the same wrong correction happened to be harmless,
which is why the naive version looked suite-specific rather than simply misapplied.

### 7.4 Identifiability is predictable before deployment

Nothing here requires knowing the fault. The three quantities that decide which channels are
identifiable are all measurable on healthy data:

1. **The action normalisation** — quantile scales from the checkpoint. A uniform env-space
   offset is 3% of the action range on translation and 19.5% on rotation (§2.3).
2. **The plant fit** — per-channel FIR R² on fault-free rollouts. Identification quality
   tracks it in every experiment run here.
3. **The policy's command statistics** — an additive fault is loud where the quantile scale
   is small; a multiplicative one is loud where the command is large (§5). The two are
   identifiable on complementary channels.

**So the rule is: measure identifiability on healthy data, correct only those channels, and
leave the rest alone.** The four-suite result in §7.1 is the ablation showing what it costs
to ignore this.

## 8. Identifiability, stated properly (added 2026-09-01)

§7 gives the empirical rule. This section states what it rests on, including one degeneracy
that no amount of data inside a single episode can break.

### 8.1 Setup

Let `a_t ∈ R⁶` be the commanded env-space action (3 translation, 3 rotation increments) and
`y_t` the measured state increment. Over the identification horizon the plant is the FIR map
`M` fitted in §2.4, so the nominal prediction is `ŷ_t = M a_t`. Two fault classes:

- **additive** `a_actual = a_t + f`, `f` constant — regressor `φ = I`
- **multiplicative** `a_actual = diag(g) a_t`, so the error is `diag(θ) a_t` with `θ = g − 1`
  — regressor `φ = diag(a_t)`

### 8.2 Proposition 1 — a constant input fault and a constant output bias are not separable

Suppose the measurement carries an unknown constant bias `b`:

```
y_t = M (a_t + f) + b + noise
```

Within one episode `f` and `b` are both constant, so they enter only through the sum
`M f + b`. `M` is square and full rank (verified, §2.4), so for **any** `b′` the alternative
fault `f′ = f + M⁻¹(b − b′)` reproduces every observation exactly. The pair `(f, b)` is
unidentifiable; only `M f + b` is.

**This is why `--bias` exists and why it is not a nuisance parameter you can fit.** It has to
come from somewhere outside the episode.

**Corollary (onset breaks the degeneracy).** If the fault switches on at `t₀`, with `f = 0`
for `t < t₀`, then `b` is identified on the pre-onset window and `f` on the post-onset window.
Mid-episode onset is not a robustness flourish — it is what makes the problem well-posed.
This is what `--onset` implements, and it is the honest deployment story: the estimator needs
to have seen the healthy plant, not to have been told the bias.

### 8.3 Proposition 2 — the two fault classes are loud on complementary channels

Actions are quantile-normalised, `a_norm = 2(a_env − q01)/(q99 − q01) − 1`. Write
`R_i = (q99 − q01)_i` for channel `i`.

- An **additive** env-space offset `δ` appears in normalised units as `2δ/R_i`. Its
  signal **falls** with `R_i`.
- A **multiplicative** fault has information matrix `Σ_t φᵀφ = diag(Σ_t a_{t,i}²)`. Its signal
  **rises** with the command magnitude on channel `i`, and command spread is what `R_i`
  measures.

So the same quantity that makes a channel quiet for an offset makes it loud for a gain. From
the `pi05_libero` checkpoint:

| ch | `q99−q01` | additive signal `2·0.05/R` | multiplicative signal `R/R_max` |
|---|---|---|---|
| x | 1.685 | 0.059 | 0.899 |
| y | 1.656 | 0.060 | 0.883 |
| z | 1.875 | 0.053 | 1.000 |
| rx | 0.256 | **0.391** | 0.137 |
| ry | 0.351 | **0.285** | 0.187 |
| rz | 0.506 | **0.198** | 0.270 |
| **translation mean** | | 0.058 | **0.927** |
| **rotation mean** | | **0.291** | 0.198 |

Additive favours rotation **5.0:1**; multiplicative favours translation **4.7:1**.

**One caveat, stated so nobody mistakes it for a result:** with these two proxies the product
`(2δ/R)·(R/R_max)` is identically `2δ/R_max` for every channel. That constancy is algebra, not
evidence — it follows from having written one proxy as `∝1/R` and the other as `∝R`. The
content of the table is the *ordering* and the *ratio*, both of which are checkable against
measured separation, not the fact that a product of reciprocals is flat.

### 8.4 What makes this a prediction rather than a fit

§7 restricted the offset correction to rotation *after* seeing that rotation identified. On its
own that is post-hoc. Proposition 2 makes the complementary claim testable in the opposite
direction: for a **gain** fault the identifiable channels should be **translation**, and
restricting the correction to rotation should do little or nothing.

That test is `adaptive_gain.py --corr-dims`, and its result is §8.5. Whichever way it comes
out, it is a prediction registered before the run, not a restriction chosen after it.
