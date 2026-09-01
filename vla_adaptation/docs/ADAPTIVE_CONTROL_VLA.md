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
to +10.

> **Read with §14 (2026-09-01).** This is measured at the uniform six-axis fault at 0.05,
> where the translation component barely damages the policy. It is a fact about *that fault*,
> not about the translation channel: on a translation-only fault at 0.15 the same correction
> takes 20% → 95%. The reason is in §6: the separation test shows the estimator identifies rotation
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

> **Superseded, 2026-09-01.** §11 refutes the mechanism claimed here. Translation is not
> structurally quiet — it identifies a single-axis fault at 79% of truth. What actually
> decides identifiability is the fault's *shape* relative to the robot's reachable set, not
> the quantile scale, and that is not readable from healthy action statistics alone. The
> empirical recommendation (correct rotation only) is unchanged and still works; the
> prediction recipe in this subsection is withdrawn. Read §11 before using §7.4.

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

> **Partly superseded, 2026-09-01.** The ordering below is real (rotation identifies at
> 101% against translation's 79% on matched single-axis faults), but the *cliff* this
> subsection implies is not: translation is not structurally unidentifiable. See §11.

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

### 8.5 The prediction, resolved: right about identification, wrong about repair

Run on 2026-09-01, `gain = 0.5` (a 50% loss of effectiveness on all six axes), n = 20 paired
episodes per arm, correction restricted to one channel group at a time.

> **Superseded, 2026-09-01 (§12).** The table below reports *raw* `β̂` against truth, with
> no matched no-fault control — the exact error §6 of this document warns against. With the
> control applied, x does not identify (1.4σ) and `ry` does (5.1σ), so the clean
> complementary split claimed here does not survive. Read §12 before using this subsection.

**Identification — the prediction holds, decisively.**

| run | `β̂` translation | `β̂` rotation | true |
|---|---|---|---|
| correcting translation | −0.616, −0.498, −0.800 | −0.029, −0.019, −0.071 | −0.50 |
| correcting rotation | −0.463, −0.490, −0.797 | −0.017, +0.020, −0.151 | −0.50 |

In both runs translation recovers the fault and rotation returns essentially zero (3–30% of
truth). This is the exact mirror of the offset fault, which identified on rotation
(+0.047…+0.049 against 0.050) and not on translation. Proposition 2 predicted the reversal
before the run, and §8 was committed while it was still computing. **Which channels carry a
fault is predictable from healthy statistics alone, and the prediction transfers across
fault classes.**

**Task success — the prediction fails.**

| arm | success |
|---|---|
| frozen, faulted | 13/20 = 65% |
| correcting translation (identifiable) | 12/20 = 60% |
| correcting rotation (non-identifiable) | 14/20 = 70% |

Neither is distinguishable from frozen at n = 20. Correcting the *identifiable* channels did
not help. So the offset chain — identify, cancel, recover — **does not transfer to a gain
fault**, and it is worth being precise about why, because the obvious explanation is wrong.

### 8.6 Why: the gain compensator amplifies its own estimation error

An external review proposed that the compensator is a first-order subtraction that
undercompensates at `g = 0.5`. It is not. The implemented correction is

```
c = a·(1/ĝ − 1),   ĝ = 1 + β̂   ⟹   a_corr = a + c = a/ĝ
```

which is the **exact** certainty-equivalent inverse gain: at `β̂ = −0.5` it commands `2.00×`
and the plant executes `0.5 × 2.00a = a` exactly. Verified symbolically and numerically.

The problem is the opposite of undercompensation. The applied factor `1/(1+β̂)` has
derivative `−1/(1+β)² = −4` at the true `β = −0.5`, so **the compensator amplifies estimation
error fourfold**, and it does so nonlinearly, blowing up as `ĝ → 0`:

| channel | `β̂` | applies | correct | error |
|---|---|---|---|---|
| dx | −0.463 | 1.86× | 2.00× | −7% |
| dy | −0.490 | 1.96× | 2.00× | −2% |
| **dz** | **−0.797** | **4.93×** | **2.00×** | **+146%** |

A 0.30 error in `β̂_z` — modest, and *better* than what several offset channels tolerate
fine — becomes a 2.9× excess command on the vertical axis. That is enough to wreck the
episode on its own, and it explains how identification can be good while repair is not.

**The structural point, which is not specific to this implementation:** an additive fault's
compensator is *linear* in the estimate, so estimate error passes through at unity gain. A
multiplicative fault's compensator *inverts* the estimate, so error passes through at
`1/ĝ²`. **Identification accuracy sufficient for an additive fault is not sufficient for a
multiplicative one, and the gap grows as authority is lost — exactly when repair matters
most.** Identifiability (§8.3) is necessary but not sufficient; the compensator's
conditioning is a second, independent requirement.

The implied fix is projection rather than a better regressor: refuse to invert a gain we do
not believe. `--g-min` floors `ĝ` at a physical prior (below it, decline to repair rather
than command a large multiple). It is a prior on the robot, not on the answer, and is not
centred on the true 0.50.

### 8.7 Projection helps the mechanism and does not rescue the task

`--g-min 0.35`, correcting translation only, `gain = 0.5`, n = 20:

| arm | success | `β̂` translation |
|---|---|---|
| frozen, faulted | 15/20 = 75% | — |
| corrected, projected | 17/20 = 85% | −0.487, −0.481, −0.797 |

Projection does what it was built to do: `ĝ_z = clip(0.203, 0.35)` caps the vertical
overcommand at **2.86×** instead of 4.93×, against a correct 2.00×. The mechanism improved.

**The +10 points are not evidence.** Across three runs of the *identical* frozen condition —
same fault, same flags — the frozen arm scored **13/20, 13/20, 15/20**. The harness is
nondeterministic (π0.5 samples its flow), so ±2/20 = ±10 points is free variation, and the
corrected arm's margin here sits inside it. **The gain fault is not repaired.** Reporting
this as a +10 improvement would be reading noise.

Two things this does establish, both from the estimates rather than the successes:

1. ~~**`β̂_z = −0.797` in all three runs, to three decimals.**~~ **Resolved in §12.1: this is
   the `--clip 0.8` projection bound, not an estimate.** It reproduces to three decimals
   because it is a constant. The estimator is diverging on z and being held by the clip.
2. **n = 20 cannot resolve what is being asked of it.** With ±10 points of free variation, a
   real repair effect would need to be very large to show. This is precisely why the paired
   record matters (§8.4 of RESPONSE.md): McNemar conditions on the discordant pairs and
   removes the shared episode-difficulty variance that is drowning the signal here. The
   gain runs still lack `per_ep`.

**Standing conclusion on the gain fault:** identification is solved and predicted
(§8.5); repair is not demonstrated. The barrier is compensator conditioning (§8.6)
compounded by a reproducible bias on one channel, not the choice of regressor.

## 9. Superposition, not curvature: a correction to this section (rewritten 2026-09-01)

**This section originally claimed that `M` was fitted across a range over which the plant is
nonlinear, and that translation is nonlinear in fault magnitude. Both claims were wrong, and
the measurement that refutes them is below. The corrected finding is narrower, and has a
different mechanism and a different fix.**

### 9.1 What was wrong

`M` is not fitted from the four uniform-magnitude probe rows. `openloop_id.py` builds it from
**per-axis central differences at ±0.02** — one axis at a time. The four-magnitude `rows` are
a separate diagnostic that never enters `M`. The original §9.1 read the CV of those rows as
evidence of per-axis nonlinearity; they measure something else entirely.

Measuring `M` directly at the operating point settles it. `--probe 0.05` against `--probe
0.02`, per-axis diagonals:

| | x | y | z | rx | ry | rz |
|---|---|---|---|---|---|---|
| M @ 0.02 | 0.297 | 0.272 | 0.126 | 0.253 | 0.276 | 0.244 |
| M @ 0.05 | 0.295 | 0.251 | 0.144 | 0.251 | 0.251 | 0.240 |
| ratio | 0.99 | 0.92 | **1.14** | 0.99 | 0.91 | 0.99 |

Every channel agrees within 14%, and `cond(M)` *improves* at the larger probe (3.0 → 2.2).
**Per axis, the plant is linear to 0.05, and translation is no worse than rotation.** The
claim that `M` over-predicts translation by 2.4× is withdrawn.

### 9.2 What is actually true: superposition fails on translation

The 2.4× came from comparing `M`'s **row sums** against the **uniform six-axis** probe. That
is a test of superposition, not of magnitude linearity. Measured uniform-fault sensitivity
divided by the sum of the single-axis columns:

| probe `f` | x | y | z | rx | ry | rz |
|---|---|---|---|---|---|---|
| +0.01 | 0.96 | 0.83 | 0.89 | 1.02 | 0.99 | 1.01 |
| +0.02 | 0.88 | 0.77 | 0.84 | 1.04 | 0.95 | 1.02 |
| +0.05 | **0.07** | 0.91 | **0.50** | 1.06 | 0.67 | 0.89 |
| −0.05 | **0.78** | 1.10 | **0.27** | 0.88 | 1.11 | 0.93 |

At ±0.01 and ±0.02 superposition holds everywhere (0.77–1.04). At ±0.05 it collapses on
translation and **asymmetrically**: x gives 0.07 in one direction and 0.78 in the other; z
gives 0.50 and 0.27. Rotation superposes at every magnitude tested (0.67–1.11).

**Direction-dependence is the signature of a contact or a joint limit, not smooth
curvature.** A uniform +0.05 on all six axes drives the arm into something that a uniform
−0.05 does not. Each axis alone is linear; all six together at 0.05 are not, because the
combination puts the arm somewhere the individual probes never reach.

### 9.3 What survives, and what it means

The practical conclusion from the original §9 stands, for a different reason: **`M⁻¹` applied
to the translation residual of a uniform six-axis fault at 0.05 is not measuring what it
claims to.** But the cause is the fault *shape and operating point*, not a defect in `M`, and
the proposed fix — refit `M` on the linear region — is moot, because `M` was already fitted
there and does not change when refitted at 0.05.

It remains a confound for the scope condition (§7, §8), now with a concrete mechanism:
translation correction may fail because superposition fails for this fault, not because the
quantile scale makes translation quiet.

### 9.4 The distinguishing experiment, corrected

Since `M` is magnitude-independent, refitting proves nothing. The two accounts separate on a
**single-axis translation fault at 0.02**, where superposition cannot fail (one axis) and `M`
is valid:

- **Proposition 2** predicts translation still fails to identify — the quantile scale is
  unchanged, so the SNR argument is untouched.
- **The superposition account** predicts translation should now identify cleanly, since the
  only reason it failed has been removed.

This is an estimation test and needs no task rollouts.

### 9.5 Still unexplained

`β̂_z = −0.797` remains open. Neither account explains it: the z diagonal is *larger* at the
0.05 probe (0.144 vs 0.126), which would push the estimate down, not up.

## 10. The paired analysis, finally computable (added 2026-09-01)

Every p-value previously reported was Fisher's exact, which treats the two arms as
independent samples. They are not: both run the same `(task, init)` episodes with the same
policy and seeds. The runs that produced those numbers stored only per-arm totals, so the
pairing was unrecoverable. These are fresh runs with per-episode records.

Rotation-only correction, `sev = 0.05`, n = 20 per suite.

| suite | frozen | adaptive | adaptive-only | frozen-only | exact McNemar |
|---|---|---|---|---|---|
| `libero_spatial` | 8/20 = 40% | 18/20 = 90% | 10 | 0 | **0.0020** |
| `libero_goal` | 8/20 = 40% | 13/20 = 65% | 5 | 0 | 0.0625 |
| `libero_object` | 5/20 = 25% | 16/20 = 80% | 11 | 0 | **0.00098** |
| `libero_10` | 0/20 = 0% | 5/20 = 25% | 5 | 0 | 0.0625 |
| **pooled** | **21/80 = 26%** | **52/80 = 65%** | **31** | **0** | **9.3×10⁻¹⁰** |

### 10.1 The result that only paired data could show

**`frozen-only wins = 0`, in all four suites, across 80 paired episodes.** Every episode the
frozen policy solved, the corrected policy also solved. The correction never broke a working
episode.

This could not be seen in any earlier analysis. Unpaired totals are consistent with a method
that fixes fifteen episodes and breaks four; the paired record shows it fixed thirty-one and
broke none. For a method meant to run continuously on hardware, "never makes things worse"
is a stronger and more useful claim than the success delta, and it was invisible until the
pairing was kept.

### 10.2 Two suites do not reach significance, and the reason is power, not weakness

`goal` and `libero_10` both land at exactly `p = 0.0625`. That is not marginal evidence —
**it is the floor of the exact test.** With 5 discordant pairs all favouring the correction,
`2⁻⁵ × 2 = 0.0625` is the smallest p-value attainable; a perfect result cannot do better. The
test is saturated, not equivocal:

| discordant pairs, all one way | best attainable p |
|---|---|
| 4 | 0.125 |
| 5 | **0.0625** |
| 6 | 0.031 |
| 7 | 0.016 |

Reaching `p < 0.05` on these suites requires ≥6 discordant pairs, which at these effect sizes
means n ≈ 30–40. That is a sample-size decision, not a result.

### 10.3 Effect sizes moved, as §8.7 warned they would

Against the earlier Fisher-tested runs, `goal` came in at +25 where it previously showed
+45, and `libero_10` at +25 against +35. `spatial` (+50) and `object` (+55) held. This is
consistent with the ±10-point run-to-run variation measured in §8.7 and is the reason the
per-suite numbers should be quoted with intervals rather than as point estimates. **The
pooled effect is what survives: 26% → 65%, p = 9.3×10⁻¹⁰, with zero regressions.**

### 10.4 A note on the permutation test

The paired permutation test reports `p = 5×10⁻⁶` pooled, which is its own resolution floor at
200 000 iterations (`1/(N+1)`), not a disagreement with McNemar. With 31 discordant pairs all
in one direction, exact McNemar is the accurate figure.

## 11. The scope condition is about fault shape, not the quantile scale (added 2026-09-01)

§9.4 set up a test that separates two explanations for why translation never identified.
It has run, with the magnitude control, and it comes down against Proposition 2.

### 11.1 The measurement

Separation test (faulted estimate minus matched no-fault estimate) on the x axis:

| fault | separation on x | % of truth | largest off-axis leak |
|---|---|---|---|
| uniform six-axis @ 0.05 | −0.013 | **sign-wrong** | — |
| **single-axis @ 0.05** | **+0.0244** | **49%** | 0.0189 |
| **single-axis @ 0.02** | **+0.0158** | **79%** | 0.0089 |
| rotation `rx`, single-axis @ 0.02 | +0.0201 | 101% | 0.0040 |

### 11.2 What it says

**Fault shape dominates.** At the *same* magnitude 0.05, a single-axis fault identifies at
49% with the right sign while the uniform six-axis fault is sign-wrong. Nothing about the
quantile scale differs between those two conditions — only whether the other five axes are
also faulted. This is the superposition/contact mechanism of §9.2, and it is the primary
cause of the translation failure.

**Magnitude matters secondarily.** Single-axis degrades 79% → 49% going from 0.02 to 0.05,
so there is real saturation on top of the superposition effect.

**Proposition 2's cliff is refuted; its ordering survives.** §8 predicted translation would
be structurally quiet because its quantile range is 5× larger. Translation is not
structurally quiet — it recovers 79% of a single-axis fault. Rotation is still better (101%
vs 79%, and less than half the off-axis leak), so the *ordering* Prop 2 predicts is real, but
it is a graded effect, not the on/off distinction §7 and §8 built on.

### 11.3 What this costs, and what it does not

**It does not touch the headline result.** Rotation-only correction still gives 26% → 65%
pooled, `p = 9.3×10⁻¹⁰`, with zero regressions (§10). The recommendation to correct rotation
only is unchanged and still correct *for this fault*. What changes is why.

**It substantially weakens §7.4.** That section claims identifiability is predictable before
deployment from three healthy-data quantities, of which the quantile scale is the first. The
quantile scale is not the operative mechanism, so the prediction recipe is not established.
The honest replacement: identifiability depends on the *fault's shape relative to the
robot's reachable set* — whether the perturbed command drives the arm into contacts or
limits — and that is not readable from action statistics alone. It needs the fault, or at
least a fault class, plus the plant.

**§8.5 needs re-reading in this light.** Prop 2 predicted the gain fault would identify on
translation, and it did. That remains a correct prediction made in advance. But it now has a
competing explanation — the gain fault is multiplicative, so it perturbs each axis in
proportion to its own command rather than uniformly, which is a gentler excursion than the
uniform offset. The gain result no longer uniquely supports Prop 2.

### 11.4 Standing summary

| claim | status |
|---|---|
| Rotation-only correction recovers the fault across four suites | **holds**, p = 9.3×10⁻¹⁰, 0 regressions |
| The correction never breaks a working episode | **holds**, 0/80 paired |
| Translation is structurally unidentifiable | **refuted** — 79% at single-axis 0.02 |
| Quantile scale explains which channels identify | **not established** — shape dominates |
| Identifiability is predictable from healthy data alone | **withdrawn** — needs the fault class |
| `M` is unreliable / nonlinear on translation | **withdrawn** (§9.1) — M is magnitude-independent |
| `β̂_z = −0.797` | still unexplained |

## 12. The gain law: β̂_z was a clip, and §8.5 was missing its control (added 2026-09-01)

Two corrections, both to claims made earlier today in this same document.

### 12.1 `β̂_z = −0.797` is not an estimate, it is the projection bound

§8.7 flagged this as a reproducible systematic 60% overestimate and called it "the concrete
defect to chase, diagnosable offline". It was diagnosable offline, and the answer is that
**`--clip` defaults to 0.8 and `β` is clipped to `±clip`**. In 17 of 20 episodes `β̂_z` is
exactly −0.800, sd 0.007. It reproduces to three decimals because it is a *constant*, not
because it is a stable estimate.

An estimator whose true target is −0.5 should never reach a bound at 0.8. Reaching it means
**`β_z` is diverging and being held by the projection**, which is a different and worse fault
than bias.

### 12.2 The estimator diverges on a healthy robot

The no-fault control (`gain = 1.0`, true `β = 0`) had been run and never analysed this way:

| | x | y | z | rx | ry | rz |
|---|---|---|---|---|---|---|
| mean `β̂`, **healthy robot** | **−0.327** | 0.043 | **−0.337** | 0.082 | 0.156 | −0.138 |
| clip-rail rate | 10% | 0% | **40%** | 0% | 0% | 0% |

On a robot with no fault at all, the estimator reports a 33% loss of effectiveness on x and
z, and rails `β_z` at the bound in 40% of episodes. Fed to the inverse-gain compensator this
commands **1.49× on x and 1.51× on z on a healthy robot**. The measured cost is 10/10 → 9/10
on the no-fault arm — small at n = 10, but in the wrong direction, and the mechanism is
plainly unsafe.

This is the sharpest available contrast with the offset law, which broke **zero** of 80
paired episodes (§10). The two laws are not comparable in maturity, and the report should
stop presenting them as parallel results.

### 12.3 §8.5's confirmation of Proposition 2 lacked its control

§6 of this document establishes that a raw `f̂` cannot distinguish identification from plant
bias, and that only the separation test — faulted estimate minus **matched no-fault
estimate** — is evidence. That standard was applied to the offset law and **not** to the gain
law. §8.5 compared raw `β̂` against truth directly.

With the control applied (true `β = −0.5`):

| | x | y | z | rx | ry | rz |
|---|---|---|---|---|---|---|
| raw `β̂` (what §8.5 reported) | −0.487 | −0.481 | −0.797 | 0.000 | −0.094 | −0.120 |
| healthy `β̂` | −0.327 | 0.043 | −0.337 | 0.082 | 0.156 | −0.138 |
| **separation** | −0.160 | **−0.525** | **−0.460** | −0.082 | **−0.250** | +0.018 |
| % of truth | 32% | 105% | 92% | 16% | **50%** | −4% |
| `|sep|/se` | 1.4 | **13.5** | **3.8** | 1.9 | **5.1** | 0.3 |

The clean mirror image §8.5 claimed — all translation identifies, rotation is essentially
zero — **does not survive its own control**:

- **x does not identify** (1.4σ), though its raw `β̂` of −0.487 looked like a near-perfect
  recovery of −0.5. It was mostly the healthy-robot phantom of −0.327.
- **`ry` does identify** (50% of truth, 5.1σ), though its raw `β̂` of −0.094 looked like
  nothing. Against a healthy baseline of +0.156 it is a −0.250 shift.

So the measured pattern is two of three translation channels and one of three rotation
channels, not 3/0. Proposition 2 predicted a clean complementary split and did not get one.

### 12.4 Standing on Proposition 2

Combined with §11, which refuted the cliff for the offset fault, Prop 2 is now unsupported in
both directions it was tested. Its *ordering* remains weakly consistent with the data
(translation carries more of the gain fault than rotation: y and z at 105% and 92% against
`ry` at 50%), but the sharp complementarity claim should be dropped rather than defended.

**What survives untouched:** the offset result of §10 — rotation-only correction, four
suites, 26% → 65%, `p = 9.3×10⁻¹⁰`, zero regressions. That result never depended on Prop 2
being the right explanation, only on the empirical separation test that selected the
channels.

## 13. Decomposing the fault: rotation carries the damage (added 2026-09-01)

§11 and §12 removed both explanations offered for the scope condition. This section tests a
simpler one by splitting the uniform fault into its halves at the same magnitude, on
`libero_spatial`, n = 20.

### 13.1 Damage

| fault | frozen success | damage vs healthy |
|---|---|---|
| none | ~100% (10/10) | — |
| translation half `[.05,.05,.05,0,0,0]` | 18/20 = 90% | **−10** |
| rotation half `[0,0,0,.05,.05,.05]` | 13/20 = 65% | **−35** |
| uniform six-axis `[.05]×6` | 8/20 = 40% | **−60** |

**Rotation does 3.5× the damage of translation at equal magnitude.** This is the simplest
explanation yet for why rotation-only correction captures the whole benefit: rotation is
where the damage is. It requires no claim about quantile scales (§8, refuted in §11) and no
claim about which channels identify.

**The halves are also super-additive.** Independent damage would predict
`100 − 10 − 35 = 55%`; the measured combination is 40%. The missing 15 points are the
interaction, and they are consistent with §11: six axes faulted together drive the arm into
contacts and limits that neither half reaches alone. So §11's superposition failure shows up
in task success, not only in the sensitivity measurement.

### 13.2 Repair

| condition | frozen | corrected | McNemar |
|---|---|---|---|
| translation fault, translation correction | 18/20 | 18/20 | 1.0 |
| rotation fault, rotation correction | 13/20 | **18/20** | 0.0625 |

Correcting rotation on a rotation fault recovers 5 of the 7 lost episodes, with zero
regressions, at the exact test's floor for 5 discordant pairs.

Correcting translation on a **purely translational** fault does nothing — 1 episode fixed, 1
broken. This is the third independent time translation correction has contributed exactly
zero (§7.3 on `libero_object`, the uniform-fault ablation, and now a fault that is *only*
translation).

**The honest caveat:** with frozen already at 18/20, this test has almost no power. Only 2
episodes were available to fix, so "does nothing" here is weak evidence taken alone. It is
the *consistency* across three different setups that carries the claim, not this run.

> **Superseded, 2026-09-01 (§14).** The mechanism below rests on translation correction
> contributing nothing. At a fault magnitude the policy actually notices, it contributes a
> great deal: 20% → 95% at 0.15, p = 6.1×10⁻⁵. §13.2's own caveat about the ceiling was
> correct and this subsection ignored it. Read §14.

### 13.3 What this replaces

The scope condition survives, with its third and simplest mechanism:

> Correct rotation because rotation is where this policy's damage comes from — 3.5× the
> translation half at equal magnitude — and because translation correction has never, in
> three separate tests, contributed anything.

This makes no appeal to quantile scales or channel identifiability. It is a statement about
**this policy's sensitivity**, measurable directly by faulting each half and reading the
frozen success rate, and it should be checked per policy rather than assumed.

### 13.4 Open

Whether a *large* translation fault is both damaging and repairable is untested: at 0.05 the
policy barely notices, and the single-axis sweep (§11) found no damage at all. A translation
fault big enough to hurt would settle whether translation correction is useless or merely
untested against a ceiling.

## 14. Translation correction works. §13.3 was a ceiling artifact (added 2026-09-01)

§13.4 flagged that a translation fault large enough to hurt had never been tried. It has now,
and the result overturns the central framing of §7.3, §13.2 and §13.3.

### 14.1 The measurement

Translation-only fault on x, y, z; translation-only correction; `libero_spatial`, n = 20
paired, `--clip 0.30` so the estimator is not capped below its own target (the §12.1 lesson).

| fault magnitude | frozen | corrected | fixed | broken | exact McNemar |
|---|---|---|---|---|---|
| 0.05 | 18/20 = 90% | 18/20 = 90% | 1 | 1 | 1.0 |
| **0.10** | 13/20 = 65% | **19/20 = 95%** | **6** | **0** | **0.031** |
| **0.15** | 4/20 = 20% | **19/20 = 95%** | **15** | **0** | **6.1×10⁻⁵** |

At 0.15 the fault destroys the policy — 20% — and translation correction recovers **15 of the
16 lost episodes**, breaking none.

### 14.2 The claim that is now withdrawn

Three times this document asserted that translation correction "contributes exactly zero",
and §13.3 built a mechanism on it: *correct rotation because that is where the damage is.*

**That was an artifact of fault magnitude, not a property of the channel.** Every test of
translation correction had been run at 0.05, where a translation fault leaves the policy at
90% and there are at most two episodes available to fix. §13.2 stated this caveat and then
§13.3 drew the conclusion anyway. The caveat was right and the conclusion was wrong.

The method is **more** general than the last three sections claimed, not less. It repairs
translation faults, rotation faults, and mixed-sign structured faults, each when that fault
is what is present and large enough to matter.

### 14.3 The corrected account

Nothing in the data requires a channel to be privileged. What the results say:

1. **A single-family fault is identified and repaired**, translation or rotation alike —
   translation at 0.10 (+30, p = 0.031) and 0.15 (+75, p = 6.1×10⁻⁵), rotation at 0.05 (+25),
   structured mixed-sign rotation at 0.06 (+66).
2. **The uniform six-axis fault is the pathological case.** It drives the arm into contacts
   where superposition fails (§11, §13.1 super-additivity), the translation estimates go
   sign-wrong, and correcting translation then hurts. Rotation-only is the right restriction
   **for that fault**, and §10's four-suite result stands exactly as measured.
3. So the scope condition is a statement about **the fault**, not about the channel: restrict
   the correction when the fault drives the plant out of the regime where the residual is
   informative. For faults inside that regime, correct the channels the fault is on.

### 14.4 The safety property, across everything run

Aggregating every paired run to date — 200 episodes, ten conditions, both fault families and
all four suites:

| | |
|---|---|
| episodes fixed | **61** |
| episodes broken | **3** |
| regression rate | **1.5%** |

All three regressions occur in runs where the frozen policy was already at 18–19 of 20, i.e.
where there was nothing to gain and only noise to lose. In the four-suite headline (80
episodes) and in both translation-magnitude runs (40 episodes) the correction broke
**nothing**.

### 14.5 Method note

This is the second time today a "does nothing" conclusion came from a condition with no
headroom, and the second time the fix was to raise the fault until the frozen policy actually
fails. **A null result against a ceiling is not a null result.** Any future "channel X
contributes nothing" claim needs the frozen arm below roughly 70% before it means anything.
