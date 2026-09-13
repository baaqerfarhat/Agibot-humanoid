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

## 15. `--law innov` does not measurably help (added 2026-09-01)

§8/§12 shipped `--law innov` as the fix for the biased fixed point the external review
identified. It had never been tested. It has now, and the honest result is a null on my own
fix.

### 15.1 Design

The bias is `1/(1 + ‖r‖²/ρ²)`, so it only becomes visible when the residual is large. At the
0.05 fault it is ~5% — unresolvable at n = 10, which is why testing there would have wasted
the GPU. A 0.15 translation fault makes the residual roughly 3× larger. Each law was run with
its **own** matched no-fault control, because the plant-bias phantom differs between them,
and with `--estimate-only` so the comparison is of estimators rather than closed loops.

### 15.2 Result

Separation against each law's own control, true fault +0.15 on x, y, z:

| law | mean translation separation | % of truth | per-channel se |
|---|---|---|---|
| legacy | 0.1387 ± 0.0137 | 92% | 0.020, 0.016, 0.023 |
| innov | 0.1429 ± 0.0334 | 95% | 0.037, 0.041, 0.051 |

**Difference: +0.0042 ± 0.0361, i.e. 0.12σ.** The two laws are indistinguishable.

Two things worth keeping:

1. **The predicted bias did not appear.** I forecast ~31% attenuation from an assumed
   `‖r‖ ≈ 0.10`. The actual `‖Mf‖` at this fault is **0.0617**, which predicts 86%, and the
   measured legacy value is 92%. My own prediction was wrong because I guessed the residual
   norm instead of computing it from `M f` — which took one line.
2. **`innov` is ~2.2× noisier per channel.** The innovation form carries an extra `M f̂` term
   and its variance with it. It buys an unbiased fixed point at a real cost in variance, and
   at these residual magnitudes there is no bias worth buying.

### 15.3 Standing

`legacy` remains the default. `innov` stays available and documented, and would matter if the
residual ever approached `ρ` — but on this plant, at faults up to 0.15, it does not.

**On the review's criticism:** the algebra was correct and I confirmed it. The practical
magnitude is ~8%, not the 31% I estimated, and correcting it changes no result. That is worth
stating plainly rather than shipping a fix and implying it mattered.

## 16. Loss of effectiveness: the intercept fix failed, and the law remains unsafe (added 2026-09-01)

> **Superseded, 2026-09-02 (§21).** The diagnosis here was incomplete. The problem was not
> a missing parameter but a wrong regressor: the law regressed on the instantaneous command
> when the plant model says the residual is proportional to the FIR-weighted command
> history. Correcting that shrinks the healthy-robot phantom 9× and takes x from 1.4σ to
> 18.1σ. §16.4 (rotation is unexcitable) survives unchanged and was re-confirmed.

§12 showed the gain law reports a 33% loss of effectiveness on a healthy robot. This section
attempts a fix, fails, and records both the failure and one solid finding that came out of it.

### 16.1 The attempted fix

The model was `z_i ≈ β_i·ψ_i` with no intercept, while the residual carries a constant
plant-model bias `b` (the phantom the offset law subtracts via `--bias`). With no home for
`b`, β absorbs `b/ψ`. Synthetically this reproduces exactly: with a **one-sided** command the
no-intercept law converges to a phantom (+0.10 where truth is 0) and the intercept form
returns 0.000. With symmetric commands both are fine, so the one-sidedness is essential —
and a reach is one-sided.

### 16.2 It made things worse on the real robot

| | healthy phantom (mean \|β̂\|) | healthy task success | sep on x | sep on z |
|---|---|---|---|---|
| no intercept | 0.181 | 9/10 | −0.160 | −0.460 |
| **with intercept** | **0.224** | **7/10** | **−0.045** | **−0.299** |

Worse on every measure. `--intercept` stays **off by default** and is retained only as a
documented negative result.

### 16.3 The obvious explanation is ruled out

The natural diagnosis is collinearity: if `ψ` barely varies within an episode, `[1, ψ]` are
nearly the same direction and `b` cannot be separated from `β·ψ`. Measured on the recorded
commands, restricted to steps that pass the PE gate:

| ch | mean\|ψ\| | sd(ψ) | sd/\|mean\| | cond([1,ψ]) |
|---|---|---|---|---|
| x | 0.515 | 0.468 | 0.91 | **2.4** |
| y | 0.423 | 0.405 | 0.96 | **2.6** |
| z | 0.644 | 0.660 | 1.02 | **1.6** |
| ry | 0.169 | 0.011 | 0.06 | 93.6 |
| rx, rz | — | — | — | **no episodes pass the gate** |

Translation is well conditioned (1.6–2.6). **The collinearity explanation is false for the
channels that got worse**, and I do not have a confirmed mechanism for why the intercept
hurt. Recorded as an open failure rather than given a story.

### 16.4 The finding that does hold: rotation gain faults are unidentifiable here

`rx` and `rz` have **no episodes at all** in which the command exceeds the PE gate, and `ry`
has sd 0.011 about a mean of 0.169. The policy simply does not rotate the wrist enough to
excite a multiplicative fault. This is direct measurement of the persistency-of-excitation
argument, and it is a property of **the policy's behaviour**, not of the estimator: no gain
law, however constructed, can identify a rotation gain fault from data with no rotational
excitation. It would need deliberate dither, which perturbs the task.

### 16.5 Status: loss of effectiveness is not solved

The faulted run looks good in isolation — frozen 14/20 → corrected **20/20** — and it must
not be reported that way. The matched healthy control is **10/10 → 7/10**: the same law
damages a robot with nothing wrong with it. A repair number is not meaningful while the
no-fault control shows harm, so the honest summary is:

> **Additive faults: solved.** 61 fixed / 3 broken over 200 paired episodes, four suites,
> both fault families, magnitudes 0.05–0.15.
>
> **Multiplicative (loss of effectiveness): not solved.** Identification works on 2 of 3
> translation channels, is structurally impossible on rotation for this policy (§16.4), and
> the estimator still hallucinates faults on healthy hardware. It should be presented as an
> open problem with a diagnosed cause, not as a second result.

## 17. Time-varying faults: repair without accurate tracking (added 2026-09-02)

Three profiles at amplitude 0.10 on translation, n = 20 paired, `libero_spatial`.

### 17.1 Task outcome

| profile | frozen | corrected | fixed | broken | McNemar |
|---|---|---|---|---|---|
| **ramp** (0 → full over 60 steps) | 12/20 = 60% | **18/20 = 90%** | 6 | 0 | **0.031** |
| sine (period 80) | 20/20 = 100% | 19/20 = 95% | 0 | 1 | 1.0 |
| intermittent (50 on, 50 off) | 17/20 = 85% | 20/20 = 100% | 3 | 0 | 0.25 |

**Only the ramp is a real test.** A zero-mean sine averages out over an episode and does not
damage the policy at all (frozen 20/20), and the intermittent fault leaves only 3 episodes to
recover. Both are ceilings — the mistake §14.5 warned about, made again. A proper oscillatory
test needs a **non-zero-mean** sine (bias plus oscillation), and the intermittent case needs a
larger amplitude.

On the one condition with headroom, the estimator handles a gradually degrading fault:
+30 points, `p = 0.031`, zero regressions.

### 17.2 Tracking is poor, and it does not matter much

The estimate was compared against the analytic ground truth per step:

| profile | RMS tracking error | as % of amplitude | after best lag correction |
|---|---|---|---|
| ramp | 0.0446 | 45% | 44% (lag 5) |
| sine | 0.0789 | 79% | 69% (lag 5) |
| intermittent | 0.0718 | 72% | 64% (lag 10) |

Shifting the estimate in time barely helps, so **the error is not lag** — with `γ = 0.08` the
EMA time constant is ~12.5 steps and a pure lag would have been removed by the shift. The
estimator is genuinely failing to follow the waveform, and is capturing something closer to
its running average.

**And yet the ramp condition still recovers 6 of 8 lost episodes.** This matches §15, where a
92%-accurate estimate and a 95%-accurate one were indistinguishable in outcome: **the task
tolerates a lot of estimator error.** The correction has to be roughly right in direction and
scale; it does not have to be a good tracker.

That is a useful property to state plainly — it is why a six-parameter, CPU-side update is
enough — but it also means **tracking accuracy is the wrong headline metric** for this method.
Task outcome under a matched control is the metric that has survived every test today.

### 17.3 What is now covered

| fault type | status |
|---|---|
| constant additive, uniform 6-axis | solved, 4 suites, p = 9.3×10⁻¹⁰ |
| constant additive, single-family, 0.05–0.15 | solved, up to 20% → 95% |
| structured mixed-sign | solved, +66 |
| mid-episode onset (step) | solved (§3) |
| **ramp / gradual degradation** | **solved, +30, p = 0.031** |
| oscillatory | **untested** — the sine chosen did no damage |
| intermittent | inconclusive — underpowered |
| multiplicative (loss of effectiveness) | **not solved** (§16) |
| sensor bias, camera shift | structurally invisible (§4) |

## 18. The integral baseline: the plant model is load-bearing (added 2026-09-02)

The paper claims that identifying a plant model on healthy data and measuring a sensitivity
matrix `M` is what makes single-episode repair possible. Nothing in §1–§17 tested that claim.
A control engineer's first move needs neither: integrate the raw motion error.

```
e_t = y_t − a_t        (achieved minus commanded, in action units)
f̂ ← clip(f̂ + kᵢ·e_t)   no FIR plant, no M
```

Same fault (translation 0.15), same episodes, same corrected channels, `--clip 0.30`.

### 18.1 The gain sweep, in full

| `kᵢ` | frozen | integral baseline | Δ | McNemar |
|---|---|---|---|---|
| 0.001 | 4/20 | 5/20 | +5 | 1.0 |
| **0.005** | 3/20 | **7/20** | **+20** | 0.125 |
| 0.02 | 5/20 | **0/20** | −25 | — |
| 0.05 | 2/20 | 1/20 | −5 | — |
| 0.15 | 4/20 | 1/20 | −15 | — |
| **ours** | **4/20** | **19/20** | **+75** | **6.1×10⁻⁵** |

The baseline is **not** useless. At `kᵢ = 0.005` it recovers 4 of the 17 lost episodes
(+20 points, though `p = 0.125` — not significant at n = 20). Above that it is actively
harmful, taking the policy to zero at `kᵢ = 0.02`.

**Correction to an earlier draft of this section.** Having seen only `kᵢ ≥ 0.02` fail, I wrote
that the baseline's best attainable result is a *tie* with frozen, on the argument that
`kᵢ → 0` reduces it to the frozen policy. The low-gain runs refute that: at 0.005 it is
clearly better than frozen. The argument was wrong because the plant-gain phantom is
proportional to the **command**, which changes sign through an episode, while the fault is
constant. A slow integrator low-passes the phantom toward zero and accumulates the fault. It
is a bad estimator, not a structurally incapable one.

### 18.2 Why it is bad: the raw error is dominated by the plant gain

The baseline implicitly assumes achieved motion ≈ commanded action. The measured translation
DC gain is **0.23**, so with **no fault at all**

```
e = y − a = (0.23 − 1)·a = −0.77·a
```

At a command magnitude of 0.5 that phantom is **0.384** against a real fault of **0.150** —
more than twice as large, and opposite in sign. The estimator's job is then to average away a
disturbance 2.5× the size of its target, which is why it needs a gain small enough to be
nearly inert, and why it saturates and oscillates as soon as the gain is large enough to move.

The trajectories, on a true fault of **+0.15**:

| | first 40 steps of `f̂ₓ` | final | sign flips | episode |
|---|---|---|---|---|
| `kᵢ`=0.02 | +0.001, **−0.104, −0.277, −0.283** | +0.300 (railed) | 2 | 220 (timeout) |
| `kᵢ`=0.15 | +0.004, **−0.300, −0.300**, −0.175 | −0.300 (railed) | 5 | 220 (timeout) |
| **ours** | +0.006, **+0.080, +0.117, +0.157** | +0.178 | **0** | **74 (success)** |

### 18.3 What the plant model buys

Against the **best** baseline gain, tuned in its favour: **35% versus 95%**. Predicting
`y ≈ 0.23a` instead of `y ≈ a` is the entire difference, and it converts a correction that
must creep to avoid instability into one that converges to the truth in ~15 steps with no
sign changes.

**This is the ablation the method needed**, and it is a real comparison rather than a
strawman: the baseline was swept over five gains across two orders of magnitude and is
reported at its best.

## 19. The recoverability map (added 2026-09-02)

Fault family × severity, `libero_spatial`, n = 20 paired per cell, correction applied to the
channels each family's evidence supports (rotation faults and uniform faults both on
rotation, per §14.3).

| fault | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| **translation** | 18/20 → 18/20 (ceiling) | 13/20 → **19/20**, p=0.031 | 4/20 → **19/20**, p=6.1×10⁻⁵ |
| **rotation** | 13/20 → 18/20, p=0.063 | 0/20 → **17/20**, p=1.5×10⁻⁵ | 0/20 → **9/20**, p=0.0039 |
| **uniform 6-axis** | 8/20 → 18/20, p=0.0020 | 2/20 → 11/20, p=0.0039 | 0/20 → 2/20, p=0.50 |

**Zero regressions in every cell.**

### 19.1 Rotation is where the policy breaks

At 0.10 and 0.15 a rotation-only fault takes the frozen policy to **exactly zero**, while a
translation fault of the same magnitude leaves it at 13/20 and 4/20. This is §13.1's 3.5×
damage ratio holding across severities, and it is the clearest single statement of the
policy's sensitivity: **π0.5-LIBERO tolerates translation error and does not tolerate
rotation error.**

Recovery from that floor is the strongest result in the table — `0/20 → 17/20` at 0.10, from
a policy that never once succeeds unaided.

### 19.2 Single-family faults degrade gracefully; the uniform fault falls off a cliff

Reading down the severity axis:

- translation: −, 95%, 95% — no degradation at all up to 0.15
- rotation: 90%, 85%, 45% — graceful
- **uniform: 90%, 55%, 10%** — collapse

### 19.3 The collapse is the translation component, and it is not a correction failure

The sharpest comparison in the table is `rotation 0.15` against `uniform 0.15`. **Both are
corrected identically — rotation only.** The uniform fault is the rotation fault plus a
translation component that the correction never touches:

| | frozen | corrected |
|---|---|---|
| rotation 0.15, corrected on rotation | 0/20 | **9/20 = 45%** |
| uniform 0.15, corrected on rotation | 0/20 | **2/20 = 10%** |

Adding an *uncorrected* translation component drops recovery from 45% to 10%. Since the
correction is the same in both rows, the loss cannot be an estimation or correction failure
on rotation. It is the translation component **destroying recoverability by driving the arm
into contacts and limits** — §11's superposition failure and §13.1's super-additive damage,
now visible directly in task outcome.

**This is the method's boundary, and it is a property of the plant, not the estimator.** Once
a fault is large enough and spread across enough axes to put the arm outside the regime where
its own motion is informative, no amount of correction on the identifiable channels recovers
the task.

### 19.4 Safety, over everything

Across **340 paired episodes** — four suites, three fault families, three severities, four
time profiles: **107 fixed, 4 broken, a 1.2% regression rate.** Every regression occurs where
the frozen policy was already at 18–19 of 20.

## 20. Time-varying faults, completed (added 2026-09-02)

§17 left two of four profiles unproven, and said why: the sine and intermittent cells were
ceilings, with the frozen policy at 20/20 and 17/20. Both are now re-run against a fault the
policy is actually sensitive to.

### 20.1 What changed, and why it was the fault rather than the analysis

Two corrections to the original design, both applied **before** running rather than after
finding a null:

1. **Moved to rotation.** §19 shows rotation is where this policy breaks — a 0.10 rotation
   fault takes it to 0/20 while translation at the same magnitude leaves it at 13/20.
2. **Made the oscillation non-zero-mean.** The original `sine` averaged to zero over an
   episode, which is precisely why it did no damage. `sine_bias` swings between zero and full
   fault instead, which is what a thermal or load cycle produces.

### 20.2 All four profiles

| profile | frozen | corrected | fixed | broken | McNemar |
|---|---|---|---|---|---|
| step / mid-episode onset | — | — | — | — | solved (§3) |
| ramp, translation 0.10 | 12/20 | **18/20** | 6 | 0 | **0.031** |
| **sine_bias, rotation 0.10** | 13/20 | **20/20** | 7 | 0 | **0.016** |
| **intermittent, rotation 0.10** | 12/20 | **19/20** | 8 | 1 | **0.039** |

**The time-varying row is now complete and every profile is significant.**

`sine_bias` reaching **20/20** is the strongest cell in the whole record: perfect recovery on
a fault whose magnitude is changing continuously throughout the episode. Set against §17.2,
where tracking error was 45–79% of amplitude and not attributable to lag, it sharpens the
same conclusion — **the correction does not need to track the waveform.** It needs to be
roughly right in direction and scale, and the task absorbs the rest.

### 20.3 Aggregate

Across **380 paired episodes** — four suites, three fault families, three severities, four
time profiles, both correction restrictions: **122 fixed, 5 broken, a 1.3% regression rate.**

The intermittent cell contributes the fifth regression, and it is the first one that did not
occur against a ceiling: frozen was at 12/20 there, not 18–19/20. Worth noting rather than
smoothing over — a fault that switches off entirely is the one condition where a stale
correction has something to damage.

### 20.4 Coverage, updated

| fault type | status |
|---|---|
| constant additive: uniform, single-family, structured | solved |
| mid-episode onset | solved |
| ramp / gradual degradation | solved, p = 0.031 |
| **oscillatory (non-zero-mean)** | **solved, p = 0.016** |
| **intermittent** | **solved, p = 0.039** |
| multiplicative (loss of effectiveness) | not solved (§16) |
| sensor bias, camera misalignment | structurally invisible (§4) |

## 21. Loss of effectiveness, solved: the regressor was wrong (added 2026-09-02)

§16 left multiplicative faults unsolved with a diagnosed cause, and recorded a failed fix.
The diagnosis was incomplete: the problem was not a missing parameter but a **wrong
regressor**, and the intercept patch added a parameter to a model that was mis-specified.

### 21.1 The derivation

The world executes `g·u`, and the FIR plant responds to what it is given:

```
y_i = Σ_ℓ W[i,ℓ]·(g·u_{k−ℓ,i}) + W[i,−1] = g·(pred_i − W[i,−1]) + W[i,−1]
```

so

```
y_i − pred_i = β_i · φ_i        with   φ_i = pred_i − W[i,−1]
```

The regressor is the **FIR-weighted command history**, not the instantaneous command `ψ_i`
the law had been using, and the regression lives in motion units with **no `M⁻¹` at all**.
Two mismatches, both removed by deriving the regressor instead of assuming it.

### 21.2 Sizing the gate before running

`φ` is the command scaled by the plant gain (~0.25), so it is ~4× smaller than `ψ`. The
inherited `pe_min = 0.15` would have gated out 77% of usable steps on x and 91% on y.
Measured on healthy rollouts:

| `pe_min` | x | y | z | rx | ry | rz |
|---|---|---|---|---|---|---|
| 0.15 (inherited) | 23% | 9% | 46% | 0% | 0% | 0% |
| **0.04** | **74%** | **46%** | **78%** | 0% | 0% | 0% |

0.04 restores the coverage `ψ` had. Rotation passes **0% at every threshold from 0.15 down to
0.02** — §16.4 confirmed independently, and not a tuning problem.

### 21.3 Safety: the phantom is gone

The failure that made §16 declare this unsolved was a 33% hallucinated gain loss on a healthy
robot. With the corrected regressor, on `gain = 1.0`, true `β = 0`:

| | mean \|β̂\| | task success |
|---|---|---|
| `cmd` (old) | 0.181 | 10/10 → **9/10** |
| **`fir` (new)** | **0.020** | **10/10 → 10/10** |

**A 9× smaller phantom, and no episodes lost on a healthy robot.**

### 21.4 Identification: all three translation channels, against the control

Separation (faulted minus matched healthy), true `β = −0.500`:

| | x | y | z | rz |
|---|---|---|---|---|
| `cmd` (old) | −0.160 (32%, 1.4σ) | −0.525 (105%) | −0.460 (92%) | +0.018 (0.3σ) |
| **`fir` (new)** | **−0.515 (103%, 18.1σ)** | **−0.500 (100%, 60.6σ)** | **−0.457 (91%, 9.1σ)** | −0.122 (2.5σ) |

x went from **not identifying at all** (1.4σ, 32% of truth) to **103% at 18σ**. All three
translation channels now recover the fault to within 9%.

### 21.5 Task outcome, and what is still owed

Frozen 15/20 → corrected **18/20**, 3 fixed, 0 broken, `p = 0.25`.

**Not significant**, and it should not be presented as though it were: with frozen at 15/20
there were only 5 episodes available to fix. This is the ceiling problem of §14.5 again, and
the honest next step is the one that worked for translation — raise the severity until the
frozen policy actually fails, then measure.

**What is established** is the part that was actually broken: the estimator is now safe on
healthy hardware and identifies the fault on every excited channel. Whether that converts to
task repair at a severity with headroom is untested.

| | before (§16) | now |
|---|---|---|
| phantom on healthy robot | 0.181, costs an episode | **0.020, costs nothing** |
| channels identified | 1 of 3 translation | **3 of 3** |
| x separation | 1.4σ | **18.1σ** |
| task repair | uninterpretable | +15, `p` = 0.25, needs headroom |

## 22. Loss of effectiveness repairs at severity (added 2026-09-03)

§21 fixed identification and safety for the multiplicative fault but could not test repair:
at `gain = 0.5` the frozen policy still scored 15/20, leaving five episodes to fix. Lowering
the gain until the policy actually fails settles it.

### 22.1 Result

| gain (β) | frozen | corrected | fixed | broken | exact McNemar |
|---|---|---|---|---|---|
| 0.50 (−0.50) | 15/20 = 75% | 18/20 = 90% | 3 | 0 | 0.25 (ceiling) |
| **0.30 (−0.70)** | 1/20 = 5% | **17/20 = 85%** | 16 | 0 | **3.1×10⁻⁵** |
| **0.20 (−0.80)** | 0/20 = 0% | **17/20 = 85%** | 17 | 0 | **1.5×10⁻⁵** |

At a 80% loss of actuator effectiveness the frozen policy **never once succeeds**, and the
correction recovers 17 of 20 episodes while breaking none.

**Loss of effectiveness is solved.** It was the last fault family listed as unsolved, and the
fix was §21's regressor correction — nothing here changed but the severity.

### 22.2 One estimate is confounded, and it is not the one that matters

`β̂` at `gain = 0.20` reads −0.800, −0.796, −0.797 against a true −0.800, which looks like a
0.5% estimate. It is not, on two of three channels: **`--clip` is 0.8, so the bound and the
truth coincide.** Checking per-episode saturation:

| gain | x at clip | y at clip | z at clip |
|---|---|---|---|
| 0.30 | 10% | 0% | 0% |
| 0.20 | **60%** | 0% | **35%** |

At `gain = 0.20`, x and z are pinned at the projection bound in 60% and 35% of episodes, so
their agreement with truth **cannot be distinguished from saturation**. Only y (0% railed,
−0.796 against −0.800) is a genuine measurement there.

`gain = 0.30` is the clean cell: −0.723, −0.699, −0.717 against −0.700, **within 3%**, with
only x touching the bound and only in 10% of episodes.

This is the §12.1 trap in a new place — an estimate reading exactly its own clip — and it is
recorded because the accuracy claim at 0.20 would otherwise be wrong. **The task results are
unaffected:** success does not depend on how the estimate is read out, and a saturated
estimate that happens to equal the truth still produces the right correction.

### 22.3 Coverage, complete

| fault type | status |
|---|---|
| constant additive: uniform, single-family, structured | solved |
| mid-episode onset, ramp, oscillatory, intermittent | solved |
| **multiplicative (loss of effectiveness)** | **solved — 0/20 → 17/20 at 80% loss, p = 1.5×10⁻⁵** |
| sensor bias, camera misalignment | structurally invisible (§4) |

Rotation gain faults remain unidentifiable for this policy (§16.4): the commands never excite
those channels, at any gate threshold. That is a property of π0.5's behaviour, not of the
estimator, and no gain law can recover it without deliberate dither.

### 22.4 Aggregate

**450 paired episodes** — four suites, three additive fault families, three severities each,
four time profiles, and a multiplicative severity sweep: **158 fixed, 5 broken, 1.1%
regression.**

## 23. Second suite: all three fault families replicate (added 2026-09-03)

The map, the baseline and the gain sweep all ran on `libero_spatial`. These are the three
headline cells — one per fault family, each at the severity where spatial showed the largest
effect — repeated on `libero_object`, n = 20 paired.

**The plant model `W` and the sensitivity matrix `M` were not re-identified.** Both are the
ones fitted on `libero_spatial` healthy rollouts, so this tests calibration transfer as well
as replication.

### 23.1 Result

| cell | `libero_spatial` | `libero_object` | McNemar (object) |
|---|---|---|---|
| rotation 0.10 | 0/20 → 17/20 (85%) | **0/20 → 10/20 (50%)** | **0.0020** |
| translation 0.15 | 4/20 → 19/20 (95%) | **0/20 → 15/20 (75%)** | **6.1×10⁻⁵** |
| gain 0.20 | 0/20 → 17/20 (85%) | **0/20 → 19/20 (95%)** | **3.8×10⁻⁶** |

**Zero regressions in all three.** The frozen policy scores **0/20 in every cell** — at these
severities `libero_object` is destroyed by all three fault families.

All three replicate, and **the offline calibration transfers**: a plant model and sensitivity
matrix identified on one task suite repair faults on another without re-identification. That
matters for deployment, because it means the healthy-data calibration is a property of the
robot rather than of the task distribution.

### 23.2 What does not transfer cleanly, stated honestly

Recovery on the two additive cells is lower than on spatial — 50% against 85%, 75% against
95%. Two explanations push the same way and **this design cannot separate them**:

1. `libero_object` is the harder suite; its unfaulted baseline is lower.
2. The calibration is foreign to it.

The gain cell argues against (2) being dominant — it *exceeded* spatial (95% vs 85%) on the
same foreign calibration — but that is one cell, not a control. Separating the two needs `W`
and `M` re-identified on `object` healthy rollouts, which is cheap: open-loop replay is CPU
and takes seconds. **Until that is run, the transfer claim should be stated as "works without
re-identification", not as "loses nothing".**

### 23.3 The clip confound, again

`gain = 0.20` on object gives `β̂ = −0.798, −0.797, −0.791` against a true −0.800, and
`--clip` is again 0.8. Per-episode: **x is pinned at the bound in 45% of episodes**; y and z
are clean at 0%. So y and z are genuine measurements here (0.4% and 1.1% error) and x is not.

This is the fourth time an estimate has read approximately its own projection bound. The
lesson has been learned repeatedly and not yet acted on: **the clip should be set from the
expected fault magnitude at run time, not left at a default that can coincide with truth.**

### 23.4 Aggregate

**510 paired episodes** across two suites, four fault families, three severities, four time
profiles: **202 fixed, 5 broken, 1.0% regression.**

## 24. `libero_90`: a floor, not a repair result (added 2026-09-03)

The four-suite headline was extended to `libero_90` — 90 tasks, sampled every 4th, same
configuration (uniform +0.05, rotation-only), same spatial-identified `W` and `M`. The
faulted run came back **4/20 → 5/20, p = 1**, with 15 episodes where neither arm succeeded.

That number cannot be read without knowing what the policy can do *unfaulted*. The same 20
episodes, three conditions:

| condition | success |
|---|---|
| healthy, frozen | **6/20** |
| faulted, frozen | 4/20 |
| faulted, corrected | 5/20 |
| healthy, with the law running | 5/20 |

**The policy fails 14 of these 20 tasks with no fault present.** There is nothing for a
correction to restore on them. Of the 6 it can do, the fault breaks 4 and the correction
recovers 1 — too few episodes to say anything about repair, and within the ±10-point
run-to-run noise (tasks 28 and 44 succeeded *faulted* but not *healthy*).

### 24.1 Why: these tasks are outside the checkpoint's competence

openpi's LIBERO fine-tune ingests exactly four raw datasets — `libero_10`, `goal`, `object`,
`spatial` — and the converted set on the Hub has 40 tasks, all from those suites. **`libero_90`
is not in it.** π0.5-LIBERO has never seen these tasks, and 6/20 is what an
out-of-distribution suite looks like for it. The keyword overlap with `goal`/`10` tasks
("open the top drawer...", "turn on the stove...") is vocabulary, not training coverage.

The estimator is not implicated: rotation `f̂` converged to `[0.040, 0.020, 0.044]` here
against `[0.040, 0.030, 0.045]` on spatial, and the trajectory on the longest episode climbs
cleanly to 0.051. The correction was applied correctly to a policy that could not complete
the task either way.

### 24.2 What this is, and is not

It is **a fact about the policy**: this checkpoint does not generalise to `libero_90`. That is
worth one sentence in the paper and no more. It is **not** a fault-repair result, positive or
negative, and it must not be tabulated as one — the faulted and corrected arms are both
sitting on the floor.

Safety on the healthy robot: 5/20 against 6/20 frozen, one regression, inside the noise.

### 24.3 The design rule, completed

§14.5 said a null against a *ceiling* is not a null. This is the same rule from the other
side: **a null against a floor is not a null either.** Before any repair experiment, the
frozen policy's unfaulted success on the exact episodes must be measured and must sit well
away from both 0 and 20. Three of today's cells were ceilings; this one is a floor; all four
were avoidable with a healthy-frozen control run first. That control is now a prerequisite,
not an afterthought.

### 24.4 Aggregate, unchanged in substance

530 paired episodes, 203 fixed, 5 broken (0.9%). `libero_90` contributes one fix and no
regressions, and is listed for completeness rather than as evidence.

## 26. A second manipulator: ALOHA in joint space — design, validated before the policy (added 2026-09-03)

Bimanual ALOHA (two ViperX arms) in `gym_aloha`, driven by `pi0_aloha_sim`. The action
interface is **14 absolute joint targets in radians** at 50 Hz, the state is the same 14
measured joint positions, and the episode cap is 300 steps. Nothing here is Cartesian and
nothing rotates: joint space is a vector space, so the whole class of SO(3) bug from §2.2b
cannot occur.

The pipeline was exercised end to end with a **stub policy** before spending any server
time, and that dry run changed the design twice.

### 26.1 The residual must live at position level, not increment level

The LIBERO plant predicts the *increment* `dq` from the command, because an OSC command *is*
an increment. Ported verbatim, that plant gave `M ≈ 0.008` on every joint: a target offset is
absorbed by the position servo within a few steps and leaves almost nothing in `dq`, so
`M⁻¹ ≈ 125×` amplified pure noise and `f̂` oscillated to 0.26 with sign flips.

Regressing **position** on the target history instead — `q_t = Σ_k h_k u_{t−k} + c` — gives
R² 0.986–0.998 on the arm joints, taps summing to ≈ 1 (a servo tracks its target), and
`M ≈ I` with cond 1.1 by construction. The law is otherwise unchanged.

### 26.2 Whether the fault is observable depends on the policy, not the estimator

A +0.05 rad offset on joints 0–2, traced under two stubs:

| stub | residual on faulted joints | residual on clean joints | net drift, joint 0 |
|---|---|---|---|
| planned targets, no fault | 0.032, 0.015, −0.003 | ≈ 0 | −0.08 |
| **planned targets, +0.05** | **0.082, 0.065, 0.047** | ≈ 0 | −0.04 |
| anchored targets, no fault | ≈ 0 | ≈ 0 | 0.47 |
| **anchored targets, +0.05** | ≈ 0 | **−0.19 on joint 4** | **1.39 (runaway)** |

With targets planned from the task, the separation is **exactly +0.050 on every faulted
joint**. With targets anchored to the *measured* state — "go to where I am, plus δ" — the
same fault becomes an integrator: each re-anchor adds `f`, the arm runs away 1.4 rad, the
drift is *inside the command* so a command-based residual cannot see it, and a clean joint
is driven into a limit.

This is the mechanism, stated once: **on an absolute-position interface, a policy that
re-anchors to measured state converts a constant fault into a drift and hides it from
proprioceptive identification.** LIBERO never posed the question because its commands are
increments. π0 takes state as input, so how strongly it anchors is an empirical property of
the checkpoint — the frozen-faulted run will show either a bounded tracking error (fault
visible, repairable) or a runaway (fault invisible, and a different problem).

### 26.3 What is fixed before the real run

- the healthy-frozen control runs **first**, on the exact episodes (§24.3)
- the plant and `M` are identified on the real policy's healthy rollouts, not the stub's
- grippers (joints 6, 13) are never corrected: R² 0.75 and no excitation
- `--clip` is set from the fault magnitude, not left at a default that can coincide with it

## 25. Second backbone: OpenVLA-OFT, with the π0.5 calibration (added 2026-09-03)

OpenVLA-OFT (7 B, PyTorch, L1-regression action head, LIBERO-spatial fine-tune) served behind
the same websocket protocol via `oft_server.py`. **Every experiment script, the FIR plant
model and the sensitivity matrix `M` are the ones used for π0.5, unchanged** — `W` and `M`
were identified with π0.5 driving the arm and were never re-fitted. `libero_spatial`, n = 20
paired per cell, healthy control run first.

| cell | frozen | corrected | fixed | broken | exact McNemar | π0.5 (same cell) |
|---|---|---|---|---|---|---|
| healthy control | 20/20 | 19/20 | 0 | 1 | 1.0 | 10/10 → 10/10 |
| rotation 0.10 | 0/20 | **11/20 = 55%** | 11 | 0 | **0.00098** | 0/20 → 17/20 |
| translation 0.15 | 0/20 | **14/20 = 70%** | 14 | 0 | **0.00012** | 4/20 → 19/20 |
| gain 0.20 | 0/20 | **17/20 = 85%** | 17 | 0 | **1.5×10⁻⁵** | 0/20 → 17/20 |

### 25.1 What this establishes

**Nothing in the method was π0.5-specific.** A different architecture — autoregressive
backbone with a regression head instead of flow matching, PyTorch instead of JAX, a different
training pipeline and image preprocessing — is repaired by the identical law, on all three
fault families, with zero regressions on the faulted cells. The gain estimate is
`β̂ = −0.823, −0.805, −0.817` against −0.800, within 3%, with `--clip 0.95` so the bound
cannot coincide with the truth (10% railing on x and z, flagged, not confounding).

**The calibration is a property of the robot, not the policy.** `W` and `M` transferred
across backbones with no re-identification. Combined with §23 (transfer across task suites),
one healthy-data calibration covers a robot, whichever policy drives it and whatever it is
asked to do.

### 25.2 What is different, stated plainly

OFT is **more fragile**: the frozen policy scores 0/20 on all three faults where π0.5 kept
4/20 on translation. It also recovers less on rotation (55% vs 85%) though the same on gain
(85% vs 85%). Two causes are plausible and this design does not separate them: OFT's action
chunk is executed more open-loop (8 steps vs 5 before replanning), and its policy is less
robust to off-distribution states. The healthy-robot offset phantom is also larger than
under π0.5 (x 0.035, z 0.055 vs 0.022, 0.042) — expected, since the plant was fitted on
π0.5's command distribution — and it did not matter for rotation-only correction.

The healthy control shows **one regression (20/20 → 19/20)** with the law running on a
robot with nothing wrong — and inspecting that episode, **it was the law's doing, not
noise.** Its rotation estimate reached |f̂| = 0.037, 0.003, 0.062 against a typical healthy
phantom of 0.009, 0.014, 0.006: four to ten times the usual push, applied to a robot that
needed none, and the episode failed. One in twenty, but a real harm mode — a transient
phantom integrated and acted on. It is the first regression in this record with a
diagnosed cause.

The obvious mitigation — a confidence gate, acting only once `|f̂|` has exceeded a threshold
for K consecutive steps — was evaluated **offline on the stored trajectories before touching
the law**, and it does not remove the harm:

| gate | healthy OFT episodes that open | the harmful episode | faulted episodes open at (median step) |
|---|---|---|---|
| none | 20/20 | opens | ~7 |
| \|f̂\| > 0.02 for 5 steps | 13/20 | **opens, step 38** | 6–10 |
| \|f̂\| > 0.03 for 5 steps | 5/20 | **opens, step 40** | 7–16 |
| \|f̂\| > 0.03 for 10 steps | 4/20 | **opens, step 45** | 12–21 |

The phantom in that episode was *sustained*, not transient, so it is indistinguishable from
a real fault by any test on the estimate alone, while every gate delays genuine repair by
one to two seconds. A dwell gate is therefore not implemented. Separating that episode
would need information the estimator does not have — most plausibly the task-level
observation that the robot was already succeeding — which is a different design.

**Identification is uneven on OFT.** Separation against the matched healthy run on the
rotation fault (true +0.100): rx **87%**, ry **23%**, rz **87%**. π0.5 identified all three at
90–100% (§19). `ry` is the least-excited rotation channel, and OFT's command distribution
evidently excites it less still; the correction on `ry` was therefore mostly absent, which
is consistent with OFT recovering 55% where π0.5 recovered 85% on this cell.

### 25.3 Aggregate

**610 paired episodes** across two backbones, two suites, four fault families, three
severities and four time profiles: **245 fixed, 6 broken** (1.0% regression).

## 27. ALOHA on the real policy: a null, its cause, and the rerun (added 2026-09-03)

### 27.1 Calibration on `pi0_aloha_sim` is clean, and the policy does not re-anchor

Eight healthy rollouts of the real policy (transfer-cube, **4/8** successes — a modest
policy) give a per-joint FIR plant with **R² 0.989–1.000 on all 14 joints**, and open-loop
replay gives **`M ≈ I`** (diagonals 0.98–1.02, cond 7.3). The healthy control on the twenty
paired episodes: frozen 5/20, law running 7/20 — no harm on a healthy robot.

The open question of §26.2 — does π0 re-anchor its targets to measured state and hide a
constant fault? — was answered by a log-mode run under the fault (three episodes,
`faulted_log_off005.json`). Under +0.05 rad on joints 0–5:

| | tracking `q − u`, joints 0–5 | residual `q − pred`, joints 0–5 | net drift, joint 1 |
|---|---|---|---|
| healthy | ≈ 0 | ≈ 0 | 0.76 |
| **faulted +0.05** | 0.049, 0.050, 0.054, 0.048, 0.061, 0.050 | **0.050, 0.056, 0.049, 0.051, 0.050, 0.050** | 0.62 |

**The residual reads the injected fault exactly, on every faulted joint, and drift is
unchanged.** π0 does not re-anchor. The fault is fully observable to the estimator.

### 27.2 And yet: 0/20 → 0/20 at both severities — because of one constant

With the fault this visible, the paired runs returned `off005` **0/20 → 0/20** and `off010`
**0/20 → 0/20**, with the estimate sitting at `f̂ ≈ 0.004` against a true 0.050. The law was
not integrating a residual that was plainly there.

The cause is the normaliser `est / (1 + ‖r‖²/ρ²)`. `norm_r` had been carried over as 0.05 —
sized, in LIBERO, against a 6-D residual of ~0.034. Over 14 ALOHA joints the residual norm
is **0.192**. Replaying the law offline on the stored residuals:

| `norm_r` | attenuation `1/(1+(‖r‖/ρ)²)` | replayed `f̂` (true +0.050) |
|---|---|---|
| **0.05 (as run)** | **0.06** | **+0.003** ← matches the observed 0.004 |
| 0.1 | 0.21 | +0.011 |
| 0.2 | 0.52 | +0.026 |
| 0.4 | 0.81 | +0.041 |
| 0.8 | 0.95 | +0.048 |

Every update was cut to six percent. This is the **fourth** instance in this record of a
constant set without measuring the quantity it is compared against — the deadzone (§4),
`pe_min` for the FIR regressor (§21.2), the projection clip (§12.1, §22.2), and now the
residual normaliser on a plant of different dimension. The rule that follows is mechanical:
**every threshold in the law is a ratio against a measured scale, and the scale must be
re-measured on every new plant.** A robustness term tuned on one robot is a wrong constant
on the next.

The stub dry run of §26 did not catch it because the stub's fault visibility was assessed at
the residual, not through the law — the law's own output under the *planned* stub was never
read. It should have been.

The rerun uses `norm_r = 0.4` on the same paired episodes.

### 27.3 With the normaliser sized correctly: identified, and still not repaired

Same paired episodes, `norm_r = 0.4`:

| fault | frozen | corrected | separation on joints 0–5 (vs healthy control) | clean joints, max \|sep\| |
|---|---|---|---|---|
| +0.05 rad, j0–5 | 0/20 | **0/20** | 0.042, 0.045, 0.042, 0.042, 0.043, 0.043 (**85–91%**) | 0.007 |
| +0.10 rad, j0–5 | 0/20 | **0/20** | 0.069, 0.071, 0.069, 0.069, 0.069, 0.069 (69–71%) | 0.006 |

**Identification transfers to the second manipulator.** The estimate lands within 10–15% of
truth on every faulted joint, leaks under 0.007 rad onto clean joints, and reaches 80% of its
final value by step ~30 — 0.6 s of a 6 s episode. The clip guard fired only on joint 13, the
right gripper, which is never corrected: gripper commands are binary and the servo cannot
track them, so its "estimate" is a constant that saturates. Cosmetic, and noted so it is
not mistaken for a problem.

**Repair does not.** 0/20 → 0/20 at both severities. Against a healthy baseline of 5/20 on
these episodes, `P(0/20 | p = 0.25) = 0.003`, so this is a real failure to restore, not noise
around a low floor.

Three measured facts bound the explanation:

1. A +0.05 rad offset on six left-arm joints displaces the left gripper by **5.3 cm** — two
   to three cube widths. During the ~30-step transient the arm is that far off; afterwards
   ~0.8 cm.
2. **Both arms begin moving at step 1** in every healthy episode. There is no idle window in
   which the estimate can converge before the faulted arm is asked to do something precise.
3. The healthy policy itself succeeds only 25% of the time. The task has almost no margin.

So the leading hypothesis is that the **transient loses the task**: 0.6 s at 5 cm on a
task with no margin, from the first step. The decisive test is an *oracle* — the exact
fault subtracted from step 0, no estimator. If it restores ~5/20, the transient is the
cause and the fix is on the law's speed or on carrying the estimate across episodes; if it
also returns 0/20, the fault path differs from the healthy one and the bug is in the
plumbing. That run is in progress.

The transient, converted to gripper displacement with the measured 5.3 cm per 0.05 rad:

| step | 0 | 5 | 10 | 20 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| residual gripper error (cm) | 5.3 | 5.4 | 4.1 | 2.5 | 1.6 | 1.0 | 0.7 | 0.8 |

**24 steps above one cube width (2 cm), 56 steps above 1 cm, steady state 0.76 cm.** Half a
second at more than a cube width, on a task whose healthy success is 25% and whose faulted
arm is in motion from step 1.

### 27.4 The oracle: the transient is the cause

Same paired episodes, the exact fault subtracted from step 0, no estimator:

| arm | success |
|---|---|
| healthy, frozen (control) | 5/20 |
| faulted, frozen | 0/20 |
| faulted, **oracle static correction** | **4/20** |
| faulted, adaptive (85–91% identified, 0.6 s transient) | 0/20 |

The oracle restores the task to the healthy rate (4/20 against 5/20; the exact test sits at
its floor of 0.125 with four discordant pairs). So the faulted execution path is sound and
the fault is recoverable — **what loses the task is the half-second at more than a cube
width while the estimate converges**, on a task with no margin and an arm that is precise
from step 1.

This is a different failure from anything in LIBERO. There, the transient was ~15 control
steps at 20 Hz on a task whose first second is a free-space reach; the arm had time. Here
the transient is ~30 steps at 50 Hz — shorter in seconds — but the task punishes the first
centimetres immediately. **Identification speed, not identification accuracy, is the
binding constraint on this manipulator.**

Two remedies follow, both testable on the same episodes: a smaller fault, where the
transient stays under a cube width (0.02 rad → 2.1 cm peak, ~0.3 cm steady); and carrying
the estimate across episodes, so that after the first episode there is no transient at all.
The second is a deployment choice rather than a change to the law — a hardware fault is
persistent, and resetting the estimate every episode was a deliberate constraint for the
LIBERO claims ("no learning across episodes"), not a requirement of the method.

### 27.5 A smaller fault does not help, and jitter is not the cause

Same episodes, fault reduced to **+0.02 rad** on joints 0–5 — 2.1 cm at the gripper, under a
cube width from step 20 on, 0.4 cm at steady state:

| | frozen | corrected | identification (true +0.020) |
|---|---|---|---|
| +0.02 rad | 0/20 | **0/20** | 0.019, 0.021, 0.019, 0.019, 0.019, 0.019 (**94–106%**) |

Identification is now essentially exact, the transient is a third of what it was, and the
task still does not come back — while the oracle at a fault 2.5× larger restored 4/20. The
frozen policy is destroyed by a 2 cm offset on one arm (0/20 against a healthy 5/20), which
says how little margin this task has.

**Jitter is ruled out.** After the transient, the applied correction changes by
0.00004–0.00006 rad per step — **0.004–0.006 cm at the gripper, at 50 Hz** — a fifth of
LIBERO's 0.019 cm per step, where repair works. The estimator is not shaking the arm.

What separates the oracle from the adaptive arm is therefore confined to the first ~20
steps. The warm-start run, in which episodes 1–19 begin at the converged estimate and have
no transient at all, is the decisive test of that and is in progress. If it restores the
task, identification speed is the whole story on this manipulator; if it does not, the
difference between an exact static correction and a converged adaptive one is something
this analysis has not found.

On the oracle itself: it solves episodes {2, 4, 5, 9} where the healthy policy solves
{2, 5, 6, 13, 14}. The overlap is partial, but the policy's own two healthy arms differ on
four episodes at identical seeds, so equivalence holds at the rate level (4/20 vs 5/20), not
episode for episode.

### 27.6 Warm start removes the transient, and the task still does not come back

Estimate carried across episodes, same paired episodes, +0.05 rad on joints 0–5:

| | success |
|---|---|
| episode 0 (full transient) | 0/1 |
| **episodes 1–19, starting at `f̂ ≈ 0.037–0.041` (no transient)** | **0/19** |
| oracle, exact −0.050 from step 0 | 4/20 |

The carry worked — every later episode began within 20% of the fault — and it changed
nothing. **The transient is not the cause.** Between the oracle and a warm-started
adaptive arm two things remain: a steady residual of ~0.01 rad (≈ 1 cm at the gripper, half
a cube width), and the estimator continuing to update during the episode.

Two static runs separate them, both queued: the law's own converged estimate (−0.041)
applied *frozen*, and a 90% correction (−0.045). If the frozen estimate restores the task,
online updating during the episode is what hurts and the remedy is to stop adapting once
converged; if neither restores it, this task tolerates less than a centimetre of residual
offset and the law's 80–90% identification is short of that by the task's margin, not by
design.

The 0.02 rad cell of §27.5 already hints at the second reading — identification there was
94–106%, residual ≈ 0.1 cm, and it still failed with the transient present — so the two
effects may each be sufficient on a task this fragile.


**Addendum: the estimate wanders in a fixed pattern, and it is a closed-loop effect.** Within
a warm-started episode the applied correction on joints 0–5 swings by a median 0.019 rad —
**2.0 cm, a cube width** — and the *mean* trajectory across episodes has the same shape every
time: 0.040 at step 0, a dip to 0.036 near step 25, a climb to 0.043–0.044 by step 75, then
flat. Phase-locked, not noise. Replaying the law on stored open-loop residuals produces only
0.4–0.7 cm of wander, and a slower gain makes it worse (γ 0.08 → 0.01: 0.38 → 0.72 cm), so
the gain is not the lever.

**It is not error in the healthy plant model.** With no fault present, the residual on joints
0–5 is 0.0001–0.0003 rad (0.01–0.03 cm) in every phase of the episode, rms ≤ 0.0025 — the
FIR fits the healthy robot essentially exactly at motion onset, mid-episode, and at the end.
So the early dip in `f̂` arises only under the fault and the correction, in closed loop:
whatever the estimator is tracking during the first second, it is not something the plant
model gets wrong on a healthy arm.

### 27.7 The dip is the zero-initialised FIR history

The phase-locked dip has a mechanical cause, found by replaying the law on the stored faulted
log from a warm `f̂ = 0.040`:

| history initialised as | `f̂[0:6]` at steps 0, 2, 4, 6, 10, 15, 25 | `‖r‖` at step 0 |
|---|---|---|
| **zeros (as run)** | 0.037, 0.031, 0.026, **0.024**, 0.027, 0.032, 0.037 | **1.415** |
| current joint position `q₀` | 0.037, 0.036, 0.036, 0.035, 0.035, 0.037, 0.039 | 0.198 |
| closed-loop warm run, measured | 0.040 (step 0), **0.025 (step 5)**, 0.028, 0.032, 0.036 (step 25) | — |

The FIR history is seeded with zeros at every episode start. For delta commands that is a
valid history — "no motion" — and in LIBERO it was harmless. For absolute joint targets it
means "target = 0 rad" on an arm resting at `[0, −0.96, 1.16, …]`: for the first `K_FIR`
steps the prediction is wildly wrong (`‖r‖ = 1.4`), the normaliser zeroes every update, and
`f̂` decays toward zero at `γ` per step. A **1.6 cm dip in the first ten steps of every
episode, cold or warm** — precisely when the faulted arm is already moving and the task has
no margin. Seeding the history with the current joint position removes it.

This is the fourth defect in the ALOHA port and the first that was invisible in LIBERO by
construction rather than by luck. It was also nearly missed a second time: the offline gate
I wrote to confirm it demanded the `q₀` replay stay above 0.038, and it bottoms at 0.0347 —
a 0.5 cm dip against the 1.7 cm one under test — so the gate rejected a mechanism the
numbers plainly show. Thresholds, again.

One more thing the replay exposes, checked and dismissed: the FIR's tap at lag 6 is 0.286,
nearly as large as lag 0 (0.292), which looked like servo settling extending past the
window. Refitting with 12 and 20 lags does not make the taps decay — the last tap still
carries 0.15 at K = 12, the taps scatter at K = 20, and R² moves from 0.99970 to 0.99973.
It is collinearity: the target and the position track each other so closely that
least-squares can place weight anywhere along that direction while keeping the sum at 1.
Harmless for prediction, harmless for an offset residual (the sum is what matters), and not
the cause of anything. No action.

The fixed-history cell runs on the same paired episodes after the static bracket.

### 27.8 The task's margin: under half a centimetre

Static corrections, frozen for the whole episode, no estimator, same paired episodes:

| correction applied | residual offset at the gripper | success |
|---|---|---|
| none (frozen, faulted) | 5.3 cm | 0/20 |
| −0.041 rad (82%, the law's converged estimate) | 0.95 cm | **0/20** |
| −0.045 rad (90%) | 0.53 cm | **0/20** |
| −0.050 rad (100%, oracle) | 0 | **4/20** |
| healthy, no fault | 0 | 5/20 |

**Transfer-cube tolerates less than 0.5 cm of steady offset on the left arm.** A 10%
under-correction is as fatal as no correction. That is the margin the law has to hit, and
an estimate at 85–91% of the fault leaves 0.5–0.95 cm — outside it. Nothing about the
estimator's speed, its wander, or the history dip changes this bound; it is a property of
the task and the policy's 25% healthy success.

Turning the margin into a requirement — the residual must stay under 0.5 cm, i.e. under
0.0047 rad of uniform offset on these six joints — gives the identification accuracy the
law must reach as a function of fault size:

| fault | gripper displacement | required identification |
|---|---|---|
| 0.02 rad | 2.1 cm | **76%** |
| 0.05 rad | 5.3 cm | **91%** |
| 0.10 rad | 10.6 cm | **95%** |

Measured: 94–106% at 0.02 rad (inside), 84–91% at 0.05 rad (outside), 69–71% at 0.10 rad
(far outside). The bound explains every ALOHA success count in this section.

So the ALOHA result decomposes cleanly:

1. **Identification transfers** — 85–91% at 0.05 rad, 94–106% at 0.02 rad, ≤0.007 rad leak
   onto clean joints, on a plant identified from eight healthy rollouts.
2. **Repair on transfer-cube requires >95% identification held from step 1**, because the
   residual must stay under 0.5 cm and the arm is precise from the first step. At 0.05 rad
   the law's steady accuracy is short of that by the task's margin. At 0.02 rad it is inside
   the margin — and that cell failed for a different reason, the zero-history dip, which let
   the full fault act during the first ten steps (§27.7).

The one cell that can demonstrate repair on this task is therefore **0.02 rad with the
history fixed**: identification inside the margin, no dip. It is queued after the fixed
0.05 cells. If it restores the task, the ALOHA story is "identification transfers; repair
needs identification accuracy matched to the task's margin"; if it does not, the remaining
suspect is the first ten steps, where the estimate is still converging from zero, and the
honest fix is a warm start — which is what a persistent hardware fault gets in deployment
anyway.

### 27.9 With the history fixed: the dip is gone, and the 0.05 rad result is closed

Same paired episodes, FIR history seeded at the current joint position:

| | early `f̂[0:6]` (steps 0, 5, 10, 25, 50) | identification | success |
|---|---|---|---|
| cold start | 0.001, 0.012, 0.020, 0.036, 0.041 | 84–90% | **0/20** |
| warm start | 0.041, 0.037, 0.036, 0.038, 0.040 | 85–91% | **0/20** |

The dip is gone: the cold start now converges monotonically, and the warm start holds
within 12% of its seed from step 0 instead of collapsing by 40%. Identification is
unchanged at 84–91%. And the task does not come back — **which is what §27.8 says it must
not**: an estimate at 84–91% leaves 0.5–0.8 cm at the gripper, and a frozen static
correction at 90% (0.53 cm) already returned 0/20.

So the 0.05 rad result is closed, and it is a clean statement rather than a defect list:
**on transfer-cube, repair requires holding the residual under 0.5 cm for the whole
episode, and the law's steady-state accuracy on a 0.05 rad fault is 84–91%, which leaves
0.5–0.8 cm.** Every dynamic effect investigated — the transient, the wander, the
zero-history dip — was real, was fixed or ruled out, and none of them was the binding
constraint. Accuracy relative to the task's margin is.

The 0.02 rad cell with the history fixed is inside that margin (identification 94–106%,
residual ≈ 0.1 cm) and is running; a static oracle at 0.02 is queued behind it as its
ceiling.

### 27.10 Inside the margin at steady state, outside it during the climb

0.02 rad, history fixed, cold start, same paired episodes:

| | value |
|---|---|
| identification, joints 0–5 | **95%, 104%, 94%, 95%, 95%, 95%** |
| residual at the gripper, steps 0 / 10 / 20 / 30 / 50 / 100 | 2.08 / 1.30 / 0.85 / 0.61 / 0.37 / 0.08 cm |
| steps above the 0.5 cm margin | **39** |
| success | **0/20** |

Steady state is inside the margin by a wide factor — the residual is 0.08 cm by step 100 —
and the task still fails, because a cold start spends the first 0.8 s above 0.5 cm while the
estimate climbs from zero. The bound of §27.8 is not "reach the margin"; it is **"never leave
it"**, and on an arm that is precise from step 1 no cold-started estimator can satisfy that.

That leaves exactly one adaptive configuration that can: a warm start at 0.02 rad with the
history fixed, which begins each episode at ~95% of the fault and holds the residual near
0.1 cm throughout. It is the last cell, queued behind a 0.02 oracle that gives its ceiling.
If it repairs, the ALOHA statement is complete and honest: **identification transfers;
repair requires the residual to stay inside the task's margin for the whole episode, which
this law achieves at 0.02 rad only with a persistent estimate.** A persistent estimate is
what a persistent hardware fault gets in deployment, and the per-episode reset in the LIBERO
results was a constraint chosen to make a stronger claim there, not a property of the method.

**The 0.02 oracle: 8/20, p = 0.0078.** The exact fault subtracted from step 0 restores the
task to 40% — above the 5/20 healthy-frozen control and inside the noise band the policy's
own two healthy arms span (5/20 and 7/20). So the ceiling for the warm-started 0.02 cell is
roughly 5–8 of 20, and a repair anywhere in that range would be the healthy rate.


### 27.11 Warm-started 0.02 rad, history fixed: still 0/20 against an oracle of 8/20

The last adaptive configuration that could stay inside the margin for a whole episode:

| | success |
|---|---|
| frozen, faulted | 0/20 |
| **adaptive, warm start, history fixed** | **0/20** |
| oracle, exact −0.020 from step 0 | **8/20**, p = 0.0078 |

From the warm episodes' own records:

```
identification j0-5 (true 0.020): ['93%', '98%', '92%', '91%', '93%', '93%']
warm episodes: f_hat at step 0 median 0.0181 rad (90% of fault)
within-episode range median 0.0041 rad = 0.43 cm;  worst-case min 0.0146 rad -> residual up to 0.57 cm
residual gripper error, warm episodes: mean over steps 0.17 cm; fraction of steps > 0.5 cm: 2%; > 0.25 cm: 28%
```

The estimate begins each episode near the fault and identification is essentially exact, yet
the task fails where an exact static correction succeeds 8 times in 20. **The margin bound
of §27.8 is therefore necessary but not sufficient.** Two things separate the oracle from
this arm: the law keeps updating during the episode, and its correction is what the plant
residual says it should be rather than exactly the fault. The clean test is the law's own
converged estimate (0.019 rad, 95%) applied *frozen* — no estimator, no wander — on the same
episodes. If it recovers ~8/20, mid-episode updating is what breaks the task and the remedy is
to stop adapting once converged; if it returns 0/20, this task distinguishes 95% from 100%
correction, and no estimator that stops short of exact can repair it. That run is in progress.

### 27.12 It is the updating, not the estimate

The same correction magnitude, applied two ways on the same paired episodes, 0.02 rad fault:

| correction | mean residual | success |
|---|---|---|
| none (frozen, faulted) | 2.12 cm | 0/20 |
| **0.019 rad (95%) applied frozen** | 0.11 cm | **5/20**, p = 0.0625 |
| 0.019 rad reached by the **updating** law (warm, history fixed) | 0.17 cm | **0/20** |
| 0.020 rad (100%) oracle | 0 cm | 8/20 |

**The law's own converged estimate restores the task — but only when it stops moving.** Rows
two and three carry the same number to the same joints; the only difference is whether the
estimator keeps writing to it. Frozen: 5/20, the healthy-frozen rate. Updating: nothing.

The mechanism is in §27.11's measurements. The updating correction wanders within an episode
by a median **0.0041 rad = 0.43 cm**, with worst-case excursions to 0.57 cm — comparable to
the entire 0.5 cm margin — even though its *mean* residual, 0.17 cm, is comfortably inside.
A mean inside the margin is not enough: the correction has to *stay* inside it, and a
continuously updating estimator on a 14-joint plant does not.

**This is the ALOHA finding, and it is not a defect in the estimator.** Identification
transfers to a second manipulator and a second action interface — 94–106% at 0.02 rad, from
a plant fitted on eight healthy rollouts. What does not transfer is the assumption that a
*continuously adapting* correction is harmless. On LIBERO it was: §18 measured 0.019 cm per
step of jitter there and repair worked anyway, because a 20 Hz Cartesian reach has
centimetres of slack. On transfer-cube, with a sub-half-centimetre margin and an arm that is
precise from step 1, the same jitter is the difference between 5/20 and 0/20.

The obvious remedy is to adapt until the estimate settles and then hold it
(`--freeze-after`). **That run has now returned 0/20, and it refutes the remedy as I
proposed it.** The reason is worth more than the proposal was:

| | froze at | residual held | success |
|---|---|---|---|
| warm + freeze after 30 steps | 0.0155 rad (78%) | **0.48 cm** | 0/20 |
| pure static 0.019 rad | 0.019 (95%) | 0.11 cm | 5/20 |

Thirty steps is not convergence at `γ = 0.08` — the estimate is only at 78% there, and
freezing locks in a 0.48 cm residual, sitting exactly on the margin. Worse, **combined with
`--warm-start` the flag is self-defeating**: each episode carries forward whatever value was
frozen, so the estimate gains only 0.0009 rad per episode and never reaches the fault. I
wired the two flags together without noticing they fight.

So the corrected claim is narrower than §27.12's first draft. What the frozen-versus-updating
pair establishes is that **a stationary correction at 95% repairs where a moving correction
of the same mean does not**. Whether an estimator can *become* stationary at a good enough
value on this task is a separate question, and freezing after 100 and 200 steps — where the
updating run has actually reached ~0.019 — is the test now running.

### 27.13 Identify once, then hold: the ALOHA result, positive

Freezing *within* an episode failed at every cut point (30, 100, 200 steps → 0/20), because
each still adapts during the opening steps and, combined with the carry, never converges.
The scheme the data actually implies is to adapt for one **episode** and then hold:

| scheme | within-episode wander | residual | success |
|---|---|---|---|
| frozen, faulted | — | 2.12 cm | 0/20 |
| adaptive, updating throughout | 0.43 cm | 0.17 cm mean | 0/20 |
| freeze after 30 / 100 / 200 steps | — | 0.13–0.49 cm | 0/20 |
| **adapt episode 0, then hold** | **0.000 cm** | **0.12 cm** | **5/20**, p = 0.0625 |
| oracle, exact fault from step 0 | 0 | 0 | 8/20 |
| healthy, no fault | — | — | 5/20 |

`--identify-episodes 1`: episode 0 pays the transient and fails; episodes 1–19 run a
perfectly stationary correction at `f̂ = 0.0189` — **94% of the true fault** — and recover
**5 of 19**, the healthy-frozen rate, from a frozen policy that scores zero.

**The second manipulator works, with a stated precondition.** The law identifies a
joint-space fault on a 14-DOF bimanual arm to 94% from eight healthy rollouts, and repairs
the task to the healthy rate — provided the correction is *stationary while the task runs*.
On LIBERO that precondition was invisible because a 20 Hz Cartesian reach tolerates
centimetres of jitter; on transfer-cube, with a sub-half-centimetre margin, it decides
everything.

That is a real limit on "fully online" adaptation and it should be stated as one: **on a
tight-margin task the estimate must be identified in a sacrificial episode and then frozen,
not updated continuously.** For a persistent hardware fault this costs one episode, which is
what a calibration procedure costs anyway.

### 27.14 The predictive criterion

The ALOHA and LIBERO outcomes are both predicted by one comparison, measurable before any
repair run:

> **Compare the task's spatial margin against the correction's within-episode variation.**
> The margin is measured by a static-correction sweep (how much residual offset the frozen
> policy tolerates); the variation is measured from the estimator's own trajectory on a
> single faulted episode. Repair works when the variation fits inside the margin.

| | task margin | correction wander | outcome |
|---|---|---|---|
| LIBERO, translation 0.15 | ≫ 1 cm (a reach) | 0.019 cm/step | repair, 4/20 → 19/20 |
| ALOHA, transfer-cube, updating | 0.5 cm | 0.43 cm within episode | no repair, 0/20 |
| ALOHA, transfer-cube, held | 0.5 cm | 0.000 cm | repair, 0/20 → 5/20 |

Neither quantity needs the fault to be known, and neither needs a repair experiment. This is
the deployment test the method was missing.

## 28. The two floor-limited suites at n = 40 (added 2026-09-04)

§10.2 said `goal` and `libero_10` sat at `p = 0.0625` because five one-way discordant pairs
is the exact test's floor, not because the effect was weak, and that n ≈ 30–40 would settle
them. Same configuration (uniform +0.05, rotation-only), inits 45–48, paired.

### 28.1 `libero_goal`

| n | frozen | corrected | fixed | broken | exact McNemar |
|---|---|---|---|---|---|
| 20 | 8/20 = 40% | 13/20 = 65% | 5 | 0 | 0.0625 (floor) |
| **40** | 15/40 = 38% | **29/40 = 72%** | **15** | 1 | **0.00052** |

Resolved. The effect size did not move (+25 → +35 points, inside run-to-run noise); the test
simply gained the discordant pairs it needed. One regression appears at n = 40 — the first
on this suite — and it is one in forty against fifteen repairs.

The regression is task 5, init 46. Its correction was ordinary-sized (|f̂| 0.042, 0.034, 0.048
against a run mean of 0.035, 0.031, 0.037), and the frozen policy itself flips on that task —
3 of 4 inits either way in both arms. Policy stochasticity, not the law. Unlike the OFT
healthy-control regression of §25.2, which the law caused.

### 28.2 `libero_10`, and the headline at its best n

| n | frozen | corrected | fixed | broken | exact McNemar |
|---|---|---|---|---|---|
| 20 | 0/20 | 5/20 | 5 | 0 | 0.0625 (floor) |
| **40** | 0/40 | **15/40 = 38%** | **15** | 0 | **6.1×10⁻⁵** |

The long-horizon suite, from a frozen policy that never succeeds, to 15 of 40. Zero
regressions.

**The four-suite headline, each suite at its best n:**

| suite | n | frozen | corrected | fixed | broken | McNemar |
|---|---|---|---|---|---|---|
| `libero_spatial` | 20 | 8/20 = 40% | 18/20 = 90% | 10 | 0 | 0.0020 |
| `libero_goal` | 40 | 15/40 = 38% | 29/40 = 72% | 15 | 1 | 0.00052 |
| `libero_object` | 20 | 5/20 = 25% | 16/20 = 80% | 11 | 0 | 0.00098 |
| `libero_10` | 40 | 0/40 = 0% | 15/40 = 38% | 15 | 0 | 6.1×10⁻⁵ |
| **pooled** | **120** | **28/120 = 23%** | **78/120 = 65%** | **51** | **1** | **6.1×10⁻¹³** |

Every suite is individually significant. §10's pooled figure (26% → 65%, `p = 9.3×10⁻¹⁰`,
80 episodes) is superseded by this one.

**Aggregate for the online law, LIBERO and OpenVLA-OFT, every paired run:** 690 episodes,
**275 fixed, 7 broken** (1.0%). ALOHA is reported separately (§27) because its repair uses
the identify-then-hold scheme, not continuous adaptation.

### 28.3 The video script was not the benchmark condition (found 2026-09-05)

Rendering a `libero_10` comparison video from five pairs that the n = 40 run had fixed gave
**0 of 5** repaired. The benchmark had not changed; `compare_video.py` had two differences
from `adaptive_law.py` that the LIBERO benchmark never had:

1. **The widened recording camera was the policy's input.** `--rec-cam agentview
   --rec-fovy 62` set the fov of the *same* camera the policy observes, so the policy saw a
   62° view where it was trained at 45°. The docstring said the recording camera was
   separate; the code did not do that. Mean pixel difference wide-vs-native: 28 of 255.
2. **The rotation residual was an axis-angle difference**, not the so3 increment the
   `*_so3` plant and M were fitted on (§so3). The estimator therefore ran on a residual its
   calibration did not describe.

Both fixed: the policy always receives the native `agentview` image, the video frame is a
second render with the fov swapped in and out around it, and `rot_delta` is used. Rerun on
the same five pairs: **1 of 5 repaired** (task 1 init 48, fault estimate +0.046/+0.024/+0.044
against a true +0.05 on rotation, corrected run done at step 314; frozen timed out at 520).
One in five on a 38 % suite, conditioned on a previous success, is inside stochastic
variation; it is not the 0 of 5 the confounded script gave.

**What this means for the earlier videos.** `adaptive_vs_frozen.mp4` (spatial),
`_goal`, `_object`, `_oft` were all rendered with the widened camera reaching the policy.
Both panels of each video saw the same input and the same fault, so they are still fair
frozen-versus-corrected comparisons on that input; they are not renders of the benchmark
condition, and the success rates in the record come from `adaptive_law.py`, never from the
video script. `_libero10` is the first video rendered under the benchmark condition.
`--only-repaired --max-clips N` was added so a demonstration video can be assembled from
the pairs the correction actually repairs on that render, with the skip count printed.

**The assembled video.** Walking the fifteen pairs fixed at n = 40 in order, the corrected
run succeeded on **4 of the first 10** (tasks 9, 0, 5, 0 at inits 47, 47, 45, 48; done at
steps 328, 332, 188, 312 against a frozen timeout of 520 on every one); the render stopped
at four clips. 4 of 10 on pairs that had *all* succeeded before is the suite's marginal
38 %, not something higher: **conditioning on a previous success buys almost nothing**,
so the set of "fixed" pairs is not a stable set of episodes but a draw from a policy whose
outcome is dominated by its own sampling. This is the n = 20 ±11-point noise of §10 seen
from the other side, and it is why every claim in this record is a paired rate, never a
list of episodes.

**The four earlier videos, re-rendered under the benchmark condition (2026-09-05).**
Same recipe as `_libero10`: candidates are the pairs fixed in the stored run, distinct
tasks first, `--only-repaired --max-clips 4`; every final frame was checked by eye.

| video | candidates walked | frozen succeeded (skipped) | corrected failed (skipped) | clips | estimate range on rotation |
|---|---|---|---|---|---|
| `adaptive_vs_frozen.mp4` (spatial, π0.5) | 6 | 2 | 0 | 4 | 0.028–0.056 (true 0.05) |
| `_goal` (π0.5) | 9 | 4 | 1 | 4 | 0.015–0.055; one clip finished at 0.02 on rz with ry/rx near zero |
| `_object` (π0.5) | 9 | 2 | 3 | 4 | 0.021–0.053 |
| `_oft` (OpenVLA-OFT, rotation +0.10) | 6 | 0 | 2 | 4 | 0.043–0.099 (true 0.10) |

Over the pairs where the frozen policy failed on the re-run, the correction repaired
16 of 22. All five LIBERO videos in `results/phase05/` are now benchmark-condition renders;
the earlier confounded versions are in git history before this commit. The goal clip that
finished with a small estimate is left in: the video is a demonstration of the paired
protocol, and it would misrepresent that protocol to cut the pairs where the policy's own
draw did some of the work.

### 28.4 ALOHA identify-then-hold at n = 40 (added 2026-09-05)

§27.13's `5/20`, `p = 0.0625`, sat on the same exact-test floor as `goal` and `libero_10`
did before §28. Same configuration (+0.02 rad on left-arm joints 0–5, `--identify-episodes
1`, norm_r 0.4, clip 0.08), seeds 200–239, paired.

| n | frozen | held correction | fixed | broken | exact McNemar | held f̂, joints 0–5 |
|---|---|---|---|---|---|---|
| 20 | 0/20 | 5/20 = 25% | 5 | 0 | 0.0625 (floor) | 0.0183–0.0191 (92–96%) |
| **40** | 0/40 | **15/40 = 38%** | **15** | 0 | **6.1×10⁻⁵** | **0.0190–0.0202 (95–101%)** |

Resolved, with zero regressions, and the identification episode (episode 0) failed as it
must. The rate moved from 25 % to 38 %, which is the ±11-point noise of §10 and §28.3, not
a change in the mechanism. The estimator's clip guard fired on joint 13 (the right
gripper) at 100 % of episodes: that joint is not in the correction set and its "estimate"
is the projection bound, exactly the coincidence §12.1 warned about — reported here so the
number is never read as an identification. 
**Healthy control, same seeds, same law, no fault (n = 40):**

| arm | success | vs healthy-frozen | exact McNemar |
|---|---|---|---|
| healthy, frozen | 17/40 = 42% | — | — |
| healthy, law running (identify-then-hold) | 14/40 = 35% | 5 fixed / 8 broken | 0.58 (null) |
| **faulted, repaired** (above) | **15/40 = 38%** | 4 / 6 | 0.75 (null) |

The held phantom estimate on a healthy arm is at most 0.0008 rad on joints 0–5, 4 % of
the fault, and the law neither helps nor harms a healthy robot at this n. The repaired arm
is indistinguishable from the healthy one on the same seeds. This is the full cell the
protocol requires: floor (0/40), repair (15/40), healthy ceiling (17/40), and a null for
the law on healthy data.

## 29. Faults below the controller: joint-level faults in simulation (added 2026-09-06)

Every fault in §1–§28 enters at the Cartesian action interface, above LIBERO's
operational-space controller. J-PARC's faults (joint lock, limited range of motion, friction)
enter below it, in the actuator, where the fault-to-motion map depends on the arm's
configuration through the Jacobian; §Limitations of the draft says that class is open. It
can be tested without a robot: `joint_fault.py` edits the MuJoCo model of the Panda after
each reset (friction loss, damping, actuator gain, joint range) or applies a constant joint
torque every step (`qfrc_applied`), and restores the model before the next episode. The
policy, the controller, the plant model and M are untouched.

### 29.1 Measure before setting constants: what each fault does to the end effector

40 control steps from task 0 init 45, holding still and under a constant +x command,
difference from the healthy motion:

| fault | holding | commanded +x |
|---|---|---|
| torque bias, joint 1 (shoulder), 5 N·m | +4.0 cm x | +4.4 cm x |
| torque bias, joint 3 (elbow), 5 N·m | +9.1 cm x, +7.8 cm z | +9.5 cm x, +8.3 cm z |
| friction loss +2.0, joint 3 | 0 | −4.0 cm x, −2.9 cm z |
| actuator gain 0.5, joint 1 | +25 cm x | +18 cm x, −12 cm z |
| joint lock ±0.05 rad, joint 3 | 0 | −10.3 cm x, −7.1 cm z |

A torque bias leaks through the OSC controller (no integral action) as a near-constant
displacement whether or not the arm is commanded: an additive fault, arriving from below.
Friction and a lock do nothing while holding and remove centimetres of commanded motion:
a loss of effectiveness, arriving from below. A halved gain drops the arm by a quarter
metre and is not a fault the policy could be asked to survive. So the two fault families
of §19 both exist at the joint level, and the magnitudes chosen (5 N·m, +2.0 friction,
±0.05 rad) produce 1–3 mm per step, the scale of the Cartesian faults already studied.

### 29.2 Estimate-only probe: does the Cartesian residual see them?

Three episodes each, π0.5, `libero_spatial`, estimator running but never applied
(`--estimate-only`), clip 0.30. Statistic: mean of the estimate over the last 50 steps
(the final value alone swings at the grasp). Separation = faulted − healthy phantom, in
units of the healthy phantom's across-episode standard deviation:

| fault | frozen | separation x, y, z (action units) | in sd of healthy | rotation channels |
|---|---|---|---|---|
| healthy phantom | 3/3 | 0.007, −0.008, 0.016 (sd 0.015, 0.009, 0.008) | — | ≤ 0.002 |
| torque j1 5 N·m | 3/3 | 0.003, 0.018, −0.068 | 0.2, 1.9, 8.4 | none |
| torque j3 5 N·m | 1/3 | 0.054, −0.049, 0.099 | 3.7, 5.3, 12.2 | none |
| friction j3 +2.0 | 3/3 | −0.074, −0.009, −0.076 | 5.1, 1.0, 9.5 | none |
| **lock j3 ±0.05** | **0/3** | **−0.224, −0.051, −0.183** | **15, 5.5, 23** | none |

Every joint-level fault is identified on translation and nowhere else, which is the
opposite of the Cartesian uniform fault (§14.3: rotation identified, translation not).
The lock, which zeros the frozen policy, is the cleanest identification in the record so
far: 15–23 standard deviations, consistent sign on all three episodes. The torque bias at
the elbow is identified at 4–12 sd with a positive z that matches §29.1's +8 cm.

**What the estimator is actually reading.** For a lock the physical fault is a range
limit, not an offset; the residual reports "commanded +x and +z did not happen", and the
estimator calls that a negative offset of 0.2. Correcting it means commanding more +x/+z,
which on a redundant 7-DOF arm the six free joints can partly deliver and a locked joint
cannot. Whether that repairs the task is the experiment; the estimate itself is
state-dependent by construction, so the law must track rather than converge. Paired cells
follow in §29.3, translation channels corrected, the healthy phantom subtracted as
`--bias` per Proposition 1.

### 29.3 Paired cells, π0.5, `libero_spatial`, n = 20, translation corrected, phantom subtracted

**Joint lock ±0.05 rad at the elbow: identified, corrected, not repaired.** `0/20 → 0/20`.
The estimate converges within 30 steps to `x −0.23, z −0.16` on every episode (sd 0.02
on x, no clip hit), the correction is applied, and nothing changes. This is a different
boundary from §19's: there the plant left the informative regime; here the plant's
*structure* changed. A lock removes a degree of freedom, so the reachable motion is a
subspace, and no additive command on the action interface moves the arm along a direction
the arm can no longer produce. The residual reports the missing motion faithfully, the
law inverts it faithfully, and the inverse of a rank-deficient map through a full-rank M is
a command the plant discards. **Identifiability is necessary for repair and not
sufficient**: the fault must be an input to the plant, not a change in its rank. J-PARC reports +6.8 points on π0.5 under a joint lock with its offline residual, which is
consistent with this reading rather than against it: a residual trained on faulted rollouts
can learn a *different route* through the six free joints, whereas our law can only push
harder along the direction the residual says is missing. On a lock that is the difference
between a learned re-plan and an inverted disturbance, and it is a real limitation of the
method to state. The torque and friction cells follow.

**Elbow torque bias 5 N·m: repaired, state-dependent estimate, n = 20 short of significance.**
`11/20 → 17/20`, 8 fixed, 2 broken, exact McNemar `p = 0.11`. The last-50-step estimate is
`x +0.13, z +0.11` with an across-episode sd of 0.07 — three to four times the sd of any
Cartesian-fault estimate in the record, and the signature §29.2 predicted: a constant joint
torque maps to a Cartesian offset through the Jacobian, so the offset the estimator sees
changes with the arm's configuration and the law tracks it rather than converging. The
+30 points are the same size as the Cartesian cells' effects and the test is at n = 20
with two regressions, so n = 40 is queued (as for `goal`, `libero_10` and ALOHA) before
this cell is claimed. This is the first repair of a fault that enters below the controller.

**Elbow friction +4.0: a ceiling, and a null.** `20/20 → 20/20`. Doubling §29.2's friction
still does not damage the policy on this suite: the OSC loop drives through a friction
deadband on its own, and the estimator running on top of a healthy outcome changes nothing
(zero regressions). The identification is there (a negative x/z estimate, the loss-of-
motion signature of §29.2) but there is no task loss for it to repair. A friction level that
does damage the policy is queued after the n = 40 torque run; §19's lesson applies — a cell
where frozen scores 100 % proves only that the law is harmless.

## 30. Third backbone: NVIDIA GR00T N1.7 (added 2026-09-06)

The official `nvidia/GR00T-N1.7-LIBERO` finetune (3B; Cosmos-Reason2 VLM, flow-matching
DiT action head, 16-step chunks; NVIDIA reports 97.65 % on `libero_spatial` at 720 steps)
is served by `openpi/groot_server.py` behind the same websocket protocol as π0.5 and
OpenVLA-OFT, so every script, the plant model and M are unchanged. Conventions were read
from Isaac-GR00T's own LIBERO wrapper, not assumed: same 180°-rotated 256×256 cameras, same
8-D state, same 7-D action with the gripper normalised then inverted. Three things had to
be fixed for this machine (gated backbone repo, no FlashAttention on Turing, a venv
without pip; `SETUP.md`). Inference is 3.8 s per chunk, so `--replan-steps 8` (GR00T's own
evaluation horizon) and a 20-episode paired cell takes about 45 minutes. A two-episode
healthy smoke through our client scored 2/2 before any fault.

### 30.1 Healthy control, `libero_spatial`, n = 20, law running on rotation

| arm | success | vs frozen |
|---|---|---|
| frozen, no fault | 18/20 | — |
| law running, no fault | 18/20 | 1 fixed / 1 broken |

A null, as required, with π0.5's plant and M. The rotation phantom is ≤ 0.005 (last-50-step
mean, sd ≤ 0.009); translation shows the same 0.02–0.03 phantom as on π0.5, which is why
translation stays uncorrected. GR00T's healthy rate at our 220-step cap is 90 %, against
NVIDIA's 97.65 % at 720 steps: the shorter cap costs a few slow episodes and is kept so
the cell is comparable to the other two backbones.

### 30.2 Rotation fault +0.10, rotation-only correction, `libero_spatial`, n = 20, paired

| arm | success | fixed | broken | exact McNemar |
|---|---|---|---|---|
| frozen, faulted | 0/20 | | | |
| **corrected** | **14/20 = 70%** | **14** | **0** | **1.2×10⁻⁴** |

**The third backbone is repaired with the calibration identified on the first.** A
rotation fault that zeros GR00T is recovered to 70 % with π0.5's plant model and M, no
retuning, no clip hit. The estimate settles at `rx 0.081, ry 0.042, rz 0.082` (last-50
mean, sd ≤ 0.011) against a true 0.10 — 81 %, 42 %, 82 %, and it is 62 % of the way there
by step 15. `ry` under-identified by half is the same channel that lagged on π0.5 (Fig. 1
of the draft) and on OFT; it is a property of the plant and M, not of the policy, which
is the point: nothing in the repair depends on which network is being repaired.

Same cell across the three backbones (rotation +0.10, rotation-only correction,
`libero_spatial`, n = 20):

| backbone | developer / family | frozen | corrected | fixed / broken | p |
|---|---|---|---|---|---|
| π0.5 (calibration source) | Physical Intelligence, flow matching | 0/20 | 17/20 | 17 / 0 | 1.5×10⁻⁵ |
| OpenVLA-OFT | Stanford/Berkeley, autoregressive + L1 head | 0/20 | 11/20 | 11 / 0 | 9.8×10⁻⁴ |
| **GR00T N1.7** | NVIDIA, Cosmos VLM + flow-matching DiT | 0/20 | 14/20 | 14 / 0 | 1.2×10⁻⁴ |

Three developers, three architectures, one calibration, 42 repairs and zero regressions
in 60 paired episodes.

**Elbow torque bias 5 N·m at n = 40: resolved.** `20/40 → 32/40`, **15 fixed, 3 broken**,
exact McNemar `p = 0.0075`. The first fault below the controller that the law repairs,
and it is claimed with its cost stated: three regressions in forty (7.5 %) against 1 % on
the Cartesian aggregate, two of them on the same task (6, inits 47 and 48). The estimate's
across-episode sd is 0.075 on x and 0.078 on z, unchanged from n = 20, and that spread is
the mechanism for the regressions: a joint torque maps to a Cartesian offset through the
Jacobian, so the offset the estimator sees, and cancels, is right for the configuration it
was measured in and wrong for the one the arm moves into. A tracking law with
one-episode memory follows that drift with a lag; on a task whose trajectory changes
configuration quickly the lag is a correction applied in the wrong direction for a few
steps, which is what a regression looks like here. The gap to a joint-space law (§27, which
would see the torque bias as a constant) is the obvious next step and is not run.

**Elbow friction, escalated until it damages: +10 (frozen 1/3), +20 (frozen 0/3).**
The +4.0 ceiling was the policy's tolerance, not the method's. Paired at +20:

| arm | success | fixed | broken | exact McNemar |
|---|---|---|---|---|
| frozen, faulted | 0/20 | | | |
| **corrected** (x,y,z, phantom subtracted, clip 0.30) | **8/20 = 40%** | **8** | **0** | **0.0078** |

**A loss-of-effectiveness fault below the controller, from zero to 40 % with zero
regressions.** The estimate reads `x −0.21, z −0.19` (last-50 mean; one clip hit in 20 on
z, the rest inside the box), the same loss-of-motion signature as the lock — and the
lock got nothing. The difference is physical, not statistical: friction is a force the
controller can overcome by commanding more, and the correction commands more; a lock is a
constraint no command overcomes. Same residual, same estimate, opposite outcome, decided
by whether the fault is an *input* the plant still responds to. That is the precise form
of the "necessary, not sufficient" statement in the lock paragraph, now with its positive
half. The +10 cell (clip 0.50) follows.

**Elbow friction +10 (clip 0.50): not a claim.** `11/20 → 14/20`, 6 fixed, **3 broken**,
`p = 0.51`. The milder friction leaves the frozen policy at half and the law both helps
and hurts. The estimate spread (sd 0.09 on x and z) is the torque cell's again, and the
mechanism is specific to a loss of effectiveness from below: friction removes motion only
*while the joint moves*, and an additive estimate that was right during a reach is a
push in the wrong direction the moment the arm pauses to grasp. At +20 the deadband is
present at every speed and the constant estimate is right more often than wrong (0 → 8,
no regression); at +10 it is right about as often as it is wrong. This is where a
velocity-dependent (multiplicative, §22) model of the fault would belong, and it is not run.

### 29.4 Summary of the joint-level cells

| fault (elbow, below the controller) | frozen | corrected | fixed / broken | p | reading |
|---|---|---|---|---|---|
| torque bias 5 N·m, n = 40 | 20/40 | **32/40** | 15 / 3 | **0.0075** | repaired; Jacobian makes the estimate state-dependent |
| friction +4 | 20/20 | 20/20 | 0 / 0 | ceiling | policy tolerates it; law harmless |
| friction +10 (clip 0.5) | 11/20 | 14/20 | 6 / 3 | 0.51 | motion-dependent fault vs constant estimate |
| friction +20 | 0/20 | **8/20** | 8 / 0 | **0.0078** | repaired from zero |
| lock ±0.05 rad | 0/20 | 0/20 | 0 / 0 | — | identified, not an input: not repairable |

Two of five cells repaired and claimed; one ceiling; one boundary of kind (rank); one
boundary of degree (a motion-dependent fault under a constant estimate). Every one is
identified on the translation channels by the Cartesian residual with the healthy
calibration. The draft's limitation ("that class appears only in the hardware protocol")
is closed and replaced by these five rows.

## 31. Recoverability-map cells at n = 40, and the constants ablation (added 2026-09-06)

### 31.1 Rotation 0.05, the last map cell on the exact-test floor

§19 had rotation 0.05 at `13/20 → 18/20, p = 0.063` — five one-way pairs, the floor.

| n | frozen | corrected | fixed | broken | exact McNemar |
|---|---|---|---|---|---|
| 20 | 13/20 = 65% | 18/20 = 90% | 5 | 0 | 0.063 (floor) |
| **40** | 27/40 = 68% | **39/40 = 98%** | **12** | 0 | **0.00049** |

Resolved, zero regressions; the effect (+25 → +30 points) did not move. Every cell of the
map that showed an effect at n = 20 is now individually significant except uniform 0.15,
which is the superposition boundary (§19.3) and is not expected to be.

### 31.2 Translation 0.10 at n = 40

§19: `13/20 → 19/20, p = 0.031`.

| n | frozen | corrected | fixed | broken | exact McNemar |
|---|---|---|---|---|---|
| 20 | 13/20 = 65% | 19/20 = 95% | 6 | 0 | 0.031 |
| **40** | 30/40 = 75% | **38/40 = 95%** | 9 | 1 | **0.021** |

Significant at both n, and the cell is a near-ceiling: the frozen rate rose from 65 % to
75 % on the second twenty initial states (the ±11-point noise of §10) while the corrected
arm stayed at 95 %, so the discordant count barely grew. One regression appears at n = 40,
the second on `libero_spatial` in the whole record. With §31.1, every map cell that showed
an effect at n = 20 is individually significant, and the summary of §28.2 stands.

### 31.3 Ablation of the law's constants and of the calibration size

Headline cell (π0.5, `libero_spatial`, uniform +0.05, rotation-only correction), n = 20
paired per row, one constant changed per row. Reference constants: γ = 0.08, ρ = 0.15,
deadzone 0.008, plant fitted on three healthy episodes (285 steps). The last column is the
rotation estimate, last-50-step mean, against a true 0.05 on each axis.

| setting | frozen | corrected | fixed | broken | exact McNemar | r̂x, r̂y, r̂z |
|---|---|---|---|---|---|---|
| **reference** | 8/20 | **18/20** | 10 | 0 | 0.0020 | 0.044, 0.020, 0.044 |
| γ 0.02 (¼) | 8/20 | 17/20 | 9 | 0 | 0.0039 | 0.036, 0.016, 0.036 |
| γ 0.32 (4×) | 7/20 | 18/20 | 11 | 0 | 0.00098 | 0.044, 0.019, 0.044 |
| ρ 0.05 (⅓) | 11/20 | 14/20 | 5 | **2** | **0.45** | 0.027, 0.011, 0.027 |
| ρ 0.50 (3⅓×) | 8/20 | **20/20** | 12 | 0 | 0.00049 | 0.048, 0.022, 0.049 |
| deadzone 0 | 8/20 | 19/20 | 11 | 0 | 0.00098 | 0.044, 0.018, 0.044 |
| deadzone 0.03 (≈ residual scale) | 8/20 | 17/20 | 9 | 0 | 0.0039 | 0.031, 0.016, 0.032 |
| plant from **1** healthy episode (75 steps) | 10/20 | 17/20 | 8 | 1 | 0.039 | 0.027, 0.008, 0.044 |
| plant from 2 healthy episodes | 8/20 | 17/20 | 10 | 1 | 0.012 | 0.045, 0.021, 0.043 |

**Reading.** The gain is flat across a 16× range: γ only sets how many steps the transient
takes (a quarter gain still converges well inside a 220-step episode), and the estimate at
the end is the same. The deadzone is flat until it reaches the residual's own scale, where it
starts eating signal (estimate 0.031 against 0.044) and costs one episode. The normaliser is
the one constant that decides the outcome, in exactly the direction §12's derivation
says: the legacy law's fixed point is f/(1 + |r|²/ρ²), so ρ = 0.05 against a residual of
0.034 puts the estimate at 0.027 — 55 % of the fault — and the cell drops to a null with two
regressions; ρ = 0.50 removes the bias (0.048) and the cell reaches 20/20. **This is the
"measure before setting constants" rule with its price tag:** the only setting that fails is
the one chosen below the measured residual scale. On calibration size, **one healthy episode
is enough for the plant** — 17/20 from 75 steps of healthy motion, with r̂x under-identified
at 0.027 and the result carried by the other two axes; two episodes recover the full
estimate. The plant needs seconds of healthy data, not a dataset.

### 30.3 Translation fault +0.15, translation-only correction, `libero_spatial`, n = 20

| arm | success | fixed | broken | exact McNemar |
|---|---|---|---|---|
| frozen, faulted | 2/20 | | | |
| **corrected** | **15/20 = 75%** | **13** | **0** | **2.4×10⁻⁴** |

Second fault family on the third backbone, same calibration. The estimate reads
`x 0.19, y 0.07, z 0.21` (last-50 mean, sd ≤ 0.05, two clip hits in 60 channel-episodes)
against a true 0.15: x and z carry the translation phantom (§29.2 measured it at +0.02 to
+0.03 on π0.5, and GR00T's healthy control showed the same) on top of the fault, and y is
under-identified — the same per-channel pattern π0.5 (§19) and OFT (§25) showed. The
gain-fault cell follows.

### 30.4 Gain fault 0.20 on translation, multiplicative law, `libero_spatial`, n = 20

| arm | success | fixed | broken | exact McNemar | β̂ (true −0.80) |
|---|---|---|---|---|---|
| frozen, faulted | 0/20 | | | | |
| **corrected** (FIR regressor, g-min 0.12, clip 0.95) | **19/20 = 95%** | **19** | **0** | **3.8×10⁻⁶** | **−0.816, −0.804, −0.813** |

The multiplicative law identifies the loss of effectiveness to within 2 % on all three
translation axes (clip 0.95, so the value is an estimate, not the bound) and takes the third
backbone from zero to 19/20, the same cell OFT reached 17/20 on (§25) and π0.5 17/20 (§22).

### 30.5 The third backbone, complete

| GR00T N1.7 cell (`libero_spatial`, n = 20) | frozen | corrected | fixed / broken | p |
|---|---|---|---|---|
| healthy control (law running) | 18/20 | 18/20 | 1 / 1 | null |
| rotation +0.10, rotation-only correction | 0/20 | 14/20 | 14 / 0 | 1.2×10⁻⁴ |
| translation +0.15, translation-only correction | 2/20 | 15/20 | 13 / 0 | 2.4×10⁻⁴ |
| gain 0.20, multiplicative law | 0/20 | 19/20 | 19 / 0 | 3.8×10⁻⁶ |

Three fault families, 46 repaired, 0 broken, on a policy from a third developer with a
third architecture, using the plant model and M identified on π0.5 and never retuned. Every
number in OFT's row (§25) now has its GR00T counterpart.

### 30.6 Videos

`results/phase05/adaptive_vs_frozen_groot.mp4` (rotation +0.10, rotation-only correction)
and `adaptive_vs_frozen_groot_translation.mp4` (translation +0.15, translation-only
correction), rendered under the benchmark condition (§28.3) with `--only-repaired
--max-clips 4` from the pairs the stored runs fixed, distinct tasks first, every final
frame checked. Rotation: 4 clips from 5 candidates (one corrected re-run failed), estimates
0.06–0.09 against 0.10, the frozen gripper approaching tilted in every clip. Translation:
4 clips from 4 candidates, estimates x 0.13–0.22, y 0.08–0.10, z 0.16–0.27 against 0.15
(x and z carry the translation phantom on top of the fault), the frozen gripper landing
past the bowl in every clip. 48 s each.

## 32. A humanoid: GR00T N1.5 on the Fourier GR1 (added 2026-09-07)

**Why N1.5 and not N1.7.** N1.7's only humanoid simulator embodiment (RoboCasa GR1
tabletop, 24 pick-and-place tasks in MuJoCo) is finetune-only and NVIDIA released no
checkpoint for it; the X2 has no GR00T embodiment at all, and fine-tuning is the compute we
do not have. The N1.5 base model lists GR1 as a *pretrained* embodiment ("humanoid robots with
dexterous hands using absolute joint space control"), usable zero-shot. So: a second clone at
`n1.5-release`, python 3.10, patched to SDPA attention for the Turing GPU (three files;
`SETUP.md`), `groot15_server.py` behind the same websocket protocol, and `gr1_adapt.py`, the
ALOHA joint-space script with a GR1 adapter. N1.5 runs at 0.4 s per 16-step chunk here, ten
times faster than N1.7.

**The robot and its interface.** Fourier GR1 upper body: two 7-joint arms, two 6-joint
dexterous hands, a 3-joint waist, 29 absolute joint targets at 20 Hz (`control_delta =
False` in the benchmark's own wrapper), one egocentric camera rendered at 1280×800 and
crop-padded to 256×256. Task: `PnPCanToDrawerClose` — pick up the can, place it in the
drawer, close the drawer; 720-step cap as in NVIDIA's evaluation; success is the
benchmark's own check. Both arms move on every episode (the left arm is the one that
handles the can; the right arm the drawer).

### 32.1 Measure before setting constants

| quantity | value |
|---|---|
| healthy success, N1.5 zero-shot, 5 episodes | 3/5 (NVIDIA reports 70 % for the finetuned N1.7 on this task) |
| plant FIR R² on the 14 arm joints | 0.94–1.00 (hands and waist not modelled; not corrected) |
| healthy residual norm on the arm joints | median 0.064 rad, 90th pct 0.18, max 0.58 (contact) |
| servo tracking of a held +0.05 rad target offset | ratio 1.00 (j8, j10) |
| M, right arm, open-loop replay of 120 healthy commands | block = I exactly, cond 1.0 |
| M, left arm, same replay | block diag 0.26–0.91, off-diag to 1.2: **contaminated** — the left arm is in contact during the replay window, and the probe measures the divergence of a contact trajectory, not the servo |
| M, left arm, direct step response from a held pose (30 settle + 30 probe) | diag 0.984–0.991, off-diag ≤ 0.002, cond 1.01 — used |
| frozen under +0.05 rad on the right arm (5 eps) | 3/5, not damaging |
| frozen under +0.15 rad on the right arm | 2/5 |
| **frozen under +0.15 rad on the left arm** | **0/5**: the grasping arm; this is the cell |

Constants from the table: deadzone 0.03 (half the median residual), normaliser 0.35 (twice
the 90th percentile), clip 0.30 (twice the fault), γ 0.08. The replay-based M probe of
`aloha_adapt.py` is *wrong on an arm that contacts objects during the replay*, which ALOHA's
transfer-cube arm did not in its first 120 steps; the direct step-response probe is the
general method and is what `openloop_left_direct.json` records.

### 32.2 Run 1: 0/10 → 0/10, and why — the normaliser was looking at the hands

Left arm +0.15 rad, correct the left arm, continuous adaptation, constants from §32.1
(γ 0.08, deadzone 0.03, ρ 0.35, clip 0.30), 10 paired episodes: frozen **0/10**, corrected
**0/10**. The estimate rose to 0.059 by step 50 and then *decayed* to 0.02 — 13 % of the
fault — on every episode.

The constants were set from the residual on the arm joints (median 0.064), but the law's
normaliser took the norm of the residual on **all 29 joints**, and the twelve hand joints,
commanded open/closed and fitted with R² ≈ 0, carry a residual three times the arm's
(median 0.17, 90th pct 0.61; the 29-joint norm is median 0.19, 90th pct 0.63). Every update
on the arm was attenuated four to ten times by a signal from joints the law was not
estimating, and the legacy fixed point f/(1 + |r|²/ρ²) landed near zero. ALOHA did not show
this because its two gripper joints are a small share of its fourteen.

Two changes, both in `gr1_adapt.py`: the normaliser and deadzone act on the residual of the
joints being corrected, and `--law innov` (the innovation form of §12, unbiased fixed point)
is available. Run 2 uses both with ρ = 0.30 and deadzone 0.03 from the *left-arm* residual
(median 0.054, 90th pct 0.142). Measure-before-setting has a corollary: **measure the
quantity the law actually consumes**, not a neighbour of it.

### 32.3 Run 2: identified to 85–95 %, still 0/10 — the ALOHA pattern

Same cell, innovation law, normaliser on the left-arm residual: frozen **0/10**, corrected
**0/10**. The identification now works: the last-50-step estimate on the seven left-arm
joints is 0.13–0.16 on eight of ten episodes (85–105 % of the 0.15 fault; two episodes at
0.085), and it is at 0.12 by step 50 and 0.15 by step 100. The correction is applied and the
task still fails. The within-episode wander of the applied correction is **0.056 rad median**
(sd over steps after step 50, mean over joints), with contact spikes driving single joints to
the ±0.30 clip on two episodes. That is the ALOHA signature of §27.12: a continuously
updating correction on a task whose margin is smaller than the estimator's motion.
Proposition 2 says the next two runs, not a retune: the oracle (exact −0.15 applied from step
0, no estimator), which measures whether the task is recoverable at all at this fault and
sets the ceiling; and identify-then-hold (adapt on episode 0, hold on episodes 1–9), the
deployable scheme that gave ALOHA its result.

### 32.4 Oracle 1/10, identify-then-hold 0/10: the ceiling is the question

| scheme (left arm +0.15, seeds 100–109) | success |
|---|---|
| frozen, faulted | 0/10 |
| continuous adaptation (run 2) | 0/10 |
| **oracle: exact −0.15 on the left arm from step 0, no estimator** | **1/10** |
| identify episode 0, hold 1–9 (held estimate 0.086–0.204 per joint, 57–136 %) | 0/10 |
| healthy, seeds 100–104 (from the log run) | 3/5 |

The oracle's executed commands are identical to the healthy policy's — a_cmd − 0.15 + 0.15 —
and it succeeded on the one seed of the first five the healthy run failed and failed on the
three it passed. So on this task the policy's own sampling noise (flow-matching, 16-step
chunks, 720 steps) is at least as large as any effect being measured, and every number above
is inside it. Two consequences before any more repair runs: the healthy rate on the same
ten seeds is being measured (the control §14.5 requires), and the held estimate now has a
`--hold-stat mean50` option, because the final value of a contact-rich identification
episode is one contact spike away from the fault (0.086 on joint 6, 0.204 on joint 5) and
the record's statistic has always been the last-50-step mean. If the healthy control comes
back low, the task, not the law, is the problem, and a screen of five other GR1 tasks for a
higher healthy rate is queued behind it.

### 32.5 The healthy control came back 0/10, and the scenes are not the same across processes

Healthy, no fault, seeds 100–109, a fresh process: **0/10** — against 3/5 on seeds 100–104 in
the first log run. Not sampling noise alone: on the same seed the two processes start from
initial joint states 0.026 rad apart and the first commands differ by 0.12 rad, so the
*scenes* differ. Within a process, `reset(seed=100)` is exactly repeatable (q₀ identical to
four decimals, three resets), so the two arms of a `run` — which share a process — are
paired on identical scenes, but any number from a separate process (the log run, the
oracle, identify-then-hold, the healthy control) is on a different draw of the scene. The
wrapper seeds `np.random` only; whatever robocasa uses for the rest of the scene is not
under that seed. Consequence: the healthy ceiling must be measured **inside the same
process** as the arms it is compared with. `gr1_adapt.py run --with-healthy` now runs a
third arm, healthy, on the same seeds before the two faulted ones. Nothing in §32.2–32.4
is retracted, but none of those rows can be compared to the 3/5 either.

### 32.6 Task screen and the cell that has a ceiling: plate-to-plate

Healthy N1.5 zero-shot, six episodes each, one process per task:

| task | healthy | steps to success |
|---|---|---|
| **PosttrainPnPNovelFromPlateToPlate** | **5/6** | 160–180 |
| PosttrainPnPNovelFromTrayToPlate | 3/6 | |
| PnPBottleToCabinetClose | 1/6 | |
| PnPCupToDrawerClose | 0/6 | |
| PnPWineToCabinetClose | 0/6 | |
| PnPCanToDrawerClose (§32.1–32.5) | 3/5 then 0/10 | 250–560 |

N1.5 zero-shot is weak on the "close the drawer/cabinet" tasks and strong on the short
pick-and-place ones; plate-to-plate is a **right-arm** task (the left arm moves under 0.5
rad, the right 1–2 rad) and finishes in 170 steps, which also makes every cell four times
cheaper. Plant on its six healthy episodes: R² 0.998–0.999 on the right arm; healthy
right-arm residual norm median 0.027, 90th pct 0.054 → deadzone 0.013, normaliser 0.11.
M: direct step response, right-arm block 0.990–0.997, off-diagonal ≤ 0.005 (the left arm is
in contact in this scene's held pose and its block is taken from the can-drawer probe).

Damage, frozen, six episodes: left arm +0.10 → 3/6, +0.20 → 1/6 (the idle arm still
matters: it is in the camera view and the policy reads it); **right arm +0.10 → 0/6**. The
cell is right arm +0.10 rad (5.7°), corrected on the right arm, clip 0.20, two schemes
(continuous; identify-then-hold with the last-50 mean), each with an in-process healthy
arm on the same ten seeds.

### 32.7 Plate-to-plate, right arm +0.10 rad, three arms in one process, n = 10

| arm | success | per episode (seeds 100–109) |
|---|---|---|
| healthy (no fault) | **6/10** | 0 0 1 1 0 0 1 1 1 1 |
| frozen, faulted | **0/10** | 0 0 0 0 0 0 0 0 0 0 |
| **corrected, continuous adaptation** | **3/10** | 0 0 1 0 1 1 0 0 0 0 |

**The law repairs a humanoid.** From a frozen zero to half the healthy rate, 3 fixed, 0
broken — at n = 10 that is p = 0.25 on the exact test, so it is a result to extend, not yet
to claim; seeds 110–129 are queued for both schemes. What the trajectories say: the three
successes are the three episodes whose estimate reached 0.098–0.099 (98 % of the fault) with
a within-episode wander of 0.013–0.017 rad; the failures split into episodes where the
estimate stayed at 0.035–0.05 with wander 0.04–0.06 (contact-driven, under-identified) and
episodes with a good estimate that failed anyway — as the healthy arm failed four of ten.
Two of the three repaired episodes (seeds 104, 105) are ones the *healthy* policy failed,
which is the sampling noise of §32.4 again and the reason the healthy arm has to sit in
the same table. The wrist joint (j13) touched the 0.20 clip on three episodes.

### 32.8 Identify-then-hold on the humanoid: 1/10 → 8/10, above the healthy arm

Same cell, same ten seeds, same process: adapt on episode 0, hold the last-50-step mean
(`0.090–0.112` rad on the seven right-arm joints, 90–112 % of the 0.10 fault) on episodes
1–9.

| arm | success | per episode |
|---|---|---|
| healthy | 6/10 | 0 1 0 1 1 1 1 0 1 0 |
| frozen, faulted | 1/10 | 0 0 0 0 0 0 0 0 0 1 |
| **identify episode 0, then hold** | **8/10** | 1 1 1 1 0 1 1 1 0 1 |

The identification episode itself succeeded (the estimate is at 0.1 within 100 steps of a
170-step task), and the held correction takes the humanoid from one in ten to eight in ten,
two above its own healthy arm on these seeds — inside noise, and the reading is "to the
healthy rate". Against continuous adaptation on the same seeds (§32.7: 3/10, wander
0.013–0.058 rad) this is Proposition 2 on a third manipulator: the estimate is the same
number either way; what decides the task is whether it moves while the hand is at the
plate. Exact McNemar against frozen: **7 fixed, 0 broken, p = 0.016**; against healthy:
4 up, 2 down, p = 0.69 (indistinguishable). Seeds 110–129 are running for both schemes.

### 32.9 The humanoid result at n = 30

Seeds 100–129, three arms per process, right arm +0.10 rad, right arm corrected, plant and
M from §32.6, constants from the measured residual. Exact McNemar on the paired outcomes.

| scheme | healthy | frozen | **corrected** | fixed / broken vs frozen | p vs frozen | vs healthy |
|---|---|---|---|---|---|---|
| continuous adaptation | 23/30 | 1/30 | 8/30 = 27% | 8 / 1 | 0.039 | −18 / +3, p = 0.0015 (below) |
| **identify episode 0, then hold** | 21/30 | 2/30 | **21/30 = 70%** | **19 / 0** | **3.8×10⁻⁶** | +8 / −8, **p = 1** (at the ceiling) |

**A frozen humanoid VLA is repaired from 7 % to its healthy rate by six numbers held after
one identification episode.** The held estimate is 0.083–0.118 rad per joint on the second
batch (83–118 %), 0.090–0.112 on the first. Continuous adaptation on the same seeds is
significant but stays at a third of the ceiling and is *significantly below* healthy: the
same estimate applied while it moves. This is the third manipulator on which Proposition 2
decides the scheme — LIBERO (large margin: continuous works), ALOHA (sub-centimetre margin:
hold), GR1 (a dexterous-hand grasp on a plate: hold) — and the first humanoid. The clip
warnings in the log are the hand joints, which are estimated as a by-product, never
corrected, and pinned at the bound because their plant is R² ≈ 0; they are noise in the
report and nothing in the result.

Cost of the whole humanoid chain from a cold start, GPU time: six healthy episodes for the
plant (≈ 2 min), a direct M probe (≈ 1 min, no policy), 6+6 damage probes, and 6 × 30
paired episodes — under three hours, with one 3B model that was never fine-tuned.

### 32.10 Correction: the humanoid arms are NOT paired; unpaired statistics replace §32.7–32.9's

§32.5 said scenes repeat per seed within a process. That test compared robot joint state and
the *fixed furniture* across resets and both matched; it did not look at the manipulated
objects. A second test (plate-to-plate; reset 100, reset 101, reset 100 again) shows 45 of 86
bodies moved between the two seed-100 resets, the manipulated object changed (a squash, then
a bell pepper; 0.19 m apart), and so did the task language. `np.random.seed` in the wrapper
does not govern robocasa's object sampling. **Every arm in §32.7–32.9 saw its own draw of
the scene**, so the McNemar "fixed / broken" counts there are not pair counts and the
paired p-values are not valid. The three arms are independent samples of the same scene
distribution and the right test is unpaired (Fisher exact, two-sided):

| comparison, n = 30 per arm | rates | Fisher p |
|---|---|---|
| **identify-then-hold vs frozen** | **21/30 vs 2/30** | **5.5×10⁻⁷** |
| identify-then-hold vs healthy | 21/30 vs 21/30 | 1 |
| continuous adaptation vs its own frozen control | 8/30 vs 1/30 | 0.02569 |
| continuous adaptation vs the held scheme's frozen control | 8/30 vs 2/30 | 0.07972 (cross-scheme comparison) |
| continuous adaptation vs healthy | 8/30 vs 23/30 | 2.3×10⁻⁴ |

The original version of this paragraph incorrectly used the held scheme's frozen
2/30 for the continuous comparison and said its significance disappeared.
Continuous adaptation improves over its own frozen 1/30 (p = 0.02569), while
remaining below its healthy control. Held correction reaches the same observed
21/30 as its healthy control; that unresolved difference does not establish
equivalence. The humanoid table uses unpaired Fisher tests, without fixed/broken
columns. Frozen and healthy are being run for 30 more episodes each on fresh
seeds to tighten both ends.

**A second error caught in the same hour.** The first humanoid video was rendered on the
can-to-drawer task (the client's default `--task`, not passed) with the plate-to-plate plant,
M and held estimate; its frozen arm succeeded 4/8 because that task is barely damaged by a
right-arm offset. The render is discarded and the plate-to-plate video is queued. Nothing
from it is reported anywhere.

### 32.11 Frozen and healthy on thirty fresh seeds; the humanoid table, final

The new standalone files report frozen under the right-arm +0.10 rad offset
**1/30** and healthy **20/30**, documented as seeds 130–159. The episode-list
files contain outcomes and trajectories but no embedded seed or run arguments;
the seed assignment is documented here and in their filenames.

The expanded controls pool the original continuous cohort (frozen 1/30, healthy
23/30), original held cohort (2/30, 21/30), and these standalone controls (1/30,
20/30): frozen **4/90 = 4 %**, healthy **64/90 = 71 %**. The original cohorts
combine `p2p_right010_cont.json`/`p2p_right010_cont_s110.json` and
`p2p_right010_hold.json`/`p2p_right010_hold_s110.json`; the new files are
`p2p_frozen_right010_s130.json` and `p2p_healthy_s130.json`, all under `results/gr1`.
The separate null-study healthy 18/30 below is not included. This pooling follows
the observed corrected results; it is a subsequent unpaired analysis, not a new
preregistered confirmation of the unchanged corrected cohorts.

| arm | rate | Fisher p vs frozen (4/90) | Fisher p vs healthy (64/90) |
|---|---|---|---|
| frozen, right arm +0.10 rad | 4/90 | — | — |
| healthy | 64/90 | — | — |
| **identify episode 0, then hold** | **21/30 = 70 %** | **8.9×10⁻¹³** | 1.0 |
| continuous adaptation | 8/30 = 27 % | 0.0016 | 2.8×10⁻⁵ (below) |

These pooled controls give continuous adaptation 27 % against frozen 4 %
(Fisher p = 0.00159), while held correction reaches 70 % against healthy 71 %.
The latter difference remains unresolved; this does not establish equivalence.
The original same-process continuous comparison already favored adaptation:
8/30 versus its own frozen 1/30 gives p = 0.02569. Every comparison here is
unpaired. The pooled healthy comparison uses p = 2.7675×10⁻⁵; the earlier
2.3×10⁻⁴ belonged to 8/30 versus the original healthy 23/30 and was stale after
changing the table's denominator.

### 32.12 Video

`results/phase05/adaptive_vs_frozen_gr1.mp4` (148 s): plate-to-plate, right arm +0.10 rad,
the corrected panel applying the held estimate (`0.103, 0.102, 0.102, 0.101, 0.112, 0.090,
0.090` rad on the seven right-arm joints, identified in one episode) from step 0. Four of
six candidate seeds kept (`--only-repaired`): the frozen panel times out at 720 steps in
every clip — reaching past the plate, or turning away from it — and the corrected panel
finishes in 166–182 steps. First render (2026-09-07): the two panels of a clip showed
different objects because the scene was redrawn per reset. Re-rendered 2026-09-08 with the
env generator reseeded (§32.15): **both panels now show the same scene, object and task
language** (pear, squash, ...), four clips from six candidate seeds, the frozen panel
timing out at 720 steps with the hand beside the plate and the corrected panel finishing
in 166–211 steps. The file is re-encoded with the letterbox of the 256-pixel egocentric render
cropped out (header bars kept, same frames, CRF 26): 4 MB instead of 36.

### 32.13 The null: the law on a healthy humanoid

Seeds 160–189, no fault, identify-then-hold on the right arm with the same constants: the
healthy arm in the same process **18/30**, the law running on the healthy arm **19/30**
(Fisher p = 1.0; against the pooled healthy 64/90, p = 0.5). The held phantom on the seven
right-arm joints is 0.0005–0.0055 rad on six of them and **0.012 rad on the wrist (joint
13)** — 12 % of the 0.10 fault on that one joint, under 6 % elsewhere — and applying it to a
healthy humanoid gives a numerically similar success rate, with the difference
unresolved at this sample size. (The first version of this paragraph said "at most 0.004
rad"; that was written before the number was read and is corrected here.) The clip warning is joint 24, a left-hand joint that is estimated as a
by-product and never corrected. With this the humanoid cell has every row the protocol of
§14.5 requires: floor (4/90), repair (21/30), ceiling (64/90), and a null for the law on
healthy data (19/30 against 18/30).

### 32.14 Not pursued: a below-the-controller fault on the humanoid

A joint-torque bias (`qfrc_applied`) on a right-arm joint of the GR1, as in §29 on the
Panda, was probed from a held pose: 30 settle steps, 40 steps with the bias, against a
zero-bias run of the same length. The zero-bias run itself is not repeatable across two
resets of the same seed (max 0.18 rad difference on the right arm over 70 held steps), because
the held arm drifts into whatever the redrawn scene puts under it, and the torque responses
(0.1–0.2 rad, spread over all seven joints, non-monotonic in the torque) are inside that
drift. A clean probe would need a free-space pose away from the table and a fixed scene;
neither is available from the benchmark wrapper without modifying it. Left here as a
measured dead end; the humanoid section stands on the action-interface offset.

Merge audit (2026-09-08): the numeric torque-repeatability/contact account in
§32.14 is an exploratory report; these incoming commits do not include its raw
probe trace or script. It is not counted as independently verified GR1 torque
repair evidence in the all-joint follow-up.

### 32.15 Pairing restored: the scene comes from the env's own generator

The scene randomness of §32.10 is `env.rng`, a numpy `Generator` the tabletop environment
creates once and advances at every `_load_model`; the wrapper's `np.random.seed` never
touches it. Reseeding it before each reset (`env.unwrapped.env.rng =
np.random.default_rng(seed)`, plus `random.seed`) makes `reset(seed)` repeat exactly: 0 of
86 bodies move between two seed-100 resets, the object and the task language are the same,
and seed 101 still differs in 45 bodies. `gr1_adapt.GR1.reset` now does this, so every
humanoid run from here is **paired** in the LIBERO sense and the video's two panels show the
same task and object. §32.10–32.13 stand as stated (unpaired); the plate-to-plate cell is
being rerun paired (three arms, seeds 100–129) so the humanoid table can carry McNemar
counts like every other table, and the tray-to-plate chain runs paired from the start.


> **Audit-branch note kept from the 2026-09-08 merge on `review/dual-track-audit`** (written before the paired GR1 result files existed; the results below supersede its last sentence):
>
> and seed 101 still differs in 45 bodies. `gr1_adapt.GR1.reset` now does this, so new runs
> can share the initial task and scene across comparison arms. The video's
> two panels now show the same task and object. This change does not retroactively pair
> historical results; new benchmark pairing still requires matching run provenance and
> verification of the complete reset state. §32.10–32.13 stand as stated (unpaired).
> The incoming main-branch notes report a plate-to-plate rerun in progress (three arms,
> seeds 100–129) and a tray-to-plate chain using the reset change. These commits do not
> include their benchmark result files; new McNemar counts are not available in this merge.
>
> Merge verification (2026-09-08): the reset change passes a nested-wrapper mock check
> for repeated and distinct seeds, and both changed Python files compile. This audit did
> not rerun the actual simulator or independently reproduce the reported 86-body probe.
> A reset-only comparison of task, object poses, full physics and controller state across
> repeated seeds and a fresh instance remains appropriate before treating new result files
> as paired. No new paired GR1 benchmark counts are included in the manuscript.

### 32.16 Plate-to-plate, PAIRED (identical scenes), identify-then-hold, n = 30

| arm | success | vs frozen | vs healthy |
|---|---|---|---|
| healthy | 22/30 | | |
| frozen, right arm +0.10 rad | 2/30 | | |
| **identify episode 0, then hold** | **15/30 = 50 %** | **13 fixed / 0 broken, p = 2.4×10⁻⁴** | −13 / +6, p = 0.17 |

Paired at last, and the number is lower than the unpaired 21/30: not because of pairing but
because of the **one identification episode**. Its held estimate this time is `0.117, 0.092,
0.096, 0.115, 0.095, 0.146, −0.096` — joint 13 (wrist pitch) came out with the **wrong sign**
and joint 12 at 146 %; the earlier run's held vector was 0.090–0.112 on all seven. One
episode of identification on a contact-rich task can be one contact spike away from the
fault on a joint, and the hold scheme then carries that error for 29 episodes. The repair
is still 13 fixed and 0 broken against frozen, but it sits below the healthy arm (p = 0.17)
where the earlier draw sat on it. Identifying over three episodes and holding their mean is
the obvious robustness step (§27 identified over one because ALOHA's estimate was
uniform to 2 %) and is queued, paired, on the same seeds.

## 33. Review pass on the draft (2026-09-08)

Eleven wording corrections (the manipulators' own calibration, the pooled 4 % humanoid floor,
the humanoid's unpaired status in the contributions, "a few tens of steps" for convergence,
the GR1 in the setting and method, the corrected-channel norm and the innovation law on the
humanoid, six measured failures of the constants rule). Then every success-rate cell in the
draft's tables — 38 cells across the headline, map, time-varying, gain, OFT, GR00T,
joint-level, ALOHA, ablation and n = 40 tables — was recomputed from its stored
per-episode file by script: **38 of 38 match** (the one apparent mismatch was the script
reading `tv_intermittent.json`, the underpowered §17 cell, instead of `tv2_intermit.json`,
the §20 cell the draft reports; the draft is right). The humanoid pooled floor and ceiling
(4/90, 64/90) also recompute from their six files.

### 32.17 Tray-to-plate, paired, identify-then-hold: 6/30, and the identification episode is the weak link

| arm (tray-to-plate, right arm +0.10 rad, seeds 100–129, identical scenes) | success |
|---|---|
| healthy | 16/30 |
| frozen | 2/30 |
| identify episode 0, then hold | 6/30 — 5 fixed / 1 broken, p = 0.22; below healthy, p = 0.006 |

Not a repair. The held vector is `0.085, 0.081, 0.097, 0.055, 0.083, −0.107, 0.179`: the two
wrist joints are wrong by 0.2 rad in opposite directions, and the identification episode
itself failed at 720 steps — it spent most of the episode in contact, and the last-50-step
mean is a contact standoff, not the fault. §32.16 was the same failure on a milder draw.
The law identifies the fault well while the arm reaches (the trajectories are at 90–110 %
by step 100 on the shoulder joints) and badly once the hand is on the plate; a single
episode's ending is the wrong place to read the estimate. Two fixes, both queued paired on
both tasks: read the **median over the episode after the transient** (`--hold-stat
median`, added), and identify over three episodes before holding.

**Which statistic of the identification episode to hold** (offline, on the four stored
identification episodes, max |error| over the seven joints against 0.10 rad):

| episode | last-50 mean | median after 50 | mean 50–200 | median 50–200 |
|---|---|---|---|---|
| plate-to-plate, good draw (held → 21/30) | 0.012 | 0.008 | 0.009 | 0.008 |
| plate-to-plate, paired draw (held → 15/30) | 0.196 | 0.165 | 0.131 | 0.165 |
| tray-to-plate, paired (held → 6/30) | 0.207 | 0.123 | 0.064 | 0.055 |
| plate-to-plate continuous, ep 0 (720 steps) | 0.299 | 0.116 | 0.015 | 0.009 |

The reach window (steps 50–200) is where the law reads the fault; the episode's end is
where the hand is on the plate. On three of four episodes the window median is within
0.06 rad on every joint where the last-50 mean is off by 0.2–0.3; the fourth (the paired
plate-to-plate draw) had the wrist wrong from the start of that episode, which only more
identification episodes can fix. `--hold-stat window` now holds the median over steps
50–200, and with `--identify-episodes k` the median of the k per-episode windows. Both
tasks are rerun paired with k = 3 and the window statistic.

### 32.18 Plate-to-plate, paired, identify over three episodes, hold the window median

| arm (seeds 100–129, identical scenes) | success | vs frozen | vs healthy |
|---|---|---|---|
| healthy | 22/30 | | |
| frozen, right arm +0.10 rad | 1/30 | | |
| **identify episodes 0–2 (median of the three reach windows), hold 3–29** | **19/30 = 63 %** | **18 fixed / 0 broken, p = 7.6×10⁻⁶** | +4 / −7, p = 0.55 |

Held vector `0.104, 0.097, 0.098, 0.099, 0.100, 0.093, 0.096` — **93–104 % of the fault on
every joint**, where the single-episode draw of §32.16 had a wrong-signed wrist. On the 27
held episodes the corrected arm scores 17/27 against the healthy arm's 20/27 on the same
scenes; two of the three identification episodes succeeded as well. This is the humanoid
result the paper carries: paired, McNemar, at the healthy rate, with the identification
cost stated as three episodes rather than one. The tray-to-plate rerun with the same scheme
is running.

### 32.19 Tray-to-plate, paired, identify over three episodes: at the healthy rate too

| arm (seeds 100–129, identical scenes) | success | vs frozen | vs healthy |
|---|---|---|---|
| healthy | 17/30 | | |
| frozen, right arm +0.10 rad | 1/30 | | |
| **identify episodes 0–2 (window median), hold 3–29** | **13/30 = 43 %** | **12 fixed / 0 broken, p = 4.9×10⁻⁴** | +6 / −10, p = 0.45 |

Held vector `0.102, 0.097, 0.102, 0.098, 0.100, 0.126, 0.073` (73–126 %); on the 27 held
episodes 13/27 against the healthy arm's 15/27. All three identification episodes failed
the task (they run with the estimate still moving) and the held correction repairs
anyway. The law on a healthy arm on this task: 16/30 against 15/30 (§ C4 of the queue log,
`t2p_null_hold_s160.json`), a null.

**The humanoid, final: two tasks, paired, at the healthy rate.**

| task | healthy | frozen | corrected (identify 3, hold) | fixed / broken | McNemar p | vs healthy |
|---|---|---|---|---|---|---|
| plate-to-plate | 22/30 | 1/30 | **19/30** | 18 / 0 | 7.6×10⁻⁶ | p = 0.55 |
| tray-to-plate | 17/30 | 1/30 | **13/30** | 12 / 0 | 4.9×10⁻⁴ | p = 0.45 |
| pooled | 39/60 | 2/60 | **32/60** | 30 / 0 | 1.9×10⁻⁹ | — |

Zero regressions in sixty paired episodes; both tasks indistinguishable from their healthy
arms; nulls on both (19/30 vs 18/30; 16/30 vs 15/30). What it cost: six healthy episodes
for the plant, a one-minute direct probe for M, and three identification episodes per
task. What it took to get right: the normaliser on the corrected joints (§32.2), the
innovation law (§32.3), the scene generator (§32.15), and reading the estimate in the reach
window over three episodes instead of at the end of one (§32.17). Each of those is a
measured failure that is now a sentence in the method.

### 32.20 Second humanoid video

`results/phase05/adaptive_vs_frozen_gr1_tray.mp4` (148 s, 5 MB): tray-to-plate, right arm
+0.10 rad, the corrected panel applying the three-episode held vector (`0.102, 0.097,
0.102, 0.098, 0.100, 0.126, 0.073`) from step 0, labelled as identified over three
episodes. Paired scenes: both panels show the same task and object (can, bell pepper,
squash, can). Four clips from seven candidates; the frozen panel times out at 720 steps in
each — in one it swings the arm up into the camera — and the corrected panel finishes in
150–177 steps. Both humanoid videos are re-encoded with the letterbox cropped.

### 32.21 Humanoid recoverability map, plate-to-plate, paired, identify 3 then hold

| right-arm offset | healthy | frozen | corrected | fixed / broken | McNemar | vs healthy | held estimate |
|---|---|---|---|---|---|---|---|
| **0.05 rad** | 19/30 | 7/30 | **24/30** | 17 / 0 | 1.5×10⁻⁵ | +7 / −2, p = 0.18 | 0.045–0.054 (90–108 %) |
| 0.10 rad (§32.18) | 22/30 | 1/30 | **19/30** | 18 / 0 | 7.6×10⁻⁶ | +4 / −7, p = 0.55 | 0.093–0.104 |
| **0.20 rad** | 25/30 | 0/30 | **21/30** | 21 / 0 | 9.5×10⁻⁷ | +2 / −6, p = 0.29 | 0.188–0.197 (94–99 %) |

At half the headline fault the frozen humanoid keeps a quarter of its successes and the
held correction takes it above its own healthy arm on these seeds (inside noise); at twice
the headline fault (11.5° on every right-arm joint) the frozen policy never succeeds and
the held correction recovers 21 of 30, at the healthy rate, with the estimate within 6 % on
every joint. **Zero regressions across the three magnitudes (56 fixed / 0 broken in 90
paired episodes)**, and no sign yet of the superposition boundary §19 found on the Panda:
the identification stays at 90–108 % from 0.05 to 0.20 rad. The clip warnings in the log
are hand joints, uncorrected.

### 32.22 Third humanoid video: the largest fault

`results/phase05/adaptive_vs_frozen_gr1_020.mp4` (148 s, 3.6 MB): plate-to-plate, right arm
**+0.20 rad** (11.5° on every joint), the corrected panel applying the three-episode held
vector (0.188–0.197) from step 0. Paired scenes, four clips from five candidates; the
frozen arm is driven so far off that the source plate leaves the camera's view, and the
corrected panel finishes in 161–184 steps.

### 32.23 Time-varying faults on the humanoid

A held estimate cannot follow a moving fault by construction, so these cells run
continuous adaptation (the scheme that reached 8/30 on a constant fault, §32.9), paired,
n = 20, right arm, amplitude 0.10 rad.

| profile | healthy | frozen | corrected (continuous) | fixed / broken | McNemar |
|---|---|---|---|---|---|
| **ramp to full over 60 steps** | 16/20 | 1/20 | **9/20** | 9 / 1 | 0.021 (vs healthy −8/+1, p = 0.039) |
| non-zero-mean sine, period 120 | 11/20 | 1/20 | 6/20 | 6 / 1 | 0.13 (vs healthy −9/+4, p = 0.27) |
| intermittent, 60 on / 60 off | 14/20 | 0/20 | 4/20 | 4 / 0 | 0.13 (vs healthy −11/+1, p = 0.006) |
| ramp, **identify 3 then hold** | 13/20 | 1/20 | **11/20** | 10 / 0 | **0.002** (vs healthy −5/+3, p = 0.73) |

The ramp is the one profile a held estimate can serve, because after its 60-step rise the
fault is constant for the rest of every episode and the three-episode window median reads
that constant: 11/20, at the healthy rate. Under continuous adaptation the ramp is repaired
significantly (9/20, p = 0.021) but stays below healthy — the estimate lags the rise by
about 30 steps and reaches 92 % by step 100, and then it keeps moving, which is §32.9's
problem again. The two profiles that never settle are the boundary: a slow oscillation
gets 6/20 and an intermittent fault 4/20, neither significant against frozen at n = 20 and
the intermittent one significantly below healthy. On the Panda the same profiles were
repaired (§20) because a Cartesian reach tolerates a correction that is wrong for a few
steps; a dexterous-hand grasp on a plate does not. The humanoid's repairability criterion
is therefore stricter than the Panda's on exactly the axis Proposition 2 names: the
correction must be stationary while the hand is at the object, and a fault that keeps
moving cannot be given a stationary correction.

### 33.1 Two consistency fixes after the humanoid work (2026-09-09)

The title and abstract say single-episode; ALOHA holds an estimate from one sacrificial
episode and the GR1 from three. The abstract's "no learning across episodes" is replaced by
the statement that on tasks whose margin is below the estimator's motion the estimate
identified in one to three episodes is held, and the first contribution says the same. The
contributions' aggregate was stale (750): a script over every stored paired result file
with both arms (59 files, controls and ablations included, probes and logs excluded)
counts **1,330 paired episodes, 544 fixed, 32 broken**; the paper now says "over 1,300
paired episodes in aggregate across cells, controls and ablations", and the abstract's
curated 690/275/7 (LIBERO and OFT, the online law) is unchanged and separately verified
(§33).

## 34. A second simulator and a fourth robot: WidowX in SimplerEnv with GR00T N1.7 (2026-09-09)

**Why.** Every result so far lives in MuJoCo (LIBERO, ALOHA, RoboCasa). SimplerEnv is the
benchmark J-PARC and most VLA papers report on, it runs in SAPIEN, and its WidowX tasks have
real-robot counterparts (Bridge V2). GR00T N1.7 ships a SimplerEnv-Bridge finetune
(`GR00T-N1.7-SimplerEnv-Bridge`, 6.5 GB), so the same backbone family serves a third robot
with no training on our side.

**Setup** (`openpi/groot_widowx_server.py`, port 8005; `openpi/widowx_adapt.py`; venv
`simpler-venv`). SAPIEN 2.2.2 headless needs `unset DISPLAY` and the NVIDIA Vulkan ICD;
SimplerEnv pulls numpy 2 which segfaults in `compute_fk` -> pinned numpy 1.26.4. Task
`widowx_spoon_on_towel`, 150-step cap (the wrapper default is 120 for Bridge; 150 gives the
policy slack), seeds 100+. Controller `PDEEPoseController use_delta=True use_target=True`:
each 7-vector action is an end-effector pose *increment* (xyz in [-1,1], rpy in [-1.571,1.571],
gripper) applied to an accumulating target. The plant output is the measured pose increment per
step, normalised per channel by the per-unit-action motion scale (`healthy_log_scale.json`:
0.93/0.94/0.77 on xyz), so the fault, the residual and f_hat all live in action units.

**Healthy log and plant.** 10 healthy episodes, 6/10 success. FIR K=6 plant R^2 on xyz
0.97/0.995/0.92.

**Sensitivity probe: the accumulating target bites.** The first M probe (offset 0.02 over
120 steps, the LIBERO recipe) saturated: 0.02 per step on an accumulating target drives the
arm to the workspace limit (0.4462) in a few dozen steps and the response goes to zero. The
probe is 0.005 over 60 steps from the held pose (`openloop_p005.json`): diag
[0.78, 1.07, 0.51, 0.99, 0.93, 1.10], cond 9.7. Z is the weakest channel (gravity and the
table).

**Residual scale before constants (the rule from Sec 22).** Healthy residual norm on xyz in
normalised units: median 0.0020, 90th pct 0.0047, max 0.023. The client defaults
(dead 0.01, norm_r 0.15) were carried over from LIBERO and sit 5x and 16x above this scale;
they were not used. Constants: dead 0.001 (half the median), norm_r 0.009 (2x the 90th
pct), clip 0.03, gamma 0.08, innovation form, xyz corrected.

**Damage probes (frozen policy, 10 eps each unless noted).** A constant offset on the
action's xyz: +0.04, +0.02, +0.01, +0.005, +0.003 -> 0/10 each (seeds 100-109, 150-step cap). On an
accumulating target even 0.003 per step (1.5x the residual median) is a 0.3 unit drift over
the episode, and the vision loop does not undo it: the frozen policy fails at every
magnitude tried. The fault for the paired cell is 0.005 on xyz (2.5x the residual median,
identifiable inside the deadzone budget), with 0.003 as the harder cell.

**First paired attempt aborted: the probe's M was wrong (seventh entry for Sec 22).**
With `openloop_p005.json` the adaptive arm's first two episodes failed with the z estimate
at the clip (0.03, six times the fault) and x barely moving. Replaying the law offline on
the frozen +0.005 log showed the residual itself is right (mean residual on x,y,z 0.0048,
0.0046, 0.0058: the fault, to the third decimal) and the probe's M is not: its z entry 0.51
inflates the z estimate 2.5x, and its rotation rows carry cross-couplings of order 1 (a
held-pose step response in SAPIEN mixes in contact and the joint limits). Under the FIR
plant an additive offset on the command appears in the residual scaled by the plant's DC
gain, so M = diag(sum of FIR taps) is derivable from the healthy log alone: DC gains
1.04/1.04/1.12/1.01/1.02/1.07 (`openloop_dc.json`). Offline on the faulted log this M
recovers 0.0047-0.0055 on x,y,z per episode (median over steps). The paired cell was
restarted with it; the aborted file is kept as `cell_tra005_probeM_aborted.json`
(healthy 11/20, frozen 0/20, adaptive 0/2 before the stop). The update is also masked to
the corrected dimensions (uncorrected entries of f_hat previously integrated with no
feedback and drifted to the clip; harmless for the correction, misleading in the printout).

### 34.1 Continuous adaptation, +0.005 on x,y,z (`cell_tra005.json`, seeds 100-119)

| arm | successes / 20 |
|---|---|
| healthy (no fault, no law) | 13 |
| frozen under the fault | 0 |
| adaptive, online from step 0 (gamma 0.08) | 6 |

Paired: adaptive-only 6, frozen-only 0, McNemar p = 0.031. Final estimate median
(0.0050, 0.0051, 0.0049) on x,y,z for a fault of 0.005, IQR <= 0.0007: the estimator is
exact. The healthy arm was 11/20 on the same seeds an hour earlier (the aborted file):
SAPIEN's reset is not bit-repeatable across processes at the two-episode level, so
"same seed" here means the same object draw with a small pose jitter, as on the humanoid
before Sec 32.15 (pairing is within one process, arm after arm on the same seed list).

**Why 6 and not 13: the transient is permanent on this controller.** The estimate
reaches half the fault at step 14 and 80% at step 27 (all episodes). On LIBERO's delta
controller the uncorrected part of the fault during those steps is forgotten; on the
WidowX `use_target` controller it stays in the target: 0.07-0.09 action units accumulated
per episode (about 7 cm at 0.93 m per unit, with the policy's own commands at 0.01 per
step), more than the spoon. After convergence the residual bias is <= 0.001 per step. Two
schemes address exactly this and both are already in the paper: the identify-then-hold
scheme (Sec 32.17: identify over 3 episodes, apply the held correction from step 0 of the
next), and a faster gain. Both queued: hold3w (23 episodes, 20 held) and gamma 0.2.

### 34.2 Identify-then-hold, +0.005 on x,y,z (`cell_tra005_hold3w.json`, seeds 100-122)

Three identification episodes online (seeds 100-102: adaptive 2/3, healthy 3/3), the held
correction = median over the three per-episode window medians (steps 30-150):
(0.0049, 0.0048, 0.0046). Then 20 episodes (seeds 103-122) with that correction applied
from step 0, no adaptation.

| arm, held 20 | successes / 20 |
|---|---|
| healthy | 14 |
| frozen under the fault | 0 |
| adaptive, held correction | 14 |

Paired: adaptive-only 14, frozen-only 0, McNemar p = 1.2e-4. Adaptive vs healthy on the
same seeds: 5 each way, p = 1.0. The held correction restores the healthy rate exactly;
the leftover per-step bias of 0.0003 (about 4 cm over a full 150-step episode, but most
successes come at 30-80 steps) does not show in the success rate. Over all 23 episodes:
healthy 17, frozen 0, adaptive 16.

This is the same pattern as the humanoid (Sec 32.17-32.19): on a controller that keeps
its target, the online scheme pays for its transient every episode, and identifying once
and holding is the right deployment. On LIBERO the online scheme reaches the healthy rate
because its delta controller forgets the transient; ALOHA sends absolute joint targets and
needed the hold scheme for a different reason, the sub-centimetre margin (Sec 28.4). (Sentence
corrected 2026-09-10 after the dual-track audit flagged it: ALOHA is not a delta controller.)

### 34.3 The transient explanation, tested: gamma 0.2 online (`cell_tra005_g02.json`, seeds 100-119)

If the online scheme loses to the drift accumulated before convergence, a faster gain
should recover part of the gap in proportion. Gamma 0.2, everything else unchanged:

| gamma (online, from step 0) | steps to 80% of the fault (median) | drift accumulated by episode end (median, action units) | successes / 20 |
|---|---|---|---|
| 0.08 | 27 | 0.09 | 6 |
| 0.20 | 13 | 0.032 | 12 |

Paired against frozen (0/20): 12 fixed, 0 broken, p = 4.9e-4; against the healthy arm's
13/20 on the same seeds: 3 vs 4 discordant, p = 1.0. Final estimate median
(0.0048, 0.0052, 0.0048). Halving the transient doubles the success count and closes the
gap to the healthy rate; the hold scheme (Sec 34.2) removes the transient entirely and
reaches it exactly. The explanation is quantitative, not a story.

### 34.4 Null: the law on a healthy arm (`null_tra000.json`, seeds 100-119, gamma 0.08)

| arm | successes / 20 |
|---|---|
| healthy, no law | 15 |
| healthy, law running | 16 |

Discordant 3 vs 2, p = 1.0. Phantom at episode end: median |f| per axis 0.0003/0.0001/0.0002,
max 0.0009, a fifth of the fault of 0.005 and at the residual floor. The healthy rate in
this process was 15/20 against 13/20 and 11/20 in earlier processes on the same seeds:
the spread of a 20-episode binomial at p ~ 0.65 plus SAPIEN's reset jitter, and the reason
every comparison here is within-process and paired.

### 34.5 A second magnitude, +0.003 on x,y,z, hold scheme (`cell_tra003_hold3w.json`, seeds 100-122)

1.5x the healthy residual median, the mildest fault that damages this policy. Identification
episodes (seeds 100-102): adaptive 1/3, frozen 0/3; held correction (0.0028, 0.0030,
0.0030) = 93-100% of the fault.

| arm, held 20 (seeds 103-122) | successes / 20 |
|---|---|
| frozen under the fault | 2 |
| adaptive, held correction | 13 |

Paired: 11 fixed, 0 broken, p = 9.8e-4. Against the healthy arm of Sec 34.2 on the same
seeds (14/20, a different process): 4 vs 5 discordant, p = 1.0. The frozen policy scored
2/23 here against 0/10 in the probe (Sec 34): the same reset jitter as the healthy arm's
11-15/20. Two magnitudes, both at the healthy rate under the hold scheme, none broken.

**WidowX totals.** Paired episodes on this robot: continuous 20 + 20 (gamma 0.08, 0.2),
hold 20 + 20, null 20, healthy controls 20 + 23: 163 paired episodes, 57 fixed, 0 broken
across the four faulted cells.

**Aggregate after Sec 34.** 1,330 paired episodes (Sec 33.1) + 163 on the WidowX = 1,493;
the paper's contributions say "over 1,400".

### 34.6 Video (`results/phase05/adaptive_vs_frozen_widowx.mp4`)

`openpi/widowx_video.py`, the compare_video.py style: same seed in both panels (SimplerEnv
seeds its scene from `env.reset(seed)`, so the same spoon and towel placement), the
benchmark's own 256x256 third-person camera, which is exactly what the policy sees; the
task language, the fault and the held estimate on the frame. Hold scheme: the held vector
(0.0049, 0.0048, 0.0046) applied from step 0, as episodes 3-22 of Sec 34.2. Seeds 106 and
107, `--only-repaired`: frozen fails at the 150-step cap on both (the arm drifts up and
away from the spoon), corrected succeeds at steps 66 and 23. A first render (seeds 103,
104: corrected at 80 and 73) was redone for a title that overran the panel; in the redo
seeds 103-105 failed corrected before 106 and 107 succeeded, the same reset jitter as the
11-15/20 healthy spread, and the reason `--only-repaired` exists. Verified frame by frame
on a contact sheet before shipping.

## 35. OpenVLA on SimplerEnv: the Google robot, move-near (2026-09-09)

**Why not WidowX with OpenVLA.** The published SimplerEnv table puts base OpenVLA-7B at
0% on all four WidowX tasks; a healthy control at 0 is a floor, not a null (Sec 33 protocol).
On the Google robot it reports 46% on move-near, so that is the task.

**Setup.** `openpi/openvla_google_server.py` (port 8006, openvla-oft venv, transformers
4.40.1, bf16 emulated on the Turing card: 0.84 s per action). Base `openvla/openvla-7b`
(15 GB, OXE-pretrained, never finetuned), SimplerEnv's own conventions reproduced: prompt
"In: What action should the robot take to {instruction}?\nOut:", unnorm_key
`fractal20220817_data`, rpy delta -> axis-angle, raw open-ness in [0,1] to the GR00T gym
wrapper whose Google branch applies the relative/sticky gripper rule. Client
`widowx_adapt.py --task simpler_env_google/google_robot_move_near --horizon 1 --max-steps 80`
(pose from the wrapper's quaternion; image key `video.image`, 256x320). 3 Hz control, 80
steps, seeds 100+.

**Healthy log: 2/10.** Successes at 79 and 41 steps; the arm moves 0.2-0.4 m net per
episode. Against the published 46%: P(<=2 | 0.46, n=10) = 4%; the seed draw here is
uniform over the task's object pairs rather than the benchmark's fixed 60 configurations,
and nothing in the conventions was found wrong. Taken as measured; the healthy arm of the
paired cell will add 40 more.

**The weakest plant in the paper.** The arm controller is
`arm_pd_ee_delta_pose_align_interpolate_by_planner`: a per-step delta (use_target False, so
no accumulation, unlike the WidowX) executed by a motion planner within the 3 Hz step. Only
20-30% of a commanded delta is realised in the step, and the fraction falls with the command
(x: 0.28 below 0.01, 0.20 above 0.03): a mild saturation. FIR K=6 R^2 on x,y,z
0.57/0.62/0.35, unchanged at K=2 or K=10 (so lag is not the missing part; the nonlinearity
is); rotation channels R^2 ~ 0 (quaternion-derived rpy, unmodelled, never corrected).
Per-unit motion scale 0.28/0.25/0.22 on x,y,z; DC gain in normalised units
0.85/0.91/0.80 (`openloop_dc.json`, the WidowX lesson applied from the start). Healthy
residual norm on x,y,z: median 0.017, 90th pct 0.042, with commands of 0.013-0.022: the
residual is as large as the command. Per-step identification SNR for a fault f is ~f/0.017;
the innovation form integrates it over an episode. Constants: dead 0.008, norm_r 0.08,
clip 0.1, gamma 0.08.

**Damage probes** queued: frozen under +0.02 and +0.04 on x,y,z (10 eps each).

**Damage probes and the close-out (2026-09-10).** Frozen under +0.02: 0/10; under +0.04:
0/10 (the fault damages; the healthy 2/10 is itself near the floor). Offline replay of the
law on the faulted logs against the healthy phantom, window estimate (median over steps
30-80) per episode:

| fault on x,y,z | window-estimate separation (% of the fault; sd) | mean-residual separation (% of DC x fault) |
|---|---|---|
| +0.02 | x 2% (0.1 sd), y 19% (0.4), z 19% (0.5) | 37 / 42 / 57 |
| +0.04 | x 1% (0.0 sd), y 8% (0.2), z 6% (0.2) | 19 / 18 / 22 |

A saturating and a quadratic plant model do not change this (R^2 0.52-0.64, separation
30-64% at 1-2 sd). The larger fault separates *less*: the planner realises a smaller
fraction of a larger delta, so the offset is absorbed by the controller rather than
passed to the motion, and the residual is dominated by the plant's own model error (median
0.017, as large as the command). Compare LIBERO: 87-100% of the fault at >10 sd.

**Decision: no paired cell.** Both conditions the paper states as necessary fail here at
once: the healthy control is at the floor (2/10 against a published 46%), and the fault
is not identifiable from the residual (Proposition 1's premise, an action-interface fault
that reaches the measured motion through a plant the healthy data identifies, does not
hold on a planner-interpolated controller). A cell would report a null on a floor, which
Sec 33's protocol rejects. Recorded as a boundary: the method needs a plant the healthy
log can model (R^2 well above 0.9 on the corrected channels), and the SimplerEnv Google
robot's 3 Hz planner controller is not one. GPU released; server stopped. Total cost: 30
episodes, 45 minutes.

**What this does and does not say.** It does not say OpenVLA cannot be repaired: on
LIBERO its OFT variant is (Sec 30, three families). It says the SimplerEnv Google-robot
plant is outside the method's stated condition, for a reason measured before any repair
was attempted, which is the point of stating the condition.

## 36. Integrating the dual-track audit branch (2026-09-10)

`review/dual-track-audit` (120 commits, author fengze, Sept 7-9) was merged into
`integrate/audit-fixes` following its own handoff (`docs/branch_comparison_20260909/CLAUDE_HANDOFF.md`).
What came in: the McNemar relative-tolerance fix and duplicate-key guard, the independent
cross-check of every printed p-value (15 cells reproduce), the fault-lifecycle and reset-fingerprint
protocol with tests, the ALOHA telemetry and innovation variants, the weighted allocator and
composite estimator (experimental, with their counterexamples), the all-joint physical-fault
confirmation (1,440 Panda rollouts, 780 ALOHA), the matched-estimator tuning study (5,424
rollouts: no estimator beats Kalman; the original law scores 50% balanced), the descriptor-form
module `learned_adaptation/` (ARX identification, constant basis; 112/140 -> 135/140 on the seven
Spatial torque cells against its own frozen controls), 51 bibliography entries, and the audit's
paper draft with its conditional analysis.

**Paper.** The audit draft is the base (it carries the corrected claims); main's newer evidence
was ported into it: the paired two-task GR1 table and magnitude map with the time-varying
paragraph (superseding the unpaired cohorts, which stay in Sec 32.10-32.13), the WidowX
subsection, the OpenVLA Google-robot boundary, the SimplerEnv citation. The four-suite, three-backbone
and hold-scheme results now sit in the appendix "Extended experiments and historical evidence",
which is the audit's framing: the main text leads with the conditions under which identification
becomes repair, the all-joint confirmation and the estimator study. Main-text page count 9,
verified with the audit's `check_submission_layout.py` (backmatter boundary from the aux file).

**A build defect found on the way.** This machine's tectonic bundle silently substitutes Latin
Modern for the `times` package; every earlier page-9 check in this record (Secs 33-35) ran in a
font about 6% wider than the submission font, so those checks were conservative, never
permissive. `paper/build_tectonic.tex` loads `newtxtext` (Times-metric) at the end of the preamble
without touching the submission source.

**Not imported.** The branch's `AGENTS.md`, its canonical-repository declaration and dual-push
policy (local configuration and project identity, per the handoff). Kept unchanged from main:
`gr1_adapt.py`, the WidowX and Google-robot scripts, all GR1/WidowX/Google results and videos.

**Tests in the merged tree.** joint-fault lifecycle, calibration pipeline, weighted DOB, composite
observer and validation, tuning and summary preparers, the LIBERO and ALOHA selftests,
`mcnemar_crosscheck`, and the 124 `learned_adaptation` tests (one skipped, MuJoCo optional): all pass.

**Still open from the audit, needing GPU time.** (1) The channel mask: the rotation-only
restriction on the headline cells came from a separation test on faulted rollouts; the audit's
proposed fix is an online gate on the healthy phantom's standard deviation (healthy data only),
one to two GPU-days, and it decides whether the "no faulted data" claim is true end to end.
(2) Held-out recalibration: `M` was measured on init 45, which the headline cells evaluate.
(3) Reruns of the historical friction/lock cells under the new fault-restoration protocol.

## 37. The channel mask from healthy data alone (2026-09-10)

**The objection** (audit §4.5): the headline cells correct rotation only, and that mask came from
a separation test on faulted rollouts, so "no faulted data" was not true end to end.
Preregistered in `prereg_records/PREREG_HEALTHY_GATE.md` before anything was measured.

**The gate.** Healthy phantom from 20 healthy `--estimate-only` scenarios (the headline's 20
(task, init) pairs, inits 45/46; healthy 20/20 in both arms), statistic = per-episode mean
estimate over the last 50 steps, b = mean and sd = across-episode sd per channel
(`results/gate/phantom_stats.json`):

| channel | b | sd | 3 sd |
|---|---|---|---|
| x | +0.015 | 0.039 | 0.117 |
| y | −0.003 | 0.015 | 0.045 |
| z | +0.029 | 0.040 | 0.119 |
| rx | +0.000 | 0.003 | 0.010 |
| ry | −0.000 | 0.013 | 0.038 |
| rz | +0.001 | 0.002 (floor) | 0.006 |

Rule, every step, no memory: correct channel i iff |f̂_i − b_i| > 3 sd_i. The estimator runs on
all six channels with the headline constants (legacy law, γ 0.08, dead 0.008, ρ 0.15, clip 0.15).
Nothing in the rule has seen a fault. The translation phantom is wide (3 sd > the 0.05 fault on x
and z), which is the healthy-data reason translation is untrustworthy on this plant.

**The cell** (`results/gate/spatial_uniform005_gate3.json`): π0.5, `libero_spatial`, uniform +0.05
on all six action dimensions, the headline's 20 scenarios, paired.

| arm | successes / 20 |
|---|---|
| frozen under the fault | 9 |
| gated correction, k = 3 | **19** |

10 fixed, 0 broken, exact McNemar p = 0.0020. The rotation-only headline on the same scenarios
was 8/20 → 18/20 (10 fixed / 0 broken). **Prediction 1 confirmed**: the healthy-only gate
reproduces the headline, so the restriction to trustworthy channels needs no faulted rollout.

**Prediction 2, half refuted.** Gate occupancy after step 20 (median over episodes): rx 100 %,
rz 100 %, ry 28 %; x 17 %, y 36 %, z 20 %. Rotation was predicted > 80 % open: true for rx and
rz, false for ry, whose estimate settles at 0.021 against a 3 sd threshold of 0.038 (the channel
that lags on every backbone, §30, stays mostly closed). Translation was predicted < 20 % open:
true for x and z, false for y at 36 %. So the gate applied a translation correction on a third
of the steps on y and the cell still repaired 19/20 with no regression. §19.3's finding that
correcting translation hurts under the uniform fault was about correcting it *always*; an
intermittent, thresholded translation correction did not hurt here. That is a new fact, not
a contradiction, and it is recorded as such.

**What this changes in the paper.** The channel restriction is now derivable from the healthy
calibration: measure the phantom, gate at 3 sd. The rotation-only mask of the headline cells
stands as the fixed-mask version of the same decision. The sentence "our mask ... selection
use faulted development outcomes" is amended to: the mask was first found by a faulted
separation test and is reproduced by a healthy-only gate (§37), so the deployed method needs
no faulted data; hyperparameters remain development-tuned.

**Healthy control with the gate running** (`results/gate/healthy_gate3.json`, same 20 scenarios):
healthy no-law 20/20, gated 19/20, 0 fixed / 1 broken, p = 1.0. The lost episode (task 5, init 45)
ran to the 220-step cap with the gate open on 4–10 % of its steps per channel. **Prediction 3,
half refuted:** the healthy rate is unchanged within the registered bound (one discordant), but
the gate was predicted to open on < 5 % of healthy steps and opened on 30 % (any channel;
per channel: z 12 %, rx 10 %, rz 9 %, y 7 %, x 3 %, ry 2 %, means over episodes). A 3 sd
threshold on a statistic averaged over 50 steps is not a 3 sd threshold on the instantaneous
estimate, which is what the gate reads; the instantaneous estimate is noisier, so the gate
flickers open on a healthy arm. That flicker cost nothing measurable on 19 episodes and may
have cost the twentieth: the same order of harm as the OFT healthy control (one in twenty,
§25), and reported the same way. A dwell requirement (open only after m consecutive steps
beyond the threshold) is the obvious fix and is NOT applied here, because k and the rule were
fixed in the preregistration; it is a second preregistration if it is run.

**Bottom line for §37.** With the mask derived from healthy data alone, the headline cell is
9/20 → 19/20 (10/0, p = 0.002) and the healthy control 20/20 → 19/20 (0/1, p = 1.0). The
"no faulted data" claim holds end to end for the channel selection; the hyperparameters
remain development-tuned, and the gate's healthy-arm flicker is the stated cost.

## 38. The paper: original story, audit corrections folded in (2026-09-10)

The merged draft of Sec 36 carried the collaborator's framing (their all-joint and estimator
studies in the main text, ours in an appendix). At the user's direction the main text now tells
the original story again, from the pre-merge draft, with every audit correction folded in:

- pooled four-suite p = 2.4e-14 (relative-tolerance instrument); "zero regressions in every cell"
  → every regression sits in a ceiling cell (translation 0.05 is 1 fixed / 1 broken); fixed/broken
  printed beside every healthy-control total (GR00T 1/1, ALOHA 5/8, GR1 7/6 on both tasks,
  WidowX 3/2); "same (task, init, seed)" → same (task, init), policy sampling not pinned;
  the proprioceptive bias damages nothing (15/15 frozen) and the wrist fault is an image roll;
  the regression denominator that can regress (1 of 28 in the four-suite cell); M probed at
  init 45 disclosed, held-out recalibration registered (`PREREG_HELDOUT_CALIBRATION.md`) and
  pending in an appendix; the channel mask reproduced by the healthy gate (Sec 37).
- related work rebuilt on the merged 51-entry bibliography; "the machinery earns its place" →
  the plant model earns its place, matched estimators sharing it repair as well (19/17/17/18/15
  of 20, oracle 20), the collaborator's 5,424-rollout ALOHA study finds no estimator beating
  another; both their studies reported in an appendix with their tables, attributed.
- the abstract's tally is now computed: `openpi/aggregate_tally.py` sweeps every stored paired
  cell under written rules and writes `results/aggregate_manifest.json`: 110 cells, 2,240 paired
  episodes, 625 fixed, 66 broken (2.9 % of all, 10.3 % of episodes the frozen policy was winning).
  The old hand-assembled "690 / 275 / 7" is gone.
- the audit's framing is kept verbatim as `paper/iclr_draft_audit_framing.tex`.
- 9 main pages under Times metrics, no overfull boxes, verified with the audit's checker.

## 39. Held-out calibration (2026-09-10, running)

Prereg `PREREG_HELDOUT_CALIBRATION.md`. Healthy FIR log at `--init-base 25` (10 episodes,
`results/heldout/error_signal_init25.json`), M by the same 0.02 probe at init 25
(`openloop_init25.json`).

| | shipped (init 45) | held-out (init 25) | ratio |
|---|---|---|---|
| M diagonal x, y, z | 0.297, 0.272, 0.126 | 0.268, 0.221, 0.100 | 0.90, 0.81, 0.79 |
| M diagonal rx, ry, rz | 0.253, 0.276, 0.244 | 0.233, 0.259, 0.244 | 0.92, 0.94, 1.00 |
| condition number | 3.0 | 3.1 | |

**Third prediction (M within 15 %) refuted on y and z** (19–21 % lower at init 25); rotation
within 8 %. The initial state matters to the translation sensitivity at the 20 % level, which
is the magnitude the audit's "M is magnitude independent to 14 %" finding did not cover
(different quantity: magnitude there, configuration here). The rotation-corrected headline
cells depend on the rotation block, which held.

**The held-out plant fits as well, and the provenance is better.** FIR R^2 on the shipped
3-episode log (init 45) vs the held-out 10-episode log (init 25): translation
0.96/0.98/0.98 vs 0.95/0.98/0.96; rotation 0.52/0.33/0.97 vs 0.81/0.42/0.97 (rx improves
with more episodes). `ry` is the worst-fit channel in both (0.33-0.42) and is the channel
whose estimate lags on all three backbones and stays mostly closed under the healthy gate
(Sec 37): one explanation, three symptoms. The held-out log is schema v2: source hashes for
all five scripts, `reset_protocol: libero-reset-v1`, the calibration episode list, and
`policy_rng_pinned: false` recorded. The shipped log has none of that, which is the
provenance gap the audit named.

**The cells** (same faults, channels and constants; only `--log`/`--openloop` changed):

| suite | original | held-out calibration |
|---|---|---|
| `libero_spatial` n=20 | 8/20 -> 18/20, 10 fixed / 0 broken, p = 0.0020 | 9/20 -> 18/20, 9 / 0, p = 0.0039 |
| `libero_object` n=20 | 5/20 -> 16/20, 11 / 0, p = 0.00098 | 7/20 -> 15/20, 10 / 2, p = 0.039 |
| `libero_goal` n=40 | 15/40 -> 29/40, 15 / 1, p = 0.00052 | 18/40 -> 28/40, 11 / 1, p = 0.0064 |
| `libero_10` n=40 | 0/40 -> 15/40, 15 / 0, p = 6.1e-5 | **0/40 -> 7/40, 7 / 0, p = 0.016** |

**Prediction 1 holds on three suites and is refuted on the fourth.** Corrected counts 18, 15,
28, 7 against 18, 16, 29, 15: within the registered +-3 on spatial, object and goal, and
**8 below** on `libero_10`, where the registered refutation threshold was a drop of more than 5.
Every cell is still individually significant and the pooled broken count is 3 against a
threshold of 5, so repair still happens everywhere; it is *weaker on the long-horizon suite*
when the calibration has not seen the evaluated initial state. Pooled: held-out
34/120 -> 68/120 (37 fixed, 3 broken) against the original 28/120 -> 78/120 (51 fixed, 1 broken).

**What this means, stated as the refutation it is.** The audit was right that calibration and
evaluation overlapped, and the overlap was partly load-bearing: not on the three shorter
suites, where the corrected count is unchanged to within one episode, but on `libero_10`,
whose 40 episodes are long-horizon compositions where a 20 % error in the translation block
of M (Sec 39 above) has 200+ steps to accumulate. The honest headline is therefore two
numbers, not one: with a calibration probed at the evaluated initial state, 28 -> 78; with one
identified on disjoint initial states, 34 -> 68. The paper reports the held-out row as the
primary result for `libero_10` and keeps both for the other three.

The frozen arms differ by 0-3 episodes (9 vs 8, 7 vs 5, 18 vs 15, 0 vs 0) with no calibration
involved at all, which is the unpinned policy sampling the protocol section now states.

## 40. Joint-level cells under the corrected fault protocol (2026-09-11)

Prereg `PREREG_JOINT_RERUN.md`. The audit's last rerun item: the historical friction and lock
cells (§29.3) ran with model mutation on a cached environment without guaranteed restoration,
and with `reset(); set_init_state(...)`, which does not prove the two arms start from the same
physical scene. Reruns use `JointFault`'s `finally`-restoration and `--scenario-reset`
(exposed today; the merged branch shipped `libero_reset` with the flag off, the partial fix the
handoff warned about). Everything else is unchanged: π0.5, `libero_spatial`, n = 20, elbow,
translation corrected, phantom subtracted, clip 0.30.

| cell | historical | rerun, corrected protocol |
|---|---|---|
| elbow friction +20 | 0/20 → 8/20, 8 fixed / 0 broken, p = 0.0078 | 1/20 → 4/20, 3 / 0, **p = 0.25** |
| elbow lock ±0.05 rad | 0/20 → 0/20, 0 / 0 | 0/20 → 0/20, 0 / 0 |
| healthy control, law running | not previously run | 19/20 → 20/20, 1 / 0, p = 1.0 |

**The damage was not an artefact.** Every frozen arm reproduces: friction 1/20 against 0/20,
lock 0/20 against 0/20. The audit's first worry, that the historical damage came partly from
uncleared force state, is answered: it did not.

**The lock cell is reproduced exactly, estimate included.** Final estimate x −0.232, z −0.189
against the recorded −0.23, −0.16, one clip hit in sixty channel-episodes. Identified,
corrected, not repaired — the rank argument survives the corrected protocol, which is the
result the paper leans on for Proposition 1's boundary.

**The friction repair does not reproduce at n = 20** (prediction 1 required ≥ 5 corrected;
4/20). The estimate is nearly unchanged between runs (final median x, z: −0.23, −0.23
historical against −0.19, −0.20 rerun), so identification is not what differs; the task
outcome is. At n = 20 on a benchmark with ±11 points of noise, 4/20 against 8/20 cannot
separate "the protocol removed the effect" from "the effect is half this size and noisy", so
an n = 40 extension was registered before running it (amendment in the prereg) with the
decision rule written down: ≥ 10/40 with ≤ 2 broken and p < 0.05 keeps the row, anything else
withdraws the friction claim from the paper's joint-fault table.

**A phantom that does no harm.** On a healthy arm the translation law holds a large phantom
(final |f̂| median 0.055, 0.015, 0.080 on x, y, z; max 0.167) after the §29.2 bias subtraction,
and costs nothing: 20/20 against 19/20. Consistent with §19's reading that a Cartesian reach
tolerates centimetres of translation jitter, and a contrast with ALOHA, where 0.43 cm of
within-episode variation was the whole margin.

**Resolved at n = 40: the friction repair survives the corrected protocol.** `0/40 → 13/40`,
13 fixed / 0 broken, exact McNemar p = 2.4e−4 (`results/jf_rerun/jf_friction_3_20_reset_n40.json`).
The registered keep-rule is met, so the row stays, with these numbers replacing the historical
n = 20 ones: 33 % from a floor of zero, no regression in forty paired episodes, and a p-value
thirty times smaller than the historical cell's on a cleaner protocol. By block: inits 45/46
give 6/20, inits 47/48 give 7/20. The same twenty 45/46 scenarios therefore scored 8, 4 and 6
across three runs, which is exactly the ±11-point noise of a single n = 20 LIBERO cell and the
reason the n = 20 rerun could not decide the question. The audit's rerun item is closed: the
damage is real, the lock boundary is reproduced, the friction repair holds at a measured 33 %
rather than the historical 40 %, and the law does no harm on a healthy arm.

## 41. Figures corrected by the re4 forensics (2026-09-11)

The re4 evidence plan's items G.3 and G.4 (`results/re4_evidence/G_forensics/`, a CPU-only
recomputation from stored data) found five figures in the paper and report that the stored data
do not support. Each was recomputed independently before correcting it:

| figure | where | stored data say | correction |
|---|---|---|---|
| ALOHA plant R² "0.989–1.000 on all joints" | §26, paper ALOHA subsection | runner's own `fit_plant`: arm joints 0.9891–1.0000; grippers j6 0.9915, **j13 0.768** | "on the twelve arm joints; the uncorrected right gripper 0.77" |
| n = 40 identify-then-hold "mean residual 0.12 cm" | paper ALOHA table | held vector (0.0193, 0.0202, 0.0190, 0.0191, 0.0192, 0.0192) against 0.02 at 106 cm/rad: **0.071–0.078 cm**; 0.12 cm is the n = 20 run's value | 0.07 cm |
| GR1 "R² ≥ 0.998 on the arm" | paper GR1 subsection | true of the plate-to-plate plant; tray-to-plate cells use `t2p_healthy_log.json`, a separate ten-episode plant, 0.994–0.9997 | both plants named |
| ALOHA identification "92–96 %" | Figure 1 caption | n = 20 run: 0.01835–0.01907 rad = **92–95 %** | 92–95 % |
| LIBERO "σ of 0.019 cm per step" set against ALOHA "σ = 0.43 cm" | Prop. 2 discussion; report criterion table | 0.019 reproduces from `tmag_015.json` as the **mean per-step change of the translation correction** after step 15, × 5 cm per commanded unit (0.0191); the arm achieves **0.0044 cm**. It is a per-step change of a *velocity* offset, while ALOHA's 0.43 cm is the **median within-episode range** (not an SD; SD 0.13 cm) of a *position* offset at the uniform-offset 106 cm/rad | the two are not on one scale; the Prop. 2 comparison across the interfaces is qualitative |

The last row matters for the argument, not just the arithmetic. Proposition 2 says repair
needs the correction's within-episode variation to fit inside the task margin. On ALOHA both
sides of that inequality are positions and the comparison is quantitative (0.43 cm range
against a sub-centimetre margin). On LIBERO the correction commands a velocity, so its
variation integrates into position drift that the vision loop keeps re-closing; no single
number on the LIBERO side is directly comparable to ALOHA's. The criterion's prediction for
LIBERO (repair) holds, but the paper no longer presents the two figures as one scale.

G.3's reading is recorded here as well: the FIR's position R² is within 0.001 of the
persistence predictor's, so position R² does not show that the model captures command-induced
motion. On increments, the held-out FIR reaches 0.53 (ALOHA left arm) and 0.38 (GR1
plate-to-plate) against persistence at about 0, and −1.77 on GR1 tray-to-plate, where it is
worse than persistence. The correction relies on the plant's steady-state gain, which is about
1 on every corrected joint except the tray-to-plate wrist (0.91 and 0.94).

**Documents left as written.** The same superseded figures appear in the collaborator's audit
(`report/DUAL_TRACK_AUDIT.md`: the 0.019 cm / 0.43 cm comparison), in the audit's framing of the
paper (`paper/iclr_draft_audit_framing.tex`: the n = 40 row's 0.12 cm and the all-joints R²), and
in a historical preregistration (`prereg_records/PREREG_GR1_MARGIN.md`: "σ measured? yes (0.019
cm/step)"). They are not edited: the first two are another author's documents kept verbatim, and a
preregistration is a timestamped record. This section is the correction of record for all three.

## 42. What the re4 provenance map found (2026-09-11)

`results/re4_evidence/A_provenance/implementation_map.{md,json}` (re4 plan Part A and G.2)
recovers, for all nine cohorts, the code path, gate convention, clipping order, K, law and
constants, and calibration. Every field was recovered; for the older result files, which store
no settings, the exact launch commands came from this project's session transcript (outside the
repository) and were cross-checked against the stored per-step estimates. The paper cannot cite
that transcript; the cross-check against stored data is what it can cite.

**Answers to the plan's questions.** Every published cohort subtracts the correction outside
the network; the native `action_out_proj/bias` edit exists only in the ACE, oracle and G.1
scripts, so re4's "code transition" hedge can go. K = 6 in every cohort, ALOHA and GR1
included. Gate convention: leakage deadzone on LIBERO (π0.5, OFT, GR00T, joint-level) and
ALOHA; innovation law with a hold deadzone on GR1 and WidowX. The deadzone never fired in the
four headline samples or in ALOHA's identification episodes, so the convention did not affect
those outcomes; it did fire on some steps of the joint-level cells, the healthy controls, the
deadzone-0.03 ablation row, and GR1/WidowX. All four headline suites used the spatial
calibration. M used 17 replays on LIBERO, 29 on ALOHA, 29 on the GR1 right arm, none on WidowX
(DC gain from the healthy log). G.2: only the joint-level cohort subtracts a healthy phantom,
the settled attenuated estimate, measured open-loop and applied closed-loop.

**Inconsistencies, and what was done** (✓ = verified independently before acting):

| finding | action |
|---|---|
| ✓ report said the correction edits `action_out_proj/bias`; the runner adds it externally | report corrected |
| ✓ paper said the norms run over the corrected channels; LIBERO uses all 6, ALOHA all 14 (runner default, no stored override) | paper corrected per robot |
| ✓ paper never said WidowX runs the innovation law | paper corrected |
| ✓ ALOHA "M ≈ I, cond 7.3": cond is 1.17 on the arm; 7.3 comes from the right gripper's 0.145 | report corrected |
| ✓ GR1 plate-to-plate null (§32.13) predates the reseeding fix (§32.15), so it is unpaired; I had attached paired fixed/broken counts to it | paper now gives Fisher for it, paired counts only for the tray-to-plate null (§32.19) |
| ✓ WidowX +0.003 cell compared against the healthy arm of the +0.005 run, a different process | paper wording states it |
| unpaired four-suite tables (§7.1–7.2) used per-suite calibration, the paired headline the spatial one, a switch the record never mentioned; §23.2 says object re-identification "has not been run" though it had | recorded here |
| the shipped M replay ran past the end of healthy episode 0 (75 steps) into 5 commands from another task | recorded; the held-out M of §39 does not share this |
| calibration–evaluation overlap, precisely: 2 of the 20 spatial episodes share a (task, init) with the plant log, and constants were tuned on the same init band | recorded; §39's held-out cells are the answer |
| OFT ran 5 of each 8-step chunk (default), not 8 as the record says | record corrected here |
| the ablation's "reference" row is the 09-01 headline file, not a run made with the ablation | recorded |
| the runner's default constants never matched the published ones | recorded; `run_configuration.json` now stores every constant |

## 43. Matched healthy controls on the exact primary scenarios (re4 Part C, 2026-09-11)

Prereg `PREREG_RE4_C_HEALTHY_CONTROLS.md`; runs under `results/re4_evidence/C_healthy/`, every
one with `--scenario-reset`, the section-0 records and per-step timing. The exact (task, init)
lists of the four headline cells, no fault, two arms: healthy frozen, and healthy with the law
running on the rotation channels. Twice, with the shipped and the held-out calibration.

| calibration | spatial | object | goal | libero_10 | pooled (no law → law; fixed / broken) |
|---|---|---|---|---|---|
| shipped | 19→20 | 20→20 | 39→40 | 38→37 | 116/120 → 117/120; 4 / 3, p = 1.0 |
| held-out | 20→20 | 20→20 | 40→39 | 38→39 | 118/120 → 118/120; 2 / 2, p = 1.0 |

Both registered predictions hold under both calibrations: every healthy rate sits above its
suite's corrected rate, and the law on a healthy robot changes no suite by more than one
episode net, with at most three broken (the shipped-calibration `libero_10` cell, 37 against
38, the weakest of the eight and at the registered bound). The headline table's corrected
rates are therefore 67 % (shipped) and 58 % (held-out) of health measured on the same 120
scenarios, which is the denominator the paper lacked. Adapter compute per control step over
the eight runs: 0.17 ms median, 0.46 ms at the 99th percentile, 0.77 ms maximum; the policy
call is about 900 ms on this GPU and the simulator step 30–40 ms, neither of which is the
controller's.

## 44. The decisive missing baselines (re4 Part D, 2026-09-12)

Prereg `PREREG_RE4_D_BASELINES.md` (outcome appended there). Headline fault, headline constants,
`--scenario-reset`, full logs; each arm against its own frozen arm, baselines compared with the
method by scenario key.

| run | spatial (n=20) | libero_10 (n=40) |
|---|---|---|
| D.0 the method, rerun with full logs | 6 → 18 (12/0) | 0 → 15 (15/0) |
| D.1 static observer, K = 0 | 11 → 18 (7/0) | 0 → 9 (9/0) |
| D.2 innovation law, same constants | 9 → 18 (9/0) | 1 → 12 (12/1) |
| D.3a known −f, rotation channels (oracle) | 8 → 19 (12/1) | 1 → 22 (22/1) |
| D.3b known −f, all six channels | 9 → 20 (11/0) | — |

**What the baselines say.** (i) The headline reproduces under the corrected protocol to the
episode (18/20, 15/40). (ii) A static gain with no FIR memory does as well on the short suite and
loses six of forty on the long-horizon suite (D.0-only 7, D.1-only 1, p = 0.07; pooled p = 0.11):
FIR memory is worth something on long horizons and the registered significance bar was not met.
(iii) The innovation law is within three of the legacy law on both cohorts, as registered; it
removes the attenuation bias (0.049 against 0.044 on rx, rz) and does nothing for ry, which is
under-identified under both. (iv) The known correction through the same channels is the ceiling
for this interface: the method achieves 1.09 of it on spatial and 0.71 on libero_10, where the
exact rotation correction reaches 22/40 and the estimated one 15/40; the seven-episode gap is
the price of identification, mostly ry. (v) With all six channels corrected exactly, spatial is
20/20 against a healthy 19/20: the fault is fully cancellable through the interface, and the
rotation-only mask costs at most one episode of ceiling on this suite.

## 45. M at a third initial state (re4 Part H, 2026-09-12): a refutation

Prereg `PREREG_RE4_H_THIRD_INIT.md`. Probed at initial state 5 (states 5–14 for the healthy log):
diagonal 0.253, 0.259, **0.262**, 0.252, 0.246, 0.243, condition number **1.29**. Against the
shipped (state 45) matrix: x −15 %, y −5 %, **z +107 %**, rx 0 %, **ry −11 %**, rz 0 %. Two of the
three registered bands fail (rotation within 10 %; condition number in [2.5, 3.5]). The shipped
matrix's small z entry (0.126, half the other translation axes) is a property of state 45, not of
the plant; at state 5 the matrix is nearly isotropic. §39's reading that "the rotation block held"
was true of one pair of states and is withdrawn as a general statement: which entry of M moves
depends on the configuration probed. The held-out cells of §39 remain what they are; this adds
that a calibration should be probed at, or near, the deployment configuration, and that a probe
at a single state can carry a factor-of-two error in one axis.

## 46. A native decoder-bias edit does not realise an intended correction (re4 G.1, 2026-09-12)

Prereg `PREREG_RE4_G1_DECODER.md`; `openpi/g1_decoder.py`; 30 fixed observations, sampler
pinned, bias edits at ±0.02, ±0.05, ±0.1 on the seven LIBERO decoder dims. At a fixed observation
the response is deterministic (repeat error 0.0), linear (‖D−J‖/‖J‖ = 0.035, quadratic part
≤ 8 % at the largest edit) and diagonal-dominant (off-diagonal ≤ 15 % of the diagonal). Across
observations the gain varies 27–41 % (coefficient of variation of J's diagonal), so a single bias
vector computed from the mean Jacobian realises an intended ±0.05 correction with a median error
of **43 %** (ζ p95 1.9, max 4.5). Predictions 1–3 confirmed, 4 and 5 refuted. The conclusion for
both papers: applying the correction outside the network, which every published cohort does
(§42), is exact by construction; editing the decoder bias is not, and the measured ζ is the number
re4's decoder bound should carry.

## 47. Held vs continued from a matched estimate (re4 Part F, 2026-09-12): the hold scheme, re-read

Prereg `PREREG_RE4_F_HELD_VS_CONTINUED.md` (both outcomes appended there); runs under
`results/re4_evidence/F_held_vs_continued/`. Every arm starts every episode from the stored
identified estimate; the arms differ only in what happens after step 0.

| robot, fault | held | continued, innovation law | continued, legacy law |
|---|---|---|---|
| ALOHA, +0.02 rad on joints 0–5, n = 40 | 15/40 | **14/40** | 0/40 |
| GR1, +0.10 rad on the right arm, n = 30 | 22/30 | 17/30 | 1/30 |

**What was confounded, and what it separates into.** The paper's ALOHA argument was: the
updating correction moves 0.43 cm within an episode, the margin is under 0.5 cm, so updating
must stop. With the initial estimate matched, the innovation law updating throughout repairs
14/40 against the held 15/40 (p = 1.0) while moving 0.30 cm; the legacy law repairs 0/40 while
moving 0.40 cm. The difference is bias, not movement: the legacy law's estimate is pulled to
0.016 by step 20 (its attenuated fixed point, plus deadzone leakage once the fault is cancelled),
a 0.42 cm residual during the grasp; the innovation law's estimate stays at the fault. On the
humanoid the same two arms give 17/30 and 1/30 against the held 22/30: there the innovation
law does lose five episodes to updating (registered margin met, p = 0.23), and the legacy law's
estimate decays to the deadzone within 200 steps.

**What changes in the paper.** (i) "ALOHA with continuous adaptation: no repair" becomes
"with the legacy law"; the innovation law repairs ALOHA while updating. (ii) The deployable
scheme is stated as: use an update whose correct estimate is a fixed point (the innovation
form); hold the estimate where the task margin is smaller than the update's own movement, which
on this evidence is the humanoid, not ALOHA. (iii) Proposition 2 stands as a necessary
condition and its ALOHA example is relabelled. (iv) The GR1 innovation-law bias figure of 50 %
was the attenuation reading; the measured mechanism under the legacy law is deadzone leakage
to near zero, which is stronger, and the paper says leakage.

This is the fourth registered refutation of the re4 programme (libero_10 under held-out
calibration, M at a third state, decoder-bias realisation, and this), each reported as primary.
