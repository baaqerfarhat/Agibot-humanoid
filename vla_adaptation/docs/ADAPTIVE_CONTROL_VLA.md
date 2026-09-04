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
