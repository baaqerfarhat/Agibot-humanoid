# Online adaptive correction of a frozen VLA — theory and results

**Result in one line.** A fault that halves task success on a frozen 3.35 B vision-language-
action model is corrected *within a single episode*, from the robot's own motion, with no
gradients through the task and no search over task success: **45% → 95%, p = 1.1×10⁻⁶**
(n = 40 per arm), against a 99% fault-free ceiling. It generalises to a fault pattern it was
never tuned on (17% → 83%), tracks a fault that appears mid-episode within 0.75 s, and costs
nothing when no fault is present.

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
