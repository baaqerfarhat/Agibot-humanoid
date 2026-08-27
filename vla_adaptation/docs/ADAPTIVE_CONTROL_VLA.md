# Online adaptive correction of a frozen VLA — theory and results

**Result in one line.** A fault that halves task success on a frozen 3.35 B vision-language-
action model is corrected *within a single episode*, from the robot's own motion, with no
gradients through the task and no search over task success: **47% → 87%**, against a 99%
fault-free ceiling.

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

### 2.3 Error

With `u_t = a_t + c_t + f` the executed action (policy command, our correction, the unknown
fault), and `P̂` the identified fault-free plant:

```
r_t  =  y_t  -  P̂(a_t + c_t)   ≈   M f
```

`r_t` does not depend on `c_t`, so the estimator is not chasing its own correction — an
important property, and the reason this form was chosen over comparing against a
reference trajectory.

### 2.4 The map, and why it must be a matrix

`M = ∂(achieved motion)/∂f`, measured **open loop**: a recorded command sequence is replayed
with and without a fault, so the commands are identical by construction and any difference in
motion is the fault propagating through the plant. One input axis at a time, ±0.02 central
difference:

```
              dx       dy       dz      drx      dry      drz   <- fault applied to
    dx     0.291    0.024   -0.115   -0.025    0.068   -0.004
    dy     0.006    0.253    0.024   -0.012    0.018    0.005
    dz     0.027    0.030    0.130   -0.025    0.013   -0.004
   drx    -0.004   -0.000   -0.001    0.251   -0.001    0.059
   dry     0.001    0.003   -0.002   -0.024    0.010    0.383
   drz    -0.013    0.000    0.001    0.014   -0.424    0.009
```

Three properties decide the design:

1. **Attenuation.** The diagonal is 0.13–0.29 — the plant damps the fault 3–8×, so the law
   needs roughly 4× gain on the observed error.
2. **The rotation block is a swapped, sign-flipped pair.** `dry` answers a `drz` fault at
   **+0.383**; `drz` answers `dry` at **−0.424**; their own diagonals are ≈ 0.01. Off-diagonal
   mass is **59%** of `|M|`. A per-axis law would drive `dry` from `drz`'s error *with the
   wrong sign* — positive feedback, guaranteed divergence. This is not a subtlety; it is the
   difference between a law that works and one that destroys the policy.
3. **It inverts.** cond(M) = **3.6**, so `M⁻¹` is numerically safe and the problem is
   well-posed.

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

| arm | success | note |
|---|---|---|
| nominal (no fault) | 99% | ceiling |
| **oracle** — computed edit on `action_out_proj/bias` | **100%** | knows the answer analytically |
| frozen, faulted | 7/15 = **47%** | floor |
| **adaptive** | **13/15 = 87%** | **+40 points**, Fisher p = 0.0502 |

The adaptive arm recovers **~77% of the available headroom** using only proprioception.
p = 0.0502 is *at* the threshold: suggestive, not significant, and one episode either way
flips it. n ≈ 40 would settle it.

**Fault estimates**, mean over 15 episodes, true value 0.05 on every dimension:

```
f̂      [0.070  0.038  0.082  0.037  0.017  0.024]
ratio  [ 1.40   0.76   1.65   0.74   0.34   0.49]
```

**Sign correct on 6 of 6.** Translation and `drx` land within about ±65%. `dry` and `drz`
undershoot badly (0.34, 0.49) — exactly the pair whose plant model does not fit (R² 0.11,
0.17) and whose sensitivity block is the swapped one. The weakness appears precisely where
§2.2 and §2.4 predict it.

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

- **p = 0.0502 at n = 15.** Suggestive. Not significant.
- **Rotation is poorly identified** (R² 0.11–0.49), and `dry`/`drz` estimates undershoot by
  half. The likely cause is the error metric: orientation deltas are taken as differences of
  axis-angle vectors, which is not a valid metric on SO(3). Fixing that is the first thing to
  try next.
- **One fault type, one suite.** A constant additive offset is the easiest case — it is
  constant, so a constant correction cancels it. A state-dependent fault (a gain error) needs
  a regressor, not a single vector.
- **`M` is identified open loop** and assumed valid in closed loop, where the policy
  compensates and the arm visits different states. Not verified.
- The oracle reaches 100% and the adaptive law 87%, so ~13 points remain on the table —
  consistent with the undershoot in §3.
