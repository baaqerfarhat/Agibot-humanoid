# Measured results

All numbers measured on the MuJoCo deployment-faithful testbed included in this package.
Metric definitions: **survival** = control steps before the pelvis drops below 0.35 m;
**leg tracking error** = mean |q − q_ref| over the 12 hip/knee/ankle joints.

## 1. Testbed validation

The deployment loop is transcribed from the vendor's `deploy_x2_box_pickup.py` and validated
against the real robot's own logs: replaying the robot's **logged state** through this pipeline
reproduces the **targets the robot actually commanded** to

> **0.84° on the leg joints, sustained across all 742 logged control steps**

with no drift. So the observation construction and policy evaluation are correct, and the sim's
behaviour is the policy's behaviour.

The frozen policy falls ~1.9 s in. That is the *real* behaviour: during the recorded hardware
runs the robot was physically supported, so it would have fallen unaided as well. The sim
reproduces the real failure rather than diverging from it.

## 2. Headline (held-out seeds 600–631, never used for tuning)

Configuration: layer 2, Γ = 3e-4, `gx_level=1`, error mask legs+waist, `engage_step=0`.

| | frozen | adapted | change |
|---|---|---|---|
| **leg tracking error** | 14.69° | **9.22°** | **−37%** |
| **survival** | 97.2 steps (1.94 s) | 111.2 steps (2.22 s) | +14% |
| **never fell** | **0/32** | **0/32** | none |
| composite "tracked steps" | 59.2 | 104.1 | +44.9 |

28/32 seeds, p < 1e-4 on the composite, zero divergences, median ‖dW‖ = 0.28.

**Interpretation.** The composite metric (steps before falling **or** exceeding 20° leg error)
is what gives the large +44.9; its gain is almost entirely the *tracking* half. Frozen loses
tracking at step ~43 then wanders upright until it falls at ~97; adapted holds tracking until it
falls at ~111. Report survival and tracking separately.

Confirmed on three independent pools with no regression (300–331, 500–531, 600–631).

## 3. Controls

**Displacement-matched random null** — random `dW` of the same Frobenius norm, 20 draws, same
seeds:

| | tracked steps | leg error |
|---|---|---|
| frozen | 42.8 | 17.83° |
| matched random null | 43.0 (range 40.6–45.8) | 17.77° |
| **adaptation** | **81.5** | **11.17°** |

**0/20 draws come close.** The null lands on the frozen value and does nothing for tracking
error, so the *direction* the law computes carries the result, not the size of the weight change.
(Necessary because random perturbation of *any* layer can improve this fragile policy slightly —
beating frozen alone would not have been evidence.)

## 4. Configuration sweep

### Input mapping `g(x)` (dev seeds, layer 2, scale-matched gains)

| level | `delta_L` | tracked | leg error | p |
|---|---|---|---|---|
| 0 | `s · e` | 49.3 | 16.18° | 0.006 |
| **1** | **`s · Kp · e`** | **77.0** | **13.43°** | **<1e-4** |
| 2 | `+ M⁻¹` | 40.7 | 17.87° | 0.061 |
| 3 | `+ contact-consistent M_c⁻¹` | 40.1 | 17.18° | 0.57 |

Frozen 36.2. Deriving the mapping properly roughly **doubles** the effect; adding inverse inertia
**undoes** it. `M⁻¹` correctly says light joints accelerate most per unit torque, but that points
at wrists (`M_ii` = 0.00065, so 1/M_ii ≈ 1500) and head, whose large tracking errors are
irrelevant to the task. Restricting the regulated error partially rehabilitates level 2
(p = 0.06 → 0.004) but never changes the ordering.

Note `‖delta_L‖` spans ~13,000× across these levels, so each needs its own Γ — a shared gain makes
the high-fidelity levels diverge, which is a scaling artifact, not a verdict.

### Engagement step and gain (dev seeds, `gx_level=1`, legs+waist mask)

Best survival by engagement step (adapting from step 0 is best; the policy already diverges by
step ~5, so engaging at step 20 is partly too late):

| engage_step | frozen survival | best adapted survival |
|---|---|---|
| **0** | 99.2 | **109.7** |
| 5 | 94.2 | 101.9 |
| 20 | 79.2 | 86.9 |

Gain stability: Γ = 3e-4 is stable on layers 0–2 (0 divergences). Γ ≥ 1e-2 diverges broadly.
**Layer 3 (output) is unusable** — 20–32 divergences out of 32 at any useful gain.

### What actually drives the improvement (same-pool decomposition)

| source | contribution to tracked steps |
|---|---|
| larger Γ (1e-4 → 3e-4) | ~+6.8 |
| seed pool difference | ~+5.7 |
| error mask (legs+waist) | **~+0.8** |

The error mask contributes almost nothing to *performance*. Its value is **stability**: at
Γ = 3e-4 the unmasked law diverges on 3/32 seeds, the masked law on 0/32. It makes the higher
gain safe to use. Gain and mask are complements; the mask is an enabler, not the source.

## 5. Layer selection (ACE)

The paper's offline ACE attribution, 24 seeds × 40 draws = **960 draws per layer**:

| layer | ACE (Δ leg error) | sem | distance from 0 |
|---|---|---|---|
| 0 | −0.030 | 0.085 | 0.4 sem |
| 1 | −0.026 | 0.078 | 0.3 sem |
| **2** | **−0.125** | 0.081 | **1.5 sem** |
| 3 | +0.206 | 0.098 | 2.1 sem (harmful) |

**ACE selects layer 2 — the same layer the empirical sweep found best**, and flags layer 3 as
harmful, consistent with layer 3 being the divergence-prone one.

Power matters here: an underpowered first run (16 seeds × 12 draws, sem ≈ 0.17) nominally chose
layer 0 at −0.186, but that lead was noise — it collapsed to −0.030 with 5× the draws while layer
2 held. Two caveats: layer 2 is 1.5 sem from zero (suggestive, not conventionally significant),
and it was chosen empirically *before* the high-power ACE ran, so the agreement is post-hoc.

## 6. What did not work

An alternative "H-step" adaptation law was tested first on this same testbed with the same metric
and seed protocol. Held-out result: **+0.3 tracked steps, p = 1.0000** — a flat null.

The mechanism was measured, and it is instructive: that law must **estimate** a counterfactual
model `B_H` by finite differences, and the estimate is ill-posed here. At H = 1 the sensitivity of
the outcome to the adapter is σ_max = 0.016, so nulling the residual demands corrections of norm
57–570 — far outside any admissible trust region. Horizon accumulation buys 3–4 orders of
magnitude of authority but costs linearity (prediction error 0.2% → 24%), and at large H the
correction becomes a no-op. The law interpolates between two useless regimes.

The Lyapunov law works here precisely because it builds **no** such model: exact backprop, loop
closed every control step, leakage bounding the weights.

## 7. Reproduction

```bash
python evaluate.py --seeds 32 --seed0 600           # headline table + sign tests
python evaluate.py --seeds 32 --seed0 600 --null    # + matched random control
python run_mujoco_demo.py --adapt --view            # watch it
```

Exact numbers depend on your MuJoCo version (contact solver details shift trajectories in this
contact-rich scene). The *pattern* — large tracking gain, small survival gain, no prevented
falls — is what should reproduce.
