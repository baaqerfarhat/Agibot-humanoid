# The ACC-2026 ACE principle is the thing that survived — carry it into the online stage

18 Aug 2026. After re-reading Taheri, Chung, Hadaegh, *Closing the Loop Inside Neural
Networks: Causality-Guided Layer Adaptation for Fault Recovery Control*, ACC 2026
(`5 - Final - After Acceptance/ACC26_main (1).pdf`).

## 1. What the paper actually does (for citation accuracy)

- **Offline layer selection by causal intervention, not gradients.** The DNN controller is
  recast as an SCM; layer importance is `ACE_ℓ = E[‖ě‖ | do(W_ℓ+Δ_ℓ)] − E[‖ě‖ | W_ℓ]`,
  with `Δ_ℓ ~ N(0, ρ_ℓ²I)`, estimated by **Monte-Carlo sampling of interventions, each
  scored by a closed-loop rollout** under held-out fault/disturbance scenarios (§III-B).
  (It is MC sampling of do-interventions — no tree search in the published version.)
- **Online adaptation by a Lyapunov gradient law** on the selected layer:
  `Ẇ_ℓ* = Γ δ_ℓ* z_ℓ*−1ᵀ − γW_ℓ*`, `δ_L = g(x)ᵀPe` (eq. 4) — **exactly the `acc` arm
  deployed in this project.**
- Table I's stated reason for rejecting Jacobian/gradient attribution: it *"captures only
  short-term correlations; neglects closed-loop dynamics and long-horizon fault effects."*
- Case study: 6-layer net, spacecraft attitude, ACE selects **layer 4** (an interior layer)
  over the last layer; margins small (RMSE 0.00882 vs 0.00890; ACE −0.004387 vs −0.004256).

## 2. What the walker measured, restated in the paper's own vocabulary

The project's negative results are a *proof, with mechanism, that Table I's critique applies
to the online stage too* — to eq. 4 itself, not only to layer selection:

| Table I says gradient attribution fails because… | …and the walker measured exactly that |
|---|---|
| "captures only short-term correlations" | the tq05 fix moves **no observable by > 0.4σ** at one step; its entire effect is a horizon-scale gait reshaping (`what_the_fix_does`) |
| "neglects closed-loop dynamics" | the residual acts through `S=(I+GK)⁻¹`; the base policy rejects exactly the rows the surrogate is built on — velocity-row sign flips across 3 conditions |
| "neglects long-horizon fault effects" | no ε-plateau beyond h≈10: the horizon cost is not differentiable in the parameters; 1600× eta sweep never positive while a constant recovers +100% |

So the two stages of the ACC framework have opposite fates on this system: the
**intervention-scored stage survives** (it is the only thing that ever produced a held-out,
control-beating positive) and the **instantaneous-gradient stage is structurally dead**
(condition D, certified at chance agreement with the fix).

## 3. The constructive consequence — the NeurIPS method IS ACE carried online

**Adaptation = do-intervention search scored on the realised closed-loop metric.**
The confirmed tq05 result (`PREREG_CONST_TQ05.md`, +1.70 m held-out) is precisely
MC-do on `mlp.6`'s bias: sample `do(b₆ + r)`, score by whole-episode outcome, refine.
`PREREG_ONLINE_TQ05.md` (running) makes it sequential — one intervention per fresh
deployment episode. No gradient anywhere, which also removes the ONNX/VLA blocker:
do-interventions need only forward passes.

**Layer selection = the ACE screen, and we have been computing it by accident.** The paper's
estimator is the *mean effect of N random interventions on layer ℓ*. The "norm-matched
random constant" control column IS that estimator for `mlp.6`, per fault:

| fault | mean effect of random b₆ intervention | CEM recovery |
|---|---|---|
| tq05 | **+0.26…+0.58 m (ACE negative)** | **44.9–68%** |
| s2r_moderate | **−0.30 m (ACE positive)** | 0.9% |
| kp05 | −0.02 | 0.7% |
| tq04 | +0.05 | 1.2% |

The sign of the ACE estimate predicts where the searched fix exists — on the cells measured
so far, perfectly. That is a deployable, gradient-free layer/cell selection stage inherited
directly from the ACC paper, and it costs N episodes.

## 4. The 2×2 correspondence experiment is layer-level ACE with a mechanism

`layer_fault_gate.py` / the coming 2×2 search is `ACE_ℓ` for ℓ ∈ {mlp.0 bias, mlp.6 bias}
under faults {obs_bias, joint_offset}, with the *prediction* that ACE selects the layer whose
coordinates contain the fault's inverse (backend docstring: obs_bias exactly cancellable at
the first layer; joint_offset only at the last). The ACC paper found an interior layer wins
on the spacecraft but could not say *why* (ACE values nearly tied); the walker version makes
the mechanism explicit and the separations order-of-magnitude rather than 1%.

## 5. What to pre-register on top of the 2×2 (the ACE arm)

For each (layer, fault) cell, N=8 norm-matched random interventions → mean effect =
`ACE_hat`. **Prediction, stated before the run: `ACE_hat` is negative on the two diagonal
cells and non-negative off-diagonal, and its sign predicts whether the CEM search in that
cell recovers ≥25% of the analytic ceiling.** If ACE fails to predict, that is reportable
against §3's screen; if it holds, the paper has a two-stage, fully gradient-free method —
ACE screen → intervention search — that is the ACC architecture with the broken middle
(eq. 4) removed and each surviving stage validated at scale.

## 6. One honest caveat to carry

ACE with `Δ ~ N(0, ρ²I)` measures the *class* effect (any perturbation of that layer), not
the *direction* effect — the same distinction the 20-draw control forced on tq05 (p=0.095).
That is a feature for **selection** (it is exactly what makes it cheap) and a limitation for
**credit**: where ACE is negative, part of the recovery belongs to the class itself. Both
numbers must be reported per cell: `ACE_hat` (class) and `CEM − mean(random)` (direction).
