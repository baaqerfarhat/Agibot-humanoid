# Review of `vla_adaptation_report.pdf`

**Received 2026-09-01.** Reproduced verbatim. This is a review of the report in this
folder; it is not a record of work done. Action items are listed in the tables and
in the "Minimum experimental program" section.

---

I reviewed [vla_adaptation_report.pdf](vla_adaptation_report.pdf). The 47% → 93% result is a strong pilot, but the theory needs to be rebuilt rather than lightly revised. At present, Section 6 is an empirically tuned disturbance estimator—not yet contraction-based adaptive control.

The winning ICLR story should be:

> A frozen VLA is repaired within the first faulted episode by adapting a low-dimensional physical action interface from proprioceptive prediction errors—without task reward, fault-conditioned training, reference trajectories, or backpropagation through the VLA—with explicit recoverability and trajectory-error bounds.

The six-parameter result should be the motivating case study, not the whole contribution.

## Problems that must be fixed immediately

| Current issue                                                            | Required correction                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Equation (2), \(r_t\approx Mf\), contradicts the \(K=6\) FIR model       | Use the full history-dependent regressor. For constant offsets after the transient, \(M_{\rm off}=\sum_{\ell=0}^{K}H_\ell\).                                                                                                                          |
| Equation (4) is biased                                                   | With \(r=Mf\), it converges to \(f/(1+\|Mf\|^2/\rho^2)\), not \(f\). Normalize the innovation, not \(M^{-1}r\) alone.                                                                                                                                 |
| Equation (4) uses full \(M^{-1}\), but Section 7 blames a “diagonal law” | Determine what the code actually implements and make the equation, implementation, and interpretation identical.                                                                                                                                      |
| Gain-fault law ignores FIR history                                       | Its regressor must contain all delayed commands.                                                                                                                                                                                                      |
| Gain compensation is not defined correctly                               | Use an inverse-gain compensator, not a first-order subtraction.                                                                                                                                                                                       |
| No contraction condition appears                                         | Add a contraction condition for the healthy VLA–robot closed loop and derive a trajectory-recovery bound.                                                                                                                                             |
| \(+0.05\) is injected at the Cartesian VLA action interface              | Until the fault is injected below the OSC or in joint dynamics, call it a Cartesian action-interface bias, not a demonstrated actuator hardware fault.                                                                                                |
| Native output-bias edit is treated as direct action addition             | Because π0.5 generates actions through flow matching, changing `action_out_proj/bias` is not automatically equivalent to adding \(c_t\) to the final action. This mapping must be derived or measured. [π0.5 paper](https://arxiv.org/abs/2504.16054) |
| “77% of headroom”                                                        | It is \((93.3-46.7)/(99-46.7)=89.1\%\), or 87.5% relative to the 100% oracle.                                                                                                                                                                         |
| “Four of six estimates within ±20%”                                      | Only three are: 0.96, 0.82, and 0.94.                                                                                                                                                                                                                 |
| Six parameters presented as minimal                                      | The tested \(0.05\mathbf1\) fault is one fault direction. If that direction is known, one scalar is sufficient. Six parameters become necessary only for arbitrary independent six-axis offsets.                                                      |

Also, \(14/15\) and \(7/15\) have very wide confidence intervals. Fisher’s \(p=0.0142\) is correct only for independent arms. If identical tasks, initial conditions, and policy seeds were paired, use exact McNemar or a paired randomization test.

## Recommended mathematical formulation

Let the frozen VLA produce normalized actions

$$
a_k=\pi_\omega(o_k,\ell),\qquad \omega\ \text{fixed},
$$

and let physical action conversion be

$$
u_{\pi,k}=Da_k+b,\qquad
D=\frac12\operatorname{diag}(q_{99}-q_{01}).
$$

Perform estimation in physical action coordinates. Write the realized input as

$$
u_k^{\mathrm{act}}
=
u_{\pi,k}
+
\Phi_k(\theta^\star-\hat\theta_k)
+
d_k^\perp ,
$$

where \(\theta^\star\) is the unknown fault, \(\hat\theta_k\) is the online estimate, and \(d_k^\perp\) contains unmatched faults, saturation, and modeling error.

For an additive offset, \(\Phi_k=I\). For diagonal gain and offset, use

$$
u_k^{\mathrm{act}}=G^\star v_k+f^\star,
$$

with the exact certainty-equivalent command

$$
\boxed{
v_k=\hat G_k^{-1}(u_{\pi,k}-\hat f_k).
}
$$

For \(g=0.5\), the first-order correction \(c=-\hat\beta\odot a\) undercompensates substantially and may explain why the gain estimate improves without improving success.

### Correct residual regression

Write the healthy FIR model as

$$
\hat y_k^0
=
b_P+\sum_{\ell=0}^{K}\hat S_\ell v_{k-\ell}.
$$

Then

$$
r_k=y_k-\hat y_k^0=H_k\theta^\star+\nu_k .
$$

For combined offset and diagonal gain,

$$
\theta^\star=
\begin{bmatrix}f^\star\\ \beta^\star\end{bmatrix},
\qquad
H_k=
\begin{bmatrix}
\sum_{\ell=0}^{K}\hat S_\ell
&
\sum_{\ell=0}^{K}
\hat S_\ell\operatorname{diag}(v_{k-\ell})
\end{bmatrix}.
$$

This should replace the present Equations (1), (2), and the gain equation. The current \(M\) is only the constant-offset block.

Crucially, \(H_k\) should be obtained from healthy command-response calibration, not by injecting the target fault family. Otherwise the “unknown fault” claim becomes vulnerable.

### Correct adaptive law

Whiten using the nominal residual covariance:

$$
\bar r_k=R_k^{-1/2}r_k,\qquad
\bar H_k=R_k^{-1/2}H_k,
$$

and define the innovation

$$
e_k=\bar r_k-\bar H_k\hat\theta_k.
$$

A defensible projected normalized adaptive law is

$$
\boxed{
\hat\theta_{k+1}
=
\Pi_{\Theta}^{\Gamma^{-1}}
\left[
\hat\theta_k+
\frac{
\alpha\Gamma\bar H_k^\top
\mathcal D_{\delta_k}(e_k)
}{
\mu+\|\bar H_k\Gamma^{1/2}\|_2^2
}
\right],
\qquad 0<\alpha<2 .
}
$$

Here \(\mathcal D_{\delta}\) is a deadzone applied to the whitened innovation, and \(\Pi_\Theta\) enforces physical offset and gain bounds.

This fixes several problems simultaneously:

* The correct parameter remains the fixed point.
* No direct \(M^{-1}\) noise amplification is needed.
* Full MIMO and rectangular sensitivities are allowed.
* The threshold is based on measured model uncertainty.
* If the fault disappears, \(e_k=-H_k\hat\theta_k\), so the estimate returns toward zero. A deadzone on raw \(r_k\) could instead freeze a harmful correction.

For a constant fault, avoid leakage; leakage creates steady-state bias. Use a sliding window or forgetting-factor RLS only for genuinely time-varying faults.

## The contraction-theory spine

The paper should contain three separate results.

### 1. Adapter representability and minimum dimension

For candidate adaptation site \(\ell\), let \(J_{\ell,k}\) be its parameter-to-final-action Jacobian. Stack the contraction-weighted physical effects:

$$
\mathcal D_T
=
\operatorname{col}_k
\left(
\mathcal P_{k+1}^{1/2}B_kD_k
\right),
\qquad
\mathcal B_{\ell,T}
=
\operatorname{col}_k
\left(
\mathcal P_{k+1}^{1/2}B_kJ_{\ell,k}
\right).
$$

Exact local compensation is possible iff

$$
\boxed{
\operatorname{Range}(\mathcal D_T)
\subseteq
\operatorname{Range}(\mathcal B_{\ell,T}).
}
$$

The minimum adapter dimension satisfies

$$
p_{\min}\ge \operatorname{rank}(\mathcal D_T).
$$

This replaces the underpowered random site screen. Site selection should depend on:

* unmatched projection error;
* smallest singular value/conditioning;
* number of updated parameters;
* contraction-weighted effect on robot motion.

I strongly recommend appending an explicit six-parameter adapter after final action generation and unnormalization. That makes the mapping exact and architecture-independent. If the native π0.5 bias is retained, \(J_{\ell,k}\) and its nonlinear remainder through flow integration must be measured.

### 2. Adaptive-estimation theorem

Require interval excitation:

$$
\sum_{j=k}^{k+T-1}
H_j^\top R_j^{-1}H_j
\succeq \alpha_{\mathrm{PE}}I.
$$

Then the projected adaptive law gives

$$
\|\tilde\theta_k\|
\le
C\rho_\theta^{k-k_0}\|\tilde\theta_{k_0}\|
+
C_\nu\bar\nu+
C_v\bar v ,
$$

where \(\tilde\theta=\theta^\star-\hat\theta\), \(\bar\nu\) bounds predictor uncertainty, and \(\bar v\) bounds fault variation.

Without excitation, do not claim parameter convergence. Only the excited action-mismatch component \(H_k\tilde\theta_k\) can be guaranteed to shrink. This directly explains the current gain-fault problem: the rotation commands do not provide enough excitation.

### 3. Contraction-to-recovery theorem

Define the healthy composite map

$$
x_{k+1}^{h}=F_k^0(x_k^h)
$$

for the same frozen VLA on healthy hardware. Assume local contraction on a task rollout tube:

$$
A_k^\top\mathcal P_{k+1}A_k
\preceq
\rho_x^2\mathcal P_k,
\qquad \rho_x<1.
$$

Assume input mismatch obeys

$$
d_{\mathcal P}
\bigl(F_k(x,u+q),F_k(x,u)\bigr)
\le L_u\|q\|+\bar d .
$$

If the estimator satisfies

$$
\|\tilde\theta_k\|
\le
\rho_\theta^k\|\tilde\theta_0\|+\bar\theta ,
$$

then

$$
\boxed{
\begin{aligned}
d_{\mathcal P,k}(x_k,x_k^h)
\le {}&
\rho_x^k d_{\mathcal P,0}\\
&+
L_u\bar\Phi\|\tilde\theta_0\|
C_k(\rho_x,\rho_\theta)\\
&+
\frac{
L_u\bar\Phi\bar\theta+\bar d
}{
1-\rho_x
}
(1-\rho_x^k),
\end{aligned}
}
$$

where

$$
C_k(a,b)=
\begin{cases}
\dfrac{a^k-b^k}{a-b},&a\neq b,\\[2mm]
ka^{k-1},&a=b.
\end{cases}
$$

Therefore:

* Exact matching, exact modeling, and sufficient excitation give exponential recovery to healthy behavior.
* Noise, plant mismatch, saturation, and unmatched faults give an explicit ultimate recovery tube.
* The recovery rate is governed by \(\max\{\rho_x,\rho_\theta\}\).

This follows established adaptive contraction ideas, but the paper’s novelty would be connecting physical-consistency adaptation of a frozen VLA to an explicit behavior-recovery bound. Relevant foundations include [adaptive neural contraction metrics](https://arxiv.org/abs/2103.02987) and [adaptive robust control contraction metrics](https://arxiv.org/abs/2310.13655).

Important limitation: contraction must apply to the composite VLA–OSC–robot map, not merely the OSC. Otherwise the theorem should be explicitly limited to command-response recovery under a shared action sequence. LIBERO is hybrid/contact-rich, so use a local rollout-tube or mode-wise result rather than claiming global contraction.

Contraction does not automatically prove binary task success. A task-success corollary requires the healthy trajectory’s distance from the failure/contact-mode boundary to exceed the recovery bound.

## Minimum experimental program for ICLR

The closest current competitor, J-PARC, already evaluates frozen π0.5 and OpenVLA-OFT across all four LIBERO suites and includes WidowX hardware. Its opening for your work is that it uses offline learned residual calibration, reference rollouts, and teacher targets, whereas your method can adapt physically meaningful parameters online without those requirements. [J-PARC](https://arxiv.org/abs/2606.10501)

To make that distinction convincing:

1. **Faults**

   * Random sparse and dense Cartesian offsets with independent signs and magnitudes.
   * Diagonal gain/loss-of-effectiveness.
   * Combined affine gain-plus-offset.
   * One dynamic or joint-level fault: friction, delay, saturation, deadzone, or joint efficiency.
   * Random mid-episode onset, removal, switching, and simultaneous faults.
   * Adaptation continuously active under healthy operation.

2. **Breadth**

   * π0.5 on all four LIBERO suites.
   * At least one second backbone, preferably OpenVLA-OFT.
   * Predictor calibration on disjoint healthy tasks.
   * At least one real robot with genuine payload, controller-gain, friction, or joint fault—not only a software action offset.

3. **Baselines**

   * Frozen and oracle.
   * Current EMA/\(M^{-1}\) law.
   * RLS/Kalman disturbance estimator.
   * Integral/disturbance observer.
   * External output correction versus native bias edit.
   * J-PARC.
   * Reward-based CEM/CMA-ES.
   * Small last-layer/LoRA online adaptation.

4. **Statistics**

   * Identical paired initial states, task instances, and policy seeds.
   * Prefer 50 rollouts per task/condition to match the contemporary comparison; at minimum, 20–30.
   * Task-stratified paired confidence intervals or hierarchical bootstrap.
   * Separate calibration, validation, and untouched test faults.
   * Freeze the implementation and hyperparameters after the SO(3) correction.

5. **Theory validation**

   * Predicted versus measured parameter convergence rate.
   * Predicted versus measured contraction recovery tube.
   * Success versus smallest singular value of the excitation Gramian.
   * Maximum recoverable severity versus contraction margin/control authority.
   * Adapter-rank versus residual error and success.
   * Failure cases correctly predicted by rank loss, saturation, or loss of contraction.

The strongest figure would be a recoverability phase diagram over fault severity and sensitivity conditioning, showing where the theorem predicts recovery and where experiments actually recover.

## Recommended paper structure

1. Problem formulation: frozen VLA and physical execution mismatch.
2. Low-dimensional adaptive action interface.
3. Healthy one-step predictor and \(SE(3)\) residual.
4. Representability, estimation, and contraction-recovery theory.
5. Multi-fault simulation evaluation.
6. Hardware and failure-boundary evaluation.
7. Limitations.

Move the chronological debugging, site screen, long negative-result narrative, artifact paths, and video description out of the main paper. Keep the SO(3) bug only as a clean ablation comparing chart subtraction against the correct Lie-group residual.

My recommended title is:

**Frozen Yet Adaptive: Contraction-Guided Online Repair of Vision–Language–Action Policies**

The decisive next step is to lock the corrected FIR regression and adaptive law, rerun the offset experiment on untouched paired trials, and then fix the gain experiment using the history-dependent regressor and exact inverse-gain compensation. If gain plus at least one genuine joint/dynamic fault cannot be recovered, the current result should remain a supporting case study rather than the central ICLR contribution.
