# Source equations and an independent adaptation formulation

This memo reads the locally supplied author PDFs and TeX of MAGIC-VFM, Neural-Fly, and HMAC. It separates their original formulations from a proposed first-order controller. **A learned dynamics representation with composite coefficient adaptation is a sound direction to investigate; none of these papers establishes that composite adaptation will rank first on every manipulation benchmark.** Historical FIR/observer experiments are not evaluations of this new method.

All page references below are **one-based pages of the exact local PDFs**, not journal pagination. [Source metadata and file hashes](adaptation_sources_metadata.json) identify the versions and inspected equations. Publisher-deposited bibliographic exports already preserved in `paper/citation_sources/` establish publication metadata. Author-version equations may differ from a publisher's final typeset version; inconsistencies below concern the local versions actually read.

| Source | Published record | Local version and principal locations |
|---|---|---|
| MAGIC-VFM, Lupu et al. | IEEE Transactions on Robotics **41, 180–199 (2025)**; [DOI](https://doi.org/10.1109/TRO.2024.3475212) | [arXiv:2407.12304v2](https://arxiv.org/abs/2407.12304v2), 20 pages. Dynamics pp.3–4; learning pp.4–5; tracked-vehicle control pp.7–9. |
| Neural-Fly, O'Connell et al. | Science Robotics **7(66), eabm6597 (2022)**; [DOI](https://doi.org/10.1126/scirobotics.abm6597) | [arXiv:2205.06908v2](https://arxiv.org/abs/2205.06908v2), 41 pages, including supplement. Dynamics p.16; learning pp.17–19; control pp.20–22; implementation/proof pp.34–37. |
| HMAC, Xie et al. | **ICRA 2024, 18309–18315**; [DOI](https://doi.org/10.1109/ICRA57147.2024.10611562) | [arXiv:2311.12367v2](https://arxiv.org/abs/2311.12367v2), 7 pages. Dynamics p.2; hierarchical learning pp.3–4; control p.4. |

## What the three methods learn

| Method | Offline learned object and objective | Online adapted object | Important distinction |
|---|---|---|---|
| Neural-Fly | Shared state-dependent force basis `phi(q,qdot)`, trained through least-squares coefficient fitting and a domain-adversarial objective | Linear force coefficients `a`; covariance-like gain `P` | The basis is encouraged to be wind-invariant; online coefficients carry condition dependence. |
| HMAC | Two additive state-dependent force bases: manageable-disturbance basis via DAIML, latent-disturbance basis via smoothed streaming meta-learning; training alternates | Both coefficient blocks; full covariance-like gain with separate intended timescales | The hierarchy is in the learned disturbance decomposition and offline training. It is not a new position controller with an unrelated estimator. |
| MAGIC-VFM | Small basis network from state and pretrained visual features, optimized through regularized coefficient fitting over short trajectory windows | Linear coefficients of a disturbance/effectiveness model | In the tracked-vehicle controller, the disturbance multiplies the command, so the controller inverts an estimated input matrix. |

All three use learned nonlinear features with a small online linear parameter vector. They do **not** simply sum the current tracking residual into a command. They still use a dynamics residual as a measurement: measured physical evolution minus a nominal model driven by the applied input. The distinction is the meaning of this measurement, its regressor, and its role in a jointly designed controller and estimator.

## Neural-Fly: additive force model and composite Riccati adaptation

The original dynamics and representation are (p.16, Eqs.1–2)

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)=u+f(q,\dot q,w),
\qquad f=\phi(q,\dot q)a(w)+d.
\]

Here `u` is a generalized force, `M` is a symmetric positive-definite inertia matrix, `w` is the environmental condition, and `d` is representation error. The force measurement is `y=f+epsilon`, obtained using measured state derivatives and the nominal dynamics (p.17, Eq.3). It is not the raw joint-position tracking error.

With the paper's error convention (p.21, Eq.10),

\[
\tilde q=q-q_d,\quad
\dot q_r=\dot q_d-\Lambda\tilde q,\quad
s=\dot q-\dot q_r=\dot{\tilde q}+\Lambda\tilde q,
\]

its controller and adaptation laws are (p.20, Eqs.7–9)

\[
u=M\ddot q_r+C\dot q_r+g-Ks-\phi\hat a,
\]
\[
\dot{\hat a}=-\lambda\hat a
 +P\phi^\top R^{-1}(y-\phi\hat a)
 +P\phi^\top s,
\qquad
\dot P=-2\lambda P+Q-P\phi^\top R^{-1}\phi P.
\]

The **positive tracking term** is determined by the disturbance and error signs. With `tilde a = hat a - a`, the error dynamics are

\[
M\dot s+(C+K)s=-\phi\tilde a+d
\quad\text{(p.34, Eq.20).}
\]

In a Lyapunov function containing `s^T M s` and `tilde a^T P^{-1} tilde a`, the tracking term cancels the state–parameter cross term. Removing it gives a Kalman-Bucy-style parameter estimator, but does not inherit this particular closed-loop cancellation proof. `Q`, `R`, and `lambda` have process-variation, measurement-error, and mean-reversion interpretations; their Kalman interpretation does not turn a non-Gaussian physical disturbance into a known stochastic model.

The learning objective (p.18, Eqs.4–6) combines force prediction and a discriminator for the training environment index:

\[
\max_h\min_{\phi,a_1,\ldots,a_K}
\sum_{k,i}\left[
\|y_{ki}-\phi(x_{ki})a_k\|^2
-\alpha\,\mathrm{CE}(h(\phi(x_{ki})),k)\right].
\]

Algorithm 1 (p.19) fits coefficients by least squares on an adaptation batch and updates the feature network on another training batch. It also bounds fitted coefficients and uses spectral normalization. The adversarial term discourages the state feature network from encoding the training environment merely through correlated state distributions. The final feature network remains fixed during deployment; the linear coefficients adapt. The existence result for analytic functions (supplement S2) does not guarantee an arbitrarily small fixed feature dimension for any disturbance.

**Actual features differ from the compact dynamics notation.** In the quadrotor implementation (p.23), the network receives 11 values: velocity (3), attitude quaternion (4), and rotor PWM commands (4). Four shared scalar features are repeated across three force directions, giving 12 adaptive coefficients. The controller uses the previous time step's PWM to avoid an implicit dependence of the current correction on its own network input. Thus `phi(q,qdot)` describes the paper's abstract formulation, not the complete deployed feature vector; command history can be a legitimate learned-feature input when its timing is specified.

**Guarantee and scope.** Theorem 1 (p.22, Eq.12) gives exponential approach to a tracking-error ball depending on representation error, measurement noise, parameter variation, and the leakage bias `lambda ||a||`. The detailed argument (pp.35–37, Eqs.22–39, Theorem 4) uses uniformly bounded, positive-definite state/parameter metrics, bounded features, positive gains, and the mechanical identity `Mdot - 2C` skew-symmetric. Robust bounded tracking without persistent excitation is not exact identification of every coefficient without excitation. The implemented drone computes a desired force and then a desired attitude/thrust for an inner flight controller (p.22, Eq.13); this is not direct unconstrained control of every physical degree of freedom.

**Discrete implementation requires an independent sign and units check.** Supplement S4 (p.34, Eqs.15–19) recommends propagation and a Joseph-form covariance update because naive Euler integration can lose positive definiteness. However, the displayed Eq.18 prints a **negative** tracking increment, inconsistent with Eq.8 and the above error convention; it also does not display an explicit time-step factor on that increment. Do not copy it as the discretization of Eq.8. Specify whether measurement noise is a continuous-time intensity or a per-sample covariance, and derive a discrete update with an explicit time step and a positive tracking increment. This memo does not infer what the authors' running code actually used from that display.

## HMAC: two learned disturbance components

HMAC retains the same mechanical dynamics (p.2, Eq.1) and assumes the approximate additive decomposition (Eq.2)

\[
f(q,\dot q,w)\approx
\phi_m(q,\dot q)a_m(w_m)
+\phi_r(q,\dot q)a_r(w_r).
\]

`w_m` denotes conditions that can be managed or labeled during training, such as a commanded wind-fan setting. `w_r` denotes other latent variation. The decomposition is a modeling assumption; it is not automatically an identifiable separation of physical causes.

Figure 3 and Section III-A (p.3) alternate two learning modules. Training starts with `y_m=y`; fitted manageable effects leave the latent target `y_r=y-y_m`, and fitted latent effects update the manageable target `y_m=y-y_r`. The figure uses the same symbols for evolving predicted components and their training targets, so an implementation should name these separately. DAIML fits the manageable feature basis with the domain-adversarial objective (p.3, Eq.3). Smoothed Streaming Meta-Learning fits the latent basis using sequential windows, approximately constant local coefficients, a support/query split, and a penalty on network-weight movement (pp.3–4, Eqs.4–5, Algorithm 1):

\[
a_r^*=\arg\min_a\sum_{i\in\mathcal S}
\|y_{r,i}-\phi_r(x_i)a\|^2,
\]
\[
\mathcal L_r=\sum_{i\in\mathcal Q}
\|y_{r,i}-\phi_r(x_i)a_r^*\|^2
+(\vartheta_r-\vartheta_r^{\rm previous})^\top W(\vartheta_r-\vartheta_r^{\rm previous}).
\]

Here `vartheta_r` denotes latent-basis network weights, distinct from environmental `w_r`. Both networks are trained offline. A streaming training algorithm is not evidence that the deployed system updates all network weights online. As in Neural-Fly, the actual quadrotor feature input augments velocity and quaternion with four rotor inputs (11 values in total); HMAC learns three shared manageable features and two shared latent features, repeated across force directions (p.6, neural-network implementation section).

The online controller is Neural-Fly's controller with

\[
\phi=[\phi_m\ \phi_r],\qquad
\hat a=[\hat a_m^\top\ \hat a_r^\top]^\top
\quad\text{(p.4, Eqs.6–10).}
\]

HMAC prints

\[
\dot{\hat a}=-\Lambda\hat a
+P\phi^\top R^{-1}(y-\phi\hat a)+P\phi^\top s,
\qquad
\dot P=-2\Lambda P+Q-P\phi^\top R^{-1}\phi P,
\]

with separate damping and process-noise blocks for the two representations. Its stability discussion points to Neural-Fly supplement S5.

Two notation issues must be resolved in an independent implementation:

1. The text below Eq.10 prints `qdot_r=qdot_d-Lambda(q_d-q)` while defining `tilde q=q-q_d` and `s=tilde qdot+Lambda tilde q`. A consistent reference velocity is `qdot_r=qdot_d-L_q(q-q_d)`. The tracking gain `L_q` and parameter damping `D=diag(lambda_m I,lambda_r I)` also have different dimensions and roles.
2. For unequal parameter damping blocks, the symmetric covariance equation is
   \[
   \dot P=-DP-PD^\top+Q-P\phi^\top R^{-1}\phi P.
   \]
   The printed `-2 Lambda P` is equivalent only under a commutation condition, including scalar damping. A block-diagonal initial `P` does not ensure continued commutation: the measurement information can generate cross-block covariance. This is an algebraic consistency correction, not an assertion about undocumented implementation behavior.

## MAGIC-VFM: visually informed input-effectiveness adaptation

The general model (pp.3–4, Eqs.1–2) is

\[
\dot x=f_{\rm nom}(x,u,t)+d,\qquad
d=\Phi(x,u,E)\theta+\delta,
\]

where `E` contains visual-foundation-model features. The specialization used for controller synthesis (p.4, Eq.3) is

\[
d\approx\sum_{i=1}^{p}\theta_i\Phi_i(x,E)u
=H(x,E,u)\theta,\quad
H=[\Phi_1u\ \cdots\ \Phi_pu].
\]

Each `Phi_i` is a matrix. This displayed specialization is homogeneous linear in the command; it has no independent additive intercept unless the regressor is augmented. It models terrain-dependent actuation effectiveness and slip. The small basis network receives state and visual features; the pretrained DINO visual model provides those features. The learned small-network weights are fixed after offline training (pp.4–6); online adaptation updates `theta`, not the full visual model.

**Meta-learning differs from DAIML.** The training data are continuous driving trajectories that can cross terrain transitions. On a sufficiently short sampled window, `theta` is approximately constant. The inner objective on p.5 is ridge regression, and Eq.4 differentiates the residual fitting loss through that solution to train the feature network. In unambiguous regressor notation, the solution to the printed objective is

\[
\theta^*(w)=\left(\sum_{t\in\mathcal W}H_t(w)^\top H_t(w)+\lambda_r I\right)^{-1}
\left(\sum_{t\in\mathcal W}H_t(w)^\top y_t+\lambda_r\theta_r\right),
\]
\[
J(w)=\mathbb E_{\mathcal W}\sum_{t\in\mathcal W}
\|y_t-H_t(w)\theta^*(w)\|^2.
\]

These are window-dependent coefficients and one shared feature network, with spectral normalization (Algorithm 1, p.5). Unlike Neural-Fly's support/query algorithm, the displayed MAGIC objective uses the same window for inner fitting and outer loss. It does not include DAIML's adversarial environment discriminator. **The unnumbered closed-form equality printed immediately before Eq.4 is inconsistent with its own minimization objective:** it retains `theta` inside the putative regressor and sums individual inverses. The equations above follow from differentiating the stated objective, not from copying that equality. Uniqueness requires positive ridge regularization or a full-rank unregularized normal matrix.

The tracked-vehicle reduction (p.7, Eqs.11–12) uses measured forward/yaw velocity, desired-velocity commands, and an identified first-order internal response:

\[
\dot q=S(q)v,\quad
\dot v=A_n v+(B_n+\Phi\theta)u+\delta.
\]

The nominal reduced `A_n` has negative diagonal entries. The controller uses the plain reduced-state error (p.7, Eqs.17–18)

\[
s=v-v_{\rm ref},\quad
u=-(B_n+\Phi\hat\theta)^{-1}
[Ks+A_n v_{\rm ref}-\dot v_{\rm ref}].
\]

Consequently, with `tilde theta=hat theta-theta`,

\[
\dot s=(A_n-K)s-H\tilde\theta+\delta.
\]

This is an example where a **first-order controlled-state error is sufficient**. The position/yaw reference generator and convergence argument are additional layers (pp.7,9, Eqs.14–16, Theorem 2). It does not prove that any robot's joint-position interface already obeys an equally valid first-order model.

For the full-matrix gain, Proposition 1 gives (p.9, Eq.26)

\[
\dot{\hat\theta}=-\lambda\hat\theta
+\Gamma H^\top R^{-1}(y-H\hat\theta)+\Gamma H^\top s,
\qquad
\dot\Gamma=-2\lambda\Gamma+Q-\Gamma H^\top R^{-1}H\Gamma.
\]

The residual measurement is defined in physical derivative coordinates (p.8, Eq.20), with filtering/noise represented as measurement error. Its regressor includes the **actual command**. The online inverse requires the estimated input matrix to stay invertible with a bounded inverse; bounded disturbances/features alone do not guarantee this.

**The diagonal-gain law is a different case.** Theorem 1 prints a positive quadratic gain term (p.8, Eq.19),

\[
\dot\gamma_i=-2\lambda\gamma_i+q_i
+\gamma_i^2 u^\top\Phi_i^\top R^{-1}\Phi_i u.
\]

Do not identify this with the negative-information-term Kalman covariance law in Eq.26. The subsequent bound uses a uniformly coercive combined metric. Such boundedness needs explicit checking for a positive quadratic gain equation: even constant scalar values `lambda=q=H^T R^{-1}H=1` and `gamma(0)=2` give `gamma_dot=(gamma-1)^2`, which escapes at time 1. This algebraic example shows why the claimed convergence-speed comparison cannot replace a gain-boundedness condition. Use the full negative-sign Riccati form as the initial independent composite variant, with a discretization that preserves positive definiteness.

## Assessment of `B xdot = A x + a + Phi z`

The following is **our proposed derivation**, not an equation copied from the papers. Use different symbols from MAGIC's input matrix: here `B` multiplies a state derivative. Initially take known constant invertible `B`, known `A`, and an effective applied input `a` with the same dimension as `B xdot`:

\[
B\dot x=A x+a+\Phi(x,c)z_*+\delta,
\qquad a=a_{\rm nom}-\Phi(x,c)\hat z.
\tag{I1}
\]

The context `c` may contain visual features, command history, or other justified observables. A short-window approximately constant coefficient is a modeling hypothesis. It should be checked on disturbances and task phases outside feature training.

For a shared exogenous nominal input, define a **healthy dynamic reference** by

\[
B\dot x_d=A x_d+a_{\rm nom}(t),\quad
e=x-x_d,\quad\tilde z=\hat z-z_*.
\]

Then

\[
\dot e=F e-G\tilde z+B^{-1}\delta,
\qquad F=B^{-1}A,\quad G=B^{-1}\Phi.
\tag{I2}
\]

The raw position command need not equal `x_d`: a healthy actuator has transient lag. If the nominal policy depends on the measured state, its feedback difference must be included in the healthy closed-loop vector field. A trajectory generated from the current faulted observation is not automatically the counterfactual healthy policy trajectory. A practical formulation can condition the actuator reference on the same held command over each control interval, but its guarantee must use that stated reference definition.

Suppose the healthy linear loop is Hurwitz and choose a positive-definite **state metric** `W_x` satisfying

\[
F^\top W_x+W_xF=-Q_x,\qquad Q_x\succ0.
\]

The Euclidean special case `W_x=I` requires a negative-definite symmetric part of `F`; stable eigenvalues alone are insufficient. For nonlinear healthy feedback, a suitable incremental Lyapunov/contraction metric and its derivative terms must replace this constant linear identity.

For the first-order state, the tracking contribution required by cross-term cancellation is

\[
\boxed{\quad+P\Phi^\top B^{-\top}W_x e\quad}.
\tag{I3}
\]

Thus **joint-position error can be enough** when joint position is the validated first-order controlled state. A filtered velocity error is not mandatory for that model. However, plain `+P Phi^T e` requires the coordinate/metric condition `B^{-T}W_x=I`. This holds for `B=I,W_x=I`, and also for constant symmetric positive-definite `B` with `W_x=B` when the corresponding stability inequality holds. With `B=I`, the familiar state metric is simply omitted only when the healthy loop is Euclidean-contractive.

Use the dynamics measurement

\[
y=B\dot x-Ax-a=\Phi z_*+\delta+\epsilon,
\tag{I4}
\]

where `a` is the **actually applied corrected input**. Subtracting only `a_nom` would instead measure `Phi(z_*-hat z)` and change the estimator observation equation. Causal differentiation or integral/filtered regression must treat both sides consistently, including actuator delay and timestamps. The measurement regressor in these coordinates is `Phi`; the state coupling is `G=B^{-1}Phi`. Mixing them without transforming the noise model changes the algorithm.

A consistent scalar-damping composite candidate is

\[
\boxed{
\dot{\hat z}=-\lambda(\hat z-z_0)
+P\Phi^\top R^{-1}(y-\Phi\hat z)
+P\Phi^\top B^{-\top}W_xe,
\quad
\dot P=-2\lambda P+Q_z-P\Phi^\top R^{-1}\Phi P.
}
\tag{I5}
\]

`z_0` is an explicit prior center; zero-centered leakage is the source law's special case. With constant `z_*`, zero representation/noise error, and `z_0=z_*`, differentiating

\[
V=\tfrac12e^\top W_xe+\tfrac12\tilde z^\top P^{-1}\tilde z
\]

gives

\[
\dot V=-\tfrac12 e^\top Q_x e
-\tfrac12\tilde z^\top
(\Phi^\top R^{-1}\Phi+P^{-1}Q_zP^{-1})\tilde z.
\tag{I6}
\]

For an unknown nonzero prior error, varying coefficients, and noisy residuals, additional forcing terms produce practical bounds under the required uniform bounds. This derivation explains the sign and weighting; it is not yet a proof for the entire sampled manipulation stack with contact, saturation, and a learned nominal model. A freely tuned extra tracking coefficient also changes the cancellation argument; its proof cannot simply be inherited with the same Lyapunov weights.

**Matching remains a physical-interface condition.** Equation I1 assumes the disturbance can be canceled through available input directions. For a real input map `D(x)u`, cancellation requires `Phi z_*` to lie in its range, with feasible corrective commands. Torque faults can be matched for joint-torque control but partly uncorrectable through a restricted Cartesian translation interface. A learned basis can describe that component without making it actuated. Similarly, an additive model can represent an effectiveness fault only if its regressors include the relevant command dependence. A separate multiplicative-effectiveness model, inspired by MAGIC, should retain its own input-map and invertibility analysis.

## Euclidean composite adaptation versus mirror geometry

A directly relevant additional source is Tang, Sun, and Azizan, **Meta-Learning for Adaptive Control with Automated Mirror Descent**, [L4DC 2025, PMLR 283:1025–1037](https://proceedings.mlr.press/v283/tang25b.html). The local extended author PDF `2407.20165v5.pdf` has 20 pages. Its Sections 2.2 and 4 (pp.4–7, Eqs.5–10) learn a state feature network **and** a mirror potential/coordinate transformation using a closed-loop tracking-plus-control-effort meta-objective. Its matrix named `P` is a learned coordinate transformation/gain, **not Neural-Fly's evolving Riccati covariance**. It is a useful geometry reference, not evidence that the original three methods already use mirror descent.

For a fixed, twice-differentiable strongly convex potential `psi` with an invertible gradient on the relevant domain, introduce

\[
\xi=\nabla\psi(\hat z),\qquad
\hat z=\nabla\psi^*(\xi).
\]

With our sign convention and a constant true coefficient, choose the parameter storage

\[
D_\psi(z_*\|\hat z)=\psi(z_*)-\psi(\hat z)
-\nabla\psi(\hat z)^\top(z_*-\hat z).
\]

Its derivative is `tilde z^T xidot`. Therefore a compatible **dual-space composite extension**, derived here, is

\[
\boxed{\dot\xi=
\Phi^\top R^{-1}(y-\Phi\hat z)
+\Phi^\top B^{-\top}W_xe},
\qquad
\dot{\hat z}=[\nabla^2\psi(\hat z)]^{-1}\dot\xi.
\tag{I7}
\]

For exact constant-coefficient dynamics and noiseless measurements, the sum of state storage and this Bregman divergence has derivative

\[
\dot V=-\tfrac12 e^\top Q_xe
-\tilde z^\top\Phi^\top R^{-1}\Phi\tilde z.
\]

This establishes the basic cancellation and nonincreasing storage under the stated ideal assumptions; asymptotic or exponential claims require the corresponding additional regularity/excitation/damping arguments. A quadratic potential `psi(z)=0.5 z^T Gamma^{-1}z` recovers a **constant-gain** Euclidean composite law. Leakage and time-varying `z_*` add terms that need separate treatment. Dual leakage toward `grad psi(z_0)` is generally different from primal leakage toward `z_0`.

The local mirror paper prints a negative sign in its Eqs.5 and 8b (pp.4,6), although its plant has a positive disturbance, its controller subtracts the estimate, and its Euclidean Eq.3b has a positive sign. Under those definitions the mirror update needs the positive sign to cancel the cross term. This can be checked even in one dimension: `edot=-e-tilde z` with `zhatdot=+e` is stable; replacing the update by `-e` gives an eigenvalue `(-1+sqrt(5))/2>0`. This memo uses the independently checked sign, without attributing an unverified correction to the authors. Also, an unsmoothed `l_p^p` potential does not supply an everywhere invertible Hessian for every printed `p>=1`; any implementation needs a valid domain or smooth, strongly convex parameterization.

**A Riccati metric and a mirror potential are not interchangeable.** A fixed potential supplies a parameter-dependent Hessian; Riccati `P(t)` depends on measurement history. They coincide in a fixed quadratic special case, not generally. If a time-varying quadratic potential is defined by `psi_t(z)=0.5 z^T P(t)^{-1}z`, then differentiating the dual coordinate produces an additional `d(P^{-1})/dt * zhat` term, and differentiating the Bregman storage produces an explicit time derivative. Multiplying a mirror inverse Hessian by a Riccati matrix without these terms does not preserve the existing proof; the product may not even be a symmetric Hessian inverse. The initial independent study should therefore keep the Riccati composite law and fixed-potential mirror composite law as distinct, explicitly derived variants.

## Concrete design consequences

The source-supported starting point is a validated nominal actuator model, a learned disturbance/effectiveness basis, and a small online coefficient vector. Neural-Fly supplies the composite prediction/tracking structure and Riccati gain; MAGIC supplies visual context and short-window coefficient meta-learning; HMAC supplies a possible two-timescale additive representation when the data support that separation. The mirror paper motivates a separately derived learned geometry. Combining these ingredients is a new method design, not a demonstrated theorem or performance result inherited from the cited papers.

Before calling the joint-position version valid, establish the healthy incremental model and state metric over the intended operating range, define the reference and actual-input measurement precisely, and identify the corrective input directions. Feature training should then be judged both by residual prediction on held-out trajectories and by downstream control performance; a low prediction loss or smaller tracking error alone is not a manipulation-success guarantee. Separate data are needed for training features, selecting adaptation parameters, and testing the final controller. None of these choices requires importing or automatically merging the historical repository's main branch.
