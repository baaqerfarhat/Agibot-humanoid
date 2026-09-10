# Geometry and symmetry for independent joint-space adaptation

Reviewed 2026-09-08. This memo concerns the five user-provided PDFs in [manipulator_manifold_paper](../../manipulator_manifold_paper/). Page numbers below are **one-based PDF pages of those exact versions**, not another publication's pagination. The accompanying [source metadata](manifold_sources.json) records titles, authors, version dates, hashes, and primary records. The PDFs were read locally; arXiv records were checked for identity and version, and the GUFIC authors' implementation was inspected to resolve one printed sign discrepancy. This is a technical reading and integration proposal, not a complete formal verification of every paper or a new experimental result.

**Recommendation.** Use geometry to construct a state-dependent disturbance representation and to make coordinate transformations consistent. Use verified morphological symmetries to share statistical strength across genuinely equivalent limbs. Keep disturbance estimation, correction feasibility, and closed-loop stability as separate requirements. A Riemannian planner and a passive force–impedance controller are useful optional components when their interfaces and assumptions are available; neither is a direct replacement for an online disturbance estimator.

**Specification follow-up.** [GEOMETRIC_METHOD.md](GEOMETRIC_METHOD.md) now defines
the proposed wrench/servo units, context, symmetry family, effectiveness tensor,
estimator and allocation transformations, with a finite-group reference
implementation and exact identity tests. The source reading below remains a
review; the follow-up derives the method and records its unresolved physical
assumptions without attributing new results to these papers.

## 1. Sources and what each contributes

| Local PDF | Verified title and version | Most useful contribution here |
|---|---|---|
| [2211.07945v4.pdf](../../manipulator_manifold_paper/2211.07945v4.pdf) | **Geometric Impedance Control on SE(3) for Robotic Manipulators**; Seo, Prakash, Rose, Choi, Horowitz; v4, 2025-03-05; IFAC World Congress 2023 paper | Consistent pose and velocity errors, a potential-derived wrench, and conditional model-based tracking results |
| [2402.15552v4.pdf](../../manipulator_manifold_paper/2402.15552v4.pdf) | **Morphological Symmetries in Robotics**; Ordoñez-Apraez et al.; v4, 2025-03-24; IJRR, DOI 10.1177/02783649241282422 | Conditions under which robot morphology induces joint-space transformations and equivariant models |
| [2412.05197v3.pdf](../../manipulator_manifold_paper/2412.05197v3.pdf) | **A Riemannian Take on Distance Fields and Geodesic Flows in Robotics**; Li, Qiu, Calinon; v3, 2026-01-27 | Joint-space metrics, learned distance fields, and explicit separation of geometric guidance from constrained control |
| [2503.09829v3.pdf](../../manipulator_manifold_paper/2503.09829v3.pdf) | **SE(3)-Equivariant Robot Learning and Control: A Tutorial Survey**; Seo et al.; v3, 2025-04-23; record states accepted to IJCAS | Representation conventions and conditions connecting equivariant perception, actions, and control |
| [2504.17080v3.pdf](../../manipulator_manifold_paper/2504.17080v3.pdf) | **Geometric Formulation of Unified Force-Impedance Control on SE(3) for Robotic Manipulators**; Seo et al.; v3, 2026-06-09 | Energy-tank accounting for contact-force and moving-reference ports |

Version identities are supported by the primary arXiv records: [GIC](https://arxiv.org/abs/2211.07945v4), [morphological symmetries](https://arxiv.org/abs/2402.15552v4), [Riemannian distance fields](https://arxiv.org/abs/2412.05197v3), [tutorial survey](https://arxiv.org/abs/2503.09829v3), and [GUFIC](https://arxiv.org/abs/2504.17080v3). A journal-template header or placeholder DOI in a PDF was not used to infer a finalized publication version.

### 1.1 Geometric impedance control: adopt the coordinates, verify the plant assumptions

GIC defines the relative configuration through rigid transforms rather than subtracting orientation coordinates. Its left-invariant pose error function is
\[
\Psi(g,g_d)=\tfrac12\|I-g_d^{-1}g\|_F^2
=\operatorname{tr}(I-R_d^\top R)+\tfrac12\|p-p_d\|^2
\]
(Eqs. 12–16, p. 3). The corresponding body-coordinate error is
\[
e_g=
\begin{bmatrix}
R^\top(p-p_d)\\
(R_d^\top R-R^\top R_d)^\vee
\end{bmatrix}
\]
(Eq. 20). Desired and actual body velocities are compared after transport,
\[
V_d^*=\operatorname{Ad}_{g^{-1}g_d}V_d^b,\qquad e_V=V^b-V_d^*
\]
(Eqs. 22–24). This is directly useful if our state representation includes end-effector pose, twist, or wrench; those quantities must share a declared frame and origin.

For anisotropic stiffness, multiplying the unweighted error by an arbitrary gain is not generally the gradient of the chosen potential. The paper derives the elastic wrench from its weighted potential (Eqs. 29–31, p. 4), then uses it in the model-based wrench law (Eq. 32). This is a useful warning against inserting a geometry-based weighting into a feedback update without redoing its energy calculation.

Theorems 4–6 (pp. 4–5) assume the stated manipulator dynamics, a nonsingular body Jacobian, reachable smooth references, and zero external torque for their tracking conclusions. Their controller uses inertia, Coriolis and gravity compensation; the operational-space expression explicitly contains Jacobian inverses. The UR5e simulation (p. 6, Table 1) compares matched initial conditions and stiffness/damping on a prescribed trajectory. It is not a multi-fault adaptation comparison.

**Integration limit.** These results do not prove contraction of an unknown position-controlled robot, or stability of a VLA plus an adaptation wrapper. Also, the displayed smooth rotational error has undesired critical points: for scalar rotational stiffness, any relative rotation of 180 degrees makes the skew error zero despite a nonzero pose error. Do not turn the paper's equilibrium statements into a global unique-attractor guarantee. A seven-joint arm also requires explicit treatment of redundancy and null-space motion rather than a literal inverse of a nonsquare Jacobian.

### 1.2 Morphological symmetries: useful constraints on models, conditional on the complete system

Definition 1, Proposition 1 and Eqs. 8–10 (p. 6) relate an invariant Lagrangian to equivariant dynamics. For the orthogonal joint representations relevant here, the mass matrix transforms by conjugation and the generalized force transforms with the joint representation. The paragraph following Proposition 1 explicitly requires contacts and control actions to remain related by the transformation.

Definition 2 and Eq. 12 (p. 8) distinguish an ordinary spatial isometry from a **reachable morphological transformation** that reproduces its dynamics. Replicated kinematic branches and compatible mass distributions are required, not merely a visually similar silhouette. Eqs. 18–21 and the identification procedure (pp. 10–11, Fig. 7) construct joint representations from branch permutations and within-branch axis transformations, then check kinematics and energy. For one-dimensional joints, signed permutations are a common representation after choosing appropriate coordinate origins.

The learning applications are unusually relevant to adaptation. Section 6.1 (p. 14) transforms joint quantities, contact forces, surface normals, sensor positions, and limb identities together. Eq. 26 (p. 16) distinguishes polar and axial quantities under reflection: linear momentum transforms by \(R\), while angular momentum transforms by \(\det(R)R\). The experiments demonstrate supervised centroidal-momentum estimation and contact-state classification on Atlas/Solo/Mini Cheetah, including histories for contact detection (pp. 15–18). They do not establish fault compensation or composite-observer superiority.

**Integration limit.** A left-only fault breaks a left/right symmetry of that particular faulted system unless the fault parameters and affected-joint labels are also transformed. A left/right-specific task or asymmetric contact can likewise break the task symmetry. Similar-looking arms, different payloads, actuator gains, limits, or mounting geometries need explicit checks. Do not impose arbitrary permutations of the joints of a serial arm.

**Do not overinterpret the harmonic decomposition.** Eqs. 22–25 (pp. 12–13) give an orthogonal change of coordinates and a work decomposition. These are useful for diagnostics and structured priors, but orthogonality alone does not prove independent nonlinear subsystem dynamics. In particular,
\[
M(gq)=\rho(g)M(q)\rho(g)^{-1}
\]
does not imply that \(M(q)\) commutes with every \(\rho(g)\) at a nonsymmetric configuration. A simple counterexample to automatic nonlinear decoupling is a limb-swap symmetry, with symmetric/antisymmetric coordinates \(q_+,q_-\), unit mass and
\[
U=\tfrac12(q_+^2+q_-^2)+\kappa q_+^2q_-^2,\qquad \kappa>0.
\]
This potential is invariant under \(q_-\mapsto-q_-\), but
\[
\ddot q_+=-q_+-2\kappa q_+q_-^2,\qquad
\ddot q_-=-q_--2\kappa q_-q_+^2
\]
remain coupled. This is our integration caution, not an experimental finding about the authors' robots. Separate per-mode filters require additional block-structure assumptions or measured approximation error.

### 1.3 Riemannian distance fields: a planning/reference component, not an observer

The key relation is the dual-metric eikonal equation
\[
\nabla U(q)^\top G(q)^{-1}\nabla U(q)=1,
\]
with backward flow \(\dot q=-G^{-1}\nabla U\) (Eqs. 6–10, pp. 3–4). Where the exact distance is differentiable and the prescribed flow is realized, \(\dot U=-1\). This is a statement about a kinematic flow of an exact field; it is not a contraction proof for the second-order physical robot or a learned approximation.

The most reusable geometry is \(G(q)=M(q)\), or a task pullback
\[
G(q)=J(q)^\top G_XJ(q)+\epsilon I,\qquad \epsilon>0
\]
(Eqs. 11–15, p. 4). The regularizer matters: a task pullback is generally semidefinite, particularly for redundant robots. The Jacobi metric \(2(H-U_{\rm pot}(q))M(q)\) is appropriate to a conservative fixed-energy setting with \(H>U_{\rm pot}\), not automatically to active contact, friction, or time-varying actuation.

NES learns a source–goal distance using an eikonal residual and a Laplace–Beltrami regularizer (Eqs. 19–24, p. 7). Boundary-conditioned NES replaces a specific joint goal by an end-effector target set (Eqs. 25–27, pp. 7–8). Metric-conditioned NES includes physical/task parameters in the input (Eq. 28, p. 8). These ideas could guide reference trajectories toward configurations where corrective control is easier, or provide geometry-aware features, but that is a new hypothesis.

Crucially, the paper itself separates guidance from actuation. Section 6.2 (p. 14) reports instability from directly following the learned field near the target, blends it with a local attracting vector (Eq. 33), and uses velocity/torque QPs with explicit limits (Eqs. 34–37). Null-space projection can change the original optimum (p. 15, Eqs. 38–39). The experiments include a planar robot and a seven-axis Franka, while the conclusion acknowledges inaccurate neural solutions on strongly anisotropic metrics (p. 21). Page 20 also explicitly distinguishes geodesic optimality from real execution-energy optimality.

**Integration limit.** Low potential energy is not a general Lyapunov or contraction certificate, and a geodesic does not guarantee feasibility after faults. Appendix A.3 (p. 25, Eqs. 73–75) discusses squared power \((\dot q^\top\tau)^2\), which differs from squared torque. For a free unit-mass particle, circular motion \(q=(\cos t,\sin t)\) has perpendicular torque \(\tau=\ddot q=-q\), hence zero mechanical power but nonzero torque and a nongeodesic Euclidean path. Thus a zero-power objective does not uniquely select geodesics or minimal torque. Keep these objectives distinct when designing adaptation penalties.

### 1.4 The tutorial: a guide to representations and symmetry breaking

The survey's equivariant-map definition (Section 3.1.3, p. 11), representation/tensor-product discussion (pp. 12–16), and relative pose/body-error construction (Eqs. 37–43, pp. 22–24) help specify a network's inputs and outputs. It also distinguishes scene and grasp transformations in bi-equivariant manipulation (Fig. 7, pp. 17–18). An end-effector pose distribution is a different output object from a vector of joint commands.

Section 4.2 (pp. 19–20) requires reward and transition symmetry for its optimal-policy discussion. Symmetric dynamics alone do not make an arbitrary frozen VLA equivariant, and a task instruction can distinguish left/right, directions, or particular objects. For nonunique optimal actions, equivariance of the optimal action set does not guarantee that an arbitrary deterministic tie-breaker is equivariant.

Section 5.5.2 (p. 25) explains how body-frame invariant feedback gives spatial-frame equivariant feedback. Section 6.2 (p. 26) explicitly names singular configurations, input constraints, camera geometry, and occlusion as symmetry-breaking factors. These limitations should be carried into our design. Several tutorial equations contain typographical inconsistencies, so use the original GIC definitions for implementation; for example, SE(3) matrix inverse is not generally transpose.

### 1.5 GUFIC: useful energy accounting, with a verified printed sign discrepancy

GUFIC starts with a commanded **body wrench** and external measured wrench (Eqs. 2–10, p. 2). Its passivity port is \((V^b,F_e)\), with power \((V^b)^\top F_e\) (Eq. 11, p. 3). Eq. 14 shows why simply adding force tracking to impedance does not preserve passivity: force control and a moving reference introduce additional power terms. Separate force and reference tanks scale these contributions (Eqs. 15–20, p. 3). Theorem 3 (p. 4), with the proof in Appendix A.1 (p. 6), gives
\[
\dot S_{\rm tot}\le (V^b)^\top F_e
\]
for the specified model, tank dynamics and controller. This is a useful template for accounting for energy injected by adaptation, **if the actual power port is available**.

The paper transports a desired wrench with the dual adjoint (Eq. 26, p. 4) and sends joint torque \(J_b^\top F'\) (Eq. 29). Its Indy7 simulations use force sensing and 1000 Hz control on planar/spherical contact tasks (p. 5); the force-error improvement is not uniformly accompanied by lower translational error in both scenarios (Table 1). It does not test latent disturbance adaptation.

**Verified equation discrepancy.** Eq. 21 (p. 4) prints \(+\zeta\nabla_1\Psi\), while Eqs. 22–23 define its translational component as \(R^\top(p-p_d)\), with \(\zeta>0\). For a stationary translational target, that sign produces an outward field. The authors' [implementation at commit b3a2afc](https://github.com/Joohwan-Seo/GUFIC_mujoco/blob/b3a2afcef4ff93175b3327e51bc1cee31a7678aa/gufic_env/env_gufic_velocity_field.py#L296) instead uses **minus** the translational and rotational error terms, at lines 296–297. The PDF page was also rendered to verify the printed sign. Re-derive the intended attracting field; do not copy Eq. 21 literally. This does not refute the separate tank power-balance calculation.

**Integration limit.** Passivity alone is not an unconditional contact-safety, force-limit, collision-avoidance, or task-success guarantee. Environment/interconnection assumptions, sensing error, model mismatch, discretization, saturation, and reference modifications matter. A joint-position command offset has units of position; its squared magnitude is not mechanical energy. An energy tank attached to that offset requires a valid account of the underlying actuator/controller power, rather than relabeling a numerical correction budget as passivity.

## 2. Which symmetry is valid for which variable?

Use separate symbols for a symmetry element \(h\), an end-effector pose \(T\), a configuration-space metric \(\mathcal G(q)\), and the actual correction map \(B\).

| Transformation | Joint state and action | External/task quantities | What it can support |
|---|---|---|---|
| Passive change of world coordinates \(h=(S,t)\in SE(3)\) | The same physical joint angles, joint velocities, scalar joint torques and joint targets remain unchanged | All poses, point clouds, gravity, external vectors and wrench origins are re-expressed consistently | Coordinate consistency; no extra physical experiments or actuation |
| Active repositioning of task/object with fixed robot base | Joint configurations solving the task usually change nonlinearly; no generic seven-dimensional rotation matrix acts on \(q\) | Robot base, gravity, contacts and obstacles are not implicitly moved | Only a conditional/local task symmetry when reachability, dynamics and sensing permit it |
| Verified morphological symmetry | Joint map is specified, often an affine signed permutation about calibrated zero angles; actions, faults and limits transform compatibly | Permute limb contact identities and transform vectors, surfaces, task labels and payloads | Data augmentation or architecture constraints for that verified subgroup |
| Arbitrary joint permutation or pose-space rotation applied directly to \(q\) | Generally invalid for serial chains with different joint axes, gains and limits | Usually inconsistent | Do not enforce |
| Change of robot morphology or a joint lock | Changes kinematics, constraints, and potentially action dimension | Contact/reachable sets may change | A model-family or mode change; not a coordinate symmetry |

A polar vector transforms as \(v'=Sv\), while under reflection an axial vector transforms as \(\omega'=\det(S)S\omega\). For a spatial twist/wrench at a declared origin, use the adjoint and its power-preserving dual: if \(V'=\operatorname{Ad}_hV\), then \(F'=\operatorname{Ad}_h^{-\top}F\). Two independent rotated three-vectors are insufficient when the wrench origin changes, because the moment arm contributes torque.

A fixed gravity vector restricts active rotational symmetry to its stabilizer unless gravity is transformed or supplied as an input. Expressing it in a body frame, for example \(R^\top g_{\rm world}\), preserves the information; deleting it to obtain an invariant feature does not. The same applies to contact normals, slip directions, force measurements, object properties and support geometry.

Revolute joints with limits are not automatically circles on which every wraparound is feasible. A sine/cosine encoding is useful for periodic geometry, but retain joint-limit margins, the actual permissible interval and, where needed, turn count or actuator history. A joint lock cannot be fixed by choosing another chart or metric: it removes feasible motion directions. Geometry may help find another feasible trajectory if redundancy remains.

## 3. Mapping to a learned disturbance \(\Phi(x)z\)

The following is a proposed integration, not a result proved by these five sources.

### 3.1 Define the modeled quantity and its units first

Let \(s_t\) denote physical state and let the frozen policy supply action \(a_t\). Define the model input \(x_t\) to contain the needed state, current known command and history:
\[
x_t=(q_t,\dot q_t,a_t,a_{t-1:t-L},q_{t-1:t-L},
       T_{\rm base}^{-1}T_{\rm ee},T_{\rm ee}^{-1}T_{\rm object},
       g_{\rm body},\mathcal C_t,\text{robot/actuator parameters}).
\]
Here \(\mathcal C_t\) includes available contact mode, normals, relative velocity/slip, force measurements and object/payload covariates. Missing or inferred quantities must be marked as such; a latent contact estimate is not measured ground truth.

Choose one output quantity:
\[
d_\tau(x)=\Phi_\tau(x)z,\quad
d_{\ddot q}(x)=\Phi_{\ddot q}(x)z,\quad\text{or}\quad
d_{\dot q}(x)=\Phi_{\dot q}(x)z.
\]
These are torque, acceleration and effective closed-loop velocity disturbances, respectively. They are not interchangeable. With known inertia, \(d_{\ddot q}=M(q)^{-1}d_\tau\); with a position servo, the relation between a command correction and either disturbance still requires its own model.

The action must remain in the model input when disturbance depends on commanded effort. Examples include actuator gain loss and command scaling. Velocity-dependent friction, configuration-dependent payload/gravity mismatch and external contact also cannot generally be represented by one constant joint offset.

A sensible initial basis combines a small learned geometric component with interpretable features such as joint velocity for viscous friction, a regularized sign of velocity for Coulomb friction, and \(J_b(q)^\top\) applied to body-wrench features. Known physical terms should only be used when their quantities are available. General rigid-body parameter regressors involving acceleration require an appropriate measured/filtered regression; noisy numerical differentiation is not an innocuous substitution.

### 3.2 Enforce the correct transformation of the latent model

Suppose \(d(h\cdot x,\rho_z(h)z)=\rho_d(h)d(x,z)\). Then the factorized model must satisfy
\[
\boxed{\Phi(h\cdot x)\rho_z(h)=\rho_d(h)\Phi(x).}
\]
An invariant latent, \(\rho_z=I\), is suitable only when its semantics actually remain unchanged under that transformation. A per-joint fault vector permutes with the limbs. An external force represented in world coordinates rotates with the world frame; a body-frame force has a different transformation. A learned basis can change with configuration while the physical latent remains constant, but this must be defined, not assumed.

For a passive world-frame change, joint torque is invariant and a joint-torque disturbance model can use body-relative geometric features plus \(q,\dot q\) and all relevant external covariates. For actual left/right symmetry, use the robot-specific joint representation. If exact symmetry is only approximate, retain robot-specific parameters or a symmetry-breaking residual component instead of enforcing a false equality.

A latent vector is basis-dependent: \(\Phi S\) and \(S^{-1}z\) represent the same disturbance for invertible \(S\). Fix the learned basis before online adaptation and evaluate physical disturbance/prediction error as well as parameter error. Changing the basis online would require transport of the latent and covariance.

### 3.3 Geometry does not replace identifiability or correction authority

If measured regression data satisfy
\[
y_t=F_t z+\nu_t,\qquad F_t=C_t\Phi(x_t),
\]
then estimation depends on \(F_t\), not \(\Phi\) alone. A filtered or indirect measurement may erase disturbance directions. Excitation/information must be checked using the actual noise-weighted regression over the observed history. Symmetry can reduce the number of free parameters or enforce a valid prior, but transformed copies of a recorded sample are not independent physical excitation.

Likewise, cancellation requires an admissible correction \(c\) such that
\[
B(x)c=\Phi(x)z,\qquad c\in\mathcal U(x),
\]
in a model where these terms share the same state-effect units. Equivariance may improve the estimate on the right-hand side; it does not enlarge \(B(x)\mathcal U(x)\). A lock, saturation, contact constraint, or missing correction channel remains an authority limitation. A new full-joint controller can have greater authority than a restricted Cartesian interface, but that improvement comes from the controller/interface change.

## 4. Composite adaptation: what geometry changes and what it does not

For a first-order error model with compensation subtracted,
\[
\dot e=F_0(e,x)+B(x)\Phi(x)(z-\hat z)+w,
\]
and a verified healthy-loop storage \(V(e,x)\), the relevant tracking covector is
\[
h_z=\Phi(x)^\top B(x)^\top\,\partial_eV.
\]
A positive \(+\Gamma h_z\) term in \(\dot{\hat z}\) cancels the corresponding state/parameter cross-term under the usual fixed-metric assumptions and the convention \(\tilde z=\hat z-z\). If \(\Phi\) already maps latent coefficients directly into state-error dynamics, omit the extra \(B\); it must not be counted twice. The prediction term uses the measured regressor \(F_t\), not automatically the same matrix as the control term.

This explains when **joint-position error alone can be appropriate**. If the relevant healthy first-order position loop is actually contractive in a justified metric, its storage derivative provides the needed covector. An arbitrary second-order mechanical plant does not inherit that property merely because position eventually tracks. The GIC papers' velocity-error construction is essential to their particular energy proof, not a requirement to impose a filtered velocity error on every independent position-loop design.

A state-dependent Riemannian metric introduces derivative/transport terms. For example, a quadratic local storage \(V=\tfrac12e^\top\mathcal G(x)e\) has the additional term \(\tfrac12e^\top\dot{\mathcal G}(x)e\). The metric must be checked for the actual closed-loop dynamics. Inertia, a pullback metric, an estimator covariance, a noise precision and a contraction metric serve different purposes and should have distinct notation.

For an equivariant implementation, the update's other quantities must transform too. If \(z'=\rho_z z\), then covariance transforms as \(P'=\rho_z P\rho_z^\top\); the measurement regressor/noise covariance and tracking covector must transform compatibly. The tracking covector obeys \(h_z'=\rho_z^{-\top}h_z\), so \(P'h_z'=\rho_z Ph_z\). Coordinatewise clipping, unequal gains, action masks and whitening can break equivariance unless the feasible set, metric and parameters respect the same group. Scalar gains with identity covariance are naturally compatible with orthogonal representations, not every nonorthogonal adjoint transformation.

Neither these sources nor the existing composite formulation establishes that composite adaptation must outperform Kalman filtering. Prediction and tracking residuals can share model error; extra feedback can add bias, noise or instability. Retain the zero-tracking nested Kalman case, tune on independent development data, and test whether the learned basis improves held-out physical prediction before crediting closed-loop improvement to geometry.

## 5. Concrete research choices supported by this reading

1. **Start with joint-space adaptation and a small state/action-dependent basis.** Use geometric relative poses and properly transformed external vectors as features. This is the closest integration with \(\Phi(x)z\); it does not require a full Riemannian planner or a new force controller.
2. **Identify a legitimate symmetry subgroup from the robot model.** Check joint maps, kinematics, inertia, limits, servo parameters and contact/task transformations. Use a nontrivial group only if those checks support it. Keep a plain non-equivariant model as a comparator.
3. **Separate tests of representation, estimation and control.** Compare matched-capacity bases with/without valid symmetry; Kalman versus its composite extension with the same basis and measurements; and correction interfaces/allocators with the same estimate. Hold out fault identities, configurations, payloads and contact regimes.
4. **Treat changing action authority explicitly.** Test each joint and fault family, including torque/friction/gain changes and locks as distinct cases. Report prediction error, excitation, saturation, feasible residual and task success. A result on smooth additive faults does not establish performance on locked-joint constraints.
5. **Add NES guidance only as a separate planning study.** Its hypothesis is that reference/posture selection can avoid poor correction configurations. That changes trajectories and needs its own comparison; it should not be mixed with an observer-only claim.
6. **Consider GUFIC-style tanks only with a measured/modelled physical power port.** Preserve the discrete-time energy budget and actual actuator limits, and resolve the printed sign using the authors' implementation and a fresh derivation. A tank can trade tracking for bounded supplied energy; it cannot guarantee success.

The strongest supported direction is a **geometry-informed disturbance representation with conditional symmetry**, coupled to independently justified estimation and control. The useful novelty would have to come from the resulting model, adaptation law, verified assumptions, and held-out behavior—not from calling joint space a manifold or asserting that equivariance repairs unactuated directions.
