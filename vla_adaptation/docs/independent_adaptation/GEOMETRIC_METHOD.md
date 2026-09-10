# Geometric specification of the composite adapter

2026-09-08. This completes the geometric construction for the constant local
descriptor law in [FORMULATION.md](FORMULATION.md), especially Eqs. (11), (12),
(15)–(21). It specifies a model family and exact algebraic consistency, with an
untrained reference implementation. It does not validate any robot's symmetry,
servo reduction, feature quality, or closed-loop sampled implementation.

## 1. Wrench features have generalized-force units

Let the robot have `d` joint coordinates, calibrated about symmetry-compatible
zero positions, and `C` declared candidate wrench slots. Slot `c` belongs to a
specified link and a **link-fixed** frame with origin `O_c`. Let
`T_Bc(q)=(R_Bc(q),p_Bc(q))` map its coordinates into the robot base frame `B`.
Use linear-first body twists and force-first wrenches:

\[
 V_c=\begin{bmatrix}v_{O_c}^{c}\\\omega_c^c\end{bmatrix}
     =J_c(q)\dot q,\quad J_c\in\mathbb R^{6\times d},\qquad
 F_c=\begin{bmatrix}f^c\\\mu_{O_c}^c\end{bmatrix}
     =\Phi_{F,c}(\xi)z_d,\quad\Phi_{F,c}\in\mathbb R^{6\times p_d}.
 \tag{G1}
\]

The first three feature rows have newtons per coefficient; the last three have
newton-metres per coefficient. Fix dimensionless coefficient scales for the
implementation. Generalized forces have units N m for revolute joints and N for
prismatic joints; writing radians as dimensionless does not turn torque into a
position target. A force applied at a different point `r_c`, expressed relative to
`O_c`, contributes moment `r_c cross f^c` in addition to any free couple. Use that
moment in `F_c`. Do not pair an end-effector Jacobian with a wrench about the
world origin. A sliding contact can be represented by its changing application
point relative to the fixed slot; do not differentiate that point as though it
were a material frame attached to the link.

A concrete admissible context schema, fixed before feature fitting, is

\[
\begin{split}
 \xi_t=(&q_t,v_t,\{q_{t-\ell},v_{t-\ell},a_{\mathrm{sent},t-\ell}\}_{\ell=1}^{H},
       \mathfrak m,\{T_{Bc}(q_t),T_{Bc}^{-1}T_{Bo,t},R_{Bc}^{\top}g_B,\\
       &\quad m_c,r_c^c,n_c^c,v_{\mathrm{slip},c}^c,
                   F_{\mathrm{meas},c}^c,b_{\mathrm{available},c}\}_{c=1}^{C},
       \mathcal V_t).
\end{split} \tag{G2}
\]

Here `mathfrak m` contains joint types, axes/zeros, link/limb labels, morphology,
healthy actuator parameters and limits. `T_Bo` is the relevant object's pose in
the base frame; `m_c` is a measured or inferred contact activation, `n_c` a normal,
and `b_available` distinguishes missing quantities from measured zeros. Missing
fields must be masked or omitted by a fixed declared schema; an inferred contact
is not ground truth. `mathcal V` is optional visual input together with its
camera/frame metadata. `v` may be an explicitly identified estimate. If it or
history is needed, this is no longer a claim of position-only sensing.

Use the nuisance component of the wrench, with nominal/desired contact already
accounted for. Activations multiply the corresponding feature blocks. The schema
does **not** consume the unknown current corrected command inside its networks.
The effectiveness regressor introduces that command explicitly in Section 4.
Omitting unavailable context changes the approximation hypothesis and can increase
representation error; geometry does not supply the missing information.

Stack `J=[J_1;...;J_C]` and `Phi_F=[Phi_F,1;...;Phi_F,C]`. Virtual power gives

\[
 \dot q^\top\tau_d=\sum_c V_c^\top F_c,\qquad
 \boxed{\Phi_\tau(q,\xi)=\sum_cJ_c(q)^\top\Phi_{F,c}(\xi)},\quad
 \tau_d=\Phi_\tau z_d\in\mathbb R^d. \tag{G3}
\]

This is a pullback of forces, requiring no inverse Jacobian. Its image is limited
to `range(J.T)`: one six-dimensional wrench cannot represent every generalized
force of a redundant arm. Internal joint friction or faults outside that image
need a separately typed joint-force basis, concatenated with the wrench basis.
Identification still depends on the rank/excitation of the observed regressor.

GIC's local PDF defines the linear-first body twist and `V_b=J_b qdot` on
**p. 2, Eqs. (2), (5)–(7)**. GUFIC uses wrench transport and `tau=J_b.T F` on
**p. 4, Eqs. (26), (29)**. These are existing mechanics, not a new force law.
Sources: [GIC v4](../../manipulator_manifold_paper/2211.07945v4.pdf),
[GUFIC v3](../../manipulator_manifold_paper/2504.17080v3.pdf).

### Frame and origin changes

For a constant coordinate change `r'=O r+t`, `O in O(3)`, `t` in metres, define
`D=det(O) O` and the skew map `[t]_cross`:

\[
 X=\begin{bmatrix}O&[t]_\times D\\0&D\end{bmatrix},\quad
 V'=XV,\quad F'=X^{-\top}F
 =\begin{bmatrix}Of\\D\mu+t\times(Of)\end{bmatrix}. \tag{G4}
\]

For `det(O)=1`, `X` is the usual SE(3) adjoint in this ordering; reflections use
the O(3) extension and are **not** SE(3) elements. Since `J'=XJ` and
`Phi_F'=X^{-T}Phi_F`, the joint pullback is unchanged under passive frame changes.
This preserves power even when the origin changes. Morphological Symmetries
**p. 16, Eq. (26)** explicitly distinguishes polar and axial transformations;
GIC **p. 3, Eq. (23)** supplies the proper adjoint. The reflection-plus-translation
formula above is derived from these conventions and power duality.

The estimator coefficients use a fixed gauge, not a new moving body chart at each
timestep. If a constant world force is represented by body-frame coefficients,
those coefficients generally vary as the body rotates. Put that known rotation
into the basis when a constant-coefficient hypothesis is intended. A varying
latent chart `W(t)` adds `dot W W^{-1} z'` and corresponding covariance terms.
Neither `LinearAction` nor the original constant-coordinate proof includes them.

### Descriptor compatibility: two explicit cases

**Position servo.** For `x=q`, `B=T`, `A=-I`, `E_0=I`,

\[
 T\dot q=-q+a_{\mathrm{sent}}+\Phi_a(\xi)z_d, \tag{G5}
\]

every residual row is in joint-position command units. **Inserting `Phi_tau`
directly here is dimensionally wrong.** `T` has time-scale units and is not mass.
There is no universal Jacobian-only conversion. An identified map
`C_a: generalized force -> command residual` gives

\[
 \Phi_a=C_a\Phi_\tau. \tag{G6}
\]

For the *additional*, qualified overdamped PD assumption
`K_d qdot=-K_p q+K_p a+tau_d`, invertible `K_p` gives
`T=K_p^{-1}K_d`, `C_a=K_p^{-1}`. This discards inertial transients and assumes
the remaining nominal terms have been compensated or modeled. The product `T`
need not be symmetric for coupled gains; use the actual contraction metric.
Arbitrary installed position servos do not establish this reduction. If `C_a`
is unavailable, use a direct learned basis **in command-residual units**, with
geometric context and symmetry, and make no physical-wrench attribution.

**Mechanical state.** The wrench basis belongs directly in

\[
 \underbrace{\begin{bmatrix}I&0\\0&M(q)\end{bmatrix}}_{\mathcal B(q)}
 \begin{bmatrix}\dot q\\\dot v\end{bmatrix}
 =\begin{bmatrix}v\\-K_pq-K_dv-C(q,v)v-g(q)-\tau_f(q,v)\end{bmatrix}
  +\begin{bmatrix}0\\K_p\end{bmatrix}a_{\mathrm{sent}}
  +\underbrace{\begin{bmatrix}0\\\Phi_\tau\end{bmatrix}}_{\Phi_d}z_d.
 \tag{G7}
\]

For direct torque commands the lower input block is `I`, with a different action
interface. `mechanical_rows` implements only this feature embedding. The full
model requires state `[q,v]`, nonlinear drift and `B(q)`, a matching residual and
integration rule, and an incremental/mechanical storage proof. In particular,
`integral B(x) xdot dt` cannot be replaced by `B(x_start) Delta x` in general.
The constant `NominalModel` in `core.py` does not implement this nonlinear plant.
It can represent a frozen/linearized local model with the operating-point bias
removed and with constant `B,A,E`. Hurwitz stability of that local model does not
prove contraction throughout the mechanical state space.

An inertia/pullback metric is not automatically the tracking metric `L`.
The Riemannian distance-field PDF **p. 4, Eqs. (11)–(15)** distinguishes such
metrics and makes the task pullback semidefinite before regularization;
**p. 14, Section 6.2** separates geometric guidance from constrained execution.
Neither result closes this adapter's nonlinear stability argument.
[Source v3](../../manipulator_manifold_paper/2412.05197v3.pdf).

## 2. Symmetry acts on the disturbance family

Take a verified finite morphological group `G`, with
`q'=rho_g q`, `v'=rho_g v`, signed permutation `rho_g`. Affine joint actions
must first be centered at consistent calibrated zeros. Permute only physically
compatible joints/limbs: types, axes, mass, healthy servo parameters, limits and
mounts must agree under the candidate transformation. External geometry, gravity,
payloads, contact identities and observations must transform too. The verified
subgroup can be the identity. No nontrivial group for ALOHA/Panda/GR1 is certified
by this implementation.

This qualification follows the Morphological Symmetries PDF **p. 6, Definition 1,
Proposition 1 and the following force/contact qualification; p. 8, Definition 2;
pp. 10–11, Eqs. (18)–(21), Fig. 7**. Its **p. 14, Section 6.1** transforms limb
identities and sensor quantities together.
[Source v4](../../manipulator_manifold_paper/2402.15552v4.pdf).

Choose representations, satisfying the same multiplication table,

\[
 x'=T_gx,\quad y'=S_gy,\quad a'=U_ga,\quad z'=W_gz. \tag{G8}
\]

For joint-position servo rows and targets, `T=S=U=rho`. For the mechanical
descriptor, `T=diag(rho,rho)` and `S=diag(rho,rho^{-T})`; signed orthogonality
makes the blocks numerically equal. Full joint targets/torques have `U=rho`.
A restricted action interface needs its own verified `U`, and may admit a smaller
group. `W` is a chosen latent representation, fixed with the basis gauge; it is
not determined by `rho` alone. A fixed nonorthogonal change of latent gauge is
allowed, so the formulas retain inverses and transposes separately.

The exact condition is, for all contexts, group elements and coefficients,

\[
 d(g\cdot\xi,W_gz)=S_gd(\xi,z)
 \quad\Longleftrightarrow\quad
 \boxed{\Phi(g\cdot\xi)W_g=S_g\Phi(\xi)}. \tag{G9}
\]

For wrench slots let `X_g` be the stacked twist transformation, including the
permutation of slots. Require the kinematic identity
`J(g q) rho_g=X_g J(q)` and wrench-feature identity
`Phi_F(g xi) W_d,g=X_g^{-T} Phi_F(xi)`. Then

\[
 \Phi_\tau(g\xi)W_{d,g}
   =\rho_g^{-\top}\Phi_\tau(\xi)=\rho_g\Phi_\tau(\xi). \tag{G10}
\]

The servo conversion must also obey
`C_a(g xi)=S_g C_a(xi) rho_g^T` (typically `rho_g C_a rho_g^T`). A false
compliance symmetry cannot be repaired by a symmetric wrench network.

**The family is equivariant; its individual member can be asymmetric.** In a
two-limb example with swap matrix `Pi`, choose a per-limb additive latent with
`W_d=Pi` and `Phi_d=I`. A left fault `(b,0)` is carried to a right fault `(0,b)`.
The model with the original fixed fault is not swap-invariant. More generally,
only the stabilizer of the particular coefficient, context, task and fixed
parameters remains a symmetry of that particular experiment. The environment's
distribution need not be symmetric either; an invariant statistical prior is a
separate modeling choice.

Do **not** impose `W_g z=z`, or project the online estimate/covariance onto the
invariant subspace. With `W=I` at a symmetric two-limb context, (G9) forces the
two rows of `Phi` to agree and cannot express `(b,0)`. An invariant scalar latent
can still be useful for a shared severity if an explicitly transforming fault
label or context carries the side information. Both semantics are coherent;
silently discarding the side information is not.

Symmetry of the adapter is conditional on its nominal command input. A frozen
VLA need not output `U_g a_nom` under a scene transformation. Left/right task
instructions may intentionally break that symmetry. The tutorial **p. 11,
Section 3.1.3** defines equivariant maps, and **p. 26, Section 6.2** identifies
constraints, camera geometry and occlusion as symmetry-breaking factors.
[Source v3](../../manipulator_manifold_paper/2503.09829v3.pdf).

## 3. A constructive basis, with a fixed latent gauge

An exact finite-group construction from any seed `f_psi(xi)` of shape `(n,p)` is

\[
 \Phi_\psi(\xi)=\frac1{|G|}\sum_{g\in G}S_g^{-1}f_\psi(g\cdot\xi)W_g. \tag{G11}
\]

Substituting `k=gh` proves `Phi(h xi)=S_h Phi(xi) W_h^{-1}`. Averaging an
already equivariant basis leaves it unchanged. This is the standard Reynolds
projection, not a new neural architecture. It costs `|G|` seed evaluations and
may remove useful components if the asserted symmetry or latent representation
is wrong. To preserve the wrench factorization explicitly, apply (G11) in the
stacked wrench representation `S_g=X_g^{-T}`, then use (G3), and finally (G6) or
(G7). A single arbitrary scalar invariant coefficient is not generally enough.

`act_context(g,xi)` must implement a left action on the whole schema (G2). A raw
image transformation must be justified by the camera/scene model; a generic VFM
embedding has no presumed channel representation. Re-evaluate the encoder on
the correctly transformed observation, use an encoder with a known action, or
restrict the claimed group. The library checks the finite matrix representations,
not correctness of a caller's physical context transformation.

Offline, one may train the seed through the existing support/query ridge objective
in FORMULATION Eq. (18), using the projected basis throughout. The ridge prior
transforms as `Omega'=W^{-T}Omega W^{-1}`, and each observation weight as below.
Freeze the seed and its coefficient gauge online. No training pipeline or trained
weights are produced here. Synthetic orbit copies are not independent excitation.

## 4. Effectiveness and allocation

Preserve the typed decomposition `z=[z_E;z_d]`,
`W_g=diag(W_E,g,W_d,g)`, with `E_j(xi)` of shape `(n,m)`:

\[
 E(\xi,z_E)=E_0(\xi)+\sum_{j=1}^{p_E}z_{E,j}E_j(\xi),\qquad
 \Phi_{\rm all}(\xi,a)=[E_1(\xi)a,\ldots,E_{p_E}(\xi)a,\Phi_d(\xi)].
 \tag{G12}
\]

The nominal and full maps must satisfy

\[
 E_0(g\xi)=S_gE_0(\xi)U_g^{-1},\quad
 E(g\xi,W_{E,g}z_E)=S_gE(\xi,z_E)U_g^{-1}. \tag{G13}
\]

Equating latent coefficients gives the **tensor** law

\[
 \boxed{E_k(g\xi)=\sum_j(W_{E,g}^{-1})_{jk}\,S_gE_j(\xi)U_g^{-1}},\qquad
 \Phi_d(g\xi)=S_g\Phi_d(\xi)W_{d,g}^{-1}. \tag{G14}
\]

Thus `Phi_all(g xi,U_g a) W_g=S_g Phi_all(xi,a)`. The coefficient index must
transform in addition to the rows and columns. A per-joint scalar gain deficit
normally transforms by the **unsigned** joint permutation `abs(rho)`, while an
additive signed joint offset transforms by `rho`. For example,
`E=I+diag(z_E)`, `rho=-Pi` still takes a left deficit `(-0.5,0)` to `(0,-0.5)`;
it does not turn reduced effectiveness into increased effectiveness.

The library rejects mixing effectiveness and additive blocks in `W`: a general
mixture destroys their separate command dependence and the explicit allocator.
`project_effectiveness` implements the Reynolds projection of this tensor and
the additive block together using (G14). Current `a_sent` enters only after
allocation, so this preserves the linear solve and avoids an algebraic loop.

Given `d_hat=Phi_d z_hat_d`, the unique nearest nominal feasible action is

\[
 a_{\rm sent}=\mathop{\arg\min}_{a}\tfrac12(a-a_{\rm nom})^\top M_a(a-a_{\rm nom})
 \quad\text{subject to}\quad
 \widehat E a=E_0a_{\rm nom}-\hat d,\qquad M_a\succ0. \tag{G15}
\]

`core.matched_action` implements this equality-constrained problem and raises on
missing authority. Under transformed data, `E_hat'=S E_hat U^{-1}`,
`d_hat'=S d_hat`, `a_nom'=U a_nom`, and `M_a'=U^{-T}M_a U^{-1}` imply
`a_sent'=U a_sent`: both feasibility and objective are preserved, and the SPD
metric makes the minimizer unique. This holds for redundant and rank-deficient
input maps whenever the target is feasible. It creates no missing input direction.

Bounds would require the feasible set to transform to `U A`; a bounded solver
is not implemented here. For a signed permutation, transform a box by
`lower'=U_+ lower+U_- upper`, `upper'=U_+ upper+U_- lower`, where positive and
negative parts retain their signs. Transform command conversion, masks and
priors too. Keeping an asymmetric bound fixed instead yields a different problem.
Numerical rank/feasibility thresholds are finite-precision choices; the identities
are exact-real algebra, verified within tolerances away from rank transitions.

## 5. Every estimator and controller object

The following are **forced transports once `T,S,U,W` and the original design
are chosen**, not independently adjustable gains after a transformation.

| Object | Transport | Which part is a design choice? |
|---|---|---|
| True/estimated coefficients | `z'=W z`, `z_hat'=W z_hat` | Latent dimension, fixed gauge and representation `W`; do not assume invariance |
| Estimate gain/covariance | `P'=W P W.T` | Initial SPD `P_0`; subsequent `P` follows the Riccati law |
| State/reference/error | `x'=T x`, `x_r'=T x_r`, `e'=T e` | Valid modeled state and reference definition |
| Tracking metric | `L'=T^{-T} L T^{-1}` | SPD `L` satisfying the actual nominal contraction inequality |
| Residual and basis | `y'=S y`, `Phi'=S Phi W^{-1}` | Row units, learned seed and typed physical model |
| Measurement weight | `R'=S R S.T` | SPD noise/design weight; its precision transforms by `S^{-T} R^{-1} S^{-1}` |
| Process weight | `Q_z'=W Q_z W.T` | SPD design/intensity in coefficient units |
| Forgetting operator | `Lambda'=W Lambda W^{-1}` | Forgetting rates/operator; it is not a covariance |
| Descriptor and drift | `B'=S B T^{-1}`, `A'=S A T^{-1}` | Identified nominal model; a fixed symmetric model must obey these identities |
| Input effectiveness | `E_0'=S E_0 U^{-1}`, `E_hat'=S E_hat U^{-1}` | Representation/fitting of nominal and faulted input maps; tensor law (G14) is forced |
| Allocation metric | `M_a'=U^{-T} M_a U^{-1}` | Initial SPD distance in action units |
| Constraints/conversion | `A_set'=U A_set`; `Gamma'(raw')=U Gamma(raw)` | Original actuator constraints and raw-input convention |
| Ridge prior/center | `Omega'=W^{-T} Omega W^{-1}`, `z_0'=W z_0` | Regularization and prior center (the implemented leakage center is zero) |
| Dissipation/noise/error terms | `Q_e'=T^{-T} Q_e T^{-1}`, `epsilon_y'=S epsilon_y`, `epsilon_x'=T epsilon_x` | Original model/error assumptions and scales |

Here `R,Q_z` have the continuous-law meanings from `core.py`. A window covariance
must be derived/calibrated for that window and then transported; it is not equal
to `R` simply because it has the same shape. Constant-row integrated observations
obey `Y'=S Y`, `Psi'=S Psi W^{-1}`, including consistent quadrature and timestamps.

**Transport of a design family versus one fixed symmetric design.** Arbitrary
anisotropic priors/weights are allowed if transported in the comparison. If the
same numerical hyperparameters are to be reused under every group element, they
must be invariant under the table's respective actions. For example,

\[
 T_g^\top L T_g=L,\quad S_g R S_g^\top=R,\quad
 W_g Q_z W_g^\top=Q_z,\quad W_g\Lambda=\Lambda W_g,\quad
 U_g^\top M_aU_g=M_a. \tag{G16}
\]

A symmetry-neutral initialization also has `W_g z_hat_0=z_hat_0` and
`W_g P_0 W_g.T=P_0`. Such initialization is optional for a family of transported
initial conditions. Subsequent `P(t),z_hat(t)` need not be invariant on a single
asymmetric trajectory. Their values on *transformed histories* are related by
the transport. Do not average them across limbs during online adaptation.

Scalar identities are sufficient in orthogonal coordinates, not in an arbitrary
nonorthogonal latent gauge. Equal diagonal gains on exchanged coordinates are
required for an invariant diagonal design; distinct rates in invariant latent
blocks are allowed. More general invariant matrices can have off-diagonal terms.
Neither symmetry nor an isotypic decomposition justifies dropping covariance
cross-blocks created by the measured information.

SPD invariant covariance designs can be constructed by group averaging
`W_g Q_0 W_g.T`; metrics by averaging `T_g^{-T} L_0 T_g^{-1}`. However, an
averaged tracking metric still needs a contraction check. For a *constant*,
equivariant Hurwitz `F=B^{-1}A` commuting with all `T_g`, averaging a valid
Lyapunov metric preserves its strict dissipation. This argument does not apply
to arbitrary scheduled dynamics or an arbitrary inertia matrix.

### Commutation of the composite update

Let `H=B^{-1}Phi`, `nu=y-Phi z_hat`. Then

\[
 H'=THW^{-1},\quad\nu'=S\nu,\quad
 \Phi'^\top R'^{-1}\nu'=W^{-\top}\Phi^\top R^{-1}\nu,\quad
 H'^\top L'e'=W^{-\top}H^\top Le. \tag{G17}
\]

Consequently the existing law, with its positive tracking term,

\[
 \dot{\hat z}=-\Lambda\hat z+P\Phi^\top R^{-1}\nu+PH^\top Le,\quad
 \dot P=-\Lambda P-P\Lambda^\top+Q_z-P\Phi^\top R^{-1}\Phi P,
\]

satisfies

\[
 \boxed{\dot{\hat z}'=W\dot{\hat z},\qquad \dot P'=W\dot P W^\top}. \tag{G18}
\]

No commutation of `Lambda` with `P` was used. For constant group actions and
unique solutions, transformed initial conditions and input histories therefore
give transformed solution histories. A common scalar Euler increment also
commutes algebraically; this is not a positivity or sampled-stability guarantee.
The implementation continues to expose a continuous RHS, not a certified stepper.

The storage `V=0.5 e.T L e+0.5 (z-z_hat).T P^{-1}(z-z_hat)` is unchanged by
the transports. Thus the exact Lyapunov identity and its qualifications in
FORMULATION Eq. (13) remain the same; geometry introduces no extra stabilizing
term, unbiased-convergence theorem or superiority over prediction-only filtering.
If one changes to a nonlinear mechanical nominal model or a state-dependent
tracking metric, the necessary drift/metric derivative terms must be established
separately. The wrench embedding alone does not provide them.

## 6. Prior work and the narrow contribution

The literature comparisons below were checked against these exact PDF pages,
not inferred from paper titles or from the earlier review's prose.

| Existing component | Source that occupies it | What this specification adds locally |
|---|---|---|
| Meta-learned shared force basis with online linear coefficients | Neural-Fly v2, **pp. 18–19**, Eqs. (4)–(6), Algorithm 1 | A typed geometric basis and a declared nontrivial action on fault coefficients |
| Prediction-plus-tracking adaptation with a Riccati-like gain | Neural-Fly **p. 20**, Eqs. (7)–(9) | Consistent descriptor-row/state metrics already in FORMULATION; no new composite update is claimed |
| Visual/context matrix basis, short-window coefficient fitting, and effectiveness control | MAGIC-VFM v2, **pp. 4–5**, Eqs. (2)–(4), Algorithms 1–2; full composite law **p. 9**, Eq. (26) | Morphological action on all three tensor indices, plus compatible latent/estimator/allocation transport |
| Multiple learned additive coefficient blocks | HMAC v2, **p. 4**, Eqs. (6)–(10) | No hierarchy novelty claimed; retain the symmetric noncommuting forgetting formula from the existing core |
| Morphological representations, equivariant networks and augmented measurements | Morphological Symmetries v4, **pp. 10–11, 14, 16**, Eqs. (18)–(21), (26) | A specified composite coefficient estimator and allocator coupled to that model-family symmetry; that paper's presented learned estimators are supervised momentum/contact models, not this online composite fault estimator |
| Wrench pullbacks, adjoint/dual transport and geometric force control | GIC **pp. 2–3**, GUFIC **p. 4** as cited above | Correct placement in command or mechanical descriptor units; no new wrench geometry or passivity theorem |
| Geometry-consistent observers | Gada et al., *Equivariant Filters are Equivariant*, arXiv:2210.13728v1, **p. 1, abstract** | This report does not claim to invent equivariant estimation or coordinate/noise-consistent filters |

Adaptation primary PDFs:
[Neural-Fly](../../2205.06908v2.pdf),
[MAGIC-VFM](</home/fengze/online_adaptation_research/references/MAGIC-VFM: Meta-learning Adaptation for Ground Interaction Control with Visual Foundation Models/2407.12304v2.pdf>),
[HMAC](</home/fengze/online_adaptation_research/references/Hierarchical meta-learning-based adaptive controller/2311.12367v2.pdf>).
Their version hashes are recorded in [source metadata](adaptation_sources_metadata.json).
The additional observer precedent was retrieved from its
[primary PDF](https://arxiv.org/pdf/2210.13728v1), whose first page identifies v1.

The defensible contribution **within this repository and relative to these named
architectures** is a fully specified composition: a force/command distinction,
an equivariant family with asymmetric fault coefficients, and a compatible
observation–Riccati–tracking–effectiveness–allocation path at the modeled policy
interface, with exact commutation tests. These are a method specification and
implementation contribution. The transport identities and Reynolds projection
are standard algebra, and several transports were already written in FORMULATION
Eqs. (20)–(21). Combining known components is not enough, by itself, to establish
a publishable conceptual novelty or priority over all equivariant adaptive control.
No first-of-kind, transfer benefit, data-efficiency benefit or robot result is
claimed. Such claims need further prior-art comparison and new held-out evidence.

## 7. Implemented scope

[`geometry.py`](../../learned_adaptation/geometry.py) provides `SpatialFrameChange`,
`wrench_pullback`, `mechanical_rows`, `servo_rows`, `LinearAction`,
`EffectivenessFeatures`, and `FiniteSymmetry`. Shapes are explicit; incompatible
frames must be resolved by the caller and incompatible shapes/group laws raise.
NumPy arrays have no physical unit tags: names and explicit conversion arguments
prevent a default conversion but cannot detect a caller's falsely labeled values.

[`test_geometry.py`](../../learned_adaptation/test_geometry.py) checks virtual
power, reflection and origin transport, contact permutations, descriptor conversion,
Reynolds equivariance/idempotence, effectiveness tensors, actual recomputation of
group-transformed features before estimator updates, storage, reference/residual/
window/allocation commutation, asymmetric faults, and invalid inputs. Its S3
tests include noncommuting group elements and nonorthogonal latent gauges.
Existing core tests remain intact. Run both suites:

```sh
python -m unittest learned_adaptation.test_core learned_adaptation.test_geometry -v
```

The tests certify algebraic identities within numerical tolerance on synthetic
arrays. They do not identify a physical `C_a`, check a robot's kinematics/limits,
implement a visual context action, train a feature network, or certify nonlinear
contact control. Those limits are part of the method's assumptions, not results
to infer from the tests.
