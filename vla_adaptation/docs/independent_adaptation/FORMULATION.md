# Learned dynamics and composite adaptation at a frozen-policy interface

Design date: 2026-09-08. This is the formulation for the independent
`fxxie2/vla_adaptation` project. It is a proposed method with a tested mathematical
reference implementation, not a trained controller or new robot result.

The starting point is the user's internal control model. The change is substantive:
identify the nominal actuator/servo dynamics, learn representations of the remaining
dynamics across environments, and adapt their coefficients online. The inherited FIR
residual smoother remains a historical baseline. The new estimator is not obtained
by renaming its channel estimates. A linear combination of learned basis functions
is retained because that is the representation used by the proposed model.

## 1. Conventions that fix the signs and dimensions

Use the more general local descriptor model

\[
 \mathcal B\dot x=\mathcal A x+E_0 a_{\rm sent}
                    +\Phi(\xi)z+\epsilon_d. \tag{1}
\]

| Quantity | Dimension | Meaning |
|---|---|---|
| \(x\) | \(n\) | Modeled actuator/robot state, with coordinates and units fixed |
| \(a_{\rm nom}\), \(a_{\rm sent}\) | \(m\) | Nominal policy command expressed at the modeled boundary, and corrected command in those same coordinates |
| \(\mathcal A,\mathcal B\) | \(n\times n\) | Nominal dynamics; \(\mathcal B\) is nonsingular in this local model |
| \(E_0\) | \(n\times m\) | Nominal available-input map in descriptor-row units |
| \(\Phi(\xi)\) | \(n\times p\) | Learned disturbance representation |
| \(z,\hat z\) | \(p\) | Actual latent coefficients and their online estimate |
| \(\xi\) | specified per model | State, morphology, visual/contact context and, when needed, the sent command |
| \(\epsilon_d\) | \(n\) | Representation error and remaining nominal-model error |

The policy emits \(\bar a=\pi(o)\). Define \(a_{\rm nom}=\Gamma(\bar a)\), where
\(\Gamma\) contains known conversion and nominal command limits at the chosen
model boundary. Both commands in (1) use those same units. If correction must be
requested upstream of a nonlinear \(\Gamma\), its feasible inverse/allocator must
be included; subtracting a metre-valued correction from a normalized action is invalid.
Alternatively model that nonlinear command map explicitly in the nominal dynamics.

The user's \(B\dot x=Ax+a\) is the special case \(m=n\), \(E_0=I\).
It does **not** hold just because a robot has joint-position observations: the
available input can have a different dimension and different units.

With a **plus** disturbance in (1), the exact full-input compensation is

\[
 \boxed{a_{\rm sent}=a_{\rm nom}-\Phi(\xi)\hat z.} \tag{2}
\]

Substitution gives
\(\mathcal B\dot x=\mathcal A x+a_{\rm nom}
+\Phi(z-\hat z)+\epsilon_d\).
A plus correction instead gives \(\Phi(z+\hat z)\): a correct estimate doubles the
disturbance. A plus correction is consistent only if its coefficients are explicitly
defined to estimate **negative disturbance**, with every identification/update sign
changed accordingly. We use the disturbance convention throughout.

For \(E_0\ne I\), solve \(E_0 c=\Phi\hat z\) and send \(a_{\rm nom}-c\).
Exact cancellation needs both the required input directions and feasible bounds.
A pseudoinverse cannot recover a missing direction or a locked actuator. A nearest
nominal solution preserves the unused input nullspace. The reference implementation
raises when the unconstrained equality is infeasible; deployment needs a bounded
allocator and explicit accounting for its remaining error.

## 2. What the nominal internal model should represent

### A first-order position servo is a useful special case

For a qualified overdamped position servo,

\[
 T\dot q=-q+a_{\rm sent}+d_a,\qquad T\succ0. \tag{3}
\]

Here \(\mathcal B=T\), \(\mathcal A=-I\), and \(d_a\) has command-coordinate
units. \(T\) is a servo time-scale matrix, **not** the rigid-body inertia matrix.
A bias/equilibrium term is generally needed; augment the model or center coordinates
before fitting. Coupled servos may require a full matrix or a scheduled nonlinear
model. The eigenvalues of \(\mathcal B^{-1}\mathcal A\), the prediction errors and
the model's validity region all require validation.

A first-order \(q\)-only model is an approximation. At identical \((q,a)\), a
second-order mechanism can have different velocities; contact and actuator memory
can also produce different futures. A learned basis cannot generally make an omitted
state Markov without access to that state, a history encoder, or an explicitly
mode-dependent model.

### Physical joint disturbances generally need mechanical dynamics

For position-target access, start from

\[
 M(q)\ddot q+C(q,\dot q)\dot q+g(q)+\tau_f(q,\dot q)
 =K_p(a_{\rm sent}-q)-K_d\dot q+\tau_{\rm ext}+\tau_{\rm fault}. \tag{4}
\]

This is illustrative PD servo structure, not a claim that every installed controller
uses constant gains or lacks feedforward compensation. For \(x=[q^\top,v^\top]^\top\),
one corresponding descriptor form is

\[
 \begin{bmatrix}I&0\\0&M(q)\end{bmatrix}\dot x
 =\begin{bmatrix}v\\-K_pq-K_dv-C(q,v)v-g(q)-\tau_f(q,v)\end{bmatrix}
  +\begin{bmatrix}0\\K_p\end{bmatrix}a_{\rm sent}
  +\begin{bmatrix}0\\\Phi_\tau(\xi)\end{bmatrix}z. \tag{5}
\]

This is nonlinear and its input map is rectangular. It cannot be rewritten as the
user's constant \(B\dot x=Ax+a\) with \(x=[q,v]\) without changing the meaning of
\(a\) or adding assumptions. With direct torque access, replace the lower input block
\(K_p\) by \(I\), while retaining the different action semantics. Such a change
requires a separately reported benchmark protocol.

URDF/MJCF and controller configuration supply morphology, joint types and limits,
inertial priors, transmissions, actuator ranges and often nominal gains. They do not
by themselves identify real servo lag, friction, backlash, hardware gain calibration,
communication delay, or contact with the current object. Use those files to structure
the nominal model and use independent healthy recordings to identify its effective
parameters. A matrix inferred from an input/state fit is not automatically a mass
matrix.

If the identified first-order model is \(\dot x=Fx+Ga\), one can set
\(B=G^{-1}, A=G^{-1}F\) only when \(G\) is square and nonsingular. Otherwise retain
\(F,G\) or the explicit descriptor/input maps. Enforcing an invertible \(G\) merely
to obtain the preferred notation would hide restricted actuation.

### Background identification must not absorb the fault

Fit nominal parameters on healthy support data; check predictions and residual scale
on separate healthy validation data, including contact modes used at deployment.
Freeze that fit during a confirmation experiment. Simultaneously adapting \(A,B\)
and \(z\) to the same unexplained motion is generally nonidentifiable: either model
can absorb the discrepancy. Later background re-identification needs qualified
healthy intervals, versioned models, and an explicit transition protocol. Historical
faulted test data cannot become training data and remain untouched confirmation.

## 3. A healthy reference permits position-only feedback under explicit assumptions

For the constant local model, run the internal healthy response to the **same nominal
command stream**:

\[
 \mathcal B\dot x_r=\mathcal A x_r+E_0a_{\rm nom},\qquad
 x_r(0)=x(0),\qquad e=x-x_r,\quad \tilde z=z-\hat z. \tag{6}
\]

Assume cancellation is feasible and define

\[
 F=\mathcal B^{-1}\mathcal A,\qquad
 H(\xi)=\mathcal B^{-1}\Phi(\xi).
\]

Then

\[
 \dot e=F e+H\tilde z+\epsilon_x,\qquad
 F^\top L+LF=-Q_e,\quad L\succ0,\ Q_e\succ0. \tag{7}
\]

\(\epsilon_x\) includes model mismatch, actuation/allocation error and unmodeled
reference discrepancy. The healthy linear system is contractive in the constant
metric \(L\). For a valid first-order \(x=q\) model, \(e=q-q_r\) uses position only;
a velocity-combination error is **not mandatory**. However, the properly scaled
feedback entering the parameter law is

\[
 \boxed{H^\top Le=\Phi^\top\mathcal B^{-\top}Le,} \tag{8}
\]

not automatically \(\Phi^\top e\). If \(\mathcal B\) is symmetric positive definite
and \(\mathcal A+\mathcal A^\top\prec0\), choose \(L=\mathcal B\), and (8) reduces
to \(\Phi^\top e\) **in a compatible nondimensionalized coordinate convention**.
More generally choose \(L=\kappa\mathcal B\) with the required scale/units;
then the score is \(\kappa\Phi^\top e\). This includes (3) after choosing the
coordinate/time scales. After a unit change, transform \(L\) by (21); do not
reset it to the numerical value of the new \(B\). A generic Hurwitz \(F\) admits a Lyapunov
metric, but that metric need not equal \(B\).

For a nonlinear nominal field \(f_h(x,a)\), use its actual incremental contraction
condition. Pointwise stability of a fitted \(A(x)\) is insufficient. With a constant
metric, require the symmetric metric Jacobian inequality over the validated state
and command region. With a state-dependent metric, its directional time derivative
also appears. Contact switching/reset maps need their own storage conditions.

The reference above is an internal response to commands produced on the **actual**
trajectory. It is not the visual policy's counterfactual healthy rollout. Stable
tracking of that response does not certify stability of the entire vision–policy
loop or successful manipulation. If second-order state is required, position-only
feedback needs an additional observer/storage argument; simply deleting velocity
from a mechanical-controller proof is not valid.

## 4. Residual measurement must subtract the command actually sent

For (1), use

\[
 y=\mathcal B\dot x-\mathcal A x-E_0a_{\rm sent}
   =\Phi z+\epsilon_y,\qquad
 \nu=y-\Phi\hat z. \tag{9}
\]

\(a_{\rm sent}\) means the known command after **our** allocation, coordinate
conversion and clipping at the modeled boundary, aligned with the observation.
An unknown downstream offset is part of the disturbance; do not subtract a
simulator-privileged post-fault command and thereby erase the target being estimated.
Any known actuator lag belongs in the model or timestamp alignment.

Subtracting \(a_{\rm nom}\) instead would measure a remaining corrected disturbance,
approximately \(\Phi(z-\hat z)\). Treating that as a measurement of \(\Phi z\) would
produce a second subtraction of the estimate and the wrong fixed point. This is a
separate issue from the choice of estimator.

Avoid differentiating noisy positions unnecessarily. For constant descriptor
matrices and a slowly changing coefficient within \([t_k,t_{k+1}]\), use

\[
 Y_k=\mathcal B(x_{k+1}-x_k)
       -\int_{t_k}^{t_{k+1}}(\mathcal A x+E_0a_{\rm sent})\,dt,
 \quad \Psi_k=\int_{t_k}^{t_{k+1}}\Phi(\xi(t))\,dt,
 \quad Y_k\simeq\Psi_kz_k+v_k. \tag{10}
\]

Integrate the feature along the window; using only its last value is another
approximation. A varying \(z\) creates a window-model error. For \(B(x)\), the left
side is \(\int B(x)\dot x\,dt\), not \(B(x_k)(x_{k+1}-x_k)\). An alternative is to
integrate the already inverted state dynamics. Overlapping windows and endpoint
state noise are correlated; their observation covariance must be modeled or
calibrated, not copied from a per-step derivative filter.
The reference code offers explicitly labeled trapezoidal endpoint interpolation
and left-interval quadrature. For commands held constant between updates, the next
command issued at \(t_{k+1}\) did not generate motion on the preceding interval.
The left rule takes only the commands/features for completed intervals and incurs
its stated state/feature quadrature error. For action-dependent features, their
quadrature must respect these same input discontinuities.

## 5. The composite coefficient law and its covariance dynamics

Let \(P(0)\succ0\), \(R\succ0\), \(Q_z\succ0\). Let \(\Lambda_z\) describe
coefficient forgetting; a usual choice is \(\lambda I\), or different positive
rates for explicitly modeled latent blocks. For (7) and (9), use

\[
 \boxed{\dot{\hat z}=-\Lambda_z\hat z
       +P\Phi^\top R^{-1}(y-\Phi\hat z)
       +P\Phi^\top\mathcal B^{-\top}Le,} \tag{11}
\]

\[
 \boxed{\dot P=-\Lambda_zP-P\Lambda_z^\top
       +Q_z-P\Phi^\top R^{-1}\Phi P.} \tag{12}
\]

These are regularization, prediction-error learning and position-tracking learning,
with one sign convention throughout. The scalar forgetting case reduces to
\(-2\lambda P+Q_z-P\Phi^\top R^{-1}\Phi P\). For unequal block rates,
\(-2\Lambda_zP\) is generally wrong because it need not be symmetric. A covariance
with off-diagonal blocks must use both terms in (12).

This follows the **continuous composite structure** of Neural-Fly, adapted to the
user's servo model and its metric. The original Neural-Fly mechanical tracking error
is different. Exact source expressions and version-specific inconsistencies are
recorded in [the source comparison](adaptation_sources.md); do not copy a displayed
discrete expression without deriving its signs, time factors and covariance units.

Removing the final term gives the corresponding prediction-only Kalman–Bucy-style
parameter estimator. Keeping \(\Phi=I\) gives a constant-basis comparison. Neither
reduction makes the learned controller statistically optimal for task success.
Under composite feedback, \(P\) is an adaptation/preconditioning matrix with an
estimator interpretation, not automatically the true posterior error covariance.

### Why the positive tracking term is correct

Use

\[
 V=\tfrac12 e^\top Le+\tfrac12\tilde z^\top P^{-1}\tilde z.
\]

The state derivative contains \(+e^\top LH\tilde z\). Because
\(\tilde z=z-\hat z\), the tracking part of (11) contributes
\(-\tilde z^\top H^\top Le\), canceling it exactly. Including noise and drift gives

\[
\begin{aligned}
 \dot V={}&-\tfrac12e^\top Q_e e
 -\tfrac12\tilde z^\top\Phi^\top R^{-1}\Phi\tilde z
 -\tfrac12\tilde z^\top P^{-1}Q_zP^{-1}\tilde z\\
 &+e^\top L\epsilon_x
 +\tilde z^\top P^{-1}(\dot z+\Lambda_z z)
 -\tilde z^\top\Phi^\top R^{-1}\epsilon_y. \tag{13}
\end{aligned}
\]

The matrix-forgetting terms cancel through the derivative of \(P^{-1}\), without
assuming that \(P\) and \(\Lambda_z\) commute. The reference tests check this exact
identity for 128 random cases, including nonidentity/nonsymmetric descriptors,
unequal forgetting rates and disturbance/noise terms.

Equation (13) is a deterministic/pathwise identity for ordinary error signals.
It is not an Itô stochastic Lyapunov calculation. Interpreting a continuous
observation as white noise requires an SDE formulation and its diffusion terms;
the ordinary derivative and bounded-noise conclusion cannot be reused unchanged.

Under uniform positive lower/upper bounds on \(P\), bounded features and the stated
model/contraction conditions, (13) supports a conditional ultimate tracking bound
in terms of model error, measurement noise and \(\dot z+\Lambda_z z\). It is not a
new universal robot stability theorem. With nonzero forgetting and a constant
nonzero disturbance, the driving term includes \(\Lambda_z z\); **exact unbiased
parameter convergence is not promised**. With no forgetting and an unexcited latent
direction, covariance upper bounds and coefficient identification cannot simply be
assumed. Excitation, feature identifiability and representation error still matter.

An arbitrary gain multiplying only the tracking term breaks the displayed
cancellation unless the storage/metric and other terms are adjusted consistently.
If coefficient constraints are needed, the continuous proof uses a tangent-cone
projection in the \(P^{-1}\) metric with the true coefficient in the feasible set.
Ordinary coordinatewise clipping under a full \(P\) does not automatically satisfy
that projection argument. Saturating the robot command is also a separate effect
in \(\epsilon_x\).

### A discrete implementation must be derived separately

One consistent first-order splitting starts with

\[
 F_z=e^{-\Lambda_z h},\quad
 Q_d=\int_0^h e^{-\Lambda_z s}Q_ze^{-\Lambda_z^\top s}\,ds,\quad
 \hat z^-=F_z\hat z_k,\quad P^-=F_zP_kF_z^\top+Q_d.
\]

For the integrated observation in (10), with calibrated covariance \(R_k^I\),

\[
 K_k=P^-\Psi_k^\top(\Psi_kP^-\Psi_k^\top+R_k^I)^{-1},\quad
 \hat z^{\rm pred}=\hat z^-+K_k(Y_k-\Psi_k\hat z^-),
\]
\[
 P^+=(I-K_k\Psi_k)P^-(I-K_k\Psi_k)^\top+K_kR_k^IK_k^\top,
\]
\[
 \hat z_{k+1}=\hat z^{\rm pred}+hP^+H_{k+1}^\top Le_{k+1}. \tag{14}
\]

The next action uses this updated estimate; an observation cannot update the action
that already generated it. For an ideal continuous white residual-noise intensity
\(R_c\) and constant feature over a window, \(R^I=hR_c\), \(\Psi=h\Phi\);
using the averaged observation instead gives covariance \(R_c/h\). Endpoint state
noise generally has a different covariance. Mixing these conventions changes the
gain by powers of the timestep.

Equation (14) is a candidate splitting, **not** an exact discretization or a proved
sampled-data stability result. The effect of a drifting coefficient within the
window, delay, covariance bounds, constraints and contact must be checked before
robot integration. The current reference code deliberately implements the
continuous right-hand side and window regression, not a purportedly certified
robot update. No old tuning parameters should be copied without converting units.

## 6. Include effectiveness changes: the part closest to MAGIC

An additive state-only disturbance is not enough for reduced motor gain or a
changing input matrix. Use the model

\[
 \mathcal B\dot x=\mathcal A x+
 \underbrace{\left(E_0+\sum_{j=1}^{p_E}z_{E,j}E_j(\xi)\right)}_{E(\xi,z_E)}a_{\rm sent}
 +\Phi_d(\xi)z_d+\epsilon_d. \tag{15}
\]

Equivalently, its linear coefficient regressor is

\[
 \Phi_{\rm all}(\xi,a_{\rm sent})=
 [E_1(\xi)a_{\rm sent},\ldots,E_{p_E}(\xi)a_{\rm sent},\Phi_d(\xi)],
 \qquad z=[z_E^\top,z_d^\top]^\top. \tag{16}
\]

The compensation equation is

\[
 \widehat E(\xi,\hat z_E)a_{\rm sent}
       =E_0a_{\rm nom}-\Phi_d(\xi)\hat z_d. \tag{17}
\]

For the explicit linear solve, require that \(E_j(\xi)\) and \(\Phi_d(\xi)\)
depend on known state/history/context, not on the unknown current \(a_{\rm sent}\).
The action dependence in (16) is then explicit and linear. If a feature network
also consumes the current sent command internally, both (2) and (17) become
implicit nonlinear control equations. Use a justified nonlinear solve or a
specified delayed-command approximation; its approximation error remains in the model.

If square and safely nonsingular, solve this equation directly; otherwise use a
feasible bounded allocation and report its residual. This handles the action's
appearance in the regressor without an unexamined algebraic loop: assemble the
estimated matrix and solve for the command, then evaluate (16) on that command.
With exact allocation, subtracting the healthy reference gives the same error
structure as (7), with \(H=B^{-1}\Phi_{\rm all}\). Use this same feature in the
prediction and tracking update. Under allocation residual \(r_a\), add
\(B^{-1}r_a\) to \(\epsilon_x\).

The summation in (15) specifies a learned matrix basis; it is not the inherited
residual-averaging update. It can equally be written as a tensor contraction.
For a scalar gain \(g\), exact compensation is \(a_{\rm sent}=a_{\rm nom}/\hat g\),
not a state-only constant subtraction. Near zero effectiveness, inversion becomes
unbounded; at a lock there is a lost feasible direction. Parameter constraints,
singular-value monitoring, a bounded optimizer and fallback behavior belong in the
controller design. Learning more accurately does not restore that direction.

## 7. The gripper needs an actuator and contact model

The local interface audit found a concrete ALOHA mismatch: normalized finger target
\(a_g\) maps to \(0.01844+0.03956a_g\) metres, while the configured actuator range is
\([0.021,0.057]\) metres. In normalized coordinates this is approximately
\([0.06471,0.97472]\). The historical healthy right-gripper requests exceed the upper
range on 889 of 2,253 recorded steps (39.46%). These are continuous signals.
See [the interface audit](interface_model_audit.md) for pinned source/configuration
provenance and counting convention. This mismatch motivates a controlled test; it
does not alone explain the entire historical squared-residual share.

A nominal gripper model should contain:

1. The known coordinate conversion, mimic/tendon coupling, command saturation and
   any incremental target accumulator or rate limit.
2. Servo dynamics, with velocity/actuator state included if a first-order reduction
   is inaccurate.
3. Free-motion, contact and joint-stop behavior, with object geometry/contact
   context when needed. Intended contact is not an unexplained fault by definition.
4. Noise weights in stated physical coordinates. Arm radians and normalized finger
   targets should not share an unscaled residual norm.

For example, a free-motion first-order reduction can be
\(T_g\dot g=-g+\operatorname{clip}(\psi(\bar a_g))+\phi_g(\xi)z_g\), where
\(\bar a_g\) is the raw gripper request and the clipped conversion is the modeled
position-target input.
For contact, a mechanical model instead adds a constraint/contact force and can
require second-order dynamics. Contact forces that maintain a grasp must be
preserved or regulated against a desired force, not automatically canceled as an
external disturbance. Contact mode can condition the nominal model, the tracking
metric and what is treated as nuisance dynamics. A mode-changing metric requires
its own stability accounting; the constant-metric result above does not cover it.

The old global normalizer is absent from (11), so the particular path by which an
uncorrected gripper attenuated every arm target is removed. New cross-channel
coupling can still occur through a learned basis, covariance or shared latent.
Use calibrated observation weights and physically justified feature blocks, then
measure arm/gripper interference. A new model can reduce deterministic nominal
mismatch; it does not prove that grasp success increases.

## 8. What is learned offline, and what is adapted online

**Healthy fit:** morphology/configuration priors plus separate healthy data estimate
the nominal dynamics, command conversion, contact modes and observation weights.
A sufficient first-order stable parameterization is \(B\succ0\) and
\(A=J-D\), \(J=-J^\top\), \(D\succ0\); then a suitably scaled \(L=\kappa B\)
works in the specified coordinates. This restriction must
fit the measured dynamics well enough; it is not imposed on arbitrary mechanics.

**Shared feature training:** across disturbance environments, learn the feature
parameters \(\psi\) of \(\Phi_\psi\), potentially including the input-matrix features
in (16). For support samples \(S_e\) from environment/window \(e\), a dimensionally
consistent ridge coefficient fit is

\[
 z_e^*(\psi)=\left(\sum_{i\in S_e}\Phi_i^\top R_i^{-1}\Phi_i+\beta\Omega\right)^{-1}
             \sum_{i\in S_e}\Phi_i^\top R_i^{-1}y_i,
 \quad \Omega\succ0. \tag{18}
\]

For effectiveness learning, each \(\Phi_i\) includes its **known sent action**;
the unknown \(z_e\) is not inserted into its own regression design matrix.
Optimize query prediction loss after fitting on support, and, where a simulator
and valid differentiable controller permit it, a separately specified control
objective. Keep the adaptation horizon, available sensors and information matched
to deployment. Avoid fitting and reporting on the same query trajectory.

Neural-Fly motivates invariant shared features with environment-specific linear
coefficients. MAGIC motivates visual/context-dependent **input effectiveness**;
its feature treatment need not be invariant to all environment information.
HMAC motivates distinct manageable/contextual and latent/time-varying disturbance
representations, plus streaming adaptation. Do not blindly impose invariance on
visual terrain/contact information whose variation the model needs to use.
A hierarchy can concatenate \([\Phi_m,\Phi_l]\) and use different coefficient
priors/forgetting rates. Correlated blocks can remain coupled through \(P\).

**Online:** freeze nominal models and feature weights for the initial study; update
only \(\hat z,P\), state estimates and internal reference state. Later training of
feature weights during deployment would be another method requiring its own
stability and evaluation treatment. Coefficient identifiability also has a basis
gauge: \(\Phi T^{-1},Tz\) represent the same disturbance. Evaluate physical
prediction, tracking and task outcome; compare coefficient error only when the
basis and parameter convention are fixed.

## 9. Geometric structure: where it helps and where it does not

The completed [geometric method specification](GEOMETRIC_METHOD.md) makes this
section constructive: declared wrench frames/origins and context, explicit
force-to-servo conversion or mechanical embedding, finite-group basis projection,
effectiveness tensor transformations, asymmetric fault semantics, and transport
of the estimator and allocator. It is implemented in
[`geometry.py`](../../learned_adaptation/geometry.py). In particular,
\(J_c^\top\Phi_F\) has generalized-force units and cannot be inserted unchanged
in the first-order position-servo residual. The abstract identities below do not
identify a robot's symmetry or justify a nonlinear mechanical contraction model.

The [five-paper manifold review](manifold_review.md) distinguishes several different
objects. For a manipulator, use joint configuration space and its tangent/cotangent
coordinates for joint state/velocity/torque, \(SE(3)\) for end-effector pose, and the
appropriate twist/wrench transformations. A joint vector is not itself an \(SE(3)\)
pose. Limited revolute joints are intervals; only genuinely periodic coordinates
admit unrestricted angle wrapping.

A useful representation of an external wrench contribution is

\[
 \tau_d=J_c(q)^\top F_c,\qquad
 F_c=\Phi_F(\xi)z,\qquad \Phi_\tau(q,\xi)=J_c(q)^\top\Phi_F(\xi). \tag{19}
\]

This gives a physically structured map from learned wrench features to joint
forces, provided the contact point/frame and nuisance-versus-desired-force meaning
are defined. It is not the inverse map from joint torque to a restricted Cartesian
command. Use the latter interface's own actuation map for allocation.

For a legitimate symmetry \(g\), a basis/latent pair should satisfy

\[
 \Phi(g\!\cdot\!\xi)\rho_z(g)=\rho_d(g)\Phi(\xi). \tag{20}
\]

Latents tied to particular joints must transform under a valid limb permutation;
assuming all latents are invariant would mislabel left/right faults. A robot
morphology, gains, limits and environment may break a proposed symmetry. Gravity,
contact normals, object poses and external forces must transform with the state
when claiming coordinate covariance. Rotating only the robot while holding gravity
fixed is usually a different physical experiment. Visual foundation features do not
automatically obey the required group representation.

For a proper rigid transformation, a spatial twist transforms by the adjoint and a
wrench by its inverse transpose so that power pairing is preserved. Reflection is
not an element of \(SE(3)\). Under an orthogonal reflection, polar force and axial
torque have different rules: \(f'=Rf\), \(\tau'=\det(R)R\tau\). Joint-axis signs,
limits and handedness must be transformed consistently before using mirrored data.

### The estimator must transform along with the basis

For constant invertible coordinate changes \(x'=Tx\), row units \(y'=Sy\),
action \(a'=Ua\), latent \(z'=Vz\), set

\[
 B'=SBT^{-1},\quad A'=SAT^{-1},\quad E_0'=SE_0U^{-1},\quad
 \Phi'=S\Phi V^{-1},
\]
\[
 L'=T^{-\top}LT^{-1},\quad P'=VPV^\top,\quad R'=SRS^\top,
 \quad Q_z'=VQ_zV^\top,\quad\Lambda_z'=V\Lambda_zV^{-1}. \tag{21}
\]

Then (11)–(12) give \(\dot{\hat z}'=V\dot{\hat z}\),
\(\dot P'=V\dot PV^\top\). In particular,
\(H'=THV^{-1}\) makes the tracking term transform correctly. This identity is
verified numerically with nonorthogonal coordinate and latent changes. It is a
useful consistency requirement for a geometry-aware adaptation implementation;
it is not a claim that all coordinate changes are physical robot symmetries.
State-dependent coordinate changes, Lie-group errors and switching modes need
transport/connection terms or an appropriate geometric storage proof.

For the **whole controller**, also transform the action constraint set to
\(U\mathcal U\), the known command conversion consistently, and the nearest-action
metric to \(M_a'=U^{-\top}M_aU^{-1}\). An ordinary Euclidean nearest-action rule
is not invariant under nonorthogonal input scaling when the input has a nullspace.
The reference allocator accepts \(M_a\) and tests its transformation. Offline ridge
priors likewise require \(\Omega'=V^{-\top}\Omega V^{-1}\). Learned features alone
do not provide these controller/training invariances.

Promising contribution: couple an equivariant learned representation with the
**whole** observation, coefficient, metric and action-allocation transformation,
then test whether it improves data efficiency and transfer at the actual policy
interface. This is more specific than adding a geometric feature to Neural-Fly.
No novelty priority is claimed without comparison to existing equivariant
adaptive-control work.

### Mirror descent is a distinct optional geometry

A reflection symmetry and optimization by mirror descent are different concepts.
For a fixed strongly convex potential \(h\), set
\(\eta=\nabla h(\hat z)\) and, without forgetting, consider

\[
 \dot\eta=\Phi^\top R^{-1}(y-\Phi\hat z)+H^\top Le. \tag{22}
\]

For constant true \(z\), use
\(V=\tfrac12 e^\top Le+D_h(z,\hat z)\), where
\(D_h(z,\hat z)=h(z)-h(\hat z)-\nabla h(\hat z)^\top(z-\hat z)\).
Its parameter derivative is \(-\tilde z^\top\dot\eta\); tracking terms cancel and
the noiseless prediction term is \(-\|\Phi\tilde z\|_{R^{-1}}^2\).
This yields a separate fixed-geometry composite design. Coefficient drift,
regularization and a time-dependent potential add terms that need analysis.

Do not multiply a learned inverse Hessian by an unrelated Riccati \(P\) and assume
both proofs survive. Their matrix product need not be symmetric and the storage
identity changes. A quadratic fixed potential recovers a fixed-gain Euclidean law;
a time-varying quadratic potential involves \(\dot P^{-1}\). Start with one derived
geometry, then compare alternatives with matched information and tuning budgets.

## 10. Scientific scope and next experiment

This formulation plausibly addresses nominal servo mismatch, state-dependent
residuals, gripper saturation and some input-effectiveness failures. It does not
establish repair of a joint lock, missing Cartesian actuation directions, contact
force misclassification, or a policy whose healthy trajectory already fails.

The first new study should validate a joint-target model on ALOHA, including the
actual finger conversion/limits and contact regimes, then compare constant and
learned bases under the **same** prediction-only/composite laws. Add effectiveness
learning as a separate factor. Test all arm joints and grippers, healthy controls,
held-out object/contact conditions, matched random seeds and full reset fingerprints.
Panda's current Cartesian interface is a distinct setting with explicit authority
limits; any new torque-access experiment must be labeled as such. The complete
study and prospective paper plan are in [the project entry](README.md).

No historical success number is evidence for (11) with a learned feature network.
The implementation in `learned_adaptation/core.py` is a reference for equations
(1), (9), (11), (12), (17) and the quadrature in (10). Its tests establish algebraic
and numerical consistency in synthetic cases, not stability of real robot dynamics,
learned-feature quality or an estimator ranking.
