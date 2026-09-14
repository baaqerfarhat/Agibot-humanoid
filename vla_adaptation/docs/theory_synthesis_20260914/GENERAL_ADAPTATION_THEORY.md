# A general adaptation theory for frozen VLA execution

Analysis date: 2026-09-14. Evidence snapshot: commit b540d52.
This is a proposed synthesis and a retrospective analysis, not a preregistered
prediction or a certificate for the evaluated robots. The existing manuscript
and historical results are unchanged.

Read the [standalone theory PDF](general_adaptation_theory.pdf) for a compact
mathematical draft, and the [constructed examples](memory_example.pdf) for
the difference between servo decay, accumulated transient error, and joint
feedback stabilization.

**Recommendation.** Build the paper around a finite-horizon incremental
Lyapunov/storage analysis of the **combined execution and adaptation system**.
Strict physical contraction is one sufficient special case. It is unnecessary
for finite-horizon improvement, and it should not be an assumption of the
general theorem. Keep disturbance estimation, feasible correction, execution
deviation, and task success as four separate statements.

There is also a stronger Lyapunov route: an appropriately signed composite
update can stabilize a joint system even when the execution subsystem has a
neutral mode. This is a conditional design result, not an explanation already
verified by the existing composite experiments.

## 1. What the newest evidence actually establishes

| Evidence | Result | Consequence for the theory |
|---|---|---|
| Historical four-suite command offsets | 28/120 to 78/120; disjoint-calibration rerun 34/120 to 68/120 | Repair exists; calibration dependence must enter the uncertainty term. These are different cohorts/protocols, not a paired test between calibrations. |
| Healthy-data channel gate | Faulted 9/20 to 19/20; healthy 20/20 to 19/20; any-channel healthy opening about 30% | Channel selection can use healthy data, but the gate has no demonstrated 3-sigma instantaneous false-opening guarantee. Its calibration statistic was a 50-step mean. |
| Physical Panda joint faults | Joint-5 legacy 15/60 to weighted 41/60, off 36/60; Object j3 legacy 16/20 to weighted 2/20, off 8/20 | Estimation and allocation quality do not imply physical improvement or task improvement. Retain the authority analysis. |
| Corrected friction/lock protocol | Friction 0/40 to 13/40; lock 0/20 to 0/20 | Some below-controller faults remain repairable. The lock result alone does not prove a rank-loss mechanism; authority must be measured or derived. |
| Matched-estimate ALOHA study | Held 15/40, continued innovation 14/40, continued legacy 0/40 | Holding is not necessary on this cell. Bias, update dynamics, and task phase matter; correction movement alone cannot explain the outcomes. Similar totals do not prove equivalence. |
| Same-command Panda replays | Median step ratios near 1; roughly half the free-space weighted end-effector ratios at least 1 | No uniform strict contraction certificate in the tested metrics. A stable motion-increment model can coexist with persistent pose displacement. |
| Archived tube/small-gain analysis | Final bound about 98 times the largest logged residual; nominal small-gain margin about -0.00095 | These calculations do not certify recovery. Their measured variable and constants also differ from what a physical-state theorem requires; see Section 2. |
| ARX intervention | Spatial 9/20 to 20/20, but rotation-y estimate falls to 32% of truth; exploratory libero_10 0/40 to 11/40 | Better healthy prediction does not establish better disturbance identification. The proposed AR-model-order explanation was refuted. |
| Pinned FIR versus static observer | 10/80 versus 9/80; five method-only, four static-only; exact McNemar p=1 | No demonstrated FIR-memory advantage in this test. Do not claim equivalence or that the earlier gap was definitively caused by sampling noise. |
| Pinned method versus raw integral | 18/20 versus 3/20; fifteen method-only, none integral-only; p=0.0000610 | This comparator supports calibrated model-based correction. It does not isolate the necessity of six FIR taps or one particular update law. |
| Earlier descriptor prototype | Spatial 112/140 to 135/140; Object j3 7/20 to 18/20 | Favorable integrated-controller evidence with a supplied constant basis; no learned-feature, geometry, or isolated composite-term benefit is established. |
| Earlier matched ALOHA tuning | Composite minus Kalman -0.95 percentage points, interval [-9.52,+7.62] | An extra tracking term has no established task benefit and needs a joint stability analysis. |

Primary local records: [new theory results](../../results/re4_theory/README.md),
[new evidence](../../results/re4_evidence/README.md),
[historical and new experiment record](../ADAPTIVE_CONTROL_VLA.md),
[physical-fault confirmation](../../report/FOLLOWUP_STATUS.md),
[matched tuning](../../report/COMPOSITE_TUNING.md), and
[descriptor audit](../branch_comparison_20260909/results_audit.md).
Pinned runs still have some GPU execution nondeterminism and condition on a
fixed sampler realization. Repeated tasks/fault cells require clustered
inference; a count increase is not automatically an independent treatment effect.

## 2. Corrections needed before using the existing theory scores

Inspection of [recursions.py](../../openpi/re4_theory/recursions.py) reveals
four distinctions that matter to a paper theorem.

1. Part 3 sets the measured deviation to a weighted **FIR residual**, not the
   deviation between a faulted physical trajectory and a matched healthy
   simulator replay. It takes lambda from median perturbation ratios and L
   from mean sensitivities. Its eta is a percentile from the same faulted
   telemetry, including both arms. These are not uniform, independent bounds
   in a consistently validated physical metric.
2. Part 4.2 uses an innovation-like factor 1-gamma*s and constructs an error
   using r-M(f-fhat), while the implemented legacy law predicts the total
   fault and has coefficient 1-gamma on the old estimate. Its 100% coverage
   does not validate the implemented legacy recursion as written.
3. Part 5 regresses the norm of M times the estimate update, rather than the
   predictor-mismatch quantity specified in the evidence plan. Ordinary
   least-squares coefficients are not uncertainty envelopes. Failure of this
   particular sufficient-condition diagnostic does not prove instability.
4. Files called metric certificates contain fitted per-channel models and
   held-out prediction checks. They do not certify a nonlinear, coupled
   physical execution model throughout a domain. In particular, the Panda
   fit concerns motion increments, not decay of absolute pose differences.

We independently replayed both actual update equations from telemetry:

| Run | Adaptive steps | Maximum update error, all six channels | Corrected bound coverage | Median bound/error |
|---|---:|---:|---:|---:|
| T1 legacy | 2,634 | 2.78e-17 | 100% | 1.237 |
| T4 innovation | 2,292 | 2.78e-17 | 100% | 1.182 |

These corrected envelopes use the **observed same-run uncertainty** and known
injected fault. They verify arithmetic and the decomposition below, not
prospective uncertainty calibration. Median last-50-step rotation-y observation
bias is -0.02723 and -0.02785 against a +0.05 fault, respectively: changing the
update does not remove this bias.

Reproduction: [audit_recursions.py](audit_recursions.py);
output: [recursion_audit.json](recursion_audit.json).

## 3. Separate observation from the physical effect of correction

Let a_k be the frozen policy command, c_k the correction to subtract, and
u_k=a_k-c_k the known command sent. The predictor must use u_k. Its residual
r_k is an **observation**, not the physical error remaining after correction.

For a local linear model at a specified state and horizon, write

\[
r_k=H_k\theta_k+\varepsilon_k,\qquad
v_k=F_k\theta_k-D_kc_k .
\tag{1}
\]

Here theta is a disturbance parameter, H its observation map, F its physical
effect, and D the physical effect of the available correction inputs. Model
memory, contact dependence and calibration error can make epsilon depend on
the executed command and the adaptation state. It must not silently be treated
as independent sensor noise.

For a nominal allocation K_k and arbitrary actual feasible correction c_k,
the following is an exact identity:

\[
v_k=(F_k-D_kK_k)\theta_k+
       D_kK_k(\theta_k-\hat\theta_k)+
       D_k(K_k\hat\theta_k-c_k).
\tag{2}
\]

These are, respectively, a mismatch in the assumed cancellation map, estimation
error propagated into physical coordinates, and allocation/realization error.
The first term can be nonzero even when exact cancellation by another
allocation is possible. A genuine authority floor is
inf over feasible c of ||F theta-Dc||, not the error of one chosen inverse.

In the noiseless, unconstrained linear setting, a residual-based linear rule
c=Lr can cancel every disturbance exactly iff

\[
\ker H\subseteq\ker F,\qquad
\operatorname{range}F\subseteq\operatorname{range}D.
\tag{3}
\]

Proof: necessity follows from indistinguishable disturbances and the correction
range. For sufficiency, define T on range H by T(H theta)=F theta, which is
well-defined by the kernel inclusion; lift T through D and extend linearly.
With constraints, pointwise cancellation additionally requires the particular
F theta to belong to D C. A single feasible linear rule over a disturbance
set further requires L H theta to belong to C throughout that set; pointwise
feasibility does not establish that common linear rule. This retains the existing
[authority appendix](../../paper/authority_appendix.tex).

A native decoder edit and external action subtraction are separate cases.
The latter can be exact at the request boundary while downstream saturation
and controller memory remain. The measured native-decoder realization error
(median zeta about 0.43) must be included if that path is claimed.

For multiplicative command faults, the executed discrepancy is
(g/ghat-1)a, so an estimate bounded away from zero gives
|discrepancy| <= |a| |g-ghat|/gmin.
The sensitivity of the compensator a/ghat to its estimate is instead
|a|/ghat^2. These are different bounds. A zero true effectiveness can remove
authority even if the estimated inverse is numerically bounded.

## 4. Estimator lemma: leakage and uncertainty are different terms

For the command-equivalent additive case, define

\[
z_k=M^{-1}r_k-b=f_k+w_k,\quad
e_k=\hat f_k-f_k,\quad E_k=\|e_k\|,\quad
\nu_k=\|f_{k+1}-f_k\|.
\]

Assume f_k belongs to the estimator's closed convex projection set C and use
Euclidean projection/norm, or a matched projection and norm in another metric.
Let ||w_k|| <= epsilon_k. Without such a bound, the following is an algebraic
decomposition but not a predictive certificate.

**Legacy update.** With 0<gamma<=1 and effective attenuation s_k in [0,1], including the
zero-target deadzone,

\[
\hat f_{k+1}=\Pi_C[(1-\gamma)\hat f_k+\gamma s_kz_k],
\]
\[
e^{\rm pre}_{k+1}
=(1-\gamma)e_k+\gamma s_kw_k
 -\gamma(1-s_k)f_k-(f_{k+1}-f_k),
\]
\[
E_{k+1}\le(1-\gamma)E_k+\gamma s_k\epsilon_k+
             \gamma(1-s_k)\|f_k\|+\nu_k .
\tag{4}
\]

The pre-projection identity is exact. The bound follows from projection
nonexpansiveness relative to f_k and then adding drift. An implementation
that explicitly holds inside the deadzone uses a separate hold branch.

**Innovation update.** With alpha_k in [0,1], including alpha=0 when gated,

\[
\hat f_{k+1}=\Pi_C[\hat f_k+\alpha_k(z_k-\hat f_k)],
\qquad
E_{k+1}\le(1-\alpha_k)E_k+\alpha_k\epsilon_k+\nu_k .
\tag{5}
\]

Here alpha_k=gamma*s_k is evaluated on the implementation's innovation norm.
There is no legacy leakage term. For constant alpha>0 and bounds epsilon,nu,
the steady error bound is epsilon+nu/alpha. This supports the ramp-lag
interpretation, conditional on an independent mismatch bound. In the presence
of persistent w, innovation is not an unbiased fault estimator.

For constant f,s and zero-mean exogenous w, the unclipped legacy mean tends
to s*f. More generally its stationary moment is E[s_k z_k], not a formula
obtained by substituting a mean residual into a nonlinear normalizer.
Deadzone intervals, finite windows and projection change that calculation.

For hold from a calibrated estimate at time T,

\[
\|\hat f_T-f_{T+j}\|\le E_T+\sum_{\ell=T}^{T+j-1}\nu_\ell .
\tag{6}
\]

Delay/snapshot age has the same bound with T=k-tau. Holding removes update
noise and feedback during that interval, but retains calibration error and
accumulates sensitivity to fault drift. It need not improve over a well-behaved
innovation update. RLS, Kalman and descriptor/composite observers fit the same
framework through their own matrix error recursions; equations (4)-(5) must
not be applied to them without deriving those recursions.

## 5. Main theorem: finite-horizon execution and adaptation storage

Let Z_k contain the deviations needed to make the local system Markov:
physical state, relevant controller/decoder memory, predictor history,
estimator state, and, when comparing closed-loop tasks, policy/chunk state.
Take a specified nominal trajectory and a horizon N.

Assume the actual incremental dynamics on a declared domain admit

\[
Z_{k+1}=\mathcal A_k Z_k+\mathcal B_k d_k+\rho_k .
\tag{7}
\]

The signed matrices include estimator-execution coupling. The inputs d and
the remainder rho account for unmodeled dynamics, disturbance drift,
unmatched effects, nonlinear approximation and stochastic innovations.
This is an assumption to validate, not an identification supplied by an
end-effector FIR fit. A bound must hold throughout the reachable domain,
not merely at the logged center trajectory.

Choose P_k positive definite, with reported coercivity bounds on 0..N, and set

\[
V_k=Z_k^\top P_k Z_k,\quad
a_k=\|P_{k+1}^{1/2}\mathcal A_kP_k^{-1/2}\|_2,\quad
\|\mathcal B_kd_k+\rho_k\|_{P_{k+1}}\le b_k.
\]

**Theorem.** If R_0 >= sqrt(V_0), then for every k<=N,

\[
\sqrt{V_k}\le R_k,\qquad R_{k+1}=a_kR_k+b_k,
\tag{8}
\]
\[
R_n=
\left(\prod_{j=0}^{n-1}a_j\right)R_0+
\sum_{k=0}^{n-1}
\left(\prod_{j=k+1}^{n-1}a_j\right)b_k .
\tag{9}
\]

Proof: the induced-norm inequality and triangle inequality give
sqrt(V_{k+1}) <= a_k sqrt(V_k)+b_k; induction proves both statements.
If the assumptions are local, use a stopping-time argument: the calculated
tube must stay inside their domain to conclude no exit before N.

This is a finite-horizon storage bound. It permits a_k=1 or a_k>1.
It neither proves asymptotic stability nor becomes useful merely because a
time-varying metric exists. Shrinking P_k cannot hide growing physical error:
state/output conversion and the minimum eigenvalue of P_k must be included.

For a known signed local model, preserve the transition products
Phi(n,k)=A_{n-1}...A_k and the expression
Z_n=Phi(n,0)Z_0+sum Phi(n,k+1)(B_k d_k+rho_k)
as long as possible. Multiplying scalar per-step norms discards cancellations
and can be much more conservative. Channel-aware reachable sets or an
augmented quadratic metric are preferable when validated data support them.

**Contraction corollary.** If a_k<=a<1 and b_k<=b, then
R_n<=a^n R_0+b(1-a^n)/(1-a). This recovers the older contraction tube.
It is a special case, not the entry condition of the theory.

**Physical-error interface version.** If a separately justified execution
inequality is X_{k+1}<=lambda_k X_k+L_k||v_k||+eta_k,
use b_k=L_k||v_k||+eta_k and equation (2). The input is the physical
remaining disturbance v_k, not the observed predictor residual r_k.
Physical-unit conversion and cross-axis input sensitivity belong in L_k.

**Hybrid execution.** Include contact/controller resets explicitly:
X^+<=j_k X^-+d^J_k. Insert that map in the product bound. A common metric
avoids artificial metric jumps; different metrics require their transition
factors. Gating or holding does not automatically satisfy a dwell-time
theorem, and differing contact sequences need separate domain assumptions.

## 6. Why this covers both position servos and integrating interfaces

Consider the exact scalar illustrative channel with 0<alpha<=1:

\[
p_{k+1}=\lambda p_k+g(f-\hat f_k),\qquad
\hat f_{k+1}=(1-\alpha)\hat f_k+\alpha f,\quad \hat f_0=0.
\]

Then f-hat f_k=f(1-alpha)^k and, for p_0=0,

\[
p_n=
\begin{cases}
gf\,\dfrac{\lambda^n-(1-\alpha)^n}{\lambda-(1-\alpha)},
 &\lambda\ne1-\alpha,\\
gf\,n\lambda^{n-1}, &\lambda=1-\alpha.
\end{cases}
\tag{10}
\]

For 0<=lambda<1 the displacement eventually decays. For lambda=1,

\[
p_n=\frac{gf}{\alpha}[1-(1-\alpha)^n],
\qquad p_\infty=\frac{gf}{\alpha}.
\tag{11}
\]

An unbiased observer stops new drift but leaves an offset caused by its
transient. No time-invariant metric, or family uniformly equivalent to the
physical norm over an infinite horizon, can turn the exact unforced integrator
into strict contraction with a uniform rate below one.

With 0<gamma<=1 and constant legacy attenuation 0<=s<1, the same integrating channel gives

\[
p_n=gf\left[(1-s)n+
              \frac{s}{\gamma}(1-(1-\gamma)^n)\right].
\tag{12}
\]

The two terms are persistent bias and transient identification cost. This
explains why mean error, variance, convergence speed, interface memory and
task horizon all matter; estimator range alone is insufficient.

For an ideal integrator with hold and drift bound nu, the subsequent h steps
contribute at most

\[
|g|\,[hE_T+\nu h(h-1)/2]
\tag{13}
\]

to displacement. For a stable servo, replace these sums by geometrically
weighted sums. Identify-then-hold **with a task reset** also removes the
previous execution deviation; pausing the estimate halfway through an
already damaged task does not.

These are analytical examples, not fitted causal descriptions of a robot.
Panda's neutral same-command pose response and WidowX's accumulated target
are related memory phenomena but are not identical controller implementations.

## 7. A joint Lyapunov theorem can be stronger than subsystem contraction

This provides a constructive Lyapunov-based extension worth including in the
paper, provided its limited empirical status remains explicit.

For an ideal neutral execution channel with normalized known input gain one,
constant fault, exact observation z=f, and no projection, let delta=hat f-f.
Use actual next-state tracking error through the implementable update
hat f_next = hat f + alpha(z-hat f) + kappa e_next:

\[
e_{k+1}=e_k-\delta_k,\qquad
\delta_{k+1}=(1-\alpha)\delta_k+\kappa e_{k+1}.
\tag{14}
\]

The augmented transition is

\[
\mathcal A=
\begin{bmatrix}1&-1\\\kappa&1-\alpha-\kappa\end{bmatrix}.
\]

**Proposition.** This system is Schur stable iff

\[
0<\alpha<2,\qquad 0<\kappa<4-2\alpha .
\tag{15}
\]

Proof: det A=1-alpha and tr A=2-alpha-kappa. The three strict
second-order Jury inequalities reduce to alpha>0, kappa>0,
and 4-2alpha-kappa>0, which also imply alpha<2.
Consequently for every Q positive definite there is P positive definite
satisfying A^T P A-P=-Q. Thus V=Z^T P Z is a strict joint Lyapunov
function even though the physical subsystem alone is neutral.

With additive error w and q=lambda_min(Q),

\[
V_{k+1}-V_k
\le-\frac q2\|Z_k\|^2+
\left(\|P\|+\frac{2\|\mathcal A^\top P\|^2}{q}\right)\|w_k\|^2 .
\tag{16}
\]

Proof: expand the quadratic and apply Young's inequality to
2 Z^T A^T P w. This is a conditional joint ISS bound. If alpha or kappa
vary, stability of each frozen matrix is insufficient; establish a common
metric or a justified time-varying/multiple-metric condition.

At kappa=0 the prediction-only controller retains the neutral eigenvalue.
Too much composite feedback is unstable. Both match the reason an extra
tracking term requires analysis rather than a presumed superiority claim.
The stable GR1 fitted servo is another possible model class, not a prerequisite.

This proposition is not yet a certificate for the descriptor/composite
robot controllers: their reference, state, covariance, clipping, timing and
input map differ. In particular, a tracking error formed only from motion
increments may not observe the neutral **pose** mode. The physical reference
needed by (14) must actually be available, and its error must be bounded.

## 8. The final step to task recovery needs an additional assumption

There are three different comparisons:

1. Healthy predictor versus measured output: produces the residual r.
2. Healthy physical execution driven by the **same issued nominal commands**:
   defines an execution discrepancy, but its reference need not succeed.
3. A successful healthy **closed-loop policy trajectory**: appropriate for
   task recovery, but the faulted policy generally issues different commands.

For comparison 3, include policy/chunk/observation feedback in the augmented
dynamics, or separately bound the command difference. Frozen weights do not
make the commands identical. Conditioning on a shared sampler realization
is useful for coupling trajectories but is not a population success theorem.

**Conditional task-transfer corollary.** Suppose a successful healthy
trajectory has a validated robustness radius mu in a specified weighted
supremum trajectory metric, and the physical-output conversion of the bound
in (9) gives a deviation strictly below mu in that metric, with compatible contact
and reset assumptions. Then the corrected trajectory succeeds.
Proof: it remains inside the assumed success neighborhood.

For example, use d_task=max_k ||W_k(x_k-x_k^0)|| and require
max_k ||W_k C_k P_k^(-1/2)|| R_k < mu when x_k-x_k^0=C_k Z_k.
Other trajectory metrics require their own conversion, including horizon
factors for accumulated or L2 error.

This assumption must include grasp timing, object configuration, orientation
and any relevant contact history. A static-offset sweep measures a restricted
family, not a margin against arbitrary time-varying deviations. Lower
position RMS does not prove this corollary.

The manuscript's condition "correction range sigma < task margin mu" should
therefore be presented as a restricted empirical diagnostic, not a general
necessary or sufficient condition. A constant wrong estimate has zero range
and can fail; exact cancellation of a varying fault can have large correction
range and still reproduce healthy execution. Equations (9)-(13) supply the
proper interface- and time-dependent replacement.

At a population level, if the sufficient conditions hold outside a set of
coupled scenarios of probability at most delta, then
P(corrected success) >= P(healthy success)-delta by the union bound.
Existing success counts do not identify such a delta or establish equivalence.

## 9. Rotation-y: a concrete model-consistency test, not a settled mechanism

The gain mismatch is real: healthy fitted DC gain about 0.10-0.12, probed
sensitivity about 0.276. But their ratio is not, by itself, the equilibrium
of every adaptive loop.

For a fixed-command scalar or matrix example with true map G, fitted map
Ghat, bias b0, and c=hat f,

\[
r=Gf+(G-\widehat G)(a-\hat f)+b_0 .
\]

An unclipped innovation observer calibrated with S has stationary equation

\[
[S+G-\widehat G]\hat f=Gf+(G-\widehat G)a+b_0 .
\tag{17}
\]

For S=G, a=0 and b0=0 this gives
hat f/f=1/(2-Ghat/G), about **0.61**, not 0.37 when Ghat/G=0.37.
The approximate ratio Ghat/G can emerge under additional command/trajectory
conditions, for example when feedback restores the measured output so that
the known command shifts by approximately -f. The ARX residual also has
different conditioning and sensitivity scaling.

Therefore the observed ratios support a model-consistency problem, but the
short residual equation in record 50 does not derive the claimed equilibrium.
Retain command distribution, state dependence and predictor memory in the
analysis; test a DC-consistent model intervention before assigning causality.

## 10. What to put in the paper and what to measure next

The main text can carry (2), the estimator bounds (4)-(5), and the finite-horizon
storage/task-transfer result (8)-(9). Put the joint Lyapunov construction
(14)-(16), gain-mismatch derivation and proofs in the appendix. Contraction
becomes one corollary. A suitable claim is:

> Calibrated action adaptation can reduce the disturbance presented to a frozen
> policy's execution interface. Whether that reduction repairs a task depends
> on feasible correction directions, residual-model bias, execution memory,
> and the trajectory's remaining tolerance. We provide finite-horizon
> conditions covering contracting and noncontracting interfaces, and a
> conditional joint Lyapunov analysis for composite adaptation.

The comparison inequalities and Lyapunov tools are standard. The defensible
contribution is the faithful model of the implemented interface, correct
estimator recursions, explicit separation of the four claims, and experiments
that test their assumptions. This note does not establish novelty relative to
all literature or universal real-robot recovery.

Three discriminating next measurements, to preregister before running:

1. **DC consistency with matched budgets.** Compare unconstrained and
   probe-constrained predictor gains, sharing data, observer, clipping and
   fresh paired scenarios. Measure observed target bias, actual local
   correction response, healthy harm and task success. Improved rotation-y
   identification without improved long-horizon success would refute a
   sufficient "identification price" explanation.
2. **Actual trajectory error and memory.** From matched complete simulator
   states, record healthy physical replay and corrected physical trajectories.
   Distinguish increment error, pose error and controller target state.
   Test the signed cumulative disturbance prediction and uncertainty bounds
   on held-out episodes. Do not score a residual as a state deviation.
3. **Joint feedback on a qualified interface.** Compare prediction-only,
   composite and hold from the same initial estimate with matched tuning and
   fresh healthy/faulted cohorts. Validate the augmented dynamics and a
   common quadratic metric, including reference error and contact limits.
   Measure task success independently of tracking improvement.

Use multiple sampler seeds as well as paired initial states, with task-cluster
uncertainty. Do not select an intervention from these outcomes and describe
its evaluation on the same scenarios as independent confirmation.

## 11. Primary theoretical grounding

- [Forni and Sepulchre, A differential Lyapunov framework for contraction
  analysis, TAC 2014](https://arxiv.org/pdf/1208.2943): Theorem 1 distinguishes
  nonstrict incremental stability from asymptotic and exponential contraction.
  Dropping task-relevant coordinates to obtain a pseudometric does not bound
  their physical error.
- [Forni and Sepulchre, On differentially dissipative dynamical systems,
  2013](https://arxiv.org/pdf/1305.3456): storage with an input supply provides
  the appropriate conceptual extension to forced deviations.
- [Geiselhart and Wirth, Relaxed ISS Small-Gain Theorems for Discrete-Time
  Systems](https://arxiv.org/pdf/1406.3224): finite-step analysis can exploit
  interconnection dynamics without requiring every subsystem to be ISS.
  It does not make a true unforced integrator asymptotically stable.
- [Cheng et al., Improving the Robustness of Reinforcement Learning Policies
  with L1 Adaptive Control, RA-L 2022](https://arxiv.org/pdf/2112.01953):
  pretrained-policy adaptive augmentation is established; its guarantees
  depend on model/regularity and matched-disturbance assumptions.
- [Slotine and Li, Composite adaptive control of robot manipulators,
  Automatica 1989](https://doi.org/10.1016/0005-1098(89)90094-0):
  prediction-plus-tracking adaptation is classical; its manipulator proof
  does not automatically apply to this discrete stochastic policy interface.
- [Hespanha and Morse, Stability of Switched Systems with Average
  Dwell-time, CDC 1999](https://www.ece.ucsb.edu/~hespanha/published/avedwell.pdf):
  switching guarantees require explicit mode and transition conditions;
  they do not automatically validate identify-and-hold.

Numerical checks of the constructed identities, bounds and joint Lyapunov
example accompany this note. They check the mathematics implemented here,
not robot certification. See [verify_theory.py](verify_theory.py) and
[verification.json](verification.json). A compact standalone LaTeX draft is
[general_adaptation_theory.tex](general_adaptation_theory.tex).

From the repository root, using the existing LIBERO environment:

    ../openpi/examples/libero/.venv/bin/python docs/theory_synthesis_20260914/audit_recursions.py
    ../openpi/examples/libero/.venv/bin/python docs/theory_synthesis_20260914/verify_theory.py

Both print results; --out explicitly writes a receipt. The mathematical
checker additionally accepts --figure followed by a PDF path.
