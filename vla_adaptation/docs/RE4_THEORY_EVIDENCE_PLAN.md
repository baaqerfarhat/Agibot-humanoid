# re4 theory-evidence plan — instantiating the certificates on the real plants

**Target.** The re4 submission's theory (execution tube, estimator recursions,
small-gain coupling, held-estimate bound, authority conditions, composite
extension). The first evidence plan closed the *empirical* hedges; this one
closes the gap the paper still states in its own words: *"trajectory-dependent
predictor mismatch and the evaluated robots' metrics remain unverified"* and
*"a physical execution metric and its domain remain to be established for the
reported VLA robot/controller."* Everything here turns a conditional theorem
into a measured instance — or into a recorded refutation of its constants.

**Rules, same as before.** Global logging contract of `RE4_EVIDENCE_PLAN.md` §0
applies to every run. Every part gets a prereg in `prereg_records/` with the
statistic, band, and refutation condition written before running. Outputs under
`results/re4_theory/<part>/...`, committed incrementally as
`re4 theory: <part> <cohort>`. Refutations are primary. Where truth injection is
used, it is evaluation-only, never fed to the estimator.

**Shared tooling to build once (T.0).** A paired-rollout driver that (a) saves a
simulator state, (b) runs the *same recorded command sequence* from the saved
state and from a perturbed state, with matched RNG, and (c) logs the state gap
per step; plus the nominal-replay driver the paper already specifies (replay
recorded $u_k$ through the healthy model from matched initial conditions) to
produce $\xi_k^{\rm nom}$ and $X_k=d(\xi_k,\xi_k^{\rm nom})$ for real episodes.
Both are prerequisites for Parts 1–4.

---

## Part 1 — Measure the execution contraction constants (the paper's biggest IOU)

**Theory targeted:** Theorem (scheduled same-command execution tube),
Eq. (sampled-contraction), Eq. (sampled-transition); the hybrid caveats.

1. **Sampled contraction rate.** On LIBERO Panda (OSC) and on the GR1 arm servo:
   from a mid-episode saved state, perturb the physical state by small $\delta\xi$
   (grid over magnitude and direction), run the same commands, and record the
   per-step gap ratio in a candidate metric (start with $P$ from Part 2; also
   report Euclidean). Deliverables: distribution of per-step $\hat\lambda_k$ in
   free space vs. near/at contact; the domain where $\hat\lambda<1$ holds;
   explicit flagging of contact steps where it fails (this *validates* the
   paper's hybrid restriction rather than embarrassing it).
2. **Input sensitivity $L$.** Same protocol, but perturb the command by $\delta u$
   instead of the state; report $\hat L$ = gap per unit $\|\delta u\|$ per step.
3. **Refutation condition (write it first):** if no metric among the candidates
   gives $\hat\lambda<1$ over a usable free-space domain, the paper's servo
   example stays constructed-only and the limitation sentence stays; do not
   shop metrics post hoc beyond the preregistered candidate list.

## Part 2 — Construct the metric from the controller, as Appendix (gen-innerloop) prescribes

**Theory targeted:** the metric-construction route; PD/first-order channel claims.

1. **GR1/ALOHA joint servos:** fit per-joint first-order lag $\dot q=-a(q-q_d)$
   from the stored open-loop step probes (`openloop*.json` already exist);
   $\hat\lambda$ per joint = fitted pole, $M_c=I$. Check predicted vs. measured
   step decay on held-out probes. This is nearly free and gives real robots
   their first certified channel.
2. **LIBERO Panda OSC:** identify linearized tracking dynamics $A_{\rm cl}$ on
   the operational-space error from healthy logs; solve the Lyapunov equation
   for $P$; verify the metric inequality numerically along visited states
   (sampled Jacobian check with declared tolerance); hand $P$ to Part 1.
3. **Artifact:** per-robot `metric_certificate.json` (metric, domain, margin,
   check residuals) — the object the paper can cite instead of "unverified".

## Part 3 — Propagate the tube on real episodes and overlay the measured error

**Theory targeted:** the tube recursion $R_{k+1}=\lambda R_k+L(\ldots)+\eta$;
the servo figure's real-robot counterpart.

Using Part 1–2 constants, logged $\chi_k,\tau_k$, measured $\epsilon_k$ (truth
injection in sim), and $\zeta=0$ (external subtraction): propagate $R_k$ for
$N\ge20$ corrected LIBERO episodes and $10$ GR1 episodes; overlay measured
$X_k$ from nominal replay. Report the fraction of steps with $X_k\le R_k$, the
median slack, and every violation with its cause class (contact step, domain
exit, calibration error beyond band). **Acceptance:** violations only at steps
the theory already excludes (contact/hybrid, domain exit). One figure per
robot; this becomes the paper's strongest new panel.

## Part 4 — Estimator recursions and constants, quantitatively

**Theory targeted:** Proposition (selected-coordinate target error), the
deadzone ball, applied-delay Eq., the fixed-point/bias analysis, drift $\nu/\alpha$.

1. **Onset transient:** at fault onset, overlay measured $w_k$ against the exact
   $K$-step history prediction $-S^{-1}\sum_{\ell>j}G_\ell\Delta f$; it must die
   in $K=6$ responses.
2. **Recursion envelope:** per episode, overlay $E_k^D$ against the propagated
   bound with measured $\epsilon_k$, actual $\chi_k$, and snapshot age; include
   hold windows. Check the deadzone ball $\epsilon+\|S^{-1}\|\delta$ on gated cohorts.
3. **Fixed points:** attenuation settle vs. predicted $\mathbb E[s_kz_k]$ computed
   from the *logged residual distribution* (not the single-residual formula the
   ablation caveat rejects), across the $\rho=.05/.15/.50$ rows; innovation
   settle vs. truth. Closes the "predict that magnitude" caveat.
4. **Drift floor:** on the ramp cells, compare measured estimate lag to
   $\nu/\underline\alpha$ with $\underline\alpha$ from logged gains.

## Part 5 — Small-gain constants on corrected trajectories

**Theory targeted:** Proposition (conditional small-gain), Corollary
(recovery region).

Regress $\|Dw_k\|$ on $(E_k^D, X_k)$ over corrected episodes (truth-injected,
replay-based $X_k$) to estimate $(\epsilon_0,k_E,k_X)$; take $(\lambda,b{=}L)$
from Parts 1–2. Report whether $ac>bk_X$ holds with margin, the implied
rectangle, and whether measured trajectories stay inside it. **Refutation:**
$ac\le bk_X$ on the deployed cohort is reported as-is — it would say the
coupling condition, not just the constants, is load-bearing.

## Part 6 — The $r_y$ question, as a theory test

**Theory targeted:** the "identification price" explanation of the oracle gap;
the excitation/identifiability discussion.

Diagnose $r_y$: sensitivity column vs. excitation vs. model error (targeted
single-axis probes at multiple states; per-channel regressor energy from logs).
Then one registered intervention (e.g., $r_y$-informative probe added to
calibration) with the written prediction: **if the $r_y$ estimate improves, the
long-horizon corrected count moves toward the oracle's 22/40; if it improves
and the count does not move, the identification-price explanation is refuted.**

## Part 7 — Composite law on a real channel (stretch; first robot test of Theorem gen-adaptive)

On GR1 joint space, $\xi^{\rm nom}$ is computable online from the first-order
reference channel, so $\sigma_{\rm exec}=\Psi^\top e$ is available. Run
composite ($\kappa=1$) vs. prediction-only from matched estimates on one fault
cell, with the theorem's envelope propagated from measured constants.
**This is the only part that evaluates the composite extension on a robot; if
skipped, the paper keeps its "unevaluated" label — do not soften it without
this run.**

## Part 8 — Statistics that the theory sections promised

1. Task-clustered bootstrap for the headline (paper currently episode-iid only).
2. Seed-pinned repeats of the two decision cells (static-observer gap at
   $n=80$ with the registered rule; D.4 paired integral baseline, still unrun).
3. Recovery time to a prespecified sustained threshold, compared against the
   geometric-decay time implied by $\hat\lambda$ (Part 1) — the last unmeasured
   sentence in the Evidence boundary.

---

## Acceptance map (paper sentence → part)

| re4 sentence (current) | closed or instantiated by |
|---|---|
| "the evaluated robots' metrics remain unverified" | 1, 2 |
| "A physical execution metric and its domain remain to be established" | 2 |
| "the VLA experiments do not certify the physical contraction and transition bounds" | 1, 3 |
| "The constants $k_E,k_X,\lambda$ are not certified by the experiments" | 5 |
| "fitted gains and sampled Jacobians remain empirical diagnostics" | 2, 3 |
| ablation caveat "must also be accounted to predict that magnitude" | 4.3 |
| "sustained execution-recovery time remains unmeasured" | 8.3 |
| "the composite extension is unevaluated in the reported VLA trials" | 7 |
| oracle gap "the price of identification" (currently an interpretation) | 6 |

Order of value per GPU-hour: T.0 → 2.1 → 4 → 1 → 3 → 5 → 6 → 8 → 7.
Parts 2.1 and 4.1–4.2 are CPU-only against stored logs and can start immediately.
