# Exact-tip implementation audit (2026-09-09)

This is a read-only comparison of `main` at
`7ca2f26fa919c8b4cb8e4532b64c8072e36368db` and
`review/dual-track-audit` at `cda6c92357e884ef6e2b1195abda673fc18853ac`.
Their merge base is `96555f3fac7431c8f9e8a8c9b0362121ca7d5a83`.
Line references below are at those fixed revisions, not a moving branch.
No simulator experiments, branch merge, or runtime changes were performed for this audit.

## Actual defects still present on this main tip

### 1. Exact McNemar implementation and silent duplicate pairing

- Main `openpi/mcnemar.py:29–32` compares binomial probabilities with an absolute
  `+1e-12` tolerance. For discordances `(1,51)`, it returns
  `6.123990203832363e-13`; the independent integer-tail calculation is
  `2.353672812205332e-14`. This was conservative, not inflated significance.
- Review `openpi/mcnemar.py:29–41` uses a relative comparison and reproduces the
  independent value. Review `:75–90` rejects duplicate `(task, init)` keys rather
  than silently dropping repeated episodes while retaining the old totals.
- Added `openpi/mcnemar_crosscheck.py` independently calculates the exact tail.
  `openpi/score_joint_followup.py:46–92` requires a declared complete scenario set,
  correct totals, no duplicates, and agreement between both instruments. It does
  not accept a convenient intersection of available outcomes.
- Earliest implementation commits: `722576d` (tolerance), `dda9040` (duplicate
  guard); import the current small files plus scorer dependencies if desired.
  Do not blindly replay their historical manuscript edits over main's newer results.
- Recalculate statistics from stored outcomes; simulator reruns are unnecessary
  for this arithmetic correction. Missing/invalid pairing cannot be repaired by
  a different p-value implementation.

### 2. Joint-fault lifecycle can contaminate later episodes

- Main `openpi/adaptive_law.py:104–109` constructs/applies a model fault but has no
  corresponding `finally` restoration. A later object can snapshot an already
  modified cached model as its baseline; friction, damping, actuator gain and
  locks can persist/compound, and exceptions can leave the fault installed.
- Review `openpi/adaptive_law.py:411–414,573–575` encloses the entire episode in
  `try/finally` and restores on normal completion, early success, exceptions and
  interrupts. `openpi/joint_fault.py:53–63,83–87` saves/restores preexisting external
  forces and adds the injected torque to the prior force rather than zeroing
  unrelated external loads.
- Main already has a `restore()` helper; the defect is its lifecycle use and
  baseline handling, not the complete absence of a restoration function.
- Commit `c32f318`, with `openpi/test_joint_fault_lifecycle.py` and
  `openpi/test_aloha_joint_fault.py`, is the relevant change family.
- This explains a possible integrity failure, not automatically the historical
  joint-5 performance drop. Corrected-protocol fresh experiments are necessary
  before making causal claims about affected cached-environment runs.

### 3. LIBERO scenario IDs do not guarantee identical physical scenes

- Main's runner calls `env.reset(); env.set_init_state(...)`. The stored initial
  state does not restore model fields such as reset-randomized fixture geometry.
- Review adds `openpi/libero_reset.py:18–95`: deterministic scenario-derived seed,
  clearing regenerated stock property samplers before a hard reset, force
  cleanup, and model/data fingerprints. The sampler fix matters because reseeding
  alone did not prevent a changing number of RNG draws.
- `openpi/run_joint_followup.py:91–109,158,209–218` requests the new reset and
  compares model/data fields plus full qpos/qvel across arms. The scorer requires
  verified pairing. Policy RNG remains independent and is explicitly recorded.
- **Partial deployment caveat:** `openpi/adaptive_law.py:393` still defaults
  `scenario_reset=False`, and the historical CLI call at `:1152–1163` does not set
  it. This preserves old behavior. Importing `libero_reset.py` alone, or using the
  legacy CLI unchanged, does not fix every future run. Use/port the follow-up
  runner and explicitly require the corrected reset/fingerprint protocol.
- Commit `c32f318`; dependencies include `paired_probe.py`, calibration callers,
  `check_libero_reset.py`, and the tests/audit artifacts.
- Passing new reset tests does not retroactively validate old pairing; compare
  old fingerprints if recorded or rerun the claims requiring physical pairing.

### 4. Calibration lacks provenance and can overlap evaluation

- Main `openpi/openloop_id.py:63–75` uses a flat prefix of commands, task 0 and
  hard-coded initial state 45. It has no source-episode binding or replay-source
  provenance. The historical replay also substitutes an open gripper command.
- Review `openpi/openloop_id.py:57–97` selects exactly one nominal episode, checks
  boundaries/full command consistency, retains recorded gripper commands when
  available, and labels the legacy missing-gripper fallback. `:100–129` aligns
  plus/minus sensitivity probes to a common prefix; `:132–205` exposes task/init
  choice, validates arguments, and writes source hashes and replay provenance.
- Review `openpi/error_signal.py:132–176` writes schema-v2 calibration artifacts
  with status, episode keys, source hashes, control acknowledgment and failures;
  `:189–198` supports healthy-only data and refuses accidental overwrite.
  `adaptive_law.py:320–369` accepts the old and new artifact formats.
- **Not a universal held-out fix:** review retains `--init-base=45`,
  `--probe-init=45`, and `--eval-init=45`. The historical CLI's overlap check
  (`adaptive_law.py:1097–1115`) is a warning, keyed primarily to FIR provenance;
  it is not a hard, complete union-of-FIR/M/reference/tuning overlap validator.
  A new experiment must explicitly separate every fitted/probed source from its
  evaluation scenarios. The fresh tuning/follow-up pipelines provide stronger
  source binding and declared splits.
- Commits `a10d5f3` (expose provenance problem), `c32f318` (completed pipeline and
  episode-safe collection/replay). Do not pick only the CLI flag hunk: reader,
  writer, replay and schema changes belong together.
- Refit and rerun before claiming held-out historical performance; metadata added
  now cannot recover unrecorded provenance or quantify existing optimism.

## Algorithm extensions and their actual limits

### Legacy normalizer, gripper coupling and innovation

Both old and review default ALOHA implement, ignoring clipping and with the gate open,

\[
 \hat f^+=(1-\gamma)\hat f+\gamma\alpha(M^{-1}r-b),\qquad
 \alpha=[1+\|r\|^2/\rho^2]^{-1}.
\]

The normalizer scales the target, so its fixed point for an ideal constant residual
is approximately `alpha*f`, not merely slower convergence. Main ALOHA computes the
norm over all 14 channels (`openpi/aloha_adapt.py:119–124`) although correction can
be limited to six arm joints. Review adds:

- `--norm-channels corrected`: restrict only the gate/normalizer signal to the
  correction channels; the full calibrated residual remains available to the
  inverse. This is a diagnostic/variant, not a proved cure for task failure.
- `--law innov`: use `e = r - M*f_hat - M*bias` and
  `f_hat += gamma*M_inv*e/(1+||e_selected||²/rho²)`; the unnormalized ideal fixed
  point is the fault when invertibility/matching assumptions hold. A nonzero
  deadzone leaves a neighborhood, not exact identification.
- `--deadzone-mode hold`: skip the gated update entirely. Legacy `zero` sets the
  target to zero and therefore decays the existing estimate; it does not freeze
  it. The mode must be included in fault-removal comparisons.
- Per-step raw action, correction, known command, residual, attenuation, estimate
  before/after, gains and source/configuration telemetry.

Review locations: `aloha_adapt.py:27–82,107–125,188–194,247–282,594–613`;
LIBERO counterpart `adaptive_law.py:148–183,386–393,551–563,939–941`.
Commits: `f801087` (telemetry/channel controls), `5936510` (ALOHA innovation),
`f20ee8b` (bias/removal investigation), `9ef518f` (residual-channel instrument).

**Do not claim innovation is new everywhere:** main already implements it for
LIBERO (`adaptive_law.py:218–241`) and GR1 (`gr1_adapt.py:140–146`). Main GR1
already restricts its normalizer to selected channels (`:139`). ALOHA is the
important missing port. Review still defaults to legacy/all/zero in both old
runners, deliberately preserving previous experiments.

**Do not call the gripper problem solved:** the old and review ALOHA fit is still
an FIR on commanded targets versus measured positions; none of these flags adds
servo saturation/contact dynamics. Both command and measured gripper position
can be continuous. Binary switching is not an established mechanism, and a
linear FIR can represent a step. The independent-model audit provides a concrete
conversion/limit hypothesis, but an identified/trained gripper-and-contact model
is not deployed by these ALOHA runners. A separate constant-basis LIBERO adapter
exists and must not be confused with that missing gripper model.

### Matched observer alternatives

`openpi/adaptive_law.py:105–146` adds:

- DOB: exponential average of calibrated `M_inv*r-bias`.
- RLS: recursive constant-parameter fit with forgetting/covariance.
- Kalman: random-walk parameter state with healthy-residual measurement covariance.
- Calibrated integral: accumulate the calibrated residual. Since this residual
  already accounts for the known corrected command, a persistent nonzero ideal
  fault can drive this comparator to its bound. This is not evidence against all
  integral feedback designs; integrating the innovation yields a different law.
- Oracle remains a known-fault experimental control, not a deployable estimator.

They share the FIR/calibration/correction interface; all matched estimators retain
clipping/projection (`:146`). DOB removes normalization and deadzone, not projection.
Their sample results do not establish equivalence or a universal method ranking.
The Kalman default can be set to the same asymptotic effective gain as DOB, so a
startup covariance advantage must not be called a universal tracking advantage.
Commit family `fdb74d8`; the later ALOHA/tuning port includes `acd3b7f`.

### Bounded weighted allocation for available correction directions

Main obtains an unconstrained full inverse estimate and then masks/clips its
correction (`adaptive_law.py:144–156,250 onward`). Inverting all input directions
and subsequently removing some of them need not minimize the remaining residual.
Review `openpi/weighted_dob.py:129–203` instead filters the residual and solves

\[
 \bar r^+=(1-\gamma)\bar r+\gamma(r-Mb),\qquad
 \hat f_J=\arg\min_{|v|\le c}\|R^{-1/2}(M_Jv-\bar r^+)\|^2
                       +\|v/\sigma\|^2,
 \qquad u=a-\hat f_J.
\]

The correction mask is enforced inside the solve, correlated healthy residual
noise determines the weighting, and box limits are part of optimization. The
small active-set solver checks feasibility, descent and KKT residuals; it raises
on nonconvergence. It supports at most 12 active inputs. Persistent state contains
the filtered residual; carrying only `f_hat` does not resume the filter.

This corrects a nominal allocation mismatch. It cannot create missing actuation,
make a stale local map physically valid, or guarantee task recovery. An improved
calibrated objective is not automatically an improved physical displacement or
success rate. Preserve the documented improvements **and** harmful cells.
Commits `207feaf` plus `1df4db6` (ALOHA CLI integration); dependencies include
healthy covariance estimation and `openpi/test_weighted_dob.py`.

### Composite adaptation and qualified reference model

Review adds `openpi/composite_observer.py` and ALOHA integration. A healthy fit
`q_ref+ = A*q_ref + B*a_raw + offset` generates a position reference from the
live raw policy command. With actual-minus-reference error and `D=B*mask`,

\[
 \hat\theta^- = e^{-\lambda\Delta t}\hat\theta,\qquad
 \hat\theta^+=\Pi[\hat\theta^-+K(r-H\hat\theta^-)
                  +\Delta t\eta P^+D^TLe^+].
\]

`H` is the residual observation map; `D` is the action-to-next-position tracking
map. They must not be silently equated. The physical action uses the previous
estimate, then the observer updates from the resulting observation; signs and
sample timing are tested. The positive tracking term is consistent with
`u=a-theta` and `e=q-q_ref`.

- `composite_observer.py:192–280` fits on whole healthy episodes and qualifies
  held-out prediction, independent reference rollout, rank/conditioning and a
  contraction metric.
- `:284–360` implements damped KF prediction, Joseph-form covariance and tracking;
  process noise is per second, measurement noise per sample. Zero tracking **and
  zero damping**, with matching noise/time units, recovers the KF ablation.
- `:363–394` checks the augmented frozen-gain unsaturated recursion. A contractive
  healthy fitted model alone is insufficient: the added estimator feedback can
  destabilize it. Neither local check certifies the contact-rich full VLA loop.
- `aloha_adapt.py:633–681` requires the appropriate artifact and, for composite,
  checks qualified mask/rate/damping and revalidates source-bound calibration.
- Commit families `d89d7bb`, `7961ce0`, `acd3b7f`, `16b2574`; import qualification,
  validation, runner and tests together. Trained-basis adaptation in the new
  `learned_adaptation/` package is a separate prospective method, not this tested
  legacy composite observer.

### Independent-method implementation details should not be conflated

The new `learned_adaptation/libero_adapter.py` uses a supplied constant matched
basis `Phi=E`, not a trained MAGIC/Neural-Fly basis (`:13–20,159`). Its reference
is Euler at one policy interval, `B=I, A=F-I, E=G`; this exactly implements the
identified discrete ARX recurrence (`:8–11,143–159`). This is not continuous
zero-order-hold reference propagation. Independently, the invoked
`loop.AdaptationLoop.observe` **does** use Van Loan forgetting discretization
and Joseph-form covariance (`learned_adaptation/loop.py:197–211`). Do not call
these estimator equations absent merely because the reference uses Euler.
Neither fact proves the full sampled-data closed loop inherits a continuous-time
Lyapunov theorem. The actual adapter clips coefficients before allocation,
which its own docstring correctly distinguishes from a proved metric projection.
For matrix-valued forgetting, the continuous covariance drift is
`-Lambda P - P Lambda.T`; writing `-2 Lambda P` requires an appropriate
scalar/commuting symmetric special case.

## Shared behavior and main-only work that a merge must preserve

- The correction sign is not generally broken: the legacy adaptive command is
  nominal minus estimate on both branches. `--static-corr` means an **additive
  correction** in LIBERO but a **fault vector to subtract** in ALOHA/GR1. Review
  documents/tests the distinction rather than silently changing old CLI semantics.
- ALOHA FIR history initialization at current position was already on main
  (`aloha_adapt.py:81–86`); do not credit review for inventing it.
- GR1 environment-generator reseeding (`gr1_adapt.py:55–64`) is shared from before
  divergence. The legacy unpaired data remain unpaired, but main now has separate
  paired GR1 studies. Do not overwrite their provenance with an old blanket claim.
- Main uniquely adds `--hold-stat median/window` and accumulation across
  identification episodes (`gr1_adapt.py:187,257–266`). Review only has last/mean50;
  copying its whole file would regress this functionality.
- Main uniquely adds `openpi/groot_widowx_server.py`, `openpi/widowx_adapt.py`, and
  `openpi/widowx_video.py` with WidowX/SAPIEN results at `7ca2f26`.
- Main's latest paired GR1 maps, videos, time-varying-fault study and paper changes
  are independent additions, not duplicates of review's Panda/ALOHA diagnostics.
- Preserve the independent-project instructions, authorship, and both branches'
  result provenance. This comparison does not authorize an actual merge.

## Practical import order for an agent working on main

1. Create a fresh review branch from the exact main tip; fetch the audit ref and
   record both SHAs. Preserve main-only GR1 and WidowX files/results.
2. Import statistics/cross-checks first and re-score stored data. Keep older and
   newer protocols separate; correct only claims actually supported by each.
3. Port lifecycle cleanup and the scenario-reset/fingerprint protocol with tests.
   Ensure every intended new runner explicitly opts into the new reset. Treat
   old physical-pairing claims as requiring fingerprints or new measurements.
4. Port the calibration producer/consumer/schema/provenance changes together;
   implement a strict experiment-level disjointness check covering FIR, M,
   reference, search, selection and confirmation. Preserve legacy readers.
5. Add telemetry and opt-in estimator variants while preserving default numerical
   behavior. Explicitly select/record new semantics in a new protocol.
6. Import weighted allocation and composite/reference modules with their tests,
   qualification and source validation; tune them on development data with the
   same budget and keep confirmation data untouched. Do not force a ranking.
7. Merge scientific evidence/figure captions selectively, preserving main's new
   paired GR1 and WidowX material. Keep prospective learned-basis work explicitly
   separate from legacy FIR experimental evidence.

These are **change families**, not a safe sequential `git cherry-pick` list.
Several commits include large datasets and older paper rewrites or depend on
prior refactors. Inspect each diff and port the current dependency group; do not
replace main's entire tree with review. No runtime port was executed here.

## Checks actually run for this audit

At review `cda6c92`, all completed successfully:

```sh
python -m unittest discover -s openpi -p 'test_joint_fault_lifecycle.py' -v # 6
python -m unittest discover -s openpi -p 'test_calibration_pipeline.py' -v # 5
python -m unittest discover -s openpi -p 'test_weighted_dob.py' -v         # 9
python -m unittest discover -s openpi -p 'test_composite_observer.py' -v   # 20
python openpi/adaptive_law.py --selftest
python openpi/aloha_adapt.py --selftest
```

The self-tests include 400 exact legacy/innovation/integral arithmetic comparisons,
matched stationary covariance/RLS/KF checks, masks, clipping, fixed points and
fault removal. The McNemar calculation above separately executes each exact tip's
source against an integer-tail reference. These are mathematical/software checks,
not new simulated or physical task-success evidence.
