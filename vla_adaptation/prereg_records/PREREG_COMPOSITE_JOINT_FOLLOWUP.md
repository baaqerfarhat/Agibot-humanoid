# Composite correction and joint-fault follow-up: staged evaluation plan

2026-09-08. This is a prospective plan for **new qualification, development, and
confirmation runs**, written while the implementation is being completed. It is
not a claim that the earlier joint-map outcomes were unseen. The architecture and
constants for each task experiment will be recorded in a run manifest before that
stage is launched. A confirmation manifest additionally records the selected
development configuration and its code/calibration hashes.

The existing study is explicitly development evidence: with a 5 N·m torque bias
on each LIBERO Panda joint, the Cartesian correction scored 103/140 versus a
frozen 103/140, with 20 episodes fixed and 20 broken. Joint 3 gained seven net
successes; the other six lost seven net successes, including joint 5 at 12/20 to
3/20. Those outcomes motivated this follow-up. Their initial states 45–46, as well
as previously evaluated initial states 20, 47, and 48, are not a fresh confirmation
set. Reproducing them is useful debugging, not an independent test of a new idea.

## Questions and controller under investigation

The practical question is whether a controller combining an estimated disturbance
correction with feedback from joint-position error can improve task recovery
without the regressions observed with the original Cartesian correction.

The planned reference model is fitted on healthy transitions `(q_t, u_t, q_t+1)`:

`q_ref[t+1] = A q_ref[t] + B u_raw[t] + c`.

The feedback component uses the joint error and the identified action-to-joint
response through a term of the form `dt * k * P * B_mask.T * L * error`; the
disturbance-estimation component and the precise sign/error convention are
specified by the implementation and saved configuration. This plan does not
predeclare a particular gain as effective, assume that the two components improve
performance, or describe the resulting empirical controller as proven stable.

ALOHA's absolute joint targets are the first qualification setting. LIBERO's
relative Cartesian action targets need a separate qualification: a good fit of
joint increments does not imply a contractive absolute joint-position reference
model. A failed qualification is a finding about that reference/model interface,
not evidence that all joint-state feedback is impossible.

## Stage 0: software checks and model qualification

No success-rate claim is made from this stage.

1. Exercise fault-off, feedback-off, estimation-off, and zero-error behavior.
   Check correction sign, action units, controller step duration, masking,
   projection, controller clipping, reset state, and model-history initialization.
   Synthetic checks test known equilibria and the complete correction path rather
   than mirroring implementation statements.
2. Collect healthy joint-state/action transitions, recording both raw policy
   actions and the commands the environment actually executes. Record full
   episode/reset provenance, the simulator/controller versions, code revision,
   calibration sample identifiers, model coefficients, and fitting settings.
3. Fit the reference model on calibration data and assess prediction error on
   separate healthy trajectories. Record per-joint error, bias, fitted response
   scales, and fit sensitivity to episode selection. Inspect the reference model's
   contraction/stability diagnostic in the coordinates actually used by feedback.
   A single stable fitted matrix or successful finite rollout is not a global
   stability certificate for a contact-rich robot.
4. Probe feedback response with bounded commands and check that the implemented
   sign reduces the relevant error locally. Record correction clipping and
   actuator/controller saturation. Gains outside the implementation's stability
   guard are rejected before policy repair trials; the guard and its inputs are
   included in the manifest. Guard passage is qualification, not a proof of task
   repair or physical safety.

For LIBERO, collect joint telemetry and assess whether the proposed absolute
reference model is usable before transferring the ALOHA feedback design. If its
model or feedback check fails, retain the probe results, stop that configuration,
and record any revised model as a new development revision. Do not treat a good
Cartesian FIR fit as satisfying this joint-reference qualification.

## Data separation and provenance

An audit of committed result `per_ep` records before this plan found LIBERO
evaluation initial-state indices 20 and 45–48. It found no recorded evaluation
outcomes for the following proposed bands. This is a statement about committed
records, not a claim that no unrecorded trial has ever used those states.

- LIBERO sensitivity calibration: task 0, initial state **25**.
- LIBERO healthy model calibration: initial state **26**, with tasks/episodes
  specified in the calibration manifest.
- LIBERO development: tasks 0–9, initial state **30**.
- LIBERO reserved confirmation: tasks 0–9, initial states **35–38** (up to 40
  scenario pairs per cell).

These choices are conditional on the environment providing those initial states.
Their availability and calibration provenance are checked before collection. If a
band is unavailable or has previously been used, a replacement is documented
before observing its outcomes. The historical `--probe-init 45` default is not used
for new calibration in this study. A legacy calibration may be used for a clearly
marked smoke test or developmental screen; it cannot qualify the reserved
confirmation experiment.

For ALOHA, new disjoint calibration, development, and confirmation base-seed bands
are recorded in the run manifest before each collection stage. Scenario identity
is `(task=0, actual reset seed)`, where `actual reset seed = args.seed + per_ep.init`;
the local episode index alone is not a scenario identifier. The seed must reproduce
the complete task-relevant initial state, including the cube, across treatment
arms. Confirmation seeds must not occur in calibration, tuning, or historical
task-success comparisons used to select the candidate.

## Stage 1: bounded development

The run manifest fixes the exact candidate versions, parameter settings, task/fault
conditions, episode keys, and expected run files before launch. It also names the
calibration used by every comparator and any model-interface differences. An
initial development screen can use fewer conditions than the eventual joint map;
there is no requirement to spend 40 episodes on every candidate or joint before
the controller is qualified.

The relevant comparators are the frozen faulted policy and the existing calibrated
correction under the same fault. When the composite candidate is qualified, compare
its disturbance-only, feedback-only, and combined forms on the development
conditions named in the manifest. Shared settings such as calibration, masking,
authority bounds, reset procedure, and rate are held equal where the comparison
allows; unavoidable differences are stated. A healthy/no-fault condition is
included to measure whether enabling correction changes already functioning
behavior. Oracle correction, if used for a truly additive target fault, is labeled
as using privileged fault information and is not an equally informed estimator.

For physical torque faults, the first development conditions should include the
previous elbow case and the previously harmful joint 5 case. This is an informed
stress test, not a blind prediction. A constant torque is not assumed to correspond
to a constant Cartesian action offset, and equal torque magnitude is not assumed
to give equal task difficulty. Exact joint indices, magnitudes, correction methods,
and any additional fault types are fixed in the manifest before their screen.

Hyperparameters may be selected from these development results. The selection
record states the grid, observed scores, diagnostics, tie rule, and reasons for
discarding any configuration. A rule decided after seeing the development results
is labeled development-informed. The run is not called fully preregistered merely
because this high-level plan existed first. New tuning means a new development
revision; confirmation outcomes are not used to tune a candidate and then reported
as its independent confirmation.

## Stage 2: confirmation, if a candidate qualifies

Before any reserved outcome is inspected, freeze one candidate, its constants,
calibration, fault matrix, exact number of scenarios, comparison list, and
multiplicity families in a confirmation manifest. The manifest identifies the
development selection record and lists forbidden calibration/development scenario
keys. A 40-pair cell is the proposed upper initial allocation, not an automatic
requirement or a guarantee of power. The actual sample count is fixed before the
cell runs. Any later extension is labeled as an extension; there is no repeated
testing until a p-value crosses a threshold.

Confirmation compares the selected candidate against its own frozen control and
against the calibrated legacy comparator on the same task scenarios. The joint
matrix should measure both positive recovery and harm; it cannot be reduced after
seeing which cells favor the candidate. If resources only support a subset of
joints or faults, report that subset as the scope of confirmation. Do not claim
general physical-fault repair from an elbow/joint-5 screen alone.

## Measurements and statistical analysis

Task success is primary. Per condition and comparator, report:

- successes/episodes and absolute percentage-point difference;
- gains and losses on matched scenarios;
- losses/all scenarios and losses/baseline-success opportunities;
- the exact **one-sided 95%** upper bound on conditional loss rate, with the
  denominator visible (no opportunities means the conditional rate is undefined);
- exact two-sided McNemar p-values for prespecified matched comparisons, with
  Bonferroni adjustment across every comparison in the registered family.

The scorer `openpi/score_joint_followup.py` consumes an explicit manifest and the
existing runner result format. It rejects duplicate, missing, or mismatched
scenario keys, inconsistent totals, differing recorded fault conditions, and
unexpected settings. It scores only complete declared studies, emits source-file
hashes, and checks McNemar through both existing implementations. A manifest's
matching labels do not themselves verify physical scene identity; the reset/state
check remains part of the experiment's provenance.

Development p-values are descriptive and labeled exploratory. A nonsignificant
contrast is **not resolved**, not equivalent; lack of a significant healthy effect
does not establish noninferiority. A success-floor condition has few opportunities
to measure harm. The current sample budget may be unable to resolve moderate
differences, particularly against a comparator near perfect success. No arbitrary
fixed/broken pairing is used if an embodiment fails the scenario-matching check:
that study requires an explicitly unpaired design/analysis, not a seed-based
shortcut or silent fallback to a different test.

Secondary diagnostics include held-out joint prediction error, joint-reference
error over the trajectory, disturbance-estimate behavior, feedback and feedforward
correction magnitudes, clipping/guard events, and task completion time when
recorded. Report the relevant coordinates, physical units, time window, and
aggregation. Do not compare a Cartesian disturbance estimate numerically against
a true torque in N·m or infer a mechanism from endpoint means alone. Shared held
estimates are counted as shared identification realizations, not one independently
learned estimate per recipient episode.

## Interpretation and reporting commitments

A positive outcome supports the specific controller/robot/fault conditions tested.
It does not by itself establish that a new update law is novel, that the method is
safe to enable on unknown physical faults, or that the fitted reference model is
globally contractive. The earlier matched-observer results already show that
several calibrated estimators can recover the same action-interface fault.

A failed qualification, no resolved task advantage, or increased regressions is
reported with the same artifacts as a positive result. No old outcome is replaced
in place. New calibration, telemetry, run manifests, result files, and interpretation
are stored under a new follow-up directory. This document is amended with stage
locks and outcomes in order, keeping prior designs visible. At initial writing,
there are **no confirmation results for this follow-up**.

## Development screen lock 1: existing correction under joint 5 torque

Recorded 2026-09-08 before these new rollouts. This screen measures whether
correction coordinates, observer choice, attenuation, or holding the estimate
reduce the previously observed joint-5 harm. It precedes a qualified composite
controller; none of its arms is labeled composite feedback.

The first cell is `libero_spatial`, physical zero-based Panda joint **5**, constant
**+5 N·m**, with task IDs 0–9 at initial-state index **30**: ten scenarios per
arm, seven arms, **70 scored rollouts**. The Cartesian injected bias is zero. The
episode cap is 220 policy action steps after the runner's ten warmup
steps, and the policy replans every five steps. Root coordinates the policy server
and launches the run; the wrapper records the actual host/port and control ack.

The ordered input arm list is:

| Arm | Executed correction coordinates | Observer/application |
| --- | --- | --- |
| `off` | none | Frozen faulted policy; shared control run once |
| `legacy_translation` | Cartesian translation 0–2 | Original legacy update |
| `legacy_full` | Cartesian action coordinates 0–5 | Original legacy update |
| `legacy_rotation` | Cartesian rotation 3–5 | Original legacy update |
| `dob_translation` | Cartesian translation 0–2 | Calibrated DOB update |
| `legacy_half` | Cartesian translation 0–2 | Legacy estimate, half correction authority |
| `legacy_hold30` | Cartesian translation 0–2 | Legacy estimate updated for the first 30 policy actions, then held |

All arms reuse the recorded healthy FIR and sensitivity matrix supplied through
`--log` and `--openloop`. The initial development defaults are the existing
`results/phase05/error_signal_so3.json` and
`results/phase05/openloop_so3.json`. This is disclosed historical calibration
reuse, not a newly separated calibration or a qualification of the later
confirmation study. Common adaptation constants are `gamma=0.08`,
`dead=0.008`, normalization radius `0.15`, estimate clipping `0.30`, all-channel
normalization, and zero deadzone mode. The bias correction is `None`. Coordinate
masks and authority changes are deliberately treatment differences. The old joint
map and this new cell must not be silently pooled if their recorded settings
differ.

For scenario counter `i`, the wrapper cyclically rotates the ordered input list
by `i modulo 7` and reverses the rotated list on odd `i`. It completes that
scenario across all arms before advancing, independent of outcomes. The policy
sampling RNG remains unpinned, as in the original runs; scenario pairing does not
imply identical sampled policy actions. The first pre-action callback records full
simulator `qpos` and `qvel` before the first torque injection. Every later arm in
the scenario is compared with the first arm's snapshot, including state hashes
and maximum absolute differences; a discrepancy above `1e-9` aborts and marks the
study invalid. The task-0/init-8 control-handshake probe is recorded as setup and
does not count among the 70 outcomes.

`study.json` records the complete arm schedule, actual args, arm configurations,
source/calibration hashes, state comparisons, progress, and final status. One
telemetry JSONL contains every arm with explicit scenario and arm labels. Each
candidate JSON exposes standard `frozen_faulted` and `adaptive` arms for scoring;
their frozen outcomes are views of **the same** shared control cohort, identified
by one `shared_control_id`. They must not be pooled as six independent controls.
Progress snapshots are replaced atomically after each episode. Partial or failed
studies are retained and cannot be reported as the complete cell.

All six candidate-versus-shared-off contrasts form one exploratory family of six;
all five alternative-versus-`legacy_translation` contrasts form a second
exploratory family of five. Report the counts and paired discordances for every
arm, including failures and regressions. Development-informed choices and their
diagnostics are written in a selection record; no positive outcome is assumed.

The user-authorized expansion covers every physical Panda joint 0–6 and multiple
LIBERO benchmarks (spatial, object, and goal), with ALOHA robot evaluation after
its own qualification and complete-state pairing check. The runner therefore
accepts all seven joints, selectable suites, and configurable scenario counts.
An initial broader cell allocation of **20 scenarios per arm** is planned after
selection; this paragraph does not pre-lock its candidate, fault/healthy matrix,
calibration, seed bands, or multiplicity family. Those are fixed in a separate
manifest before broader outcomes are observed. Benchmark-specific episode caps
are preserved, and scenario identity includes the benchmark as well as task and
initial-state index. This initial 70-rollout screen does not fulfill the broader
multi-benchmark or multi-robot request by itself.

## Development screen lock 2: qualified ALOHA position-reference feedback

Recorded 2026-09-08 before the following development outcomes. This allocation
uses `gym_aloha/AlohaTransferCube-v0`, actual reset seeds **2700–2709**, nine
arms, and two conditions, for **180 scored rollouts**. Each arm runs once per
condition and seed. The two conditions are healthy commands and an additive
**+0.02 rad target offset on left arm joints 0–5**, with the remaining eight
command coordinates undisturbed. This screen qualifies neither multiplicative
actuator gain faults nor physical torque cancellation.

The ordered arms are:

| Arm | Observer and tracking feedback |
| --- | --- |
| `off` | No correction; one shared control per condition/seed |
| `legacy_all` | Existing legacy law, all 14 residual channels in its norm |
| `legacy_corrected` | Existing legacy law, left six residual channels in its norm |
| `dob` | Calibrated DOB, coefficient 0.08 |
| `kalman` | Matched random-walk Kalman estimator; no position feedback |
| `composite_001` | Kalman plus position feedback, calibrated strength 0.01 |
| `composite_005` | Kalman plus position feedback, calibrated strength 0.05 |
| `composite_010` | Kalman plus position feedback, calibrated strength 0.10 |
| `composite_025` | Kalman plus position feedback, calibrated strength 0.25 |

Every adaptive arm corrects only left arm joints 0–5. Common settings are
`gamma=0.08`, `dead=0.002`, normalization radius **0.4**, estimate clip **0.08**,
zero deadzone mode, no residual bias subtraction, no estimator warm start, and no
within-episode freezing. The legacy normalization and deadzone do not apply to
the DOB/Kalman/composite baselines. Parameter damping is zero. The policy runs at
50 Hz, replans every ten commands, and retains the 300-step episode cap. Each
episode resets the estimate, estimator covariance, reference position, and FIR
history. The reference is driven by that arm's actual raw policy commands. It
does not replay commands from a privileged healthy rollout. This screen's
Kalman arm supplies the zero-tracking ablation; feedback-only and damping
ablations are not included in this particular allocation.

All nine arms use exactly the same W/M/Q/R in
`results/composite_followup/calibration/aloha_reference.json`, SHA-256
`374df77ed662e914bf9a396b2a0f92269c96b8e42910d7f243cd69f2f71bc836`.
The fresh healthy collection command used seeds 2600–2609; its source JSON has
SHA-256 `25ea6ee5d343eb4c230cabea0142d3aaf85bcbadc21783a08e58c8d63d70a4a2`.
The original collection JSON records episode ordering but does not itself embed
the reset seeds, so seed provenance also relies on the recorded collection
command. Episodes 0–5 fit the reference/FIR/noise; episodes 6–9 validate the
reference. M is explicitly reused from historical
`results/aloha/openloop.json`, SHA-256
`b09669da5575f42db91fc09a167179eff3a1543b7abf09dde7a3013c607f0e7a`.
Thus this is fresh reference/noise fitting with historical sensitivity reuse,
not wholly fresh calibration of every component. Seeds 2500–2501 were physical
contraction probes; seed 2690 was a software smoke test and is excluded from
development/confirmation outcome counts and gain selection.

The predeclared validation gate was **0.005 rad RMSE per tracked state**, for
both one-step prediction and autonomous healthy reference rollout. The fitted
model passed: spectral radius 0.88214, Euclidean contraction factor 0.92428,
maximum held-out per-state one-step RMSE 0.001169 rad, and maximum held-out
per-state rollout RMSE 0.003737 rad. The maximum instantaneous six-joint reference
error norm was 0.02510 rad; passing the RMS gate is not a pointwise task-margin
bound. Hidden velocity, contacts, policy feedback, and correction clipping remain
outside the fitted first-order contraction claim.

The strength grid was fixed from calibration before development task outcomes.
Strength is normalized by `||dt P_inf D.T L D||_2`; the four corresponding raw
tracking rates are approximately 686923.6043, 3434618.0216, 6869236.0433, and
17173090.1082. These values depend on the recorded small covariance, input map,
and metric. They are not meaningful transferable scalar gains. The artifact
stores full precision and checks the local unsaturated augmented error matrix
along the covariance transient and at steady state. The implementation verifies
the calibration matrices, mask, rate, time step, and damping before running.
This is a local fitted-model screening diagnostic, not a physical stability
certificate or inherited Neural-Fly theorem. No raw rate outside the declared
qualified grid will be introduced using this screen's task outcomes. Any selected
strength and its healthy/offset tradeoff will be documented as a development
choice before a separate confirmation lock; this allocation does not itself
specify a confirmation winner or establish noninferiority.

`openpi/run_aloha_followup.py` completes the two conditions and all arms for each
seed before advancing. For episode counter `i`, condition order uses the same
cyclic-rotation/odd-reversal rule as the joint runner. For condition position `j`
within that episode, arm order uses that rule with counter `2*i+j`. Scheduling is
fixed without reference to outcomes. The policy sampling RNG remains unpinned;
paired initial physical scenes do not imply identical sampled policy commands.

After each `Aloha.reset`, the wrapper reads full MuJoCo `qpos` and `qvel` through
`A.env.unwrapped._env.physics`. It changes no renderer or reset behavior. Every
arm and both conditions for the same seed are compared with the first snapshot,
including cube coordinates, before any policy action or injected offset. A
maximum absolute difference above `1e-9` aborts the study and marks pairing
invalid. Observed joint positions alone cannot satisfy this check.

The runner locks a `run_config.json` with code, preregistration, and artifact
hashes before environment creation. `study.json` records the schedule, progress,
per-condition arms, state checks, and completion/failure. Telemetry labels every
step as `condition/arm`. Standard result views are named
`healthy_<arm>.json` and `offset_<arm>.json`; their `frozen_faulted` arms identify
the same shared off cohort within that condition. These copies must not be pooled
as independent controls. Outputs are never overwritten by a new invocation,
and progress JSON is replaced atomically after every completed episode.

For this screen, all eight candidate-versus-off contrasts across both conditions
form one exploratory family of **16**. All seven non-off alternatives versus
`legacy_all` across both conditions form a second family of **14**. The four
composite-versus-Kalman contrasts across both conditions form a third family of
**8**, directly testing the addition of tracking feedback. Counts, paired gains
and losses, exact two-sided McNemar values, within-family Bonferroni values, and
conditional regression rates/bounds are reported for all declared comparisons.
These are exploratory development statistics, not independent confirmation or a
claim of safety from nonsignificance. Further observer/robot/torque evaluations
remain separate stages of the broader authorized follow-up.

## Development screen lock 3: all physical ALOHA arm joints

Recorded 2026-09-08 before physical-torque policy rollouts. This development
screen uses TransferCube, seeds 2900–2909, and thirteen cells: one healthy cell
and each of twelve arm joints faulted separately. Physical indices 0–5 are the
left arm and 6–11 the right arm; grippers are excluded. Four interleaved arms
(`off`, `legacy_allchannels`, `legacy_correctedchannels`, `kalman`) give **520
rollouts**. The healthy cell is run once, not duplicated for each faulted joint.

All candidates correct the twelve arm command coordinates 0–5 and 7–12. They use
the same fourteen-channel W/M/Q/R from the existing left-arm reference artifact,
solely as shared observer calibration. Constants are gamma .08, deadzone .002,
normalization radius .4, estimate clip .08, and the existing 300-step ALOHA cap,
20 ms control period, and ten-action policy horizon. Policy sampling is unpinned;
estimates reset every episode. The twelve-joint reference failed the .005 rad
held-out RMS criterion, so **no composite arm** is tested in this screen.

Each torque is the live nominal position-actuator command-to-torque gain times
+0.02 rad: [16,32,16,.2,1,.4] N·m on each side. This defines an unsaturated static
reference scale, not equal physical torque across joints, equal dynamic
severity, or guaranteed 20 mrad displacement under force/position/control limits.
Torque is injected below the servo through qfrc_applied. The estimator receives
zero action fault and no torque truth; the experiment wrapper records the fault
separately. Measured actuator limits, force/control saturation, correction clipping,
and commanded out-of-range targets are retained in telemetry.

Per-cell arm order rotates by episode and reverses on odd episodes. One off
cohort is shared by the three candidates. Full physical qpos/qvel, including the
cube, must agree within 1e-9 after reset and before torque or policy inference.
Each cell refuses existing outputs, saves source/argument/calibration hashes,
and preserves incomplete runs as incomplete. Bounded parallel cell execution
may change policy sample order; it does not change the paired scenario definition.

All candidate-versus-off contrasts form one exploratory family of 39. The two
alternative-versus-legacy_allchannels contrasts per cell form a family of 26.
Report every cell, exact paired discordances, adjusted p-values, and conditional
regression denominators/bounds. Seeds 3000–3019 are reserved for a later separate
confirmation lock; these development results will not be relabeled confirmation.

## LIBERO reset correction and restart

The first joint-5 development run stopped at task4/init30 when the required
pre-action pairing check failed. Its partial counts are not a valid completed
comparison and are preserved under `failed_reset_screen`. Upstream LIBERO resets
randomize fixture model positions/orientations, which saved initial data states
do not restore. Hard resets also accumulate generated object-property initializers,
changing random-number consumption even after reseeding.

`libero-reset-v1` seeds by suite/task/init, clears the stock generated property
initializers before hard reset, and clears external forces. Actual cached-env
repetition on task4/init30 passes exact model/data comparisons immediately after
reset and after ten dummy steps; the legacy path reproduces a mismatch. Follow-up
runs now compare the model/data fingerprint as well as qpos/qvel before policy
actions. Policy sampling remains unpinned. This change does not retroactively
validate historical pairing.

The seven-arm development design is restarted in a new directory at the same
task/init allocation, explicitly with this corrected reset protocol. No result
from the aborted run is pooled with it. In parallel, fresh healthy calibration
will collect all ten tasks at init26 in each of spatial/object/goal and identify
M at task0/init25 using the corrected reset. These states remain separate from
development and later confirmation. Because fresh calibration changes the
observer, its selected candidate will receive a development recheck before a
confirmation configuration is locked.

## Development screen lock 4: mask-aware bounded weighted DOB

Recorded after the corrected-reset seven-arm screen and the local authority
diagnostics, but before any weighted-DOB task rollout. The candidate estimates a
low-pass residual (gamma .08), then minimizes
`||R^(-1/2) (M[:,active] f - residual_estimate)||^2 + ||f/0.1||^2`
subject to the existing coordinate bounds `|f| <= .3`. Inactive estimates are
zero. R is the centered healthy FIR residual covariance with the existing
eigenvalue floor; M, W, and R use fresh spatial calibration at inits25/26.
The prior standard deviation .1 action units is fixed prospectively. This is an
input-allocation/regularization variant of DOB, not a novelty claim.

The box constraints are part of the solve. The static fitted-model objective is
compared with zero correction, but this does not certify real nonlinear task
behavior, changing residuals, or contraction of an adaptive closed loop. In
particular, historical M and measured local faulted response maps differ.

This screen uses spatial tasks0–9 at init32 in three cells: healthy (zero torque),
joint3 +5N·m, and joint5 +5N·m. Six arms are interleaved in the fixed input order
`off,legacy_translation,legacy_full,legacy_rotation,weighted_translation,weighted_full`,
giving **180 rollouts**. Other constants, bounds, episode caps, bias=None, model
fingerprint checks, and unpinned policy sampling match the corrected follow-up.
Every cell has one shared off cohort and is written to a new directory.

All five candidate-versus-off comparisons across three cells form an exploratory
family of15. The two weighted-versus-same-mask-legacy comparisons across three
cells form a family of6. Report all outcomes and regressions. This is a development
recheck with fresh calibration, not independent confirmation. Broader all-joint,
three-benchmark evaluation will use a separate prospective lock and untouched
initial states, and will retain negative results.

## Confirmation lock 1: all Panda joints across three LIBERO suites

Recorded after all 180 weighted development rollouts and both local authority
grids, before collecting any confirmation outcome. Select `weighted_full`: the
healthy/joint3/joint5 development counts are 9/10, 8/10, 8/10, versus off
10/10, 6/10, 8/10 and legacy translation 10/10, 8/10, 1/10. This is an explicit
development selection; the healthy regression and remaining local physical harm
are retained. Weighted translation and alternative legacy masks are not carried
into this confirmation. The exact source-result hashes are in the selection
record supplied to every rollout.

Use spatial, object, and goal, all ten tasks in each, at untouched initial states
35 and 36. Each suite contains one healthy cell and seven separate +5 N m torque
cells, one per Panda arm joint. Each cell interleaves the same three arms: off,
legacy_translation, weighted_full. Total: 24 cells x 20 scenarios x 3 arms =
1,440 rollouts. Fixed +5 N m is not equal fractional actuator severity: installed
limits are +/-80 N m for joints0–4 and +/-12 N m for joints5–6. Claims must be
restricted to the tested severity and these scenario distributions.

The candidate uses all six Cartesian correction inputs, gamma .08, bounds .3,
prior standard deviation .1 action units, no bias correction, and continuous
updates. M, FIR, and healthy residual covariance come from the completed fresh
calibrations at inits25/26 within each suite. No outcome-dependent gain, matrix,
channel, or stopping changes are allowed. Episode caps are 220/280/300 for
spatial/object/goal. All arms use libero-reset-v1 and full pre-policy physical
fingerprint checks; policy randomness is not pinned. Four or six independent
cell workers may share the serialized policy-control handshake.

The prespecified primary contrast pools the within-suite paired discordances
for joint5 weighted_full versus legacy_translation across all three suites
(60 distinct suite/task/init scenarios), with one exact two-sided McNemar test.
This tests the motivating joint5 repair claim. A positive primary result alone
does not establish benefit against correction off, healthy safety, or universal
repair. Joint5 candidate-versus-off pooling is descriptive secondary analysis.

Report every individual healthy/joint cell and its paired fixes/breaks. Three
global families contain 24 comparisons each: legacy versus off, candidate versus
off, and candidate versus legacy. Use Bonferroni within each full family; do not
reduce its size if a cell fails. Report conditional regression denominators and
95% one-sided upper bounds. No significance result or nonsignificance result
constitutes a claim of no harm. All incomplete or failed runs remain documented;
validate all 24 cells before the planned pooled analysis. Frozen source, input,
selection, allocation, and collection manifests are saved before launch.
