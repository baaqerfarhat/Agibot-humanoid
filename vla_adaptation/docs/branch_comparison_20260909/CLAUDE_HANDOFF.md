# Claude handoff: improve main using the dual-track audit

This is an actionable integration specification, not a record of a completed
merge. First read [HUMAN_SUMMARY.md](HUMAN_SUMMARY.md). Detailed evidence is in
[code_audit.md](code_audit.md), [results_audit.md](results_audit.md), and
[main_preservation_and_merge.md](main_preservation_and_merge.md).

## 0. Repository and evidence boundaries

```yaml
audit_date: 2026-09-09
repository: git@github.com:mtaheriee/vla-adaptation.git
destination_branch: main
main_sha: 7ca2f26fa919c8b4cb8e4532b64c8072e36368db
source_branch: review/dual-track-audit
review_sha: cda6c92357e884ef6e2b1195abda673fc18853ac
merge_base: 96555f3fac7431c8f9e8a8c9b0362121ca7d5a83
main_only_commits: 8
review_only_commits: 120
scope: comparison_and_integration_plan
merge_performed: false
new_robot_experiments_for_this_audit: false
```

[snapshot.json](snapshot.json) lists all 1,867 endpoint file differences and the
six paths changed on both sides since the merge base. Most differences are
experimental artifacts. An endpoint `D` can mean a file added only on main;
it is **not** an instruction to delete that file when integrating review.

Review now contains three distinct bodies of work:

1. Integrity corrections and audits of the inherited calibrated FIR method.
2. Optional weighted/composite estimators and their completed experiments.
3. An independent descriptor/learned-representation research direction. Its
   supplied constant-basis prototype has simulator results; learned features,
   effectiveness adaptation and geometric control do not yet have those results.

Keep these identities, original authorship, data and failed experiments intact.
Do not transfer review's independent-project `AGENTS.md`, canonical-repository
choice or dual-push policy into main as part of an algorithm import. Remote push
URLs are local configuration and do not transfer in a Git merge.

## 1. What needs fixing on main

| Priority | Verified issue at the pinned main | Import or implementation requirement |
|---|---|---|
| P0 | McNemar uses an absolute probability tolerance; duplicate scenario keys can be collapsed silently. | Import `openpi/mcnemar.py` and `mcnemar_crosscheck.py`; validate complete, unique, identical pairing keys before scoring. |
| P0 | LIBERO model-changing faults lack guaranteed episode cleanup. Cached faults can persist/compound after a run or exception. | Port `joint_fault.py` and `adaptive_law.run` lifecycle changes with `test_joint_fault_lifecycle.py`; restore all touched model/force fields in `finally`. |
| P0 | `reset(); set_init_state(...)` does not prove equal physical scenes. | Port `libero_reset.py`, use it explicitly, and require model/data fingerprints through the runner and scorer. Validate on the actual simulator. |
| P0 | Calibration sources and evaluation can overlap; episode/source provenance is incomplete. | Port the producer, replay, reader and validation changes together. Keep final evaluation disjoint from all data-dependent fitting, probing and selection; disclose any reuse within development. |
| P1 | Legacy ALOHA normalizes using unrelated channels and attenuates the estimated target. | Add telemetry, corrected-channel selection and innovation as named variants; measure healthy/task effects. Do not silently relabel old results. |
| P1 | Inverting a full map and then masking/clipping need not solve the restricted correction problem. | Add the bounded weighted allocator and retain its significant counterexample. Default rollout must be a research decision backed by fresh validation. |
| P1 | The raw-error integral does not isolate the proposed update law. | Import matched estimators using the same residual, interface and projection; disclose tuning and calibrated-integral semantics. |
| P1 | Healthy fitted contraction is insufficient to certify an added tracking-feedback loop. | Import composite qualification, source binding and augmented-recursion checks together. Test matched tracking-off ablations. |
| P2 | Command-only prediction omits measured state/memory; separately calibrated maps can disagree. | Evaluate the descriptor prototype under matched conditions. Distinguish predictive fit from causal input-map identification. |

**Important partial-fix caveats:** review's historical `adaptive_law` CLI still
leaves `scenario_reset=False`, and legacy calibration/evaluation defaults still
use initial state 45. Its overlap warning is not a complete hard exclusion across
FIR, sensitivity map, reference and tuning. Copying files or adding a flag alone
does not establish an honest held-out, physically paired experiment. The new
follow-up/tuning/descriptor paths implement stronger protocols; port that behavior.

### Correct interpretation of the legacy control law

For calibrated residual `r`, sensitivity `M`, bias `b`, and open deadzone gate,
the legacy update before clipping is

\[
\hat f^+=(1-\gamma)\hat f+\gamma\alpha(M^{-1}r-b),\qquad
\alpha=(1+\|r\|^2/\rho^2)^{-1},\qquad a_{sent}=a_{nom}-\hat f.
\]

For an ideal constant measurement, its equilibrium is approximately `alpha*f`.
Normalization changes estimation bias, not only adaptation speed. ALOHA can
compute this norm over 14 channels while correcting only six arm joints. The
healthy right-gripper residual accounts for 93.8% of **squared residual energy**;
this is not 93.8% of the norm itself. The gripper command and state in these ALOHA
logs are continuous. Servo conversion, limits, memory and contact require an
identified model; neither binary switching nor an inability of FIR filters to
represent steps establishes the mechanism.

Review's ALOHA innovation variant uses `r - M*f_hat - M*b` as its update error.
Its ideal fixed point avoids the same target-attenuation bias, subject to
matching/model accuracy and a deadzone neighborhood. Main already has innovation
for LIBERO and GR1, and selected-channel normalization for GR1; do not claim these
are entirely new. Legacy defaults remain available for reproducing old data.

Both branches already use the known corrected, pre-fault request in the legacy
residual history. Do not introduce nominal-action subtraction as a supposed fix:
that can mistake the adapter's own correction for an external disturbance.
Similarly, the adaptive subtraction sign is not globally reversed. The historical
`--static-corr` CLI has different conventions across runners; preserve/document
them or introduce an explicitly versioned interface.

### Physical faults and correction authority

The original joint map tested continuously updated **translation-only** correction
under constant +5 N·m torque, not a constant correction. It totals 103/140 in both
arms; joint 5 drops 12/20 to 3/20 with seven-test adjusted p=0.02734375. Joint 6
improves numerically 7/20 to 11/20. Do not write “only the elbow ever repairs.”

An additive torque disturbance is matched to direct torque input. Whether it is
compensable through the exposed Cartesian correction channels is a separate
question. Configuration dependence does not by itself establish unmatchedness.
Equal absolute torque also has different relative severity: +5 N·m is about 6.25%
of installed Panda joint 0–4 limits but 41.7% for joints 5–6.

The weighted solver includes channel selection, noise weighting, regularization
and box constraints inside the solve. That repairs a nominal optimization defect,
but a smaller modeled residual does not imply a smaller physical response or
greater task success. Probe state-dependent sensitivity and actual actuator
limits, and evaluate full trajectories/healthy harm. Do not use a fitted-map norm
or synthetic counterexample as a proved cause of the original task failure.

## 2. Estimators: retain exact meanings and evidence

| Variant | Meaning in this branch |
|---|---|
| Proposed/legacy | Calibrated estimate with the inherited normalization, gate and clipping. |
| DOB | Exponential filtering of the calibrated disturbance target. |
| RLS | Recursive coefficient fit with covariance and forgetting. |
| Kalman | Coefficient-state filter with process and calibrated observation noise. |
| Calibrated integral | Accumulation of the calibrated residual; persistent nonzero disturbance can drive it to its bound. It is not every possible integral-feedback design. |
| Oracle | Uses exact command-offset truth, or a declared physical-fault proxy such as nominal torque/kp, for a privileged diagnostic. The proxy is not ground-truth command compensation; neither guarantees maximum task success. |
| Composite | Prediction-based coefficient update plus qualified position-reference tracking feedback and leakage. |

All matched legacy estimators retain projection/clipping. DOB removes the
deadzone/normalization, not projection. The small comparison is
19/20, 17/20, 17/20, 18/20, 15/20 and 20/20 in the first six rows' order; it
establishes neither equivalence nor a universal ranking.

The larger ALOHA tuning study separates healthy fitting (seeds 3100–3129), search
(3200–3203), validation (3300–3307), and confirmation (3400–3429). Its 5,424 outcomes
comprise 2,608 search + 896 validation + 1,920 confirmation. Balanced scores are
legacy 50.00%, RLS 43.33%, DOB 42.14%, Kalman 36.90%, composite 35.95%, integral
31.90%. Composite minus Kalman is **−0.95 pp**, 95% whole-seed bootstrap interval
**[−9.52,+7.62] pp**, exact whole-seed sign-flip **p=0.86261**.

The selected composite reduces fitted-reference error in all eight conditions,
but tracking-update energy concentrates on one joint, and parameter error does
not consistently improve. Other selected hyperparameters differ too. Therefore:

- Run same-model/same-noise/same-clipping/same-leakage tracking-off/on ablations
  and independently tuned families with equal declared budgets.
- Include task success, healthy harm, reference error, coefficient error where
  observable, transient behavior and per-joint update energy.
- Expand omitted valid tuning limits such as static-parameter `Q=0` where the
  implementation supports them; do not claim the finite search found a global
  optimum or keep reusing confirmation seeds until a preferred method wins.

## 3. What the new descriptor method actually implements

### System identification: agent integration specification

**Objective:** import the new identification capabilities without conflating
the LIBERO descriptor prototype with the ALOHA composite reference model.
Main's default predictor uses per-channel command-history FIR filters and a
separately calibrated sensitivity map `M`. Main already supports optional MIMO
FIR; cross-channel prediction alone is not the new contribution.

| Change | Implementation and agent action |
|---|---|
| **State-aware prediction** | Import `learned_adaptation/identify.py`. Its ARX2 model predicts motion from the command and two previous measured-motion increments. Preserve causal history, centering and episode boundaries. |
| **Consistent input map** | In the descriptor path, use the same fitted `G` as `E` in reference propagation, the known-input residual and correction allocation. Do not substitute the legacy separately fitted `M` into this model. |
| **Composite reference** | Import the separate healthy joint-position reference fit in `openpi/composite_observer.py`, with preparation and validation tools. This ALOHA path retains its FIR observation model and sensitivity calibration. |
| **Better validation** | Retain whole-task holdout prediction, free-running prediction, excitation/conditioning diagnostics and source provenance. Keep final evaluation separate from fitting and model selection. |

The descriptor artifact has this interpretation:

```yaml
identification_target: discrete_nominal_interface_dynamics
source: learned_adaptation/identify.py
training_data: healthy_episodes
state:
  kind: arx2
  dimension: 12
  history: two_completed_six_dimensional_end_effector_motion_increments
  motion_coordinates: normalized_translation_and_relative_SO3_rotation
  centering: fitted_from_training_data_only
input: centered_known_sent_command_before_unknown_fault
identified_recurrence: x_next = F @ x + G @ a_sent_centered
descriptor_embedding:
  dt: one_policy_interval
  B: identity_by_convention
  A: F_minus_identity
  E: G
tested_disturbance_basis: constant_Phi_equals_E
```

**Identifiability constraint:** this fits discrete `F,G`. With
`xdot := x_next - x`, the assignments `B=I`, `A=F-I`, `E=G` reproduce that
recurrence. They do not separately identify physical `A,B`, inertia, or a
continuous-time plant. Premultiplying a descriptor equation by a nonsingular
matrix leaves its trajectories unchanged. A good fit to closed-loop policy
data also does not establish causal response to arbitrary corrective inputs.

The **ALOHA composite reference** is a different fit:
`q_ref_next = A_q @ q_ref + B_q @ a_nom + c_q`, using the live raw nominal
policy command. Here `B_q` is a fitted discrete input matrix; it is not the
descriptor's conventional `B=I`. Keep the position-reference input map separate
from the FIR residual observation map. Reference-model contraction qualifies
that fitted model, not the complete adaptive VLA/robot/contact loop.

Integration and acceptance requirements:

1. Port identification artifacts, readers and runner coordinate conventions
   together: `identify.py`, `libero_adapter.py`, `core.py` and `loop.py`.
   Preserve training episode keys, source hashes, scaling, means and model schema.
2. Port ALOHA reference fitting with `prepare_composite_reference.py`,
   `validate_composite_reference.py`, observer integration and their tests.
   Reject stale or incompatible artifacts before starting an experiment.
3. Check command/observation timing, no cross-episode history, and equality of
   the ARX recurrence with its descriptor embedding. The residual must subtract
   the known command that generated the observation.
4. Evaluate one-step and free-running prediction on held-out tasks; record
   excitation, conditioning, fitted stability and input-map variation. Retain
   covariance provenance: in-sample innovation covariance is not automatically
   a calibrated continuous-time observation-noise intensity.
5. Compare task success and healthy harm on fresh scenarios with matched
   calibration, horizon and tuning budgets. Predictive improvement alone does
   not prove task repair, learned-basis superiority or gripper/contact modeling.

Relevant existing checks:

```sh
python -m unittest learned_adaptation.test_identify learned_adaptation.test_libero_adapter -v
python -m unittest discover -s openpi -p 'test_composite_observer.py' -v
python -m unittest discover -s openpi -p 'test_composite_validation.py' -v
python -m unittest discover -s openpi -p 'test_calibration_pipeline.py' -v
```

### Descriptor adaptation equations

Use the complete conventions in
`docs/independent_adaptation/FORMULATION.md`. With an additive coefficient estimate,
the consistent nominal-map special case is

\[
\begin{aligned}
B\dot x&=Ax+E_0a_{sent}+\Phi_dz_d, &
B\dot x_r&=Ax_r+E_0a_{nom},\\
e&=x-x_r, & y&=B\dot x-Ax-E_0a_{sent},\\
E_0a_{sent}&=E_0a_{nom}-\Phi_d\hat z_d.&&
\end{aligned}
\]

For `H=B^{-1}Phi_d`, the continuous coefficient/metric equations are

\[
\begin{aligned}
\dot{\hat z}_d&=-\Lambda\hat z_d+
 P\Phi_d^TR^{-1}(y-\Phi_d\hat z_d)+PH^TLe,\\
\dot P&=-\Lambda P-P\Lambda^T+Q-P\Phi_d^TR^{-1}\Phi_dP.
\end{aligned}
\]

The tracking sign is positive for actual-minus-reference error and subtractive
compensation. For a matrix `Lambda`, do not replace the covariance drift with
`-2 Lambda P` without the scalar/commuting symmetric special-case assumptions.
The reference uses the live nominal command sequence; it is not a full
counterfactual rollout of the VLA observing a different healthy scene.

If estimating effectiveness, expand
`E(xi,z_E)=E0+sum_i E_i(xi)*z_E[i]` and form the joint regression against **E0**:
`y0=B*xdot-A*x-E0*a_sent = [E_1*a_sent,...,E_p*a_sent,Phi_d]*[z_E,z_d]`.
Allocate with `E_hat*a_sent=E0*a_nom-Phi_d*z_d_hat`. Do not subtract an unknown
true `E` while claiming to estimate its same coefficients; prediction residuals,
units and the sampled action timing must use one explicit convention. Loss of
rank cannot be undone by increasing an additive estimate.

Actual task-tested implementation:

- `learned_adaptation/identify.py` fits healthy ARX models. In the recorded runs,
  the state is 12-dimensional measured motion history, not seven joint positions.
  A URDF alone does not identify the deployed servo/OSC/contact response.
- `libero_adapter.py` embeds the ARX2 recurrence as `dt=1 policy interval`,
  `B=I`, `A=F-I`, `E=G`. Euler reference propagation and the left-state residual
  reproduce this identified **discrete** recurrence exactly.
- `loop.AdaptationLoop.observe` does use Van Loan forgetting propagation and
  Joseph covariance, followed by endpoint tracking injection. It is a split
  estimator, not an exact discretization or stability proof for the entire
  adaptive robot loop. The separate `loop_codex.DescriptorLoop` has a different
  state-propagation implementation; do not confuse them.
- The tested basis is **constant `Phi=E`**, with six action-equivalent additive
  coefficients. The gripper passes through. There is no trained feature network,
  identified physical wrench bridge, or online effectiveness adaptation in these
  task results. Coefficients are box-clipped before allocation; that is not a
  proved metric projection.
- Geometry/equivariance code has mathematical/kinematic checks. Wrench features
  need an identified torque-to-interface response map before entering motion
  residual rows. A valid coordinate transformation is not automatically a symmetry
  of a bounded robot, frozen policy, contact scene or fitted nominal controller.

### Latest descriptor evidence and corrections to its prose

Seven Spatial joint-fault cells total **112/140 → 135/140**, 28 fixed/5 broken.
Adding Object joint 3 (7/20 → 18/20) gives **119/160 → 153/160**, 39 fixed/5 broken.
All compare against their own frozen-faulted control. Healthy training inits are
35/36; evaluation inits are 37/38. The study is a promising integrated-controller
result, not a matched descriptor/composite/Kalman comparison.

Before importing `SWEEP_RESULT.md` or `paper/independent_method.tex` claims:

1. Correct coverage to **seven joints, seven Spatial cells plus one Object cell**.
2. Treat repeated `(suite,task,init)` scenarios across faults as clusters. The
   naive 39/5 pooled McNemar p assumes independent rows. The audit's exploratory
   cluster sign-flip remains favorable, but is post-hoc and not selection-adjusted.
3. Disclose that the 0.25 bound was chosen after the first 0.15 Object result.
   The reported 7.4% clipped fraction misses the originally registered below-5%
   manipulation check. “Unclipped” is a filename, not absence of clipping.
4. Do not compare historical joint 5's 12→3 with the new 14→19 as a controlled
   method effect: initial states, horizon and deployed fault source differ.
5. Without a same-scenario healthy arm, do not call all baseline failures
   fault-caused or claim return to healthy performance. An earlier ±3 episode
   fluctuation is not a statistical noise floor.
6. Add matched state-term, consistent-map, prediction-only/composite, feature and
   geometry ablations before assigning the improvement to any one component.
7. Correct the independent README's stale “no runner” status while retaining its
   accurate “no trained feature model” limitation. Reconcile paper equations,
   feasibility/constraint claims and actual runner behavior before publication.

## 4. Preserve these main-only additions

- Keep GR1's three-episode identification, `median/window` hold statistics,
  window **50–200**, videos and new paired datasets. The two-task +0.10 rad data
  are healthy 39/60, off 2/60, held 32/60. Identification episodes are included in
  these totals. Older GR1 files remain unpaired; the newer files have their own
  paired protocol. The scene-generator reseeding fix is already shared ancestry.
- Keep `groot_widowx_server.py`, `widowx_adapt.py`, `widowx_video.py`, setup and
  SimplerEnv/SAPIEN results. Continuous success is 6/20 versus 0/20; the held-only
  window is 14/20 versus 0/20 after three identification episodes, using window
  **30–150**. Report identification cost. `cell_tra005_g02.json` is incomplete.
- Do not overwrite main's `gr1_adapt.py` with review's older version. Preserve
  `gr1_video.py` scheme labels and the SimplerEnv bibliography entry.
- Equal success totals or an unresolved difference from healthy do not establish
  equivalence. WidowX also needs its own hidden-state/reset audit; LIBERO reset
  code does not validate SAPIEN. ALOHA sends absolute joint targets, not deltas.

## 5. Integration procedure

When implementing this handoff under an authorized development task, start a
clean integration worktree from main and import changes in reviewable groups.
These example commands create a separate branch; they do not merge/push main:

```sh
git fetch origin main review/dual-track-audit
git worktree add -b integrate/audit-fixes ../vla-main-audit-integration origin/main
git log --left-right --oneline origin/main...origin/review/dual-track-audit
git diff 96555f3..cda6c92 -- openpi/mcnemar.py openpi/joint_fault.py
```

First verify that `origin` is the intended mtaheriee repository and compare the
new remote SHAs with section 0. Re-audit changed files if they moved. Use a fresh
worktree/branch name if those names already exist. Do not reset a user's checkout.

| Group | Source files/change families | Integration notes |
|---|---|---|
| Statistics | `mcnemar.py`, `mcnemar_crosscheck.py`, scoring/key validation; `722576d`, `dda9040` | Smallest independent import. Recompute old statistics without changing outcomes. |
| Fault/reset/provenance | `joint_fault.py`, `libero_reset.py`, `paired_probe.py`, `adaptive_law.py`, `error_signal.py`, `openloop_id.py`, follow-up/calibration runners and tests; `c32f318` | Port as a producer/consumer/lifecycle group; require the new protocol explicitly. |
| ALOHA diagnostics/variants | `aloha_adapt.py`, `residual_channels.py`, telemetry/schema; `f801087`, `5936510`, `f20ee8b` | Preserve defaults and label selectable behavior. Tests must cover gate and fault-removal semantics. |
| Weighted allocation | `weighted_dob.py`, both runner integrations, healthy covariance, tests; `207feaf`, `1df4db6` | Include solver feasibility/KKT validation; preserve harmful cells. |
| Composite/tuning | `composite_observer.py`, reference preparation/validation, tuning/analyzers and tests; `d89d7bb`, `acd3b7f`, related validation commits | Model qualification, state/action ordering, source hashes and sample timing are dependencies. |
| Descriptor/geometry | `learned_adaptation/`, corresponding formulation, preregistrations and results | Experimental module; do not replace FIR evidence or imply trained-feature validation. |
| Paper/evidence | Figures, receipts, theory appendices, bibliography sources, result audits | Integrate after runtime/protocol decisions; regenerate the paper from final source. |

Commit IDs identify **change families**, not a safe ordered cherry-pick script.
Some commits contain earlier manuscript rewrites, many results or dependent
refactors. Inspect their contents; port complete current dependency groups into
small attributed commits rather than replaying all 120 commits blindly.

Read-only three-way inspection predicts conflicts in `README.md`,
`docs/ADAPTIVE_CONTROL_VLA.md`, `paper/iclr_draft.tex` and the binary PDF.
`openpi/gr1_video.py` and `paper/refs.bib` also changed on both branches and need
semantic review even if Git merges the text. Preserve main's newer results and
review's negative evidence in a coherent rewrite. Regenerate the PDF; choosing
one old binary cannot resolve a scientific source conflict.

## 6. What must be rerun, and what can be rescored

| Claim or change | Required action |
|---|---|
| McNemar tolerance / duplicate-key correction | Re-score valid archived outcomes. Investigate invalid pair sets; do not silently intersect/drop episodes. |
| Fully held-out historical calibration | Refit with recorded, disjoint sources and rerun the dependent comparisons. Adding metadata now does not repair historical provenance. |
| Historical friction/lock/gain efficacy with cached model mutation | Rerun affected cells with healthy/off/on controls, restoration and initial-state/model fingerprints; preserve historical data as such. |
| Paired physical-scene claims without fingerprints | Recover equivalent reset evidence if it exists; otherwise rerun under the verified protocol. Policy draws need not be identical, but record the sampling scheme. |
| Existing weighted/composite confirmation | Retain completed results and limitations; no need to rerun merely to obtain a different ranking. Changes to model, gains or rule require a new confirmation split. |
| Descriptor advantage and failure repair | Freeze deployed sources, calibration, horizon, bound and analysis; run healthy/off/legacy/prediction-only/composite on fresh scenarios with the same protocols. Extend Object/Goal and then robot-specific adapters. |
| Learned basis/effectiveness/geometry contribution | Train/identify missing components on separate data first, then run controlled ablations. Algebra tests and old FIR trials cannot substitute. |
| GR1 held robustness / WidowX generalization | Confirm the selected hold rule across independently identified vectors and fresh scenarios; preserve per-robot action semantics and add reset evidence. |

Choose estimands and multiple-comparison families before new outcomes. Cluster
shared task/initial-state scenarios across faults and whole seeds across tuning
conditions. Report healthy harm, all registered cells, identification cost,
selection history, saturation and failed/partial runs. “Fixed/broken” denotes
observed paired outcomes, including policy sampling variation.

## 7. Acceptance checks and reporting

The comparison audit actually passed **40 focused legacy-module unit tests**,
both legacy runner self-tests, and **124 independent-module tests (one skipped
because optional native MuJoCo is unavailable)**. It independently recounted
stored outcome tables and checked exact statistics. It did not launch new
simulator experiments, validate a completed merge, or rerun the full archival
hash/telemetry/bootstrap pipeline. See the supporting memos for exact scope.

Useful post-import checks from the integration repository root:

```sh
python -m unittest discover -s openpi -p 'test_joint_fault_lifecycle.py' -v
python -m unittest discover -s openpi -p 'test_calibration_pipeline.py' -v
python -m unittest discover -s openpi -p 'test_weighted_dob.py' -v
python -m unittest discover -s openpi -p 'test_composite_observer.py' -v
python -m unittest discover -s openpi -p 'test_composite_validation.py' -v
python openpi/adaptive_law.py --selftest
python openpi/aloha_adapt.py --selftest
python -m unittest discover -s learned_adaptation -t . -p 'test_*.py'
```

Also run affected ALOHA/tuning/collection tests and simulator-specific reset and
interface checks. An estimator passing NumPy tests does not certify contact-rich
task stability. Reproduce statistics using the commands in the results memo.

For the paper, preserve verified citations and source metadata, rebuild plots
from their own method/protocol's data, and separate constructed theory diagrams
from robot evidence. Retain main's newer GR1/WidowX findings and review's failures.
The nine-page predecessor manuscript and `independent_method.tex` represent
different studies; do not substitute the latter while retaining the former's
success tables. After combining the paper tooling, rebuild and check:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr_draft.tex
python check_submission_layout.py --require-main-pages 9
```

Finish an integration with an explicit list of imported fixes, preserved main
features, unchanged historical evidence, tests actually passed, new measurements
still required and unresolved regressions. Keep any subsequent publication/merge
within the user's authorization and the destination repository's policy.
