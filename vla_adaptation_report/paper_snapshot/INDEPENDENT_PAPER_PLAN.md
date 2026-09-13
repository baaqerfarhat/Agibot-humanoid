# Prospective paper: learned dynamics adaptation at a frozen-policy interface

This plan supersedes the old method as the direction of the independent repository.
The existing `iclr_draft.tex`/PDF remains the nine-page predecessor study until new
training and confirmation results exist. Replacing its method equations while retaining
its old success tables would falsely attribute those results to a different algorithm.

A possible descriptive title is **Geometry-Aware Composite Dynamics Adaptation for
Frozen Robot Policies**. This is a working title, not a claim that the architecture
or its equivariant variant is already validated.

## The central research question

Can a nominal actuator/contact model and a transferable learned representation of
physical disturbance and input effectiveness make rapid adaptation more reliable at
a frozen policy's actual control interface?

The proposed method adapts latent dynamics coefficients rather than VLA weights or
legacy per-channel correction averages. The main conceptual separation is between
known actuator behavior, learned nuisance dynamics, intended contact, and loss of
actuation authority. Geometry should impose correct transformation laws on both the
representation and the adaptation state, not merely decorate the network input.

## Candidate contributions and the evidence they need

| Candidate contribution | Already available | Required before claiming the contribution empirically |
|---|---|---|
| Nominal actuator model plus learned residual/effectiveness representation at a VLA interface | Equations, interface audit and concrete gripper saturation mismatch | Held-out actuator/contact prediction; trained feature model; demonstration that the chosen state is sufficient |
| Composite coefficient learning compatible with healthy position tracking | Continuous descriptor-model derivation, Lyapunov identity and reference tests. **New:** a sampled implementation now runs end to end on a frozen policy under physical joint faults with the task evidence in `SWEEP_RESULT.md` | Validated region/metric, matched prediction-only comparison isolating the tracking term causally, constraints/contact analysis |
| Geometry-consistent adaptation | [Typed geometric specification](../docs/independent_adaptation/GEOMETRIC_METHOD.md), wrench/servo separation, constructive finite-group basis and effectiveness tensor, exact estimator/allocation commutation tests. **New:** gravity-preserving yaw *placement* covariance of the wrench basis is established (defect 4.4e-15 on Panda over 64 configurations x 8 yaws; 1.3e-15 on an independent analytic chain, against an invariance defect of 5.66), giving the commutant reductions Q 21->7 and Lambda 36->12 | Identified physical conversion `C_tau` (rectangular, state dependent, 6 x n_q) at the same interface and interval; a nominal model that respects the same law, which the current ARX does not (defects .080/.153 at .37 rad); available transformed context; trained features and cross-configuration or embodiment evaluation. No nontrivial *active* symmetry of the deployed experiment is claimed or expected |
| Hierarchical context and latent disturbances | HMAC source comparison and separate coefficient-block design | Evidence that both blocks contribute beyond an equal-capacity single block; streaming versus stationary disturbance ablation |
| Reliable manipulation under physical change | Predecessor failures motivate the question | Independent task confirmation with all joints/grippers, healthy controls, contact conditions and disclosed authority limits |

Learned bases and composite adaptation are established ideas. Applying them to a
manipulator is not by itself a defensible novelty claim. The strongest potential
contribution is the connection between physical interface modeling, geometric
representation and a consistently transformed adaptive estimator/controller,
supported by independent manipulation evidence.

The geometric addition is a specified composition of established ingredients;
it does not introduce the composite Riccati law, wrench pullback, Reynolds
projection, or equivariant estimation. The specification's prior-work table also
names the EqF coordinate/noise-consistency precedent. A publishable novelty claim
still needs a broader adaptive-control comparison and evidence beyond these
algebraic tests. No predecessor result evaluates this geometric implementation.

## Introduction structure and an honest opening

Suggested opening, while results are pending:

> A frozen robot policy specifies an action, but the physical effect of that action
> depends on the actuator, the mechanism and its interaction with the environment.
> These dependencies change under wear, friction, contact and external loading.
> An adapter must distinguish such changes from the actuator's ordinary response:
> a gripper that cannot close through an object is not necessarily malfunctioning,
> and a disturbance estimate does not create a missing actuation direction.

Continue with four precise steps:

1. Explain why adapting the command-to-motion dynamics is useful while keeping the
   policy fixed. Cite the evaluated policy and deployment setting.
2. Introduce the existing learning/control lineage: Neural-Fly's shared features and
   composite adaptation, MAGIC's context-dependent input map, and HMAC's multiple
   disturbance sources. State what they already establish in their own settings.
3. Identify the manipulator/VLA interface problem: mixed command coordinates,
   internal servo behavior, contact objectives and restricted control inputs. Use
   the gripper conversion mismatch as a motivation with its provenance limit.
4. Introduce the proposed model and the role of geometry; state contributions only
   at the level supported by the completed training, proofs and experiments.

After confirmation, replace prospective language with concrete outcomes and their
comparators/denominators. If geometry or composite feedback does not help, report
that result and narrow the contribution. Do not write a desired estimator ranking
into the abstract before observing the held-out comparison.

## Related work as a comparison map

Organize around questions rather than a list of acronyms:

- **Where is the model adapted?** Frozen-policy correction, learned residual policies,
  context adaptation, actuator/servo identification and physical VLA fault work.
- **What information produces the learned representation?** Neural-Fly/DAIML,
  MAGIC visual and command-dependent features, HMAC manageable/latent hierarchy.
- **What closes the loop?** Prediction-only identification, composite adaptation,
  covariance/metric preconditioning and optional learned mirror geometry. Cite the
  original control results; distinguish their mechanical tracking state from the
  proposed qualified position-servo model.
- **Which geometry is relevant?** Configuration/tangent/cotangent spaces, body-frame
  pose/wrench features and legitimate morphological symmetries. Force-impedance
  control and geodesic planning are nearby alternatives, not automatic components.

Nearest learned adaptive controllers and closest physical-fault methods belong in
the main paper. Broader taxonomy and all five local-paper readings can expand in
the appendix. Keep verified published citation records already in `refs.bib`; add
new records only from primary metadata, without copying placeholder journal/DOI
fields from a manuscript template.

## Method presentation

Start with one model diagram: frozen policy → known command map → feasible action
solve → servo/robot/contact. Show the healthy reference, residual measurement and
composite coefficient dynamics as explicit feedback paths. Draw unknown additive
forces and input effectiveness at their distinct locations. Use the same symbols
in the figure and equations.

Then explain in execution order: modeled state/input and units; healthy model and
its identified parameters; shared feature training; prediction measurement from the
known sent input; bounded action solve; coefficient/covariance update; feedback timing.
The full derivation is in [FORMULATION.md](../docs/independent_adaptation/FORMULATION.md).
State whether the implemented basis depends on the current or delayed input; a
current-input feature requires a well-defined implicit solve or explicit input-map
parameterization.

The main method must disclose what is trained offline, what is tuned on faulted
development data, what is fixed during evaluation and what updates online. An
unavailable effort/contact sensor cannot silently enter the estimator. Privileged
simulator force labels may train a model only if that information split is explicit.

## Evaluation sequence

1. **Nominal qualification.** Fit on healthy data from multiple motion/contact modes.
   Validate disjoint trajectories with command conversion, saturation and delay
   represented. Compare q-only, q/velocity and actuator-memory/contact models at
   comparable capacity. Reject a preferred structure if its error is materially worse.
2. **Feature learning.** Separate environment/task support and query trajectories.
   Reserve whole fault/contact environments for final transfer. Use a fixed healthy
   model initially to avoid nominal/disturbance identification ambiguity.
3. **Model-matched ablations.** Compare constant basis, unconstrained learned basis,
   context-conditioned basis and geometric basis at stated parameter counts. Cross
   each with prediction-only and composite updates using a common model and sensors.
   Add input effectiveness and HMAC-style hierarchy as independently attributable factors.
4. **Fair development.** Fix tuning budgets and selection metrics before search.
   For a causal test of tracking feedback, hold all other parameters identical in
   an additional ablation. Selected-best comparisons answer a separate question.
5. **Task confirmation.** Test all arm joints and grippers; offsets, torque, friction,
   effectiveness loss and time changes; retain complete-lock cells as authority
   limits. Report healthy harm and success changes per task/condition, not only an
   average. Register comparison families and use paired reset fingerprints where valid.
6. **Transfer.** Test new contacts/objects and robot configurations first, then a
   second embodiment with its own explicit nominal model and input map. A Panda
   torque interface and its original Cartesian interface are different protocols.

Track task outcome, reference/state error, force/prediction error, allocation
residual, saturation duration, contact/force violations, parameter uncertainty,
compute/latency and calibration budget. Evaluate coefficients against truth only
when their basis gauge is fixed. Joint torque magnitude should be reported both
absolutely and relative to each joint's available authority where appropriate.

The selected basis must be trained before claiming a Neural-Fly-style learned
controller. The old 103/140 totals or composite/Kalman tuning scores cannot serve as
its confirmation cohort. A first benchmark for this plan has now run: eight
paired cells over six joints and two suites, 39 of 41 opportunities repaired
against 5 of 119 healthy episodes broken (`SWEEP_RESULT.md`). It evaluates the
plain supplied-basis descriptor controller with an untrained constant basis, so
it does not confirm the learned-feature or geometric components.

## Figures and tables

| Display | Main question | Content |
|---|---|---|
| Figure 1 | What is modeled, learned and adapted? | Symbolic online loop plus offline feature-learning inset, robot/interface images and one provenance-backed outcome illustration when available |
| Figure 2 | Does the nominal model explain the gripper correctly? | Requested versus feasible finger target; free/contact residuals; held-out predictions with units and per-mode sample counts |
| Figure 3 | Does the learned representation transfer? | Query residual/error across unseen environments, data-efficiency curves and a geometric transform-consistency diagnostic |
| Figure 4 | Does better dynamics prediction improve tasks? | Complete joint/task success matrix, healthy harm and prespecified paired contrasts with uncertainty |
| Table 1 | What is the experimental setting? | Robots, modeled state, actual input interface, available sensors, training/selection/confirmation split and fault types |
| Table 2 | What earns the improvement? | Matched basis/update/context/effectiveness ablations, exact counts, compute and calibration budget |

Use vector plots at final width, physical units, consistent method colors and
paired difference intervals where appropriate. A learned-feature embedding plot
alone is weak evidence of control quality. Avoid selected successes without the
full failure/healthy-control accounting. All plots need source artifacts and
reproduction scripts; use authentic images for observed behaviors.

## Main-paper budget: nine pages, references excluded

A starting budget including floats is: introduction/overview 1.7 pages; related
work 0.8; method 2.0; conditional analysis 1.2; experiments 2.9; limitations and
conclusion 0.4. This totals nine pages. Revise the allocation once real figures and
results exist; do not fill the remaining space with prospective claims.

Keep the modeled actuator state, command boundary, feature-training split, exact
online update, central assumptions, primary contrast and important negative results
in the main paper. Move full proofs, all symmetry transformations, controller/MJCF
parameters, full seed tables, extended source comparisons, extra transfer cases and
video provenance to the appendix. Cite assumptions and inherited methods where
used; there is no citation quota. The [ICLR writing study](style_study/ICLR_WRITING_STUDY.md)
provides the detailed editorial standards.
