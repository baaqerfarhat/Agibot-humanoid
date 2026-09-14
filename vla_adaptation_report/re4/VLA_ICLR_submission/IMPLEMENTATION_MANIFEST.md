# Implementation and physical-model evidence manifest

This document identifies the implementation evidence represented in the manuscript and the fields that remain unresolved. It does not supply missing code or reconstructed observations.

## Documented interface families

| Interface | Predictor response | Offset update | Recorded evaluation schedule |
|---|---|---|---|
| LIBERO Panda | Position increments and relative rotations, FIR K=6 | Observation attenuation | Within episode |
| ALOHA | Absolute joint positions; K unrecorded | Observation attenuation | Prior identification, then hold |
| Fourier GR1 | Absolute joint positions; K unrecorded | Innovation normalization | Three prior episodes; held median |

New robots receive new healthy calibration. For the primary LIBERO condition, six coordinates are estimated and three rotation coordinates corrected. Correction support was informed by fault diagnostics; a held-out support-selection split is not documented. Healthy calibration and sensitivity probes form part of the information budget.

The headline aggregate comparison remains 28/120 frozen and 78/120 corrected,
with 51 recoveries and one regression. Figure 1 is a mechanism diagram and
does not introduce an additional empirical cohort.

## Per-cohort records still required

Recover the correction site and exact decoder/action mapping; units and signs; healthy centering; residual normalizer coordinates and scale; update gain and gate behavior; projection bounds; clipping order; snapshot application time and age; interpolation/chunk schedule; fault insertion; task/seed/initial-state pairing; calibration arrays; raw issued commands and timestamped motion. The original description names native output-bias editing; later prose gives an action-subtraction model without documenting the transition. The recorded outcomes are not retrospectively assigned to a newly implemented external adapter.

The current update equations are the documented prediction-only attenuation and innovation families. The continuous composite controller in Appendix D and the fixed-data gradient candidate in Appendix B.3 are unevaluated extensions. No new online tracking signal or geodesic computation is attributed to the reported experiments.

## Required execution-model record

Specify the robot and low-level controller, command interface and admissible command set, operating region, complete execution state, and nominal same-command comparison. Include object/contact states when their evolution determines the transition, or provide a uniform omitted-effect bound for the reduced model. Record common fixed geometry and prescribed external motion separately from endogenous object state.

Establish the direct fixed-command distance contraction throughout the declared region. Declare a fixed physical baseline distance and uniform lower/upper equivalence constants for any time-varying certificate distance, so rescaling cannot create artificial contraction. In smooth modes, the metric Jacobian inequality is a sufficient test; a stable linearization alone does not establish a nonlinear/contact certificate. Record flow assumptions, event/reset and event-time sensitivity, and the treatment of command/reference jumps. At impacts or mode changes, define cross-mode distances for compared states, verify the complete sampled transition, or stop the certificate at the unverified event.

Keep unsupported fault, estimation, snapshot-age, and correction-realization terms in the interface mismatch budget. Allocate base, direct history, and omitted contact transition forcing once. Retain any dependence on execution/estimation error in the small-gain condition. The supplied run template contains nullable fields for this evidence; no robot metric or contact bound has been filled in by inference.

## Decisive matched comparisons

Recover existing implementation records before repeating recoverable experiments. Add matched healthy and calibrated static/FIR controls; identical-episode attenuation versus innovation; known-fault correction through the same interface and limits; and held versus continued adaptation from the same identified estimate.

For execution-envelope validation, use fixed command replay with each adapter observing its own physical response. Common-residual replay only compares estimator dynamics. Independent VLA task rollouts evaluate task benefit and can produce different commands; cross-adapter trajectory comparisons then retain the nominal-response divergence term. Record measured execution error, calibrated budgets, clipping and projection activation, holds, age, contact events, and any domain exit.

All current task-success results are simulated. The constructed servo is an analytical instance with a matched predictor and fixed centering bias. It is not a uniform manipulation certificate, a real-robot recovery demonstration, or a measured VLA execution trace.
