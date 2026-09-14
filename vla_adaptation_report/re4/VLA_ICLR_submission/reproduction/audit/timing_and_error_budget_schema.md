# Trace schema, timing, and empirical recovery budget

This is an instrumentation specification for new or recovered execution traces. No timing or robot-trajectory measurements are supplied by this template. Use injected truth only in the simulator's offline evaluator or an explicitly labeled privileged comparator.

## One row per executed physical-interface command

Store a CSV index plus vector arrays in NPZ/Parquet. Use immutable episode/configuration/calibration IDs. Record each signal before it is overwritten; store units, coordinate order, and clock domains in the configuration JSON.

| Fields | Required meaning |
|---|---|
| `episode_id`, `step_id`, `chunk_id`, `action_index_in_chunk` | Executed command identity, including unused/discarded chunks separately. |
| `sensor_timestamp_s`, `sensor_available_s`, `update_start_s`, `update_end_s`, `command_apply_s` | Monotonic measured times. Record simulator physical time separately from wall time. |
| `policy_start_s`, `policy_end_s`, `decode_start_s`, `decode_end_s` | Base-policy inference and decoding, distinct from adapter work. |
| `estimate_update_id`, `estimate_applied_id` | Which estimator state was updated and which one actually generated the command; equal indices cannot be assumed. |
| `nominal_action`, `requested_corrected_action`, `pre_fault_command`, `executed_interface_action` | All in the same declared physical-interface units. The pre-fault command includes correction realization and upstream clipping. Fault ordering must match the model. |
| `vla_supplied_command`, `supplied_command_sequence_id`, `command_source_step_id` | Actual supplied command `a_k` and its timing, units and mapping to `nominal_action`; preserve this stream for the nominal physical comparison. Do not replace it with commands from another rollout. |
| `execution_state`, `low_level_controller_state`, `execution_delay_state` | Robot and required execution-controller coordinates defining `xi_k`, including physical/controller delays or their valid effective-input realization. VLA internal memory is outside the primary metric. |
| `nominal_execution_state`, `nominal_replay_id`, `reference_initialization_id`, `reference_model_error` | Nominal robot-controller response `xi_k^{nom}` to the logged supplied command stream; record whether it is analytical, rigorously computed, approximate, or unavailable. |
| `command_admissible`, `command_hold_or_interpolation`, `controller_reset`, `reference_jump` | Declared command-set membership, between-sample command generation, resets, and any jump term required by the chosen reference coordinates. |
| `measurement`, `prediction`, `raw_residual`, `centered_fault_observation` | Preserve the raw residual `e_k^{res}` and processed observation `z_k`, with their exact timestamp convention. |
| `estimate_before`, `estimate_after`, `estimate_applied` | Estimator state at update and at application. Log initialization and reset events. |
| `gain`, `attenuation`, `gate_open`, `closed_gate_operation` | Exact update coefficients and hold/leakage choice; do not infer from a threshold afterward. |
| `normalization_indices`, `normalization_scales`, `P_r`, `D`, `P_b` | Keep residual normalization, physical correction support, and native edit map separate; log their coordinate units and counts. Prefer fixed values referenced by config ID. |
| `projection_active`, `command_clip_active`, `actuator_saturation` | Distinguish estimate bounds, command clipping, and lower-level saturation. Log requested/applied values. |
| `contact_label`, `joint_limit_flags`, `task_phase` | Measured/deployed signals versus simulator-only evaluation labels must be distinguishable. |
| `fault_truth`, `fault_onset_time`, `fault_removal_time` | Evaluator-only values; record access separation from the deployed update. |
| `decoder_base_action`, `decoder_edited_action`, `decoder_draw_id`, `decoder_state_id` | Native-edit diagnostic at identical observation, policy/controller memory, action-buffer state, and random draw; include denormalization and a declared clipping convention. |
| `uncentered_transformed_observation`, `centering_value`, `centering_stage` | Save `v_k = S^{-1}e_k^{res}`, the baseline, and exact order relative to gate, attenuation, and projection. |
| `persistence_prediction`, `predicted_increment`, `measured_increment` | Evaluate trivial position persistence and actual motion prediction on identical timestamps. |
| `metric_id`, `metric_domain_id`, `metric_status`, `physical_baseline_distance_id`, `distance_equivalence_lower`, `distance_equivalence_upper`, `phase_rule_id`, `admissible_command_set_id` | Use `analytic_uniform`, `rigorously_verified_domain`, `empirical_candidate`, or `unassessed`; identify the robot-controller model and verification artifact. For a time-varying certificate distance, record uniform comparison constants to a fixed physical baseline distance. Never infer status from the plot. |
| `trajectory_metric_distance`, `empirical_envelope`, `envelope_violation`, `domain_exit` | For the primary analysis use `d_Mc(xi_k, xi_k^{nom})` under the same supplied commands. Log state, norm, reference, scope, and bound status. A missing reference gives an unavailable distance, not zero. |
| `additional_history_input`, `history_reference`, `history_forcing_bound`, `history_bound_status` | History effects on execution not captured by `(xi,a,q)`, with norm and sensitivity; log `L_h * h_bound` separately from baseline forcing. |

## Physical closure, contact and comparison records

The current symbols are `a_k` (supplied command), `e_k^{res}` (raw residual), `xi_k^{nom}` (same-command nominal physical response), and `lambda_k^{exec}` (sampled execution amplification). Retain existing machine-readable keys for backward compatibility; populate their meanings according to this mapping. Continuous-time commands remain `r(t)` where used.

| Additional fields | Required meaning |
|---|---|
| `robot_state_components`, `actuator_state_components`, `controller_state_components`, `object_state_components`, `compliance_state_components` | Exact coordinate names, order, units, dynamics and observability; include every retained physical degree of freedom needed for closure. |
| `geometry_parameter_id`, `geometry_trajectory_id`, `geometry_common_status`, `geometry_dynamic_state_components` | Fixed geometry versus externally prescribed common geometry versus dynamically evolving object/environment state; identify the nominal counterpart. |
| `contact_mode`, `nominal_contact_mode`, `mode_observation_source`, `contact_model_id` | Actual and nominal modes; simulator-only labels must not be treated as deployed estimator inputs. A common command does not enforce common modes. |
| `contact_state_included`, `omitted_contact_effect`, `contact_bound_domain`, `contact_bound_status` | State inclusion or bounded-remainder choice and its validity region; unavailable terms remain unavailable. |
| `d_base`, `d_hist`, `d_contact`, `continuous_bound_units` | Disjoint continuous vector-field contributions, separate from matched parameter coupling; trace each to its physical/model source. |
| `eta_base`, `eta_hist`, `eta_contact`, `forcing_allocation_id`, `continuous_to_sampled_bound_artifact` | Disjoint metric-distance forcing over each execution interval; include finite-step sensitivity/propagation. Do not double count terms already in `q`. |
| `guard_id`, `actual_event_time`, `nominal_event_time`, `pre_event_mode`, `post_event_mode`, `reset_map_id` | Actual versus nominal events and time stamps, reset rules, unmatched-event status and clock alignment. |
| `jump_gain`, `saltation_bound_artifact`, `event_timing_bound`, `event_forcing_allocation`, `hybrid_comparison_status` | Verified jump/event-time comparison or explicit bounded unmatched-event effect. A bounded flow force does not cover an impulse. Record the single forcing component receiving each jump contribution. |
| `command_replay_protocol_id`, `fixed_command_trace_hash`, `adapter_measurement_stream_id` | Identify fixed-command execution replay; each adapter uses its own physical measurements. A common residual-stream replay is an observer-only comparison. |
| `adapter_id`, `nominal_response_id_by_adapter`, `nominal_response_divergence`, `command_divergence`, `command_sensitivity_bound_status` | For independently generated command streams retain per-arm nominal responses and their additional distance; keep task outcomes as separate measurements. |
| `manipulation_certificate_scope`, `contact_closure_verification_artifact` | Distinguish a verified servo/free-space region, a contact-mode region, a verified hybrid manipulation region, and an empirical diagnostic. |

Allocate each omitted physical effect once. Use `d_eff = d_base + d_hist + d_contact` during flow intervals, with separate justified hybrid jumps, and `eta_k = eta_k^{base} + eta_k^{hist} + eta_k^{contact}` for the sampled transition. Baseline must exclude explicitly named history/contact terms. Contact-induced changes already in retained state dynamics or in `q` are not additional contact forcing. Contact-event/timing mismatch can be allocated to `eta_contact`; purely reference-coordinate jumps can instead be allocated to baseline with an explicit reference label. Preserve the allocation table so neither is silently duplicated. Do not substitute zero for an unknown component. Retain execution-error-dependent contact/history terms in the coupled comparison and its small-gain/domain check. A shared physical cause may have separate predictor and direct plant effects; document those distinct channels while avoiding duplicate accounting of the same transition effect.

A manipulation trace requires closure over robot-object/contact dynamics or a justified remainder. Treat fixed geometry/common external geometry as prescribed only when it is actually common; endogenous object motion and contact modes require states or error bounds. Verify mode/reset conditions and event-time regularity for the declared comparison. If such evidence is missing, restrict the certificate's label to the verified servo/free-space regime; plotted contact envelopes remain empirical.

For a cross-adapter physical comparison under a common metric and domain, retain the nominal-response term in `d_k(xi_k^i,xi_k^j) <= X_k^i + d_k(xi_k^{nom,i},xi_k^{nom,j}) + X_k^j`. Fixed command replay with matched nominal initialization/geometry makes the two nominal comparison systems identical when the stated model is deterministic and well posed. Independent task rollouts generally do not. If replacing the nominal-response distance by a command-divergence bound, log the verified command sensitivity and any additional geometry, reset and reference discrepancy. No policy Jacobian is thereby required for the execution theorem.

## Align observations with commands before computing a residual

Recover the actual predictor convention from code. Explicitly record whether `measurement_t` is caused by command `u_t` or `u_{t-1}` and whether rotation is a group difference or subtraction of chart coordinates. FIR history must contain the actual command at its modeled pre-fault interface. If native decoder editing or clipping makes `a-D*fhat` differ from that command, log the latter. Do not align channels by array index when sensor delays differ.

For action chunks, record the application map `j(k)`: the update whose estimate is used for physical command step `k`. Define age as `command_apply_s - update_end_s[j(k)]` and maintain the whole buffered command state if the predictor uses it. Delay and held estimates belong in the empirical error budget.

## Evaluate the error decomposition on a common interface

For additive input mismatch, use the nominal action at the adapted trajectory's **current observation/state**, not an action from a different healthy trajectory. With applied estimate `fhat_applied`, compute

\[
w_k=z_k-f_k,\qquad
\zeta_k=u_k-a_k+D\hat f_{\mathrm{applied},k},\qquad
q_k=a^{\mathrm{exec}}_k-a_k.
\]

Check the identity

\[
q_k=(I-D)f_k+D(f_k-\hat f_{\mathrm{applied},k})+\zeta_k
\]

only when the logged injection is exactly `a_exec = u+f` at the stated interface. With post-fault clipping, gain loss, or joint-dynamics faults, use the actual actuator map and log its additional remainder; forcing this identity onto another interface creates a false diagnostic. The nominal/base decoder call and edited call must also share internal policy memory, controller state, and decoder random draw when those affect decoding. Save the diagnostic state rather than comparing actions at different healthy/adapted rollout states. Include the actual clipping operation in the compared map or log its remainder separately.

For the selected physical corrections report

\[
E_{D,k}^{\rm app}=\|D(f_k-\hat f_{\mathrm{applied},k})\|,
\]

in addition to the full estimate error. A six-coordinate estimator and three-coordinate physical correction have different dimensions from a native tensor edit. Unused estimate coordinates may still affect attenuation through `P_r`; record that indirect path rather than declaring them irrelevant. For a delayed estimate, split

\[
f_k-\hat f_{\mathrm{applied},k}
=(f_k-\hat f_{k})+(\hat f_k-\hat f_{\mathrm{applied},k}).
\]

Report vector components and declared scaled norms. The decomposition can contain cancellation; show the signed/component values as well as the conservative sum of norms. Plot `w`, estimation error, uncorrected coordinates, realization/clipping error, and applied-estimate age against seconds from fault onset. Shade contact and FIR history-fill windows. A sensitivity condition number measures inversion conditioning, not prediction accuracy or superposition validity on those trajectories.

For absolute-position predictors, include persistence `yhat_t=y_{t-1}` and compare both position and increment error on held-out episodes. For healthy centering, preserve raw residual means, transformed means, mean attenuated targets, and settled estimates separately; nonlinear attenuation generally makes `E[s z]` differ from `E[z]`. Log averaging windows, units, and projection activity. Compare actual recorded ordering with the constant-data centering check in `IMPLEMENTATION_MANIFEST.md`; do not assign its hypothetical result to an unobserved run.

For a uniform six-axis fault, compare the simultaneous response with the sum of the individually probed responses. Repeat on independently signed/magnituded faults before claiming arbitrary six-coordinate identification.

## Execution contraction and phase diagnostics

The primary metric describes the robot plus necessary low-level execution controller, conditional on the supplied VLA command stream. Record `F_exec(xi,r,t)` or its sampled map `F_exec,k(xi,a_k)`, controller gains, state coordinates, command set, and operating domain. The reference is the nominal physical response to the **same supplied commands**, with a declared initial condition. Commands may have been generated in feedback during deployment; the comparison holds their realized stream common. A separately evolving healthy VLA rollout is a different empirical task baseline. Neither that rollout's distance nor the same-command execution distance can be inferred from the command residual alone.

Form the execution Jacobian with respect to `xi` at fixed `r` and include the low-level feedback derivative. There is no VLA Jacobian in the primary certificate. Prefer `M_c(xi,t)` independent of `r` and the adapted estimate. Save the exact differential/discrete inequality, its uniformity over admissible commands, lower/upper eigenvalue bounds, geodesic containment argument, and verification artifact. A metric based on a Hurwitz linear tracking model can be constructed from its Lyapunov equation; a stable linearization does not certify the full nonlinear robot over a region. Identify model uncertainty and its forcing treatment. A newly designed controller and an analysis of the deployed controller are distinct records.

Declare execution, observer, and command-source states separately before forming a distance. The VLA's observation memory and nominal pre-correction chunks are command-source records, not coordinates required by the primary metric. Log them if needed for native-edit diagnostics, without expanding the execution certificate. Keep actual issued-command FIR history and corrected queues separate where their effects are captured by the supplied command and mismatch. At fixed execution state, supplied command and other declared inputs, verify that equal `q` implies equal execution transition across admissible observer states. If a history directly affects the low-level controller or plant beyond these inputs, retain the necessary state or an extra input with a reference, uniform bound and sensitivity. Actuator delay states cannot be dropped because they are stored in software. A history that only alters future VLA commands does not by itself violate this same-command comparison.

Perfect physical compensation can leave the observer's issued-command buffer offset from healthy, so raw-buffer mismatch is not an execution-state tracking failure. Report any extra history forcing separately; it need not vanish when `q=0`. Preserve command changes, interpolation/holds and controller resets. The nominal physical response generally remains continuous across a piecewise constant command change; if using a directly commanded target or jumping error coordinates instead, include the associated jump bound or a feasible reference construction.

For prediction-only deployment, the nominal physical response may be an analytical comparison system, with no online metric/reference/geodesic calculation. Offline replay must use the actual logged command sequence and record reference-model and initialization error. For implemented composite tracking feedback, record the online reference, metric/geodesic computation and approximation errors because they enter the adaptive law. Unavailable reference/state values must remain unavailable. The constructed servo provides one verified example; this schema does not supply a certificate for the reported robots.

For learned/fitted metrics or amplification models save the training/evaluation partition and finite sample checks, labeling the result `empirical_candidate`. Lock phase rules and coefficients on development episodes; plot held-out envelope violations and exits without excluding failures. Contact, task phase, joint limits, and history fill are diagnostic strata, not automatic certificates. A local slope fit does not establish a uniform small-gain condition. Track nominal decay/amplification, corrected-coordinate forcing, predictor error dependence and reference/history approximation separately. Gains and information matrices should be logged with update times to distinguish poor conditioning, persistent holds and stale informative data.

Full-policy stability or task preservation is an optional separate extension with its own policy/observation sensitivity, reference relation, domain and task-success margin. It is not an assumption needed to state the primary execution bound. A tube below a visual threshold does not establish task success; supplied-command admissibility and continued domain containment also require evidence in the claimed use case.

## Timing and recovery protocol

1. Specify processor, OS/runtime, CPU/GPU placement, synchronization, and actual servo/policy/chunk periods. Distinguish simulated physical time from wall-clock runtime.
2. Time the deployed adapter scope: signal preparation, residual evaluation, estimate update, correction generation, and required transfer/synchronization. Record total sensor-to-command latency separately. Extra decoder calls used only for offline diagnostics must not be charged as required deployment calls or silently included in the deployed method.
3. Report sample count, median, p95, p99, maximum observed adapter latency, total sensor-to-command latency, and missed deadlines. Save raw samples. A maximum observed value is not a worst-case execution-time guarantee.
4. Fix tolerances before test execution. Identification recovery is the first post-onset time at which a stated scaled estimation error stays below a declared threshold for a declared duration. Report interface-mismatch recovery from actual `q` separately from physical execution recovery using `d_Mc(xi,xi^{nom})` under the same supplied commands. A small `q` alone does not measure state recovery during transients. If `xi^{nom}` is unavailable, report that limitation; task success remains a separate outcome.
5. Use elapsed seconds and report history-fill time, identification delay, application delay, and sustained execution recovery. For estimates that never meet the criterion, report the nonrecovery count and censoring/episode horizon. Give a recovery fraction by fixed deadlines; do not report a mean only over recoveries without the denominator.
6. Report healthy false-correction magnitudes and healthy task regressions on the same cohort. A held fault-informed vector is not an uninformed healthy monitor.

## Required output per representative experiment

- Original trace bundle and configuration ID.
- Error-budget figure with aligned onset/contact/saturation markers and units.
- Table of median/p95/max component norms, latency quantiles, projection/clipping fractions, and estimate ages; label fitted VLA envelopes **empirical** and identify the verification artifact for any separate analytical example.
- Phase-stratified prediction/increment errors, corrected-coordinate estimation errors, and envelope violations on held-out episodes. Keep pre-onset, history-fill, onset recovery, changed-fault, and removal intervals explicit.
- Paired success and recovery fractions with nonrecovery counts.
- Clear distinction between measured finite-sample diagnostics and uniformly verified bounds over an operating region. Fitting a slope or taking a sample maximum does not establish the paper's conditional certificate.

## Mechanism figure

Figure 1 is a schematic of the evaluated prediction-only information path.
It supplies no empirical record, rollout, success count, nominal replay, or
composite-branch evaluation. Populate the schemas above from actual run records,
not from diagram labels.
