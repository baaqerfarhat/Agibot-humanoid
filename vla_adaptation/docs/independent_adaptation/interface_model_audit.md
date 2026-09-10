# Actuation interfaces and nominal model audit

Date: 2026-09-08. Repository snapshot inspected: `687650f93ce6657d5fb8e3f8429a61970e634c30`.

This memo supports the new nominal servo model plus learned disturbance/effectiveness representation direction. It audits existing code and model files; it does **not** report an implementation or evaluation of that new method. No simulator was launched. The only numerical calculation below reprocesses an existing healthy ALOHA log.

## Findings that affect the design

1. **ALOHA and GR1 already expose joint position targets; Panda/LIBERO exposes Cartesian pose increments.** Access to Panda joint telemetry is not access to independent joint commands. A joint-space correction on Panda requires an explicit new controller interface and a fresh healthy comparison.
2. **A first-order position servo is a candidate reduced model, not a consequence of position control.** Velocity, internal controller targets, saturation, and contact can make position alone insufficient. The new method can use position tracking error without introducing a sliding variable, provided its nominal error dynamics and actual correction map satisfy the required assumptions.
3. **ALOHA's gripper has a concrete command-coordinate mismatch.** The installed normalization maps a normalized target of 1 to 0.058 m, whereas the positive finger actuator's control range ends at 0.057 m. The historical healthy right-gripper request exceeds that actuator range on 889/2,253 steps (39.46%). This is a plausible source of nonlinear residuals, not an attribution of the entire reported 93.8% residual share.
4. **Known action processing belongs in the nominal model; unknown injected faults do not.** The existing residuals already include the applied adaptation correction. Feeding privileged post-fault commands to the observer would change the estimation problem and can remove the command-offset signal being estimated.
5. **Task contact must remain distinguishable from nuisance disturbance.** A finger stopped by a grasped object can have persistent position error while performing exactly the intended task. Cancelling all such error is not an appropriate gripper objective.

## 1. What each current adapter can command and observe

| Environment | Current issued command | State used by the adaptation runner | Internal mechanism and relevant limitation |
|---|---|---|---|
| Panda / LIBERO | Six Cartesian pose-increment channels plus one gripper channel; adaptation adds to the first six, with a selectable subset | FIR residual from three translation increments and a geometrically computed three-component rotation increment; joint position/velocity and controller torque are optional diagnostics | `OSC_POSE` computes torques through state-dependent inertia/Jacobian maps and null-space posture control. Seven independent arm torques are not exposed by the seven-component policy action: its seventh component is the gripper. |
| ALOHA / transfer cube | Fourteen absolute targets: six arm joints and one normalized gripper per side | Fourteen measured positions in the corresponding policy coordinates; ordinary policy observation omits velocity | Arm targets feed position actuators. One gripper target expands into two opposing finger targets. Gripper coordinates have a separate length normalization and contact/limit behavior. |
| GR1 / RoboCasa tabletop | Twenty-nine absolute targets: left arm 7, right arm 7, left hand 6, right hand 6, waist 3 | The corresponding twenty-nine position coordinates; ordinary runner does not consume velocity or contact force | Pinned implementation uses joint-position arm control and Fourier hand position actuators. Each six-channel hand command drives eleven physical joints through copying. This task uses a fixed-base arms-and-waist embodiment, not walking control. |

Local command and observation anchors: [Panda policy input and correction](../../openpi/adaptive_law.py#L450), [Panda residual and telemetry](../../openpi/adaptive_law.py#L519), [ALOHA policy observation](../../openpi/aloha_adapt.py#L148), [ALOHA command execution](../../openpi/aloha_adapt.py#L218), [GR1 part dimensions and mapping](../../openpi/gr1_adapt.py#L29), and [GR1 action dictionaries](../../openpi/gr1_adapt.py#L75). The controller-source provenance is specified in Section 8.

### Panda: Cartesian target control is not a joint-torque interface

The installed OSC configuration clips normalized arm inputs to [-1, 1] and scales translation to ±0.05 m and rotation to ±0.5 rad. These are **target increments**, not guaranteed motion over one policy step. The distinction is already explicit in [error_signal.py](../../openpi/error_signal.py#L85). The controller uses pose and velocity error, operational-space inertia and Jacobians, gravity/bias compensation, and a null-space posture term; the robot then clips requested torques to actuator limits.

Consequently, changing configuration changes the local map from Cartesian correction to joint torque. That alone does not make an additive joint torque disturbance unmatched relative to a torque interface. It does mean that a translation-only correction, or even a six-dimensional Cartesian correction, need not span a seven-joint disturbance's required compensation at a particular configuration. Saturation further restricts the feasible set.

The gripper path has its own memory. GR00T and OFT explicitly binarize the LIBERO gripper command ([GR00T](../../openpi/groot_server.py#L44), [OFT](../../openpi/oft_server.py#L77)). Independently, the installed default `PandaGripper.format_action` uses the sign of its input to increment a bounded internal target; the two physical finger positions remain continuous. A model of finger motion should include that internal target or its reconstructed history, rather than treating the instantaneous sign command as an absolute finger position.

### ALOHA: direct joint targets, but mixed units and a coupled gripper

The runner uses 20 ms control steps and a seven-tap position FIR, covering the current command and six earlier commands. Despite the `fit_plant` docstring saying `(u, dq)`, its actual target is absolute position `qq[t]` ([fit](../../openpi/aloha_adapt.py#L171), [timing constants](../../openpi/aloha_adapt.py#L26)). Its observations combine arm radians and normalized finger displacement. Squaring and summing those coordinates does not yield a physical energy or force norm.

In the installed `gym_aloha` implementation, each normalized gripper target \(g\) becomes

\[
\ell(g)=0.01844+0.03956g\quad\text{metres},
\]

and the two actuator targets are \([\ell,-\ell]\). The measured normalized coordinate is reconstructed from the positive finger alone. The normalization itself neither clips nor binarizes the signal. However, the XML declares positive finger position-control limits [0.021, 0.057] m and corresponding negative finger limits. Thus the positive actuator's effective target range corresponds to

\[
g\in[0.0647118301,\;0.9747219414].
\]

Reprocessing [the historical healthy log](../../results/aloha/healthy_log.json) gives:

| Coordinate | Command minimum–maximum | Measured minimum–maximum | Requests above the mapped upper control limit |
|---|---:|---:|---:|
| Left gripper, index 6 | 0.274500–1.107641 | 0.114184–0.980158 | 317 / 2,253 = 14.07% |
| Right gripper, index 13 | 0.106346–1.175206 | 0.114479–0.975033 | 889 / 2,253 = 39.46% |

Neither coordinate has a below-range request in this log. These are **requested target** counts relative to the inspected actuator configuration, not measurements of per-substep force saturation. Soft physical constraints can allow small measured joint-limit violations; the state should not be described as hard-clipped to the command range. The current installed files were inspected directly, but this audit does not establish that every historical recording used byte-identical assets.

This yields a concrete mechanism candidate: a linear fit to unbounded normalized targets must represent a response that saturates, experiences friction, and changes on contact. In these coordinates a 1 mm finger error is approximately 0.02528, which can dominate squared arm errors numerically. The large right-gripper residual share therefore cannot be interpreted as its share of physical disturbance magnitude. Nor does it prove that target clipping explains all of that share: contact, friction, asymmetric task roles, model history, and data coverage remain alternatives. An FIR can represent a step; continuous versus binary commands is not the relevant impossibility argument.

The legacy law additionally scales its **estimation target** by the residual normalizer ([lines 67–80](../../openpi/aloha_adapt.py#L67)). If unrelated gripper error enters an all-channel norm, the resulting effect includes estimation bias, not only slower adaptation. Restricting or whitening residual channels can address that coupling, but does not itself identify a physical gripper model.

### GR1: verify actual actuator semantics rather than variable names

The repository pins RoboCasa commit `4840e671596f93ca03651524b9f72ffb1aadfeff` and robosuite `v1.5.1` ([SETUP.md](../../SETUP.md#L229)). In those sources, the tabletop wrapper selects a basic composite controller with absolute targets; arm control forms a PD acceleration-like request and multiplies it by the joint-space mass matrix before adding compensation. The Fourier hand formatter copies six continuous values into eleven outputs. The hand XML uses **position** actuators with joint-specific limits and gains.

The generic hand controller's variables `goal_qvel` and `vels` are therefore misleading if interpreted without the XML. They do not establish velocity control here. Similarly, the historical [GR1 normalization comment](../../openpi/gr1_adapt.py#L134) describing open/closed hands is not evidence of an adapter-level binary map. The observation converter selects six physical hand coordinates and reverses their order to match the robot convention. A new physical hand model must explicitly preserve that ordering and the six-to-eleven actuation map; it must not assume twenty-nine mutually independent scalar actuators.

## 2. Where the fault enters

| Fault family | Existing injection | Implication for the nominal model and compensation |
|---|---|---|
| Command offset | After our correction, before the environment/controller: `a_exec = a_corr + f_now` | Shares the issued-command coordinates before clipping and controller dynamics. Estimate against the known corrected command. Beyond limits, different offsets can produce the same effective actuator target. |
| Command motion scaling on ALOHA/GR1 | `q + gain * (target - q)` | Scales requested motion relative to current position. It is not simply a constant scalar multiplying an absolute target. See [ALOHA](../../openpi/aloha_adapt.py#L233) and [GR1](../../openpi/gr1_adapt.py#L122). |
| Panda joint torque bias | Adds to `qfrc_applied` at the physical joint before each environment step | Enters generalized force below OSC, independently of the Cartesian request. Its correction requires the actual torque authority of the Cartesian controller. |
| Panda friction/damping/gain/lock | Modifies model friction loss, damping, actuator gain, or joint range | Changes physical dynamics or constraints. The gain hook sets a gain parameter to the supplied value; the lock hook narrows the range around the current joint position. Neither should be silently redescribed as a command offset. |
| ALOHA physical arm torque | Wrapper adds torque on one of twelve arm joints; grippers excluded | Command/observation pathways are unchanged and the estimator does not receive torque truth. Static target-to-torque conversion is available locally before force/target limits; it does not guarantee task repair. |

Sources: [Panda offset and physical step](../../openpi/adaptive_law.py#L496), [Panda fault implementation and restoration](../../openpi/joint_fault.py#L44), [ALOHA torque mapping/wrapper](../../openpi/aloha_joint_fault.py#L14). The ALOHA follow-up checks the actual position-actuator gains and uses torque magnitude `0.02 * command_to_torque_gain` ([metadata](../../openpi/run_aloha_joint_followup.py#L64)). The arm gains are not uniform: the registered per-arm sequence is [800, 1600, 800, 10, 50, 20].

For an unclipped scalar position servo \(\tau=k_p(u-q)\), an additive external torque \(d\) has an algebraic compensating target shift \(c=-d/k_p\). This cancellation ceases to be automatic if the shifted target or resulting actuator force clips, the gain changes, another controller transforms the input, or the required state/contact trajectory is infeasible. A physical joint lock is a constraint change; it is not generally cancellable by adding bounded torque.

## 3. When the proposed first-order model is appropriate

To distinguish the descriptor matrix from the input map, write the proposed model as

\[
E\dot x=A_sx+F_u u+b+\Phi(\xi)z,
\]

where \(u\) is a precisely identified control-interface command. This corresponds to the user's \(B\dot x=Ax+a+\Phi z\) when \(E=B\) and \(a=F_u u+b\). If \(a\) instead means an unknown constant, the equation has omitted the time-varying policy input. Even with an invertible \(E\), position/input observations generally identify the products \(E^{-1}A_s\), \(E^{-1}F_u\), and \(E^{-1}\Phi\); they do not uniquely identify each physical matrix without additional structure or information.

A physical arm is initially better described by

\[
M(q)\ddot q+h(q,\dot q)
=\tau_{\rm servo}(q,\dot q,u,\eta)
+\tau_{\rm nuisance}+J_c(q)^\top f_c,
\]

where \(\eta\) contains controller/actuator memory and \(f_c\) represents contact. The generalized-force separation agrees with [MuJoCo's equations and data fields](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework). For revolute/prismatic robot joints, \([q,\dot q,\eta]\) is a natural candidate state; full floating-body quaternion coordinates need their proper tangent-space velocity convention.

An overdamped reduction such as

\[
D_{\rm eff}\dot q=-K_{\rm eff}q+K_{\rm eff}u+b+\Phi z
\]

can be useful in a declared operating region when velocity transients are fast relative to the modeled control step and omitted modes have bounded error. Position-target access alone does not establish those conditions. A decisive empirical test compares position-only, position/velocity, and finite-history models on **held-out entire healthy episodes**, including multi-step rollout error and residual dependence on velocity, command changes, limits, and contact phase. Persistent dependence indicates omitted state or an inappropriate mode, even if aggregate one-step RMSE is small.

The current [position reference fit](../../openpi/composite_observer.py#L192) already handles post-action alignment, disjoint episode splits, rank/conditioning, and one-step/rollout scores. Its [contraction check](../../openpi/composite_observer.py#L93) certifies a fixed fitted discrete matrix in a selected metric; its [qualification gate](../../openpi/composite_observer.py#L251) explicitly does not prove physical-plant or full VLA closed-loop stability. The reference is driven by the live policy's raw command ([reference_step](../../openpi/composite_observer.py#L124)); it is not the hypothetical command sequence a healthy robot would have received under identical policy randomness.

Using joint-position tracking error directly is compatible with the new direction. What remains necessary is the correct error dynamics and correction-to-state map. The existing [masked tracking map](../../openpi/composite_observer.py#L139) explicitly assumes coefficient coordinate \(j\) means an additive correction in command channel \(j\); it cannot convert a learned joint torque basis into Cartesian commands by naming the coordinates the same. Composite adaptation must use the actual map, sign, units, and timing.

## 4. Residuals from the known applied input

Keep distinct records for

\[
u_t^\pi\quad\longrightarrow\quad u_t^{\rm issued}=u_t^\pi+c_t
\quad\longrightarrow\quad u_t^{\rm eff}=h(u_t^{\rm issued},\eta_t),
\]

where \(h\) denotes **known** normalization, clipping, rate limiting, internal target integration, and other modeled action processing. Disturbance placement must be specified relative to this chain. A nominal transition residual is

\[
r_{t+1}=x_{t+1}-\widehat F_h(x_t,u_t^{\rm issued},\eta_t,m_t),
\]

or equivalently a prediction using \(u_t^{\rm eff}\) if the healthy processing \(h\) has already been evaluated. Here \(m_t\) is a declared mode/context, not an oracle fault label.

The current FIR histories already use `a_corr`, including our correction: [Panda](../../openpi/adaptive_law.py#L524), [ALOHA](../../openpi/aloha_adapt.py#L241), [GR1](../../openpi/gr1_adapt.py#L126). Replacing that input with raw policy output would make the model misinterpret its own compensation as disturbance. Conversely, `a_exec` includes the injected command fault. It is recorded for audit, but using that privileged signal in the predictor would explain away an upstream command offset and change the information available to the estimator. For a physical torque fault, measured actuator control and external injected force are likewise separate signals.

Correct alignment and action accounting remove avoidable bias; they do **not** guarantee a statistically unbiased residual. Conditional model error, errors in measured regressors, closed-loop correlations, changing contact, and unmodeled dynamics can violate a zero-mean noise assumption. Healthy bias should be examined conditionally on state, command, and mode, rather than subtracting one global mean and calling the remainder physical disturbance. A learned basis can model some of this structure, but its coefficients then represent the declared residual model, not uniquely identified external forces unless further identifiability conditions hold.

## 5. Saturation, lag, and observability

Control-target clipping, actuator-force clipping, and joint-level force clipping are different mechanisms; MuJoCo documents them separately and permits combinations ([force limits](https://mujoco.readthedocs.io/en/stable/modeling.html#force-limits)). A requested `data.ctrl` value is not by itself evidence of the internally clamped effective value. Joint limits are a further constraint, not another synonym for control clipping.

On a saturated interval the local derivative of an effective input with respect to its request can vanish. Several fault magnitudes then produce indistinguishable motion. A bounded estimate or anti-windup mechanism can limit consequences, but it cannot recover information absent from that interval. Excitation in an unsaturated region, actuator/current measurements, or additional structure is needed to distinguish loss of effectiveness from target/force saturation. Delay and internal target integration also require an augmented state or action history; estimating them as a constant additive disturbance generally confounds distinct mechanisms.

Current diagnostic access is useful but incomplete:

* Panda logs joint velocity and `robot.torques` ([telemetry](../../openpi/adaptive_law.py#L551)). In the installed robot implementation, this is a clipped **commanded** torque. It is not a measurement of total joint force including external torque/contact, nor a hardware torque-sensor reading.
* The ALOHA physical wrapper logs `qpos`, `qvel`, `qfrc_applied`, actuator force, and raw actuator control ([PhysicalTelemetry](../../openpi/run_aloha_joint_followup.py#L85)). Its rail checks are evaluated after an environment step, so they can miss a force limit reached only during an internal physics substep. Its arm-coordinate comparison also should not be reused for grippers without their length conversion.
* GR1's ordinary runner logs position/command trajectories. Available simulator velocity and force information is not automatically part of the declared online observation interface.

A future collector should distinguish policy output, issued correction, converted target, effective limited target, controller goal/memory, commanded actuator force, total joint actuator force, external force, and contact state. It should record actual control timestamps and summarize substep limit occupancy where available. These are proposed measurements; this memo adds none to the runners.

## 6. Preserve useful contact while adapting to nuisance forces

The installed ALOHA transfer reward explicitly depends on finger–box contact and lifting the box off the table. Contact is therefore part of the desired task dynamics. During a grasp, the object can prevent a finger from reaching its no-object target while the servo produces the necessary holding force. A method that interprets this healthy position residual as unwanted force can open the grasp or increase squeezing, depending on its sign and allocation.

A useful nominal gripper model should distinguish free motion, closing, first contact, sustained grasp/slip, and release. Its state should include physical finger position/velocity and known target memory; its context can include available contact/force/current evidence and task phase. Paired fingers or a multi-joint hand also require an explicit actuation and observation map. Healthy contact examples can inform the expected load response; a free-motion model alone cannot establish that every contact residual is a fault.

Contact forces and nuisance torques can occupy the same generalized-force directions. Position observations alone need not distinguish them, even with a perfectly learned basis. An orthogonal projection or larger representation does not automatically solve that ambiguity. The practical choices are to obtain discriminating observations, restrict when coefficient learning is trusted, or define a force/impedance/contact objective consistent with task intent. For example, uncertainty or contact-transition gates can limit adaptation during impacts; they require validation and should not be described as proof that desired contact is preserved.

## 7. What model files provide, and what changing the interface entails

URDF/MJCF can supply kinematics, coordinate conventions, inertial parameters, declared joint/actuator limits, and whatever friction, transmission, mimic, and contact parameters are actually specified. They are useful structural priors. They do not identify the real device's closed-loop gains, delay, backlash, motor efficiency, sensor offsets, or contact properties merely by being present. The simulator's compiled model plus controller implementation is the relevant nominal simulation plant; a physical robot still needs an identified operating envelope. Inspect resolved defaults rather than interpreting a missing XML attribute as “unlimited” or “zero.” See [MuJoCo's model/default semantics](https://mujoco.readthedocs.io/en/stable/modeling.html#default-settings).

For ALOHA, modeling the existing joint-position interface first is the least disruptive route. The arm target-to-torque gains can seed a local actuation map, with declared saturation/contact qualifications. For GR1, preserve the exact arm controller and six-to-eleven hand map and attest the live installed version before new experiments. For Panda, independent joint-space correction would require either a torque hook at a declared point before torque limiting, or a joint-position controller together with a conversion of the VLA's Cartesian target and a specified redundant posture. Both alter the control interface; healthy performance must be revalidated before comparing repairs. Merely reading `robot._joint_positions` does not implement either route.

The repository's hardware analysis, such as [hw_matched_position.py](../../hardware/hw_matched_position.py#L5), reads recorded X2 logs. It does not establish a live Panda/ALOHA/GR1 actuator API. Hardware availability and simulator access should remain separate claims.

## 8. Evidence provenance and limits

### Installed files inspected read-only on `lambda`

ALOHA prefix: `/data/fxxie/vla/envs/aloha/lib/python3.10/site-packages/gym_aloha/`.

| Relative file and lines | Evidence | SHA-256 where recorded |
|---|---|---|
| `constants.py:71–75,94–105` | Finger endpoint normalization and inverse | `76880e66502671296010e16e58c66fa7a0eba668ad2abf60adcd3ef29df32bd6` |
| `tasks/sim.py:38–80,125–148` | 14→16 target mapping, observed finger selection, contact reward | `842a8c8d03f425830412e0606a6124ea56f289b2d4a9cb2befc3df8f031caafe` |
| `assets/bimanual_viperx_transfer_cube.xml:16–35` | Arm and finger position actuators, gains and limits | `f29eda25c5ac8df9388efa743bcbeeea8b3f49cb7126962fb510879f60381317` |
| `assets/vx300s_left.xml:7–50` | Joint friction, finger slides/limits/contact geometries | Inspected, no separate hash retained |
| `env.py:138–147,173–187` | Position-only selected state; action forwarding and success | Inspected, no separate hash retained |

Panda prefix: `/data/fxxie/vla/envs/libero/lib/python3.8/site-packages/robosuite/` (installed version 1.4.1, also recorded in [confirmation environment metadata](../../results/joint_followup/confirmation_plan/environment.json)).

| Relative file and lines | Evidence | SHA-256 where recorded |
|---|---|---|
| `controllers/config/osc_pose.json:2–16` | Clipping/scales, gains, pose-delta mode | Inspected, no separate hash retained |
| `controllers/osc.py:234–265,314–351` | Goal update, PD wrench, inertia/Jacobian map, null-space torque | `cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb` |
| `controllers/base_controller.py:104–123` | Input clipping and affine scaling | Inspected, no separate hash retained |
| `robots/single_arm.py:245–261` | Goal update, torque clipping and actuator write | Inspected, no separate hash retained |
| `models/grippers/panda_gripper.py:43–62` | Sign-driven bounded internal target, speed 0.01 | `a3760a8fc4599fa6f79914304b82466f3c1bda1a945943ca693bb57f00dc59eb` |
| `models/assets/grippers/panda_gripper.xml:9–11` | Position actuators and finger force/control bounds | Inspected, no separate hash retained |

The installed LIBERO wrapper selects `OSC_POSE`, control frequency 20 Hz, and the default `PandaGripper`: `/data/fxxie/vla/openpi/third_party/libero/libero/libero/envs/env_wrapper.py:17–27` and `envs/robots/mounted_panda.py:28`. No simulation was needed to read these files.

### GR1 pinned source, not a live runtime attestation

The documented GR1 installation was not located for direct byte inspection in this audit. These are the exact pinned public source versions from [SETUP.md](../../SETUP.md#L229):

* [RoboCasa wrapper, lines 95–106](https://github.com/robocasa/robocasa-gr1-tabletop-tasks/blob/4840e671596f93ca03651524b9f72ffb1aadfeff/robocasa/utils/gym_utils/gymnasium_basic.py#L95): controller selection and absolute control.
* [GR1 arms-and-waist model, lines 37–42](https://github.com/robocasa/robocasa-gr1-tabletop-tasks/blob/4840e671596f93ca03651524b9f72ffb1aadfeff/robocasa/models/robots/manipulators/gr1_robot.py#L37): removed leg/head actuation and free joint.
* [Observation converter, lines 109–127](https://github.com/robocasa/robocasa-gr1-tabletop-tasks/blob/4840e671596f93ca03651524b9f72ffb1aadfeff/robocasa/models/robots/__init__.py#L109): physical hand coordinate selection and reversal.
* [robosuite v1.5.1 GR1 configuration](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/controllers/config/robots/default_gr1.json#L28), [joint-position controller](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/controllers/parts/generic/joint_pos.py#L225), and [generic gripper controller](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/controllers/parts/gripper/simple_grip.py#L141): absolute goals, arm PD/mass-matrix control, and unscaled hand output.
* [Fourier hand command formatter](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/models/grippers/fourier_hands.py#L21) and [left-hand actuator XML](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.1/robosuite/models/assets/grippers/fourier_left_hand.xml#L180): six-to-eleven copying and position-actuator semantics.

### Recomputed historical log statistic

Source: `results/aloha/healthy_log.json`, SHA-256 `d4dd02228bdef5412ea8abcdd2370fedc5a52d266d703d9e010b31368a5e5e4f`. Concatenate each episode's `u` and `q`, select columns 6 and 13, and compare requested `u` against `(0.057-0.01844)/(0.058-0.01844)`. All 2,253 steps are used; no episode or time-window selection is involved. This audit did not recompute the 93.8% FIR residual decomposition or claim that the limit calculation causally explains it.
