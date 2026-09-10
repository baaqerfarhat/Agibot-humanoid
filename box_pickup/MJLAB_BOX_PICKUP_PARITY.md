# X2 box pickup — MJLab native task, parity checklist

The policy now learns inside MJLab. Isaac/HoloSoma is used for nothing except as the
origin of the OmniRetarget clip, and that dependency ends once the clip is converted.

## Robot interface: what changed vs the Isaac task

The MJLab walking/squat setup is the source of truth. HoloSoma disagreed with it on
most of the robot interface, which is the most likely reason Isaac policies looked fine
in Isaac and fell over everywhere else.

| Component | MJLab walking/squat (canonical) | HoloSoma box pickup | Real deployment |
| --- | --- | --- | --- |
| DOF order | 31, identical | 31, identical | 31, identical |
| control rate | 50 Hz (dt 0.005, decim 4) | 50 Hz | 50 Hz |
| PD gains | per-joint, 29/31 differ from HoloSoma | flat kp 20 / kd 2 on every joint | onboard PD, gains sent with the policy |
| action def | `q_des = default + scale * a` | same form | same form |
| action scale | per-joint MJLab constants, 29/31 differ | `0.25 * effort / kp` | from policy metadata |
| effort limits | per-joint, 14/31 differ | rescaled | actuator firmware |
| base frame | pelvis | pelvis | pelvis (reconstructed) |
| actor obs | ang vel, proj gravity, q, qd, prev action, command | no projected gravity; adds `motion_ref_ori_b` | IMU + encoders |
| quaternion | wxyz | xyzw | xyzw |
| torque path | MuJoCo implicit position servo | explicit PD, clipped | onboard PD |
| self-collision | off, feet-only colliders | on, full meshes | real |
| box | real dynamic body | kinematic reference only | real |

The box task inherits the canonical column verbatim: same X2 articulation, same gains,
same action scale, same `implicitfast` integrator, same 50 Hz.

## Actor observation parity

Every actor input is reproducible on hardware. Actor is 164-dim:

| Term | Source on the real robot |
| --- | --- |
| `base_ang_vel` | IMU gyro |
| `projected_gravity` | IMU attitude |
| `joint_pos`, `joint_vel` | encoders |
| `actions` | previous policy output, kept in the deploy loop |
| `command` | clock + clip lookup, see below |

The command vector is `[sin(phase), cos(phase), reference pelvis height,
reference joint positions relative to default, reference joint velocities]`. All of it
comes from the clip and a step counter, so deployment reads it out of the same npz.

**The actor never sees the box.** No box pose, no box velocity, no palm-relative box
vector. The box only enters through the critic (privileged) and through the rewards.
This is deliberate: the real cell has the box at a fixed surveyed pose, so the clip
already encodes where it is, and giving the actor ground-truth box XYZ would be a
capability deployment does not have.

## Train/deploy checklist

| Item | Status |
| --- | --- |
| joint ordering | identical to squat export, verified against the model |
| units | rad, rad/s, m, m/s |
| frames | pelvis-local ang vel and gravity, same as squat |
| quaternion | wxyz throughout MJLab; the export converts once, as squat does |
| control rate | 50 Hz train and deploy |
| action scale | MJLab per-joint constants, carried in the export metadata |
| PD gains | MJLab per-joint constants, carried in the export metadata |
| action clipping | rsl_rl `clip_actions`, same value as squat |
| previous action | fed back pre-clip, same as squat |
| observation history | none, same as squat |

## Contact logic

Three MuJoCo contact sensors, so the distinction is real collision-pair data rather than
the geometric guess Isaac needed:

- `feet_ground` — ankle-roll colliders against the floor geom
- `hand_ground` — palm colliders against the floor geom
- `hand_box` — palm colliders against the box geom

Hand-box is rewarded, hand-ground is penalised and, if sustained under load, terminates.
Palm colliders are 50 mm spheres at `(0.01, 0.0, -0.10)` in the wrist-roll frame, with
torsional friction so a grip can actually hold.

## Box

Real dynamic MuJoCo free body. Not welded, not teleported, no fake grasp.

- size 0.4712 x 0.4587 x 0.4079 m (from the OmniRetarget asset)
- mass 1.0 kg (measured on the real box)
- friction (1.0, 0.02, 0.001)
- spawned from the clip's frame-0 pose with roll/pitch stripped so it rests flat

## Reference validation

`sub3_largebox_003_squat_lift.npz`, 584 frames at 50 Hz (11.68 s):

- joint order identical to the model, 31/31
- root quaternion is wxyz; FK reproduces `body_pos_w` to 0.0001 mm
- body quaternions wxyz, 0.0000 deg error
- 0/584 frames violate joint limits
- pelvis 0.283 – 0.666 m, feet never below the floor
- box lifts 0.646 m, xy travel 0.186 m
- hands stay 0.212 m and 0.331 m above the floor, so the reference never uses hands as legs

Known caveat: at the grasp the clip still wants the palm spheres about 93 mm inside the
box surface, and the free-root inverse dynamics reads over the waist effort limit. The
policy cannot put a palm inside a rigid box, so it settles on the surface and pays a
small body-tracking cost; the lift is driven by the box reward, not by tracking. This is
the first thing to tighten in the retarget if v1 plateaus.
