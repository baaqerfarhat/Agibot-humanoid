"""X2 box pickup, clean retrain + a grasp that does not use the floor.

Supersedes ``box_clean.py``. That run fixed the waist, the feet and the action
bound, but a rollout at iteration 9000 showed the policy had found a different
exploit: it plants both HANDS on the ground during the grasp and again during
the set-down, and pushes off them like a third and fourth foot.

The diagnosis was that this is NOT a bad reference. Measuring the lowest point
of the hand mesh (114k vertices, rotated by the link quaternion) against the
floor plane:

    reference   left  +0.193 m   right +0.216 m   -- minimum over all 584 frames
    iter 9000   left  -0.001 m   right -0.002 m   -- 50 / 39 frames within 2 cm

The retargeted clip is clean: it never brings a hand within 19 cm of the floor,
and it grasps the box on the upper side faces, which is already the 0.20-0.25 m
palm height that a human would use. Nor is the squat to blame -- reference
pelvis bottoms out at 0.636 m and the policy at 0.600 m, a 3.6 cm difference.
The hands alone go 20 cm lower than they should while the pelvis stays put.

So the reference was left untouched and the objective was fixed instead. Three
things had made hands-on-the-floor free:

  1. ``undesired_contacts`` excludes every wrist link (so that gripping the box
     is not punished), which also legalised gripping the FLOOR.
  2. ``motion_relative_body_position_error_exp`` averages over 14 bodies, so two
     hands 0.17 m off-reference cost about 5% of that term.
  3. ``hands_to_object_distance_exp`` saturates for any hand within 0.25 m of the
     box centre -- including a hand resting on the floor beside the box.

This preset otherwise keeps everything box_clean.py established:

  1. waist tracking      -- the reference bend is +18 deg but nothing in the
                            original objective constrained waist_pitch/yaw, so
                            the actor drove them the wrong way (IRL pitch
                            collapse, and a waist-yaw side-carry twist).
  2. planted feet        -- nothing penalized skating or lifting a foot, which
                            became a real-world pivot/step under load.
  3. bounded actions     -- ``action_clip_value = 100`` let the actor park
                            ankle_roll and the wrists far outside their
                            mechanical range to demand torque that only
                            simulated contact could absorb.
  4. hip-hinge, not squat -- exp + L2 on hip_pitch/knee, after an exp-only term
                            at sigma 0.20 turned out to be exp(-16), i.e. dead.
  5. reachable objective  -- the hands are excluded from relative-body
                            ORIENTATION tracking, because the reference asks for
                            +84 deg of wrist roll that the bounded, 4.8 Nm wrist
                            can never produce.

Everything else (motion-tracking weights, object weights, action-rate, init-pose
noise, termination thresholds, domain randomization, PPO recipe) is the ORIGINAL
configuration, NOT the v11..v33 accumulation. The one exception is the
hardware-transfer randomization block (PD-gain spread, control latency, encoder
bias): deployment demonstrably has those effects, and the guiding principle here
is that training must represent what deployment does.

Reference motion: ``sub3_largebox_003_mj_w_obj.npz`` (584 frames @ 50 Hz,
11.68 s) -- the complete in-place clip: settle -> bend -> grasp -> lift -> hold
-> set-down -> stand. The feet never leave the ground in this reference (max
0.1 mm), which is what makes the planted-feet penalties coherent; the original
6.5 s clip's lateral carry-walk is handled by the separate payload walking
policy, not by this task.
"""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values import algo, robot, simulator, terrain
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos
from holosoma.config_values.wbt.g1.randomization import (
    base_reset_terms,
    base_setup_terms,
    base_step_terms,
)
from holosoma.config_values.wbt.x2 import curriculum, observation
from holosoma.config_values.wbt.x2.randomization import x2_robot_material_dr

# ---------------------------------------------------------------------------
# Command / reference motion  (ORIGINAL values)
# ---------------------------------------------------------------------------

# Original init-pose noise. The later configs escalated dof_pos 0.1 -> 0.5 rad
# and root xy 0.05 -> 0.12 m to absorb operator stance variation; that is
# unrelated to waist/feet/actions, so it is not carried over.
box_grasp_init_pose = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

# When the REFERENCE has a foot planted, for the gated foot penalties below. The
# ankle_roll_link rides 68 mm above its contact spheres, so a flat foot sits at
# ~0.068 m; 0.10 m admits it with margin without admitting a swing. Height alone
# cannot decide, though -- the swings in this clip only clear 2-6 cm -- so a foot
# has to be slow as well as low to count as planted.
X2_STANCE_HEIGHT = 0.10
X2_STANCE_SPEED = 0.20

X2_BODY_NAMES_TO_TRACK = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]

# Original sampling: adaptive sampler with a 30% uniform floor and 15% of
# episodes from t=0. Single motion file, not the v29 multispeed directory.
box_grasp_motion_config = MotionConfig(
    # v6: the full OmniRetarget clip WITH the walk, repaired for X2 feasibility --
    # stances pinned and levelled, pelvis placed where the legs can actually reach,
    # CoM held over the feet, the clock stretched 19% where the ZMP left the support
    # polygon, the grip raised from 7 to 20.5 cm so the palms never load the floor,
    # and a 1.2 s rise appended so the episode ends standing rather than folded over
    # at 22 deg. The box pickup spot and its resting pose are untouched.
    motion_file="holosoma/data/motions/x2_31dof/whole_body_tracking/sub3_largebox_003_walk_feasible.npz",
    body_names_to_track=X2_BODY_NAMES_TO_TRACK,
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    adaptive_uniform_ratio=0.3,
    # v2: 0.15 -> 0.35. Deployment ALWAYS starts at t=0 from a standing pose, and
    # the standing-start bend is the phase that failed on the robot; mid-motion
    # resets spawn the robot already bent with the box at the reference, so they
    # never exercise it. Kept well below the later configs' 0.70 so the hold and
    # set-down still get most of the sampling.
    start_at_timestep_zero_prob=0.35,
    noise_to_initial_pose=box_grasp_init_pose,
)

x2_31dof_box_grasp_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": box_grasp_motion_config},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand")
    },
)

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_X2_UNDESIRED_CONTACTS_REGEX = (
    "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    "(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)"
    "(?!left_wrist_pitch_link$)(?!right_wrist_pitch_link$)"
    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)

x2_31dof_box_grasp_reward = RewardManagerCfg(
    terms={
        # ---- ORIGINAL motion tracking, unchanged -------------------------
        # v2: 0.5 -> 1.0. This is the only term that sees a uniform drop of the
        # whole body (the relative-body terms are measured against this same ref
        # frame, so they cancel it out). At 0.5 the iter-2500 rollout squatted
        # 0.23 m below the reference through the set-down for free.
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        # v2: hands excluded from ORIENTATION tracking only. The reference asks
        # for +84 deg of right_wrist_roll; with the sim2real action bound the
        # wrist can only reach +-13.7 deg (scale 0.06 rad x 4), and the 4.8 Nm
        # wrist could not hold that angle under the box load on hardware either.
        # Leaving it in made a permanently unreachable target the policy paid for
        # every step, and it paid by contorting shoulder_yaw (+49 vs -13 deg) and
        # elbow (-83 vs -35 deg) to compensate. Hand POSITION is still tracked
        # below, which is what actually guides the grasp.
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={
                "sigma": 0.4,
                "exclude_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
            },
            weight=1.0,
        ),
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # v4: restored from v33. Every other tracking term above is measured in
        # Cartesian space over 14 bodies, which leaves the 31-dim joint vector
        # itself only weakly pinned -- the arm can reach the same hand pose
        # through many joint paths, and nothing prefers the smooth one. This is
        # the only term that tracks joint angles directly, so it is a second
        # brake on chatter alongside action_rate_l2, and dropping it was part of
        # why the from-scratch runs shook.
        "motion_dof_pos_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_error_exp",
            params={"sigma": 0.25},
            weight=1.0,
        ),
        # ---- CARRIED OVER (1/3): waist tracking ---------------------------
        # The 14 tracked bodies pin the torso pose but barely constrain how the
        # waist chain gets there, so waist_pitch/yaw free-ride. Each exp term
        # returns [0, 1]; sigma sets where the gradient lives. At sigma 0.20 a
        # 9 deg pitch error still returns ~0.57 and a 15 deg error ~0.18, so
        # there is real gradient across the whole band the policy drifted
        # through -- unlike the original v33 sigma 0.10, which was ~0 outside
        # 6 deg. Weights are the low end of what v33 converged on (pitch 4 ->
        # 3, yaw 2 -> 1.5): waist is then 29% of the positive budget (4.5 of
        # 15.5) instead of v33's 34%, enough to steer without outvoting the
        # body/object tracking that defines the task.
        "motion_waist_pitch_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={"sigma": 0.20, "joint_names": ["waist_pitch_joint"]},
            weight=3.0,
        ),
        "motion_waist_yaw_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={"sigma": 0.12, "joint_names": ["waist_yaw_joint"]},
            weight=1.5,
        ),
        # v4: restored from v33. Pitch and yaw were carried over but roll was
        # not, and roll is the one the reference holds near zero throughout --
        # so it was the cheapest axis to fidget on. It came out of the iter-9000
        # rollout at |da| 0.276/step against v33's 0.164, third worst of the 31.
        "motion_waist_roll_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={"sigma": 0.12, "joint_names": ["waist_roll_joint"]},
            weight=1.5,
        ),
        # v2: squat-vs-hinge. The reference lifts by hinging at the hips with
        # near-straight knees (knee +11 deg, pelvis never below 0.636 m); the
        # iter-2500 policy instead squatted (knee +80 deg, pelvis 0.43 m). The
        # cartesian body terms barely see the difference because every tracked
        # body moves down together. These four joints are the difference, they
        # are the strongest actuators on the robot (120 Nm), and unlike the wrist
        # target above the reference angles are fully reachable.
        #
        # v3: sigma 0.20 -> 0.35. At sigma 0.20 this term was DEAD, not weak: it
        # returns exp(-mean(err^2)/sigma^2), and the measured 46-53 deg knee error
        # gives mean(err^2) ~ 0.64 rad^2, i.e. exp(-16) ~ 1e-7 with a gradient to
        # match. It contributed literally nothing for 3000 iterations. 0.35 keeps
        # it near zero at the current error but alive from ~25 deg inward, so it
        # takes over for fine tracking once the L2 below has done the pulling.
        "motion_leg_pitch_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={
                "sigma": 0.35,
                "joint_names": [
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                ],
            },
            weight=2.0,
        ),
        # v3: the far-field companion the exp term was missing -- the same
        # exp-plus-L2 pairing the waist joints already use, and for the same
        # reason: an exp term saturates to zero exactly in the large-error regime
        # that has to be escaped, so something with a constant gradient has to do
        # the escaping. Mean |error| in radians over the four joints, currently
        # ~0.80 rad, so this starts as about -2.0 against an ~18 positive budget
        # and shrinks to nothing as the hinge is recovered.
        "motion_leg_pitch_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_l2",
            params={
                "joint_names": [
                    "left_hip_pitch_joint",
                    "right_hip_pitch_joint",
                    "left_knee_joint",
                    "right_knee_joint",
                ],
            },
            weight=-2.5,
        ),
        # Linear far-field pull. The exp terms saturate to 0 once the error is
        # large, which is exactly the wrong-sign regime that has to be escaped;
        # these keep a constant gradient there. Mean |error| in radians, so a
        # 0.3 rad pitch error costs 0.9 -- meaningful against a ~15.5 positive
        # budget without the -6.0 that v33 needed to break an established
        # plateau (there is no plateau to break when training from scratch).
        "motion_waist_pitch_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_l2",
            params={"joint_names": ["waist_pitch_joint"]},
            weight=-3.0,
        ),
        "motion_waist_yaw_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_l2",
            params={"joint_names": ["waist_yaw_joint"]},
            weight=-2.0,
        ),
        # ---- CARRIED OVER (2/3): planted feet, now gated on the reference --
        # foot_slip sums |v_xy| over feet in contact; contact_loss counts
        # unloaded feet (0/1/2). They are kept in proportion (-4 / -3): lifting
        # a foot zeroes foot_slip, so if contact_loss were much cheaper the
        # policy would escape slip by stepping. 20 N demands the foot be loaded,
        # not just grazing the floor.
        #
        # v7: both are gated on the reference's own stance. They were written
        # for an in-place clip where the reference never moves a foot, so
        # charging for any foot motion cost nothing correct. This clip walks in,
        # so ungated they bill the policy for the steps the reference is asking
        # for -- and, far worse, for the catch step that is the only thing that
        # saves it once the CoM leaves the feet. v6 shuffled instead of stepping
        # and toppled. Gated, they still forbid skating a planted foot and
        # hovering one the reference has planted; they just stop pricing the
        # steps the clip contains and the recovery the robot needs.
        "foot_slip": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_foot_slip",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_force_threshold": 1.0,
                "reference_stance_only": True,
                "stance_height": X2_STANCE_HEIGHT,
                "stance_speed": X2_STANCE_SPEED,
            },
            weight=-4.0,
        ),
        "feet_contact_loss": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_feet_contact_loss",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_force_threshold": 20.0,
                "reference_stance_only": True,
                "stance_height": X2_STANCE_HEIGHT,
                "stance_speed": X2_STANCE_SPEED,
            },
            weight=-3.0,
        ),
        # ---- NEW: keep the CoM inside the feet ----------------------------
        # v6 fell the same way every rollout: the CoM crossed outside the
        # support polygon at the top of the lift and the robot toppled 1.1 s
        # later, 82% of the escape lateral. Nothing in the reward could see it.
        # Tracking error only reports the fall once it is already unrecoverable
        # -- statically, once the CoM is out, no ankle torque brings it back --
        # so the policy never had a gradient toward staying balanced, only
        # toward matching poses. This charges for the margin itself, from 4 cm
        # of clearance inward, while there is still a polygon to steer into.
        # Weight is deliberately below the tracking terms: it should break ties
        # between equally on-reference poses, not buy a crouch that ignores the
        # clip.
        "com_support_margin": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ComSupportMarginPenalty",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "margin": 0.04,
                "cap": 0.20,
                "contact_force_threshold": 1.0,
                # A foot has to be on the ground, not merely reporting force, before
                # it counts towards the polygon -- see ComSupportMarginPenalty.
                "contact_height": 0.02,
                "directions": 16,
            },
            # Reward terms are accumulated as weight * value * dt, so -2.0 bought a
            # penalty of -0.026 an episode against a mean reward of 65 -- unmeasurable.
            # At -20 the term stays near -0.04 for a policy standing where this one
            # already stands, and a CoM parked on the edge of the feet costs about a
            # tenth of everything else the step earns. The cap is only reached 160 mm
            # outside, which double support never gets near.
            weight=-20.0,
        ),
        # v31's feet_anchor / foot_not_flat / feet_edge_contact are NOT included.
        # They were three successive attempts to stop ankle-roll edge-standing
        # through the reward, which the bounded ankle_roll action now prevents
        # at the source.
        # ---- ORIGINAL regularization, unchanged ---------------------------
        # v4: -0.1 -> -1.0. THIS IS THE JITTER FIX, and -0.1 was a mistake of
        # method rather than of judgement: it is the ORIGINAL value, and the
        # later escalation to -1.0 was filed under "not waist, feet or action
        # clip, so do not carry it over". It is a smoothness fix and it belonged
        # in the carry-over set from the start.
        #
        # Measured on the robot, leg targets, mean |delta target| per 50 Hz step
        # and how often that delta changes sign:
        #     v33          @ -1.0      12-19 mrad     15-25%
        #     clean        @ -0.1      45-159 mrad    39-64%
        #     clean_grasp  @ -0.1      205 mrad       67%
        # Two thirds of steps reversing direction is bang-bang chatter at the
        # Nyquist frequency, and it made the policy unrunnable on hardware. The
        # same gap is visible in sim (|da| 0.209 vs v33's 0.095), so this is a
        # property of the policy, not of the deployment path -- which is why it
        # has to be fixed here and not with a filter on the robot.
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        # v5: from mtaheriee's lift-feasibility analysis of why the pickup
        # completes in Isaac and topples on the robot. Verified independently on
        # our own iter-9000 rollout, peak demand vs each joint's limit: hip_pitch
        # 0.87x and knee 0.57x never saturate, while ankle_roll hits 5.96x and is
        # saturated for 84-99% of the episode and ankle_pitch 2.97x for 18-60%.
        # clip_torques then throws the excess away and sim contact absorbs the
        # shortfall, so in Isaac it is free; on hardware there is no lateral ankle
        # authority left and the feet roll onto their edges.
        #
        # Two deliberate departures from his config:
        #
        # require_lifted_z is 0.0 here, not 0.30. His gate exists so the policy
        # cannot dodge the penalty by never lifting, but 60% of OUR ankle
        # saturation happens before the box clears 0.30 m -- it is the squat and
        # the grasp that pin the ankles, and that is the phase where the feet roll.
        # Gating it away would discard most of the signal. The degenerate optimum
        # is not competitive for us anyway: object tracking alone is worth +7.0
        # (3.0 + 2.5 + 1.5) against roughly -0.4/step here.
        #
        # position_term_only isolates the sustained kp*(target-q) command from the
        # kd*qd damping spikes, which are transient and legitimate. This term is the
        # backstop, not the ankle_roll fix -- that is the action_scale cap on the
        # control block, after three weights of this penalty failed to price the
        # exploit out. With the cap in place it still earns its keep on waist_pitch
        # (1.82x demand / 11% of the episode saturated on the v31 baseline).
        "penalty_joint_torque_saturation": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_joint_torque_saturation",
            params={
                "joints": "waist_pitch,ankle_pitch,ankle_roll",
                "ramp_steps": 24_000,
                "require_lifted_z": 0.0,
                "position_term_only": True,
            },
            weight=-0.10,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": _X2_UNDESIRED_CONTACTS_REGEX,
            },
            weight=-0.1,
        ),
        # ---- ORIGINAL object tracking, unchanged --------------------------
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=3.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.5,
        ),
        "hands_to_object_distance_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hands_to_object_distance_exp",
            params={
                "sigma": 0.25,
                "hand_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
                "surface_offset": 0.25,
            },
            weight=1.5,
        ),
        # ---- NEW (1/3): where the palms should actually be ------------------
        # Tracks each hand's offset FROM THE BOX against the same offset in the
        # reference, so it states the grasp geometry positively instead of only
        # forbidding the floor. Anchored to the SIMULATED box because the box is
        # reset with +-5 cm of noise, which a world-frame target would ignore.
        #
        # This is the term that makes the two penalties below survivable: without
        # somewhere for the hands to go, "not on the floor" is satisfied just as
        # well by holding them in the air.
        #
        # sigma 0.20 is set from the measured failure, not from the tolerance we
        # would like. err is summed over both hands, so the observed 0.17-0.20 m
        # drop is err ~ 0.07 and the term returns 0.17 -- small but with real
        # gradient. At 5 cm per hand it returns 0.88, at 2 cm 0.98, so it still
        # discriminates a good grasp from a sloppy one. A tighter sigma 0.10
        # would return 0.001 at the current error, which is the dead-term trap
        # the leg-pitch reward already fell into once.
        "hands_to_object_relative_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hands_to_object_relative_position_error_exp",
            params={
                "sigma": 0.20,
                "hand_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
            },
            weight=2.5,
        ),
        # ---- NEW (2/3): the hands are not feet ------------------------------
        # Counts hands (0/1/2) that carry contact force while their lowest mesh
        # point is under 6 cm. Force AND height, because the net-force sensor
        # cannot tell the box from the floor: the reference never brings a hand
        # below 0.193 m, so 0.06 m leaves 13 cm of margin and cannot fire on a
        # correct grasp, while the 5 N floor means a hand may still swing low
        # through the air -- only load-bearing contact is charged.
        #
        # -5.0 puts both planted hands at -10/step against an ~20.5 positive
        # budget. The comparison that matters is not to zero but to what the
        # support buys: propping up the torso improves nearly every tracking
        # term at once, so the penalty has to be worth more than that combined
        # improvement, and half the positive budget is.
        "hand_ground_contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_hand_ground_contact",
            params={
                "hand_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
                "contact_force_threshold": 5.0,
                "ground_height": 0.06,
            },
            weight=-5.0,
        ),
        # ---- NEW (3/3): a slope to follow on the way down -------------------
        # The contact penalty is a step function: zero until the hand is already
        # down, so by itself it tells the policy nothing while it is descending.
        # This is a hinge in metres below 0.12 m, i.e. -0.24 m for two hands flat
        # on the floor, or -2.4/step at this weight. Still 7 cm clear of the
        # reference minimum, so tracking the reference costs exactly zero.
        "hand_floor_clearance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_hand_floor_clearance",
            params={
                "hand_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
                "min_clearance": 0.12,
            },
            weight=-10.0,
        ),
    }
)

# ---------------------------------------------------------------------------
# Termination  (ORIGINAL thresholds)
# ---------------------------------------------------------------------------

x2_31dof_box_grasp_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                # v2: 0.25 -> 0.40. Only the two hands are on this list, and the
                # bounded wrist cannot reproduce the reference hand pose, so 0.25 m
                # killed episodes for a grasp that was actually holding the box.
                "bad_motion_body_pos_threshold": 0.40,
                "body_names_to_track": X2_BODY_NAMES_TO_TRACK,
                # Ankles dropped from the kill list (the one feet-related change
                # kept here): ending an episode over ankle drift rewards thrashing
                # the legs to save the episode. The foot_slip / feet_contact_loss
                # penalties price that behaviour instead of terminating on it.
                "bad_motion_body_pos_body_names": [
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
                # v2: measured on the iter-2500 rollout, which lifts, holds and
                # sets the box down successfully. Peak box POSITION error was
                # 0.200 m against a 0.25 m bound, and box ORIENTATION error
                # averaged 0.746 rad with 46% of frames past the 0.8 rad bound
                # (the bounded wrist grips the box rotated). Training was
                # terminating a working grasp, which is what pinned mean episode
                # length at 1.95-1.99 s for 800 iterations. These bounds still
                # end the episode on a genuine drop: the box falls 0.65 m.
                "bad_object_pos_threshold": 0.5,
                "bad_object_ori_threshold": 2.0,
            },
        ),
        # Backstop only -- the two hand penalties carry the gradient; this makes
        # hand-supported crouching unreachable rather than merely expensive.
        # Deliberately blunt on both axes so exploration stays intact: 30 N is
        # about 3 kg through the palm (a graze or a numerical impulse never gets
        # there, propping up the torso does), and 25 consecutive steps at 50 Hz
        # is 0.5 s against the ~1 s the measured exploit held. The counter clears
        # the instant the load lifts, so only sustained support accumulates.
        # This caution is not theoretical: an over-tight bound on this task once
        # pinned mean episode length at 2 s for a thousand iterations.
        "hand_ground_support": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:HandGroundSupport",
            params={
                "hand_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
                "contact_force_threshold": 30.0,
                "ground_height": 0.06,
                "max_consecutive_steps": 25,
            },
        ),
    }
)

# ---------------------------------------------------------------------------
# Randomization  (ORIGINAL + hardware-transfer block)
# ---------------------------------------------------------------------------

# Original object DR. Mass floor/ceiling back to 0.3-1.5 kg (v27 widened the
# ceiling to 2.0 for denser demo boxes).
x2_box_grasp_object_dr = {
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [0.8, 1.4],
            "dynamic_friction_range": [0.7, 1.2],
            "restitution_range": [0.0, 0.2],
        },
    ),
    # Added to the URDF's 0.1 kg, so this is a 0.6-1.4 kg box. v7: was [0.3, 1.5],
    # i.e. 0.4-1.6 kg, which straddled rather than bracketed the real ~1 kg box and
    # spent a third of its envs above the mass the waist could lift at all under the
    # old posture. Centred on the real box, +/-40% for robustness.
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_startup",
        params={"mass_distribution_params": [0.5, 1.3]},
    ),
    "randomize_object_rigid_body_inertia_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_inertia_startup",
        params={"inertia_distribution_params_dict": {"Ixx": [0.5, 1.5]}},
    ),
}

# The single non-original block that is retained. The robot runs a 50 Hz ROS
# loop with real latency, servo gains that differ from the config, and encoder
# offsets. A policy trained without those is mismatched to deployment by
# construction, which is the failure mode this whole retrain exists to avoid.
# Values are the moderate v27 settings, not the widened v29 ones.
x2_box_grasp_hardware_dr = {
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            "kp_range": [0.70, 1.30],
            "kd_range": [0.70, 1.30],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": True,
            "enable_rfi_lim": False,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            "ctrl_delay_step_range": [0, 3],  # 0-60 ms at 50 Hz
            "enabled": True,
        },
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.05, 0.05],
            "enabled": True,
        },
    ),
}

x2_31dof_box_grasp_randomization = RandomizationManagerCfg(
    setup_terms={
        **base_setup_terms,
        **x2_robot_material_dr,
        **x2_box_grasp_object_dr,
        **x2_box_grasp_hardware_dr,
    },
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

_X2_WBT_INIT_HEIGHT = 0.9

# ---- v4: the training-side action clip is REMOVED --------------------------
# Not because the bound was wrong in magnitude, but because of what clipping
# does to an action that crosses zero. In the iter-9000 rollout these joints sit
# on the +-4 boundary for 78-94% of the episode. Saturation by itself is
# survivable: v33 trained unclipped and also exceeds 4 on 94% of its steps, but
# v33's raw actions hold one sign (right_ankle_roll [0.03, 26.40]), so clipping
# them just flattens them to a constant. Ours cross zero (left_ankle_roll
# [-10.43, +7.44]), and clipping a zero-crossing oscillation does not damp it,
# it squares it off into a full amplitude +-4 alternation. That is the 67%
# sign-reversal rate measured on the robot -- the clip took an oscillation the
# weak action_rate_l2 had allowed and converted it into the worst possible
# waveform for the hardware.
#
# Note that ankle_roll was NOT a reachability problem: the reference needs only
# |action| 1.50 there, well inside 4. The wrists are a different matter --
# right_wrist_roll needs 26.21, left_wrist_roll 12.07, both wrist_pitch 9.3 --
# so for those the clip did also make the reference unreachable, which is the
# same effect already documented above for wrist ORIENTATION tracking.
#
# The actuator bound this was meant to enforce is already enforced downstream:
# _compute_torques clips to dof_effort_limit_list on every substep, in sim and
# on hardware alike. Clipping the action too adds no torque bound, it only
# corrupts the position setpoint feeding the PD law. So: no action clip in
# training, matching v33, whose control configuration is the one we have actual
# hardware evidence for. The deployment keeps its +-4 clip as a mechanical-stop
# guard (ankle_roll: 4 x 0.06 = 0.24 rad, inside the 0.262 rad stop), which is
# exactly how v33 ran on the robot -- benign, because with action_rate_l2 back
# at -1.0 the commands are sign-stable and the clip flattens rather than squares.

x2_31dof_wbt_box_clean_grasp = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="x2_box_clean_grasp",
        num_envs=8192,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=500,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                # The original 10.0 s predates this reference: the clip is
                # 11.68 s, so at 10 s an episode starting at t=0 is cut off
                # during the set-down and never trains the stand-up that ends
                # the motion. 12.0 s covers the clip with a small margin.
                max_episode_length_s=12.0,
            ),
            scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0),
        ),
    ),
    robot=replace(
        robot.x2_31dof_w_object,
        control=replace(
            robot.x2_31dof_w_object.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
            # ankle_roll's scale must be sized to the REFERENCE RANGE, and this clip
            # walks. 0.02 was derived from the in-place clip, whose ankle_roll spanned
            # -5.2..+4.2 deg; the walking reference spans -15.0..+15.0, so reaching it
            # needs |a| = 13.1. Measured on the 2026-08-27 hardware run that played to
            # the end (115642, gain 0.9): the policy pushed |a| to 12.4 and still only
            # covered -5.0..+14.3 deg, tracking 0.65 of the reference range on the
            # right ankle and 0.74 on the left, against 0.85-0.98 for every hip and
            # knee. ankle_roll is the lateral CoP joint, so a robot that cannot roll
            # its ankles cannot shift its weight to unload a foot -- which is why the
            # feet did not step and the robot had to be held up.
            #
            # 0.06 puts the full +-15 deg back within |a| ~ 4.4. It is also what
            # cfg*effort/kp gives, so the override is now documentation rather than a
            # cap: it records that this number is derived from the clip and MUST be
            # re-derived whenever the reference changes.
            #
            # The cap it replaces existed to stop a sustained unreachable ankle
            # command (+35 deg held for 99% of the episode, pinning the 24 N-m
            # actuator, which three reward penalties failed to price out). That
            # exploit was measured on the in-place clip, where holding a constant
            # offset cost no tracking. Watch the command gap on this clip -- it was
            # 33.3 deg while exploited and 4.6 deg after the cap -- and if it grows,
            # the answer is a tighter scale ONLY down to what the reference needs.
            action_scale_overrides={"ankle_roll": 0.06},
        ),
        asset=replace(robot.x2_31dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.x2_31dof_w_object.object,
            object_urdf_path="holosoma/data/motions/x2_31dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.x2_31dof_w_object.init_state, pos=[0.0, 0.0, _X2_WBT_INIT_HEIGHT]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.x2_31dof_wbt_observation_w_object,
    action=g1_29dof_joint_pos,
    termination=x2_31dof_box_grasp_termination,
    randomization=x2_31dof_box_grasp_randomization,
    command=x2_31dof_box_grasp_command,
    curriculum=curriculum.x2_31dof_wbt_curriculum,
    reward=x2_31dof_box_grasp_reward,
    nightly=NightlyConfig(iterations=8000, metrics={}),
)

__all__ = [
    "x2_31dof_box_grasp_command",
    "x2_31dof_box_grasp_randomization",
    "x2_31dof_box_grasp_reward",
    "x2_31dof_box_grasp_termination",
    "x2_31dof_wbt_box_clean_grasp",
]
