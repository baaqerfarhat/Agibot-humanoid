"""Whole Body Tracking command presets for the AgiBot X2 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

# v11 (post hardware trial): dof_pos noise raised 0.1 -> 0.15 rad. On the real
# robot the ramp-to-start pose is imperfect (encoder offsets, gravity sag), and
# the policy fell as soon as the bend started from a slightly-off pose. Training
# must cover that error band, not just +/-0.1 rad around the reference.
init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.15,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

# Bodies tracked against the reference motion. Mirrors the G1 selection: pelvis
# anchor, the two legs (hip_roll / knee / ankle_roll), torso, and the two arms
# (shoulder_roll / elbow / wrist_roll). For X2 the arm end-effector is the
# wrist_roll_link (the hand), which is what must follow the box.
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

# Sampling structure: the adaptive sampler focuses on the failure-heavy grasp
# moment, but is capped by a 30% uniform floor. start_at_timestep_zero_prob was
# raised 0.15 -> 0.35 for v11: on hardware the failure was the standing-start
# bend, which is exactly the phase that only t=0 episodes exercise end-to-end
# (mid-motion starts spawn the robot already bent with the box at reference).
motion_config = MotionConfig(
    motion_file="holosoma/data/motions/x2_31dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=X2_BODY_NAMES_TO_TRACK,
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    adaptive_uniform_ratio=0.3,
    start_at_timestep_zero_prob=0.35,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/x2_31dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

x2_31dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

x2_31dof_wbt_command_w_object = replace(
    x2_31dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

__all__ = [
    "x2_31dof_wbt_command",
    "x2_31dof_wbt_command_w_object",
]
