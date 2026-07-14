"""Whole Body Tracking reward presets for the AgiBot X2 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

# Allow contacts on the feet (ankle_roll) and the full hand/forearm chain
# (wrist_yaw, wrist_pitch, wrist_roll) so the robot can plant its feet and hug
# the box against its forearms; penalize contacts on everything else.
_X2_UNDESIRED_CONTACTS_REGEX = (
    "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    "(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)"
    "(?!left_wrist_pitch_link$)(?!right_wrist_pitch_link$)"
    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
)

x2_31dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
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
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
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
    }
)

x2_31dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **x2_31dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
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
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

# Object tracking is weighted heavily (3.0/1.5 vs G1's 1.0/1.0): with equal
# weights the policy converged to tracking the body motion while shoving the
# box aside, since the box contributed only 2/7 of the positive reward. Making
# the box the dominant objective forces the grasp/lift to actually happen.
x2_31dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **x2_31dof_wbt_reward.terms,
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
        # Dense shaping toward the grasp: the object tracking terms above are
        # flat-zero once the box is left behind, letting the policy pantomime
        # the carry. surface_offset 0.25 = box half-width (~0.23) + palm sphere
        # standoff, so the reward saturates at contact, not penetration.
        "hands_to_object_distance_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:hands_to_object_distance_exp",
            params={
                "sigma": 0.25,
                "hand_body_names": ["left_wrist_roll_link", "right_wrist_roll_link"],
                "surface_offset": 0.25,
            },
            weight=1.5,
        ),
    }
)

__all__ = ["x2_31dof_wbt_fast_sac_reward", "x2_31dof_wbt_reward", "x2_31dof_wbt_reward_w_object"]
