"""Whole Body Tracking termination presets for the AgiBot X2 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

from holosoma.config_values.wbt.x2.command import X2_BODY_NAMES_TO_TRACK

x2_31dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": X2_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
                # object tracking (only triggered when has_object=True)
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = ["x2_31dof_wbt_termination"]
