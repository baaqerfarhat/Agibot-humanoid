"""Whole Body Tracking termination presets for the AgiBot X2 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

from holosoma.config_values.wbt.x2.command import X2_BODY_NAMES_TO_TRACK

x2_31dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        # v14: thresholds loosened after the standing-start crux plateaued
        # (fall time stuck at ~1.4 s across iters 2k/14.5k/21k). The tight
        # 0.25 m object/body bounds killed every standing-start episode at the
        # instant of an imperfect grasp, starving the stand-up-with-box phase
        # of practice. Looser bounds let imperfect grasps play on so the
        # policy can learn the recovery; tighten again once the full motion
        # survives from t=0.
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.7,
                "bad_ref_ori_threshold": 1.0,
                "bad_motion_body_pos_threshold": 0.40,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": X2_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
                # object tracking (only triggered when has_object=True)
                "bad_object_pos_threshold": 0.45,
                "bad_object_ori_threshold": 1.0,
            },
        ),
    }
)

__all__ = ["x2_31dof_wbt_termination"]
