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
                # v27: ankles removed from bad_tracking (same reason as reward
                # tracking). Killing an episode for a 40 cm ankle drift was
                # teaching the policy to thrash the legs to "save" the episode
                # rather than stay planted and recover with the upper body.
                "bad_motion_body_pos_body_names": [
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                ],
                # object tracking (only triggered when has_object=True)
                # v29: 0.45 -> 1.0 m / 1.0 -> 2.0 rad. With 0.45 m the episode
                # died a fraction of a second after any box drop, so the policy
                # NEVER saw a post-drop state -- on hardware a slipped box left
                # it in unvisited states and it thrashed. At 1.0 m a dropped
                # box (floor vs chest-height reference ~0.7 m) keeps the
                # episode alive for several seconds; the body-tracking rewards
                # still guide the robot to finish the motion upright, so it
                # learns "box lost -> stay balanced and come back up" instead
                # of undefined behavior. Pantomime regression is unlikely at
                # this warm-start stage: the object terms are ~45% of positive
                # reward and the grasp is already mastered.
                "bad_object_pos_threshold": 1.0,
                "bad_object_ori_threshold": 2.0,
            },
        ),
    }
)

# Crawl: tighter tracking kill so the policy cannot farm reward by lying
# limp after a flop. Also watch knees (primary support limbs on a crawl) in
# addition to the wrists.
# v4: slightly tighter than v3 so mid-slope flops end the episode sooner and
# the adaptive sampler focuses there.
x2_31dof_wbt_crawl_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                "bad_ref_pos_threshold": 0.40,
                "bad_ref_ori_threshold": 0.7,
                "bad_motion_body_pos_threshold": 0.30,
                "body_names_to_track": X2_BODY_NAMES_TO_TRACK,
                "bad_motion_body_pos_body_names": [
                    "left_wrist_roll_link",
                    "right_wrist_roll_link",
                    "left_knee_link",
                    "right_knee_link",
                    "torso_link",
                ],
                "bad_object_pos_threshold": 1.0,
                "bad_object_ori_threshold": 2.0,
            },
        ),
    }
)

__all__ = ["x2_31dof_wbt_termination", "x2_31dof_wbt_crawl_termination"]
