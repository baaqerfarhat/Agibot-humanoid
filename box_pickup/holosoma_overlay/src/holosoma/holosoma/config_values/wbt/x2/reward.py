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
        # v30: 0.5 -> 1.0. Hardware trial 3 showed the torso rolling 13-25 deg
        # LEFT during every pickup; at weight 0.5 / sigma 0.4 a 25 deg error
        # still kept ~30% of this reward, too cheap to matter. Upright torso
        # is a hard requirement for the demo.
        # v33: 1.0 -> 1.5 / sigma 0.4 -> 0.35. IRL pitch-collapse ~1s into the
        # squat; make torso orientation cheaper to ignore.
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.35},
            weight=1.5,
        ),
        # Motion tracking rewards - relative body frame
        # v27: exclude ankles from cartesian body tracking rewards (obs dim
        # unchanged so warm-start still works). Feet are planted in the
        # reference; rewarding ankle world-pose pulls against real contact.
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={
                "sigma": 0.3,
                "exclude_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={
                "sigma": 0.4,
                "exclude_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={
                "sigma": 1.0,
                "exclude_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={
                "sigma": 3.14,
                "exclude_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
            weight=1.0,
        ),
        # Joint-space tracking (v17): the cartesian terms above barely notice a
        # hip-yaw toe-out (rotating the leg about its axis hardly moves the
        # ankle point), which let the policy carry the box in a crab-walk. Mean
        # squared error over 31 joints; sigma 0.25 => ~0.83 reward when e.g.
        # 4 joints are each 0.3 rad off.
        "motion_dof_pos_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_error_exp",
            params={"sigma": 0.25},
            weight=1.0,
        ),
        # v33 WAIST FIX: IRL + open-loop probes showed the actor commands
        # waist_pitch NEGATIVE while the reference squat is +18 deg (and sim
        # PD targets go to -160 deg). Mean-over-31 dof tracking lets waist
        # free-ride; these terms force waist (+ hip pitch) to track the bend.
        # v33b: sigma 0.10 -> 0.12. The exp term is a razor spike that is 0
        # everywhere except within ~6 deg of ref, so it gave almost no gradient
        # during the descent (tgt lagged ref by 4-6 deg) or the hold. Widen the
        # band so measured waist is actively pulled toward ref across ~+-8 deg.
        # v33f: sigma 0.12 -> 0.20, weight 3 -> 4. The HOLD pitch PLATEAUED at a
        # -6 deg target (ref +8.6) across 229.5k->241.5k: at sigma 0.12 the exp
        # is still ~0.01 at the ~14 deg hold error, so ONLY the flat L2 gradient
        # remained and it stalled against the other rewards. sigma 0.20 keeps a
        # real, smooth gradient out to ~15 deg error (exp ~0.2 there) -- gradient
        # exactly where it is stuck -- while still rewarding tight tracking near
        # ref (~0.83 at 5 deg). Higher weight gives it authority to win the hold.
        "motion_waist_pitch_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={"sigma": 0.20, "joint_names": ["waist_pitch_joint"]},
            weight=4.0,
        ),
        "motion_waist_roll_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={"sigma": 0.12, "joint_names": ["waist_roll_joint"]},
            weight=1.5,
        ),
        # v33c WAIST-YAW FIX: rollout @220000 showed the torso twisting to
        # waist_yaw +38 deg (target +52) and holding >20 deg for 51% of the
        # clip, so the robot carried the box off to one side and pivoted on the
        # right foot (0.24 m step) to sustain the twist. The reference only
        # twists to +22 deg (mean +6) -- the policy was over-twisting ~2x. Like
        # waist_pitch before it, waist_yaw had NO dedicated tracking term and
        # free-rode the mean-over-31 dof reward. Force it to follow the ref twist
        # (which keeps the box square/front) instead of exaggerating it.
        "motion_waist_yaw_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={"sigma": 0.12, "joint_names": ["waist_yaw_joint"]},
            weight=2.0,
        ),
        "motion_hip_pitch_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_error_exp",
            params={
                "sigma": 0.15,
                "joint_names": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
            },
            weight=1.5,
        ),
        # Keep a gradient when waist is already far off (exp term saturated at 0).
        # v33b: -2.0 -> -6.0. This is the ONLY term with a gradient in the HOLD
        # phase, where @218500 the policy drifted to -10 deg while ref wants
        # +8.6 deg (an ~18 deg wrong-sign error that is WITHIN joint limits, so
        # PhysX does not mask it and this L2 sees it directly). The mid-squat
        # +55 deg target overshoot is invisible here on purpose -- ref sits at
        # the +-18 deg hard limit so measured is clamped == ref, and deploy
        # clips the target to +-18 deg == ref, making that overshoot benign.
        # The real deploy risk is the hold wrong-sign; -6.0 gives it real teeth.
        "motion_waist_pitch_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_l2",
            params={"joint_names": ["waist_pitch_joint"]},
            weight=-6.0,
        ),
        # v33c: linear far-field pull on the over-twist. The exp term above is
        # ~0 once yaw is >~10 deg past ref (e.g. the +38 vs +22 seen @220000),
        # so this keeps a constant gradient dragging the twist back to the
        # reference and stops the side-carry / foot-pivot strategy.
        "motion_waist_yaw_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_named_l2",
            params={"joint_names": ["waist_yaw_joint"]},
            weight=-3.0,
        ),
        # Regularization. v27: action_rate -0.4 -> -0.8 after hardware trials
        # showed aggressive leg thrashing at stand-up; smoother targets are
        # the cheapest sim2real win for planted-feet manipulation.
        # v28: -0.8 -> -1.0, targeting the residual stand-up "tweaking".
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
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
        # v27: plant the feet. Reference already has pinned ankles; this stops
        # the policy from skating them in sim to recover from COM shifts.
        # v28: -1.0 -> -2.0 -- sim skating becomes hardware thrashing.
        # v30: -2.0 -> -3.0. Hardware trial 3 (v27 deploy) showed stepping in
        # pickup/hold/set-down; in sim the same strategy expresses as skating
        # (raw slip ~4/episode persisted through v29). Squeeze it harder --
        # feet_contact_loss below blocks the escape into actual step-taking.
        # v33e: -3.0 -> -6.0. Once the waist tracking budget grew (pitch exp 3 +
        # L2 -6, yaw exp 2 + L2 -3, roll/hip 1.5) the -3.0 slip penalty was
        # out-competed: rollout @229500 slid in place ~11% of frames (p95 ~0.2
        # m/s, ~0.8 m cumulative path per foot) and raw_rew_foot_slip actually
        # ROSE 7.5 -> 11.5 during training. Net drift is ~0 (feet don't walk)
        # but the chatter is the visible sim2real slip risk. Scale the anti-
        # slide penalty to match the new waist budget so planted-and-still wins.
        "foot_slip": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_foot_slip",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_force_threshold": 1.0,
            },
            weight=-6.0,
        ),
        # v30: close the stepping loophole. foot_slip is zero for an airborne
        # foot, so stepping (lift + re-plant) was free; on hardware the robot
        # stepped during pickup, hold, AND set-down. The reference plants both
        # feet for the entire clip: charge -2 per unloaded foot per step.
        # v30b: threshold 1 -> 20 N. At 1 N a foot "barely touching" the floor
        # (the exact behavior seen at iter 168.5k) still counts as contact;
        # 20 N (<10% of a foot's nominal share of body weight) demands the
        # foot actually be loaded.
        # v33e: -2.0 -> -4.0. foot_slip just doubled to -6.0; if lifting a foot
        # (which zeroes foot_slip) stays cheaper than sliding, the policy will
        # escape the slip penalty by stepping. Raise the contact-loss cost in
        # lockstep so "plant AND hold still" remains the only cheap option.
        "feet_contact_loss": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_feet_contact_loss",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_force_threshold": 20.0,
            },
            weight=-4.0,
        ),
        # v30b: anchor the feet where they planted at reset. Slip charges only
        # the sliding transient, so the policy skated the stance from 0.27 m
        # out to ~0.55 m (one-time cost, permanent stability gain) -- which on
        # hardware becomes stepping, and hands the walking policy a stance it
        # was never trained on. Linear in drift distance, per foot, per step.
        # v32: -2.0 -> -1.0. During the HOLD the box shifts the COM; the policy
        # needs a few centimetres of ankle/stance micro-adjust to stay upright
        # under load. The hard -2.0 made that illegal and pushed failures into
        # the post-set-down fall. Flat-foot / edge / contact-loss terms still
        # block skating and edge-standing.
        "feet_anchor": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetAnchorPenalty",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            },
            weight=-1.0,
        ),
        # v31: kill the foot-edge failure. Hardware trial 4 (v30 deploy) stood on
        # the outer edges of both feet (soles off the floor) and rocked to
        # rebalance; deploy logs showed 8-13 deg ankle-roll deviation and the sim
        # foot-tilt never trended down across checkpoints because NOTHING in the
        # objective measured foot flatness. This reads the ankle-roll link
        # orientation and charges sin(tilt) per loaded foot (roll weighted over
        # pitch -- edge-standing is the roll axis). Smooth gradient toward a level
        # sole; gated on contact so feet_contact_loss blocks lifting as an escape.
        "foot_not_flat": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_foot_not_flat",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_force_threshold": 20.0,
                "roll_weight": 1.0,
                "pitch_weight": 0.3,
            },
            weight=-3.0,
        ),
        # v31: sim-to-real closer. A foot pressed onto its edge still registers
        # contact force in PhysX, so feet_contact_loss passes it as "planted"
        # while on hardware it is unstable. Count any foot that is in contact AND
        # tilted past ~10 deg as a lost contact -- a hard threshold that pairs
        # with the smooth foot_not_flat gradient: a foot is good only when it is
        # both down and flat.
        "feet_edge_contact": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_feet_edge_contact",
            params={
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_force_threshold": 20.0,
                "tilt_threshold": 0.17,
            },
            weight=-2.0,
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
        # The legs are never torque-limited: hip_pitch peaks at 0.95x of its 120 N-m
        # limit and the knees at 0.29x, across every policy measured. The joints that
        # actually saturate are the three small ones -- waist_pitch 48, ankle_pitch 36,
        # ankle_roll 24 N-m -- so charge on DELIVERED torque there, not on |a| at the
        # hips. See penalty_joint_torque_saturation for the measurement table.
        "penalty_joint_torque_saturation": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_joint_torque_saturation",
            params={"joints": "waist_pitch,ankle_pitch,ankle_roll", "ramp_steps": 24_000,
                    "require_lifted_z": 0.30},
            weight=-0.05,
        ),
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

# ---------------------------------------------------------------------------
# Slope / prone crawl rewards.
#
# The upright box-pickup suite (feet_contact_loss, feet_anchor, foot_slip,
# foot_not_flat, feet_edge_contact) is actively hostile to crawling: it demands
# both feet stay loaded and glued to their reset pose. Training the crawl with
# that suite produced a policy that just flopped. This preset is the G1 WBT
# tracking core (what OmniRetarget used for crawl_slope) plus crawl-specific
# contact allowances and stronger orientation tracking so the robot stays
# belly-down on the incline.
# ---------------------------------------------------------------------------
_X2_CRAWL_ALLOWED_CONTACTS_REGEX = (
    "^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    "(?!left_wrist_roll_link$)(?!right_wrist_roll_link$)"
    "(?!left_wrist_pitch_link$)(?!right_wrist_pitch_link$)"
    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
    "(?!left_knee_link$)(?!right_knee_link$)"
    "(?!left_elbow_link$)(?!right_elbow_link$).+$"
)

x2_31dof_wbt_crawl_reward = RewardManagerCfg(
    terms={
        # v5: POSITION is the crawl (+Y up the slope).
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.25},
            weight=3.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.30},
            weight=2.0,
        ),
        # Relative body-frame tracking (hands / knees / torso vs pelvis).
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.25},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.35},
            weight=1.5,
        ),
        # Velocity tracking keeps the crawl cadence in phase with the clip.
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 0.8},
            weight=1.5,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Joint-space tracking: limb phasing of the crawl gait.
        "motion_dof_pos_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_dof_pos_error_exp",
            params={"sigma": 0.20},
            weight=2.0,
        ),
        # v5: undo v4's -1.2 action-rate (over-smoothed the gait into a
        # shuffle that couldn't climb). Mild smoothness only.
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.4,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        # Allow hands / feet / knees / elbows to plant on the slope; still
        # punish belly / pelvis / head crashes that mean the robot flopped.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": _X2_CRAWL_ALLOWED_CONTACTS_REGEX,
            },
            weight=-1.0,
        ),
    }
)

__all__ = [
    "x2_31dof_wbt_crawl_reward",
    "x2_31dof_wbt_fast_sac_reward",
    "x2_31dof_wbt_reward",
    "x2_31dof_wbt_reward_w_object",
]
