"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, List

import torch
from loguru import logger

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_apply, quat_error_magnitude, quat_rotate_inverse

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_joint_torque_saturation(
    env: WholeBodyTrackingManager,
    joints: str = "",
    ramp_steps: int = 0,
    require_lifted_z: float = 0.0,
    position_term_only: bool = False,
) -> torch.Tensor:
    """Penalize demanding more torque than the actuator can deliver.

    ``action_rate_l2`` charges for how fast a command changes and
    ``limits_dof_pos`` for where the joint ends up, but nothing charges for
    asking an actuator for torque it does not have. Because ``clip_torques`` is
    on, the excess is silently discarded and the shortfall is made up by
    simulated contact -- which the real floor does not supply. That is the
    documented hardware failure: the feet roll onto their edges and the lift
    topples.

    The torque is reconstructed exactly as ``JointPositionActionTerm._compute_torques``
    builds it for control_type "P", but BEFORE its ``torch.clip``, since after
    clipping the excess is by definition zero. Normalizing each joint's excess by
    its own limit keeps a 24 N-m ankle comparable to a 120 N-m hip; without it the
    big actuators would dominate a sum they never come close to saturating.

    Note that ``|action|`` is NOT a torque criterion, which is the trap this term
    exists to avoid: torque is ``kp * (default + a*scale - q) - kd * qd``, so once a
    joint tracks its target the position error, not the action, sets the torque. A
    hip reaching |a| = 9.12 delivers only 34.8 of its 120 N-m because it follows to
    within 0.19 rad, while waist_pitch saturates at |a| = 4.44 precisely because it
    cannot follow.

    Measured on our own iter-9000 rollout, peak demand as a multiple of the joint
    limit and share of the episode saturated:

        hip_pitch     120 N-m    0.87x     0%
        knee          120 N-m    0.57x     0%
        waist_pitch    48 N-m    1.13x     0%
        ankle_pitch    36 N-m    2.97x    18-60%
        ankle_roll     24 N-m    5.96x    84-99%

    The legs are never the bottleneck. The three small joints are, and ankle_roll
    spends nearly the whole episode with no lateral authority left to balance with.

    Gains are the nominal ones rather than the per-env randomized ``_kp_scale`` /
    ``_kd_scale``: this is meant to price the policy's own demand, and using the
    randomized draw would charge two envs differently for the same decision.

    Args:
        env: The environment instance
        joints: Comma-separated substrings restricting the penalty (empty = all
            joints). "waist_pitch,ankle_pitch,ankle_roll" targets the measured
            bottlenecks; taxing joints that never saturate only adds noise.
        ramp_steps: Fade in linearly over this many calls (0 = off). Counted
            locally because ``env.common_step_counter`` does not exist on BaseTask
            envs, where reading it via ``getattr(..., 0)`` would silently pin the
            ramp at zero. One call is one env step, so at 24 steps per iteration
            24_000 is a 1000-iteration ramp -- which is what makes it safe to switch
            this on over a warm start instead of shocking a converged policy.
        require_lifted_z: Only charge while the object is above this height
            (0 = always). Guards against the degenerate optimum of never lifting so
            as never to saturate.
        position_term_only: Charge only the ``kp * (target - q)`` component. A joint
            held by ground contact cannot follow a large commanded offset, so that
            term sits pinned at the limit for the whole episode -- a sustained,
            unrealizable command. The ``kd * qd`` component spikes transiently
            whenever the joint moves fast and is legitimate, so charging it just adds
            noise to the signal we actually want.

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    torques = (
        env.p_gains * (actions * env.action_scales + env.default_dof_pos - env.simulator.dof_pos)
        - env.d_gains * env.simulator.dof_vel
    )
    if position_term_only:
        torques = env.p_gains * (
            actions * env.action_scales + env.default_dof_pos - env.simulator.dof_pos
        )
    limits = torch.clamp(env.torque_limits, min=1e-3)
    excess = torch.clamp(torch.abs(torques) - limits, min=0.0) / limits

    if require_lifted_z > 0.0:
        motion_command = _get_motion_command_and_assert_type(env)
        lifted = (motion_command.simulator_object_pos_w[:, 2] > require_lifted_z).float()
        excess = excess * lifted.unsqueeze(-1)

    if ramp_steps > 0:
        step = getattr(env, "_torque_sat_penalty_step", 0) + 1
        env._torque_sat_penalty_step = step
        excess = excess * min(1.0, step / float(ramp_steps))

    if joints:
        names = env.simulator.dof_names
        keys = [s.strip() for s in joints.split(",") if s.strip()]
        mask = torch.tensor(
            [1.0 if any(k in n for k in keys) else 0.0 for n in names],
            device=actions.device,
            dtype=actions.dtype,
        )
        excess = excess * mask

    return torch.sum(torch.square(excess), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def _body_keep_mask(
    env: WholeBodyTrackingManager,
    exclude_body_names: List[str] | None,
) -> torch.Tensor | None:
    """Boolean mask over tracked bodies; None means keep all."""
    if not exclude_body_names:
        return None
    motion_command = _get_motion_command_and_assert_type(env)
    names = list(motion_command.motion_cfg.body_names_to_track)
    keep = [n not in exclude_body_names for n in names]
    return torch.tensor(keep, dtype=torch.bool, device=env.device)


def motion_relative_body_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    exclude_body_names: List[str] | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    mask = _body_keep_mask(env, exclude_body_names)
    if mask is not None:
        error = error[:, mask]
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    exclude_body_names: List[str] | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    mask = _body_keep_mask(env, exclude_body_names)
    if mask is not None:
        error = error[:, mask]
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    exclude_body_names: List[str] | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    mask = _body_keep_mask(env, exclude_body_names)
    if mask is not None:
        error = error[:, mask]
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(
    env: WholeBodyTrackingManager,
    sigma: float,
    exclude_body_names: List[str] | None = None,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    mask = _body_keep_mask(env, exclude_body_names)
    if mask is not None:
        error = error[:, mask]
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_dof_pos_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Joint-space tracking: match the reference joint angles directly.

    The cartesian body-position rewards are nearly invariant to internal-yaw
    joints (a hip-yaw toe-out barely moves the ankle link), so the policy can
    drift into visibly wrong postures (crab-walk) at little cost. This term
    closes that gap.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.square(env.simulator.dof_pos - motion_command.joint_pos)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_dof_pos_named_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    joint_names: List[str],
) -> torch.Tensor:
    """Joint-space tracking restricted to a named subset of DoFs.

    Used to force critical joints (e.g. waist_pitch) to follow the reference
    when the mean-over-31 ``motion_dof_pos_error_exp`` lets them free-ride.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    names = list(env.simulator.dof_names)
    idxs = [names.index(n) for n in joint_names]
    err = torch.square(
        env.simulator.dof_pos[:, idxs] - motion_command.joint_pos[:, idxs]
    )
    return torch.exp(-err.mean(-1) / sigma**2)


def motion_dof_pos_named_l2(
    env: WholeBodyTrackingManager,
    joint_names: List[str],
) -> torch.Tensor:
    """Linear (mean abs) joint tracking error on a named subset.

    Complements the exp reward: once waist error is large the exp term is
    already ~0 and gives no gradient; this keeps pressure on until the joint
    re-enters the reference band.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    names = list(env.simulator.dof_names)
    idxs = [names.index(n) for n in joint_names]
    err = torch.abs(env.simulator.dof_pos[:, idxs] - motion_command.joint_pos[:, idxs])
    return err.mean(-1)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def hands_to_object_distance_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    hand_body_names: List[str],
    surface_offset: float = 0.0,
) -> torch.Tensor:
    """Smooth gradient pulling the hands toward the simulated object.

    The object tracking rewards are near-flat once the object is left behind its
    reference, so a policy that pantomimes the motion without touching the object
    gets no signal toward it. This term pays for keeping the hand bodies within
    reach of the object's actual position. ``surface_offset`` is the hand-to-center
    distance at contact (object half-extent plus palm offset), so the reward
    saturates at touch instead of encouraging penetration.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    object_pos = motion_command.simulator_object_pos_w  # (num_envs, 3)
    indices = [env.simulator.find_rigid_body_indice(name) for name in hand_body_names]
    hand_pos = env.simulator._rigid_body_pos[:, indices, :]  # (num_envs, num_hands, 3)
    dist = torch.norm(hand_pos - object_pos[:, None, :], dim=-1)
    dist = (dist - surface_offset).clamp(min=0.0)
    # Sum (not mean) over hands == product of per-hand exps: every hand must be
    # at the surface to saturate. With a mean, one touching hand masks the other
    # hovering half a meter away, which produced one-handed pseudo-grasps.
    return torch.exp(-torch.square(dist).sum(-1) / sigma**2)


# ================================================================================================
# Hand / grasp
# ================================================================================================

# Lowest point of the X2 hand collision geometry in the wrist_roll_link frame.
#
# That link carries two colliders: its own STL, which bottoms out at local
# z = -0.1817, and the 5 cm ``*_sphere_hand_link`` sphere, whose fixed joint at
# (0.01, 0, -0.10) puts its lowest point at -0.15. Isaac folds the fixed child
# into the parent, so the sphere never appears in ``body_names`` and its contact
# forces are reported on wrist_roll_link -- and the STL, being 3.2 cm deeper, is
# what actually reaches the floor first. Hence this offset, not the sphere's.
#
# Both hands are mirror images and bound out at the same depth, so one offset
# serves for each. Rotating this single point by the link quaternion reproduces
# the true minimum over all 114k mesh vertices to a mean of 6 mm (corr 0.9998,
# never optimistic by more than 2 mm), which is what makes a per-step clearance
# term affordable: no mesh query, one quaternion rotation.
_X2_PALM_TIP_OFFSET_B = (0.0, 0.0, -0.1817)


def _palm_tip_height(
    env: WholeBodyTrackingManager,
    hand_indices: List[int],
    palm_tip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """World height of the lowest point of each hand. Shape (num_envs, num_hands)."""
    pos = env.simulator._rigid_body_pos[:, hand_indices, :]  # (E, H, 3)
    quat = env.simulator._rigid_body_rot[:, hand_indices, :]  # (E, H, 4) xyzw
    offset = torch.tensor(palm_tip_offset_b, device=pos.device, dtype=pos.dtype)
    offset = offset.expand(pos.shape[0], len(hand_indices), 3)
    tip_w = pos + quat_apply(quat, offset, w_last=True).reshape(pos.shape)
    return tip_w[..., 2]


def penalty_hand_ground_contact(
    env: WholeBodyTrackingManager,
    hand_body_names: List[str],
    contact_force_threshold: float = 5.0,
    ground_height: float = 0.06,
    palm_tip_offset_b: tuple[float, float, float] = _X2_PALM_TIP_OFFSET_B,
) -> torch.Tensor:
    """Count hands that are pressing on the FLOOR (0, 1 or 2).

    The contact sensor reports only the net force per body, so it cannot say
    what a hand is touching -- and the hand must be free to touch the box. The
    two are separated geometrically instead: the reference clip never brings a
    hand mesh closer than 0.193 m to the floor, so a hand carrying real contact
    force while its lowest point is below ``ground_height`` is on the ground,
    not on the box. That leaves ~13 cm of margin under the reference, which is
    why this cannot fire on correct grasping behaviour.

    Requiring force AND proximity (rather than height alone) means a hand may
    still pass low through the air -- only load-bearing contact is charged.
    """
    idxs = [env.simulator.find_rigid_body_indice(n) for n in hand_body_names]
    forces = env.simulator.contact_forces_history[:, :, idxs, :]  # (E, H_hist, N, 3)
    peak_force = torch.max(torch.norm(forces, dim=-1), dim=1)[0]  # (E, N)
    low = _palm_tip_height(env, idxs, palm_tip_offset_b) < ground_height
    return torch.sum((peak_force > contact_force_threshold) & low, dim=1).float()


def penalty_hand_floor_clearance(
    env: WholeBodyTrackingManager,
    hand_body_names: List[str],
    min_clearance: float = 0.12,
    palm_tip_offset_b: tuple[float, float, float] = _X2_PALM_TIP_OFFSET_B,
) -> torch.Tensor:
    """Hinge penalty on hands dipping below ``min_clearance``, in metres summed.

    The contact penalty above is a step: it is zero until the hand is already
    on the floor, so on its own it gives the policy nothing to follow on the way
    down. This supplies the missing slope over the last few centimetres. It is
    deliberately shallower than the reference minimum (0.193 m) so that tracking
    the reference costs exactly zero.
    """
    idxs = [env.simulator.find_rigid_body_indice(n) for n in hand_body_names]
    tip_z = _palm_tip_height(env, idxs, palm_tip_offset_b)
    return torch.clamp(min_clearance - tip_z, min=0.0).sum(dim=-1)


def hands_to_object_relative_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float,
    hand_body_names: List[str],
) -> torch.Tensor:
    """Track each hand's offset FROM THE BOX, rather than its world position.

    ``motion_relative_body_position_error_exp`` averages over all 14 tracked
    bodies, so two hands 0.17 m off-reference move that mean by 0.004 rad^2 and
    cost about 5% of the term -- which is how hands-on-the-floor stayed cheap.
    This gives the grasp its own undiluted gradient.

    Anchoring to the simulated box rather than the reference box matters because
    the box is reset with +-5 cm of noise: the correct palm target moves with it,
    and a world-frame target would ask the hands to grasp where the box is not.
    The desired offset is taken from the reference clip itself, which places the
    palms on the upper side faces (lowest mesh point 0.19-0.22 m, i.e. the middle
    of the usable side surface) -- so this term encodes the grasp geometry that
    was already correct instead of inventing a new one.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    tracked = list(motion_command.motion_cfg.body_names_to_track)
    slots = [tracked.index(n) for n in hand_body_names]
    idxs = [env.simulator.find_rigid_body_indice(n) for n in hand_body_names]

    ref_offset = motion_command.body_pos_w[:, slots, :] - motion_command.object_pos_w[:, None, :]
    act_offset = env.simulator._rigid_body_pos[:, idxs, :] - motion_command.simulator_object_pos_w[:, None, :]
    err = torch.square(act_offset - ref_offset).sum(-1)  # (E, N)
    return torch.exp(-err.sum(-1) / sigma**2)


# ================================================================================================
# Foot / contact regularization
# ================================================================================================


def _reference_stance_mask(
    env: WholeBodyTrackingManager,
    idxs: List[int],
    height: float,
    speed: float,
) -> torch.Tensor:
    """Which feet the REFERENCE has planted right now. [E, F] bool.

    The planted-feet penalties below were written for an in-place clip where the
    answer is always "both", so they were free to charge for any foot motion at
    all. Against a clip that walks, that is simply wrong: it bills the policy for
    the steps the reference is asking it to take, and -- worse -- for the catch
    step that would save it when the CoM leaves the support polygon. Gating on the
    reference keeps the whole point of the penalties (no skating, no hovering, no
    inventing steps) while leaving the reference's own steps, and a recovery step
    where the reference is already mid-swing, unpriced.

    A foot counts as planted when the reference has it both low and slow; height
    alone cannot tell, because the swings in this clip only clear a few cm.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    t = motion_command.time_steps
    ref_pos = motion_command.motion.body_pos_w[t][:, idxs]  # [E, F, 3], floor at z=0
    ref_vel = motion_command.motion.body_lin_vel_w[t][:, idxs]
    return (ref_pos[..., 2] < height) & (torch.norm(ref_vel[..., :2], dim=-1) < speed)


def penalty_foot_slip(
    env: WholeBodyTrackingManager,
    foot_body_names: List[str] | None = None,
    contact_force_threshold: float = 1.0,
    reference_stance_only: bool = False,
    stance_height: float = 0.10,
    stance_speed: float = 0.20,
) -> torch.Tensor:
    """Penalize horizontal foot velocity while the foot is in contact.

    For the planted-feet pickup/set-down clip the reference has <0.5 mm of
    ankle drift, but without an explicit slip cost the policy can still skate
    the feet in sim (PhysX) to recover balance -- then on hardware, where feet
    cannot slide, that strategy becomes aggressive thrashing. Weight this
    negatively.
    """
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    idxs = [env.simulator.find_rigid_body_indice(n) for n in foot_body_names]
    # contact: max |F| over recent history > threshold
    hist = env.simulator.contact_forces_history  # [E, H, B, 3]
    contact = torch.max(torch.norm(hist[:, :, idxs, :], dim=-1), dim=1)[0] > contact_force_threshold
    if reference_stance_only:
        contact = contact & _reference_stance_mask(env, idxs, stance_height, stance_speed)
    vel_xy = env.simulator._rigid_body_vel[:, idxs, :2]  # [E, 2, 2]
    slip = torch.norm(vel_xy, dim=-1) * contact.float()
    return torch.sum(slip, dim=1)


class FeetAnchorPenalty(RewardTermBase):
    """Penalize each foot's horizontal distance from where it planted at reset.

    v30b: penalty_foot_slip charges only the sliding TRANSIENT, so the policy
    learned to buy stability by skating the feet outward once (stance 0.27 ->
    ~0.55 m across the clip, the right foot drifting ~30 cm) and then standing
    still: a one-time slip cost for a permanent stability gain. Anchoring makes
    the displacement itself costly at every step, so the stance stays where it
    planted -- which is also the stance the payload walking policy expects at
    the hybrid hand-off (the reference starts at the default stance and never
    moves the feet).
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        names = cfg.params.get("foot_body_names") or ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.idxs = [env.simulator.find_rigid_body_indice(n) for n in names]
        # Per-foot, per-step displacement larger than this is a teleport.
        self.teleport_threshold = float(cfg.params.get("teleport_threshold", 0.25))
        # Lazy init: rigid-body tensors are not populated at construction time.
        self.anchor: torch.Tensor | None = None
        self.prev: torch.Tensor | None = None

    def _feet_xy(self) -> torch.Tensor:
        return self.env.simulator._rigid_body_pos[:, self.idxs, :2].clone()

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        feet = self._feet_xy()
        if self.anchor is None or self.prev is None:
            self.anchor = feet.clone()
            self.prev = feet.clone()
        # Re-anchor envs whose feet jumped a physically impossible distance in
        # one step: the motion-end resample teleports the robot WITHOUT a
        # manager reset (BeyondMimic-style), which would otherwise leave a
        # stale anchor and a huge false penalty.
        jumped = (torch.norm(feet - self.prev, dim=-1) > self.teleport_threshold).any(dim=-1)
        if jumped.any():
            self.anchor[jumped] = feet[jumped]
        self.prev = feet
        return torch.sum(torch.norm(feet - self.anchor, dim=-1), dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.anchor is None or self.prev is None or env_ids is None:
            return
        feet = self._feet_xy()
        self.anchor[env_ids] = feet[env_ids]
        self.prev[env_ids] = feet[env_ids]


def penalty_feet_contact_loss(
    env: WholeBodyTrackingManager,
    foot_body_names: List[str] | None = None,
    contact_force_threshold: float = 1.0,
    reference_stance_only: bool = False,
    stance_height: float = 0.10,
    stance_speed: float = 0.20,
) -> torch.Tensor:
    """Penalize each foot that has lost ground contact (0, 1, or 2).

    v30 (post hardware trial 3): penalty_foot_slip only fires WHILE a foot is
    in contact, so the policy discovered that lifting a foot entirely and
    re-planting it -- i.e. taking a step -- evades the slip cost for free. On
    hardware this showed up as the right foot hovering and stepping backward
    during the pickup, another step during the hold, and a two-footed hop
    backward at set-down. The reference keeps both feet planted for the whole
    in-place clip, so ANY loss of foot contact is off-reference; this term
    closes the loophole by charging for the airborne phase the slip penalty
    cannot see.
    """
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    idxs = [env.simulator.find_rigid_body_indice(n) for n in foot_body_names]
    hist = env.simulator.contact_forces_history  # [E, H, B, 3]
    contact = torch.max(torch.norm(hist[:, :, idxs, :], dim=-1), dim=1)[0] > contact_force_threshold
    missing = ~contact
    if reference_stance_only:
        missing = missing & _reference_stance_mask(env, idxs, stance_height, stance_speed)
    return torch.sum(missing.float(), dim=1)


class ComSupportMarginPenalty(RewardTermBase):
    """Penalize the CoM approaching, or leaving, the edge of the support polygon.

    Rollouts of v6 fall the same way every time: the CoM crosses outside the feet
    at the top of the lift and the robot topples ~1.1 s later. Nothing in the
    reward notices. Every existing term is a tracking error or a joint-level
    penalty, and by the time tracking error reports the fall the robot is already
    committed -- statically there is no recovery once the CoM is out, whatever the
    ankles do. This gives the balance state its own dense signal, one that starts
    charging while there is still a support polygon to steer back into.

    The margin is the signed distance from the CoM's ground projection to the hull
    of the soles that are carrying load: positive inside, negative outside. Rather
    than build a hull, it is measured as the smallest clearance over a fan of
    horizontal directions, which is batched, cheap, and smooth in the foot pose.

    It only charges while BOTH feet are down. A walking robot puts its CoM outside
    its support polygon on every step -- that is what walking is, you fall forward
    onto the next foot -- and in a v12 rollout all 74 single-support frames had the
    CoM outside, by as much as 262 mm, against 21 of 431 double-support frames at
    worst 21 mm. Charging for the former would pay the policy to stop picking its
    feet up, which is the glued-foot shuffle this task has fought since v31. Static
    support is only the right test when the robot is standing on both feet, so that
    is the only place it is applied; dynamic balance during a step is the ZMP's
    business, and the reference is what supplies it.
    """

    # Sole corners in the ankle_roll_link frame, from the URDF's contact spheres.
    # The spheres are their own links but sit behind fixed joints, so the sim
    # collapses them into the ankle and they never appear in body_names; the foot
    # has to be reconstructed from the ankle pose and these offsets instead.
    SOLE = ((-0.05, 0.05), (-0.05, -0.05), (0.11, 0.05), (0.11, -0.05), (0.139, 0.0))

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        names = cfg.params.get("foot_body_names") or [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        self.idxs = [env.simulator.find_rigid_body_indice(n) for n in names]
        self.margin = float(cfg.params.get("margin", 0.04))
        self.cap = float(cfg.params.get("cap", 0.20))
        self.force_threshold = float(cfg.params.get("contact_force_threshold", 1.0))
        self.contact_height = float(cfg.params.get("contact_height", 0.02))
        k = int(cfg.params.get("directions", 16))
        ang = torch.arange(k, device=env.device, dtype=torch.float32) * (2.0 * math.pi / k)
        self.dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # [K, 2]
        sole = torch.tensor(self.SOLE, device=env.device, dtype=torch.float32)
        self.sole = torch.cat([sole, torch.full_like(sole[:, :1], -0.068)], dim=-1)  # [S, 3]
        self._mass_cache: torch.Tensor | None = None
        self._com_cache: torch.Tensor | None = None
        self._com_probed = False

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        pos = env.simulator._rigid_body_pos[:, self.idxs, :]  # [E, F, 3]
        rot = env.simulator._rigid_body_rot[:, self.idxs, :]  # [E, F, 4] xyzw
        e, f, s = pos.shape[0], pos.shape[1], self.sole.shape[0]
        corners = pos[:, :, None, :] + quat_apply(
            rot[:, :, None, :].expand(e, f, s, 4).reshape(-1, 4),
            self.sole[None, None].expand(e, f, s, 3).reshape(-1, 3),
            w_last=True,
        ).reshape(e, f, s, 3)

        hist = env.simulator.contact_forces_history  # [E, H, B, 3]
        pressed = (
            torch.max(torch.norm(hist[:, :, self.idxs, :], dim=-1), dim=1)[0] > self.force_threshold
        )  # [E, F]
        # Force on its own was calling both feet loaded almost all of the time, which
        # handed the term a two-foot polygon during single support and left it worth
        # -0.006 of a 65-point reward -- silent. The sole has to be on the ground too.
        grounded = corners[..., 2].amin(dim=-1) < self.contact_height
        loaded = pressed & grounded  # [E, F]

        proj = corners[..., :2].reshape(e, f * s, 2) @ self.dirs.T  # [E, F*S, K]
        keep = loaded[:, :, None].expand(e, f, s).reshape(e, f * s)
        proj = torch.where(keep[..., None], proj, torch.full_like(proj, -1e6))
        support = proj.amax(dim=1)  # [E, K] hull extent in each direction
        margin = (support - self._com(env) @ self.dirs.T).amin(dim=-1)  # [E]
        pen = torch.clamp(self.margin - margin, min=0.0, max=self.cap)
        # Standing on both feet is the only state a static support test describes, so
        # it is the only one charged for. That also disposes of the airborne case,
        # whose -1e6 margin would otherwise ride the cap through every flight phase.
        return pen * loaded.all(dim=-1).to(pen.dtype)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Nothing to carry between episodes: the margin is read off the current pose."""

    def _com(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        mass = self._masses(env)  # [E, B]
        pos = env.simulator._rigid_body_pos  # [E, B, 3] -- link ORIGINS
        off = self._com_offsets(env)
        if off is not None:
            # A link's origin is not its centre of mass: they sit a mean of 66 mm
            # apart on this robot and 192 mm apart on the torso, which is the
            # heaviest body of the lot.
            rot = env.simulator._rigid_body_rot  # [E, B, 4] xyzw
            n = pos.shape[0] * pos.shape[1]
            pos = pos + quat_apply(
                rot.reshape(n, 4), off.reshape(n, 3), w_last=True
            ).reshape(pos.shape)
        return (pos[..., :2] * mass[..., None]).sum(dim=1) / mass.sum(dim=1, keepdim=True)

    def _com_offsets(self, env: WholeBodyTrackingManager) -> torch.Tensor | None:
        """Each link's own centre of mass in its own frame, or None if the sim won't say."""
        if self._com_probed:
            return self._com_cache
        self._com_probed = True
        sim = env.simulator
        try:
            if hasattr(sim, "_robot") and hasattr(sim._robot, "root_physx_view"):
                off = sim._robot.root_physx_view.get_coms().to(env.device)[..., :3]
            else:  # mujoco keeps it on the model
                from holosoma.simulator.mujoco.fields import _field_view

                off = _field_view(sim, "body_ipos").to(env.device)
            if off.dim() == 2:
                off = off[None]
            off = torch.stack(
                [self._reorder(env, off[..., c]) for c in range(3)], dim=-1
            )
            if off.shape[0] == 1:
                off = off.expand(env.num_envs, -1, -1)
            self._com_cache = off.contiguous()
        except Exception:  # noqa: BLE001 -- fall back to link origins, as before
            self._com_cache = None
        return self._com_cache

    def _masses(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        if getattr(self, "_mass_cache", None) is None:
            sim = env.simulator
            if hasattr(sim, "_robot") and hasattr(sim._robot, "root_physx_view"):
                m = sim._robot.root_physx_view.get_masses().to(env.device)
            else:  # mujoco keeps it on the model
                from holosoma.simulator.mujoco.fields import _field_view

                m = _field_view(sim, "body_mass").to(env.device)
            m = self._reorder(env, m.reshape(-1, m.shape[-1]))
            if m.shape[0] == 1:
                m = m.expand(env.num_envs, -1)
            self._mass_cache = m.contiguous()
        return self._mass_cache

    @staticmethod
    def _reorder(env: WholeBodyTrackingManager, x: torch.Tensor) -> torch.Tensor:
        """Put a per-body quantity into the same body order as _rigid_body_pos.

        body_ids is a permutation, not a prefix: the simulator lists bodies depth
        first and holosoma wants its config's order, so _rigid_body_pos is gathered
        through it. Truncating to the first nb entries instead -- which is what this
        did until now -- pairs every body's position with some other body's mass. The
        resulting "CoM" sits near the pelvis whatever the robot does, so the support
        margin read healthy through every fall and the term was worth -0.006 of a
        65-point reward.
        """
        nb = env.simulator._rigid_body_pos.shape[1]
        ids = getattr(env.simulator, "body_ids", None)
        if ids is not None and len(ids) == nb and x.shape[1] > max(ids):
            return x[:, list(ids)]
        return x[:, :nb]


def _feet_gravity_in_foot_frame(env: WholeBodyTrackingManager, idxs: List[int]) -> torch.Tensor:
    """World -Z (gravity/up) direction expressed in each foot's local frame.

    Returns a [E, F, 3] tensor. For a perfectly level foot this is [0, 0, -1];
    tilting the foot moves the vector into the local XY plane. In the ankle-roll
    link frame local x ~ forward (a nonzero x means toe/heel pitch) and local
    y ~ lateral (a nonzero y means the foot is rolled onto its medial/lateral
    edge). The horizontal magnitude sqrt(x^2 + y^2) equals sin(tilt angle).
    """
    rot = env.simulator._rigid_body_rot[:, idxs, :]  # [E, F, 4], xyzw
    e, f = rot.shape[0], rot.shape[1]
    grav = torch.zeros(e * f, 3, device=rot.device, dtype=rot.dtype)
    grav[:, 2] = -1.0
    g_foot = quat_rotate_inverse(rot.reshape(e * f, 4), grav, w_last=True)
    return g_foot.reshape(e, f, 3)


def penalty_foot_not_flat(
    env: WholeBodyTrackingManager,
    foot_body_names: List[str] | None = None,
    contact_force_threshold: float = 20.0,
    roll_weight: float = 1.0,
    pitch_weight: float = 0.3,
) -> torch.Tensor:
    """Penalize each stance foot for tilting off level (the foot-edge failure).

    v31: hardware trial 4 (v30 deploy) showed the robot standing on the OUTER
    EDGES of its feet -- most of each sole off the floor -- and rocking to
    rebalance on those edges during pickup. Deploy logs measured 8-13 deg of
    ankle-roll deviation, worst on the right foot, and the sim foot-tilt bounced
    3.8-28.6 deg across checkpoints with NO downward trend: nothing in the
    objective measured foot flatness, so the policy tilted freely. PhysX never
    feels the instability a tiny contact patch causes, which is why sim looked
    fine while hardware did not.

    This term reads the ankle-roll link orientation directly and charges the
    horizontal component of gravity in the foot frame -- i.e. sin(tilt) -- while
    the foot is loaded. Roll (lateral tilt onto the edge) is weighted above
    pitch (toe/heel) because edge-standing is the roll axis. Gated on contact so
    it targets the stance phase; feet_contact_loss keeps the policy from evading
    it by lifting the foot instead of flattening it.
    """
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    idxs = [env.simulator.find_rigid_body_indice(n) for n in foot_body_names]
    g_foot = _feet_gravity_in_foot_frame(env, idxs)  # [E, F, 3]
    tilt = roll_weight * g_foot[..., 1].abs() + pitch_weight * g_foot[..., 0].abs()
    hist = env.simulator.contact_forces_history  # [E, H, B, 3]
    contact = torch.max(torch.norm(hist[:, :, idxs, :], dim=-1), dim=1)[0] > contact_force_threshold
    return torch.sum(tilt * contact.float(), dim=1)


def penalty_feet_edge_contact(
    env: WholeBodyTrackingManager,
    foot_body_names: List[str] | None = None,
    contact_force_threshold: float = 20.0,
    tilt_threshold: float = 0.17,
) -> torch.Tensor:
    """Score a foot resting on its edge as "not really planted" (0, 1, or 2).

    v31: closes the sim-to-real gap that lets an edge-standing foot look fine to
    the simulator. feet_contact_loss already flags a foot that has left the
    ground, but a foot pressed onto its edge still registers contact force in
    PhysX, so it passes as "planted" even though on hardware it is unstable.
    This companion term counts any foot that is BOTH in contact AND tilted past
    tilt_threshold (default sin(~10 deg)) as a lost contact -- a hard, thresholded
    push that pairs with the smooth penalty_foot_not_flat gradient: a foot only
    counts as good when it is down AND flat.
    """
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    idxs = [env.simulator.find_rigid_body_indice(n) for n in foot_body_names]
    g_foot = _feet_gravity_in_foot_frame(env, idxs)  # [E, F, 3]
    horiz = torch.norm(g_foot[..., :2], dim=-1)  # sin(tilt)
    hist = env.simulator.contact_forces_history  # [E, H, B, 3]
    contact = torch.max(torch.norm(hist[:, :, idxs, :], dim=-1), dim=1)[0] > contact_force_threshold
    edge = contact & (horiz > tilt_threshold)
    return torch.sum(edge.float(), dim=1)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        pattern = cfg.params.get("undesired_contacts_body_names", "")
        body_names = self.env.simulator.body_names  # type: ignore[attr-defined]
        undesired_contacts_body_names = [body_name for body_name in body_names if re.match(pattern, body_name)]
        # The default empty pattern "" matches every body, so an empty result can only come from an
        # explicit, non-empty pattern that matched nothing: warn
        if pattern and not undesired_contacts_body_names:
            logger.warning(
                f"UndesiredContacts: pattern '{pattern}' matched no body names in "
                f"{body_names}; contact penalty will be a permanent no-op (always zero)."
            )
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            body_names,
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
