"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch
from loguru import logger

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude, quat_rotate_inverse

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
# Foot / contact regularization
# ================================================================================================


def penalty_foot_slip(
    env: WholeBodyTrackingManager,
    foot_body_names: List[str] | None = None,
    contact_force_threshold: float = 1.0,
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
    return torch.sum((~contact).float(), dim=1)


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


def penalty_action_over_effort(
    env: WholeBodyTrackingManager, limit: float = 4.0, joints: str = "",
    ramp_steps: int = 0, require_lifted_z: float = 0.0
) -> torch.Tensor:
    """Penalize commanding torque the actuator cannot deliver.

    `action_scale = cfg_scale * effort / kp`, so |a| = 4 IS the effort limit by
    construction: past that the extra command produces exactly zero extra torque.
    Nothing in the existing reward set costs the policy anything for exceeding it --
    `penalty_action_rate` penalizes action CHANGES and `limits_dof_pos` penalizes
    POSITION limits, so a saturated command is free.

    That matters because simulation hides the consequence. Measured on the v33 box
    pickup, the hips command |a| up to 9.2 (2.3x the limit) for 52-75% of the rise, in
    sim and on hardware alike -- but sim's contact resists hard enough to make up the
    shortfall, so the policy is never taught that the torque is unavailable. On the real
    robot the floor yields instead and the motion fails.

    Penalizing only the EXCESS (not the magnitude) leaves the policy free to use the
    full actuator envelope, and only makes the impossible part expensive.

    Args:
        env: the environment instance
        limit: |a| at the effort limit; 4.0 for this action_scale convention
        require_lifted_z: only charge the penalty while the box is above this height
            (0 = always charge). WITHOUT this gate the term has a degenerate optimum:
            never touch the box at all. Measured on a first attempt at weight -0.5, the
            policy learned exactly that -- 5/5 seeds stood for the full episode with
            boxZmax 0.19 (the box's resting height) and carry 0.00 s, while leg tracking
            IMPROVED to 5.5-6.8 deg because a near-static reference is easy to track.
            Approaching and lifting is what costs hip effort, so if not-lifting is
            cheaper than lifting, the policy stops lifting. Gating on box height makes
            the penalty shape HOW the robot lifts rather than WHETHER it does.
        ramp_steps: fade the penalty in linearly over this many env steps (0 = off).
            NOTE the unit: `common_step_counter` advances once per `env.step()`, NOT
            per environment, and there are 24 steps per learning iteration -- so 24_000
            is ~1000 iterations regardless of how many envs are running.
        joints: comma-separated substrings to restrict the penalty to (empty = all).
            Restricting to "hip_pitch" targets the measured bottleneck without taxing
            joints that legitimately run near their limit.

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    excess = torch.clamp(torch.abs(actions) - limit, min=0.0)

    if require_lifted_z > 0.0:
        motion_command = _get_motion_command_and_assert_type(env)
        lifted = (motion_command.simulator_object_pos_w[:, 2] > require_lifted_z).float()
        excess = excess * lifted.unsqueeze(-1)

    # Ramp in. Warm-starting from a policy that already sits at |a| ~ 9 means this term
    # arrives as a large negative advantage on a previously-optimal policy, which PPO
    # handles badly. Fading it in lets the posture migrate instead of being shocked.
    #
    # Count locally rather than using env.common_step_counter: that attribute only
    # exists on LeggedRobotLocomotionManager, and the WBT envs inherit from BaseTask,
    # where _init_counters/_update_counters_each_step are no-ops. Reading it through
    # getattr(..., 0) silently pins the ramp at zero and the term becomes inert.
    if ramp_steps > 0:
        step = getattr(env, "_effort_penalty_step", 0) + 1
        env._effort_penalty_step = step
        excess = excess * min(1.0, step / float(ramp_steps))
    if joints:
        names = env.simulator.dof_names
        keys = [s.strip() for s in joints.split(",") if s.strip()]
        mask = torch.tensor(
            [1.0 if any(k in n for k in keys) else 0.0 for n in names],
            device=actions.device, dtype=actions.dtype,
        )
        excess = excess * mask
    return torch.sum(torch.square(excess), dim=1)


def penalty_joint_torque_saturation(
    env: WholeBodyTrackingManager, joints: str = "",
    ramp_steps: int = 0, require_lifted_z: float = 0.0
) -> torch.Tensor:
    """Penalize demanding more torque than the actuator can deliver.

    This replaces `penalty_action_over_effort`, which charged for `|a| > 4` on the
    hips. That premise was wrong. `action_scale = cfg * effort / kp` makes `|a| = 4`
    the effort limit ONLY if the joint stays pinned at its default: the torque is
    `kp * (default + a*scale - q) - kd * qd`, so once the joint MOVES toward its
    target the position error -- not the action -- sets the torque. Measured on the
    v31 rollout the hip commands `|a| = 9.12` while delivering 34.8 N-m of its 120
    N-m limit, because it tracks its target to within 0.19 rad. Meanwhile waist_pitch
    saturates at `|a| = 4.44`, precisely because it CANNOT follow: a 0.59 rad error
    pins the PD at the ceiling. Action magnitude is not a torque criterion.

    So compute the torque the PD actually asks for, exactly as
    `JointControlAction.apply_actions` does before its `torch.clip`, and charge for
    the part above the limit. Normalizing the excess by each joint's own limit keeps
    a 24 N-m ankle comparable to a 120 N-m hip; without it the large actuators would
    dominate a sum they are nowhere near saturating.

    Measured saturation on seed 600 (peak demand as a multiple of the joint limit,
    and share of the episode saturated):

        joint         limit   baseline     w=-0.05 +2000   w=-0.15 +2000
        hip_pitch     120     0.71x   0%   0.75x   0%      0.83x   0%
        knee          120     0.28x   0%   0.24x   0%      0.29x   0%
        waist_pitch    48     1.82x  11%   0.99x   0%      0.89x   0%
        ankle_pitch    36     1.39x   0%   1.18x   1%      1.17x   1%
        ankle_roll     24     1.99x  22%   4.17x  99%      7.61x  98%

    The legs are never the bottleneck. The joints that run out of authority are the
    three small ones, and ankle_roll gets monotonically worse the longer the old
    hip-targeted penalty trains -- the policy sheds hip command and the load lands on
    the smallest actuator on the robot, which in sim is free (the simulator delivers
    its 24 N-m and the contact model absorbs the rest) and on hardware means no
    lateral ankle authority is left.

    Args:
        env: the environment instance
        joints: comma-separated substrings to restrict the penalty to (empty = all).
            "waist_pitch,ankle_pitch,ankle_roll" targets the measured bottlenecks.
        ramp_steps: fade in linearly over this many env steps (0 = off). Counted
            locally -- `env.common_step_counter` does not exist on BaseTask envs and
            reading it through `getattr(..., 0)` silently pins the ramp at zero.
        require_lifted_z: only charge while the box is above this height (0 = always).
            Without it the term shares the degenerate optimum of its predecessor:
            never lift, never saturate.
    """
    actions = env.action_manager.action
    # Mirror JointControlAction.apply_actions, pre-clip.
    torques = (
        env.p_gains * (actions * env.action_scales + env.default_dof_pos - env.simulator.dof_pos)
        - env.d_gains * env.simulator.dof_vel
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
            device=actions.device, dtype=actions.dtype,
        )
        excess = excess * mask
    return torch.sum(torch.square(excess), dim=1)
