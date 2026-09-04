"""X2 whole-body tracking with the carried box.

The stock tracking task tracks a robot against a clip. This adds the box: a free
body in the scene, seeded from the clip's `object_*` arrays at every reset, and
rewarded for staying where the clip says it should be.

Two deliberate choices.

The actor stays blind to the box, exactly as it is in holosoma and on the robot. The
box enters only through physics -- its weight in the hands -- and through the reward.
Giving the policy box state it will not have at deploy time is the kind of mismatch
that produced the failures this port is meant to fix.

Everything is built on the no-state-estimation observation set. The stock tracking
actor reads `base_lin_vel` off an IMU sensor and `motion_anchor_pos_b`, neither of
which the robot can supply honestly; a policy trained on them cannot be deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots import X2_ACTION_SCALE, get_x2_robot_cfg
from mjlab.asset_zoo.robots.x2.x2_constants import get_spec as get_x2_spec
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# Measured off largebox.obj, the mesh holosoma spawns for this clip. The mesh is not
# centred on its body origin, and the offset matters: get it wrong and the box either
# floats or starts inside the floor and is kicked out on the first step.
BOX_SIZE = (0.4712, 0.4587, 0.4079)
BOX_OFFSET = (0.0015, -0.0007, 0.0058)
BOX_MASS = 1.0  # the real box is about a kilo
# Height of the body origin when the box rests on the floor: half the depth, less the
# mesh offset. The clip seats the box about 15 mm under the floor, so spawning it at
# the reference verbatim starts it interpenetrating and the solver kicks it out on the
# first step. The spawn is clamped to this instead. The reference itself is left alone
# -- it stays the tracking target, and 15 mm is nothing against a 0.2 m reward scale.
BOX_REST_Z = BOX_SIZE[2] / 2 - BOX_OFFSET[2]
# Palm contact sphere, copied from holosoma's halfspherehand asset.
PALM_RADIUS = 0.05
PALM_OFFSET = (0.01, 0.0, -0.10)
MOTION_FILE = "/home/baaqer/baaqer_ws/mjlab/motions/x2_box_walk_feasible.npz"

TRACKED_BODIES = (
  "pelvis",  # must stay first: the command takes the root from body_names[0]
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
)
HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")


def get_x2_box_robot_spec() -> mujoco.MjSpec:
  """The X2 spec with a contact sphere at each palm.

  The stock X2 collides on the feet and nothing else, which is right for walking and
  useless here: with no geometry on the hands there is nothing for the box to rest
  against and the robot cannot pick it up at all. Holosoma solves this the same way,
  swapping the detailed hand mesh for a rounded palm sphere
  (`x2_31dof_w_object_halfspherehand.urdf`); these are that sphere, same radius and
  same offset off the wrist.
  """
  spec = get_x2_spec()
  for side in ("left", "right"):
    body = spec.body(f"{side}_wrist_roll_link")
    if body is None:
      raise RuntimeError(f"{side}_wrist_roll_link missing from the X2 spec")
    body.add_geom(
      name=f"{side}_palm_collision",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(PALM_RADIUS, 0.0, 0.0),
      pos=PALM_OFFSET,
      mass=0.05,
      rgba=(0.9, 0.4, 0.2, 0.6),
      contype=0,
      conaffinity=1,
    )
  return spec


def get_x2_box_robot_cfg() -> EntityCfg:
  cfg = get_x2_robot_cfg()
  return replace(
    cfg,
    spec_fn=get_x2_box_robot_spec,
    collisions=(
      CollisionCfg(
        geom_names_expr=(
          r"^(left|right)_foot[1-9]_collision$",
          r"^(left|right)_palm_collision$",
        ),
        contype=0,
        conaffinity=1,
        condim={
          r"^(left|right)_foot[1-9]_collision$": 3,
          r"^(left|right)_palm_collision$": 4,  # palms need torsional friction to grip
        },
        priority={r"^(left|right)_foot[1-9]_collision$": 1},
        friction={
          r"^(left|right)_foot[1-9]_collision$": (0.6,),
          r"^(left|right)_palm_collision$": (1.2, 0.05, 0.005),
        },
      ),
    ),
  )


def get_largebox_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=tuple(s / 2 for s in BOX_SIZE),  # MuJoCo box size is a half-extent
    pos=BOX_OFFSET,
    mass=BOX_MASS,
    rgba=(0.72, 0.52, 0.30, 1.0),
    friction=(1.0, 0.02, 0.001),
    condim=4,
  )
  return spec


##
# Command: the stock motion command, plus the box.
##


class BoxMotionCommand(MotionCommand):
  """Motion command that also drives the reference box."""

  cfg: BoxMotionCommandCfg

  def __init__(self, cfg: BoxMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.box = env.scene[cfg.object_entity_name]
    d = np.load(cfg.motion_file)
    for key in ("object_pos_w", "object_quat_w"):
      if key not in d:
        raise KeyError(
          f"{cfg.motion_file} has no {key}; convert the clip with "
          "Agibot-humanoid/box_pickup/convert_clip_to_mjlab.py"
        )

    def t(name, default_shape):
      if name in d:
        return torch.tensor(d[name], dtype=torch.float32, device=self.device)
      return torch.zeros(default_shape, dtype=torch.float32, device=self.device)

    n = self.motion.time_step_total
    self.object_pos = t("object_pos_w", (n, 3))
    self.object_quat = t("object_quat_w", (n, 4))  # wxyz
    self.object_lin_vel = t("object_lin_vel_w", (n, 3))
    self.object_ang_vel = t("object_ang_vel_w", (n, 3))

    self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_object_ori"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def object_pos_w(self) -> torch.Tensor:
    return self.object_pos[self.time_steps] + self._env.scene.env_origins

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.object_quat[self.time_steps]

  @property
  def robot_object_pos_w(self) -> torch.Tensor:
    return self.box.data.root_link_pos_w

  @property
  def robot_object_quat_w(self) -> torch.Tensor:
    return self.box.data.root_link_quat_w

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    # The robot is teleported to the reference at reset, so the box has to follow it
    # or the hands close on empty air.
    steps = self.time_steps[env_ids]
    pos = self.object_pos[steps] + self._env.scene.env_origins[env_ids]
    pos[:, 2] = pos[:, 2].clamp(min=BOX_REST_Z)
    state = torch.cat(
      [
        pos,
        self.object_quat[steps],
        self.object_lin_vel[steps],
        self.object_ang_vel[steps],
      ],
      dim=-1,
    )
    self.box.write_root_state_to_sim(state, env_ids=env_ids)
    self.box.reset(env_ids=env_ids)

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["error_object_pos"] = torch.norm(
      self.object_pos_w - self.robot_object_pos_w, dim=-1
    )
    self.metrics["error_object_ori"] = _quat_error(
      self.object_quat_w, self.robot_object_quat_w
    )


@dataclass(kw_only=True)
class BoxMotionCommandCfg(MotionCommandCfg):
  object_entity_name: str = "box"

  def build(self, env: ManagerBasedRlEnv) -> BoxMotionCommand:
    return BoxMotionCommand(self, env)


##
# Rewards and terminations for the box.
##


def _quat_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Angle between two wxyz quaternions, in radians."""
  dot = (a * b).sum(-1).abs().clamp(max=1.0)
  return 2.0 * torch.acos(dot)


def object_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  err = torch.norm(cmd.object_pos_w - cmd.robot_object_pos_w, dim=-1)
  return torch.exp(-(err**2) / std**2)


def object_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  err = _quat_error(cmd.object_quat_w, cmd.robot_object_quat_w)
  return torch.exp(-(err**2) / std**2)


def hands_to_object_distance_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Reward both palms for being near the box.

  Without this the policy can score the tracking terms while never closing on the
  box, which is most of what went wrong in the Isaac runs.
  """
  cmd = env.command_manager.get_term(command_name)
  robot = env.scene[asset_cfg.name]
  hands = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
  box = cmd.robot_object_pos_w.unsqueeze(1)
  d = torch.norm(hands - box, dim=-1).sum(dim=-1)
  return torch.exp(-(d**2) / std**2)


def bad_object_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  cmd = env.command_manager.get_term(command_name)
  return torch.norm(cmd.object_pos_w - cmd.robot_object_pos_w, dim=-1) > threshold


##
# Env config.
##


def x2_box_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {
    "robot": get_x2_box_robot_cfg(),
    "box": EntityCfg(spec_fn=get_largebox_spec),
  }
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ACTION_SCALE

  # Swap the stock motion command for the box-aware one, carrying the tuning over.
  base = cfg.commands["motion"]
  assert isinstance(base, MotionCommandCfg)
  cfg.commands["motion"] = BoxMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=base.resampling_time_range,
    debug_vis=base.debug_vis,
    pose_range=base.pose_range,
    velocity_range=base.velocity_range,
    joint_position_range=base.joint_position_range,
    motion_file=MOTION_FILE,
    anchor_body_name="torso_link",
    body_names=TRACKED_BODIES,
    object_entity_name="box",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(left|right)_foot[1-6]_collision$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
  )

  cfg.rewards["object_pos"] = RewardTermCfg(
    func=object_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.2},
  )
  cfg.rewards["object_ori"] = RewardTermCfg(
    func=object_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.5},
  )
  cfg.rewards["hands_to_object"] = RewardTermCfg(
    func=hands_to_object_distance_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.6,
      "asset_cfg": SceneEntityCfg("robot", body_names=HANDS),
    },
  )
  cfg.terminations["object_pos"] = TerminationTermCfg(
    func=bad_object_pos,
    params={"command_name": "motion", "threshold": 0.5},
  )

  cfg.viewer.body_name = "torso_link"
  # The clip is 11.82 s; the stock 10 s would cut the set-down off.
  cfg.episode_length_s = 13.0
  # A free box adds contacts the stock budget does not allow for.
  cfg.sim.nconmax = 80
  cfg.sim.njmax = 400

  # No state estimation: the robot cannot supply base_lin_vel or the anchor position
  # honestly, so a policy that leans on them cannot be deployed.
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ("motion_anchor_pos_b", "base_lin_vel")
    },
    concatenate_terms=True,
    enable_corruption=True,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, BoxMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
