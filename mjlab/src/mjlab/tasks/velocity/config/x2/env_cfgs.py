"""X2 humanoid velocity environment configurations."""

from dataclasses import replace as dc_replace

import mujoco

from mjlab.asset_zoo.robots import (
  X2_ACTION_SCALE,
  get_x2_robot_cfg,
)
from mjlab.asset_zoo.robots.x2.x2_constants import get_spec
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# X2 foot collision geom names (6 spheres per foot, defined in xmls/x2.xml).
FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 7)
)
SITE_NAMES = ("left_foot", "right_foot")

# Per-joint posture tracking tolerances. Looser std => more freedom for that joint.
STD_STANDING = {".*": 0.05}
STD_WALKING = {
  # Lower body.
  r".*hip_pitch.*": 0.3,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  # Waist.
  r".*waist_yaw.*": 0.2,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.1,
  # Head.
  r".*head.*": 0.1,
  # Arms.
  r".*shoulder_pitch.*": 0.15,
  r".*shoulder_roll.*": 0.15,
  r".*shoulder_yaw.*": 0.1,
  r".*elbow.*": 0.15,
  r".*wrist.*": 0.3,
}
STD_RUNNING = {
  # Lower body.
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.2,
  r".*hip_yaw.*": 0.2,
  r".*knee.*": 0.6,
  r".*ankle_pitch.*": 0.35,
  r".*ankle_roll.*": 0.15,
  # Waist.
  r".*waist_yaw.*": 0.3,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.2,
  # Head.
  r".*head.*": 0.15,
  # Arms.
  r".*shoulder_pitch.*": 0.5,
  r".*shoulder_roll.*": 0.2,
  r".*shoulder_yaw.*": 0.15,
  r".*elbow.*": 0.35,
  r".*wrist.*": 0.3,
}


def x2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create X2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_x2_robot_cfg()}

  # Set raycast sensor frame to the X2 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.05, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = X2_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["std_standing"] = STD_STANDING
  cfg.rewards["pose"].params["std_walking"] = STD_WALKING
  cfg.rewards["pose"].params["std_running"] = STD_RUNNING

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def x2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create X2 flat terrain velocity configuration."""
  cfg = x2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan on flat ground).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def x2_flat_deploy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """X2 flat velocity config for real-robot deployment.

  Removes ``base_lin_vel`` from the *actor* observation group. The real robot has no
  way to measure pelvis linear velocity, so the deployed policy must not depend on it.
  It remains in the *critic* group (privileged, training-only), keeping value estimation
  informative while the policy stays deployable from on-board sensing alone.
  """
  cfg = x2_flat_env_cfg(play=play)
  del cfg.observations["actor"].terms["base_lin_vel"]
  return cfg


# =============================================================================
# Box-carry fine-tune (hybrid pickup->carry->setdown controller, walk segment).
# =============================================================================
#
# Upper-body pose at the carry "hold" frame (frame 211 of the in-place
# pickup/hold/setdown clip sub3_largebox_003_mj_w_obj.npz). During the hybrid
# carry the WBT policy holds the waist/arms/head essentially static at this
# pose while the walking policy drives the legs, so here we pin those joints
# (action scale 0 -> PD holds them at default) and make this pose the default,
# which also matches how deployment masks upper-body observations (offset 0).
CARRY_UPPER_POSE = {
  "waist_yaw_joint": 0.3815,
  "waist_pitch_joint": 0.1503,
  "waist_roll_joint": 0.0839,
  "left_shoulder_pitch_joint": -0.2301,
  "left_shoulder_roll_joint": -0.0048,
  "left_shoulder_yaw_joint": -0.2014,
  "left_elbow_joint": -0.6058,
  "left_wrist_yaw_joint": -0.8059,
  "left_wrist_pitch_joint": -0.5580,
  "left_wrist_roll_joint": 0.7240,
  "right_shoulder_pitch_joint": -0.4247,
  "right_shoulder_roll_joint": -0.1232,
  "right_shoulder_yaw_joint": -0.2231,
  "right_elbow_joint": -0.4472,
  "right_wrist_yaw_joint": -0.1160,
  "right_wrist_pitch_joint": -0.5574,
  "right_wrist_roll_joint": 1.4605,
  "head_yaw_joint": 0.0,
  "head_pitch_joint": 0.0,
}

# Box center relative to torso_link at the hold frame (measured via FK).
CARRY_BOX_POS = (0.331, 0.004, 0.09)
CARRY_BOX_HALF = 0.225  # 45 cm cube.


def _get_x2_carry_spec() -> mujoco.MjSpec:
  """X2 spec with a payload box welded to the chest at the carry pose."""
  spec = get_spec()
  torso = spec.body("torso_link")
  box = torso.add_body(name="carry_box", pos=CARRY_BOX_POS)
  box.add_geom(
    name="carry_box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(CARRY_BOX_HALF, CARRY_BOX_HALF, CARRY_BOX_HALF),
    mass=1.0,  # Nominal; randomized per-env by the payload_mass event.
    contype=0,
    conaffinity=0,
    rgba=(0.82, 0.71, 0.55, 0.7),
  )
  return spec


def x2_flat_carry_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Fine-tune the deploy walking policy while carrying a chest payload.

  Same observation/action interface as ``x2_flat_deploy_env_cfg`` (so it can be
  warm-started from its checkpoints and dropped into the same deploy pipeline),
  with:
    * a box payload (0.3--3.0 kg, COM jittered) welded to the chest,
    * waist/arms/head pinned at the carry pose (action scale 0),
    * command ranges narrowed to carry speeds,
    * gentler pushes (carrying a box, no arm recovery available).
  """
  cfg = x2_flat_deploy_env_cfg(play=play)

  # Swap in the payload spec and make the carry pose the default upper body.
  robot_cfg = cfg.scene.entities["robot"]
  init = robot_cfg.init_state
  assert init.joint_pos is not None
  cfg.scene.entities["robot"] = dc_replace(
    robot_cfg,
    spec_fn=_get_x2_carry_spec,
    init_state=dc_replace(init, joint_pos={**init.joint_pos, **CARRY_UPPER_POSE}),
  )

  # Pin the upper body: zero action scale => PD holds the default (carry) pose.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  scale = dict(X2_ACTION_SCALE)
  for pattern in (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_roll_joint",
  ):
    scale[pattern] = 0.0
  joint_pos_action.scale = scale

  # Payload randomization: mass 0.3--3.0 kg, COM jittered +-5 cm.
  cfg.events["payload_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("carry_box",)),
      "operation": "abs",
      "ranges": (0.3, 3.0),
    },
  )
  cfg.events["payload_com"] = EventTermCfg(
    mode="startup",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("carry_box",)),
      "operation": "add",
      "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
    },
  )

  # Carry-speed command ranges (and don't let the curriculum re-widen them).
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.3, 0.8)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)
  cfg.curriculum.pop("command_vel", None)

  # Gentler pushes: no arm swing available for recovery while carrying.
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.3, 0.3),
      "y": (-0.3, 0.3),
      "z": (-0.2, 0.2),
      "roll": (-0.3, 0.3),
      "pitch": (-0.3, 0.3),
      "yaw": (-0.5, 0.5),
    }

  return cfg
