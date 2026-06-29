"""X2 humanoid velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  X2_ACTION_SCALE,
  get_x2_robot_cfg,
)
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
