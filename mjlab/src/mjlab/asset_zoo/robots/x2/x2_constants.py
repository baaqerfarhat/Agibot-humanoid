"""X2 humanoid constants.

The X2 is a 31-DoF full-size humanoid: 12 leg joints (6 per leg), 3 waist joints,
2 head joints, and 14 arm joints (7 per arm). The raw MJCF (``xmls/x2.xml``) ships
torque ``<motor>`` actuators; those are stripped here and replaced with mjlab-native
position (PD) actuators created from this config.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

X2_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "x2" / "xmls" / "x2.xml"
assert X2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(X2_XML))


##
# Actuator config.
##
#
# Effort limits come from the motor ``ctrlrange``/``actuatorfrcrange`` in the original
# X2 MJCF. PD gains (stiffness/damping) are hand-tuned per joint group: high stiffness
# on the legs and waist for support, lower on arms/head. The implicitfast integrator
# keeps these stiff gains numerically stable at the 0.005 s physics timestep.

X2_ACTUATOR_HIP_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
  ),
  effort_limit=118.0,
  stiffness=200.0,
  damping=5.0,
)
X2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  effort_limit=36.0,
  stiffness=60.0,
  damping=2.0,
)
X2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  effort_limit=24.0,
  stiffness=40.0,
  damping=1.5,
)
X2_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"),
  effort_limit=48.0,
  stiffness=200.0,
  damping=5.0,
)
X2_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_yaw_joint", "head_pitch_joint"),
  effort_limit=2.6,
  stiffness=5.0,
  damping=0.5,
)
X2_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"),
  effort_limit=36.0,
  stiffness=60.0,
  damping=2.0,
)
X2_ACTUATOR_SHOULDER_YAW_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint", ".*_elbow_joint"),
  effort_limit=24.0,
  stiffness=40.0,
  damping=2.0,
)
X2_ACTUATOR_WRIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_yaw_joint",),
  effort_limit=24.0,
  stiffness=20.0,
  damping=1.0,
)
X2_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
  effort_limit=2.2,
  stiffness=10.0,
  damping=0.5,
)

##
# Keyframe config.
##
#
# Standing pose with slightly bent knees. Base height (0.69 m) was computed via forward
# kinematics so the feet rest just above the ground plane.

STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.69),
  joint_pos={
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_joint": 0.4,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
    ".*_elbow_joint": -0.3,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Feet-only collisions: only the foot geoms collide with the ground (condim=3 with
# friction). All other geoms (legs, torso, arms) have collisions disabled. This is the
# most robust setup for a first flat-ground walking policy and avoids self-collision
# instabilities. Switch to FULL_COLLISION once you need self-collision awareness.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-9]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# All collision geoms enabled. Feet get condim=3; everything else condim=1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-9]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-9]_collision$": 1},
  friction={r"^(left|right)_foot[1-9]_collision$": (0.6,)},
)

##
# Final config.
##

X2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X2_ACTUATOR_HIP_KNEE,
    X2_ACTUATOR_ANKLE_PITCH,
    X2_ACTUATOR_ANKLE_ROLL,
    X2_ACTUATOR_WAIST,
    X2_ACTUATOR_HEAD,
    X2_ACTUATOR_SHOULDER,
    X2_ACTUATOR_SHOULDER_YAW_ELBOW,
    X2_ACTUATOR_WRIST_YAW,
    X2_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_x2_robot_cfg() -> EntityCfg:
  """Get a fresh X2 robot configuration instance."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=X2_ARTICULATION,
  )


# Per-joint action scale for the position action term, computed as
# 0.25 * effort_limit / stiffness (same heuristic mjlab uses for the G1).
X2_ACTION_SCALE: dict[str, float] = {}
for a in X2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  assert a.effort_limit is not None
  for n in a.target_names_expr:
    X2_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x2_robot_cfg())
  viewer.launch(robot.spec.compile())
