"""G1 policy parameters — transcribed from gear_sonic_deploy policy_parameters.hpp.

Same formulas as the C++ (stiffness = armature * (2*pi*10Hz)^2; damping = 2*ratio*armature*w;
action_scale = 0.25 * effort_limit / stiffness; joint target = action*scale + default_angle).
Joint order = the deploy/lowstate motor order (29 DoF, hands excluded).
"""
import math

import numpy as np

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
EFF = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
STIFF = {k: a * NATURAL_FREQ ** 2 for k, a in ARM.items()}
DAMP = {k: 2.0 * DAMPING_RATIO * a * NATURAL_FREQ for k, a in ARM.items()}

# (name, motor_type, kp_mult)  — kp_mult 2.0 on ankles/waist-roll/pitch per the C++
_J = [
    ("left_hip_pitch", "7520_22", 1), ("left_hip_roll", "7520_22", 1),
    ("left_hip_yaw", "7520_14", 1), ("left_knee", "7520_22", 1),
    ("left_ankle_pitch", "5020", 2), ("left_ankle_roll", "5020", 2),
    ("right_hip_pitch", "7520_22", 1), ("right_hip_roll", "7520_22", 1),
    ("right_hip_yaw", "7520_14", 1), ("right_knee", "7520_22", 1),
    ("right_ankle_pitch", "5020", 2), ("right_ankle_roll", "5020", 2),
    ("waist_yaw", "7520_14", 1), ("waist_roll", "5020", 2), ("waist_pitch", "5020", 2),
    ("left_shoulder_pitch", "5020", 1), ("left_shoulder_roll", "5020", 1),
    ("left_shoulder_yaw", "5020", 1), ("left_elbow", "5020", 1),
    ("left_wrist_roll", "5020", 1), ("left_wrist_pitch", "4010", 1),
    ("left_wrist_yaw", "4010", 1),
    ("right_shoulder_pitch", "5020", 1), ("right_shoulder_roll", "5020", 1),
    ("right_shoulder_yaw", "5020", 1), ("right_elbow", "5020", 1),
    ("right_wrist_roll", "5020", 1), ("right_wrist_pitch", "4010", 1),
    ("right_wrist_yaw", "4010", 1),
]

JOINT_NAMES = [n for n, _, _ in _J]
ACTION_SCALE = np.array([0.25 * EFF[t] / STIFF[t] for _, t, _ in _J])
KPS = np.array([m * STIFF[t] for _, t, m in _J])
KDS = np.array([m * DAMP[t] for _, t, m in _J])
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])
assert len(JOINT_NAMES) == 29 and DEFAULT_ANGLES.shape == (29,)
