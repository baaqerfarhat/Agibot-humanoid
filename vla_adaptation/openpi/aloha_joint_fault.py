"""Physical torque injection for gym_aloha, separate from command offsets.

Arm indices 0..11 enumerate six left joints followed by six right joints.
Their policy coordinates are 0..5 and 7..12; grippers are excluded. This
experiment wrapper never supplies torque truth to the adaptation routine.
"""
from __future__ import annotations

import numpy as np

JOINT_NAMES = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def joint_metadata(physics, joint):
    if not isinstance(joint, (int, np.integer)) or not 0 <= joint < 12:
        raise ValueError("ALOHA physical arm joint must be an integer in 0..11")
    side, axis = divmod(int(joint), 6)
    name = f"vx300s_{'left' if side == 0 else 'right'}/{JOINT_NAMES[axis]}"
    model = physics.model
    jid = model.name2id(name, "joint")
    actuator_ids = np.flatnonzero(np.asarray(model.actuator_trnid)[:, 0] == jid)
    if len(actuator_ids) != 1:
        raise ValueError(f"expected one position actuator for {name}")
    aid = int(actuator_ids[0])
    gain = float(model.actuator_gainprm[aid, 0] * model.actuator_gear[aid, 0])
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("expected positive position-actuator command-to-torque gain")
    return dict(arm_joint=int(joint), joint_name=name, joint_id=int(jid),
                dof=int(model.jnt_dofadr[jid]), qpos=int(model.jnt_qposadr[jid]),
                command_coordinate=axis + 7 * side, actuator=aid,
                command_to_torque_gain=gain,
                force_limited=bool(model.actuator_forcelimited[aid]),
                force_range=np.asarray(model.actuator_forcerange[aid]).tolist())


class TorqueEnvironment:
    """Delegate observations/commands unchanged; add torque before each physics step."""

    def __init__(self, env, joint, magnitude):
        if not np.isfinite(magnitude):
            raise ValueError("torque magnitude must be finite")
        self.env = env
        self.physics = env.unwrapped._env.physics
        self.metadata = joint_metadata(self.physics, joint)
        self.magnitude = float(magnitude)
        self._dof = self.metadata["dof"]
        self._base = float(self.physics.data.qfrc_applied[self._dof])
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def restore(self):
        # Preserve external forces on all other coordinates.
        self.physics.data.qfrc_applied[self._dof] = self._base

    def reset(self, *args, **kwargs):
        self.restore()
        observation = self.env.reset(*args, **kwargs)
        self._base = float(self.physics.data.qfrc_applied[self._dof])
        self.steps = 0
        return observation

    def step(self, command):
        self.physics.data.qfrc_applied[self._dof] = self._base + self.magnitude
        self.steps += 1
        return self.env.step(command)


class torque_fault:
    """Temporarily wrap an Aloha client; always remove injected force on exit."""

    def __init__(self, client, joint, magnitude):
        self.client = client
        self.original = client.env
        self.wrapper = TorqueEnvironment(client.env, joint, magnitude)

    def __enter__(self):
        self.client.env = self.wrapper
        return self.wrapper

    def __exit__(self, *exc):
        self.wrapper.restore()
        self.client.env = self.original
