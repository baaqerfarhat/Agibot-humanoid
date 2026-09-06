"""Joint-level faults for the LIBERO Panda: BELOW the controller, in the MuJoCo model.

Every fault in the record so far enters at the Cartesian action interface, above the
operational-space controller. J-PARC's faults (joint lock, limited range, friction) enter
below it, where the fault-to-motion map depends on the arm's configuration through the
Jacobian. This module injects that class so the same law and the same healthy calibration
can be tested against it. Nothing here touches the policy or the controller.

Magnitudes were measured first (2026-09-06, task 0 init 45, 40 steps, hold / +x push):
  torque   j1 5 N.m  ->  +4 cm x drift either way       (OSC has no integral action)
  torque   j3 5 N.m  ->  +9 cm x, +8 cm z
  friction j3 +2.0   ->  0 cm holding, -4 cm x / -3 cm z when moving (a deadband)
  gain     j1 0.5    -> 25 cm: the arm sags; far too violent, use >= 0.8
  lock     j3 +-0.05 ->  0 cm holding, -10 cm x / -7 cm z when moving
So a torque bias acts like a state-dependent constant offset; friction and a lock act like a
motion-dependent loss of effectiveness. Those are the two fault families of the record,
arriving through a different door.

Spec string: "kind:joint:magnitude", e.g. "torque:1:5.0", "friction:3:2.0",
"gain:1:0.8", "lock:3:0.05". Joint index 0..6 on the 7-DOF arm.
"""
from __future__ import annotations

import numpy as np

KINDS = ("torque", "friction", "damping", "gain", "lock")


def parse(spec):
    if not spec:
        return None
    kind, j, mag = spec.split(":")
    assert kind in KINDS, f"unknown joint fault kind {kind}; one of {KINDS}"
    return dict(kind=kind, joint=int(j), mag=float(mag))


class JointFault:
    """Apply a fault to a robosuite env's model after reset; restore before the next episode.

    The env object is cached across episodes by paired_probe.Probe, and model parameters
    persist, so restore() is not optional: without it a friction fault would compound.
    """

    def __init__(self, env, spec):
        self.f = parse(spec)
        rs = env.env if hasattr(env, "env") else env
        self.sim = rs.sim
        m = self.sim.model
        robot = rs.robots[0]
        jn = robot.robot_model.joints
        self.dofs = [m.get_joint_qvel_addr(j) for j in jn]
        self.qadr = [m.get_joint_qpos_addr(j) for j in jn]
        self.jids = [m.joint_name2id(j) for j in jn]
        self.acts = [m.actuator_name2id(a) for a in robot.robot_model.actuators]
        self.base = dict(fl=m.dof_frictionloss.copy(), dp=m.dof_damping.copy(),
                         gp=m.actuator_gainprm.copy(), jr=m.jnt_range.copy())

    def restore(self):
        m = self.sim.model
        m.dof_frictionloss[:] = self.base["fl"]; m.dof_damping[:] = self.base["dp"]
        m.actuator_gainprm[:] = self.base["gp"]; m.jnt_range[:] = self.base["jr"]
        self.sim.data.qfrc_applied[:] = 0.0

    def apply(self):
        """Call after set_init_state. Torque faults are re-applied every step (see step)."""
        self.restore()
        if self.f is None:
            return
        m = self.sim.model; k, j, mag = self.f["kind"], self.f["joint"], self.f["mag"]
        if k == "friction":
            m.dof_frictionloss[self.dofs[j]] = self.base["fl"][self.dofs[j]] + mag
        elif k == "damping":
            m.dof_damping[self.dofs[j]] = self.base["dp"][self.dofs[j]] + mag
        elif k == "gain":
            m.actuator_gainprm[self.acts[j], 0] = mag
        elif k == "lock":
            q = self.sim.data.qpos[self.qadr[j]]
            m.jnt_range[self.jids[j]] = [q - mag, q + mag]

    def step(self, live=True):
        """Call before every env.step. Only the torque bias needs per-step re-application."""
        if self.f is None:
            return
        if self.f["kind"] == "torque":
            self.sim.data.qfrc_applied[self.dofs[self.f["joint"]]] = self.f["mag"] if live else 0.0
