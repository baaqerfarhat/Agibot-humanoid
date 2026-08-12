#!/usr/bin/env python3
"""Verify the torque feed-forward fix against the real hardware logs.

Imports the ACTUAL build_area_cmd from deploy_x2_box_pickup (ROS modules stubbed,
none of them are touched by that function), feeds it every tick of a recorded run,
and checks that the total torque the low level will produce,

    tau = effort + kp*(position - q) + kd*(velocity - dq)

equals the torque holosoma's JointPositionActionTerm would have produced for the
same target and state. Also reports what the old position-only path delivered, so
the gap the fix closes is visible on real data.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CTRL = os.path.join(REPO, "agibot_control_functions")


def stub_ros():
    """Minimal stand-ins so deploy_x2_box_pickup imports off-robot."""
    class _Msg:
        def __init__(self, **kw):
            self.name = ""
            self.position = 0.0
            self.velocity = 0.0
            self.effort = 0.0
            self.stiffness = 0.0
            self.damping = 0.0
            self.joints = []
            for k, v in kw.items():
                setattr(self, k, v)

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    rclpy = mod("rclpy", init=lambda *a, **k: None, ok=lambda: True,
                shutdown=lambda *a, **k: None)
    rclpy.executors = mod("rclpy.executors", SingleThreadedExecutor=object)
    mod("rclpy.node", Node=object)
    class _Any:
        """Swallows any attribute access, e.g. ReliabilityPolicy.BEST_EFFORT."""

        def __getattr__(self, k):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    mod("rclpy.qos", QoSProfile=_Any(), ReliabilityPolicy=_Any(),
        HistoryPolicy=_Any(), DurabilityPolicy=_Any())
    mod("sensor_msgs", )
    mod("sensor_msgs.msg", Imu=_Msg)
    aim = mod("aimdk_msgs")
    aim.msg = mod("aimdk_msgs.msg", JointStateArray=_Msg, JointCommandArray=_Msg,
                  JointCommand=_Msg)
    mod("ruckig", Ruckig=object, InputParameter=object, OutputParameter=object)


stub_ros()
sys.path.insert(0, CTRL)
sys.path.insert(0, HERE)

from deploy_x2_box_pickup import CONTROLLED_AREAS, build_area_cmd  # noqa: E402
from robot_states_control import robot_model  # noqa: E402

from _replay_deploy import Policy, replay  # noqa: E402


class _Reading:
    __slots__ = ("position", "velocity")

    def __init__(self, p, v):
        self.position = p
        self.velocity = v


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    jn = policy.meta["joint_names"]
    kp = np.array(policy.meta["joint_stiffness"], np.float32)
    kd = np.array(policy.meta["joint_damping"], np.float32)
    ascale = np.array(policy.meta["action_scale"], np.float32)
    eff = 4.0 * ascale * kp
    kp_by = dict(zip(jn, kp.tolist()))
    kd_by = dict(zip(jn, kd.tolist()))
    eff_by = dict(zip(jn, eff.tolist()))
    idx = {n: i for i, n in enumerate(jn)}
    lim = {ji.name: (ji.lower_limit, ji.upper_limit)
           for a in CONTROLLED_AREAS for ji in robot_model[a]}

    f = sys.argv[1] if len(sys.argv) > 1 else \
        "20260812_122817_box_pickup_x2_box_policy_v33_iter253000.csv"
    R = replay(os.path.join(HERE, f), policy)
    tgt, meas, vmeas, frame = R["tgt"], R["meas"], R["vmeas"], R["frame"]

    n = len(frame)
    tau_train = np.zeros((n, 31))
    tau_new = np.zeros((n, 31))
    tau_old = np.zeros((n, 31))
    pos_new = np.zeros((n, 31))

    for t in range(n):
        pos_by = {nm: float(tgt[t, idx[nm]]) for nm in jn}
        jmap = {nm: _Reading(float(meas[t, idx[nm]]), float(vmeas[t, idx[nm]])) for nm in jn}
        for area in CONTROLLED_AREAS:
            cmd, _ = build_area_cmd(area, pos_by, kp_by, kd_by, 1.0,
                                    jmap=jmap, eff_by_name=eff_by)
            old, _ = build_area_cmd(area, pos_by, kp_by, kd_by, 1.0)
            for jc, jc_old in zip(cmd.joints, old.joints):
                i = idx[jc.name]
                q, dq = meas[t, i], vmeas[t, i]
                # what the low level will produce for each command
                tau_new[t, i] = jc.effort + jc.stiffness * (jc.position - q) \
                    + jc.damping * (jc.velocity - dq)
                tau_old[t, i] = jc_old.effort + jc_old.stiffness * (jc_old.position - q) \
                    + jc_old.damping * (jc_old.velocity - dq)
                pos_new[t, i] = jc.position
                tau_train[t, i] = np.clip(kp[i] * (tgt[t, i] - q) - kd[i] * dq,
                                          -eff[i], eff[i])

    print(f"run: {f}   ticks {n}")
    print()
    err = np.abs(tau_new - tau_train)
    print(f"[1] torque with the fix vs training torque: max |err| = {err.max():.2e} Nm "
          f"({'MATCH' if err.max() < 1e-3 else 'MISMATCH'})")
    err_old = np.abs(tau_old - tau_train)
    print(f"[2] torque with the OLD path vs training:   max |err| = {err_old.max():8.2f} Nm, "
          f"mean {err_old.mean():.3f} Nm")
    print()

    # the position field must still respect the mechanical range
    bad = 0
    for nm, i in idx.items():
        lo, hi = lim[nm]
        bad += int(((pos_new[:, i] < lo - 1e-9) | (pos_new[:, i] > hi + 1e-9)).sum())
    print(f"[3] commanded positions outside the mechanical range: {bad} "
          f"({'none, firmware-safe' if bad == 0 else 'PROBLEM'})")

    print()
    print("[4] per-joint torque recovered by the fix (mean |tau|, Nm):")
    print(f"      {'joint':28s}{'training':>10s}{'old':>10s}{'fixed':>10s}"
          f"{'ff mean':>10s}{'ff max':>9s}")
    gap = np.abs(tau_train - tau_old).mean(axis=0)
    ffm = tau_new - (kp * (pos_new - meas) - kd * vmeas)
    for i in np.argsort(-gap)[:10]:
        if gap[i] < 0.05:
            break
        print(f"      {jn[i]:28s}{np.abs(tau_train[:,i]).mean():10.2f}"
              f"{np.abs(tau_old[:,i]).mean():10.2f}{np.abs(tau_new[:,i]).mean():10.2f}"
              f"{np.abs(ffm[:,i]).mean():10.2f}{np.abs(ffm[:,i]).max():9.2f}")
    print()
    print(f"[5] whole-body mean |tau|: training {np.abs(tau_train).mean():.3f}  "
          f"old {np.abs(tau_old).mean():.3f}  fixed {np.abs(tau_new).mean():.3f} Nm")
    print(f"    feed-forward magnitude: mean {np.abs(ffm).mean():.3f}  "
          f"max {np.abs(ffm).max():.2f} Nm  (all within the per-joint effort limit: "
          f"{bool((np.abs(tau_new) <= eff + 1e-6).all())})")


if __name__ == "__main__":
    main()
