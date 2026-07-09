#!/usr/bin/env python3
"""Sim-to-real observation diagnostic.

Reads the LIVE robot state, decomposes the exact observation we feed the policy,
and shows what the policy outputs. Goal: find which obs term is out-of-distribution
(why the policy freezes into a crouch instead of walking).

Run with the robot held at / near its DEFAULT pose and standing upright.
Compares against what sim expects at default pose + zero command:
    base_ang_vel      ~ [0, 0, 0]
    projected_gravity ~ [0, 0, -1]
    joint_pos (q-def) ~ 0  for every joint
    joint_vel         ~ 0
    action            ~ 0  (policy should say 'stay') when cmd = 0
"""
import json
import threading
import time

import numpy as np
import rclpy

from robot_states_control import RobotStateClient
from deploy_x2_walk import NumpyPolicy, ObservationBuilder, projected_gravity

POLICY = "policies/x2_policy.npz"
BASE_IMU = "torso"


def main():
    policy = NumpyPolicy(POLICY)
    meta = policy.meta
    joint_names = meta["joint_names"]
    default = np.array(meta["default_joint_pos"], np.float32)
    action_scale = np.array(meta["action_scale"], np.float32)
    if action_scale.ndim == 0:
        action_scale = np.full(len(joint_names), float(action_scale), np.float32)

    rclpy.init()
    client = RobotStateClient()
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(client)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not client.wait_ready(timeout_sec=10.0):
        print("[ERROR] state not ready"); return

    imus, head, waist, arm, leg = client.get_robot_states()
    jmap = {jr.name: jr for jr in (head + waist + arm + leg)}

    print("=" * 80)
    print("IMU projected_gravity (upright should be ~[0,0,-1]):")
    for src in ("torso", "chest"):
        g = projected_gravity(imus[src].quat)
        print(f"   {src:6s}: {np.round(g,3)}   ang_vel={np.round(np.array(imus[src].ang_vel),3)}")

    print("=" * 80)
    print(f"{'joint':<26} {'q(meas)':>9} {'default':>9} {'q-def':>9} {'dq':>9}")
    print("-" * 80)
    big = []
    for i, n in enumerate(joint_names):
        q = jmap[n].position
        dq = jmap[n].velocity
        d = q - default[i]
        flag = "  <== far from default" if abs(d) > 0.25 else ""
        if abs(d) > 0.25:
            big.append((n, d))
        print(f"{n:<26} {q:>9.3f} {default[i]:>9.3f} {d:>9.3f} {dq:>9.3f}{flag}")

    print("=" * 80)
    print("Joints far (>0.25 rad) from sim default pose:", big or "(none) -> good")

    # Build the obs the policy actually sees, for cmd = 0 and cmd = forward.
    for cmd in ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0]):
        ob = ObservationBuilder(meta, base_imu=BASE_IMU)
        obs = ob.build(imus, jmap, cmd)
        act = policy(obs).reshape(-1)
        tgt = act * action_scale + default
        print("-" * 80)
        print(f"command = {cmd}")
        print(f"  |action|: mean={np.mean(np.abs(act)):.3f}  max={np.max(np.abs(act)):.3f}"
              f"   (cmd=0 should be SMALL if obs is in-distribution)")
        for n in ("left_knee_joint", "right_knee_joint",
                  "left_hip_pitch_joint", "right_hip_pitch_joint"):
            i = joint_names.index(n)
            print(f"    {n:<24} action={act[i]:+.3f}  target={tgt[i]:+.3f}  "
                  f"meas_q={jmap[n].position:+.3f}")

    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
