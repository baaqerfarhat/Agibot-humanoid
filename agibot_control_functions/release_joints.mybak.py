#!/usr/bin/env python3
"""Release (limp) all X2 joints so the robot stops holding its last commanded pose.

After a deploy run exits, the motor drivers keep holding the final position with
full PD gains. This script publishes zero-stiffness / zero-damping commands to
every joint area, so torque = kp*(target-q) + kd*(0-dq) + ff = 0 -> the joints
go compliant and stop fighting.

    #####################################  SAFETY  #####################################
    #  With kp=kd=0 the joints go LIMP. Only run this while the robot is SUSPENDED
    #  or otherwise SUPPORTED -- if it is standing on its own it WILL collapse.
    #  Keep the e-stop in hand.
    ####################################################################################

Run INSIDE the robot's ROS 2 env (same as the deploy scripts), AFTER the deploy
has exited (or in a second terminal), while mc is stopped on 10.0.1.40:

    python3 release_joints.py                 # full limp (kp=kd=0)
    python3 release_joints.py --kd 3.0        # keep a little damping (gentler, less floppy)
    python3 release_joints.py --hold-seconds 1.0
"""

from __future__ import annotations

import argparse
import threading
import time

import rclpy

from robot_states_control import (
    JointArea,
    RobotStateClient,
    WholeBodyCommander,
    robot_model,
)
from aimdk_msgs.msg import JointCommand, JointCommandArray

CONTROLLED_AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)


def build_release_cmd(area, jmap, kp: float, kd: float) -> JointCommandArray:
    cmd = JointCommandArray()
    for ji in robot_model[area]:
        jc = JointCommand()
        jc.name = ji.name
        # Target = current position (irrelevant when kp=0, but sane if kd/kp>0).
        jc.position = float(jmap[ji.name].position) if ji.name in jmap else 0.0
        jc.velocity = 0.0
        jc.effort = 0.0
        jc.stiffness = float(kp)
        jc.damping = float(kd)
        cmd.joints.append(jc)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kp", type=float, default=0.0,
                    help="Stiffness for all joints (0 = fully limp).")
    ap.add_argument("--kd", type=float, default=0.0,
                    help="Damping for all joints (small value = gentler, less floppy).")
    ap.add_argument("--hold-seconds", type=float, default=0.8,
                    help="How long to keep publishing the release command.")
    ap.add_argument("--rate", type=float, default=50.0, help="Publish rate (Hz).")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  RELEASE JOINTS: kp={args.kp}  kd={args.kd}  for {args.hold_seconds}s")
    print("  >>> joints go COMPLIANT -- robot must be SUSPENDED / SUPPORTED <<<")
    print("=" * 70)

    rclpy.init()
    client = RobotStateClient()
    commander = WholeBodyCommander()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not client.wait_ready(timeout_sec=10.0, required_imus=["torso"]):
        print("[ERROR] state topics not ready.")
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown(); return

    _imus, head, waist, arm, leg = client.get_robot_states()
    jmap = {jr.name: jr for jr in (head + waist + arm + leg)}

    dt = 1.0 / float(args.rate)
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.hold_seconds and rclpy.ok():
            for area in CONTROLLED_AREAS:
                commander.publish(area, build_release_cmd(area, jmap, args.kp, args.kd))
            time.sleep(dt)
    finally:
        # One last publish so the sticky command left in the drivers is the limp one.
        for area in CONTROLLED_AREAS:
            commander.publish(area, build_release_cmd(area, jmap, args.kp, args.kd))
        print("[release] done -- joints released (limp).")
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
