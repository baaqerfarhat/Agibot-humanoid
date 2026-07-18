#!/usr/bin/env python3
"""Release all X2 joints so the robot can be moved by hand.

Run this after a deploy script (walking or box pickup) leaves the joints held
stiff at their last commanded position. It ramps the PD gains of every joint
(legs, waist, arms, head) down to zero over a few seconds, then keeps
publishing zero-gain commands for a short window so the drives stay compliant.

    python3 release_joints.py                 # ramp to fully limp over 3 s
    python3 release_joints.py --damping 2.0   # leave light damping (viscous
                                              # drag) so limbs don't swing free
    python3 release_joints.py --now           # no ramp, release immediately

################################  SAFETY  ################################
#  ZERO GAINS = THE ROBOT GOES LIMP. If it is standing unsupported it
#  WILL collapse. Only run this with the robot suspended on the gantry,
#  seated, or held. The script asks for confirmation before releasing.
###########################################################################
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

AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)
RATE_HZ = 50.0


def build_cmd(area, pos_by_name, gain_alpha, kp0, kd_floor):
    """Command current position with gains scaled by gain_alpha (1 -> 0)."""
    cmd = JointCommandArray()
    for ji in robot_model[area]:
        jc = JointCommand()
        jc.name = ji.name
        # Track the measured position so there is no residual spring torque
        # pulling toward an old target while the gains wind down.
        jc.position = float(pos_by_name[ji.name])
        jc.velocity = 0.0
        jc.effort = 0.0
        jc.stiffness = float(kp0 * gain_alpha)
        jc.damping = float(max(kd_floor, kp0 * 0.05 * gain_alpha))
        cmd.joints.append(jc)
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp-seconds", type=float, default=3.0,
                    help="Time to ramp gains from starting value to zero.")
    ap.add_argument("--start-kp", type=float, default=30.0,
                    help="Stiffness the ramp starts from (a modest hold, not "
                         "the full task gains).")
    ap.add_argument("--damping", type=float, default=0.0,
                    help="Residual damping to KEEP after release (0 = fully "
                         "limp; 1-3 = gentle viscous drag, nice for posing arms).")
    ap.add_argument("--hold-seconds", type=float, default=5.0,
                    help="Keep publishing released commands this long, then exit.")
    ap.add_argument("--now", action="store_true",
                    help="Skip the ramp: publish zero gains immediately.")
    args = ap.parse_args()

    print("=" * 70)
    print("  RELEASE JOINTS: the robot will go LIMP and cannot hold itself up.")
    print("  Make sure it is suspended, seated, or supported by hand.")
    print("=" * 70)
    input(">>> Press Enter to RELEASE (Ctrl+C to abort) <<<\n")

    rclpy.init()
    client = RobotStateClient()
    commander = WholeBodyCommander()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not client.wait_ready(timeout_sec=10.0):
        print("[ERROR] state topics not ready -- is the robot HAL running?")
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown(); return

    def read_positions():
        _imus, head, waist, arm, leg = client.get_robot_states()
        return {jr.name: jr.position for jr in (head + waist + arm + leg)}

    dt = 1.0 / RATE_HZ
    ramp = 0.0 if args.now else args.ramp_seconds
    t0 = time.perf_counter()

    try:
        # Phase 1: ramp gains down while tracking the measured position.
        while True:
            t = time.perf_counter() - t0
            alpha = 0.0 if ramp <= 0 else max(0.0, 1.0 - t / ramp)
            pos = read_positions()
            for area in AREAS:
                commander.publish(area, build_cmd(area, pos, alpha,
                                                  args.start_kp, args.damping))
            if alpha <= 0.0:
                break
            time.sleep(dt)

        print(f"[released] gains at zero (residual damping {args.damping}). "
              f"Robot is free to move for the next {args.hold_seconds:.0f}s...")

        # Phase 2: keep republishing zero-gain commands so a controller that
        # latches the last command keeps the drives compliant.
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < args.hold_seconds:
            pos = read_positions()
            for area in AREAS:
                commander.publish(area, build_cmd(area, pos, 0.0,
                                                  args.start_kp, args.damping))
            time.sleep(dt)
        print("[done] exiting; joints left compliant.")
    except KeyboardInterrupt:
        print("\n[interrupt] exiting; joints left in current gain state.")
    finally:
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
