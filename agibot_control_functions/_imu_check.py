#!/usr/bin/env python3
"""Read both IMUs and report projected_gravity (base frame) for each.

Upright + correct frame -> projected_gravity ~ [0, 0, -1].
Helps diagnose the torso/chest IMU axis convention vs. training (pelvis).
"""
import time
import threading
import numpy as np
import rclpy
from robot_states_control import RobotStateClient


def projected_gravity(quat_xyzw):
    x, y, z, w = quat_xyzw
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - w * z); r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y); r21 = 2 * (y * z + w * x); r22 = 1 - 2 * (x * x + y * y)
    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], np.float32)
    return (R.T @ np.array([0.0, 0.0, -1.0], np.float32)).astype(np.float32)


def main():
    rclpy.init()
    client = RobotStateClient()
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(client)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not client.wait_ready(timeout_sec=10.0):
        print("[ERROR] state not ready"); return

    print("Reading IMUs for 3 s (robot should be standing upright)...\n")
    for _ in range(6):
        imus, head, waist, arm, leg = client.get_robot_states()
        for src in ("torso", "chest"):
            im = imus[src]
            g = projected_gravity(im.quat)
            print(f"  IMU[{src:5s}] quat(xyzw)={tuple(round(v,3) for v in im.quat)}  "
                  f"ang_vel={tuple(round(v,3) for v in im.ang_vel)}  "
                  f"-> proj_g={np.round(g,3)}")
        print("-" * 70)
        time.sleep(0.5)

    print("\nInterpretation: the axis whose proj_g component is ~ -1 is that IMU's"
          " 'up'. Training expects proj_g ~ [0, 0, -1] (i.e. -Z).")
    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
