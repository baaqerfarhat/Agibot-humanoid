#!/usr/bin/env python3
"""Hold the robot at its DEFAULT pose and measure the observation + policy output.

This removes the 'sagging robot' confound: once the robot is actually AT the sim
default pose and upright, the observation should match what the policy saw in
training, so we can tell whether sensors/obs are correct.

Expected at default pose + upright + cmd 0 (if everything is right):
    projected_gravity ~ [0, 0, -1]
    joint_pos (q-def) ~ 0   for all joints
    |policy action|   ~ small   (stand still)

SAFETY: robot must be roped/suspended, MC (pnc) stopped, e-stop in hand.
Gentle gains; self-terminates after the hold; results written to RESULT_PATH.
"""
import json
import threading
import time

import numpy as np
import rclpy

from robot_states_control import RobotStateClient, WholeBodyCommander, robot_model
from deploy_x2_walk import (
    NumpyPolicy, ObservationBuilder, projected_gravity,
    publish_pose, CONTROLLED_AREAS,
)

POLICY = "policies/x2_policy.npz"
BASE_IMU = "torso"
GAIN_SCALE = 0.5
RAMP_S = 3.0
HOLD_S = 4.0
DT = 0.02
RESULT_PATH = "/tmp/diag_hold_result.txt"


def main():
    policy = NumpyPolicy(POLICY)
    meta = policy.meta
    jn = meta["joint_names"]
    default = np.array(meta["default_joint_pos"], np.float32)
    ascale = np.array(meta["action_scale"], np.float32)
    if ascale.ndim == 0:
        ascale = np.full(len(jn), float(ascale), np.float32)
    kp = dict(zip(jn, meta["joint_stiffness"]))
    kd = dict(zip(jn, meta["joint_damping"]))
    default_by_name = dict(zip(jn, default.tolist()))

    rclpy.init()
    client = RobotStateClient()
    commander = WholeBodyCommander()
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(client); ex.add_node(commander)
    threading.Thread(target=ex.spin, daemon=True).start()
    if not client.wait_ready(timeout_sec=10.0):
        print("[ERROR] state not ready"); return

    def read():
        imus, head, waist, arm, leg = client.get_robot_states()
        return imus, {jr.name: jr for jr in (head + waist + arm + leg)}

    imus, jmap = read()
    start = {n: jmap[n].position for n in jn}

    input(">>> Robot roped & pnc stopped & e-stop in hand? Press Enter to HOLD default <<<")

    t0 = time.perf_counter()
    nxt = t0
    lines = []
    measured = False
    while True:
        now = time.perf_counter()
        el = now - t0
        imus, jmap = read()
        if el < RAMP_S:
            a = el / RAMP_S
            tgt = {n: (1 - a) * start[n] + a * default_by_name[n] for n in jn}
        elif el < RAMP_S + HOLD_S:
            tgt = dict(default_by_name)
            # measure once, ~1s into the hold (robot has settled at default)
            if not measured and el > RAMP_S + 1.5:
                measured = True
                g_t = projected_gravity(imus["torso"].quat)
                g_c = projected_gravity(imus["chest"].quat)
                q = np.array([jmap[n].position for n in jn], np.float32)
                err = q - default
                worst = sorted(zip(jn, err), key=lambda x: -abs(x[1]))[:6]
                lines.append("=" * 70)
                lines.append("MEASURED WHILE HOLDING DEFAULT POSE")
                lines.append(f"proj_g torso = {np.round(g_t,3).tolist()}  (want ~[0,0,-1])")
                lines.append(f"proj_g chest = {np.round(g_c,3).tolist()}")
                lines.append(f"joint_pos err |q-def|: mean={np.mean(np.abs(err)):.3f} "
                             f"max={np.max(np.abs(err)):.3f}")
                lines.append("worst joints: " + ", ".join(f"{n}={e:+.3f}" for n, e in worst))
                for cmd in ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]):
                    ob = ObservationBuilder(meta, base_imu=BASE_IMU)
                    obs = ob.build(imus, jmap, cmd)
                    act = policy(obs).reshape(-1)
                    tg = act * ascale + default
                    lines.append("-" * 70)
                    lines.append(f"command={cmd}  |action| mean={np.mean(np.abs(act)):.3f} "
                                 f"max={np.max(np.abs(act)):.3f}")
                    for n in ("left_knee_joint", "right_knee_joint",
                              "left_hip_pitch_joint", "right_hip_pitch_joint",
                              "left_ankle_pitch_joint"):
                        i = jn.index(n)
                        lines.append(f"   {n:<24} act={act[i]:+.3f} target={tg[i]:+.3f} "
                                     f"meas_q={jmap[n].position:+.3f}")
                for ln in lines:
                    print(ln)
        else:
            break
        publish_pose(commander, tgt, kp, kd, GAIN_SCALE, engage=True)
        nxt += DT
        s = nxt - time.perf_counter()
        if s > 0:
            time.sleep(s)
        else:
            nxt = time.perf_counter()

    with open(RESULT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[done] wrote {RESULT_PATH}")
    client.destroy_node(); commander.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
