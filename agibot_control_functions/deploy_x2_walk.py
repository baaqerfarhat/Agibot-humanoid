#!/usr/bin/env python3
"""Deploy a trained mjlab X2 velocity-walking policy on the real AgiBot humanoid.

This runs INSIDE the ROS 2 environment (needs `rclpy` + `aimdk_msgs`, same as
`robot_states_control.py`). The policy is loaded from a self-contained `.npz`
produced by `export_policy_npz.py`, so the only extra runtime dependency is
**numpy** (no torch / onnxruntime needed on the robot).

    -------------------------------------------------------------------------
    PIPELINE (must match how the policy was trained in mjlab):
      observation (built here)  ->  policy MLP  ->  action (31)
      target_q = action * action_scale + default_q       (per joint)
      publish position targets w/ training PD gains, 50 Hz
    -------------------------------------------------------------------------

    Observation layout is read from the policy metadata. For the deployment
    policy (`Mjlab-Velocity-Flat-X2-Deploy`) it is:
        [ base_ang_vel(3), projected_gravity(3),
          joint_pos - default(31), joint_vel(31),
          prev_action(31), command(3) ]
    (If `base_lin_vel` is present it is filled with ZEROS + a loud warning,
     because the real robot cannot measure it. Use the *Deploy* policy.)

#####################################  SAFETY  #####################################
#  1. Robot MUST be FULLY SUSPENDED (feet off the ground) for the first runs.
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default mode is DRY-RUN: it computes & logs commands but DOES NOT publish.
#     Add  --engage  only once dry-run output looks sane and the robot is safe.
#  4. Escalation order:  dry-run  ->  suspended (--engage)  ->  harness/gantry
#     on the ground  ->  free walking. Do NOT skip steps.
#  5. Keep a hand on the e-stop. Ctrl+C ramps back to the default pose and exits.
#  6. When done, restart the controller:   aima em start-app mc
####################################################################################
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np
import rclpy

# Reuse the collaborators' verified client / commander / robot model.
from robot_states_control import (
    JointArea,
    RobotStateClient,
    WholeBodyCommander,
    robot_model,
)
from aimdk_msgs.msg import JointCommand, JointCommandArray


# =============================== policy (numpy MLP) ===============================
class NumpyPolicy:
    """Loads the .npz produced by export_policy_npz.py and runs the forward pass.

    Network (standard rsl_rl actor with baked-in obs normalization):
        x = (obs - mean) / std
        for W, b in hidden:  x = elu(x @ W.T + b)
        action = x @ W_out.T + b_out
    """

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)  # ELU(alpha=1)
        return x @ self.W[-1].T + self.b[-1]


# =============================== math helpers ===============================
def projected_gravity(quat_xyzw) -> np.ndarray:
    """Gravity unit vector expressed in the base frame (matches mjlab).

    quat is (x, y, z, w) base orientation in world. Returns R^T @ [0,0,-1].
    Upright robot -> approx [0, 0, -1].
    """
    x, y, z, w = quat_xyzw
    # Rotation matrix body->world.
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], np.float32)
    g_world = np.array([0.0, 0.0, -1.0], np.float32)
    return (R.T @ g_world).astype(np.float32)


# =============================== observation builder ===============================
class ObservationBuilder:
    def __init__(self, meta: dict, base_imu: str):
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.obs_names = meta["observation_names"]
        self.cmd_names = meta["command_names"]
        self.njoints = len(self.joint_names)
        self.action_dim = int(meta["action_dim"])
        self.base_imu = base_imu
        self.last_action = np.zeros(self.action_dim, np.float32)
        self._warned_lin_vel = False

    def build(self, imus, jmap, command) -> np.ndarray:
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        imu = imus[self.base_imu]
        ang_vel = np.array(imu.ang_vel, np.float32)
        proj_g = projected_gravity(imu.quat)
        command = np.asarray(command, np.float32)

        parts = []
        for name in self.obs_names:
            if name == "base_lin_vel":
                if not self._warned_lin_vel:
                    print("[WARN] policy uses base_lin_vel (unmeasurable) -> feeding ZEROS. "
                          "Use the Deploy policy instead!")
                    self._warned_lin_vel = True
                parts.append(np.zeros(3, np.float32))
            elif name == "base_ang_vel":
                parts.append(ang_vel)
            elif name == "projected_gravity":
                parts.append(proj_g)
            elif name == "joint_pos":
                parts.append(q - self.default)
            elif name == "joint_vel":
                parts.append(dq)
            elif name == "actions":
                parts.append(self.last_action)
            elif name == "command":
                parts.append(command)
            else:
                raise ValueError(f"Unhandled observation term: {name!r}")
        return np.concatenate(parts).astype(np.float32)


# =============================== command helpers ===============================
# Areas the policy drives. HEAD/WAIST/ARM/LEG all have entries in robot_model.
CONTROLLED_AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)


def build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name, gain_scale):
    """One JointCommandArray for `area`, in robot_model order, clamped to limits."""
    cmd = JointCommandArray()
    for ji in robot_model[area]:
        jc = JointCommand()
        jc.name = ji.name
        pos = float(np.clip(pos_by_name[ji.name], ji.lower_limit, ji.upper_limit))
        jc.position = pos
        jc.velocity = 0.0
        jc.effort = 0.0
        jc.stiffness = float(kp_by_name[ji.name] * gain_scale)
        jc.damping = float(kd_by_name[ji.name] * gain_scale)
        cmd.joints.append(jc)
    return cmd


def publish_pose(commander, pos_by_name, kp_by_name, kd_by_name, gain_scale, engage):
    for area in CONTROLLED_AREAS:
        cmd = build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name, gain_scale)
        if engage:
            commander.publish(area, cmd)


# =============================== main ===============================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="Path to x2_policy.npz")
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"],
                    help="Which IMU is the policy base (training was the pelvis).")
    ap.add_argument("--vx", type=float, default=0.3, help="Forward velocity command (m/s).")
    ap.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
    ap.add_argument("--wz", type=float, default=0.0, help="Yaw rate command (rad/s).")
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains (lower = gentler).")
    ap.add_argument("--ramp-seconds", type=float, default=4.0,
                    help="Time to ramp from current pose to the default pose.")
    ap.add_argument("--settle-seconds", type=float, default=2.0,
                    help="Hold the default pose before engaging the policy.")
    ap.add_argument("--cmd-ramp-seconds", type=float, default=3.0,
                    help="Time to ramp the velocity command from 0 to target.")
    ap.add_argument("--run-seconds", type=float, default=10.0,
                    help="How long to run the policy after settling.")
    ap.add_argument("--max-joint-step", type=float, default=0.15,
                    help="Max change in a joint target per 20 ms tick (rad).")
    ap.add_argument("--tilt-abort", type=float, default=-0.6,
                    help="Abort if projected_gravity z rises above this (robot tipping).")
    args = ap.parse_args()

    policy = NumpyPolicy(args.policy)
    meta = policy.meta
    joint_names = meta["joint_names"]
    default = np.array(meta["default_joint_pos"], np.float32)
    action_scale = np.array(meta["action_scale"], np.float32)
    kp_by_name = dict(zip(joint_names, meta["joint_stiffness"]))
    kd_by_name = dict(zip(joint_names, meta["joint_damping"]))
    default_by_name = dict(zip(joint_names, default.tolist()))

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  run_path:      {meta.get('run_path', '?')}")
    print(f"  obs terms:     {meta['observation_names']}  (dim {meta['obs_dim']})")
    print(f"  action joints: {len(joint_names)}")
    print(f"  command:       vx={args.vx}  vy={args.vy}  wz={args.wz}")
    print(f"  base IMU:      {args.base_imu}")
    print(f"  gain scale:    {args.gain_scale}")
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    if "base_lin_vel" in meta["observation_names"]:
        print("[WARN] This policy depends on base_lin_vel, which the robot cannot measure.\n"
              "       It will be fed zeros. Retrain with Mjlab-Velocity-Flat-X2-Deploy.")

    rclpy.init()
    client = RobotStateClient()
    commander = WholeBodyCommander()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not client.wait_ready(timeout_sec=10.0):
        print("[ERROR] state topics not ready.")
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown(); return

    obs_builder = ObservationBuilder(meta, base_imu=args.base_imu)

    def read_jmap():
        imus, head, waist, arm, leg = client.get_robot_states()
        jmap = {jr.name: jr for jr in (head + waist + arm + leg)}
        missing = [n for n in joint_names if n not in jmap]
        if missing:
            raise RuntimeError(f"State missing joints: {missing}")
        return imus, jmap

    # Sanity: print one observation in dry run so the user can eyeball it.
    imus0, jmap0 = read_jmap()
    g0 = projected_gravity(imus0[args.base_imu].quat)
    print(f"[check] projected_gravity[{args.base_imu}] = {np.round(g0, 3)} "
          f"(upright should be ~[0, 0, -1])")
    if g0[2] > -0.8:
        print("[check] WARNING: robot does not look upright, or IMU axes differ from "
              "training. Verify before engaging.")

    print("\n>>> SAFETY: robot suspended? MC stopped on .40? E-stop in hand? <<<")
    input(">>> Press Enter to START (Ctrl+C to abort) <<<")

    CONTROL_DT = 0.02  # 50 Hz, matches training decimation (4 x 5 ms).
    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)  # for per-tick step clamp

    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    phase = "ramp"  # ramp -> settle -> policy -> done
    phase_t0 = t0

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()

            # ---------------- safety: tilt abort (only once moving under policy) ----------------
            g = projected_gravity(imus[args.base_imu].quat)
            if phase == "policy" and g[2] > args.tilt_abort:
                print(f"\n[ABORT] tilt detected (proj_g_z={g[2]:.2f} > {args.tilt_abort}). "
                      "Holding default pose.")
                phase = "done"; phase_t0 = now

            elapsed = now - phase_t0

            if phase == "ramp":
                alpha = min(1.0, elapsed / max(1e-3, args.ramp_seconds))
                target_by_name = {n: (1 - alpha) * start_pose[n] + alpha * default_by_name[n]
                                  for n in joint_names}
                if alpha >= 1.0:
                    phase = "settle"; phase_t0 = now

            elif phase == "settle":
                target_by_name = dict(default_by_name)
                if elapsed >= args.settle_seconds:
                    phase = "policy"; phase_t0 = now
                    print("\n[phase] policy ENGAGED\n")

            elif phase == "policy":
                cmd_alpha = min(1.0, elapsed / max(1e-3, args.cmd_ramp_seconds))
                command = [args.vx * cmd_alpha, args.vy * cmd_alpha, args.wz * cmd_alpha]
                obs = obs_builder.build(imus, jmap, command)
                action = policy(obs).reshape(-1)
                obs_builder.last_action = action.astype(np.float32)  # raw action -> next obs
                raw_target = action * action_scale + default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    # per-tick step clamp against last commanded target
                    step = float(np.clip(raw_target[i] - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step
                if elapsed >= args.run_seconds:
                    phase = "done"; phase_t0 = now
                    print("\n[phase] run complete -> holding default\n")

            else:  # done: ramp back toward default and hold
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * default_by_name[n]
                                  for n in joint_names}

            publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                         args.gain_scale, args.engage)
            prev_target = target_by_name

            if now - last_print >= 1.0:
                last_print = now
                tag = "DRY" if not args.engage else "CMD"
                print(f"[{tag}] phase={phase:6s} t={now - t0:5.1f}s "
                      f"proj_g={np.round(g, 2)} "
                      f"knee_L={target_by_name['left_knee_joint']:+.3f} "
                      f"knee_R={target_by_name['right_knee_joint']:+.3f}")

            next_t += CONTROL_DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[interrupt] ramping to default pose and exiting.")
        # brief ramp-to-default for a soft stop
        ramp_start = dict(prev_target)
        t_stop = time.perf_counter()
        while time.perf_counter() - t_stop < 1.5 and rclpy.ok():
            a = min(1.0, (time.perf_counter() - t_stop) / 1.5)
            tgt = {n: (1 - a) * ramp_start[n] + a * default_by_name[n] for n in joint_names}
            publish_pose(commander, tgt, kp_by_name, kd_by_name, args.gain_scale, args.engage)
            time.sleep(CONTROL_DT)
    finally:
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
