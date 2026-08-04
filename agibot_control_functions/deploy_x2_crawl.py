#!/usr/bin/env python3
"""Deploy the holosoma X2 slope-crawl (whole-body tracking) policy on the real AgiBot.

Runs INSIDE the ROS 2 environment (needs `rclpy` + `aimdk_msgs`, same as
`robot_states_control.py`). Policy is a self-contained `.npz` from
`export_crawl_policy_npz.py` — runtime needs only **numpy** on the robot.

    -------------------------------------------------------------------------
    PIPELINE (must match holosoma crawl WBT training):
      observation (built here)  ->  policy MLP  ->  action (31)
      target_q = action * action_scale + default_q       (per joint)
      publish position targets w/ training PD gains, 50 Hz

    Observation layout (167 dims, holosoma concatenates terms ALPHABETICALLY):
        [ prev_action(31),
          base_ang_vel(3),                           <- torso/chest IMU gyro
          joint_pos - default(31), joint_vel(31),
          ref_joint_pos(31), ref_joint_vel(31),      <- motion clock (from npz)
          motion_ref_ori_b(6),                       <- ref torso ori rel. IMU
          projected_gravity(3) ]                     <- gravity in base frame
    -------------------------------------------------------------------------

    This is a PRONE hands-and-feet slope crawl. Frame 0 is already the crawl
    start pose — do NOT start the robot standing upright. Place/suspend the
    robot near the reference crawl pose, palms down, on a clear slope or flat
    floor for first trials.

#####################################  SAFETY  #####################################
#  1. First runs: robot SUSPENDED / gantry, NO floor contact -> verify shape.
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default mode is DRY-RUN: computes & logs commands, DOES NOT publish.
#     Add  --engage  only once dry-run output looks sane.
#  4. Escalation: dry-run -> suspended (--engage) -> gantry on surface -> free.
#  5. Crawl has large pitch; --roll-abort applies to ROLL only. After the
#     motion ends we HOLD the final crawl pose (never ramp to standing).
#  6. Keep a hand on the e-stop. Ctrl+C holds the current pose and exits.
#  7. When done, restart the controller:   aima em start-app mc
####################################################################################
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np
import rclpy

from robot_states_control import (
    JointArea,
    RobotStateClient,
    WholeBodyCommander,
    robot_model,
)
from aimdk_msgs.msg import JointCommand, JointCommandArray
from run_logger import RunLogger


# =============================== policy (numpy MLP) ===============================
class NumpyPolicy:
    """Same forward pass as the box / walking deployment (rsl_rl actor, ELU)."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))
        self.ref_joint_pos = d["ref_joint_pos"].astype(np.float32)
        self.ref_joint_vel = d["ref_joint_vel"].astype(np.float32)
        self.ref_quat_xyzw = d["ref_quat_xyzw"].astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)
        return x @ self.W[-1].T + self.b[-1]


# =============================== quaternion helpers (xyzw) ===============================
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        np.float32,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        np.float32,
    )


def yaw_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32)


def roll_of(q: np.ndarray) -> float:
    x, y, z, w = q
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def projected_gravity_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """World gravity [0,0,-1] expressed in the body frame (matches holosoma)."""
    # holosoma_inference quat_rotate_inverse expects wxyz batch.
    q = np.asarray(q_xyzw, np.float32).reshape(4)
    qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])
    v = np.array([0.0, 0.0, -1.0], np.float32)
    q_vec = np.array([qx, qy, qz], np.float32)
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(q_vec, v) * qw * 2.0
    c = q_vec * float(np.dot(q_vec, v)) * 2.0
    return (a - b + c).astype(np.float32)


# =============================== observation builder ===============================
class CrawlObservationBuilder:
    """Builds the 167-dim crawl WBT actor observation."""

    def __init__(self, policy: NumpyPolicy, base_imu: str):
        meta = policy.meta
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.action_dim = int(meta["action_dim"])
        self.base_imu = base_imu
        self.policy = policy
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.yaw_offset = np.array([0, 0, 0, 1], np.float32)

    def align(self, imu_quat_xyzw) -> None:
        q_robot_yaw = yaw_quat(np.asarray(imu_quat_xyzw, np.float32))
        q_ref0_yaw = yaw_quat(self.policy.ref_quat_xyzw[0])
        self.yaw_offset = quat_mul(q_robot_yaw, quat_inv(q_ref0_yaw))

    def build(self, imus, jmap, frame: int) -> np.ndarray:
        T = self.policy.ref_joint_pos.shape[0]
        frame = min(frame, T - 1)
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        imu = imus[self.base_imu]
        q_imu = np.asarray(imu.quat, np.float32)

        q_ref = quat_mul(self.yaw_offset, self.policy.ref_quat_xyzw[frame])
        q_rel = quat_mul(quat_inv(q_imu), q_ref)
        ori6 = quat_to_mat(q_rel)[:, :2].reshape(-1)
        grav = projected_gravity_xyzw(q_imu)

        # Alphabetical: actions, base_ang_vel, dof_pos, dof_vel,
        # motion_command, motion_ref_ori_b, projected_gravity
        return np.concatenate(
            [
                self.last_action,
                np.asarray(imu.ang_vel, np.float32),
                q - self.default,
                dq,
                self.policy.ref_joint_pos[frame],
                self.policy.ref_joint_vel[frame],
                ori6,
                grav,
            ]
        ).astype(np.float32)


# =============================== command helpers ===============================
CONTROLLED_AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)


def build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name, gain_scale):
    cmd = JointCommandArray()
    for ji in robot_model[area]:
        jc = JointCommand()
        jc.name = ji.name
        jc.position = float(np.clip(pos_by_name[ji.name], ji.lower_limit, ji.upper_limit))
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
    ap.add_argument("--policy",
                    default="../box_pickup/policy/x2_crawl_policy_v3.npz",
                    help="Path to crawl policy .npz (default: v3 tracking export)")
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"],
                    help="IMU used as the policy base (training ref body = torso_link).")
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains (lower = gentler).")
    ap.add_argument("--ramp-seconds", type=float, default=8.0,
                    help="Time to ramp from current pose to the crawl start pose.")
    ap.add_argument("--settle-seconds", type=float, default=2.0,
                    help="Hold the start pose before engaging the policy.")
    ap.add_argument("--max-joint-step", type=float, default=0.12,
                    help="Max change in a joint target per 20 ms tick (rad).")
    ap.add_argument("--roll-abort", type=float, default=1.0,
                    help="Abort if |torso roll| exceeds this (rad). Pitch is NOT "
                         "checked: the crawl motion has large pitch.")
    ap.add_argument("--hold-end-seconds", type=float, default=5.0,
                    help="Hold the final crawl pose after the motion ends.")
    ap.add_argument("--joint-filter", type=float, default=0.6,
                    help="EMA smoothing on ALL joint targets (0 = off). Crawl is "
                         "slow; mild lag is usually helpful on hardware.")
    ap.add_argument("--log-dir", default="run_logs",
                    help="Folder for per-run joint/IMU CSV logs.")
    ap.add_argument("--no-log", action="store_true",
                    help="Disable per-run data logging.")
    args = ap.parse_args()

    policy = NumpyPolicy(args.policy)
    meta = policy.meta
    joint_names = meta["joint_names"]
    default = np.array(meta["default_joint_pos"], np.float32)
    action_scale = np.array(meta["action_scale"], np.float32)
    kp_by_name = dict(zip(joint_names, meta["joint_stiffness"]))
    kd_by_name = dict(zip(joint_names, meta["joint_damping"]))

    fps = int(meta["motion_fps"])
    n_frames = int(meta["motion_frames"])
    CONTROL_DT = 1.0 / float(meta.get("control_hz", 50))
    assert abs(CONTROL_DT * fps - 1.0) < 1e-6, "control rate must match motion fps"
    assert int(meta["obs_dim"]) == 167, f"expected crawl obs_dim=167, got {meta['obs_dim']}"

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  task:          {meta.get('task')}   run: {meta.get('run_path', '?')}")
    print(f"  obs terms:     {meta['observation_names']}  (dim {meta['obs_dim']})")
    print(f"  motion:        {n_frames} frames @ {fps} Hz = {n_frames / fps:.1f} s")
    print(f"  action joints: {len(joint_names)}   gain scale: {args.gain_scale}")
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    print("\nCRAWL SETUP: start the robot in / near the PRONE crawl pose (frame 0),")
    print("palms down. Do NOT start standing. First trials: SUSPENDED / gantry.\n")

    rclpy.init()
    client = RobotStateClient()
    commander = WholeBodyCommander()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)
    threading.Thread(target=executor.spin, daemon=True).start()

    if not client.wait_ready(timeout_sec=10.0, required_imus=[args.base_imu]):
        print("[ERROR] state topics not ready.")
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown(); return

    obs_builder = CrawlObservationBuilder(policy, base_imu=args.base_imu)

    def read_jmap():
        imus, head, waist, arm, leg = client.get_robot_states()
        jmap = {jr.name: jr for jr in (head + waist + arm + leg)}
        missing = [n for n in joint_names if n not in jmap]
        if missing:
            raise RuntimeError(f"State missing joints: {missing}")
        return imus, jmap

    imus0, jmap0 = read_jmap()
    obs_builder.align(imus0[args.base_imu].quat)
    obs0 = obs_builder.build(imus0, jmap0, frame=0)
    a0 = policy(obs0)
    grav0 = projected_gravity_xyzw(np.asarray(imus0[args.base_imu].quat, np.float32))
    print(f"[check] frame-0 obs built (dim {obs0.shape[0]}), "
          f"|action|_max = {np.abs(a0).max():.3f} (should be O(1))")
    print(f"[check] torso roll = {roll_of(np.asarray(imus0[args.base_imu].quat)):+.3f} rad")
    print(f"[check] projected_gravity = ({grav0[0]:+.3f}, {grav0[1]:+.3f}, {grav0[2]:+.3f})")

    start_ref = {n: float(policy.ref_joint_pos[0][i]) for i, n in enumerate(joint_names)}
    end_ref = {n: float(policy.ref_joint_pos[-1][i]) for i, n in enumerate(joint_names)}

    # Pose check vs crawl start (not standing stance).
    pose_err = {n: jmap0[n].position - start_ref[n] for n in joint_names}
    worst = max(pose_err.items(), key=lambda kv: abs(kv[1]))
    print("\n[pose] joints vs crawl motion frame 0 (worst first):")
    for n, e in sorted(pose_err.items(), key=lambda kv: -abs(kv[1]))[:8]:
        flag = "  <-- LARGE" if abs(e) > 0.35 else ""
        print(f"    {n:32s} {e:+.3f} rad{flag}")
    if abs(worst[1]) > 0.35:
        print(f"[pose] WARNING: {worst[0]} is {worst[1]:+.2f} rad off crawl start.")
        print("[pose] Reposition toward the prone crawl pose before --engage.")

    print("\n>>> SAFETY: suspended/gantry? MC stopped on .40? E-stop in hand? <<<")
    input(">>> Press Enter to START (Ctrl+C to abort) <<<")

    run_name = f"crawl_{os.path.splitext(os.path.basename(args.policy))[0]}"
    logger = RunLogger(
        joint_names, base_imu=args.base_imu, run_name=run_name,
        meta={"script": "deploy_x2_crawl.py", "policy": args.policy,
              "gain_scale": args.gain_scale, "joint_filter": args.joint_filter,
              "engage": args.engage, "task": meta.get("task"),
              "run_path": meta.get("run_path")},
        log_dir=args.log_dir, enabled=not args.no_log)

    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)

    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    phase = "ramp"  # ramp -> settle -> policy -> done (hold final crawl pose)
    phase_t0 = t0
    frame = 0

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()

            roll = roll_of(np.asarray(imus[args.base_imu].quat, np.float32))
            if phase == "policy" and abs(roll) > args.roll_abort:
                print(f"\n[ABORT] roll {roll:+.2f} rad exceeds {args.roll_abort}. Holding pose.")
                phase = "done"; phase_t0 = now

            elapsed = now - phase_t0

            if phase == "ramp":
                alpha = min(1.0, elapsed / max(1e-3, args.ramp_seconds))
                target_by_name = {n: (1 - alpha) * start_pose[n] + alpha * start_ref[n]
                                  for n in joint_names}
                if alpha >= 1.0:
                    phase = "settle"; phase_t0 = now

            elif phase == "settle":
                target_by_name = dict(start_ref)
                if elapsed >= args.settle_seconds:
                    obs_builder.align(imus[args.base_imu].quat)
                    phase = "policy"; phase_t0 = now; frame = 0
                    print("\n[phase] policy ENGAGED -- crawl motion clock running\n")

            elif phase == "policy":
                frame = int(elapsed / CONTROL_DT)
                obs = obs_builder.build(imus, jmap, frame)
                action = policy(obs).reshape(-1)
                obs_builder.last_action = action.astype(np.float32)
                raw_target = action * action_scale + default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    tgt = float(raw_target[i])
                    if args.joint_filter > 0.0:
                        tgt = (1.0 - args.joint_filter) * tgt \
                            + args.joint_filter * prev_target[n]
                    step = float(np.clip(tgt - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step
                if frame >= n_frames + int(args.hold_end_seconds / CONTROL_DT):
                    phase = "done"; phase_t0 = now
                    print("\n[phase] motion complete -> holding final crawl pose\n")

            else:  # done: hold final crawl pose (never stand up from prone)
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * end_ref[n]
                                  for n in joint_names}

            publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                         args.gain_scale, args.engage)
            prev_target = target_by_name
            logger.log(now - t0, phase, frame, imus, jmap, target_by_name)

            if now - last_print >= 1.0:
                last_print = now
                grav = projected_gravity_xyzw(np.asarray(imus[args.base_imu].quat, np.float32))
                tag = "DRY" if not args.engage else "CMD"
                print(f"[{tag}] phase={phase:6s} t={now - t0:5.1f}s frame={frame:4d}/{n_frames} "
                      f"roll={roll:+.2f} g=({grav[0]:+.2f},{grav[1]:+.2f},{grav[2]:+.2f}) "
                      f"knee_L={target_by_name['left_knee_joint']:+.3f} "
                      f"elbow_L={target_by_name['left_elbow_joint']:+.3f}")

            next_t += CONTROL_DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[interrupt] holding current pose and exiting (no stand-up ramp).")
        hold = dict(prev_target)
        t_stop = time.perf_counter()
        while time.perf_counter() - t_stop < 1.0 and rclpy.ok():
            publish_pose(commander, hold, kp_by_name, kd_by_name, args.gain_scale, args.engage)
            try:
                imus_i, jmap_i = read_jmap()
                logger.log(time.perf_counter() - t0, "interrupt", frame, imus_i, jmap_i, hold)
            except Exception:
                pass
            time.sleep(CONTROL_DT)
    finally:
        logger.close()
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
