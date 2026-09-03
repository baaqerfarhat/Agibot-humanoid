#!/usr/bin/env python3
"""Deploy the mjlab X2 in-place squat policy on the real AgiBot humanoid.

Runs INSIDE the ROS 2 environment (needs `rclpy` + `aimdk_msgs`, same as
`robot_states_control.py`). The policy is a self-contained `.npz` from
`export_squat_policy_npz.py`; the only extra runtime dependency is numpy.

    -------------------------------------------------------------------------
    PIPELINE (must match mjlab squat training):
      observation -> policy MLP -> action (31)
      target_q = action * action_scale + default_q
      publish position targets with training PD gains, 50 Hz

    Observation (102-D, no base_lin_vel):
        [ base_ang_vel(3), projected_gravity(3),
          joint_pos - default(31), joint_vel(31),
          prev_action(31), command(3) ]

    command = [sin(2πφ), cos(2πφ), target_pelvis_height]
    One 5 s cycle (stand 0.5, down 1.5, bottom 0.5, up 1.5, stand 1.0), then
    the command FREEZES on the standing pose. It does not wrap into a second squat.
    -------------------------------------------------------------------------

    Training IMU was the PELVIS. The robot only has torso/chest IMUs, so the
    default is `--base-ang-vel pelvis` (torso IMU + waist joints). Use
    `--base-ang-vel torso` only to A/B the old walking-style substitution.

#####################################  SAFETY  #####################################
#  1. First runs: robot FULLY SUSPENDED (feet off the ground).
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default is DRY-RUN (computes + logs, does not publish). Add --engage only
#     after dry-run output looks sane.
#  4. Escalate: dry-run -> suspended --engage -> harness/gantry on the ground.
#     This policy was not domain-randomized. Treat it as a supervised test.
#  5. Bottom of the squat is ~0.28 m pelvis height. Catch the robot; keep e-stop.
#  6. Ctrl+C ramps back to the default standing pose. Then:
#         aima em start-app mc
####################################################################################
"""

from __future__ import annotations

import argparse
import json
import math
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

from base_frame import PelvisEstimator
from run_logger import RunLogger


class NumpyPolicy:
    """Loads the .npz from export_squat_policy_npz.py."""

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
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)
        return x @ self.W[-1].T + self.b[-1]


def projected_gravity(quat_xyzw) -> np.ndarray:
    """Gravity in the body frame. Upright -> ~[0, 0, -1]."""
    x, y, z, w = quat_xyzw
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
    return (R.T @ np.array([0.0, 0.0, -1.0], np.float32)).astype(np.float32)


def roll_of(q) -> float:
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def cosine_smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return 0.5 * (1.0 - math.cos(math.pi * s))


class SquatCommand:
    """Same stand-squat-stand trajectory as mjlab SquatCommand (play mode)."""

    def __init__(self, meta: dict):
        self.standing = float(meta["standing_height"])
        self.frac = float(meta["squat_height_frac"])
        self.cycle = float(meta["cycle_time_s"])
        self.t_stand = float(meta["stand_duration_s"])
        self.t_down = float(meta["descend_duration_s"])
        self.t_bottom = float(meta["bottom_duration_s"])
        self.t_up = float(meta["ascend_duration_s"])
        self.wrap = bool(meta.get("wrap_cycle", False))
        self.hold_t = float(meta.get("hold_time_s", self.cycle * 0.9))
        self.h_squat = self.standing * self.frac

    def elapsed(self, t: float) -> float:
        if self.wrap:
            return t % self.cycle
        return min(t, self.hold_t)

    def target_height(self, t: float) -> float:
        t0 = self.t_stand
        t1 = t0 + self.t_down
        t2 = t1 + self.t_bottom
        t3 = t2 + self.t_up
        if t0 <= t < t1:
            a = cosine_smoothstep((t - t0) / max(self.t_down, 1e-6))
            return self.standing + (self.h_squat - self.standing) * a
        if t1 <= t < t2:
            return self.h_squat
        if t2 <= t < t3:
            a = cosine_smoothstep((t - t2) / max(self.t_up, 1e-6))
            return self.h_squat + (self.standing - self.h_squat) * a
        return self.standing

    def command(self, t: float) -> np.ndarray:
        te = self.elapsed(t)
        phase = te / self.cycle
        h = self.target_height(te)
        two_pi = 2.0 * math.pi
        return np.array(
            [math.sin(two_pi * phase), math.cos(two_pi * phase), h], np.float32
        )


class ObservationBuilder:
    def __init__(self, meta: dict, base_imu: str, use_pelvis: bool = True):
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.obs_names = meta["observation_names"]
        self.action_dim = int(meta["action_dim"])
        self.base_imu = base_imu
        self.use_pelvis = bool(use_pelvis)
        self.pelvis_est = PelvisEstimator()
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.last_pelvis_quat = np.array([0, 0, 0, 1], np.float32)
        self.last_base_ang_vel = np.zeros(3, np.float32)

    def build(self, imus, jmap, command) -> np.ndarray:
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        imu = imus[self.base_imu]
        w_pelvis, q_pelvis = self.pelvis_est.update(imu.quat, imu.ang_vel, jmap)
        self.last_pelvis_quat = q_pelvis
        ang_vel = w_pelvis if self.use_pelvis else np.asarray(imu.ang_vel, np.float32)
        quat = q_pelvis if self.use_pelvis else np.asarray(imu.quat, np.float32)
        self.last_base_ang_vel = np.asarray(ang_vel, np.float32)
        proj_g = projected_gravity(quat)
        command = np.asarray(command, np.float32)

        parts = []
        for name in self.obs_names:
            if name == "base_lin_vel":
                parts.append(np.zeros(3, np.float32))
            elif name == "base_ang_vel":
                parts.append(np.asarray(ang_vel, np.float32))
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


CONTROLLED_AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)


def build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name, gain_scale):
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


def _default_policy_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "policies", "x2_squat_policy_40pct_iter16499.npz")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--policy", default=_default_policy_path())
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"])
    ap.add_argument("--base-ang-vel", default="pelvis", choices=["pelvis", "torso"],
                    help="pelvis (default) reconstructs training's pelvis IMU from "
                         "the torso IMU + waist joints.")
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains (lower = gentler). Try 0.9 first.")
    ap.add_argument("--ramp-seconds", type=float, default=4.0)
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    ap.add_argument("--hold-seconds", type=float, default=1.5,
                    help="Seconds to keep the standing command after the 5 s cycle.")
    ap.add_argument("--max-joint-step", type=float, default=0.15,
                    help="Max change in a joint target per 20 ms tick (rad).")
    ap.add_argument("--action-alpha", type=float, default=1.0,
                    help="EMA on the policy action. 1.0=off, 0.5=smoother.")
    ap.add_argument("--tilt-abort", type=float, default=-0.55,
                    help="Abort if pelvis projected_gravity z rises above this.")
    ap.add_argument("--roll-abort", type=float, default=0.45,
                    help="Abort if |pelvis roll| exceeds this (rad).")
    ap.add_argument("--log-dir", default="run_logs")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    policy = NumpyPolicy(args.policy)
    meta = policy.meta
    if meta.get("task") not in (None, "x2_squat"):
        print(f"[WARN] meta.task={meta.get('task')!r} (expected x2_squat)")
    squat = SquatCommand(meta)
    joint_names = meta["joint_names"]
    default = np.array(meta["default_joint_pos"], np.float32)
    action_scale = np.array(meta["action_scale"], np.float32)
    kp_by_name = dict(zip(joint_names, meta["joint_stiffness"]))
    kd_by_name = dict(zip(joint_names, meta["joint_damping"]))
    default_by_name = dict(zip(joint_names, default.tolist()))
    policy_seconds = squat.cycle + max(0.0, args.hold_seconds)
    use_pelvis = args.base_ang_vel == "pelvis"

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  run_path:      {meta.get('run_path', '?')}")
    print(f"  iteration:     {meta.get('iteration', '?')}")
    print(f"  obs terms:     {meta['observation_names']}  (dim {meta['obs_dim']})")
    print(f"  squat:         {squat.frac:.0%} of {squat.standing:.3f} m "
          f"(bottom {squat.h_squat:.3f} m)")
    print(f"  cycle+hold:    {squat.cycle:.1f}s + {args.hold_seconds:.1f}s "
          f"(wrap={squat.wrap}, freeze_t={squat.hold_t:.2f}s)")
    print(f"  base IMU:      {args.base_imu}   ang_vel={args.base_ang_vel}")
    print(f"  gain scale:    {args.gain_scale}")
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    if "base_lin_vel" in meta["observation_names"]:
        print("[WARN] this squat policy should not include base_lin_vel.")

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

    obs_builder = ObservationBuilder(meta, base_imu=args.base_imu, use_pelvis=use_pelvis)

    def read_jmap():
        imus, head, waist, arm, leg = client.get_robot_states()
        jmap = {jr.name: jr for jr in (head + waist + arm + leg)}
        missing = [n for n in joint_names if n not in jmap]
        if missing:
            raise RuntimeError(f"State missing joints: {missing}")
        return imus, jmap

    imus0, jmap0 = read_jmap()
    w0, q0 = obs_builder.pelvis_est.update(
        imus0[args.base_imu].quat, imus0[args.base_imu].ang_vel, jmap0
    )
    g0 = projected_gravity(q0 if use_pelvis else imus0[args.base_imu].quat)
    print(f"[check] pelvis proj_g = {np.round(g0, 3)}  roll={roll_of(q0):+.3f} "
          f"(upright ~[0, 0, -1])")
    if g0[2] > -0.8:
        print("[check] WARNING: robot does not look upright, or IMU axes differ. "
              "Do not --engage until this is ~[0,0,-1].")

    print("\n>>> SAFETY: robot suspended? MC stopped on .40? E-stop in hand? <<<")
    input(">>> Press Enter to START (Ctrl+C to abort) <<<")

    run_name = f"squat_{os.path.splitext(os.path.basename(args.policy))[0]}"
    extra_cols = ["cmd_sin", "cmd_cos", "cmd_height", "pelvis_roll",
                  "proj_g_x", "proj_g_y", "proj_g_z"]
    logger = RunLogger(
        joint_names, base_imu=args.base_imu, run_name=run_name,
        meta={"script": "deploy_x2_squat.py", "policy": args.policy,
              "gain_scale": args.gain_scale, "engage": args.engage,
              "base_ang_vel_source": args.base_ang_vel,
              "hold_seconds": args.hold_seconds,
              "run_path": meta.get("run_path"), "task": meta.get("task")},
        log_dir=args.log_dir, enabled=not args.no_log, extra_columns=extra_cols,
    )

    CONTROL_DT = 0.02
    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)
    filt_action = None
    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    phase = "ramp"
    phase_t0 = t0
    cmd = squat.command(0.0)
    printed_hold = False

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()
            imu = imus[args.base_imu]
            w_pelvis, q_pelvis = obs_builder.pelvis_est.update(imu.quat, imu.ang_vel, jmap)
            g = projected_gravity(q_pelvis if use_pelvis else imu.quat)
            roll = roll_of(q_pelvis)

            if phase == "policy" and (g[2] > args.tilt_abort or abs(roll) > args.roll_abort):
                print(f"\n[ABORT] tilt/roll (proj_g_z={g[2]:.2f}, roll={roll:+.2f}). "
                      "Holding default pose.")
                phase = "done"
                phase_t0 = now

            elapsed = now - phase_t0

            if phase == "ramp":
                alpha = min(1.0, elapsed / max(1e-3, args.ramp_seconds))
                target_by_name = {
                    n: (1 - alpha) * start_pose[n] + alpha * default_by_name[n]
                    for n in joint_names
                }
                if alpha >= 1.0:
                    phase = "settle"
                    phase_t0 = now
                    print("\n[phase] settle at standing default\n")

            elif phase == "settle":
                target_by_name = dict(default_by_name)
                if elapsed >= args.settle_seconds:
                    phase = "policy"
                    phase_t0 = now
                    print("\n[phase] squat policy ENGAGED\n")

            elif phase == "policy":
                cmd = squat.command(elapsed)
                obs = obs_builder.build(imus, jmap, cmd)
                action = policy(obs).reshape(-1)
                obs_builder.last_action = action.astype(np.float32)
                if args.action_alpha < 1.0:
                    if filt_action is None:
                        filt_action = action.copy()
                    else:
                        filt_action = (args.action_alpha * action
                                       + (1.0 - args.action_alpha) * filt_action)
                    applied = filt_action
                else:
                    applied = action
                raw_target = applied * action_scale + default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    step = float(np.clip(raw_target[i] - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step
                if elapsed >= policy_seconds:
                    phase = "done"
                    phase_t0 = now
                    print("\n[phase] cycle+hold complete -> ramp to default\n")

            else:
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {
                    n: (1 - alpha) * prev_target[n] + alpha * default_by_name[n]
                    for n in joint_names
                }
                if alpha >= 1.0 and not printed_hold:
                    printed_hold = True
                    print("\n[phase] done, holding default. Ctrl+C to exit.\n")

            publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                         args.gain_scale, args.engage)
            prev_target = target_by_name
            logger.log(
                now - t0, phase, 0, imus, jmap, target_by_name,
                extra={
                    "cmd_sin": float(cmd[0]), "cmd_cos": float(cmd[1]),
                    "cmd_height": float(cmd[2]), "pelvis_roll": roll,
                    "proj_g_x": float(g[0]), "proj_g_y": float(g[1]),
                    "proj_g_z": float(g[2]),
                },
            )

            if now - last_print >= 1.0:
                last_print = now
                tag = "DRY" if not args.engage else "CMD"
                print(f"[{tag}] phase={phase:6s} t={now - t0:5.1f}s "
                      f"h_cmd={cmd[2]:.3f} proj_g_z={g[2]:+.2f} roll={roll:+.2f} "
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
        ramp_start = dict(prev_target)
        t_stop = time.perf_counter()
        while time.perf_counter() - t_stop < 1.5 and rclpy.ok():
            a = min(1.0, (time.perf_counter() - t_stop) / 1.5)
            tgt = {n: (1 - a) * ramp_start[n] + a * default_by_name[n] for n in joint_names}
            publish_pose(commander, tgt, kp_by_name, kd_by_name, args.gain_scale, args.engage)
            time.sleep(CONTROL_DT)
    finally:
        logger.close()
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
