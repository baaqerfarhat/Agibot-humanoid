#!/usr/bin/env python3
"""E2: hold static poses from the box-pickup motion and log the torque each joint needs.

Why this and not another policy run. In a run log, a waist tracking error can mean
two different things and they are indistinguishable: the torso is fighting a load
the model does not know about (a mass / CoM residual, which online adaptation CAN
absorb -- it is the fault-recovery case that won +53% survival in Isaac), or the
waist is simply lagging a fast reference (a bandwidth problem, which adaptation
cannot fix). Holding the pose removes the dynamics, so whatever torque is left is
the load residual, cleanly.

The measurement is the per-joint `eff_meas` while the robot stands still at a
reference pose. `run_logs/_static_pose_compare.py` differences it against the
torque the URDF says that pose needs, computed in MuJoCo at zero velocity.

Poses are given as frame indices into the motion. Defaults:

      0   standing, the motion's start pose        -- baseline, near zero load
    160   mid-carry, box up, legs mostly extended  -- the load case, low risk
    120   end of the deep bend, still holding      -- highest torso load
     80   deepest bend                             -- most demanding, run last

SAFETY. Frames 80 and 120 are deep bends and hold the robot's own weight plus the
box on the legs and waist for several seconds. Start with `--frames 0,160`, keep a
hand on the e-stop, and only add the deep poses once the shallow ones look sane.
Aborts on |pelvis roll| like the deploy does. Dry run (no --engage) prints the
whole schedule without publishing.

  python3 static_pose_id.py --frames 0,160 --hold 5 --engage
  python3 static_pose_id.py --frames 0,160,120,80 --hold 5 --engage
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np
import rclpy

from deploy_x2_box_pickup import (
    NumpyPolicy,
    publish_pose,
    roll_of,
)
from robot_states_control import RobotStateClient, WholeBodyCommander
from run_logger import RunLogger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",
                    default="../box_pickup/policy/x2_box_policy_clean_iter9000.npz",
                    help="Only the reference poses and the PD gains are used, but keep "
                         "this the same as the policy you are about to deploy or the "
                         "poses and gains will not be the ones you are diagnosing.")
    ap.add_argument("--frames", default="0,160",
                    help="Comma-separated motion frame indices to hold, in order.")
    ap.add_argument("--hold", type=float, default=5.0, help="Seconds to hold each pose.")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="Seconds discarded at the start of each hold, before averaging.")
    ap.add_argument("--ramp", type=float, default=4.0,
                    help="Seconds to travel between poses. Slow is safe.")
    ap.add_argument("--engage", action="store_true", help="Actually publish.")
    ap.add_argument("--gain-scale", type=float, default=1.0)
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"])
    ap.add_argument("--no-torque-ff", action="store_true")
    ap.add_argument("--roll-abort", type=float, default=0.5,
                    help="Tighter than the deploy's 0.7: nothing here is dynamic, so "
                         "any real roll means something is wrong.")
    ap.add_argument("--log-dir", default="run_logs")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    frames = [int(f) for f in args.frames.split(",") if f.strip()]
    policy = NumpyPolicy(args.policy)
    meta = policy.meta
    joint_names = meta["joint_names"]
    kp_by_name = dict(zip(joint_names, meta["joint_stiffness"]))
    kd_by_name = dict(zip(joint_names, meta["joint_damping"]))
    action_scale = np.array(meta["action_scale"], np.float32)
    if "joint_effort_limit" in meta:
        eff_limit = np.array(meta["joint_effort_limit"], np.float32)
    else:
        eff_limit = 4.0 * action_scale * np.array(meta["joint_stiffness"], np.float32)
    eff_by_name = None if args.no_torque_ff else dict(zip(joint_names, eff_limit.tolist()))

    ref = policy.ref_joint_pos
    n_frames = ref.shape[0]
    for f in frames:
        if not 0 <= f < n_frames:
            raise SystemExit(f"frame {f} outside the motion (0-{n_frames - 1})")

    DT = 1.0 / float(meta.get("control_hz", 50))
    print("=" * 78)
    print("  STATIC POSE TORQUE ID (E2)")
    print(f"  poses:      {frames}   hold {args.hold}s each "
          f"(first {args.settle}s discarded)")
    print(f"  torque ff:  {'OFF' if args.no_torque_ff else 'on'}   "
          f"gain scale {args.gain_scale}")
    print(f"  MODE:       {'ENGAGED (publishing!)' if args.engage else 'DRY RUN'}")
    print("=" * 78)
    for f in frames:
        kn = ref[f, joint_names.index("left_knee_joint")]
        hp = ref[f, joint_names.index("left_hip_pitch_joint")]
        print(f"    frame {f:3d}:  knee {kn:+.2f}  hip_pitch {hp:+.2f}"
              + ("   <-- DEEP BEND, high load" if kn > 0.8 or hp < -1.0 else ""))
    print()

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

    def read_jmap():
        imus, head, waist, arm, leg = client.get_robot_states()
        jmap = {jr.name: jr for jr in (head + waist + arm + leg)}
        missing = [n for n in joint_names if n not in jmap]
        if missing:
            raise RuntimeError(f"State missing joints: {missing}")
        return imus, jmap

    imus0, jmap0 = read_jmap()
    print(f"[check] torso roll = {roll_of(np.asarray(imus0[args.base_imu].quat)):+.3f} rad")
    input(">>> Press Enter to START (Ctrl+C aborts) <<<")

    pol_tag = os.path.splitext(os.path.basename(args.policy))[0]
    logger = RunLogger(
        joint_names, base_imu=args.base_imu, run_name=f"static_pose_id_{pol_tag}",
        meta={"script": "static_pose_id.py", "policy": args.policy,
              "frames": frames, "hold_s": args.hold, "settle_s": args.settle,
              "gain_scale": args.gain_scale, "engage": args.engage,
              "torque_ff": not args.no_torque_ff,
              "joint_effort_limit": eff_limit.tolist(),
              # The offline comparison reproduces this hold in MuJoCo and needs
              # the exact gains that were used here.
              "joint_stiffness": list(meta["joint_stiffness"]),
              "joint_damping": list(meta["joint_damping"])},
        log_dir=args.log_dir, enabled=not args.no_log, log_effort=True)

    prev = {n: float(jmap0[n].position) for n in joint_names}
    t0 = time.perf_counter()
    results = {}

    def step(target, phase, frame):
        imus, jmap = read_jmap()
        roll = roll_of(np.asarray(imus[args.base_imu].quat))
        if abs(roll) > args.roll_abort:
            raise RuntimeError(f"|roll| {abs(roll):.2f} > {args.roll_abort}")
        ff = publish_pose(commander, target, kp_by_name, kd_by_name,
                          args.gain_scale, args.engage, jmap=jmap,
                          eff_by_name=eff_by_name)
        logger.log(time.perf_counter() - t0, phase, frame, imus, jmap, target,
                   effort_cmd=ff)
        return jmap

    try:
        for f in frames:
            goal = {n: float(ref[f, i]) for i, n in enumerate(joint_names)}
            print(f"\n[pose {f}] ramping {args.ramp}s ...")
            t_r = time.perf_counter()
            while (el := time.perf_counter() - t_r) < args.ramp and rclpy.ok():
                a = min(1.0, el / args.ramp)
                step({n: (1 - a) * prev[n] + a * goal[n] for n in joint_names},
                     f"ramp{f}", f)
                time.sleep(DT)
            prev = goal

            print(f"[pose {f}] holding {args.hold}s ...")
            samples = []
            t_h = time.perf_counter()
            while (el := time.perf_counter() - t_h) < args.hold and rclpy.ok():
                jmap = step(goal, f"hold{f}", f)
                if el >= args.settle:
                    samples.append([jmap[n].effort for n in joint_names])
                time.sleep(DT)

            if samples:
                tau = np.asarray(samples, float)
                results[f] = tau.mean(axis=0)
                sd = tau.std(axis=0)
                print(f"[pose {f}] {len(samples)} samples, "
                      f"steadiness (mean sd) {sd.mean():.3f} Nm")
                order = np.argsort(-np.abs(results[f]))[:6]
                for i in order:
                    print(f"      {joint_names[i]:28s} {results[f][i]:+8.2f} Nm "
                          f"+-{sd[i]:.2f}")
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\n[abort] {e}")
    finally:
        print("\n[exit] ramping to the start pose.")
        start = {n: float(jmap0[n].position) for n in joint_names}
        t_s = time.perf_counter()
        while (el := time.perf_counter() - t_s) < 3.0 and rclpy.ok():
            a = min(1.0, el / 3.0)
            try:
                step({n: (1 - a) * prev[n] + a * start[n] for n in joint_names},
                     "exit", 0)
            except Exception:
                break
            time.sleep(DT)
        logger.close()
        if logger.path:
            print(f"\n[log] {logger.path}")
            print("[next] compare against the model:")
            print(f"       python3 run_logs/_static_pose_compare.py "
                  f"{os.path.basename(logger.path)}")
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
