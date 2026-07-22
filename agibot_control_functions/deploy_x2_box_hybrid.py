#!/usr/bin/env python3
"""HYBRID box pickup-carry-setdown for the real AgiBot X2.

Sequences TWO policies:

  1. PICKUP   (WBT policy, x2_box_policy.npz): stand -> bend -> squeeze-grasp
              the box -> lift to chest. The motion clock then STOPS in the
              middle of the built-in 2 s HOLD segment (robot standing still,
              box at chest).
  2. CARRY    (walking policy, x2_policy.npz): the proven velocity-command
              walking policy drives legs/waist/head while all 14 ARM joints
              stay LOCKED at the exact targets the WBT policy commanded at the
              switch (keeps the squeeze). Velocity ramps 0 -> vx -> 0, then a
              short settle at zero command.
  3. SET-DOWN (WBT policy again): the motion clock resumes from mid-HOLD and
              plays the remaining hold + set-down + stand-up. Yaw is re-aligned
              at the switch, so heading drift during the carry is fine.

The WBT policy must be exported from a run trained on the IN-PLACE clip
(x2_box_v25_inplace_hybrid or later): pickup + 2 s hold + set-down, feet
pinned, no walking inside the clip.

    -------------------------------------------------------------------------
    Both policies run at 50 Hz and command position targets with their own
    training PD gains (they differ! gains are switched with the policy):
      WBT (holosoma):  legs hip_pitch 180 / rest 100, kd 5; arms 20/2
      Walk (mjlab):    legs 200 (ankles 60/40), kd 5/2/1.5; arms 60..10
    During CARRY the ARM joints keep the WBT gains + frozen WBT targets.
    -------------------------------------------------------------------------

#####################################  SAFETY  #####################################
#  1. Escalation ladder (do NOT skip): dry-run -> suspended, no box -> gantry on
#     ground, no box -> gantry with box -> free.  Box: 45 cm cube, 0.5-1.5 kg.
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default is DRY-RUN (no publish). Add --engage only when dry-run looks sane.
#  4. Box placement: center ~0.40 m in front of the robot on its heading.
#     Leave (walk-seconds x vx + 1) m of clear floor ahead for the carry.
#  5. Roll abort only (the pickup has a deep forward bend; pitch is normal).
#  6. Ctrl+C at any time ramps to the default pose and exits.
####################################################################################
"""

from __future__ import annotations

import argparse
import json
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

ARM_JOINT_PREFIXES = ("left_shoulder", "left_elbow", "left_wrist",
                      "right_shoulder", "right_elbow", "right_wrist")


def is_arm(name: str) -> bool:
    return name.startswith(ARM_JOINT_PREFIXES)


# =============================== numpy MLP (rsl_rl actor, ELU) ===============================
class NumpyPolicy:
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))
        if "ref_joint_pos" in d:  # WBT export carries the reference motion
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
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], np.float32)


def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], np.float32)


def yaw_quat(q):
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32)


def roll_of(q) -> float:
    x, y, z, w = q
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def projected_gravity(quat_xyzw):
    R = quat_to_mat(np.asarray(quat_xyzw, np.float32))
    return (R.T @ np.array([0.0, 0.0, -1.0], np.float32)).astype(np.float32)


# =============================== observation builders ===============================
class WbtObservationBuilder:
    """164-dim WBT actor obs; terms concatenated alphabetically (holosoma)."""

    def __init__(self, policy: NumpyPolicy, base_imu: str):
        meta = policy.meta
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.action_dim = int(meta["action_dim"])
        self.base_imu = base_imu
        self.policy = policy
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.yaw_offset = np.array([0, 0, 0, 1], np.float32)

    def align(self, imu_quat_xyzw, frame: int = 0) -> None:
        """Align the reference heading at `frame` with the robot's heading."""
        q_robot_yaw = yaw_quat(np.asarray(imu_quat_xyzw, np.float32))
        q_ref_yaw = yaw_quat(self.policy.ref_quat_xyzw[frame])
        self.yaw_offset = quat_mul(q_robot_yaw, quat_inv(q_ref_yaw))

    def build(self, imus, jmap, frame: int) -> np.ndarray:
        T = self.policy.ref_joint_pos.shape[0]
        frame = min(frame, T - 1)
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        imu = imus[self.base_imu]
        q_ref = quat_mul(self.yaw_offset, self.policy.ref_quat_xyzw[frame])
        q_rel = quat_mul(quat_inv(np.asarray(imu.quat, np.float32)), q_ref)
        ori6 = quat_to_mat(q_rel)[:, :2].reshape(-1)
        return np.concatenate([self.last_action,
                               np.asarray(imu.ang_vel, np.float32),
                               q - self.default, dq,
                               self.policy.ref_joint_pos[frame],
                               self.policy.ref_joint_vel[frame],
                               ori6]).astype(np.float32)


class WalkObservationBuilder:
    """mjlab velocity-policy obs, driven by meta['observation_names']."""

    def __init__(self, policy: NumpyPolicy, base_imu: str, grav_bias: float = 0.0):
        meta = policy.meta
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.obs_names = meta["observation_names"]
        self.base_imu = base_imu
        self.last_action = np.zeros(int(meta["action_dim"]), np.float32)
        self.grav_bias = float(grav_bias)
        # mask everything above the hips: while carrying the box, the arms /
        # waist / head are far out of the walking policy's training
        # distribution, so we show it the default (zero-offset) upper body.
        upper = ("waist", "left_shoulder", "left_elbow", "left_wrist",
                 "right_shoulder", "right_elbow", "right_wrist", "head")
        self.upper_idx = np.array(
            [i for i, n in enumerate(self.joint_names) if n.startswith(upper)],
            np.int64)

    def build(self, imus, jmap, command) -> np.ndarray:
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        q[self.upper_idx] = self.default[self.upper_idx]
        dq[self.upper_idx] = 0.0
        imu = imus[self.base_imu]
        parts = []
        for name in self.obs_names:
            if name == "base_lin_vel":
                parts.append(np.zeros(3, np.float32))
            elif name == "base_ang_vel":
                parts.append(np.asarray(imu.ang_vel, np.float32))
            elif name == "projected_gravity":
                g = projected_gravity(imu.quat)
                # bias: the box shifts the CoM forward; nudging perceived
                # pitch makes the policy lean back and hold the commanded speed
                g[0] += self.grav_bias
                parts.append(g)
            elif name == "joint_pos":
                parts.append(q - self.default)
            elif name == "joint_vel":
                parts.append(dq)
            elif name == "actions":
                parts.append(self.last_action)
            elif name == "command":
                parts.append(np.asarray(command, np.float32))
            else:
                raise ValueError(f"Unhandled walking obs term: {name!r}")
        return np.concatenate(parts).astype(np.float32)


# =============================== command publish ===============================
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
    ap.add_argument("--box-policy", required=True, help="x2_box_policy.npz (WBT, in-place clip)")
    ap.add_argument("--walk-policy", required=True, help="x2_policy.npz (velocity walking)")
    ap.add_argument("--engage", action="store_true", help="ACTUALLY publish (else dry run).")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"])
    ap.add_argument("--gain-scale", type=float, default=1.0)
    ap.add_argument("--vx", type=float, default=0.3, help="Carry walking speed (m/s).")
    ap.add_argument("--walk-seconds", type=float, default=6.0,
                    help="Carry phase duration incl. 1 s ramp-up, 1 s ramp-down, "
                         "1 s settle at zero command.")
    ap.add_argument("--hold-frame", type=int, default=None,
                    help="Motion frame (50 Hz) where the clock freezes for the carry. "
                         "Default: middle of the clip's HOLD segment from metadata.")
    ap.add_argument("--ramp-seconds", type=float, default=5.0)
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    ap.add_argument("--max-joint-step", type=float, default=0.15)
    ap.add_argument("--roll-abort", type=float, default=0.7)
    ap.add_argument("--hold-end-seconds", type=float, default=3.0)
    ap.add_argument("--grav-bias", type=float, default=0.12,
                    help="Forward pitch bias added to the walking policy's "
                         "projected-gravity x (compensates the box payload; "
                         "0.12 matched sim). Set 0 to disable.")
    args = ap.parse_args()

    box_policy = NumpyPolicy(args.box_policy)
    walk_policy = NumpyPolicy(args.walk_policy)
    bmeta, wmeta = box_policy.meta, walk_policy.meta
    joint_names = bmeta["joint_names"]
    assert joint_names == wmeta["joint_names"], "policy joint orders differ!"

    box_default = np.array(bmeta["default_joint_pos"], np.float32)
    box_scale = np.array(bmeta["action_scale"], np.float32)
    walk_default = np.array(wmeta["default_joint_pos"], np.float32)
    walk_scale = np.array(wmeta["action_scale"], np.float32)

    box_kp = dict(zip(joint_names, bmeta["joint_stiffness"]))
    box_kd = dict(zip(joint_names, bmeta["joint_damping"]))
    walk_kp = dict(zip(joint_names, wmeta["joint_stiffness"]))
    walk_kd = dict(zip(joint_names, wmeta["joint_damping"]))
    # carry gains: walking gains everywhere EXCEPT arms (keep WBT gains+targets)
    carry_kp = {n: (box_kp[n] if is_arm(n) else walk_kp[n]) for n in joint_names}
    carry_kd = {n: (box_kd[n] if is_arm(n) else walk_kd[n]) for n in joint_names}

    fps = int(bmeta["motion_fps"])
    n_frames = int(bmeta["motion_frames"])
    CONTROL_DT = 1.0 / float(bmeta.get("control_hz", 50))
    assert abs(CONTROL_DT * fps - 1.0) < 1e-6

    if args.hold_frame is not None:
        hold_frame = args.hold_frame
    elif "hold_frame_range" in bmeta:
        h0, h1 = bmeta["hold_frame_range"]
        hold_frame = (int(h0) + int(h1)) // 2
    else:
        raise SystemExit("--hold-frame required (policy meta has no hold_frame_range)")

    walk_ticks = int(args.walk_seconds / CONTROL_DT)

    print("=" * 78)
    print(f"  box policy:   {args.box_policy}  (motion {n_frames} f @ {fps} Hz)")
    print(f"  walk policy:  {args.walk_policy}  (obs {wmeta['obs_dim']})")
    print(f"  hold frame:   {hold_frame}  ({hold_frame / fps:.2f} s into the motion)")
    print(f"  carry:        {args.walk_seconds:.1f} s @ vx={args.vx} m/s "
          f"(~{args.vx * (args.walk_seconds - 2.0):.1f} m)")
    print(f"  MODE:         {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    print("\nBOX: 45 cm cube, LIGHT (0.5-1.5 kg), center ~0.40 m ahead on the heading.")
    print(f"Clear floor needed ahead: ~{args.vx * args.walk_seconds + 1.0:.1f} m.\n")

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

    wbt_obs = WbtObservationBuilder(box_policy, base_imu=args.base_imu)
    walk_obs = WalkObservationBuilder(walk_policy, base_imu=args.base_imu,
                                      grav_bias=args.grav_bias)

    def read_jmap():
        imus, head, waist, arm, leg = client.get_robot_states()
        jmap = {jr.name: jr for jr in (head + waist + arm + leg)}
        missing = [n for n in joint_names if n not in jmap]
        if missing:
            raise RuntimeError(f"State missing joints: {missing}")
        return imus, jmap

    imus0, jmap0 = read_jmap()
    wbt_obs.align(imus0[args.base_imu].quat)
    a0 = box_policy(wbt_obs.build(imus0, jmap0, frame=0))
    print(f"[check] WBT frame-0 |action|_max = {np.abs(a0).max():.3f} (should be O(1))")

    start_ref = {n: float(box_policy.ref_joint_pos[0][i]) for i, n in enumerate(joint_names)}

    print("\n>>> SAFETY: suspended first? MC stopped on .40? E-stop in hand? <<<")
    input(">>> Press Enter to START (Ctrl+C to abort) <<<")

    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)

    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    # ramp -> settle -> pickup -> carry -> setdown -> done
    phase = "ramp"
    phase_t0 = t0
    frame = 0
    walk_tick = 0
    arm_lock: dict[str, float] = {}
    kp_by_name, kd_by_name = box_kp, box_kd

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()

            roll = roll_of(np.asarray(imus[args.base_imu].quat, np.float32))
            if phase in ("pickup", "carry", "setdown") and abs(roll) > args.roll_abort:
                print(f"\n[ABORT] roll {roll:+.2f} rad > {args.roll_abort}. Holding pose.")
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
                    wbt_obs.align(imus[args.base_imu].quat)
                    phase = "pickup"; phase_t0 = now; frame = 0
                    print("\n[phase] PICKUP -- WBT motion clock running\n")

            elif phase == "pickup":
                frame = min(int(elapsed / CONTROL_DT), hold_frame)
                obs = wbt_obs.build(imus, jmap, frame)
                action = box_policy(obs).reshape(-1)
                wbt_obs.last_action = action.astype(np.float32)
                raw_target = action * box_scale + box_default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    step = float(np.clip(raw_target[i] - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step
                if frame >= hold_frame:
                    # freeze the squeeze: lock arm targets, switch leg gains
                    arm_lock = {n: target_by_name[n] for n in joint_names if is_arm(n)}
                    kp_by_name, kd_by_name = carry_kp, carry_kd
                    walk_obs.last_action = np.zeros_like(walk_obs.last_action)
                    # heading lock: remember hand-off yaw, steer back to it
                    x, y, z, w = imus[args.base_imu].quat
                    yaw_lock = float(np.arctan2(2 * (w * z + x * y),
                                                1 - 2 * (y * y + z * z)))
                    phase = "carry"; phase_t0 = now; walk_tick = 0
                    print(f"\n[phase] CARRY -- walking policy, arms locked, "
                          f"{args.walk_seconds:.1f} s @ vx={args.vx}\n")

            elif phase == "carry":
                tcmd = walk_tick * CONTROL_DT
                if tcmd < 1.0:
                    ramp = tcmd
                elif tcmd > args.walk_seconds - 2.0:
                    ramp = max(0.0, (args.walk_seconds - 1.0 - tcmd))
                else:
                    ramp = 1.0
                # heading-hold: wz feedback on IMU yaw keeps the walk straight
                x, y, z, w = imus[args.base_imu].quat
                yaw = float(np.arctan2(2 * (w * z + x * y),
                                       1 - 2 * (y * y + z * z)))
                yaw_err = (yaw_lock - yaw + np.pi) % (2 * np.pi) - np.pi
                wz = float(np.clip(1.2 * yaw_err, -0.4, 0.4))
                command = [args.vx * min(1.0, ramp), 0.0, wz]
                obs = walk_obs.build(imus, jmap, command)
                action = walk_policy(obs).reshape(-1)
                walk_obs.last_action = action.astype(np.float32)
                raw_target = action * walk_scale + walk_default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    if is_arm(n):
                        target_by_name[n] = arm_lock[n]
                        continue
                    step = float(np.clip(raw_target[i] - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step
                walk_tick += 1
                if walk_tick >= walk_ticks:
                    kp_by_name, kd_by_name = box_kp, box_kd
                    # re-align yaw at the hold frame and resume the WBT clock
                    wbt_obs.align(imus[args.base_imu].quat, frame=hold_frame)
                    wbt_obs.last_action = np.zeros_like(wbt_obs.last_action)
                    phase = "setdown"; phase_t0 = now
                    print("\n[phase] SET-DOWN -- WBT clock resumes from the hold\n")

            elif phase == "setdown":
                frame = hold_frame + int(elapsed / CONTROL_DT)
                obs = wbt_obs.build(imus, jmap, frame)
                action = box_policy(obs).reshape(-1)
                wbt_obs.last_action = action.astype(np.float32)
                raw_target = action * box_scale + box_default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    step = float(np.clip(raw_target[i] - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step
                if frame >= n_frames + int(args.hold_end_seconds / CONTROL_DT):
                    phase = "done"; phase_t0 = now
                    print("\n[phase] motion complete -> ramping to default\n")

            else:  # done
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * float(box_default[i])
                                  for i, n in enumerate(joint_names)}

            publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                         args.gain_scale, args.engage)
            prev_target = target_by_name

            if now - last_print >= 1.0:
                last_print = now
                tag = "DRY" if not args.engage else "CMD"
                print(f"[{tag}] phase={phase:7s} t={now - t0:5.1f}s frame={frame:3d}/{n_frames} "
                      f"roll={roll:+.2f} "
                      f"knee_L={target_by_name['left_knee_joint']:+.3f} "
                      f"elbow_L={target_by_name['left_elbow_joint']:+.3f}")

            next_t += CONTROL_DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[interrupt] ramping to default pose and exiting.")
        ramp_start = dict(prev_target)
        default_by_name = dict(zip(joint_names, box_default.tolist()))
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
