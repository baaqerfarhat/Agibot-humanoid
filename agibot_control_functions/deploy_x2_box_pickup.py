#!/usr/bin/env python3
"""Deploy the holosoma X2 box-pickup (whole-body tracking) policy on the real AgiBot humanoid.

Runs INSIDE the ROS 2 environment (needs `rclpy` + `aimdk_msgs`, same as
`robot_states_control.py`). The policy comes from a self-contained `.npz`
produced by `export_box_policy_npz.py`, so the only runtime dependency is
**numpy** (no torch / onnxruntime on the robot).

    -------------------------------------------------------------------------
    PIPELINE (must match holosoma WBT training):
      observation (built here)  ->  policy MLP  ->  action (31)
      target_q = action * action_scale + default_q       (per joint)
      tau      = clip(kp*(target_q - q) - kd*dq, +-effort_limit)
      publish position (clipped to the mechanical range) + the leftover torque
      as `effort`, with the training PD gains, at 50 Hz

    That last step matters more than it looks. Training clips the PD *torque* to
    the effort limit and never clips the position target, and because
    action_scale = 0.25*effort_limit/kp an action of 4 already sits exactly on the
    effort limit. This policy exploits that: it parks ankle_roll, both wrist
    pitches and the shoulder rolls far outside their mechanical range because that
    is how a position-controlled policy asks for maximum torque -- ankle roll is
    saturated on 90% of the motion, the wrist pitches ~80%.

    Clipping those targets to the mechanical limit (what this script did before
    2026-08-12) leaves the servo with almost no position error, so a max-torque
    request becomes a near-zero one. Measured on the 2026-08-12 hardware runs
    (run_logs/_torque_deficit.py) the robot delivered only 10-22% of the intended
    ankle-roll torque, 3-13% of the wrist-pitch grip torque and ~50% of the
    shoulder-roll squeeze torque, for a 20-24% whole-body torque deficit through
    the lift and stand-up. That is why the robot needed a person holding it from
    behind to avoid pitching forward, and why the arms drifted open and dropped
    the box. Sending the discarded part as feed-forward `effort` restores the
    training torque exactly. --no-torque-ff reverts to the old behaviour.

    Delivering that torque faithfully then exposed the next layer of the problem,
    and it is why --action-clip exists. The saturated requests are only safe
    because in training something PUSHES BACK: the ankle rolls press into the
    ground, the wrists into the box. Under 22 Nm of ankle-roll torque Isaac holds
    right_ankle_roll at +0.03 rad; the real robot rolls it to +0.34 (past its
    +0.263 URDF stop) and keeps it there for 96% of the motion, so it stands on
    the edges of its feet, and the same happens to the wrists (left_wrist_roll
    pinned at -1.56 against a reference of 0..+0.72). In the 2026-08-12 13:21/13:22
    runs, the two with no human support, that lateral error grew for ~100 frames
    and the robot toppled at 2.9 s. Bounding |action| to 4 on those joints removes
    only the unachievable part of the request; in Isaac it left survival unchanged
    on 7/7 seeds (box success 6/7 against 7/7 unclipped). It has to be targeted --
    clipping every joint instead fails the task on 0/9, because the legs genuinely
    need their large commands.

    Observation layout (164 dims, holosoma concatenates terms ALPHABETICALLY):
        [ prev_action(31),
          base_ang_vel(3),                           <- PELVIS gyro, see below
          joint_pos - default(31), joint_vel(31),
          ref_joint_pos(31), ref_joint_vel(31),      <- motion clock (from npz)
          motion_ref_ori_b(6) ]                      <- ref torso ori rel. to IMU torso ori
    -------------------------------------------------------------------------

    base_ang_vel is the PELVIS rate, not the torso IMU's. holosoma builds it from
    the articulation root (the pelvis freejoint body) in the pelvis frame, and the
    torso sits three waist joints above that. Feeding the raw torso gyro was the
    root cause of the v33 hardware roll failure: reproducing that exact
    substitution in Isaac (`adaptation/obs_frame_isaac.py`, 5 seeds) drops the
    policy from 100% success / 9.8 s survival to 0% / 1.6 s, falling sideways in
    the same 1.1-1.9 s window the robot did. `base_frame.PelvisEstimator` composes
    the torso IMU with the measured waist joints to recover the pelvis rate, which
    removes 90.8% of the observation error and restores 100% success in that same
    test. Use --base-ang-vel torso only to reproduce the old (broken) behaviour.

    motion_ref_ori_b is NOT affected: holosoma's x2 config sets
    body_name_ref=["torso_link"], so the raw IMU quat is the correct input there.

    The policy is BLIND: it does not perceive the box. The box must be placed
    at the reference start location (see "Box placement" below) before engaging.

    The motion (v25: IN-PLACE pickup + hold + set-down, NO walking) is 8.1 s
    at 50 Hz, feet planted at the stance position for the ENTIRE motion:
        0.0 - 0.7 s   hold the default upright pose  <- START THE ROBOT STANDING
        0.7 - 2.7 s   bend down toward the box
        2.7 - 3.2 s   two-handed squeeze grasp + lift to chest
        3.2 - 5.2 s   HOLD: stand still, box at chest (2 s)
        5.2 - 7.4 s   one continuous set-down directly in front, both feet
                      planted (the pickup played in reverse), stand back up
        7.4 - 8.1 s   hold the default upright pose
    Both the first and last frames of the reference are the robot's exact
    default standing pose at zero velocity, so the task starts from a normal
    standing start and finishes standing upright. The box is picked up and
    set back down at the SAME spot -- no floor clearance needed beyond the
    box position. (This isolated pickup/set-down is exactly what the hybrid
    controller `deploy_x2_box_hybrid.py` runs, minus the carry walk it
    splices into the hold.)

#####################################  SAFETY  #####################################
#  1. First runs: robot SUSPENDED, NO BOX -> verify the motion shape in the air.
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default mode is DRY-RUN: computes & logs commands, DOES NOT publish.
#     Add  --engage  only once dry-run output looks sane.
#  4. Escalation: dry-run -> suspended (--engage, no box) -> gantry on ground,
#     no box -> gantry with box -> free. Do NOT skip steps.
#  5. This task has a DEEP FORWARD BEND. Tilt-abort is wider than for walking
#     (--tilt-abort applies to ROLL only; pitch is part of the motion).
#  6. Keep a hand on the e-stop. Ctrl+C ramps back to a safe pose and exits.
#  7. When done, restart the controller:   aima em start-app mc
####################################################################################

Box placement: the reference box start pose (motion frame 0) is ~0.40 m in
front of the robot's initial pelvis position (box CENTER, i.e. near edge
~0.17 m from the robot), centered on its heading, resting on the floor
(45 cm cube, LIGHT: 0.5-1.5 kg; training randomized 0.4-1.6 kg and friction --
an empty/lightly-filled cardboard box is ideal; do NOT use a heavy box, the
wrist actuators cannot squeeze-hold much beyond ~2 kg).
Mark the robot's start feet position and the box position together.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np
import rclpy

from base_frame import PelvisEstimator
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
    """Same forward pass as the walking deployment (rsl_rl actor, ELU)."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))
        # Reference motion (50 Hz frames).
        self.ref_joint_pos = d["ref_joint_pos"].astype(np.float32)   # (T, 31)
        self.ref_joint_vel = d["ref_joint_vel"].astype(np.float32)   # (T, 31)
        self.ref_quat_xyzw = d["ref_quat_xyzw"].astype(np.float32)   # (T, 4)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)  # ELU(alpha=1)
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
    """Yaw-only component of q (xyzw)."""
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32)


def roll_of(q: np.ndarray) -> float:
    x, y, z, w = q
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


# =============================== observation builder ===============================
class WbtObservationBuilder:
    """Builds the 164-dim WBT actor observation.

    motion_ref_ori_b matches holosoma's `subtract_frame_transforms` +
    first-two-rotation-matrix-columns encoding: q_rel = q_torso^-1 * q_ref,
    flattened row-major from R[:, :2] -> [m00, m01, m10, m11, m20, m21].

    The motion was authored in its own world frame; at engage time we compute a
    yaw offset aligning motion frame 0 with the robot's current heading, and
    apply it to every reference quat (the robot may face any direction).
    """

    def __init__(self, policy: NumpyPolicy, base_imu: str, use_pelvis_ang_vel: bool = True):
        meta = policy.meta
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.action_dim = int(meta["action_dim"])
        self.base_imu = base_imu
        self.policy = policy
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.yaw_offset = np.array([0, 0, 0, 1], np.float32)  # set by align()
        # Always reconstructed: the pelvis attitude is the right signal for the
        # roll abort either way. The flag only decides what the POLICY is fed.
        self.pelvis_est = PelvisEstimator()
        self.use_pelvis_ang_vel = bool(use_pelvis_ang_vel)
        self.last_pelvis_quat = np.array([0, 0, 0, 1], np.float32)
        self.last_base_ang_vel = np.zeros(3, np.float32)

    def align(self, imu_quat_xyzw) -> None:
        """Rotate the reference trajectory into the robot's current heading."""
        q_robot_yaw = yaw_quat(np.asarray(imu_quat_xyzw, np.float32))
        q_ref0_yaw = yaw_quat(self.policy.ref_quat_xyzw[0])
        self.yaw_offset = quat_mul(q_robot_yaw, quat_inv(q_ref0_yaw))

    def build(self, imus, jmap, frame: int) -> np.ndarray:
        T = self.policy.ref_joint_pos.shape[0]
        frame = min(frame, T - 1)
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        imu = imus[self.base_imu]

        # motion_ref_ori_b tracks torso_link in training, so it uses the raw IMU
        # quat. Only base_ang_vel needs the pelvis.
        q_ref = quat_mul(self.yaw_offset, self.policy.ref_quat_xyzw[frame])
        q_rel = quat_mul(quat_inv(np.asarray(imu.quat, np.float32)), q_ref)
        ori6 = quat_to_mat(q_rel)[:, :2].reshape(-1)

        w_pelvis, self.last_pelvis_quat = self.pelvis_est.update(imu.quat, imu.ang_vel, jmap)
        base_ang_vel = w_pelvis if self.use_pelvis_ang_vel else np.asarray(imu.ang_vel, np.float32)
        self.last_base_ang_vel = base_ang_vel

        # Alphabetical term order (matches holosoma's group concatenation):
        # actions, base_ang_vel, dof_pos, dof_vel, motion_command, motion_ref_ori_b
        return np.concatenate(
            [
                self.last_action,
                base_ang_vel,
                q - self.default,
                dq,
                self.policy.ref_joint_pos[frame],
                self.policy.ref_joint_vel[frame],
                ori6,
            ]
        ).astype(np.float32)


# =============================== command helpers ===============================
CONTROLLED_AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)


def build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name, gain_scale,
                   jmap=None, eff_by_name=None):
    """One JointCommandArray for `area`, reproducing training's torque exactly.

    Training clips the PD torque to the effort limit and never clips the position
    target, so a target parked outside the mechanical range is a max-torque
    request (see the module docstring). `position` here still stays inside the
    range, which the firmware expects; the part that the clip would have thrown
    away is handed over as `effort`, the feed-forward torque, so the low level's
    `tau = effort + kp*(position - q) + kd*(velocity - dq)` lands on the training
    torque. `gain_scale` scales that whole torque, so 1.0 reproduces training and
    lower values stay uniformly gentler.

    With `eff_by_name=None` this degrades to the old position-only behaviour.
    """
    cmd = JointCommandArray()
    ff = {}
    for ji in robot_model[area]:
        n = ji.name
        des = float(pos_by_name[n])
        pos = float(np.clip(des, ji.lower_limit, ji.upper_limit))
        kp = float(kp_by_name[n])
        kd = float(kd_by_name[n])

        tau_ff = 0.0
        if eff_by_name is not None and jmap is not None and n in jmap:
            q = float(jmap[n].position)
            dq = float(jmap[n].velocity)
            eff = float(eff_by_name[n])
            tau_train = float(np.clip(kp * (des - q) - kd * dq, -eff, eff))
            # what the servo will make on its own from the in-limit target
            tau_pd = kp * (pos - q) - kd * dq
            tau_ff = tau_train - tau_pd

        jc = JointCommand()
        jc.name = n
        jc.position = pos
        jc.velocity = 0.0
        jc.effort = float(tau_ff * gain_scale)
        jc.stiffness = float(kp * gain_scale)
        jc.damping = float(kd * gain_scale)
        cmd.joints.append(jc)
        ff[n] = jc.effort
    return cmd, ff


def publish_pose(commander, pos_by_name, kp_by_name, kd_by_name, gain_scale, engage,
                 jmap=None, eff_by_name=None):
    """Publish every controlled area; returns the feed-forward torque per joint."""
    ff = {}
    for area in CONTROLLED_AREAS:
        cmd, area_ff = build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name,
                                      gain_scale, jmap=jmap, eff_by_name=eff_by_name)
        ff.update(area_ff)
        if engage:
            commander.publish(area, cmd)
    return ff


# =============================== main ===============================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy",
                    default="../box_pickup/policy/x2_box_policy_clean_grasp_iter9000.npz",
                    help="Path to WBT box policy .npz (default: clean-grasp retrain, iter 9000). "
                         "Trained with the same +-4 bound on ankle_roll/wrist that --action-clip "
                         "applies here, so the clip is not imposed after the fact on a policy "
                         "that had learned to rely on saturation. Waist pitch/yaw, planted feet, "
                         "and keeping the HANDS OFF THE FLOOR were objectives during training "
                         "rather than later patches: the previous policy planted both palms on "
                         "the ground during the grasp and pushed off them, which this one never "
                         "does (0 of 733 frames in floor contact, vs 103).")
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"],
                    help="IMU used as the policy base (training ref body = torso_link).")
    ap.add_argument("--base-ang-vel", default="pelvis", choices=["pelvis", "torso"],
                    help="Source for the base_ang_vel observation. 'pelvis' (default) "
                         "reconstructs the pelvis rate from the torso IMU plus the "
                         "waist joints, which is what training used. 'torso' feeds the "
                         "raw IMU gyro -- the v33 defect, kept only for A/B testing.")
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains (lower = gentler).")
    ap.add_argument("--ramp-seconds", type=float, default=5.0,
                    help="Time to ramp from current pose to the motion start pose.")
    ap.add_argument("--settle-seconds", type=float, default=2.0,
                    help="Hold the start pose before engaging the policy.")
    ap.add_argument("--no-torque-ff", action="store_true",
                    help="Send position-only commands, clipped to the mechanical joint "
                         "limits, with effort=0 -- the pre-2026-08-12 behaviour. This "
                         "silently discards every saturated-torque request the policy "
                         "makes (ankle roll, wrist pitch, shoulder roll); keep it only "
                         "for A/B testing against the old runs.")
    ap.add_argument("--action-clip", type=float, default=4.0,
                    help="Bound |action| on --action-clip-joints. |a|=4 is exactly the "
                         "effort limit, so this drops only requests for torque the "
                         "actuator cannot produce. In training those requests are "
                         "absorbed by contact; on hardware they drive the joint to its "
                         "mechanical stop. 0 disables (pre-2026-08-12 behaviour).")
    ap.add_argument("--action-clip-joints", default="ankle_roll,wrist",
                    help="Comma-separated substrings. Only the joints whose saturated "
                         "torque leans on contact belong here. Clipping every joint "
                         "fails the task on 6/6 Isaac seeds.")
    ap.add_argument("--max-joint-step", type=float, default=0.0,
                    help="Max change in a joint target per 20 ms tick (rad); 0 = off "
                         "(default). Training has no such rate limit, so any value here "
                         "is pure added lag: on the 2026-08-12 runs 0.15 clipped 6.4%% of "
                         "ticks and put the knee target up to 0.34 rad behind the "
                         "policy. Torque is already bounded by the effort limit, which "
                         "is the actual safety quantity, so this is not needed.")
    ap.add_argument("--roll-abort", type=float, default=0.7,
                    help="Abort if |pelvis roll| exceeds this (rad). Pitch is NOT "
                         "checked: the motion contains a deep forward bend. Watches the "
                         "pelvis (not the torso) so that commanded waist roll -- up to "
                         "0.49 rad of it -- does not eat the abort margin.")
    ap.add_argument("--hold-end-seconds", type=float, default=3.0,
                    help="Hold the final reference pose after the motion ends.")
    ap.add_argument("--leg-filter", type=float, default=0.0,
                    help="EMA smoothing factor on LEG joint targets (0 = off, default). "
                         "This has no counterpart in training, so it is pure added lag; "
                         "adaptation/deploy_artifacts_isaac.py measures the cost in sim "
                         "(3 seeds): 0.0 and 0.2 both give 100%% success, 0.5 drops to "
                         "67%%, and 0.9 gives 0%%, matching the hardware run that aborted "
                         "at 0.9 while 0.2 completed the motion. On hardware 0.0 got "
                         "furthest, and its logged targets sit 0.0008 rad from the raw "
                         "policy output vs 0.036 rad at 0.9. Arms are never filtered.")
    ap.add_argument("--init-tol-arm", type=float, default=0.12,
                    help="Max |meas-start| (rad) allowed on arm joints before "
                         "the policy is allowed to engage.")
    ap.add_argument("--init-tol-leg", type=float, default=0.25,
                    help="Max |meas-start| (rad) allowed on leg joints before "
                         "the policy is allowed to engage.")
    ap.add_argument("--init-timeout", type=float, default=20.0,
                    help="Seconds to wait after settle for the start pose to "
                         "be reached before aborting engage.")
    ap.add_argument("--force-engage", action="store_true",
                    help="Engage even if init pose tolerances are not met.")
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

    # Per-joint torque limit training clipped to. Newer exports carry it; for the
    # older npz files recover it from the scale relation the training config used,
    # action_scale = 0.25 * effort_limit / kp  (verified exact on all 31 joints of
    # v33 against dof_effort_limit_list).
    if "joint_effort_limit" in meta:
        eff_limit = np.array(meta["joint_effort_limit"], np.float32)
    else:
        eff_limit = 4.0 * action_scale * np.array(meta["joint_stiffness"], np.float32)
    eff_by_name = None if args.no_torque_ff else dict(zip(joint_names, eff_limit.tolist()))

    fps = int(meta["motion_fps"])
    n_frames = int(meta["motion_frames"])
    CONTROL_DT = 1.0 / float(meta.get("control_hz", 50))
    assert abs(CONTROL_DT * fps - 1.0) < 1e-6, "control rate must match motion fps"

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  task:          {meta.get('task')}   run: {meta.get('run_path', '?')}")
    print(f"  obs terms:     {meta['observation_names']}  (dim {meta['obs_dim']})")
    print(f"  motion:        {n_frames} frames @ {fps} Hz = {n_frames / fps:.1f} s")
    print(f"  action joints: {len(joint_names)}   gain scale: {args.gain_scale}")
    print(f"  torque ff:     {'OFF -- position clipped to limits (OLD, known-bad)' if args.no_torque_ff else 'on (matches training clip_torques)'}")
    print(f"  action clip:   "
          f"{'OFF' if args.action_clip <= 0 else f'+-{args.action_clip:g} on {args.action_clip_joints}'}")
    print(f"  leg filter:    {args.leg_filter}   max joint step: "
          f"{'off' if args.max_joint_step <= 0 else f'{args.max_joint_step} rad/tick'}")
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    print("\nBOX PLACEMENT: 45 cm cube, LIGHT (0.5-1.5 kg), on the floor, center")
    print("~0.40 m in front of the robot, on its heading. Start the robot STANDING")
    print("UPRIGHT; the motion holds still for ~1 s, then bends down.")
    print("First trials: NO BOX, robot suspended.\n")

    rclpy.init()
    client = RobotStateClient()
    commander = WholeBodyCommander()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(commander)
    threading.Thread(target=executor.spin, daemon=True).start()

    # Only require the IMU the policy actually uses (base_imu). The chest IMU is
    # an unused, sometimes non-publishing sensor and must not block readiness.
    if not client.wait_ready(timeout_sec=10.0, required_imus=[args.base_imu]):
        print("[ERROR] state topics not ready.")
        client.destroy_node(); commander.destroy_node(); rclpy.shutdown(); return

    obs_builder = WbtObservationBuilder(policy, base_imu=args.base_imu,
                                        use_pelvis_ang_vel=(args.base_ang_vel == "pelvis"))

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
    print(f"[check] frame-0 obs built (dim {obs0.shape[0]}), "
          f"|action|_max = {np.abs(a0).max():.3f} (should be O(1))")
    print(f"[check] torso roll = {roll_of(np.asarray(imus0[args.base_imu].quat)):+.3f} rad   "
          f"pelvis roll = {roll_of(obs_builder.last_pelvis_quat):+.3f} rad")
    print(f"[check] base_ang_vel source = {args.base_ang_vel}"
          + ("" if args.base_ang_vel == "pelvis" else "   <-- KNOWN-BAD, A/B ONLY"))
    # Standing upright the waist is ~0, so the reconstruction must be a no-op.
    # A large delta here means the IMU is not aligned with torso_link.
    d = float(np.abs(np.asarray(obs_builder.pelvis_est.last_correction)).max())
    waist0 = {n: float(jmap0[n].position) for n in ("waist_yaw_joint", "waist_pitch_joint",
                                                   "waist_roll_joint")}
    print(f"[check] waist at start = " + "  ".join(f"{k.split('_')[1]}={v:+.3f}"
                                                  for k, v in waist0.items())
          + f"   |pelvis-torso gyro| = {d:.3f} rad/s")
    if max(abs(v) for v in waist0.values()) < 0.05 and d > 0.15:
        print("[WARN] waist is ~0 but the correction is large -- check the IMU mounting "
              "frame before engaging.")

    # Which joints this policy drives past their mechanical limit on purpose, i.e.
    # which ones need the feed-forward torque to behave like training. Swept over
    # the whole reference so the operator sees it before engaging, not after.
    LIMIT_BY_NAME = {ji.name: (ji.lower_limit, ji.upper_limit)
                     for area in CONTROLLED_AREAS for ji in robot_model[area]}
    ref_tgt = policy.ref_joint_pos  # (T, 31); the policy tracks this closely
    sat = []
    for i, n in enumerate(joint_names):
        lo_i, hi_i = LIMIT_BY_NAME[n]
        over = np.maximum(ref_tgt[:, i] - hi_i, lo_i - ref_tgt[:, i])
        if over.max() > 1e-3:
            sat.append((n, float(over.max()), 100.0 * float((over > 0).mean())))
    print(f"[check] torque feed-forward = {'OFF' if args.no_torque_ff else 'ON'}"
          f"   effort limits {eff_limit.min():.1f}-{eff_limit.max():.1f} Nm")
    if sat:
        print("[check] reference itself leaves the mechanical range on "
              f"{len(sat)} joint(s) -- these need the feed-forward:")
        for n, o, f in sorted(sat, key=lambda kv: -kv[2])[:6]:
            print(f"          {n:30s} up to {o:.2f} rad past the limit, {f:.0f}% of frames")

    # Ramp / init target: the motion's own first frame (policy-correct start).
    start_ref = {n: float(policy.ref_joint_pos[0][i]) for i, n in enumerate(joint_names)}

    # Bound the action on the joints whose saturated torque is absorbed by CONTACT in
    # training -- the ankle rolls press into the ground, the wrists into the box. Sim
    # keeps them at the reference under that torque (right_ankle_roll sits at +0.03 rad);
    # hardware cannot, and pins them to their stops (+0.34 rad for 96% of the motion,
    # left_wrist_roll at -1.56), which rolls the feet onto their edges and topples the
    # robot at ~2.9 s. |a| = 4 is exactly the effort limit, so this removes only requests
    # for torque no actuator can produce. Isaac, 7 seeds: survival unchanged from the
    # unclipped baseline on 7/7, box success 6/7. Clipping ALL joints instead succeeds
    # on 0/9 -- the legs need their large commands. See
    # adaptation/action_clip_isaac.py and run_logs/_sim_vs_real.py.
    if args.action_clip > 0.0:
        act_clip_mask = np.array(
            [any(k in n for k in args.action_clip_joints.split(",")) for n in joint_names])
        print(f"[check] action clip = +-{args.action_clip:g} on "
              f"{int(act_clip_mask.sum())} joint(s) matching "
              f"'{args.action_clip_joints}'  (|a|=4 is the effort limit)")
    else:
        act_clip_mask = None
        print("[check] action clip = OFF -- saturated ankle-roll/wrist torque will be "
              "delivered in full; hardware pins those joints to their stops")
    act_clip_ticks = 0

    LEG_JOINTS = [n for n in joint_names if ("hip" in n or "knee" in n or "ankle" in n)]
    LEG_SET = set(LEG_JOINTS)
    ARM_JOINTS = [n for n in joint_names
                  if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    ARM_SET = set(ARM_JOINTS)

    def pose_errors(jmap):
        return {n: float(jmap[n].position - start_ref[n]) for n in joint_names}

    def worst_err(err, names):
        return max(((n, err[n]) for n in names), key=lambda kv: abs(kv[1]))

    def pose_ok(err):
        arm_bad = [n for n in ARM_JOINTS if abs(err[n]) > args.init_tol_arm]
        leg_bad = [n for n in LEG_JOINTS if abs(err[n]) > args.init_tol_leg]
        return arm_bad, leg_bad

    # ---- init pose check (before ramp) ---------------------------------------
    # Frame-0 reference: standing, arms slightly open (L roll +0.2 / R roll -0.2),
    # elbows ~-0.3. Planted feet cannot slide, so fix stance by hand first.
    err0 = pose_errors(jmap0)
    print("\n[init] target start pose = motion frame 0 (policy-correct):")
    print("       feet parallel ~27 cm apart; arms slightly abducted; elbows ~-0.3")
    print("[init] ARM joints vs start:")
    for n in ARM_JOINTS:
        flag = "  <-- LARGE" if abs(err0[n]) > args.init_tol_arm else ""
        print(f"    {n:32s} {err0[n]:+.3f} rad   target={start_ref[n]:+.3f}{flag}")
    print("[init] LEG joints vs start:")
    for n in LEG_JOINTS:
        flag = "  <-- LARGE" if abs(err0[n]) > args.init_tol_leg else ""
        print(f"    {n:32s} {err0[n]:+.3f} rad   target={start_ref[n]:+.3f}{flag}")
    arm_bad0, leg_bad0 = pose_ok(err0)
    if arm_bad0 or leg_bad0:
        print("[init] WARNING: not at start pose yet. Ramp/settle will pull toward it;")
        print("[init] policy will NOT engage until arms/legs are within tolerance")
        print("[init] (or use --force-engage). Fix feet by hand if on the ground.")
    # --------------------------------------------------------------------------

    print("\n>>> SAFETY: robot suspended? MC stopped on .40? E-stop in hand? <<<")
    input(">>> Press Enter to START ramp-to-init (Ctrl+C to abort) <<<")

    run_name = f"box_pickup_{os.path.splitext(os.path.basename(args.policy))[0]}"
    PELVIS_COLS = ["pelvis_roll", "pelvis_ang_vel_x", "pelvis_ang_vel_y", "pelvis_ang_vel_z",
                   "obs_ang_vel_x", "obs_ang_vel_y", "obs_ang_vel_z"]
    logger = RunLogger(
        joint_names, base_imu=args.base_imu, run_name=run_name,
        meta={"script": "deploy_x2_box_pickup.py", "policy": args.policy,
              "gain_scale": args.gain_scale, "leg_filter": args.leg_filter,
              "max_joint_step": args.max_joint_step,
              "torque_ff": not args.no_torque_ff,
              "action_clip": args.action_clip,
              "action_clip_joints": args.action_clip_joints,
              "joint_effort_limit": eff_limit.tolist(),
              "engage": args.engage, "task": meta.get("task"),
              "base_ang_vel_source": args.base_ang_vel,
              "run_path": meta.get("run_path")},
        log_dir=args.log_dir, enabled=not args.no_log,
        extra_columns=PELVIS_COLS, log_effort=True)

    def pelvis_extra(q_pelvis, w_pelvis, obs_w):
        """Both the reconstruction and what the policy actually consumed, so an
        A/B run can be told apart from its log alone."""
        return dict(zip(PELVIS_COLS,
                        [roll_of(np.asarray(q_pelvis, np.float32)),
                         *np.asarray(w_pelvis, float).tolist(),
                         *np.asarray(obs_w, float).tolist()]))

    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)
    # Stronger gains while pulling into the start pose so arms actually arrive.
    init_gain = max(float(args.gain_scale), 1.0)

    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    # ramp -> settle -> wait_init -> policy -> done
    phase = "ramp"
    phase_t0 = t0
    frame = 0

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()

            # ---- safety: roll abort (pitch is part of the motion, roll is not) ----
            # Watch the PELVIS: waist roll is commanded by the motion and would
            # otherwise consume the margin. Recomputed fresh here rather than
            # reusing the observation's value, which is one tick old.
            imu_now = imus[args.base_imu]
            w_pelvis_now, q_pelvis_now = obs_builder.pelvis_est.update(
                imu_now.quat, imu_now.ang_vel, jmap)
            obs_w_now = (w_pelvis_now if obs_builder.use_pelvis_ang_vel
                         else np.asarray(imu_now.ang_vel, np.float32))
            roll = roll_of(q_pelvis_now)
            torso_roll = roll_of(np.asarray(imu_now.quat, np.float32))
            if phase == "policy" and abs(roll) > args.roll_abort:
                print(f"\n[ABORT] pelvis roll {roll:+.2f} rad exceeds {args.roll_abort} "
                      f"(torso {torso_roll:+.2f}). Holding pose.")
                phase = "done"; phase_t0 = now

            elapsed = now - phase_t0
            gain_now = args.gain_scale

            if phase == "ramp":
                gain_now = init_gain
                alpha = min(1.0, elapsed / max(1e-3, args.ramp_seconds))
                target_by_name = {n: (1 - alpha) * start_pose[n] + alpha * start_ref[n]
                                  for n in joint_names}
                if alpha >= 1.0:
                    phase = "settle"; phase_t0 = now
                    print("\n[phase] settle -- holding start pose\n")

            elif phase == "settle":
                gain_now = init_gain
                target_by_name = dict(start_ref)
                if elapsed >= args.settle_seconds:
                    phase = "wait_init"; phase_t0 = now
                    print("[phase] wait_init -- verifying measured pose == start "
                          f"(arm tol {args.init_tol_arm:.2f}, leg tol {args.init_tol_leg:.2f})\n")

            elif phase == "wait_init":
                gain_now = init_gain
                target_by_name = dict(start_ref)
                err = pose_errors(jmap)
                arm_bad, leg_bad = pose_ok(err)
                ready = (not arm_bad and not leg_bad) or args.force_engage
                if ready:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"[init] READY  worst_arm {wa[0]}={wa[1]:+.3f}  "
                          f"worst_leg {wl[0]}={wl[1]:+.3f}")
                    if args.force_engage and (arm_bad or leg_bad):
                        print("[init] --force-engage: starting despite residual error")
                    obs_builder.align(imus[args.base_imu].quat)
                    phase = "policy"; phase_t0 = now; frame = 0
                    print("\n[phase] policy ENGAGED -- motion clock running\n")
                elif elapsed >= args.init_timeout:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"\n[ABORT] start pose not reached in {args.init_timeout:.0f}s.")
                    print(f"[ABORT] worst_arm {wa[0]}={wa[1]:+.3f}  "
                          f"worst_leg {wl[0]}={wl[1]:+.3f}")
                    print("[ABORT] Reposition (especially feet), then rerun. "
                          "Or pass --force-engage.")
                    phase = "done"; phase_t0 = now

            elif phase == "policy":
                frame = int(elapsed / CONTROL_DT)
                obs = obs_builder.build(imus, jmap, frame)
                action = policy(obs).reshape(-1)
                if act_clip_mask is not None:
                    action = np.where(act_clip_mask,
                                      np.clip(action, -args.action_clip,
                                              args.action_clip),
                                      action)
                    act_clip_ticks += int(np.any(
                        act_clip_mask & (np.abs(action) >= args.action_clip - 1e-9)))
                # Feed back the action actually applied, not the raw one.
                obs_builder.last_action = action.astype(np.float32)
                raw_target = action * action_scale + default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    tgt = raw_target[i]
                    # low-pass the LEG targets: damps the jitter loop where a
                    # small correction -> actuator overshoot -> IMU motion ->
                    # harder correction. The reference motion itself is slow,
                    # so mild lag on the legs is harmless.
                    if args.leg_filter > 0.0 and n in LEG_SET:
                        tgt = (1.0 - args.leg_filter) * tgt \
                            + args.leg_filter * prev_target[n]
                    if args.max_joint_step > 0.0:
                        tgt = prev_target[n] + float(np.clip(
                            tgt - prev_target[n],
                            -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = float(tgt)
                if frame >= n_frames + int(args.hold_end_seconds / CONTROL_DT):
                    phase = "done"; phase_t0 = now
                    print("\n[phase] motion complete -> ramping to default\n")

            else:  # done
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * float(default[i])
                                  for i, n in enumerate(joint_names)}

            ff = publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                              gain_now, args.engage, jmap=jmap, eff_by_name=eff_by_name)
            prev_target = target_by_name
            logger.log(now - t0, phase, frame, imus, jmap, target_by_name,
                       extra=pelvis_extra(q_pelvis_now, w_pelvis_now, obs_w_now),
                       effort_cmd=ff)

            if now - last_print >= 1.0:
                last_print = now
                tag = "DRY" if not args.engage else "CMD"
                extra = ""
                if phase == "wait_init":
                    err = pose_errors(jmap)
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    extra = (f" arm_err={wa[0].split('_')[1][:4]}{wa[1]:+.2f}"
                             f" leg_err={wl[0].split('_')[1][:4]}{wl[1]:+.2f}")
                elif eff_by_name is not None and ff:
                    # the saturated-torque request the old position-only command
                    # threw away; expect ankle_roll / wrist_pitch here once bending
                    wn, wv = max(ff.items(), key=lambda kv: abs(kv[1]))
                    extra = f" ff_max={wn.replace('_joint', '')}{wv:+.1f}Nm"
                    if act_clip_mask is not None and phase == "policy":
                        extra += f" aclip={100.0 * act_clip_ticks / max(frame, 1):.0f}%"
                print(f"[{tag}] phase={phase:9s} t={now - t0:5.1f}s frame={frame:3d}/{n_frames} "
                      f"roll={roll:+.2f}(pelvis) "
                      f"knee_L={target_by_name['left_knee_joint']:+.3f} "
                      f"elbow_L={target_by_name['left_elbow_joint']:+.3f}"
                      f"{extra}")

            next_t += CONTROL_DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[interrupt] ramping to default pose and exiting.")
        ramp_start = dict(prev_target)
        default_by_name = dict(zip(joint_names, default.tolist()))
        t_stop = time.perf_counter()
        while time.perf_counter() - t_stop < 1.5 and rclpy.ok():
            a = min(1.0, (time.perf_counter() - t_stop) / 1.5)
            tgt = {n: (1 - a) * ramp_start[n] + a * default_by_name[n] for n in joint_names}
            try:
                imus_i, jmap_i = read_jmap()
            except Exception:
                imus_i = jmap_i = None
            ff = publish_pose(commander, tgt, kp_by_name, kd_by_name, args.gain_scale,
                              args.engage, jmap=jmap_i, eff_by_name=eff_by_name)
            if jmap_i is not None:
                logger.log(time.perf_counter() - t0, "interrupt", frame, imus_i, jmap_i,
                           tgt, effort_cmd=ff)
            time.sleep(CONTROL_DT)
    finally:
        logger.close()
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
