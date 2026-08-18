#!/usr/bin/env python3
"""Deploy the holosoma X2 slope-crawl (whole-body tracking) policy on the real AgiBot.

Runs INSIDE the ROS 2 environment (needs `rclpy` + `aimdk_msgs`, same as
`robot_states_control.py`). The policy comes from a self-contained `.npz`
produced by `export_crawl_policy_npz.py`, so the only runtime dependency is
**numpy** (no torch / onnxruntime on the robot).

This is the crawl counterpart of `deploy_x2_box_pickup.py` and carries the same
three hardware corrections that script earned the hard way -- torque
feed-forward, pelvis-frame observations, and a targeted action bound. The
observation layout and the end-of-motion handling are what differ.

    -------------------------------------------------------------------------
    PIPELINE (must match holosoma crawl WBT training):
      observation (built here)  ->  policy MLP  ->  action (31)
      target_q = action * action_scale + default_q       (per joint)
      tau      = clip(kp*(target_q - q) - kd*dq, +-effort_limit)
      publish position (clipped to the mechanical range) + the leftover torque
      as `effort`, with the training PD gains, at 50 Hz

    Observation layout (167 dims, holosoma concatenates terms ALPHABETICALLY):
        [ prev_action(31),
          base_ang_vel(3),                           <- PELVIS gyro, see below
          joint_pos - default(31), joint_vel(31),
          ref_joint_pos(31), ref_joint_vel(31),      <- motion clock (from npz)
          motion_ref_ori_b(6),                       <- ref TORSO ori rel. IMU
          projected_gravity(3) ]                     <- gravity in PELVIS frame
    -------------------------------------------------------------------------

    TWO of these terms are in the PELVIS frame, not the torso IMU's.
    holosoma builds both `base_ang_vel` and `projected_gravity` from
    `env.base_quat`, which is the articulation root -- the pelvis freejoint body
    -- and the torso sits three waist joints above it. Feeding the raw torso gyro
    was the root cause of the v33 box-pickup hardware roll failure (reproducing
    that substitution in Isaac dropped it from 100% success to 0%), so the same
    substitution is avoided here. It matters MORE for the crawl, because
    projected gravity is this policy's only attitude signal and the crawl swings
    the waist through 0.5 rad of yaw, 0.3 of pitch and 0.5 of roll. Measured on
    the recorded v5 rollout (adaptation/verify_crawl_export.py), the raw torso
    IMU misreads the body attitude by 14.4 deg on average and up to 28.3 deg,
    which the policy would read as the robot rolling off the ramp.
    `base_frame.PelvisEstimator` composes the torso IMU with the measured waist
    joints to recover the pelvis rate and attitude; scored against the true
    pelvis over that same rollout it leaves 0.03 deg, removing 99.8% of the
    error. Rebuilding the observation this way also reproduces Isaac's recorded
    actions better than any other frame choice. Use --base-frame torso only to
    reproduce the old (broken) behaviour.

    `motion_ref_ori_b` is NOT affected: holosoma's x2 config sets
    body_name_ref=["torso_link"], so the raw IMU quat is the correct input there.
    Every observation term has scale 1.0 in this run's config, so nothing here
    is scaled.

    Torque feed-forward (the same fix as the box script): training clips the PD
    torque to the effort limit and never clips the position target, and because
    action_scale = 0.25*effort_limit/kp an action of 4 already sits exactly on
    the effort limit. This policy leans on that even harder than the box one --
    measured over the recorded v5 rollout, |action| >= 4 on 88% of wrist frames,
    69% of ankle-roll frames and 39% of all (frame, joint) pairs. Clipping those
    targets to the mechanical limit leaves the servo with almost no position
    error, so a max-torque request becomes a near-zero one. Sending the
    discarded part as feed-forward `effort` restores the training torque exactly.
    --no-torque-ff reverts to the old behaviour.

    That is also why --action-clip exists, and why it is targeted. The saturated
    requests are only safe because something PUSHES BACK in training: the palms
    press into the slope, the feet into the ramp. Sim holds the joint at the
    reference under that torque; hardware, if the contact is any softer, drives
    it to its mechanical stop instead and crawls on the edges of its feet and
    the backs of its hands. |a| = 4 is exactly the effort limit, so bounding it
    removes only the unachievable part of the request. It must stay targeted:
    on the box task, clipping every joint instead failed on 0/9 Isaac seeds
    because the legs genuinely need their large commands. NOTE: the Isaac sweep
    behind that default was run on the box task, not this one -- treat the crawl
    default as the same reasoning applied to the same joints, not as separately
    validated. The head also saturates (~89% of frames) but is deliberately NOT
    clipped: nothing pushes back on the head, so sim and hardware already agree.

    The motion (crawl_slope_palmflat_mj) is 19.2 s at 50 Hz, PALMS DOWN, and it
    advances ~2.9 m up the slope in +Y while rising ~0.4 m:
        0.0 -  3.0 s   settle from the crawl start pose onto hands and feet
        3.0 - 16.0 s   crawl up the incline, palms flat, torso near-horizontal
       16.0 - 18.0 s   rise toward upright at the top of the ramp
       18.0 - 19.2 s   upright (pelvis ~1.10 m, torso vertical)
    The reference therefore STARTS PRONE and ENDS STANDING. Frame 0 is already
    the crawl pose -- do NOT start the robot standing upright.

#####################################  SAFETY  #####################################
#  1. First runs: robot SUSPENDED / gantry, NO floor contact -> verify the shape.
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default mode is DRY-RUN: computes & logs commands, DOES NOT publish.
#     Add  --engage  only once dry-run output looks sane.
#  4. Escalation: dry-run -> suspended (--engage) -> gantry on the ramp -> free.
#     Do NOT skip steps.
#  5. The abort is a TILT check against the reference, not a roll angle: the
#     crawl pitches ~90 deg, where Euler roll is degenerate. See --tilt-abort.
#  6. Keep a hand on the e-stop. Ctrl+C holds the current pose and exits.
#  7. When done, restart the controller:   aima em start-app mc
####################################################################################

Surface: a clean, high-friction incline wide enough for the ~2.9 m of travel,
with the robot placed prone at the BOTTOM, palms flat, heading up the slope.
Mark the start position: the policy is blind and open-loop on the motion clock,
so a start offset is a persistent error for the whole climb.
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
    """Same forward pass as the box / walking deployment (rsl_rl actor, ELU)."""

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
    """Euler roll -- for REPORTING only. Degenerate at the crawl's ~90 deg pitch,
    which is why the abort uses `tilt_between` instead."""
    x, y, z, w = q
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def projected_gravity_xyzw(q_xyzw) -> np.ndarray:
    """World gravity [0,0,-1] expressed in the body frame of q (holosoma's term).

    Matches holosoma_inference.utils.math.quat.quat_rotate_inverse. Note this is
    yaw-invariant, which is what lets `tilt_between` compare measured against
    reference without any heading alignment.
    """
    q = np.asarray(q_xyzw, np.float32).reshape(4)
    qw, qx, qy, qz = float(q[3]), float(q[0]), float(q[1]), float(q[2])
    v = np.array([0.0, 0.0, -1.0], np.float32)
    q_vec = np.array([qx, qy, qz], np.float32)
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(q_vec, v) * qw * 2.0
    c = q_vec * float(np.dot(q_vec, v)) * 2.0
    return (a - b + c).astype(np.float32)


def tilt_between(g_a: np.ndarray, g_b: np.ndarray) -> float:
    """Angle (rad) between two projected-gravity directions.

    The crawl pitches through ~90 deg, where Euler roll is degenerate and a
    fixed roll threshold is meaningless. This measures how far the body attitude
    has deviated from the REFERENCE attitude instead, which stays well defined
    at any pitch and is invariant to heading.
    """
    a = np.asarray(g_a, float)
    b = np.asarray(g_b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


# =============================== observation builder ===============================
class CrawlObservationBuilder:
    """Builds the 167-dim crawl WBT actor observation.

    motion_ref_ori_b matches holosoma's `subtract_frame_transforms` +
    first-two-rotation-matrix-columns encoding: q_rel = q_torso^-1 * q_ref,
    flattened row-major from R[:, :2] -> [m00, m01, m10, m11, m20, m21]. It uses
    the raw torso IMU quat because training tracks torso_link there.

    base_ang_vel and projected_gravity use the reconstructed PELVIS state (see
    the module docstring).

    The motion was authored in its own world frame; at engage time we compute a
    yaw offset aligning motion frame 0 with the robot's current heading, and
    apply it to every reference quat (the robot may face any direction).
    """

    def __init__(self, policy: NumpyPolicy, base_imu: str, use_pelvis_frame: bool = True):
        meta = policy.meta
        self.joint_names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"], np.float32)
        self.action_dim = int(meta["action_dim"])
        self.base_imu = base_imu
        self.policy = policy
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.yaw_offset = np.array([0, 0, 0, 1], np.float32)  # set by align()
        # Always reconstructed: the pelvis attitude is the right signal for the
        # tilt abort either way. The flag only decides what the POLICY is fed.
        self.pelvis_est = PelvisEstimator()
        self.use_pelvis_frame = bool(use_pelvis_frame)
        self.last_pelvis_quat = np.array([0, 0, 0, 1], np.float32)
        self.last_base_ang_vel = np.zeros(3, np.float32)
        self.last_grav = np.array([0.0, 0.0, -1.0], np.float32)

    def align(self, imu_quat_xyzw) -> None:
        """Rotate the reference trajectory into the robot's current heading."""
        q_robot_yaw = yaw_quat(np.asarray(imu_quat_xyzw, np.float32))
        q_ref0_yaw = yaw_quat(self.policy.ref_quat_xyzw[0])
        self.yaw_offset = quat_mul(q_robot_yaw, quat_inv(q_ref0_yaw))

    def ref_grav(self, frame: int) -> np.ndarray:
        """Reference torso projected gravity at `frame` (yaw-invariant)."""
        T = self.policy.ref_joint_pos.shape[0]
        return projected_gravity_xyzw(self.policy.ref_quat_xyzw[min(frame, T - 1)])

    def build(self, imus, jmap, frame: int) -> np.ndarray:
        T = self.policy.ref_joint_pos.shape[0]
        frame = min(frame, T - 1)
        q = np.array([jmap[n].position for n in self.joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in self.joint_names], np.float32)
        imu = imus[self.base_imu]
        q_imu = np.asarray(imu.quat, np.float32)

        # motion_ref_ori_b tracks torso_link in training -> raw IMU quat.
        q_ref = quat_mul(self.yaw_offset, self.policy.ref_quat_xyzw[frame])
        q_rel = quat_mul(quat_inv(q_imu), q_ref)
        ori6 = quat_to_mat(q_rel)[:, :2].reshape(-1)

        w_pelvis, self.last_pelvis_quat = self.pelvis_est.update(imu.quat, imu.ang_vel, jmap)
        if self.use_pelvis_frame:
            base_ang_vel = w_pelvis
            grav = projected_gravity_xyzw(self.last_pelvis_quat)
        else:
            base_ang_vel = np.asarray(imu.ang_vel, np.float32)
            grav = projected_gravity_xyzw(q_imu)
        self.last_base_ang_vel = base_ang_vel
        self.last_grav = grav

        # Alphabetical term order (matches holosoma's group concatenation):
        # actions, base_ang_vel, dof_pos, dof_vel, motion_command,
        # motion_ref_ori_b, projected_gravity
        return np.concatenate(
            [
                self.last_action,
                base_ang_vel,
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
                    default="../box_pickup/policy/x2_crawl_policy_v5_iter86000.npz",
                    help="Path to crawl policy .npz (default: v5 track-xyz, iter 86000). "
                         "v5 is the last crawl policy that climbs the full slope: v4 "
                         "drifted backward and off the ramp edge because termination "
                         "only checked height, and the v6 stand-up experiment regressed "
                         "the climb and was reverted. The stand-up at the end of the "
                         "reference is NOT reliably tracked yet -- expect the climb, "
                         "not a clean finish upright.")
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"],
                    help="IMU used as the attitude source (training ref body = torso_link).")
    ap.add_argument("--base-frame", default="pelvis", choices=["pelvis", "torso"],
                    help="Frame for the base_ang_vel AND projected_gravity "
                         "observations. 'pelvis' (default) reconstructs the pelvis "
                         "rate and attitude from the torso IMU plus the waist "
                         "joints, which is what training used (env.base_quat is the "
                         "articulation root). 'torso' feeds the raw IMU -- the v33 "
                         "box defect, kept only for A/B testing.")
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains and feed-forward torque "
                         "(lower = gentler).")
    ap.add_argument("--ramp-seconds", type=float, default=8.0,
                    help="Time to ramp from the current pose to the crawl start pose. "
                         "Longer than the box default: the robot is being lowered into "
                         "a prone pose, not nudged within a stand.")
    ap.add_argument("--settle-seconds", type=float, default=2.0,
                    help="Hold the start pose before checking init tolerances.")
    ap.add_argument("--no-torque-ff", action="store_true",
                    help="Send position-only commands, clipped to the mechanical joint "
                         "limits, with effort=0. This silently discards every "
                         "saturated-torque request the policy makes -- 39%% of all "
                         "(frame, joint) pairs on this motion -- so the robot gets a "
                         "fraction of the training torque. A/B testing only.")
    ap.add_argument("--action-clip", type=float, default=4.0,
                    help="Bound |action| on --action-clip-joints. |a|=4 is exactly the "
                         "effort limit, so this drops only requests for torque the "
                         "actuator cannot produce. In training those requests are "
                         "absorbed by contact (palms on the slope, feet on the ramp); "
                         "on hardware they drive the joint to its mechanical stop. "
                         "0 disables.")
    ap.add_argument("--action-clip-joints", default="ankle_roll,wrist",
                    help="Comma-separated substrings. Only the joints whose saturated "
                         "torque leans on contact belong here (wrists saturate on 88%% "
                         "of frames, ankle rolls 69%%). The head saturates too but is "
                         "excluded on purpose: nothing pushes back on it, so sim and "
                         "hardware already agree.")
    ap.add_argument("--max-joint-step", type=float, default=0.0,
                    help="Max change in a joint target per 20 ms tick (rad); 0 = off "
                         "(default). Training has no such rate limit, so any value "
                         "here is pure added lag. Torque is already bounded by the "
                         "effort limit, which is the actual safety quantity.")
    ap.add_argument("--joint-filter", type=float, default=0.0,
                    help="EMA smoothing on ALL joint targets (0 = off, default). Has "
                         "no counterpart in training, so it is pure added lag; on the "
                         "box task 0.0/0.2 both gave 100%% Isaac success while 0.9 gave "
                         "0%%, matching a hardware run that aborted at 0.9.")
    ap.add_argument("--tilt-abort", type=float, default=0.7,
                    help="Abort if the body attitude deviates from the REFERENCE "
                         "attitude by more than this angle (rad). This is the angle "
                         "between measured and reference projected gravity, not an "
                         "Euler roll: the crawl pitches through ~90 deg, where roll is "
                         "degenerate, and pitch is part of the motion so it cannot "
                         "simply be excluded. Gravity is yaw-invariant, so no heading "
                         "alignment is needed. Over the recorded v5 rollout the climb "
                         "never exceeds 0.22 rad, so 0.7 is a wide margin. 0 disables.")
    ap.add_argument("--tilt-abort-until-s", type=float, default=16.0,
                    help="Only apply --tilt-abort for this many seconds of motion "
                         "(default: the climb). After it the reference stands up, and "
                         "v5 does NOT track that stand-up -- it stays prone -- so the "
                         "measured-vs-reference angle grows to ~1.6 rad on a rollout "
                         "that is otherwise fine. Checking it there would abort every "
                         "run at ~16 s. Set to 0 to check the whole motion.")
    ap.add_argument("--hold-end-seconds", type=float, default=5.0,
                    help="Hold the final reference pose after the motion ends.")
    ap.add_argument("--init-tol-arm", type=float, default=0.20,
                    help="Max |meas-start| (rad) allowed on arm joints before the "
                         "policy may engage. Looser than the box default: a prone "
                         "start pose is placed by hand.")
    ap.add_argument("--init-tol-leg", type=float, default=0.35,
                    help="Max |meas-start| (rad) allowed on leg joints before the "
                         "policy may engage.")
    ap.add_argument("--init-timeout", type=float, default=20.0,
                    help="Seconds to wait after settle for the start pose to be "
                         "reached before aborting engage.")
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

    # Per-joint torque limit training clipped to. Newer exports carry it; for
    # older npz files recover it from the scale relation the training config
    # used, action_scale = 0.25 * effort_limit / kp.
    if "joint_effort_limit" in meta:
        eff_limit = np.array(meta["joint_effort_limit"], np.float32)
    else:
        eff_limit = 4.0 * action_scale * np.array(meta["joint_stiffness"], np.float32)
    eff_by_name = None if args.no_torque_ff else dict(zip(joint_names, eff_limit.tolist()))

    fps = int(meta["motion_fps"])
    n_frames = int(meta["motion_frames"])
    CONTROL_DT = 1.0 / float(meta.get("control_hz", 50))
    assert abs(CONTROL_DT * fps - 1.0) < 1e-6, "control rate must match motion fps"
    assert int(meta["obs_dim"]) == 167, f"expected crawl obs_dim=167, got {meta['obs_dim']}"
    assert "projected_gravity" in meta["observation_names"], "not a crawl policy export"

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  task:          {meta.get('task')}   run: {meta.get('run_path', '?')}")
    print(f"  obs terms:     {meta['observation_names']}  (dim {meta['obs_dim']})")
    print(f"  motion:        {n_frames} frames @ {fps} Hz = {n_frames / fps:.1f} s")
    print(f"  action joints: {len(joint_names)}   gain scale: {args.gain_scale}")
    print(f"  base frame:    {args.base_frame} (base_ang_vel + projected_gravity)"
          + ("" if args.base_frame == "pelvis" else "   <-- KNOWN-BAD, A/B ONLY"))
    print(f"  torque ff:     {'OFF -- position clipped to limits (OLD, known-bad)' if args.no_torque_ff else 'on (matches training clip_torques)'}")
    print(f"  action clip:   "
          f"{'OFF' if args.action_clip <= 0 else f'+-{args.action_clip:g} on {args.action_clip_joints}'}")
    print(f"  joint filter:  {args.joint_filter}   max joint step: "
          f"{'off' if args.max_joint_step <= 0 else f'{args.max_joint_step} rad/tick'}")
    print(f"  tilt abort:    "
          + ("OFF" if args.tilt_abort <= 0 else
             f"{args.tilt_abort:g} rad vs the reference attitude"
             + (", whole motion" if args.tilt_abort_until_s <= 0
                else f", first {args.tilt_abort_until_s:g}s only (the climb)")))
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    print("\nCRAWL SETUP: place the robot PRONE at the BOTTOM of the ramp, palms flat,")
    print("heading up the slope, matching motion frame 0. Do NOT start standing.")
    print("The motion climbs ~2.9 m and rises ~0.4 m over 19.2 s.")
    print("First trials: SUSPENDED / gantry, no ramp contact.\n")

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

    obs_builder = CrawlObservationBuilder(policy, base_imu=args.base_imu,
                                         use_pelvis_frame=(args.base_frame == "pelvis"))

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
    g0 = obs_builder.last_grav
    g_ref0 = obs_builder.ref_grav(0)
    print(f"[check] frame-0 obs built (dim {obs0.shape[0]}), "
          f"|action|_max = {np.abs(a0).max():.3f}")
    print(f"[check] torso roll = {roll_of(np.asarray(imus0[args.base_imu].quat)):+.3f} rad   "
          f"pelvis roll = {roll_of(obs_builder.last_pelvis_quat):+.3f} rad  (report only)")
    print(f"[check] projected_gravity ({args.base_frame}) = "
          f"({g0[0]:+.3f}, {g0[1]:+.3f}, {g0[2]:+.3f})")
    print(f"[check] reference gravity at frame 0    = "
          f"({g_ref0[0]:+.3f}, {g_ref0[1]:+.3f}, {g_ref0[2]:+.3f})   "
          f"tilt vs ref = {tilt_between(g0, g_ref0):.3f} rad "
          f"(abort at {args.tilt_abort})")
    if tilt_between(g0, g_ref0) > args.tilt_abort:
        print("[WARN] already past the tilt abort threshold -- the robot is not in the "
              "reference crawl attitude. Reposition before engaging.")
    # Standing/prone, the reconstruction must agree with the IMU when the waist
    # is ~0. A large delta there means the IMU is not aligned with torso_link.
    d = float(np.abs(np.asarray(obs_builder.pelvis_est.last_correction)).max())
    waist0 = {n: float(jmap0[n].position) for n in ("waist_yaw_joint", "waist_pitch_joint",
                                                    "waist_roll_joint")}
    print("[check] waist at start = " + "  ".join(f"{k.split('_')[1]}={v:+.3f}"
                                                 for k, v in waist0.items())
          + f"   |pelvis-torso gyro| = {d:.3f} rad/s")
    if max(abs(v) for v in waist0.values()) < 0.05 and d > 0.15:
        print("[WARN] waist is ~0 but the correction is large -- check the IMU mounting "
              "frame before engaging.")

    # Which joints this policy drives past their mechanical limit on purpose,
    # i.e. which ones need the feed-forward torque to behave like training.
    LIMIT_BY_NAME = {ji.name: (ji.lower_limit, ji.upper_limit)
                     for area in CONTROLLED_AREAS for ji in robot_model[area]}
    ref_tgt = policy.ref_joint_pos
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

    # Ramp / init target: the motion's own first frame (prone crawl pose).
    start_ref = {n: float(policy.ref_joint_pos[0][i]) for i, n in enumerate(joint_names)}
    end_ref = {n: float(policy.ref_joint_pos[-1][i]) for i, n in enumerate(joint_names)}

    if args.action_clip > 0.0:
        act_clip_mask = np.array(
            [any(k in n for k in args.action_clip_joints.split(",")) for n in joint_names])
        print(f"[check] action clip = +-{args.action_clip:g} on "
              f"{int(act_clip_mask.sum())} joint(s) matching "
              f"'{args.action_clip_joints}'  (|a|=4 is the effort limit)")
    else:
        act_clip_mask = None
        print("[check] action clip = OFF -- saturated wrist/ankle-roll torque will be "
              "delivered in full; hardware pins those joints to their stops")
    act_clip_ticks = 0

    LEG_JOINTS = [n for n in joint_names if ("hip" in n or "knee" in n or "ankle" in n)]
    ARM_JOINTS = [n for n in joint_names
                  if any(k in n for k in ("shoulder", "elbow", "wrist"))]

    def pose_errors(jmap):
        return {n: float(jmap[n].position - start_ref[n]) for n in joint_names}

    def worst_err(err, names):
        return max(((n, err[n]) for n in names), key=lambda kv: abs(kv[1]))

    def pose_ok(err):
        arm_bad = [n for n in ARM_JOINTS if abs(err[n]) > args.init_tol_arm]
        leg_bad = [n for n in LEG_JOINTS if abs(err[n]) > args.init_tol_leg]
        return arm_bad, leg_bad

    # ---- init pose check (before ramp) ---------------------------------------
    err0 = pose_errors(jmap0)
    print("\n[init] target start pose = motion frame 0 (prone crawl, palms flat):")
    print("[init] worst offenders vs start:")
    for n, e in sorted(err0.items(), key=lambda kv: -abs(kv[1]))[:10]:
        tol = args.init_tol_arm if n in ARM_JOINTS else args.init_tol_leg
        flag = "  <-- LARGE" if abs(e) > tol else ""
        print(f"    {n:32s} {e:+.3f} rad   target={start_ref[n]:+.3f}{flag}")
    arm_bad0, leg_bad0 = pose_ok(err0)
    if arm_bad0 or leg_bad0:
        print("[init] WARNING: not at the start pose yet. Ramp/settle will pull toward")
        print("[init] it; the policy will NOT engage until arms/legs are within")
        print("[init] tolerance (or use --force-engage). Place the robot prone by hand.")
    # --------------------------------------------------------------------------

    print("\n>>> SAFETY: suspended/gantry? MC stopped on .40? E-stop in hand? <<<")
    input(">>> Press Enter to START ramp-to-init (Ctrl+C to abort) <<<")

    run_name = f"crawl_{os.path.splitext(os.path.basename(args.policy))[0]}"
    PELVIS_COLS = ["pelvis_roll", "pelvis_ang_vel_x", "pelvis_ang_vel_y", "pelvis_ang_vel_z",
                   "obs_ang_vel_x", "obs_ang_vel_y", "obs_ang_vel_z",
                   "obs_grav_x", "obs_grav_y", "obs_grav_z",
                   "ref_grav_x", "ref_grav_y", "ref_grav_z", "tilt_vs_ref"]
    logger = RunLogger(
        joint_names, base_imu=args.base_imu, run_name=run_name,
        meta={"script": "deploy_x2_crawl.py", "policy": args.policy,
              "gain_scale": args.gain_scale, "joint_filter": args.joint_filter,
              "max_joint_step": args.max_joint_step,
              "torque_ff": not args.no_torque_ff,
              "action_clip": args.action_clip,
              "action_clip_joints": args.action_clip_joints,
              "joint_effort_limit": eff_limit.tolist(),
              "tilt_abort": args.tilt_abort,
              "tilt_abort_until_s": args.tilt_abort_until_s,
              "engage": args.engage, "task": meta.get("task"),
              "base_frame": args.base_frame,
              "run_path": meta.get("run_path")},
        log_dir=args.log_dir, enabled=not args.no_log,
        extra_columns=PELVIS_COLS, log_effort=True)

    def crawl_extra(q_pelvis, w_pelvis, obs_w, obs_g, ref_g):
        """Both the reconstruction and what the policy actually consumed, so an
        A/B run can be told apart from its log alone."""
        return dict(zip(PELVIS_COLS,
                        [roll_of(np.asarray(q_pelvis, np.float32)),
                         *np.asarray(w_pelvis, float).tolist(),
                         *np.asarray(obs_w, float).tolist(),
                         *np.asarray(obs_g, float).tolist(),
                         *np.asarray(ref_g, float).tolist(),
                         tilt_between(obs_g, ref_g)]))

    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)
    # Stronger gains while pulling into the start pose so limbs actually arrive.
    init_gain = max(float(args.gain_scale), 1.0)

    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    # ramp -> settle -> wait_init -> policy -> done
    phase = "ramp"
    phase_t0 = t0
    frame = 0
    # On a clean finish the reference is UPRIGHT, so easing to it is right. On an
    # abort the robot is prone and possibly tangled: hold what it has instead of
    # commanding a stand-up it never learned.
    completed = False

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()

            # ---- safety: tilt vs the reference attitude -----------------------
            # Recomputed fresh here rather than reusing the observation's value,
            # which is one tick old.
            imu_now = imus[args.base_imu]
            w_pelvis_now, q_pelvis_now = obs_builder.pelvis_est.update(
                imu_now.quat, imu_now.ang_vel, jmap)
            if obs_builder.use_pelvis_frame:
                obs_w_now = w_pelvis_now
                obs_g_now = projected_gravity_xyzw(q_pelvis_now)
            else:
                obs_w_now = np.asarray(imu_now.ang_vel, np.float32)
                obs_g_now = projected_gravity_xyzw(np.asarray(imu_now.quat, np.float32))
            ref_g_now = obs_builder.ref_grav(frame if phase == "policy" else 0)
            tilt = tilt_between(obs_g_now, ref_g_now)
            tilt_watched = (args.tilt_abort > 0.0
                            and (args.tilt_abort_until_s <= 0.0
                                 or frame * CONTROL_DT <= args.tilt_abort_until_s))
            if phase == "policy" and tilt_watched and tilt > args.tilt_abort:
                print(f"\n[ABORT] attitude is {tilt:+.2f} rad from the reference "
                      f"(limit {args.tilt_abort}). Holding pose.")
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
                    print("\n[phase] settle -- holding crawl start pose\n")

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
                          f"worst_leg {wl[0]}={wl[1]:+.3f}  tilt_vs_ref={tilt:.3f} rad")
                    if args.force_engage and (arm_bad or leg_bad):
                        print("[init] --force-engage: starting despite residual error")
                    obs_builder.align(imus[args.base_imu].quat)
                    phase = "policy"; phase_t0 = now; frame = 0
                    print("\n[phase] policy ENGAGED -- crawl motion clock running\n")
                elif elapsed >= args.init_timeout:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"\n[ABORT] start pose not reached in {args.init_timeout:.0f}s.")
                    print(f"[ABORT] worst_arm {wa[0]}={wa[1]:+.3f}  "
                          f"worst_leg {wl[0]}={wl[1]:+.3f}")
                    print("[ABORT] Reposition the robot prone, then rerun. "
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
                    tgt = float(raw_target[i])
                    if args.joint_filter > 0.0:
                        tgt = (1.0 - args.joint_filter) * tgt \
                            + args.joint_filter * prev_target[n]
                    if args.max_joint_step > 0.0:
                        tgt = prev_target[n] + float(np.clip(
                            tgt - prev_target[n],
                            -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = float(tgt)
                if frame >= n_frames + int(args.hold_end_seconds / CONTROL_DT):
                    phase = "done"; phase_t0 = now; completed = True
                    print("\n[phase] motion complete -> easing to the reference end pose\n")

            else:  # done
                if completed:
                    # Reference ends UPRIGHT at the top of the ramp.
                    alpha = min(1.0, elapsed / 2.0)
                    target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * end_ref[n]
                                      for n in joint_names}
                else:
                    # Aborted mid-crawl: freeze, do not attempt to stand.
                    target_by_name = dict(prev_target)

            ff = publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                              gain_now, args.engage, jmap=jmap, eff_by_name=eff_by_name)
            prev_target = target_by_name
            logger.log(now - t0, phase, frame, imus, jmap, target_by_name,
                       extra=crawl_extra(q_pelvis_now, w_pelvis_now, obs_w_now,
                                         obs_g_now, ref_g_now),
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
                    wn, wv = max(ff.items(), key=lambda kv: abs(kv[1]))
                    extra = f" ff_max={wn.replace('_joint', '')}{wv:+.1f}Nm"
                    if act_clip_mask is not None and phase == "policy":
                        extra += f" aclip={100.0 * act_clip_ticks / max(frame, 1):.0f}%"
                print(f"[{tag}] phase={phase:9s} t={now - t0:5.1f}s frame={frame:4d}/{n_frames} "
                      f"tilt={tilt:.2f}{' ' if tilt_watched else '~'}"
                      f"g=({obs_g_now[0]:+.2f},{obs_g_now[1]:+.2f},{obs_g_now[2]:+.2f}) "
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
        # Hold, never ramp to the standing default: from a prone crawl that would
        # be a stand-up command the policy never produced.
        print("\n[interrupt] holding the current pose and exiting (no stand-up ramp).")
        hold = dict(prev_target)
        t_stop = time.perf_counter()
        while time.perf_counter() - t_stop < 1.0 and rclpy.ok():
            try:
                imus_i, jmap_i = read_jmap()
            except Exception:
                imus_i = jmap_i = None
            ff = publish_pose(commander, hold, kp_by_name, kd_by_name, args.gain_scale,
                              args.engage, jmap=jmap_i, eff_by_name=eff_by_name)
            if jmap_i is not None:
                logger.log(time.perf_counter() - t0, "interrupt", frame, imus_i, jmap_i,
                           hold, effort_cmd=ff)
            time.sleep(CONTROL_DT)
    finally:
        logger.close()
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
