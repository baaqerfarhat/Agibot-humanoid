#!/usr/bin/env python3
"""Deploy the mjlab X2 box-hug policy (Mjlab-BoxHug-Ref-X2-Timed) on the real AgiBot humanoid.

Runs INSIDE the ROS 2 environment (needs `rclpy` + `aimdk_msgs`, same as
`robot_states_control.py`). The policy is a self-contained `.npz` from
`mjlab/scripts/x2_box_hug/export_boxhug_policy_npz.py`; the only runtime
dependency is numpy (`boxhug_runtime.py` next to this file does the command
generation and the palm forward kinematics).

    -------------------------------------------------------------------------
    PIPELINE (must match mjlab box-hug training):
      observation (153) -> policy MLP -> action (31), clipped to +-10
      target_q = clip(action * action_scale + default_q, joint range)
      publish position targets with the training PD gains at 50 Hz; the PD torque
      the training clipped to the effort limit is reproduced with a feed-forward
      `effort` term (see build_area_cmd; --no-torque-ff disables it)

    Observation (153, no base_lin_vel):
        [ base_ang_vel(3)      PELVIS gyro, pelvis frame   <- reconstructed, see below
          projected_gravity(3) gravity in the pelvis frame (upright ~ [0, 0, -1])
          joint_pos - default(31), joint_vel(31), prev_action(31),
          command(54) ]

    command = [ phase one-hot(9), phase progress, target pelvis height,
                box centre rel. pelvis (yaw frame, 3), sin/cos box yaw (2),
                palm target - palm position (yaw frame, L then R, 6),
                bilateral grasp flag, reference joints - default (31) ]

    THE PHASES ARE DRIVEN BY A CLOCK, exactly like training (time_driven_phases):
        STAND 0.00  LOWER 1.02  GRASP 2.52  LIFT 3.62  HOLD 5.62
        LOWER_BOX 6.62  SET_DOWN 8.62  RELEASE 9.02  DONE 9.72 ... clip ends 12.22 s
    The reference joints, the target pelvis height and the nominal box pose come
    from the clip frame of that clock. The policy does NOT perceive the box: the
    box pose in the command is the clip's (relative to the pelvis), so the box has
    to be placed where the clip has it (printed at start-up). The palm positions in
    the command are real: forward kinematics of the measured joints.
    -------------------------------------------------------------------------

    Training IMU was the PELVIS (site imu_in_pelvis). The robot only has torso/chest
    IMUs, so the default `--base-ang-vel pelvis` reconstructs the pelvis gyro and
    attitude from the torso IMU + the three waist joints (base_frame.PelvisEstimator).
    `--base-ang-vel torso` only to A/B the old substitution (known-bad on walking).

    Grasp flag: in sim it is the bilateral-contact test. Here it is set from the
    clock by default (1 from the clip's grasp_hold, 3.22 s, until RELEASE, 9.02 s);
    `--grasp-flag zero` keeps it 0 throughout. Both were replayed in mjlab with this
    exact code (see the export log); the deploy default is the better one.

#####################################  SAFETY  #####################################
#  1. First runs: robot SUSPENDED, NO BOX -> verify the motion shape in the air
#     (a squat to ~0.45 m pelvis height with the arms closing at ~3 s).
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default is DRY-RUN (computes + logs, does not publish). Add --engage only
#     after the dry-run output looks sane (|action| O(1), proj_g ~ [0,0,-1]).
#  4. Escalate: dry-run -> suspended --engage, no box -> gantry on the ground, no
#     box -> gantry with the box -> free. Do NOT skip steps.
#  5. Feet stay PLANTED in this task (training ended the episode if a foot moved
#     10 cm). Fix the stance before engaging; the robot will not step to recover.
#  6. Box: 47 x 46 x 41 cm, ~1 kg (training mass 1.0 kg, not randomised). Light
#     cardboard. Centre 0.55 m ahead of the ankles, centred left/right, on the floor.
#  7. Ctrl+C ramps back to the default standing pose. Then: aima em start-app mc
####################################################################################
"""

from __future__ import annotations

import argparse
import os
import threading
import time

import numpy as np
import rclpy

from aimdk_msgs.msg import JointCommand, JointCommandArray
from base_frame import PelvisEstimator
from robot_states_control import (
    JointArea,
    RobotStateClient,
    WholeBodyCommander,
    robot_model,
)
from run_logger import RunLogger

import boxhug_runtime as rt

CONTROLLED_AREAS = (JointArea.LEG, JointArea.WAIST, JointArea.ARM, JointArea.HEAD)


# =============================== command helpers ===============================
def build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name, gain_scale,
                   jmap=None, eff_by_name=None):
    """One JointCommandArray for `area`.

    Training: tau = clip(kp*(target - q) - kd*dq, +-effort_limit), target inside the
    joint range. The firmware makes kp*(pos - q) - kd*dq on its own; the difference
    to the training torque (only non-zero when the clip binds) goes out as the
    feed-forward `effort`, the same scheme deploy_x2_box_pickup.py uses.
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
    ff = {}
    for area in CONTROLLED_AREAS:
        cmd, area_ff = build_area_cmd(area, pos_by_name, kp_by_name, kd_by_name,
                                      gain_scale, jmap=jmap, eff_by_name=eff_by_name)
        ff.update(area_ff)
        if engage:
            commander.publish(area, cmd)
    return ff


def _default_policy_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "policies", "x2_boxhug_policy_v40b2_robust_best.npz")


# =============================== main ===============================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=_default_policy_path())
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"])
    ap.add_argument("--base-ang-vel", default="pelvis", choices=["pelvis", "torso"],
                    help="pelvis (default) reconstructs training's pelvis IMU from the "
                         "torso IMU + waist joints. torso = known-bad A/B only.")
    ap.add_argument("--box-source", default="clip", choices=["clip", "hands"],
                    help="box pose fed to the policy: the clip's nominal pose (default) "
                         "or, while holding, the clip's box attached to the measured hands.")
    ap.add_argument("--grasp-flag", default="timeline", choices=["timeline", "zero"])
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains and feed-forward torque.")
    ap.add_argument("--ramp-seconds", type=float, default=5.0)
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    ap.add_argument("--no-torque-ff", action="store_true",
                    help="Do not reproduce training's torque clip as feed-forward effort.")
    ap.add_argument("--hold-end-seconds", type=float, default=2.0,
                    help="Seconds to keep running the policy after the clip ends (DONE holds).")
    ap.add_argument("--roll-abort", type=float, default=0.6,
                    help="Abort if |pelvis roll| exceeds this (rad).")
    ap.add_argument("--tilt-abort", type=float, default=-0.5,
                    help="Abort if pelvis projected_gravity z rises above this "
                         "(-0.5 = 60 deg lean; the squat itself stays well below).")
    ap.add_argument("--init-tol-arm", type=float, default=0.12)
    ap.add_argument("--init-tol-leg", type=float, default=0.25)
    ap.add_argument("--init-timeout", type=float, default=20.0)
    ap.add_argument("--force-engage", action="store_true")
    ap.add_argument("--log-dir", default="run_logs")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    policy = rt.NumpyPolicy(args.policy)
    meta = policy.meta
    if meta.get("task") != "x2_box_hug_timed":
        print(f"[WARN] meta.task={meta.get('task')!r} (expected x2_box_hug_timed)")
    joint_names = list(meta["joint_names"])
    default = np.array(meta["default_joint_pos"], np.float32)
    action_scale = np.array(meta["action_scale"], np.float32)
    target_clip = np.array(meta["joint_target_clip"], np.float32)  # (31, 2), training clip
    eff_limit = np.array(meta["joint_effort_limit"], np.float32)
    kp_by_name = dict(zip(joint_names, meta["joint_stiffness"]))
    kd_by_name = dict(zip(joint_names, meta["joint_damping"]))
    default_by_name = dict(zip(joint_names, default.tolist()))
    eff_by_name = None if args.no_torque_ff else dict(zip(joint_names, eff_limit.tolist()))
    CONTROL_DT = 1.0 / float(meta.get("control_hz", 50))
    use_pelvis = args.base_ang_vel == "pelvis"

    cmd_builder = rt.BoxHugCommandBuilder(policy, box_source=args.box_source,
                                          grasp_flag=args.grasp_flag)
    obs_builder = rt.BoxHugObservation(policy, cmd_builder)
    timeline = cmd_builder.timeline
    policy_seconds = timeline.clip_end_s + max(0.0, args.hold_end_seconds)
    feet = list(meta["feet_body_names"])
    ankle_h = float(meta["ankle_height_standing_m"])

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  run_path:      {meta.get('run_path', '?')}   iteration {meta.get('iteration', '?')}")
    print(f"  obs terms:     {meta['observation_names']}  (dim {meta['obs_dim']})")
    print(f"  timeline (s):  " + "  ".join(f"{n}@{s:.2f}" for n, s in
                                          zip(meta["phase_names"], meta["seg_start_s"])))
    print(f"  clip:          {meta['motion_frames']} frames @ {meta['motion_fps']:.0f} Hz "
          f"= {timeline.clip_end_s:.2f} s, then hold {args.hold_end_seconds:.1f} s")
    print(f"  box source:    {args.box_source}   grasp flag: {args.grasp_flag} "
          f"[{timeline.grasp_on_s:.2f}, {timeline.grasp_off_s:.2f}) s")
    print(f"  base IMU:      {args.base_imu}   ang_vel/attitude = {args.base_ang_vel}")
    print(f"  gain scale:    {args.gain_scale}   torque ff: {'OFF' if args.no_torque_ff else 'on'}")
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("=" * 78)
    bs = meta["box_size"]
    print(f"\nBOX PLACEMENT: {bs[0] * 100:.0f} x {bs[1] * 100:.0f} x {bs[2] * 100:.0f} cm, "
          f"~{meta['box_mass_kg']:.1f} kg, on the floor, long side (47 cm) ACROSS the robot,")
    print(f"centre {meta['box_ahead_of_ankles_m']:.2f} m ahead of the ankle joints "
          f"({meta['box_ahead_of_pelvis_m']:.2f} m ahead of the pelvis), "
          f"{abs(meta['box_left_of_ankles_m']) * 100:.0f} cm off-centre.")
    print("Near face therefore ~0.32 m from the ankles. Robot starts in the DEFAULT standing")
    print("pose (knees 0.4, elbows -0.3); the clip's first frame is that pose.")
    print(f"Pelvis goes {meta['clip_pelvis_z_start']:.2f} -> {meta['clip_pelvis_z_min']:.2f} m; "
          f"box rises {meta['clip_box_z_rise_m']:.2f} m. FEET STAY PLANTED.\n")

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

    pelvis_est = PelvisEstimator()

    def read_jmap():
        imus, head, waist, arm, leg = client.get_robot_states()
        jmap = {jr.name: jr for jr in (head + waist + arm + leg)}
        missing = [n for n in joint_names if n not in jmap]
        if missing:
            raise RuntimeError(f"State missing joints: {missing}")
        return imus, jmap

    def base_state(imus, jmap):
        """(pelvis ang vel, pelvis quat xyzw) the policy is fed."""
        imu = imus[args.base_imu]
        w_p, q_p = pelvis_est.update(imu.quat, imu.ang_vel, jmap)
        if use_pelvis:
            return np.asarray(w_p, np.float32), np.asarray(q_p, np.float32)
        return np.asarray(imu.ang_vel, np.float32), np.asarray(imu.quat, np.float32)

    def q_dq(jmap):
        q = np.array([jmap[n].position for n in joint_names], np.float32)
        dq = np.array([jmap[n].velocity for n in joint_names], np.float32)
        return q, dq

    # ---- start-up checks -------------------------------------------------------
    imus0, jmap0 = read_jmap()
    w0, qp0 = base_state(imus0, jmap0)
    g0 = rt.projected_gravity(qp0)
    q0, dq0 = q_dq(jmap0)
    obs0 = obs_builder.build(0.0, w0, qp0, q0, dq0)
    a0 = policy(obs0)
    L = cmd_builder.last
    print(f"[check] pelvis proj_g = {np.round(g0, 3)}  roll={rt.roll_of_xyzw(qp0):+.3f} "
          f"(upright ~[0, 0, -1])")
    if g0[2] > -0.8:
        print("[check] WARNING: robot does not look upright, or IMU axes differ. "
              "Do not --engage until this is ~[0,0,-1].")
    print(f"[check] frame-0 obs built (dim {obs0.shape[0]}), |action|_max = {np.abs(a0).max():.3f} "
          f"(should be O(1))")
    print(f"[check] palms (FK, yaw frame) L={np.round(L['palm_b'][0], 3)} R={np.round(L['palm_b'][1], 3)}"
          f"   palm->target |err| L={L['palm_err'][0]:.3f} R={L['palm_err'][1]:.3f} m")
    print(f"[check] pelvis height from feet FK = "
          f"{rt.pelvis_height_from_feet(cmd_builder.fk, L['R_rp'], q0, feet, ankle_h):.3f} m "
          f"(standing ref {meta['clip_pelvis_z_start']:.3f})")
    d = float(np.abs(np.asarray(pelvis_est.last_correction)).max())
    waist0 = {n: float(jmap0[n].position) for n in ("waist_yaw_joint", "waist_pitch_joint",
                                                   "waist_roll_joint")}
    print("[check] waist at start = " + "  ".join(f"{k.split('_')[1]}={v:+.3f}" for k, v in waist0.items())
          + f"   |pelvis-torso gyro| = {d:.3f} rad/s")
    if max(abs(v) for v in waist0.values()) < 0.05 and d > 0.15:
        print("[WARN] waist is ~0 but the correction is large -- check the IMU mounting "
              "frame before engaging.")

    LEG_JOINTS = [n for n in joint_names if ("hip" in n or "knee" in n or "ankle" in n)]
    ARM_JOINTS = [n for n in joint_names if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    start_ref = dict(default_by_name)  # clip frame 0 == default standing keyframe

    def pose_errors(jmap):
        return {n: float(jmap[n].position - start_ref[n]) for n in joint_names}

    def worst_err(err, names):
        return max(((n, err[n]) for n in names), key=lambda kv: abs(kv[1]))

    def pose_ok(err):
        arm_bad = [n for n in ARM_JOINTS if abs(err[n]) > args.init_tol_arm]
        leg_bad = [n for n in LEG_JOINTS if abs(err[n]) > args.init_tol_leg]
        return arm_bad, leg_bad

    err0 = pose_errors(jmap0)
    print("\n[init] start pose = default standing keyframe (clip frame 0):")
    for n in ARM_JOINTS + LEG_JOINTS:
        tol = args.init_tol_arm if n in ARM_JOINTS else args.init_tol_leg
        flag = "  <-- LARGE" if abs(err0[n]) > tol else ""
        print(f"    {n:32s} {err0[n]:+.3f} rad   target={start_ref[n]:+.3f}{flag}")
    arm_bad0, leg_bad0 = pose_ok(err0)
    if arm_bad0 or leg_bad0:
        print("[init] WARNING: not at the start pose. Ramp/settle pulls toward it; the policy")
        print("[init] will NOT engage until within tolerance (or --force-engage).")

    print("\n>>> SAFETY: robot suspended? MC stopped on .40? E-stop in hand? Feet planted? <<<")
    input(">>> Press Enter to START ramp-to-default (Ctrl+C to abort) <<<")

    run_name = f"box_hug_{os.path.splitext(os.path.basename(args.policy))[0]}"
    EXTRA = ["phase_idx", "phase_progress", "cmd_height", "pelvis_h_fk", "box_bx", "box_by", "box_bz",
             "palm_err_L", "palm_err_R", "grasp_flag", "pelvis_roll", "proj_g_x", "proj_g_y",
             "proj_g_z", "obs_ang_vel_x", "obs_ang_vel_y", "obs_ang_vel_z", "act_abs_max"]
    logger = RunLogger(
        joint_names, base_imu=args.base_imu, run_name=run_name,
        meta={"script": "deploy_x2_box_hug.py", "policy": args.policy,
              "gain_scale": args.gain_scale, "engage": args.engage,
              "base_ang_vel_source": args.base_ang_vel, "box_source": args.box_source,
              "grasp_flag": args.grasp_flag, "torque_ff": not args.no_torque_ff,
              "hold_end_seconds": args.hold_end_seconds, "run_path": meta.get("run_path"),
              "task": meta.get("task"), "iteration": meta.get("iteration")},
        log_dir=args.log_dir, enabled=not args.no_log, extra_columns=EXTRA, log_effort=True,
    )

    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)
    init_gain = max(float(args.gain_scale), 1.0)
    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    phase = "ramp"  # ramp -> settle -> wait_init -> policy -> done
    phase_t0 = t0
    frame = 0
    extra = {}
    action = np.zeros(len(joint_names), np.float32)

    try:
        while rclpy.ok():
            now = time.perf_counter()
            imus, jmap = read_jmap()
            w_p, q_p = base_state(imus, jmap)
            g = rt.projected_gravity(q_p)
            roll = rt.roll_of_xyzw(q_p)
            q, dq = q_dq(jmap)

            if phase == "policy" and (abs(roll) > args.roll_abort or g[2] > args.tilt_abort):
                print(f"\n[ABORT] pelvis roll {roll:+.2f} / proj_g_z {g[2]:+.2f} beyond the abort "
                      "limits. Ramping to the default pose.")
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
                    print("\n[phase] settle -- holding the default standing pose\n")

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
                if (not arm_bad and not leg_bad) or args.force_engage:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"[init] READY  worst_arm {wa[0]}={wa[1]:+.3f}  worst_leg {wl[0]}={wl[1]:+.3f}")
                    obs_builder.last_action[:] = 0.0
                    phase = "policy"; phase_t0 = now; frame = 0
                    print("\n[phase] policy ENGAGED -- task clock running (STAND 1 s, then LOWER)\n")
                elif elapsed >= args.init_timeout:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"\n[ABORT] start pose not reached in {args.init_timeout:.0f}s "
                          f"(worst_arm {wa[0]}={wa[1]:+.3f}  worst_leg {wl[0]}={wl[1]:+.3f}). "
                          "Reposition, or --force-engage.")
                    phase = "done"; phase_t0 = now

            elif phase == "policy":
                t_task = elapsed
                obs = obs_builder.build(t_task, w_p, q_p, q, dq)
                action = policy(obs).astype(np.float32)  # clipped to +-10 like training
                obs_builder.last_action = action
                L = cmd_builder.last
                frame = int(L["frame"])
                raw_target = np.clip(action * action_scale + default, target_clip[:, 0], target_clip[:, 1])
                target_by_name = {n: float(raw_target[i]) for i, n in enumerate(joint_names)}
                extra = {
                    "phase_idx": L["phase"], "phase_progress": L["progress"],
                    "cmd_height": L["target_height"],
                    "pelvis_h_fk": rt.pelvis_height_from_feet(cmd_builder.fk, L["R_rp"], q, feet, ankle_h),
                    "box_bx": L["box_b"][0], "box_by": L["box_b"][1], "box_bz": L["box_b"][2],
                    "palm_err_L": L["palm_err"][0], "palm_err_R": L["palm_err"][1],
                    "grasp_flag": L["grasp_flag"], "act_abs_max": float(np.abs(action).max()),
                }
                if t_task >= policy_seconds:
                    phase = "done"; phase_t0 = now
                    print("\n[phase] motion complete -> ramping to the default pose\n")

            else:  # done
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * default_by_name[n]
                                  for n in joint_names}

            ff = publish_pose(commander, target_by_name, kp_by_name, kd_by_name, gain_now,
                              args.engage, jmap=jmap, eff_by_name=eff_by_name)
            prev_target = target_by_name
            extra.update({"pelvis_roll": roll, "proj_g_x": float(g[0]), "proj_g_y": float(g[1]),
                          "proj_g_z": float(g[2]), "obs_ang_vel_x": float(w_p[0]),
                          "obs_ang_vel_y": float(w_p[1]), "obs_ang_vel_z": float(w_p[2])})
            logger.log(now - t0, phase, frame, imus, jmap, target_by_name, extra=extra, effort_cmd=ff)

            if now - last_print >= 1.0:
                last_print = now
                tag = "DRY" if not args.engage else "CMD"
                more = ""
                if phase == "policy":
                    L = cmd_builder.last
                    more = (f" {rt.PHASE_NAMES[L['phase']]:9s} p={L['progress']:.2f} f={frame:3d}"
                            f" h_cmd={L['target_height']:.2f} h_fk={extra.get('pelvis_h_fk', 0):.2f}"
                            f" palm_err={L['palm_err'][0]:.2f}/{L['palm_err'][1]:.2f}"
                            f" |a|={np.abs(action).max():.1f}")
                    if ff:
                        wn, wv = max(ff.items(), key=lambda kv: abs(kv[1]))
                        more += f" ff_max={wn.replace('_joint', '')}{wv:+.1f}Nm"
                elif phase == "wait_init":
                    err = pose_errors(jmap)
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    more = f" arm_err={wa[0][:14]}{wa[1]:+.2f} leg_err={wl[0][:14]}{wl[1]:+.2f}"
                print(f"[{tag}] {phase:9s} t={now - t0:5.1f}s roll={roll:+.2f} g_z={g[2]:+.2f}"
                      f" knee_L={target_by_name['left_knee_joint']:+.2f}"
                      f" elbow_L={target_by_name['left_elbow_joint']:+.2f}{more}")

            next_t += CONTROL_DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[interrupt] ramping to the default pose and exiting.")
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
