#!/usr/bin/env python3
"""Box-pickup policy + ONLINE layer adaptation on the real AgiBot X2.

Same robot interface, observation builder, PD gains, filters and safety ladder as
`deploy_x2_box_pickup.py` -- the ONLY difference is where the action comes from. Each
50 Hz tick, one weight matrix of the policy (layer 2 of 4) is nudged by a Lyapunov
adaptation law driven by the live joint tracking error. Method and provenance:
`../adaptation/README.md`.

    action  = MLP(obs; W_adapted)              instead of  MLP(obs; W_trained)
    W2     <- W2 + dt*(Gamma*delta z^T - gamma*(W2 - W2_trained))

Nothing is trained, no gradients of a learned objective, no GPU: numpy only, ~7 ms of
the 20 ms tick. Adaptation does NOT persist -- every run starts from the trained
weights.

    ---------------------------------------------------------------------------
    HOW TO GET A RESULT (the point of this script)
    ---------------------------------------------------------------------------
    Run the SAME motion twice, changing one flag, and compare the two logs:

        A (frozen control)   ... --engage --gain 0      --tag frozen
        B (adapted)          ... --engage --gain 3e-4   --tag adapted

    Both arms then run through the identical numpy code path, so the only difference
    between them is the adaptation itself (gain 0 makes the update a no-op). Then:

        python compare_adapt_runs.py run_logs/<A>.csv run_logs/<B>.csv

    Do NOT conclude anything from one pair. This task is chaotic -- in sim a 1e-6
    action perturbation moved leg tracking error by 1.7 deg over 2.4 s -- so run at
    least 3 pairs, alternating A/B/A/B, and compare medians.

#####################################  SAFETY  #####################################
#  READ THIS. Adaptation can make the robot WORSE. It has no performance monitor and
#  no reversion: nothing checks whether the robot is doing better. The only automatic
#  net is a weight-drift bound that reverts to the trained weights and latches off.
#
#  In Isaac, on a HEALTHY policy, the paper's default mask (legs+waist) at its default
#  gain dropped survival from 14.7 s to 2.5 s and failed the pickup on 6/6 seeds. Only
#  --mask waist was safe, and it only WON under an actuator fault. So:
#     - --mask waist is the default here. Changing it to legs_waist on hardware,
#       untethered, is how you break the robot.
#     - This is FAULT-RECOVERY control. On a healthy robot expect no gain at best.
#
#  1. First runs: robot SUSPENDED, NO BOX. Verify in the air, --gain 0 first.
#  2. Stop the motion controller first:   aima em stop-app mc      (on 10.0.1.40)
#  3. Default mode is DRY-RUN: computes & logs, DOES NOT publish. Add --engage.
#  4. Escalation: dry-run -> suspended gain 0 -> suspended adapted -> gantry on
#     ground, no box -> gantry with box -> free. Do NOT skip steps.
#  5. Keep a hand on the e-stop. Ctrl+C ramps back to a safe pose and exits.
#  6. Abort conditions freeze adaptation permanently for that run, they do not retry.
#  7. When done, restart the controller:   aima em start-app mc
####################################################################################

Box placement, start pose and the motion timeline are unchanged from
`deploy_x2_box_pickup.py` -- read its docstring first; it is the prerequisite. Get the
frozen policy working there before adding adaptation here.
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time

import numpy as np
import rclpy

# Reuse the validated deploy path verbatim: identical observation construction,
# identical command building, identical gains. Only the action source changes.
from deploy_x2_box_pickup import (
    NumpyPolicy,
    WbtObservationBuilder,
    publish_pose,
    roll_of,
)
from layer_adapt import MASK_PRESETS, OnlineLayerAdapter
from robot_states_control import RobotStateClient, WholeBodyCommander
from run_logger import RunLogger


class AdaptLog:
    """Per-tick adaptation trace, flushed every row so a Ctrl-C keeps everything."""

    COLS = ("t_s", "phase", "frame", "adapt_on", "drift", "drift_frac",
            "err_masked_deg", "err_leg_deg", "err_all_deg",
            "act_dev_max", "dev_clamped", "loop_ms")

    def __init__(self, path: str | None):
        self.path = path
        self._f = None
        self._n = 0
        if path:
            self._f = open(path, "w", newline="")
            self._w = csv.writer(self._f)
            self._w.writerow(self.COLS)
            self._f.flush()
            print(f"[log] adaptation trace -> {path}")

    def log(self, **kw) -> None:
        if self._f is None:
            return
        try:
            self._w.writerow([kw.get(c, "") for c in self.COLS])
            self._n += 1
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            print(f"[log] saved {self._n} adaptation rows -> {self.path}")
            self._f = None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- same knobs as the frozen deploy, same defaults -----------------------
    ap.add_argument("--policy", default="../box_pickup/policy/x2_box_policy_v33_iter253000.npz")
    ap.add_argument("--engage", action="store_true",
                    help="ACTUALLY publish commands. Without this it is a dry run.")
    ap.add_argument("--base-imu", default="torso", choices=["torso", "chest"])
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="Scale on the training PD gains (NOT the adaptation gain).")
    ap.add_argument("--ramp-seconds", type=float, default=5.0)
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    ap.add_argument("--max-joint-step", type=float, default=0.15)
    ap.add_argument("--roll-abort", type=float, default=0.7,
                    help="Abort if |torso roll| exceeds this (rad). Pitch is NOT "
                         "checked: the motion contains a deep forward bend.")
    ap.add_argument("--hold-end-seconds", type=float, default=3.0)
    ap.add_argument("--leg-filter", type=float, default=0.9,
                    help="EMA on LEG targets. Note this attenuates leg commands, so "
                         "adapting leg joints partly fights this filter -- another "
                         "reason --mask waist is the default.")
    ap.add_argument("--init-tol-arm", type=float, default=0.12)
    ap.add_argument("--init-tol-leg", type=float, default=0.25)
    ap.add_argument("--init-timeout", type=float, default=20.0)
    ap.add_argument("--force-engage", action="store_true")
    ap.add_argument("--log-dir", default="run_logs")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--hold-forever", action="store_true",
                    help="Keep holding the default pose after the motion, as "
                         "deploy_x2_box_pickup.py does, instead of exiting. Default is "
                         "to exit once the ramp to default has finished, which leaves "
                         "the robot in exactly the state Ctrl+C would; the A/B protocol "
                         "needs 6+ runs and this makes each one self-terminating.")

    # ---- adaptation ----------------------------------------------------------
    g = ap.add_argument_group("adaptation")
    g.add_argument("--gain", type=float, default=3e-4,
                   help="Adaptation gain Gamma. 0 = frozen control arm (update is a "
                        "no-op, so the action is exactly the trained policy). 3e-4 is "
                        "the paper's default and what won under fault in sim with "
                        "--mask waist. FIRST HARDWARE RUNS: use 1e-5.")
    g.add_argument("--leak", type=float, default=1e-2,
                   help="Leak gamma, pulling W back toward the TRAINED weights.")
    g.add_argument("--mask", default="waist", choices=sorted(MASK_PRESETS),
                   help="Which joints' tracking error drives adaptation. 'waist' is "
                        "the only preset that ever helped; 'legs_waist' (the paper's "
                        "default) was catastrophic in sim, healthy AND faulted.")
    g.add_argument("--layer", type=int, default=2,
                   help="Which hidden layer to adapt (0..2). 2 = 128x256, as published.")
    g.add_argument("--max-drift", type=float, default=1.0,
                   help="Revert to trained weights and latch adaptation OFF once "
                        "||W-W0||_F exceeds this. Sim used 5.0; 1.0 is deliberately "
                        "tighter for hardware. A healthy waist-only run stays ~0.05.")
    g.add_argument("--adapt-after", type=float, default=0.0,
                   help="Seconds into the motion before adaptation engages. 0 = from "
                        "the first tick, as published.")
    g.add_argument("--max-action-dev", type=float, default=0.5,
                   help="Hard clamp: the adapted action may not differ from the frozen "
                        "action by more than this, per joint (action units, pre "
                        "action_scale). Costs one extra forward pass (~2 ms). "
                        "0 disables the clamp AND skips that pass.")
    g.add_argument("--max-overrun", type=int, default=25,
                   help="Disable adaptation after this many ticks miss the 20 ms "
                        "deadline. Guards against the update starving the loop.")
    g.add_argument("--tag", default="",
                   help="Suffix for the log name, e.g. --tag frozen / --tag adapted.")
    g.add_argument("--self-check", action="store_true",
                   help="Verify the adapter's frozen forward matches the deployed "
                        "policy and time the per-tick cost, then exit. No robot "
                        "motion, no publishing. Run this first.")
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

    adapter = OnlineLayerAdapter(
        policy.W, policy.b, policy.mean, policy.std, joint_names,
        action_scale, [kp_by_name[n] for n in joint_names],
        layer=args.layer, gain=args.gain, leak=args.leak,
        mask=MASK_PRESETS[args.mask], max_drift=args.max_drift,
        engage_step=int(args.adapt_after / CONTROL_DT),
    )
    ref_pos = policy.ref_joint_pos.astype(np.float64)
    legs = [i for i, n in enumerate(joint_names)
            if any(k in n for k in ("hip", "knee", "ankle"))]
    masked = [i for i, m in enumerate(adapter.err_mask) if m > 0]
    adapting = args.gain != 0.0

    print("=" * 78)
    print(f"  policy:        {args.policy}")
    print(f"  task:          {meta.get('task')}   run: {meta.get('run_path', '?')}")
    print(f"  motion:        {n_frames} frames @ {fps} Hz = {n_frames / fps:.1f} s")
    print(f"  MODE:          {'ENGAGED (publishing!)' if args.engage else 'DRY RUN (no publish)'}")
    print("-" * 78)
    if adapting:
        print(f"  ADAPTATION:    ON   layer {args.layer} "
              f"({adapter.W0[args.layer].shape[0]}x{adapter.W0[args.layer].shape[1]} = "
              f"{adapter.W0[args.layer].size} weights)")
        print(f"  gain/leak:     {args.gain:g} / {args.leak:g}"
              f"   engage at {args.adapt_after:.1f}s")
        print(f"  error mask:    {args.mask} -> {len(adapter.masked_joints)} joints: "
              f"{', '.join(adapter.masked_joints)}")
        print(f"  guards:        max_drift {args.max_drift:g}, "
              f"max_action_dev {args.max_action_dev:g}, "
              f"max_overrun {args.max_overrun}")
    else:
        print("  ADAPTATION:    OFF (gain 0) -- this is the FROZEN CONTROL ARM")
    print("=" * 78)
    if args.mask == "legs_waist" and adapting:
        print("\n!!! --mask legs_waist is the configuration that fell in 2.5 s on 6/6")
        print("!!! seeds in sim, healthy AND faulted. Suspended runs ONLY.\n")

    # ---- self-check: no ROS traffic, no motion, just numbers ------------------
    if args.self_check:
        rng = np.random.default_rng(0)
        obs = ((rng.normal(size=policy.mean.shape[0]) * 0.3)
               + policy.mean).astype(np.float32)
        a_ref = policy(obs).reshape(-1)
        a_adp = adapter.act_frozen(obs)
        print(f"[check] frozen forward vs deployed policy: "
              f"max |diff| = {np.abs(a_ref - a_adp).max():.2e}  (float32 rounding ~1e-6)")

        n = 200
        t = time.perf_counter()
        for _ in range(n):
            adapter.act(obs)
        t_fwd = (time.perf_counter() - t) / n * 1e3
        err = rng.normal(size=len(joint_names)) * 0.05
        adapter.act(obs)
        adapter.max_drift = np.inf  # timing only: a drift latch would make it a no-op
        t = time.perf_counter()
        for _ in range(n):
            adapter.update(err, CONTROL_DT)
        t_upd = (time.perf_counter() - t) / n * 1e3
        adapter.max_drift = args.max_drift
        n_fwd = 2 if args.max_action_dev > 0 else 1
        total = n_fwd * t_fwd + t_upd
        print(f"[check] per tick: {n_fwd} x forward {t_fwd:.2f} ms + update {t_upd:.2f} ms "
              f"= {total:.2f} ms of {CONTROL_DT * 1e3:.0f} ms "
              f"({total / (CONTROL_DT * 1e3) * 100:.0f}% of budget)")
        if total > 0.5 * CONTROL_DT * 1e3:
            print("[check] WARNING: over half the tick budget. Use --max-action-dev 0, "
                  "or do not adapt on this CPU.")
        adapter.reset()
        print(f"[check] drift after reset: {adapter.weight_drift:.3e} (must be 0)")
        print("[check] done -- no commands were sent.")
        return

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

    obs_builder = WbtObservationBuilder(policy, base_imu=args.base_imu)

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
    a_ref = policy(obs0).reshape(-1)
    a_adp = adapter.act_frozen(obs0)
    print(f"[check] frame-0 obs built (dim {obs0.shape[0]}), "
          f"|action|_max = {np.abs(a_ref).max():.3f} (should be O(1))")
    print(f"[check] adapter frozen forward matches policy to "
          f"{np.abs(a_ref - a_adp).max():.2e}")
    print(f"[check] torso roll = {roll_of(np.asarray(imus0[args.base_imu].quat)):+.3f} rad")

    start_ref = {n: float(policy.ref_joint_pos[0][i]) for i, n in enumerate(joint_names)}
    LEG_JOINTS = [n for n in joint_names if ("hip" in n or "knee" in n or "ankle" in n)]
    LEG_SET = set(LEG_JOINTS)
    ARM_JOINTS = [n for n in joint_names
                  if any(k in n for k in ("shoulder", "elbow", "wrist"))]

    def pose_errors(jmap):
        return {n: float(jmap[n].position - start_ref[n]) for n in joint_names}

    def worst_err(err, names):
        return max(((n, err[n]) for n in names), key=lambda kv: abs(kv[1]))

    def pose_ok(err):
        return ([n for n in ARM_JOINTS if abs(err[n]) > args.init_tol_arm],
                [n for n in LEG_JOINTS if abs(err[n]) > args.init_tol_leg])

    err0 = pose_errors(jmap0)
    arm_bad0, leg_bad0 = pose_ok(err0)
    print("\n[init] target start pose = motion frame 0 (policy-correct)")
    for label, names, tol in (("ARM", ARM_JOINTS, args.init_tol_arm),
                              ("LEG", LEG_JOINTS, args.init_tol_leg)):
        w = worst_err(err0, names)
        print(f"[init] worst {label}: {w[0]} = {w[1]:+.3f} rad (tol {tol:.2f})")
    if arm_bad0 or leg_bad0:
        print("[init] not at start pose yet; ramp/settle will pull toward it. Policy "
              "will NOT engage until within tolerance (or --force-engage).")

    print("\n>>> SAFETY: robot suspended? MC stopped on .40? E-stop in hand? <<<")
    if adapting:
        print(">>> ADAPTATION IS ON. It can make the robot worse. <<<")
    input(">>> Press Enter to START ramp-to-init (Ctrl+C to abort) <<<")

    pol_tag = os.path.splitext(os.path.basename(args.policy))[0]
    suffix = args.tag or (f"g{args.gain:g}_{args.mask}" if adapting else "frozen")
    run_name = f"box_adapt_{pol_tag}_{suffix}"
    run_meta = {"script": "deploy_x2_box_adapt.py", "policy": args.policy,
                "gain_scale": args.gain_scale, "leg_filter": args.leg_filter,
                "engage": args.engage, "task": meta.get("task"),
                "run_path": meta.get("run_path"),
                "adapt": {"gain": args.gain, "leak": args.leak, "mask": args.mask,
                          "mask_joints": adapter.masked_joints, "layer": args.layer,
                          "max_drift": args.max_drift,
                          "max_action_dev": args.max_action_dev,
                          "adapt_after_s": args.adapt_after,
                          "control_dt": CONTROL_DT}}
    logger = RunLogger(joint_names, base_imu=args.base_imu, run_name=run_name,
                       meta=run_meta, log_dir=args.log_dir, enabled=not args.no_log)
    alog = AdaptLog(logger.path[:-4] + "_adapt.csv" if logger.path else None)

    start_pose = {n: jmap0[n].position for n in joint_names}
    prev_target = dict(start_pose)
    init_gain = max(float(args.gain_scale), 1.0)

    t0 = time.perf_counter()
    next_t = t0
    last_print = 0.0
    phase = "ramp"
    phase_t0 = t0
    frame = 0
    overruns = 0
    dev_clamped = 0
    loop_ms = 0.0
    trip_printed = False

    try:
        while rclpy.ok():
            tick_t = now = time.perf_counter()
            imus, jmap = read_jmap()

            roll = roll_of(np.asarray(imus[args.base_imu].quat, np.float32))
            if phase == "policy" and abs(roll) > args.roll_abort:
                print(f"\n[ABORT] roll {roll:+.2f} rad exceeds {args.roll_abort}. Holding pose.")
                adapter.freeze("roll abort")
                phase = "done"; phase_t0 = now

            elapsed = now - phase_t0
            gain_now = args.gain_scale
            adapt_on = 0
            err_masked = err_leg = err_all = float("nan")
            act_dev = 0.0

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
                    print("[phase] wait_init -- verifying measured pose == start\n")

            elif phase == "wait_init":
                gain_now = init_gain
                target_by_name = dict(start_ref)
                err = pose_errors(jmap)
                arm_bad, leg_bad = pose_ok(err)
                if (not arm_bad and not leg_bad) or args.force_engage:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"[init] READY  worst_arm {wa[0]}={wa[1]:+.3f}  "
                          f"worst_leg {wl[0]}={wl[1]:+.3f}")
                    obs_builder.align(imus[args.base_imu].quat)
                    # Trained weights at every engage: adaptation never carries over.
                    adapter.reset()
                    phase = "policy"; phase_t0 = now; frame = 0
                    print(f"\n[phase] policy ENGAGED -- adaptation "
                          f"{'ON' if adapting else 'OFF'}\n")
                elif elapsed >= args.init_timeout:
                    wa, wl = worst_err(err, ARM_JOINTS), worst_err(err, LEG_JOINTS)
                    print(f"\n[ABORT] start pose not reached in {args.init_timeout:.0f}s.")
                    print(f"[ABORT] worst_arm {wa[0]}={wa[1]:+.3f}  "
                          f"worst_leg {wl[0]}={wl[1]:+.3f}")
                    phase = "done"; phase_t0 = now

            elif phase == "policy":
                tick = int(elapsed / CONTROL_DT)
                frame = min(tick, len(ref_pos) - 1)
                q = np.array([jmap[n].position for n in joint_names], np.float64)

                # Close the adaptation loop on the PREVIOUS action first: `q` is the
                # state that action produced, so (q - ref[frame]) is its error. Same
                # pairing as the Isaac harness (act -> step -> measure -> update).
                err_vec = q - ref_pos[frame]
                err_masked = float(np.degrees(np.abs(err_vec[masked])).mean())
                err_leg = float(np.degrees(np.abs(err_vec[legs])).mean())
                err_all = float(np.degrees(np.abs(err_vec)).mean())
                adapter.update(err_vec, CONTROL_DT)

                obs = obs_builder.build(imus, jmap, frame)
                if args.max_action_dev > 0.0 and adapting and not adapter.disabled:
                    a_frozen = adapter.act_frozen(obs)
                    action = adapter.act(obs)
                    dev = action - a_frozen
                    act_dev = float(np.abs(dev).max())
                    if act_dev > args.max_action_dev:
                        action = a_frozen + np.clip(dev, -args.max_action_dev,
                                                    args.max_action_dev)
                        dev_clamped += 1
                else:
                    action = adapter.act(obs)
                action = np.asarray(action, np.float32).reshape(-1)
                adapt_on = int(adapting and not adapter.disabled
                               and adapter.step > adapter.engage_step)

                # Feed back the action actually APPLIED, not the raw adapted one.
                obs_builder.last_action = action
                raw_target = action * action_scale + default
                target_by_name = {}
                for i, n in enumerate(joint_names):
                    tgt = raw_target[i]
                    if args.leg_filter > 0.0 and n in LEG_SET:
                        tgt = (1.0 - args.leg_filter) * tgt + args.leg_filter * prev_target[n]
                    step = float(np.clip(tgt - prev_target[n],
                                         -args.max_joint_step, args.max_joint_step))
                    target_by_name[n] = prev_target[n] + step

                # No freeze() here: `update` only runs in the policy phase, so leaving
                # `disabled` clear keeps it meaning "a safety guard tripped".
                if tick >= n_frames + int(args.hold_end_seconds / CONTROL_DT):
                    phase = "done"; phase_t0 = now
                    print("\n[phase] motion complete -> ramping to default\n")

            else:  # done -- ramp to the default pose and hold it
                alpha = min(1.0, elapsed / 2.0)
                target_by_name = {n: (1 - alpha) * prev_target[n] + alpha * float(default[i])
                                  for i, n in enumerate(joint_names)}
                if alpha >= 1.0 and elapsed > 3.0 and not args.hold_forever:
                    publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                                 gain_now, args.engage)
                    print("[phase] at default pose -- exiting (--hold-forever to stay up)")
                    break

            publish_pose(commander, target_by_name, kp_by_name, kd_by_name,
                         gain_now, args.engage)
            prev_target = target_by_name
            logger.log(now - t0, phase, frame, imus, jmap, target_by_name)
            alog.log(t_s=f"{now - t0:.4f}", phase=phase, frame=frame, adapt_on=adapt_on,
                     drift=f"{adapter.weight_drift:.6f}",
                     drift_frac=f"{adapter.drift_fraction:.6f}",
                     err_masked_deg=f"{err_masked:.4f}",
                     err_leg_deg=f"{err_leg:.4f}",
                     err_all_deg=f"{err_all:.4f}",
                     act_dev_max=f"{act_dev:.4f}", dev_clamped=dev_clamped,
                     loop_ms=f"{loop_ms:.2f}")

            if adapter.disabled and not trip_printed:
                trip_printed = True
                print(f"\n[adapt] DISABLED: {adapter.disabled_reason}. Reverted to "
                      f"trained weights; the run continues frozen.\n")

            if now - last_print >= 1.0:
                last_print = now
                tag = "DRY" if not args.engage else "CMD"
                extra = ""
                if phase == "policy" and adapting and np.isfinite(err_masked):
                    extra = (f" drift={adapter.weight_drift:.3f}"
                             f" err_mask={err_masked:5.2f}d"
                             f" err_leg={err_leg:5.2f}d"
                             f" dev={act_dev:.2f}"
                             f"{' [OFF]' if adapter.disabled else ''}")
                print(f"[{tag}] phase={phase:9s} t={now - t0:5.1f}s "
                      f"frame={frame:3d}/{n_frames} roll={roll:+.2f} "
                      f"loop={loop_ms:4.1f}ms{extra}")

            loop_ms = (time.perf_counter() - tick_t) * 1e3
            if phase == "policy" and loop_ms > CONTROL_DT * 1e3:
                overruns += 1
                if overruns == args.max_overrun and not adapter.disabled:
                    adapter.freeze(f"{overruns} ticks missed the "
                                   f"{CONTROL_DT * 1e3:.0f} ms deadline")

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
            publish_pose(commander, tgt, kp_by_name, kd_by_name, args.gain_scale, args.engage)
            try:
                imus_i, jmap_i = read_jmap()
                logger.log(time.perf_counter() - t0, "interrupt", frame, imus_i, jmap_i, tgt)
            except Exception:
                pass
            time.sleep(CONTROL_DT)
    finally:
        w0_norm = float(np.linalg.norm(adapter.W0[args.layer]))
        print("\n" + "=" * 78)
        print(f"  adaptation:     {'ON' if adapting else 'OFF (frozen control arm)'}"
              f"   mask={args.mask}  gain={args.gain:g}")
        print(f"  peak drift:     {adapter.peak_drift:.4f} of max {args.max_drift:g}"
              f"   ({adapter.peak_drift / w0_norm * 100:.2f}% of the layer norm)")
        print(f"  guard tripped:  {adapter.disabled}"
              f"{' -- ' + adapter.disabled_reason if adapter.disabled_reason else ''}")
        print(f"  action clamps:  {dev_clamped} ticks   deadline misses: {overruns}")
        print(f"  reached frame:  {frame}/{n_frames}")
        print("=" * 78)
        alog.close()
        logger.close()
        client.destroy_node()
        commander.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
