#!/usr/bin/env python3
"""Diagnose a set of on-robot deploy runs against the reference the policy tracks.

    python analyze_deploy_logs.py '../run_logs/box_pickup_v33_last5/*.csv' \
        --policy ../box_pickup/policy/x2_box_policy_v33_iter253000.npz

Written for the v33 sim-to-real investigation. It separates three quantities that a
single "tracking error" number confuses, and that lead to opposite conclusions:

  **command saturation** -- the deploy script clamps every target into the joint's
    range before publishing, and `RunLogger` records the target BEFORE that clamp. A
    joint whose demand is outside its limit has that demand silently discarded, so a
    huge apparent error there is the clamp, not the actuator. Reported as the share of
    ticks clipped plus the worst demand as a multiple of the joint's range.

  **servo error** -- measured minus what was actually PUBLISHED (post-clamp). This is
    how well the joint obeys a command it can physically reach.

  **task error** -- measured minus the retargeted reference. This is the quantity the
    adaptation law in `layer_adapt.py` regulates.

A joint is only worth putting in an adaptation mask if it carries task error AND is not
saturated: on a joint pinned at its stop, a larger command produces no state change, so
the adaptation update is invisible to the plant while still perturbing the shared layer.

With `--sim ROLLOUT.npz` it contrasts the same metrics against a recorded Isaac rollout
(from `adaptation/adapt_experiments_isaac.py`), which is what localizes a sim-to-real
gap to specific joints.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

# Joint travel limits, mirrored from robot_states_control.robot_model (rad). Kept here
# so this script runs off-robot, without ROS on the path.
LIMITS = {
    "head_yaw_joint": (-0.366, 0.366), "head_pitch_joint": (-0.3838, 0.3838),
    "waist_yaw_joint": (-3.43, 2.382), "waist_pitch_joint": (-0.314, 0.314),
    "waist_roll_joint": (-0.488, 0.488),
    "left_shoulder_pitch_joint": (-3.08, 2.04), "left_shoulder_roll_joint": (-0.061, 2.993),
    "left_shoulder_yaw_joint": (-2.556, 2.556), "left_elbow_joint": (-2.3556, 0.0),
    "left_wrist_yaw_joint": (-2.556, 2.556), "left_wrist_pitch_joint": (-0.558, 0.558),
    "left_wrist_roll_joint": (-1.571, 0.724),
    "right_shoulder_pitch_joint": (-3.08, 2.04), "right_shoulder_roll_joint": (-2.993, 0.061),
    "right_shoulder_yaw_joint": (-2.556, 2.556), "right_elbow_joint": (-2.3556, 0.0),
    "right_wrist_yaw_joint": (-2.556, 2.556), "right_wrist_pitch_joint": (-0.558, 0.558),
    "right_wrist_roll_joint": (-0.724, 1.571),
    "left_hip_pitch_joint": (-2.704, 2.556), "left_hip_roll_joint": (-0.235, 2.906),
    "left_hip_yaw_joint": (-1.684, 3.430), "left_knee_joint": (0.0, 2.4073),
    "left_ankle_pitch_joint": (-0.803, 0.453), "left_ankle_roll_joint": (-0.2625, 0.2625),
    "right_hip_pitch_joint": (-2.704, 2.556), "right_hip_roll_joint": (-2.906, 0.235),
    "right_hip_yaw_joint": (-3.430, 1.684), "right_knee_joint": (0.0, 2.4073),
    "right_ankle_pitch_joint": (-0.803, 0.453), "right_ankle_roll_joint": (-0.2625, 0.2625),
}

DEG = np.degrees


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="Deploy run CSVs (globs are fine).")
    ap.add_argument("--policy", required=True,
                    help="Policy .npz whose ref_joint_pos the runs were tracking.")
    ap.add_argument("--sim", help="Optional recorded Isaac rollout .npz to contrast with.")
    ap.add_argument("--sim-policy", help="Policy .npz used for the sim rollout "
                                         "(defaults to --policy).")
    ap.add_argument("--top", type=int, default=12, help="Rows in the per-joint table.")
    args = ap.parse_args()

    pol = np.load(args.policy, allow_pickle=True)
    meta = json.loads(str(pol["meta_json"]))
    jn = meta["joint_names"]
    ref_all = pol["ref_joint_pos"].astype(float)
    n_frames = len(ref_all)
    lo = np.array([LIMITS[n][0] for n in jn])
    hi = np.array([LIMITS[n][1] for n in jn])

    legs = [i for i, n in enumerate(jn) if any(k in n for k in ("hip", "knee", "ankle"))]
    waist = [i for i, n in enumerate(jn) if "waist" in n]
    arms = [i for i, n in enumerate(jn)
            if any(k in n for k in ("shoulder", "elbow", "wrist"))]

    paths = sorted({p for pat in args.logs for p in glob.glob(pat)
                    if not p.endswith("_adapt.csv")})
    runs = []
    for p in paths:
        with open(p) as f:
            rows = [r for r in csv.DictReader(f) if r["phase"] == "policy"]
        if not rows:
            print(f"[skip] {os.path.basename(p)}: never reached the policy phase")
            continue
        frame = np.clip(np.array([int(r["frame"]) for r in rows]), 0, n_frames - 1)
        tgt = np.array([[float(r[f"{n}__tgt"]) for n in jn] for r in rows])
        mj = p[:-4] + ".meta.json"
        runs.append(dict(
            name=os.path.basename(p)[:15], frame=frame, tgt=tgt,
            cmd=np.clip(tgt, lo, hi),
            pos=np.array([[float(r[f"{n}__pos_meas"]) for n in jn] for r in rows]),
            ref=ref_all[frame],
            t=np.array([float(r["t_s"]) for r in rows]) - float(rows[0]["t_s"]),
            roll=np.array([float(r["roll"]) for r in rows]),
            meta=json.load(open(mj)) if os.path.exists(mj) else {}))
    if not runs:
        raise SystemExit("no usable runs found")

    print("=" * 100)
    print(f"  {len(runs)} run(s)   policy {os.path.basename(args.policy)}   "
          f"reference {n_frames} frames @ {meta['motion_fps']} Hz "
          f"({n_frames / meta['motion_fps']:.1f} s)")
    print("=" * 100)
    print(f"\n{'run':16s} {'gain':>5s} {'ticks':>6s} {'reached':>12s} {'%':>5s} "
          f"{'dur':>6s} {'end roll':>9s} {'max|roll|':>10s}")
    for r in runs:
        print(f"{r['name']:16s} {r['meta'].get('gain_scale', float('nan')):5.1f} "
              f"{len(r['frame']):6d} {r['frame'].max():5d}/{n_frames:<6d} "
              f"{r['frame'].max() / n_frames * 100:4.0f}% {r['t'][-1]:5.1f}s "
              f"{DEG(r['roll'][-1]):+8.1f} {DEG(np.abs(r['roll']).max()):9.1f}")

    pos = np.vstack([r["pos"] for r in runs])
    cmd = np.vstack([r["cmd"] for r in runs])
    raw = np.vstack([r["tgt"] for r in runs])
    ref = np.vstack([r["ref"] for r in runs])
    sat = (raw > hi) | (raw < lo)
    satf = sat.mean(0)
    servo = DEG(np.abs(pos - cmd)).mean(0)
    task = DEG(np.abs(pos - ref)).mean(0)
    task_signed = DEG(pos - ref).mean(0)

    print("\n" + "-" * 100)
    print("COMMAND SATURATION -- demands the joint cannot reach, discarded by the clamp")
    print("-" * 100)
    print(f"{'joint':28s} {'clipped':>8s} {'limit (deg)':>20s} {'worst demand':>13s} {'x range':>8s}")
    any_sat = False
    for i in np.argsort(-satf):
        if satf[i] < 0.01:
            continue
        any_sat = True
        span = max(abs(DEG(lo[i])), abs(DEG(hi[i])))
        worst = (DEG(raw[:, i]).max() if (raw[:, i] > hi[i]).any() else DEG(raw[:, i]).min())
        print(f"{jn[i]:28s} {satf[i] * 100:7.0f}% "
              f"{DEG(lo[i]):+9.1f} {DEG(hi[i]):+9.1f} {worst:+13.1f} "
              f"{abs(worst) / span:7.1f}x")
    if not any_sat:
        print("  none above 1% of ticks")

    print("\n" + "-" * 100)
    print(f"PER-JOINT ERROR, top {args.top} by task error")
    print("  servo = vs published command (post-clamp)   task = vs reference")
    print("-" * 100)
    print(f"{'joint':28s} {'|servo|':>8s} {'|task|':>8s} {'task signed':>12s} {'clipped':>8s}")
    for i in np.argsort(-task)[:args.top]:
        print(f"{jn[i]:28s} {servo[i]:8.2f} {task[i]:8.2f} {task_signed[i]:+12.2f} "
              f"{satf[i] * 100:7.0f}%")

    print("\n" + "-" * 100)
    print("GROUP SUMMARY -- candidate adaptation masks")
    print("-" * 100)
    for label, idx in (("legs", legs), ("waist", waist), ("arms+wrists", arms)):
        print(f"  {label:14s} |servo| {servo[idx].mean():6.2f} deg   "
              f"|task| {task[idx].mean():6.2f} deg   "
              f"clipped {satf[idx].mean() * 100:5.1f}% of ticks")
    usable = [i for i in range(len(jn)) if task[i] > 10.0 and satf[i] < 0.05]
    print("\n  joints with >10 deg task error AND clipped <5% of ticks -- the only ones")
    print("  where an adaptation correction can actually reach the plant:")
    for i in sorted(usable, key=lambda k: -task[k]):
        print(f"    {jn[i]:28s} task {task[i]:6.2f} deg   servo {servo[i]:5.2f} deg")

    if args.sim:
        s = np.load(args.sim, allow_pickle=True)
        sp = np.load(args.sim_policy or args.policy, allow_pickle=True)
        sm = json.loads(str(sp["meta_json"]))
        if sm["joint_names"] != jn:
            raise SystemExit("sim policy joint order differs; cannot compare per joint")
        sref = sp["ref_joint_pos"]
        q = s["dof_pos"]
        stgt = (s["actions"] * np.array(sm["action_scale"], float)
                + np.array(sm["default_joint_pos"], float))
        sfr = np.clip(s["frame"], 0, len(sref) - 1)
        ssat = (stgt > hi) | (stgt < lo)
        stask = DEG(np.abs(q - sref[sfr]))
        print("\n" + "-" * 100)
        print(f"SIM vs REAL   sim: {os.path.basename(args.sim)}, {len(q)} steps")
        if (args.sim_policy or args.policy) != args.policy:
            print(f"  NOTE: sim rollout used {os.path.basename(args.sim_policy)}, "
                  "so per-group errors mix policy versions")
        print("-" * 100)
        print(f"{'metric':44s} {'sim':>10s} {'real':>10s}")
        print(f"{'ticks with any joint clipped':44s} "
              f"{ssat.any(1).mean() * 100:9.1f}% {sat.any(1).mean() * 100:9.1f}%")
        print(f"{'mean joints clipped per tick':44s} "
              f"{ssat.sum(1).mean():10.2f} {sat.sum(1).mean():10.2f}")
        for label, idx in (("legs", legs), ("waist", waist), ("arms+wrists", arms)):
            print(f"{'task error, ' + label + ' (deg)':44s} "
                  f"{stask[:, idx].mean():10.2f} {task[idx].mean():10.2f}")
        worst = np.argsort(-(satf - ssat.mean(0)))[:5]
        print("\n  joints clipped far more on hardware than in sim (the localized gap):")
        for i in worst:
            print(f"    {jn[i]:28s} sim {ssat[:, i].mean() * 100:5.1f}%   "
                  f"real {satf[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
