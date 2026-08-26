#!/usr/bin/env python3
"""Why v16 jittered on hardware: compare the logged run against the policy and the reference.

Reads a deploy_x2_box_pickup.py CSV plus the .npz that produced it and reports, in
order, the things that can make a policy chatter on a robot but not in sim:

  1. loop timing        -- a policy trained at a fixed 50 Hz stepped at a varying
                           rate sees velocities scaled by the rate error
  2. target chatter     -- per-tick target motion and sign reversals, against the
                           same quantity computed from the reference
  3. what the robot did -- measured position vs commanded target, i.e. whether the
                           chatter is in the command or only in the servo
  4. state vs reference -- how far the robot was from the trajectory the policy was
                           trained to expect, which is what drives it off-distribution
  5. the IMU            -- whether the angular velocity fed to the policy is the
                           signal training used, and whether it is even sane

Usage:  python _analyze_v16_jitter.py <run.csv> [more.csv ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / "box_pickup" / "policy" / "x2_box_policy_walk_feasible_v16_iter30500.npz"
LEG = ("hip", "knee", "ankle")
CONTROL_DT = 0.02


def load(csv: Path):
    meta = json.loads((csv.with_suffix("")).with_suffix(".meta.json").read_text())
    lines = csv.read_text().splitlines()
    hdr = lines[0].split(",")
    # Several of these runs were aborted mid-write, so the last row is a fragment.
    good = [ln for ln in lines[1:] if ln.count(",") == len(hdr) - 1]
    raw = np.genfromtxt(good, delimiter=",", dtype=float)
    if raw.ndim == 1:
        raw = raw[None]
    col = {n: i for i, n in enumerate(hdr)}
    return meta, col, raw, hdr, good


def block(meta, col, raw, field):
    """[T, 31] of one per-joint quantity, in meta joint order."""
    return np.stack([raw[:, col[f"{j}__{field}"]] for j in meta["joint_names"]], axis=1)


def main():
    d = np.load(POLICY, allow_pickle=True)
    pmeta = json.loads(str(d["meta_json"]))
    ref_q = d["ref_joint_pos"]  # [F, 31] reference joint targets, 50 Hz
    ascale = np.array(pmeta["action_scale"], np.float32)
    default = np.array(pmeta["default_joint_pos"], np.float32)

    for path in sys.argv[1:]:
        csv = Path(path)
        meta, col, raw, hdr, rows = load(csv)
        jn = meta["joint_names"]
        leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]
        t = raw[:, col["t_s"]]
        phase = np.array([r.split(",")[col["phase"]] for r in rows])
        frame = raw[:, col["frame"]]
        run = phase == "policy"

        print("=" * 78)
        print(csv.name)
        print(f"  {len(raw)} ticks, {t[-1] - t[0]:.1f} s;  phases: "
              + ", ".join(f"{p}={int((phase == p).sum())}" for p in dict.fromkeys(phase)))
        print(f"  gain_scale={meta.get('gain_scale')}  torque_ff={meta.get('torque_ff')}  "
              f"action_clip={meta.get('action_clip')}  leg_filter={meta.get('leg_filter')}  "
              f"max_joint_step={meta.get('max_joint_step')}")
        print(f"  base_imu={meta.get('base_imu')}  base_ang_vel_source={meta.get('base_ang_vel_source')}")
        if run.sum() < 5:
            print("  never engaged the policy -- nothing to analyse\n")
            continue

        # ---- 1. loop timing -------------------------------------------------
        dt = np.diff(t[run])
        print(f"\n  [1] loop period: mean {dt.mean()*1000:.2f} ms (target {CONTROL_DT*1000:.0f}), "
              f"sd {dt.std()*1000:.2f}, min {dt.min()*1000:.1f}, max {dt.max()*1000:.1f}")
        late = (dt > 1.5 * CONTROL_DT).sum()
        print(f"      {late} ticks ({100*late/len(dt):.1f}%) took over 1.5x the period; "
              f"effective rate {1/dt.mean():.1f} Hz")

        # ---- 2. target chatter ----------------------------------------------
        tgt = block(meta, col, raw, "tgt")[run]
        dtg = np.diff(tgt, axis=0)
        rev = (np.sign(dtg[1:]) * np.sign(dtg[:-1]) < 0)
        # same quantity on the reference the policy was trained to track
        rdq = np.diff(ref_q, axis=0)
        rrev = (np.sign(rdq[1:]) * np.sign(rdq[:-1]) < 0)
        print(f"\n  [2] per-tick target motion (legs):")
        print(f"      run       mean |dtgt| {np.abs(dtg[:, leg]).mean()*1000:6.1f} mrad, "
              f"p99 {np.percentile(np.abs(dtg[:, leg]), 99)*1000:6.1f}, "
              f"direction reversals {100*rev[:, leg].mean():4.1f}% of ticks")
        print(f"      reference mean |dtgt| {np.abs(rdq[:, leg]).mean()*1000:6.1f} mrad, "
              f"p99 {np.percentile(np.abs(rdq[:, leg]), 99)*1000:6.1f}, "
              f"direction reversals {100*rrev[:, leg].mean():4.1f}% of ticks")
        worst = np.argsort(-np.abs(dtg).mean(axis=0))[:6]
        print("      worst joints by mean |dtgt|:")
        for i in worst:
            print(f"        {jn[i]:26s} {np.abs(dtg[:, i]).mean()*1000:6.1f} mrad/tick, "
                  f"reversals {100*rev[:, i].mean():4.1f}%, "
                  f"range [{tgt[:, i].min():+.2f}, {tgt[:, i].max():+.2f}] rad")

        # ---- 3. did the chatter reach the joints? ---------------------------
        pos = block(meta, col, raw, "pos_meas")[run]
        vel = block(meta, col, raw, "vel_meas")[run]
        err = tgt - pos
        print(f"\n  [3] tracking: mean |tgt-meas| {np.abs(err[:, leg]).mean()*1000:.0f} mrad "
              f"on legs, max {np.abs(err[:, leg]).max()*1000:.0f}")
        print(f"      measured leg vel: mean |v| {np.abs(vel[:, leg]).mean():.2f} rad/s, "
              f"p99 {np.percentile(np.abs(vel[:, leg]), 99):.2f}, max {np.abs(vel[:, leg]).max():.2f}")
        dvel = np.diff(vel, axis=0)
        vrev = (np.sign(dvel[1:]) * np.sign(dvel[:-1]) < 0)
        print(f"      measured leg accel reversals {100*vrev[:, leg].mean():.1f}% of ticks "
              f"(the robot physically shaking, not just the command)")

        # ---- 4. how far off the reference did the robot get? ----------------
        f = frame[run].astype(int)
        ok = (f >= 0) & (f < len(ref_q))
        if ok.sum() > 5:
            rq = ref_q[f[ok]]
            perr = pos[ok] - rq
            print(f"\n  [4] state vs reference over {ok.sum()} ticks "
                  f"(frames {f[ok].min()}-{f[ok].max()} of {len(ref_q)}):")
            print(f"      mean |q-ref| legs {np.abs(perr[:, leg]).mean()*1000:.0f} mrad, "
                  f"max {np.abs(perr[:, leg]).max()*1000:.0f}")
            bad = np.argsort(-np.abs(perr).mean(axis=0))[:6]
            for i in bad:
                print(f"        {jn[i]:26s} mean {np.abs(perr[:, i]).mean()*1000:6.0f} mrad, "
                      f"max {np.abs(perr[:, i]).max()*1000:6.0f}")

        # ---- 5. the IMU ------------------------------------------------------
        print(f"\n  [5] IMU / base state:")
        for tag, keys in (("obs base_ang_vel (fed to policy)",
                           ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]),
                          ("pelvis_ang_vel (reconstructed)",
                           ["pelvis_ang_vel_x", "pelvis_ang_vel_y", "pelvis_ang_vel_z"]),
                          ("obs_ang_vel (logged copy)",
                           ["obs_ang_vel_x", "obs_ang_vel_y", "obs_ang_vel_z"])):
            if all(k in col for k in keys):
                v = np.stack([raw[run, col[k]] for k in keys], axis=1)
                dv = np.diff(v, axis=0)
                print(f"      {tag:34s} |w| mean {np.linalg.norm(v, axis=1).mean():5.2f} "
                      f"max {np.linalg.norm(v, axis=1).max():5.2f} rad/s, "
                      f"per-tick jump mean {np.abs(dv).mean():5.3f} max {np.abs(dv).max():5.2f}")
        if all(k in col for k in ("base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z")):
            q = np.stack([raw[run, col[f"base_quat_{c}"]] for c in "xyzw"], axis=1)
            n = np.linalg.norm(q, axis=1)
            print(f"      base_quat norm {n.min():.4f}..{n.max():.4f} "
                  f"({'ok' if abs(n.mean()-1) < 1e-2 else 'NOT UNIT -- wrong field or order'})")
        print()


if __name__ == "__main__":
    main()
