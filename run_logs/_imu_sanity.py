#!/usr/bin/env python3
"""Is the IMU itself sane, and is the reconstruction on top of it sane?

Two independent checks that need no ground truth:

  1. the gyro against its own quaternion. A quaternion stream differentiated gives
     an angular velocity that must match the gyro that produced it. If it does not,
     the two are not the same body, not the same frame, or one is stale.

  2. the reconstructed pelvis rate against the torso gyro. The torso sits ABOVE the
     waist, so whatever the waist is doing gets ADDED to the pelvis motion to make
     the torso motion. Over a whole run the torso should therefore be the busier
     signal. If the reconstruction comes out consistently larger than the sensor it
     is built from, it is adding rather than removing the waist contribution -- or
     it is adding noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "20260825_173138_box_pickup_x2_box_policy_walk_feasible_v16_iter30500.csv"
    lines = csv.read_text().splitlines()
    hdr = lines[0].split(",")
    rows = [ln for ln in lines[1:] if ln.count(",") == len(hdr) - 1]
    raw = np.genfromtxt(rows, delimiter=",", dtype=float)
    col = {c: i for i, c in enumerate(hdr)}
    ph = np.array([r.split(",")[col["phase"]] for r in rows])
    m = ph == "policy"
    meta = json.loads(csv.with_suffix("").with_suffix(".meta.json").read_text())
    jn = meta["joint_names"]

    t = raw[m, col["t_s"]]
    q = np.stack([raw[m, col[f"base_quat_{c}"]] for c in "xyzw"], axis=1)
    gyro = np.stack([raw[m, col[f"base_ang_vel_{c}"]] for c in "xyz"], axis=1)
    pel = np.stack([raw[m, col[f"pelvis_ang_vel_{c}"]] for c in "xyz"], axis=1)

    print(f"{csv.name}\n{'='*74}")

    # ---- 1. gyro vs its own quaternion ----------------------------------
    R = Rot.from_quat(q)
    dt = np.diff(t)
    # body-frame rate from consecutive attitudes
    wq = (R[:-1].inv() * R[1:]).as_rotvec() / dt[:, None]
    g = gyro[:-1]
    print("\n[1] torso gyro vs the derivative of the torso quaternion (body frame)")
    print(f"    |gyro|     mean {np.linalg.norm(g, axis=1).mean():.3f} rad/s")
    print(f"    |dquat/dt| mean {np.linalg.norm(wq, axis=1).mean():.3f} rad/s")
    for i, ax in enumerate("xyz"):
        c = np.corrcoef(g[:, i], wq[:, i])[0, 1]
        sc = np.polyfit(wq[:, i], g[:, i], 1)[0]
        print(f"    axis {ax}: correlation {c:+.3f}, gyro/dquat slope {sc:+.2f}")
    err = np.linalg.norm(g - wq, axis=1)
    print(f"    residual |gyro - dquat/dt| mean {err.mean():.3f} "
          f"({100*err.mean()/max(np.linalg.norm(wq,axis=1).mean(),1e-9):.0f}% of the signal)")
    print("    -> a high correlation with slope ~1 on all three axes means the gyro")
    print("       and the quaternion describe the same body in the same frame.")

    # ---- 2. reconstruction vs the sensor it is built from ----------------
    ng, npv = np.linalg.norm(gyro, axis=1), np.linalg.norm(pel, axis=1)
    print("\n[2] reconstructed pelvis rate vs the torso gyro it is derived from")
    print(f"    |torso gyro|  mean {ng.mean():.3f}  p95 {np.percentile(ng,95):.3f}")
    print(f"    |pelvis est|  mean {npv.mean():.3f}  p95 {np.percentile(npv,95):.3f}")
    print(f"    pelvis is larger on {100*(npv>ng).mean():.0f}% of ticks "
          f"(ratio of means {npv.mean()/ng.mean():.2f})")
    print(f"    roughness: torso {np.abs(np.diff(gyro,axis=0)).mean():.3f} -> "
          f"pelvis {np.abs(np.diff(pel,axis=0)).mean():.3f} rad/s per tick")

    waist = [n for n in jn if "waist" in n]
    wv = np.stack([raw[m, col[f"{n}__vel_meas"]] for n in waist], axis=1)
    print(f"\n    waist joint velocities feeding the correction:")
    for i, n in enumerate(waist):
        print(f"      {n:20s} |v| mean {np.abs(wv[:,i]).mean():.3f} max {np.abs(wv[:,i]).max():.3f}, "
              f"per-tick jump {np.abs(np.diff(wv[:,i])).mean():.3f} rad/s")
    print("    -> the correction is only as clean as these are. Encoder-differenced")
    print("       velocity at 50 Hz is the usual reason a reconstruction is rougher")
    print("       than the gyro it started from.")


if __name__ == "__main__":
    main()
