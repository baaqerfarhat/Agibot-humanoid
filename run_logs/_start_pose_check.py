#!/usr/bin/env python3
"""What pose does the robot get ramped into before the policy engages, and is it stable?

deploy_x2_box_pickup ramps from wherever the robot is into the reference clip's
frame 0 and only then hands over. So frame 0 is a pose the robot has to hold open
loop, on its own, with a human standing next to it. If the clip does not start
from a standing pose, that hand-off happens in a crouch.

Compares frame 0 against the policy's default standing pose, and against the pose
the reference reaches later, to say how far into the motion the clip begins.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R2D = 180.0 / np.pi
POLICY = Path(__file__).resolve().parents[1] / "box_pickup" / "policy" / \
    "x2_box_policy_walk_feasible_v16_iter30500.npz"
CLIP = Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
            "x2_31dof/whole_body_tracking/sub3_largebox_003_walk_feasible.npz")


def main():
    d = np.load(POLICY, allow_pickle=True)
    m = json.loads(str(d["meta_json"]))
    jn = m["joint_names"]
    default = np.array(m["default_joint_pos"])
    ref = d["ref_joint_pos"]

    c = np.load(CLIP, allow_pickle=True)
    root = np.asarray(c["joint_pos"])[:, :3]

    print("THE POSE THE POLICY IS HANDED (reference frame 0) vs STANDING\n" + "=" * 74)
    print(f"  pelvis height at frame 0 : {root[0,2]:.3f} m")
    print(f"  pelvis height, clip max  : {root[:,2].max():.3f} m  (frame {root[:,2].argmax()})")
    print(f"  pelvis height, clip min  : {root[:,2].min():.3f} m  (frame {root[:,2].argmin()})")
    drop = root[:, 2].max() - root[:, 2].min()
    into = (root[:, 2].max() - root[0, 2]) / drop * 100
    print(f"  => frame 0 is {into:.0f}% of the way down into the squat\n")

    print(f"  {'joint':26s} {'default':>9} {'frame 0':>9} {'delta':>9}  (degrees)")
    big = np.argsort(-np.abs(ref[0] - default))[:10]
    for i in big:
        print(f"  {jn[i]:26s} {default[i]*R2D:9.1f} {ref[0,i]*R2D:9.1f} "
              f"{(ref[0,i]-default[i])*R2D:+9.1f}")
    print(f"\n  mean |frame0 - default| over all joints: "
          f"{np.abs(ref[0]-default).mean()*R2D:.1f} deg, max {np.abs(ref[0]-default).max()*R2D:.1f}")

    # how fast does it then descend?
    print("\nDESCENT RATE the policy is asked to produce\n" + "=" * 74)
    vz = np.gradient(root[:, 2]) * 50.0
    az = np.gradient(vz) * 50.0
    k = np.argmin(vz)
    print(f"  peak downward pelvis speed {abs(vz.min()):.3f} m/s at frame {k} (t={k/50:.2f} s)")
    print(f"  peak downward acceleration {abs(az.min()):.2f} m/s2 = {abs(az.min())/9.81:.2f} g")
    print(f"  pelvis falls {root[0,2]-root[:,2].min():.3f} m in "
          f"{root[:,2].argmin()/50:.2f} s from the hand-off")
    # the first half second, which is what a human would perceive as 'sudden'
    n = 25
    print(f"  in the first {n/50:.1f} s after hand-off the pelvis drops "
          f"{root[0,2]-root[n,2]:.3f} m (mean {abs((root[n,2]-root[0,2])/(n/50)):.2f} m/s)")

    # knee rate, which is what you see as a sudden buckle
    ki = [i for i, x in enumerate(jn) if "knee" in x]
    kv = np.abs(np.gradient(ref[:, ki], axis=0) * 50.0)
    print(f"  peak knee rate {kv.max()*R2D:.0f} deg/s at frame {np.unravel_index(kv.argmax(), kv.shape)[0]}")
    print(f"  knee rate in the first 0.5 s: max {kv[:25].max()*R2D:.0f} deg/s")


if __name__ == "__main__":
    main()
