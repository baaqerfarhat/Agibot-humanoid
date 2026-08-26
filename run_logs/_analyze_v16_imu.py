#!/usr/bin/env python3
"""Second stage: is the angular velocity fed to the policy the signal it trained on?

Training reads base_ang_vel off the simulator's articulation root -- the pelvis
freejoint body -- which is exact and noiseless. Hardware has no pelvis gyro, so
deploy reconstructs it from the torso IMU and the waist joint velocities. This
compares three things that should agree and usually do not:

    what the reference motion's own root actually does   (what training saw)
    the raw torso IMU                                    (measured, one sensor)
    the reconstructed pelvis rate                        (what the policy was fed)

and then walks the timeline to see whether the chatter is present from the first
tick or builds up, which separates a bad observation from a divergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / "box_pickup" / "policy" / "x2_box_policy_walk_feasible_v16_iter30500.npz"
CLIP = Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
            "x2_31dof/whole_body_tracking/sub3_largebox_003_walk_feasible.npz")
LEG = ("hip", "knee", "ankle")


def load(csv: Path):
    lines = csv.read_text().splitlines()
    hdr = lines[0].split(",")
    rows = [ln for ln in lines[1:] if ln.count(",") == len(hdr) - 1]
    raw = np.genfromtxt(rows, delimiter=",", dtype=float)
    meta = json.loads(csv.with_suffix("").with_suffix(".meta.json").read_text())
    col = {n: i for i, n in enumerate(hdr)}
    phase = np.array([r.split(",")[col["phase"]] for r in rows])
    return meta, col, raw, phase


def main():
    csv = Path(sys.argv[1])
    meta, col, raw, phase = load(csv)
    jn = meta["joint_names"]
    m = phase == "policy"
    leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]
    waist = [i for i, n in enumerate(jn) if "waist" in n]

    def blk(field, mask=m):
        return np.stack([raw[:, col[f"{j}__{field}"]] for j in jn], axis=1)[mask]

    print(f"{csv.name}\n{'=' * 78}")

    # ---- what did training actually see? --------------------------------
    d = np.load(CLIP, allow_pickle=True)
    from scipy.spatial.transform import Rotation as Rot
    names = [str(x) for x in d["body_names"]]
    bi = names.index("pelvis")
    # Training's base_ang_vel is in the BASE frame, so the world-frame clip value
    # has to be rotated in before the two are comparable.
    w_world = np.asarray(d["body_ang_vel_w"], float)[:, bi]
    Rp = Rot.from_quat(np.c_[np.asarray(d["body_quat_w"])[:, bi, 1:],
                             np.asarray(d["body_quat_w"])[:, bi, 0]])
    rav = Rp.inv().apply(w_world)
    print(f"   (reference body '{names[bi]}', world rate rotated into the base frame)")
    n = np.linalg.norm(rav, axis=1)
    print(f"\nREFERENCE root angular velocity (what the policy was trained against):")
    print(f"   |w| mean {n.mean():.3f}  p95 {np.percentile(n, 95):.3f}  max {n.max():.3f} rad/s")
    print(f"   per-frame jump mean {np.abs(np.diff(rav, axis=0)).mean():.4f} rad/s")

    # ---- what did the robot feed it? ------------------------------------
    imu = np.stack([raw[m, col[f"base_ang_vel_{c}"]] for c in "xyz"], axis=1)
    pel = np.stack([raw[m, col[f"pelvis_ang_vel_{c}"]] for c in "xyz"], axis=1)
    obs = np.stack([raw[m, col[f"obs_ang_vel_{c}"]] for c in "xyz"], axis=1)
    for tag, v in (("raw torso IMU", imu), ("reconstructed pelvis", pel), ("fed to policy", obs)):
        nn = np.linalg.norm(v, axis=1)
        print(f"\n{tag:22s} |w| mean {nn.mean():.3f}  p95 {np.percentile(nn, 95):.3f}  "
              f"max {nn.max():.3f} rad/s")
        print(f"{'':22s} per-tick jump mean {np.abs(np.diff(v, axis=0)).mean():.4f} rad/s")
        print(f"{'':22s} vs reference: {nn.mean()/max(n.mean(),1e-9):.1f}x the mean, "
              f"{np.abs(np.diff(v,axis=0)).mean()/max(np.abs(np.diff(rav,axis=0)).mean(),1e-9):.0f}x the roughness")
    print(f"\nobs == pelvis reconstruction? {np.allclose(obs, pel)}   "
          f"obs == raw torso IMU? {np.allclose(obs, imu)}")

    # is the reconstruction just adding waist-velocity noise?
    wv = blk("vel_meas")[:, waist]
    print(f"\nwaist joint velocity (the reconstruction's other input):")
    print(f"   |v| mean {np.abs(wv).mean():.3f}  max {np.abs(wv).max():.3f} rad/s, "
          f"per-tick jump mean {np.abs(np.diff(wv, axis=0)).mean():.3f}")
    extra = np.linalg.norm(pel - imu, axis=1)
    print(f"   reconstruction adds |pelvis - torso| mean {extra.mean():.3f} max {extra.max():.3f} rad/s")

    # ---- timeline: present from tick 0, or does it grow? -----------------
    tgt = blk("tgt")
    pos = blk("pos_meas")
    dtg = np.abs(np.diff(tgt[:, leg], axis=0)).mean(axis=1)
    nobs = np.linalg.norm(obs, axis=1)
    print(f"\nTIMELINE (policy phase, {m.sum()} ticks):")
    print(f"  {'t_s':>6} {'frame':>6} {'|dtgt|leg':>10} {'|w|obs':>8} {'|q-ref|leg':>11}")
    dref = np.load(POLICY, allow_pickle=True)["ref_joint_pos"]
    fr = raw[m, col["frame"]].astype(int)
    ts = raw[m, col["t_s"]]
    for k in range(0, m.sum() - 1, max(1, (m.sum() - 1) // 22)):
        f = fr[k]
        qe = (np.abs(pos[k, leg] - dref[f, leg]).mean() * 1000) if 0 <= f < len(dref) else float("nan")
        print(f"  {ts[k]-ts[0]:6.2f} {f:6d} {dtg[k]*1000:10.1f} {nobs[k]:8.2f} {qe:11.0f}")


if __name__ == "__main__":
    main()
