"""Report what a reference clip asks of the X2 that the X2 cannot deliver.

Run it on a clip before and after any edit so the change is measured, not assumed.
Every number here is a hardware fact: URDF joint limits, URDF velocity limits,
actuator effort limits, leg length, and the foot contact polygon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).parent))
from check_static_torque import effort_limits as load_efforts_urdf  # noqa: E402
from check_static_torque import subtrees  # noqa: E402
from rebuild_reference_motion import (  # noqa: E402
    FPS,
    URDF,
    Robot,
    load_vel_limits,
    support_margin,
)

SPHERES = [f"{s}_ankle_roll_sphere_{i}_link" for s in ("left", "right") for i in range(1, 6)]
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
STANCE_Z = 0.02  # m above the clip's lowest contact => that foot is carrying load
STANCE_V = 0.15  # m/s below which a loaded foot counts as planted


def analyse(path, label):
    d = np.load(path, allow_pickle=True)
    jn = [str(x) for x in d["joint_names"]]
    bn = [str(x) for x in d["body_names"]]
    qp = np.asarray(d["joint_pos"])
    rp, rq, dof = qp[:, 0:3], qp[:, 3:7], qp[:, 7:]
    bp, op = d["body_pos_w"], d["object_pos_w"]
    n = len(dof)
    robot = Robot()
    vlim = load_vel_limits(URDF)
    eff = load_efforts_urdf(URDF)
    dt = 1.0 / FPS

    print(f"\n{'='*78}\n{label}   ({n} frames, {n/FPS:.2f}s)\n{'='*78}")

    print("\n[joint position limits]")
    bad = []
    for i, nm in enumerate(jn):
        lo, hi = robot.lim.get(nm, (-np.inf, np.inf))
        over = max(lo - dof[:, i].min(), dof[:, i].max() - hi, 0.0)
        if over > 1e-6:
            bad.append((nm, np.degrees(over)))
    print(f"  violations: {len(bad)}")
    for nm, o in sorted(bad, key=lambda x: -x[1])[:6]:
        print(f"    {nm:28s} over by {o:6.1f} deg")

    print("\n[joint velocity limits]")
    dv = np.gradient(dof, dt, axis=0)
    bad = []
    for i, nm in enumerate(jn):
        vl = vlim.get(nm)
        if vl:
            pk = np.abs(dv[:, i]).max()
            if pk > vl:
                bad.append((nm, pk, vl))
    print(f"  violations: {len(bad)}")
    for nm, pk, vl in sorted(bad, key=lambda x: -x[1] / x[2])[:6]:
        print(f"    {nm:28s} {pk:6.1f} rad/s vs limit {vl:5.1f}  ({pk/vl*100:.0f}%)")

    print("\n[hands]")
    mid = (bp[:, bn.index("left_hand_contact_link")] + bp[:, bn.index("right_hand_contact_link")]) / 2
    print(f"  palm midpoint z: {mid[:,2].min():.3f} .. {mid[:,2].max():.3f} m")
    print(f"  frames with palms below 15 cm: {(mid[:,2]<0.15).sum()} of {n}"
          f"   below 10 cm: {(mid[:,2]<0.10).sum()}")

    print("\n[feet / ankles]")
    floor = min(bp[:, bn.index(s), 2].min() for s in SPHERES)
    for s in ("left", "right"):
        i = bn.index(f"{s}_ankle_roll_link")
        sph = [bn.index(x) for x in SPHERES if x.startswith(s)]
        low = bp[:, sph, 2].min(axis=1)
        vel = np.r_[0, np.linalg.norm(np.diff(bp[:, i, :2], axis=0), axis=1)] * FPS
        loaded = low < floor + STANCE_Z
        planted = loaded & (vel < STANCE_V)
        slip = vel[loaded]
        pen = (floor - low[low < floor]).max() * 1000 if (low < floor).any() else 0.0
        rollj = np.degrees(dof[:, jn.index(f"{s}_ankle_roll_joint")])
        _ = rollj
        pitchj = np.degrees(dof[:, jn.index(f"{s}_ankle_pitch_joint")])
        # sole tilt while carrying load
        tilt = []
        for f in np.where(loaded)[0]:
            R = Rot.from_quat(d["body_quat_w"][f, i][[1, 2, 3, 0]]).as_matrix()
            tilt.append(np.degrees(np.arccos(np.clip((R @ [0, 0, 1.0])[2], -1, 1))))
        # A foot swinging low still trips a height-only contact test, so measure the
        # thing that actually matters: how far it creeps across a real stance.
        runs, k = [], 0
        stance = loaded & (vel < STANCE_V)
        while k < n:
            if stance[k]:
                j = k
                while j < n and stance[j]:
                    j += 1
                if j - k >= 5:
                    runs.append((k, j))
                k = j
            else:
                k += 1
        drift = [np.linalg.norm(bp[c - 1, i, :2] - bp[a, i, :2]) for a, c in runs]
        roll_lim = robot.lim[f"{s}_ankle_roll_joint"]
        pinned = np.sum(
            np.minimum(np.abs(rollj - np.degrees(roll_lim[0])),
                       np.abs(rollj - np.degrees(roll_lim[1]))) < 0.5
        )
        print(f"  {s:5s} loaded {loaded.sum():3d}/{n} frames, planted {planted.sum():3d}")
        print(f"        {len(runs)} stances, foot creep across each:"
              f" mean {np.mean(drift)*1000:4.0f} mm, worst {np.max(drift)*1000:4.0f} mm")
        print(f"        ankle roll hard against its limit on {pinned} frames")
        print(f"        sole tilt while loaded: mean {np.mean(tilt):5.1f}, peak {np.max(tilt):5.1f} deg")
        print(f"        ankle roll {rollj.min():+6.1f}..{rollj.max():+6.1f}, pitch {pitchj.min():+6.1f}..{pitchj.max():+6.1f} deg")
        print(f"        penetration below lowest contact: {pen:.0f} mm")

    print("\n[leg reach]  (pelvis-to-ankle; straight leg is 0.618 m)")
    for b in FEET:
        dd = np.linalg.norm(bp[:, bn.index(b)] - rp, axis=1)
        print(f"  {b.split('_')[0]:5s} max {dd.max():.3f} m   frames over 0.615: {(dd>0.615).sum()}")

    print("\n[box]")
    acc = np.linalg.norm(np.gradient(np.gradient(op, dt, axis=0), dt, axis=0), axis=1)
    spd = np.linalg.norm(np.gradient(op, dt, axis=0), axis=1)
    jump = np.abs(np.diff(op, axis=0)).max() * 1000
    print(f"  z {op[:,2].min():.3f}..{op[:,2].max():.3f} m, peak speed {spd.max():.2f} m/s,"
          f" peak accel {acc.max()/9.81:.1f} g, largest 1-frame jump {jump:.0f} mm")
    off = mid - op
    held = op[:, 2] > op[:, 2].min() + 0.05
    print(f"  palm->box offset while lifted: varies by {np.ptp(off[held],axis=0).max()*1000:.0f} mm"
          f"  (0 = rigid grasp)")

    print("\n[gravity torque vs actuator effort]")
    sub = subtrees(robot.chain)
    jinfo = {j.name: j for j in robot.chain.joints}
    held = op[:, 2] > op[:, 2].min() + 0.05
    # A joint only carries the box if a hand hangs below it, and a joint with just
    # one hand below it carries half -- charging the whole box to each wrist
    # separately double-counts and reads as a 250 % overload that is not there.
    carries = {
        nm: sum(f"{s}_wrist_roll_link" in sub.get(nm, []) for s in ("left", "right")) / 2.0
        for nm in jn
    }
    for payload in (0.0, 3.0):
        peak = {nm: 0.0 for nm in jn}
        for f in range(0, n, 2):
            out = robot.fk(dof[f], jn, rp[f], rq[f])
            lm = {
                link: (m, out[link][0] + out[link][1] @ c)
                for link, (m, c) in robot.mass.items()
                if link in out
            }
            for nm in jn:
                if nm not in sub or nm not in eff:
                    continue
                jp, jR = out[jinfo[nm].child]
                tau = np.zeros(3)
                for link in sub[nm]:
                    if link in lm:
                        m_, c = lm[link]
                        tau += np.cross(c - jp, m_ * np.array([0.0, 0.0, -9.81]))
                if payload and held[f] and carries[nm]:
                    tau += np.cross(
                        op[f] - jp, carries[nm] * payload * np.array([0.0, 0.0, -9.81])
                    )
                peak[nm] = max(peak[nm], abs(float((jR @ jinfo[nm].axis) @ tau)))
        rows = sorted(((nm, peak[nm], eff[nm]) for nm in jn if nm in eff), key=lambda r: -r[1] / r[2])
        nover = sum(1 for _, t, e in rows if t > e)
        top = "  ".join(f"{nm.replace('_joint','')} {t/e*100:.0f}%" for nm, t, e in rows[:4])
        print(f"  payload {payload:.0f} kg: over limit on {nover} joints |  {top}")

    print("\n[balance]")
    com = np.array([robot.com(robot.fk(dof[f], jn, rp[f], rq[f]))[0] for f in range(n)])
    ct = np.zeros((n, len(SPHERES), 3))
    for f in range(n):
        ct[f] = bp[f, [bn.index(s) for s in SPHERES]]
    ca = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
    zmp = com[:, :2] - (com[:, 2:3] / 9.81) * ca[:, :2]
    cm = np.array([support_margin(ct[f], com[f, :2])[0] for f in range(n)])
    zm = np.array([support_margin(ct[f], zmp[f])[0] for f in range(n)])
    print(f"  planar CoM accel peak {np.linalg.norm(ca[:,:2],axis=1).max():5.2f} m/s^2")
    print(f"  CoM margin  min {cm.min()*1000:+6.0f} mm, outside on {(cm<0).sum():3d}/{n}")
    print(f"  ZMP margin  min {zm.min()*1000:+6.0f} mm, outside on {(zm<0).sum():3d}/{n}")

    print("\n[smoothness]")
    jerk = np.abs(np.gradient(np.gradient(np.gradient(dof, dt, axis=0), dt, axis=0), dt, axis=0)).max()
    print(f"  peak joint jerk {jerk:.0f} rad/s^3")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        label, path = a.split("=", 1)
        analyse(path, label)
