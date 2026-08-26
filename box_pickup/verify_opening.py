"""Check the pose the policy is handed is one the robot can actually stand in.

Everything here is about the hand-off and the first half second, because that is
where the robot needed catching. Compares against the clip as it was before the
balancing pass, which is what ran on the robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))

CUR = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
OLD = Path("/tmp/clip_pre_balance.npz")
R2D = 180.0 / np.pi
SOLE = np.array([(-0.05, 0.05), (-0.05, -0.05), (0.11, 0.05), (0.11, -0.05), (0.139, 0.0)])


def hull(p):
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def half(q):
        h = []
        for x in q:
            while len(h) > 1 and np.cross(h[-1] - h[-2], x - h[-2]) <= 0:
                h.pop()
            h.append(x)
        return h
    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def margin(pt, poly):
    """Signed distance into the polygon: positive inside."""
    best = -np.inf
    inside = True
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        nrm = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-12)
        d = float(nrm @ (pt - a))
        inside &= d >= 0
        best = max(best, -abs(d))
    d = min(
        abs(np.cross(poly[(i + 1) % len(poly)] - poly[i], pt - poly[i]))
        / max(np.linalg.norm(poly[(i + 1) % len(poly)] - poly[i]), 1e-12)
        for i in range(len(poly))
    )
    return d if inside else -d


def report(tag, path):
    d = np.load(path, allow_pickle=True)
    bn = [str(x) for x in d["body_names"]]
    jn = [str(x) for x in d["joint_names"]]
    bp, bq = np.asarray(d["body_pos_w"]), np.asarray(d["body_quat_w"])
    q = np.asarray(d["joint_pos"])
    li, ri = bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    lk, rk = jn.index("left_knee_joint"), jn.index("right_knee_joint")
    dof = q[:, 7:]

    # support polygon at frame 0 from both soles
    pts = []
    for i in (li, ri):
        R = Rot.from_quat(np.r_[bq[0, i, 1:], bq[0, i, 0]]).as_matrix()
        for s in SOLE:
            pts.append((bp[0, i] + R @ np.r_[s, -0.068])[:2])
    poly = hull(np.array(pts))

    # Proper mass-weighted CoM: each link's own centre of mass, from the URDF, taken
    # into the world through that link's pose. A link origin is not its CoM -- the
    # torso's are 192 mm apart -- and the torso is the heaviest body on the robot, so
    # a centroid of link origins is not the quantity that decides whether it topples.
    from rebuild_reference_motion import URDF, load_masses

    mass = load_masses(URDF)
    tot, acc = 0.0, np.zeros(2)
    for name, (mk, c) in mass.items():
        if name not in bn:
            continue
        i = bn.index(name)
        R = Rot.from_quat(np.r_[bq[0, i, 1:], bq[0, i, 0]]).as_matrix()
        acc += mk * (bp[0, i] + R @ np.asarray(c))[:2]
        tot += mk
    com = acc / tot
    m = margin(com, poly)

    rate = np.abs(np.gradient(dof[:, [lk, rk]], axis=0) * 50.0 * R2D)
    print(f"\n{tag}")
    print(f"  frame-0 knees         L {dof[0,lk]*R2D:5.1f}  R {dof[0,rk]*R2D:5.1f} deg"
          f"   split {abs(dof[0,lk]-dof[0,rk])*R2D:4.1f}")
    print(f"  frame-0 pelvis height {q[0,2]:.3f} m")
    print(f"  peak knee rate <0.5 s {rate[:25].max():5.0f} deg/s")
    print(f"  CoM inside the support polygon by {m*1000:+.0f} mm")
    rp = Rot.from_quat(np.c_[q[:, 4:7], q[:, 3]]).as_euler("xyz") * R2D
    print(f"  frame-0 pelvis roll {rp[0,0]:+.1f} deg, pitch {rp[0,1]:+.1f}")
    print(f"  root vertical accel over the opening "
          f"{np.abs(np.diff(q[:60,2],2)*2500).max():.2f} m/s2")
    return dof, q


def main():
    a, qa = report("BEFORE (what ran on the robot)", OLD)
    b, qb = report("AFTER  (balanced opening)", CUR)
    print(f"\n  the two clips differ by at most "
          f"{np.abs(a[40:]-b[40:]).max()*R2D:.4f} deg after frame 40, and the box path by "
          f"{np.abs(np.asarray(np.load(OLD,allow_pickle=True)['object_pos_w'])-np.asarray(np.load(CUR,allow_pickle=True)['object_pos_w'])).max()*1000:.2f} mm")


if __name__ == "__main__":
    main()
