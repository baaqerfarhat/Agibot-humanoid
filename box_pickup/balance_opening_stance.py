"""Centre the pelvis between the feet at the hand-off, so neither leg starts locked.

deploy ramps the robot into frame 0 and only then hands over to the policy, so
frame 0 is a pose the robot holds open loop with nobody driving it. The clip
currently hands over a pose standing on a nearly locked left leg: left knee 3.6
deg against the right's 28.7, a 25 deg split. On the robot that pose had to be
held up by hand.

The split is geometry, not a bad joint angle. At frame 0 the feet are 415 mm apart
laterally -- a wide straddle, taken straight from the mocap -- and the pelvis sits
31 mm off centre towards the RIGHT foot. In a stance that wide, 31 mm of offset is
20 mm of extra reach for the left leg, and with the hip that far from the ankle the
only way to cover it is to run the knee out to nearly straight. seat_opening_frames
then flexed the right knee to plant that foot and left the left one where it was,
which is what opened the split from 5 deg to 25.

So the fix is to move the pelvis, not the knees: slide it laterally to the midpoint
between the ankles and re-solve both legs onto the feet where they already are. Both
legs then need the same reach and take the same knee angle, and the pose the robot is
handed is one it can stand on.

That also fixes the lurch. The left knee has been snapping at 282 deg/s in the first
0.2 s, up from 138 in the v7 clip, because each foot-clearance pass pulled its frame-0
angle straighter -- 14.6 deg down to 3.6 -- while the descent still had to reach the
same depth at the same time. Starting the knee where the descent wants it removes
most of the catch-up, and RATE_CAP holds what is left to something the leg can do.

The correction eases to nothing over TAPER frames, well before the grasp at frame 71,
and the feet do not move at all: only the pelvis over them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_reference_motion import FIXED_FRAMES, LegChain, Robot, leg_ik
from urdf_fk import quat_wxyz_to_mat

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
SPHERES = [k for k in FIXED_FRAMES if "sphere" in k and "ankle" in k]
FLOOR = 0.011
TAPER = 30      # frames over which the pelvis slides back to the clip's own path
RESID = 0.003   # m: an IK solve worse than this is a thrown leg, not a placed foot
RATE_CAP = 160.0  # deg/s: what the opening knee rate is allowed to reach
BOX_MIN = np.array([-0.234, -0.230, -0.198])
BOX_MAX = np.array([0.237, 0.229, 0.210])
BOX_DOWN = 0.30
SPHERE_R = 0.011
R2D = 180.0 / np.pi


def log(m):
    print(m, flush=True)


def main():
    d = dict(np.load(CLIP, allow_pickle=True))
    jn = [str(x) for x in d["joint_names"]]
    q = d["joint_pos"].copy()
    root_pos, root_quat, dof = q[:, 0:3].copy(), q[:, 3:7].copy(), q[:, 7:].copy()
    box_pos = np.asarray(d["object_pos_w"], float).copy()
    box_quat = np.asarray(d["object_quat_w"], float).copy()
    n = len(dof)

    robot = Robot()
    legs = {b: LegChain(robot.chain, b) for b in FEET}
    lim = {nm: robot.lim[nm] for leg in legs.values() for nm in leg.names}
    sole = {
        b: [np.asarray(FIXED_FRAMES[s][1]) for s in SPHERES if FIXED_FRAMES[s][0] == b]
        for b in FEET
    }
    kn = {b: jn.index(("left" if "left" in b else "right") + "_knee_joint") for b in FEET}

    def feet(f, src=None, root=None):
        R0 = quat_wxyz_to_mat(root_quat[f])
        src = dof[f] if src is None else src
        root = root_pos[f] if root is None else root
        out = {}
        for b in FEET:
            p, Rm = legs[b].fk({nm: src[jn.index(nm)] for nm in legs[b].names}, root, R0)
            out[b] = (p, Rm, min((p + Rm @ s)[2] for s in sole[b]))
        return out

    def box_gap(f, src=None, root=None):
        if box_pos[f, 2] > BOX_DOWN:
            return np.inf
        Rb = quat_wxyz_to_mat(box_quat[f])
        cur = feet(f, src, root)
        worst = np.inf
        for b in FEET:
            for s in sole[b]:
                l = Rb.T @ (cur[b][0] + cur[b][1] @ s - box_pos[f])
                o = np.maximum(np.maximum(BOX_MIN - l, l - BOX_MAX), 0.0)
                sd = np.linalg.norm(o) if o.any() else -min(
                    np.min(l - BOX_MIN), np.min(BOX_MAX - l)
                )
                worst = min(worst, sd - SPHERE_R)
        return worst

    knee0 = dof[:, [kn[b] for b in FEET]].copy()
    gap0 = np.array([box_gap(f) for f in range(min(n, 60))])
    high0 = np.array([max(feet(f)[b][2] for b in FEET) for f in range(min(n, 60))])
    rate0 = np.abs(np.gradient(knee0, axis=0) * 50.0 * R2D)
    log(f"before: frame 0 knees L {knee0[0,0]*R2D:.1f} R {knee0[0,1]*R2D:.1f} deg"
        f"  (split {abs(knee0[0,0]-knee0[0,1])*R2D:.1f})")
    log(f"        peak knee rate in the first 0.5 s {rate0[:25].max():.0f} deg/s")

    # --- how far off centre is the pelvis, along the robot's own lateral axis? ---
    off = np.zeros(min(n, TAPER))
    lat = np.zeros((len(off), 3))
    for f in range(len(off)):
        cur = feet(f)
        mid = 0.5 * (cur[FEET[0]][0] + cur[FEET[1]][0])
        R0 = quat_wxyz_to_mat(root_quat[f])
        # the robot's left, flattened into the ground plane: the correction is a
        # sideways slide, not a change of height or of facing
        e = R0 @ np.array([0.0, 1.0, 0.0])
        e[2] = 0.0
        e /= max(np.linalg.norm(e), 1e-9)
        lat[f] = e
        off[f] = float((mid - root_pos[f]) @ e)
    log(f"        pelvis sits {off[0]*1000:+.0f} mm off the ankle midpoint at frame 0"
        f" (+ is towards its left)")
    stance = np.linalg.norm(feet(0)[FEET[0]][0] - feet(0)[FEET[1]][0])
    log(f"        stance {stance*1000:.0f} mm wide")

    # Full correction at frame 0, gone by TAPER, leaving and arriving with zero slope
    # so the pelvis path picks up the clip's own without a corner in it.
    ease = 0.5 * (1.0 + np.cos(np.pi * np.arange(len(off)) / len(off)))

    # Centring alone still leaves the knee snapping, because a nearly straight leg is
    # at a kinematic singularity: at 14 deg of knee, d(leg length)/d(knee) is 57 mm/rad,
    # so the 115 mm the pelvis drops in the first half second has to come out of the
    # knee at hundreds of deg/s. Seating the pelvis a little lower starts both knees
    # off the singularity, where the same descent costs roughly half the knee travel.
    # The drop is the smallest one that brings the rate under RATE_CAP, so the hand-off
    # pose stays as close to the clip's as the requirement allows.
    base_root, base_dof = root_pos.copy(), dof.copy()

    def apply(drop):
        root_pos[:], dof[:] = base_root.copy(), base_dof.copy()
        moved = skipped = 0
        worst = 0.0
        for f in range(len(off)):
            d3 = np.array([0.0, 0.0, -drop * ease[f]])
            delta = off[f] * ease[f] * lat[f] + d3
            if np.abs(delta).max() < 1e-5:
                continue
            cur = feet(f)
            root = base_root[f] + delta
            tgt = {b: (cur[b][0], cur[b][1]) for b in FEET}
            sol = leg_ik(legs, base_dof[f].copy(), jn, root,
                         quat_wxyz_to_mat(root_quat[f]), tgt, lim)
            chk = feet(f, sol, root)
            res = max(np.linalg.norm(chk[b][0] - tgt[b][0]) for b in FEET)
            if res > RESID:
                skipped += 1
                continue
            if box_gap(f, sol, root) < min(gap0[f], 0.0) - 1e-4:
                skipped += 1
                continue
            root_pos[f] = root
            dof[f] = sol
            moved += 1
            worst = max(worst, res)
        r = np.abs(np.gradient(dof[:, [kn[b] for b in FEET]], axis=0) * 50.0 * R2D)
        return r[:25].max(), moved, skipped, worst

    best = None
    for drop in np.arange(0.0, 0.081, 0.005):
        rate, moved, skipped, worst = apply(drop)
        log(f"        pelvis seated {drop*1000:4.0f} mm lower -> knee rate {rate:5.0f} deg/s"
            f"  ({moved} solved, {skipped} skipped)")
        best = (drop, rate, moved, skipped, worst)
        if rate <= RATE_CAP and skipped == 0:
            break
    drop, rate, moved, skipped, worst_res = best
    apply(drop)
    log(f"        chose {drop*1000:.0f} mm; pelvis slid {(off*ease).max()*1000:+.0f} mm sideways"
        f" and {drop*1000:.0f} mm down at frame 0, easing out by frame {len(off)};"
        f" {moved} solved, {skipped} skipped, worst residual {worst_res*1000:.2f} mm")

    rate1 = np.abs(np.gradient(dof[:, [kn[b] for b in FEET]], axis=0) * 50.0 * R2D)

    knee1 = dof[:, [kn[b] for b in FEET]]
    high1 = np.array([max(feet(f)[b][2] for b in FEET) for f in range(min(n, 60))])
    low1 = np.array([min(feet(f)[b][2] for b in FEET) for f in range(min(n, 60))])
    gap1 = np.array([box_gap(f) for f in range(min(n, 60))])
    log(f"\nafter:  frame 0 knees L {knee1[0,0]*R2D:.1f} R {knee1[0,1]*R2D:.1f} deg"
        f"  (split {abs(knee1[0,0]-knee1[0,1])*R2D:.1f})")
    log(f"        peak knee rate in the first 0.5 s {rate1[:25].max():.0f} deg/s"
        f" (was {rate0[:25].max():.0f}, cap {RATE_CAP:.0f})")
    log(f"        soles at frame 0 {(low1[0]-FLOOR)*1000:+.1f} / {(high1[0]-FLOOR)*1000:+.1f} mm"
        f" (was {(high0[0]-FLOOR)*1000:+.1f} for the higher)")
    log(f"        highest foot over the opening {(high1[:TAPER]-FLOOR).max()*1000:+.1f} mm"
        f" (was {(high0[:TAPER]-FLOOR).max()*1000:+.1f})")
    log(f"        box clearance over the first 60 frames {gap1.min()*1000:+.1f} mm"
        f" (was {gap0.min()*1000:+.1f})")
    acc = np.diff(root_pos[:60, 2], 2) * 2500.0
    log(f"        root vertical acceleration over the opening {np.abs(acc).max():.2f} m/s2")
    log(f"        peak joint jerk {np.abs(np.diff(dof, 3, axis=0)).max()*125000.0:.0f} rad/s3")
    log(f"        clip beyond the taper is untouched: max |ddof| after frame {TAPER+10} is"
        f" {np.abs(dof[TAPER+10:] - d['joint_pos'][TAPER+10:, 7:]).max()*R2D:.3f} deg")

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
