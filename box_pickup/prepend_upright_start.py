"""Start the clip standing upright and still, instead of already leaning into the squat.

deploy ramps the robot into reference frame 0 and only then hands control to the
policy, so frame 0 is the pose the robot has to be standing in when the policy takes
over. The clip opens 18.4 deg pitched forward with the waist folded another 17.3, and
already descending: the robot is handed a lean and starts moving the instant it
engages. On hardware that had to be held up by hand.

The clip already contains the pose we want. Its LAST frame is a proper stand -- pitch
0.0, waist 0.0, knees 25.9/25.5, pelvis 0.656 -- which also settles the question of
whether upright is reachable at this stance, since the clip gets there on its own with
the feet 476 mm apart.

So the opening gets a stand built from that: frame 0's own footfall and heading, with
the trunk stood up, the waist unfolded and the arms taken from the clip's final frame.
The robot holds it for HOLD frames, then eases into the clip's original opening over
BLEND. The feet never move -- they are pinned to frame 0's contact through IK for
every added frame -- so the box geometry, the grasp and everything past the join are
untouched, and the clip simply gets longer at the front.

Run after balance_opening_stance.py: that pass centres the pelvis and takes the knees
off their singularity, and this one puts a stand in front of the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_reference_motion import FIXED_FRAMES, URDF, LegChain, Robot, leg_ik, load_masses
from urdf_fk import quat_wxyz_to_mat

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
SPHERES = [k for k in FIXED_FRAMES if "sphere" in k and "ankle" in k]
HOLD = 35        # frames stood still before anything moves (0.70 s, as the v33 clip had)
BLEND = 45       # frames easing from the stand into the clip's own opening (0.90 s)
RESID = 0.003
RATE_CAP = 160.0
R2D = 180.0 / np.pi
BOX_MIN = np.array([-0.234, -0.230, -0.198])
BOX_MAX = np.array([0.237, 0.229, 0.210])
BOX_DOWN = 0.30
SPHERE_R = 0.011
# joints that make the difference between standing and leaning; the legs are not
# listed because they are solved, not blended
TRUNK = ["waist_pitch_joint", "waist_roll_joint"]


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
    lim = {nm: robot.lim[nm] for lg in legs.values() for nm in lg.names}
    leg_names = {nm for lg in legs.values() for nm in lg.names}
    sole = {
        b: [np.asarray(FIXED_FRAMES[s][1]) for s in SPHERES if FIXED_FRAMES[s][0] == b]
        for b in FEET
    }
    mass = load_masses(URDF)

    def feet_of(root, rq, src):
        R0 = quat_wxyz_to_mat(rq)
        out = {}
        for b in FEET:
            p, Rm = legs[b].fk({nm: src[jn.index(nm)] for nm in legs[b].names}, root, R0)
            out[b] = (p, Rm, min((p + Rm @ s)[2] for s in sole[b]))
        return out

    def box_gap(root, rq, src, bp, bq):
        if bp[2] > BOX_DOWN:
            return np.inf
        Rb = quat_wxyz_to_mat(bq)
        cur = feet_of(root, rq, src)
        worst = np.inf
        for b in FEET:
            for s in sole[b]:
                l = Rb.T @ (cur[b][0] + cur[b][1] @ s - bp)
                o = np.maximum(np.maximum(BOX_MIN - l, l - BOX_MAX), 0.0)
                sd = np.linalg.norm(o) if o.any() else -min(
                    np.min(l - BOX_MIN), np.min(BOX_MAX - l)
                )
                worst = min(worst, sd - SPHERE_R)
        return worst

    tgt_feet = {b: (v[0], v[1]) for b, v in feet_of(root_pos[0], root_quat[0], dof[0]).items()}
    rpy0 = Rot.from_quat(np.r_[root_quat[0, 1:], root_quat[0, 0]]).as_euler("xyz")
    log(f"clip opens: pitch {rpy0[1]*R2D:+.1f} deg, waist_pitch "
        f"{dof[0, jn.index('waist_pitch_joint')]*R2D:+.1f}, pelvis {root_pos[0,2]:.3f} m, "
        f"{n} frames")
    log(f"clip ends:  pitch "
        f"{Rot.from_quat(np.r_[root_quat[-1,1:],root_quat[-1,0]]).as_euler('xyz')[1]*R2D:+.1f} deg, "
        f"waist_pitch {dof[-1, jn.index('waist_pitch_joint')]*R2D:+.1f} -- the stand to copy")

    # ---- the standing pose: frame 0's footfall and heading, trunk stood up --------
    stand = dof[-1].copy()          # arms, head and waist as the clip's own stand has them
    for nm in leg_names:            # legs get solved onto frame 0's feet, not copied
        stand[jn.index(nm)] = dof[0, jn.index(nm)]
    up_quat = Rot.from_euler("z", rpy0[2]).as_quat()      # yaw only: no lean, no roll
    up_quat = np.r_[up_quat[3], up_quat[:3]]              # wxyz

    # Height: stand as tall as the stance allows without running a knee to its stop.
    # A 423 mm stance caps this -- the leg is only so long -- so it is searched rather
    # than assumed, and the tallest solution that keeps both feet planted wins.
    best = None
    for z in np.arange(root_pos[0, 2] + 0.05, root_pos[0, 2] - 0.06, -0.005):
        root = np.r_[root_pos[0, :2], z]
        sol = leg_ik(legs, stand.copy(), jn, root, quat_wxyz_to_mat(up_quat), tgt_feet, lim)
        chk = feet_of(root, up_quat, sol)
        res = max(np.linalg.norm(chk[b][0] - tgt_feet[b][0]) for b in FEET)
        kneeL = sol[jn.index("left_knee_joint")] * R2D
        kneeR = sol[jn.index("right_knee_joint")] * R2D
        if res < RESID and min(kneeL, kneeR) > 8.0:
            best = (z, sol, res, kneeL, kneeR)
            break
    if best is None:
        raise SystemExit("no standing height keeps both feet planted with the knees off their stops")
    z0, stand_dof, res, kL, kR = best
    stand_root = np.r_[root_pos[0, :2], z0]
    log(f"stand: pelvis {z0:.3f} m, knees L {kL:.1f} R {kR:.1f} deg (split {abs(kL-kR):.1f}),"
        f" IK residual {res*1000:.2f} mm")

    # ---- build the lead-in --------------------------------------------------------
    # Smoothstep leaves and arrives with zero slope, so the stand is genuinely still and
    # the join into the clip has no corner in it. The clip's own opening velocity is
    # picked up over the last few frames of the blend rather than stepped into.
    add = HOLD + BLEND
    u = np.clip((np.arange(add) - HOLD) / BLEND, 0.0, 1.0)
    w = u * u * (3.0 - 2.0 * u)

    new_root = np.zeros((add, 3))
    new_quat = np.zeros((add, 4))
    new_dof = np.zeros((add, dof.shape[1]))
    slerp = Rot.from_quat(np.array([np.r_[up_quat[1:], up_quat[0]],
                                    np.r_[root_quat[0, 1:], root_quat[0, 0]]]))
    from scipy.spatial.transform import Slerp

    sl = Slerp([0.0, 1.0], slerp)
    worst_res = 0.0
    worst_gap = np.inf
    for j in range(add):
        root = (1 - w[j]) * stand_root + w[j] * root_pos[0]
        rq = sl(w[j]).as_quat()
        rq = np.r_[rq[3], rq[:3]]
        base = (1 - w[j]) * stand_dof + w[j] * dof[0]
        sol = leg_ik(legs, base.copy(), jn, root, quat_wxyz_to_mat(rq), tgt_feet, lim)
        chk = feet_of(root, rq, sol)
        r = max(np.linalg.norm(chk[b][0] - tgt_feet[b][0]) for b in FEET)
        worst_res = max(worst_res, r)
        worst_gap = min(worst_gap, box_gap(root, rq, sol, box_pos[0], box_quat[0]))
        new_root[j], new_quat[j], new_dof[j] = root, rq, sol

    root_pos = np.r_[new_root, root_pos]
    root_quat = np.r_[new_quat, root_quat]
    dof = np.r_[new_dof, dof]
    box_pos = np.r_[np.repeat(box_pos[:1], add, axis=0), box_pos]
    box_quat = np.r_[np.repeat(box_quat[:1], add, axis=0), box_quat]
    N = len(dof)

    # ---- report -------------------------------------------------------------------
    kn = [jn.index("left_knee_joint"), jn.index("right_knee_joint")]
    rate = np.abs(np.gradient(dof[:, kn], axis=0) * 50.0 * R2D)
    hi = np.array([max(feet_of(root_pos[f], root_quat[f], dof[f])[b][2] for b in FEET)
                   for f in range(add + 10)])
    lo = np.array([min(feet_of(root_pos[f], root_quat[f], dof[f])[b][2] for b in FEET)
                   for f in range(add + 10)])
    rpy = Rot.from_quat(np.c_[root_quat[:, 1:], root_quat[:, 0]]).as_euler("xyz") * R2D
    log(f"\nprepended {add} frames ({add/50:.2f} s): {HOLD} standing still, {BLEND} easing in")
    log(f"  frame 0 now: pitch {rpy[0,1]:+.1f} deg, roll {rpy[0,0]:+.1f}, "
        f"waist_pitch {dof[0, jn.index('waist_pitch_joint')]*R2D:+.1f}, "
        f"knees L {dof[0,kn[0]]*R2D:.1f} R {dof[0,kn[1]]*R2D:.1f}")
    log(f"  first {HOLD} frames are still: max joint motion "
        f"{np.abs(np.diff(dof[:HOLD], axis=0)).max()*R2D:.4f} deg/frame, "
        f"root moves {np.abs(np.diff(root_pos[:HOLD], axis=0)).max()*1000:.4f} mm/frame")
    log(f"  peak knee rate over the whole lead-in {rate[:add+10].max():.0f} deg/s (cap {RATE_CAP:.0f})")
    log(f"  feet stay planted: soles {(lo.min()-0.011)*1000:+.1f} to "
        f"{(hi.max()-0.011)*1000:+.1f} mm about the floor")
    log(f"  box clearance across the lead-in {worst_gap*1000:+.1f} mm; worst IK residual "
        f"{worst_res*1000:.2f} mm")
    log(f"  root vertical acceleration {np.abs(np.diff(root_pos[:add+10,2],2)*2500).max():.2f} m/s2")
    log(f"  peak joint jerk over the join "
        f"{np.abs(np.diff(dof[max(0,add-15):add+15], 3, axis=0)).max()*125000.0:.0f} rad/s3")

    # CoM against the support polygon, at the stand
    tot, acc = 0.0, np.zeros(2)
    poses = robot.chain.fk({nm: dof[0, jn.index(nm)] for nm in jn}, root_pos[0], root_quat[0])
    for name, (mk, c) in mass.items():
        if name in poses:
            p, Rm = poses[name]
            acc += mk * (p + Rm @ np.asarray(c))[:2]
            tot += mk
    com = acc / tot
    fp = feet_of(root_pos[0], root_quat[0], dof[0])
    pts = np.array([(fp[b][0] + fp[b][1] @ s)[:2] for b in FEET for s in sole[b]])
    log(f"  CoM at the stand sits {np.linalg.norm(com - pts.mean(axis=0))*1000:.0f} mm from the"
        f" centre of the footprint (footprint spans {np.ptp(pts[:,0])*1000:.0f} x"
        f" {np.ptp(pts[:,1])*1000:.0f} mm)")

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} -> {N} frames, {N/50:.2f} s)")


if __name__ == "__main__":
    main()
