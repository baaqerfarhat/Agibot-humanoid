"""Stand the clip's first frames on the floor, on both feet.

The clip spawns the robot in the air. At frame 0 the left sole is 24 mm above the
floor and the right is 5.5 mm above it, so the simulator drops the whole robot,
the right foot lands about 20 ms before the left, and the policy answers the
asymmetric impact by picking the left leg up and replanting it -- a 79 mm swing
the reference never asks for, and the one thing still visibly wrong at the start
of the v12 rollout. It also leaves the ZMP outside the support polygon for the
first 0.4 s, with four frames that have no polygon at all.

Neither half of that is trainable: no policy can decline to fall 5.5 mm.

Only the frames that are actually off the ground are touched, and the correction
runs out on its own by frame 8, where the clip's own feet reach the floor. The
root comes down until the higher foot touches, and both legs are then solved onto
the floor; the right knee takes the difference, flexing to 29 deg where the clip
had 9. That is a real change to the opening posture, but the clip is already
descending at 270 mm/s at frame 0 -- it starts mid-squat, not standing -- so a
flexed right knee is the posture that stance is heading for anyway, and having
both feet down from the first frame is worth more than matching the mocap's
knee angle during a descent the robot cannot balance through on one foot.
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
FLOOR = 0.011  # m: the contact sphere's radius, i.e. its centre height when resting
TOL = 0.0005   # m of daylight under the higher foot before a frame counts as airborne
MAX_WINDOW = 30  # frames. A guard: this is an opening transient, not a whole-clip pass
TAPER = 24     # frames over which the root correction is eased back to nothing
BLEND = 8      # frames of joint smoothing past the window, to blend the correction out
RESID = 0.003  # m: an IK solve worse than this is a thrown leg, not a placed foot
BOX_MIN = np.array([-0.234, -0.230, -0.198])  # the box collision mesh's bounding box, m
BOX_MAX = np.array([0.237, 0.229, 0.210])
BOX_DOWN = 0.30
SPHERE_R = 0.011


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

    def feet(f, src=None, root=None):
        """World pose of each foot and the height of its lowest contact sphere."""
        R0 = quat_wxyz_to_mat(root_quat[f])
        src = dof[f] if src is None else src
        root = root_pos[f] if root is None else root
        out = {}
        for b in FEET:
            p, Rm = legs[b].fk({nm: src[jn.index(nm)] for nm in legs[b].names}, root, R0)
            out[b] = (p, Rm, min((p + Rm @ s)[2] for s in sole[b]))
        return out

    def box_gap(f, src=None, root=None):
        """Signed clearance between the sole spheres and the box; negative means inside."""
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

    high0 = np.array([max(feet(f)[b][2] for b in FEET) for f in range(n)])
    low0 = np.array([min(feet(f)[b][2] for b in FEET) for f in range(n)])
    gap0 = np.array([box_gap(f) for f in range(min(n, 60))])
    log(f"before: frame 0 soles {(low0[0]-FLOOR)*1000:+.1f} / {(high0[0]-FLOOR)*1000:+.1f} mm"
        f" -- the whole robot is {(low0[0]-FLOOR)*1000:.1f} mm off the ground")

    # The window is however long the clip keeps a foot in the air at the start, and
    # nothing beyond that: past frame 8 the feet are on or slightly under the floor,
    # which the seating pass already dealt with and which is not what provokes the step.
    w = 0
    while w < MAX_WINDOW and high0[w] > FLOOR + TOL:
        w += 1
    log(f"        {w} opening frames ({w/50:.2f} s) have a foot off the floor")
    if not w:
        log("nothing to do")
        return

    # Correcting only the airborne frames and stopping dead at touchdown is what put
    # 9 m/s2 into the root height: the correction is still closing at 4 mm/frame when
    # it gets cut off. So the whole thing rides on one ease that leaves and arrives
    # with zero slope, over four times the window that actually needs it. At frame 0
    # each foot is asked for the floor and the root drops the full 24 mm; by the end of
    # the taper both are back to whatever the clip already did, with nothing to blend.
    end = min(n, max(w, TAPER))
    t = np.arange(end) / end
    ease = 0.5 * (1.0 + np.cos(np.pi * t))

    own = np.array([[feet(f)[b][2] for b in FEET] for f in range(end)])
    want = own + ease[:, None] * (FLOOR - own)
    dz_all = ease * (FLOOR - own.max(axis=1))

    worst_res, moved, skipped = 0.0, 0, 0
    for f in range(end):
        if abs(dz_all[f]) < 1e-5 and np.abs(want[f] - own[f]).max() < 1e-5:
            continue
        cur = feet(f)
        root = root_pos[f].copy()
        root[2] += dz_all[f]
        tgt = {}
        for i, b in enumerate(FEET):
            p = cur[b][0].copy()
            p[2] += want[f, i] - own[f, i]
            tgt[b] = (p, cur[b][1])
        sol = leg_ik(legs, dof[f].copy(), jn, root, quat_wxyz_to_mat(root_quat[f]), tgt, lim)
        res = max(np.linalg.norm(feet(f, sol, root)[b][0] - tgt[b][0]) for b in FEET)
        if res > RESID:
            # A leg that cannot reach gets thrown rather than falling short, so an
            # over-budget solve is worse than leaving the frame alone.
            log(f"        frame {f}: IK residual {res*1000:.0f} mm, left as it was")
            skipped += 1
            continue
        if box_gap(f, sol, root) < min(gap0[f], 0.0) - 1e-4:
            log(f"        frame {f}: would put a foot back in the box, left as it was")
            skipped += 1
            continue
        root_pos[f] = root
        dof[f] = sol
        moved += 1
        worst_res = max(worst_res, res)

    log(f"        root moved {dz_all.min()*1000:+.1f} to {dz_all.max()*1000:+.1f} mm across"
        f" {end} frames; {moved} solved, {skipped} skipped,"
        f" worst residual {worst_res*1000:.2f} mm")

    # A 6-DOF leg has more than one way to stand, so neighbouring per-frame solves can
    # land on different ones and leave the joint traces ragged. Filtering the window on
    # its own would just move the problem to the window's edge -- that seam cost 8870
    # rad/s3 of jerk in the middle of a joint that was doing 36 -- so the filtered and
    # original traces are crossfaded with the same ease, which is zero exactly where
    # the correction is.
    leg_idx = [jn.index(nm) for lg in legs.values() for nm in lg.names]
    pad = min(n, end + BLEND)
    raw = dof[:pad, leg_idx].copy()
    sm = gaussian_filter1d(raw, 1.0, axis=0, mode="nearest")
    weight = np.r_[ease, np.zeros(pad - end)][:, None]
    dof[:pad, leg_idx] = weight * sm + (1.0 - weight) * raw

    high1 = np.array([max(feet(f)[b][2] for b in FEET) for f in range(min(n, 60))])
    low1 = np.array([min(feet(f)[b][2] for b in FEET) for f in range(min(n, 60))])
    gap1 = np.array([box_gap(f) for f in range(min(n, 60))])
    log(f"after:  frame 0 soles {(low1[0]-FLOOR)*1000:+.1f} / {(high1[0]-FLOOR)*1000:+.1f} mm")
    log(f"        opening window: highest foot now {(high1[:w]-FLOOR).max()*1000:+.1f} mm at worst"
        f" (was {(high0[:w]-FLOOR).max()*1000:+.1f})")
    log(f"        box clearance over the first 60 frames {gap1.min()*1000:+.1f} mm"
        f" (was {gap0.min()*1000:+.1f})")

    acc = np.diff(root_pos[:60, 2], 2) * 2500.0
    log(f"        root vertical acceleration over the opening {np.abs(acc).max():.2f} m/s2")
    jerk = np.abs(np.diff(dof, 3, axis=0)).max() * 125000.0
    log(f"        peak joint jerk {jerk:.0f} rad/s3")

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
