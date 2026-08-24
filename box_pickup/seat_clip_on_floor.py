"""Stand the clip on the floor and stop the feet skating out of the gate.

Two defects survive the refine pass, and neither is something training can absorb
because the policy tracks them faithfully:

1. The whole clip stands the robot IN the floor. The lowest foot contact sphere
   should sit at its own radius, 11 mm, which is where the raw mocap holds it for
   every frame. The refined clip sits at a median of 4.7 mm and dips to -7.4 mm,
   with 456 of 512 frames below floor level. The simulator hides this by ejecting
   the penetration at spawn; hardware has nothing to eject, so the same commands
   drive the feet into the ground for most of the motion.

2. The feet skate for the first 0.16 s. Over frames 1-5 the reference moves the
   FEET 15, 14, 12, 10, 7 mm per frame while moving the root only 6-10 mm -- the
   feet travel further than the body, which is a drag, not a step. The raw mocap
   does the opposite and correct thing: the root decelerates at 25 mm/frame while
   the feet hold to within 1 mm and the legs absorb it.

Both are fixed here rather than in the refine pass, which is fragile enough that
regenerating the clip to change the grip collapsed its stance detection entirely.
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
RAMP = 40  # frames (0.8 s) to spread the opening reposition over
SETTLE = 9  # frames of opening transient, for reporting
LIFT_SIGMA = 9.0  # frames. Smooths the seating so it adds no acceleration of its own


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

    def feet(f, dofs=None):
        """World pose of each foot and the lowest contact sphere under it."""
        R0 = quat_wxyz_to_mat(root_quat[f])
        src = dof[f] if dofs is None else dofs
        out = {}
        for b in FEET:
            p, Rm = legs[b].fk({nm: src[jn.index(nm)] for nm in legs[b].names}, root_pos[f], R0)
            low = min(
                (p + Rm @ np.asarray(FIXED_FRAMES[s][1]))[2]
                for s in SPHERES
                if s.startswith(b.split("_")[0])
            )
            out[b] = (p, Rm, low)
        return out

    def travel():
        """Per-frame horizontal foot travel, mm."""
        c = np.array([[feet(f)[b][0][:2] for b in FEET] for f in range(n)])
        return np.linalg.norm(np.diff(c, axis=0), axis=-1) * 1000

    low0 = np.array([min(feet(f)[b][2] for b in FEET) for f in range(n)])
    t0 = travel()
    log(f"before: lowest sole {low0.min()*1000:+.1f} mm, median {np.median(low0)*1000:.1f} mm,"
        f" {int((low0 < FLOOR - 0.002).sum())}/{n} frames in the floor")
    log(f"        opening foot travel {t0[:SETTLE].max():.1f} mm/frame worst,"
        f" against {np.median(t0):.1f} mm/frame typical")

    # 1. Seat the clip on the floor. Only ever lifts -- a foot legitimately above the
    #    floor is mid-swing and must stay there. Two passes because smoothing the lift
    #    to keep it gentle lets a little penetration back in.
    total = np.zeros(n)
    for _ in range(3):
        low = np.array([min(feet(f)[b][2] for b in FEET) for f in range(n)])
        step = gaussian_filter1d(np.maximum(FLOOR - low, 0.0), LIFT_SIGMA, mode="nearest")
        root_pos[:, 2] += step
        total += step
    # The box rides with the hands, so it has to come up by exactly the same amount
    # wherever the robot is holding it, or the grasp stops being rigid.
    # The box rides with the hands, so it comes up with them -- but by a single
    # constant, not the per-frame lift. The lift varies by a centimetre across the
    # carry and handing that variation to the box doubles its peak acceleration; a
    # constant leaves the grasp offset out by a few mm, well inside the 38 mm it
    # already varies by, and costs the box no acceleration at all.
    carried = box_pos[:, 2] > box_pos[0, 2] + 0.02
    box_pos[carried, 2] += float(total[carried].mean())

    # 2. Spread the opening reposition out. The refine pass leaves the feet 50 mm from
    #    where the clip settles and covers that in six frames, which reads as a skate
    #    and provokes the policy into lifting and replanting. It cannot simply be
    #    pinned: at frame 0 the right knee is already locked at its 0 deg limit, so the
    #    leg has no length left to reach forward with. Ramping the same displacement
    #    over 0.8 s instead brings it down to the clip's own typical rate, and because
    #    both ends of the window are left untouched nothing steps at either boundary.
    #    Orientation follows the clip -- blending that too puts a jump in the ankle
    #    worse than the skate being removed.
    ends = {b: (feet(0)[b][0], feet(RAMP)[b][0]) for b in FEET}
    worst = 0.0
    for f in range(1, RAMP):
        s = (f / RAMP) ** 2 * (3.0 - 2.0 * (f / RAMP))
        cur = feet(f)
        tgt = {b: (ends[b][0] + s * (ends[b][1] - ends[b][0]), cur[b][1]) for b in FEET}
        dof[f] = leg_ik(
            legs, dof[f], jn, root_pos[f], quat_wxyz_to_mat(root_quat[f]), tgt, lim
        )
        worst = max(worst, max(np.linalg.norm(feet(f)[b][0] - tgt[b][0]) for b in FEET))
    log(f"        opening ramp: worst leg IK residual {worst*1000:.1f} mm")

    low1 = np.array([min(feet(f)[b][2] for b in FEET) for f in range(n)])
    t1 = travel()
    log(f"after:  lowest sole {low1.min()*1000:+.1f} mm, median {np.median(low1)*1000:.1f} mm,"
        f" {int((low1 < FLOOR - 0.002).sum())}/{n} frames in the floor")
    log(f"        opening foot travel {t1[:RAMP].max():.1f} mm/frame worst")
    log(f"        clip lifted by {total.mean()*1000:.1f} mm on average,"
        f" {total.max()*1000:.1f} mm at most")

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
