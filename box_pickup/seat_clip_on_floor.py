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
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

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
FPS = 50.0
BOX_MIN = np.array([-0.234, -0.230, -0.198])  # the box collision mesh's own bounding box, metres
BOX_MAX = np.array([0.237, 0.229, 0.210])
BOX_DOWN = 0.30    # m: above this the box is off the floor and feet under it are fine
FOOT_MARGIN = 0.006  # m of daylight to leave between a sole sphere and the box
SPHERE_R = 0.011     # m: the sole contact spheres own radius
SWING = 0.045  # m of clearance above which a foot is genuinely swinging, not hovering
STILL = 0.15   # m/s of horizontal speed below which it is not swinging either
JOINT_SIGMA = 0.8  # frames of smoothing on the leg joints after per-frame IK
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
    # The URDF's real contact spheres, per foot, in that ankle's frame. An earlier
    # version of the box-clearance pass used a hardcoded five-point sole instead and so
    # kept clearing points the collision geometry does not have, reporting success while
    # 91 frames were still inside the box.
    SOLE_PTS = {
        b: [
            np.asarray(FIXED_FRAMES[k][1])
            for k in FIXED_FRAMES
            if "sphere" in k and FIXED_FRAMES[k][0] == b
        ]
        for b in FEET
    }
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
    #    Horizontal only, and after levelling: this pass has no business setting foot
    #    HEIGHT, and interpolating it undid the levelling over exactly the opening
    #    second the hover was worst in. Orientation follows the clip for the same
    #    reason -- blending it puts a jump in the ankle worse than the skate removed.
    def ramp_opening():
        ends = {b: (feet(0)[b][0][:2], feet(RAMP)[b][0][:2]) for b in FEET}
        worst = 0.0
        for f in range(1, RAMP):
            s = (f / RAMP) ** 2 * (3.0 - 2.0 * (f / RAMP))
            cur = feet(f)
            tgt = {}
            for b in FEET:
                p = cur[b][0].copy()
                p[:2] = ends[b][0] + s * (ends[b][1] - ends[b][0])
                tgt[b] = (p, cur[b][1])
            keep = dof[f].copy()
            dof[f] = leg_ik(
                legs, dof[f], jn, root_pos[f], quat_wxyz_to_mat(root_quat[f]), tgt, lim
            )
            res = max(np.linalg.norm(feet(f)[b][0] - tgt[b][0]) for b in FEET)
            if res > 0.003:
                dof[f] = keep
                continue
            worst = max(worst, res)
        return worst

    # Run once here so the levelling pass that follows sees sensible foot SPEEDS -- its
    # stance test is speed-gated, and against the raw 690 mm/s skate every opening frame
    # reads as swinging and gets skipped, which is exactly the stretch that hovers.
    log(f"        opening ramp: worst leg IK residual {ramp_opening()*1000:.1f} mm")

    # 3. Put each foot down individually. Seating fixes the clip's overall height by
    #    moving the root, so it can only ever put the LOWER foot on the ground -- it
    #    cannot close a left/right difference. The refine pass leaves one: the left foot
    #    floats a median 4.4 mm against the right's 0.9, and spends 44% of the clip in a
    #    5-40 mm limbo, neither planted nor swinging. The raw mocap has both feet at
    #    0.0 mm with a 0.2 mm gap. That hover is what reads as the leg swinging in the
    #    air before the grasp, and the policy holds it there because the clip asks it to.
    #    Only frames that are slow and already low are touched, so real swings survive.
    pos = np.empty((n, len(FEET), 3))
    clr = np.empty((n, len(FEET)))
    for f in range(n):
        ff = feet(f)
        for i, b in enumerate(FEET):
            pos[f, i], clr[f, i] = ff[b][0], ff[b][2]
    drop = np.zeros((n, len(FEET)))
    for i, b in enumerate(FEET):
        spd = np.r_[0.0, np.linalg.norm(np.diff(pos[:, i, :2], axis=0), axis=1)] * FPS
        planted = ((clr[:, i] - FLOOR) < SWING) & (spd < STILL)
        want = gaussian_filter1d(np.where(planted, FLOOR, clr[:, i]), 3.0, mode="nearest")
        drop[:, i] = np.maximum(clr[:, i] - want, 0.0)  # only ever lower a foot
    base = dof.copy()

    def try_drop(f, frac, commit):
        """Lower frame f's feet by `frac` of their drop; return the IK residual."""
        cur = feet(f)
        tgt = {}
        for i, b in enumerate(FEET):
            p = cur[b][0].copy()
            p[2] -= frac * drop[f, i]
            tgt[b] = (p, cur[b][1])
        q = leg_ik(
            legs, base[f].copy(), jn, root_pos[f], quat_wxyz_to_mat(root_quat[f]), tgt, lim
        )
        was = dof[f].copy()
        dof[f] = q
        res = max(np.linalg.norm(feet(f)[b][0] - tgt[b][0]) for b in FEET)
        if not commit:
            dof[f] = was
        return res

    # An unreachable target does not make leg_ik fall short, it makes it swing the leg:
    # with the knee already against its 0 deg stop the solver walks the hip instead and
    # throws the foot a sixth of a metre. So find how much of the drop each frame can
    # actually hold -- but accepting that per frame is what makes the joint traces
    # ragged, because neighbours settle on different fractions and a 6-DOF leg has more
    # than one way to reach a target. Smoothing the FRACTION first keeps the correction
    # continuous, which is far cheaper than smoothing the joints afterwards and having
    # to re-assert everything the smoothing undid.
    ok = np.zeros(n)
    for f in range(n):
        if drop[f].max() < 1e-4:
            ok[f] = 1.0
            continue
        for cand in (1.0, 0.75, 0.5, 0.3, 0.15, 0.0):
            if try_drop(f, cand, commit=False) <= 0.003:
                ok[f] = cand
                break
    frac = np.minimum(gaussian_filter1d(ok, 3.0, mode="nearest"), ok)
    worst_lvl, done, gave_up = 0.0, 0, int((ok < 1e-6).sum())
    for f in range(n):
        if drop[f].max() * frac[f] < 1e-4:
            continue
        res = try_drop(f, frac[f], commit=True)
        # Reach is not monotonic in the target, so a smoothed fraction can land between
        # two that solve and still diverge. Those frames go back to the seated pose,
        # which is merely un-levelled rather than thrown.
        if res > 0.003:
            dof[f] = base[f].copy()
            gave_up += 1
            continue
        worst_lvl = max(worst_lvl, res)
        done += 1
    log(f"        levelled {done}/{n} frames by up to {drop.max()*1000:.0f} mm"
        f" (worst residual {worst_lvl*1000:.1f} mm); {gave_up} out of the legs' reach")

    # The levelling solves each frame on its own, and a 6-DOF leg has more than one way
    # to reach a target, so neighbouring frames can land on different solutions and the
    # joint traces come out ragged even though every foot is where it should be -- peak
    # jerk tripled before this. Smoothing the leg joints costs a fraction of a mm of
    # foot height and takes it back out.
    # Smoothing drags the opening back towards the skate it was just pulled out of, and
    # re-asserting the placement puts some of the raggedness back, so alternate the two
    # until they agree. They converge because the ramp's targets are themselves smooth:
    # doing it once each way leaves either 11 mm/frame of skate or double the jerk.
    if JOINT_SIGMA > 0:
        leg_idx = [jn.index(nm) for lg in legs.values() for nm in lg.names]
        dof[:, leg_idx] = gaussian_filter1d(
            dof[:, leg_idx], JOINT_SIGMA, axis=0, mode="nearest"
        )
        log(f"        re-ramp after smoothing: {ramp_opening()*1000:.1f} mm residual")

    low1 = np.array([min(feet(f)[b][2] for b in FEET) for f in range(n)])
    t1 = travel()
    log(f"after:  lowest sole {low1.min()*1000:+.1f} mm, median {np.median(low1)*1000:.1f} mm,"
        f" {int((low1 < FLOOR - 0.002).sum())}/{n} frames in the floor")
    log(f"        opening foot travel {t1[:RAMP].max():.1f} mm/frame worst")
    log(f"        clip lifted by {total.mean()*1000:.1f} mm on average,"
        f" {total.max()*1000:.1f} mm at most")

    # 4. Get the feet out of the box. The retarget stands the robot with its toes inside
    #    the box footprint -- the forward sole sphere is 24 mm in at frame 0 and the
    #    overlap runs through the whole approach and the whole set-down, the two phases
    #    where the box is on the floor. It is geometrically impossible (the box is ON
    #    that floor, a foot cannot be under it), so the simulator resolves it the only
    #    way it can, by firing the box away: 145 mm in the first 0.2 s of the rollout.
    #    Everything after that is the robot chasing a box that is no longer where the
    #    reference says. It is in the raw mocap too, and the passes above deepened it,
    #    since dropping a foot onto the floor also drives its toe further under the box.
    def box_push():
        """Per-foot horizontal escape from the box footprint, world frame."""
        out = np.zeros((n, len(FEET), 2))
        for f in range(n):
            if box_pos[f, 2] > BOX_DOWN:
                continue  # carried: the box is overhead, feet underneath are fine
            # Full 3x3, transposed. Taking the top-left 2x2 block instead is only the
            # same thing for a box that is level, and this one rests tilted about 9 deg,
            # which was enough to make the pass believe it had cleared frames it had not.
            Rb3 = quat_wxyz_to_mat(box_quat[f])
            Rb = Rb3[:2, :2]
            cur = feet(f)
            for i, b in enumerate(FEET):
                pts = np.array([cur[b][0] + cur[b][1] @ s for s in SOLE_PTS[b]])
                loc = (pts - box_pos[f]) @ Rb3
                # Inflate the footprint by the sphere's radius: a centre sitting just
                # OUTSIDE the box still has the sphere lapping over the face, and testing
                # centres alone leaves exactly those frames touching.
                lo, hi = BOX_MIN[:2] - SPHERE_R, BOX_MAX[:2] + SPHERE_R
                inside = (
                    (loc[:, 0] > lo[0]) & (loc[:, 0] < hi[0])
                    & (loc[:, 1] > lo[1]) & (loc[:, 1] < hi[1])
                    & (loc[:, 2] > BOX_MIN[2] - SPHERE_R) & (loc[:, 2] < BOX_MAX[2] + SPHERE_R)
                )
                if not inside.any():
                    continue
                # Cheapest single direction clearing EVERY sphere of this foot at once.
                best = None
                for axis, sign, face in (
                    (0, -1.0, lo[0]), (0, 1.0, hi[0]),
                    (1, -1.0, lo[1]), (1, 1.0, hi[1]),
                ):
                    dist = float(((face - loc[inside, axis]) * sign).max()) + FOOT_MARGIN
                    if dist > 0 and (best is None or dist < best[0]):
                        v = np.zeros(2)
                        v[axis] = sign * dist
                        best = (dist, v)
                if best is not None:
                    out[f, i] = (Rb3 @ np.r_[best[1], 0.0])[:2]
        return out

    moved = stuck = 0
    biggest = 0.0
    # Iterate: moving a foot out along its cheapest axis can leave another sphere of the
    # same foot, or the other foot, still clipping a corner.
    for _ in range(8):
        push = box_push()
        if np.abs(push).max() < 1e-4:
            break
        # Dilate BEFORE smoothing. Smoothing a spike halves it, which is exactly the
        # frames that needed it most -- the first attempt at this left 146 frames still
        # inside. Widening first means the filter can only ever round the shoulders off
        # a plateau that is already tall enough.
        push = gaussian_filter1d(
            maximum_filter1d(np.abs(push), 13, axis=0, mode="nearest") * np.sign(
                maximum_filter1d(push, 13, axis=0, mode="nearest")
                + minimum_filter1d(push, 13, axis=0, mode="nearest")
            ),
            3.0, axis=0, mode="nearest",
        )
        biggest = max(biggest, float(np.abs(push).max()))
        for f in range(n):
            if np.abs(push[f]).max() < 1e-4:
                continue
            cur = feet(f)
            base_f = dof[f].copy()
            for frac in (1.0, 0.7, 0.4):
                tgt = {}
                for i, b in enumerate(FEET):
                    p = cur[b][0].copy()
                    p[:2] += frac * push[f, i]
                    tgt[b] = (p, cur[b][1])
                dof[f] = leg_ik(
                    legs, base_f.copy(), jn, root_pos[f],
                    quat_wxyz_to_mat(root_quat[f]), tgt, lim,
                )
                if max(np.linalg.norm(feet(f)[b][0] - tgt[b][0]) for b in FEET) <= 0.003:
                    moved += 1
                    break
            else:
                dof[f] = base_f
                stuck += 1
        # Smooth inside the loop, not after it. Solving each frame's escape on its own
        # leaves the joint traces ragged (jerk 5756 against the clip's own 4096), and a
        # filter bolted on at the end just puts the feet back in the box. Smoothing here
        # lets the next iteration re-asssert the clearance on top of it, so the two
        # converge instead of fighting.
        dof[:, leg_idx] = gaussian_filter1d(dof[:, leg_idx], 1.2, axis=0, mode="nearest")
    log(f"        cleared feet out of the box: {moved} frame-moves up to"
        f" {biggest*1000:.0f} mm; {stuck} the legs could not reach")

    # 5. Drop leading frames until the clip STARTS clear of the box. Everything else
    #    can be solved by walking a foot out of the way, but frame 0 cannot: the right
    #    knee is against its 0 deg stop there, so the leg has no length to give, and
    #    pushing harder only makes the IK swing it. The clip is clear a couple of frames
    #    later anyway, and spawning the robot already touching the box is what fires it
    #    across the floor -- 145 mm inside 0.2 s in the rollout that prompted this.
    def touching(f):
        if box_pos[f, 2] > BOX_DOWN:
            return 0.0
        Rb3 = quat_wxyz_to_mat(box_quat[f])
        cur = feet(f)
        worst = 0.0
        for b in FEET:
            for s in SOLE_PTS[b]:
                l = Rb3.T @ (cur[b][0] + cur[b][1] @ s - box_pos[f])
                o = np.maximum(np.maximum(BOX_MIN - l, l - BOX_MAX), 0.0)
                sd = np.linalg.norm(o) if o.any() else -min(
                    np.min(l - BOX_MIN), np.min(BOX_MAX - l)
                )
                worst = min(worst, sd - SPHERE_R)
        return worst

    cut = 0
    # Only genuine penetration. A sphere grazing the face at half a millimetre is
    # harmless, and trimming those too ate half a second and left the clip starting
    # half-squatted at 557 mm and already dropping at 270 mm/s -- no way to begin a
    # run on hardware.
    while cut < 25 and touching(cut) < -0.0005:
        cut += 1
    if cut:
        log(f"        trimmed {cut} opening frame(s) ({cut/FPS:.2f} s): the clip started"
            f" {touching(0)*1000:.0f} mm inside the box and is clear from there")
        root_pos, root_quat, dof = root_pos[cut:], root_quat[cut:], dof[cut:]
        box_pos, box_quat = box_pos[cut:], box_quat[cut:]
        n = len(dof)

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
