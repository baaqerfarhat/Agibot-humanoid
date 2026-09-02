"""Put the lateral weight shift back into the walk, so single support is survivable.

The clip's four single-support spells ask for up to 127 N-m of ankle_roll restoring
moment. A 50 mm half-width sole can transmit 21. That is why both ankles sit at their
effort limit for most of a rollout, why the policy has learned to keep both feet down
on 77-80% of the frames the reference wants single support, and why the robot has to
be held up on hardware.

The cause is not the stance, which is a reasonable 184-240 mm through the walk. It is
that the CoM does not go anywhere. Over all 199 single-support frames the stance ankle
sits 99 mm off the ankle midline and the CoM moves 7 mm towards it. The robot is asked
to stand on one leg while its weight stays between both.

Retargeting is where it went: a human shifts their pelvis over the stance foot on
every step, and that motion does not survive being mapped onto a body with different
segment masses and a much shorter leg.

Putting it back fixes the angle as well as the moment, which is the reason to expect
this to work rather than trade one problem for another. With the pelvis at the midline
and the foot 99 mm out, the stance leg is splayed 9.4 deg and the ankle has to roll
that far just to keep the sole flat -- which is where the +-15 deg stop contact comes
from. Move the pelvis over the foot and the same leg stands vertical, so ankle_roll
goes DOWN as the moment does.

The feet are not moved. Every foot keeps the world pose it already had, the legs are
re-solved underneath a pelvis that now travels, and the box travels with the hands
that hold it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_reference_motion import FIXED_FRAMES, URDF, LegChain, Robot, leg_ik, load_masses
from urdf_fk import quat_wxyz_to_mat

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
CONTACT = 0.020
PROTECT = 80        # the upright stand and its blend; do not touch it
RAMP = 20           # frames to ease the shift in past PROTECT
SIGMA = 6.0         # smoothing on the target, in frames
REACH = 0.85        # aim this far onto the stance foot, not the whole way
# Hard cap on the pelvis shift. The ankle only rolls +-15 deg, so with the sole held
# flat the pelvis cannot sit more than 0.6*tan(15) = 161 mm to the side of the ankle
# it is standing on. Asking for more does not bend the leg further, it drags the foot
# -- the first attempt at this pass wanted 244 mm and moved the feet 112 mm, put 82379
# rad/s3 of jerk in and pushed the soles 21 mm through the floor. Staying inside the
# cap gives up the last of the moment rather than any of that.
CAP = 0.115
SHIFT_SIGMA = 8.0   # smoothing on the shift itself; this is what bounds the jerk
RESID = 0.002       # a foot that moves more than this has not been held
BOX_DOWN = 0.30
SPHERE_R = 0.011
BOX_MIN = np.array([-0.234, -0.230, -0.198])
BOX_MAX = np.array([0.237, 0.229, 0.210])
ROLL_LIMIT = 24.0
G = 9.81
BOX_M = 1.0
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
    lim = {nm: robot.lim[nm] for lg in legs.values() for nm in lg.names}
    sole = {
        b: [np.asarray(FIXED_FRAMES[k][1]) for k in FIXED_FRAMES
            if "sphere" in k and FIXED_FRAMES[k][0] == b]
        for b in FEET
    }
    mass = load_masses(URDF)
    m_robot = sum(m for m, _ in mass.values())

    def state(f, src=None, root=None):
        """Foot poses, sole heights and the whole-body CoM at one frame."""
        src = dof[f] if src is None else src
        root = root_pos[f] if root is None else root
        poses = robot.chain.fk({nm: src[jn.index(nm)] for nm in jn}, root, root_quat[f])
        feet, h = {}, np.zeros(2)
        for i, b in enumerate(FEET):
            p, Rm = poses[b]
            feet[b] = (p, Rm)
            h[i] = min((p + Rm @ s)[2] for s in sole[b])
        acc, tot = np.zeros(3), 0.0
        for nm, (mk, c) in mass.items():
            if nm in poses:
                p, Rm = poses[nm]
                acc += mk * (p + Rm @ np.asarray(c))
                tot += mk
        return feet, h, acc, tot

    # ---- measure what is there now ------------------------------------------
    feet0, h0, com0 = [], np.zeros((n, 2)), np.zeros(n)
    for f in range(n):
        fe, h, acc, tot = state(f)
        feet0.append(fe)
        h0[f] = h
        m = tot + (BOX_M if box_pos[f, 2] > BOX_DOWN else 0.0)
        cy = acc[1] + (BOX_M * box_pos[f, 1] if box_pos[f, 2] > BOX_DOWN else 0.0)
        com0[f] = cy / m
    mass_of = lambda f: m_robot + (BOX_M if box_pos[f, 2] > BOX_DOWN else 0.0)

    down = h0 < CONTACT
    ank = np.array([[feet0[f][b][0][1] for b in FEET] for f in range(n)])

    # ---- where the CoM should be --------------------------------------------
    # The loaded feet define it: one foot down means over that foot, two means
    # between them. REACH backs off from dead centre because the last few mm cost
    # the most lean for the least moment, and the bound is 57 mm not 0.
    tgt = np.zeros(n)
    for f in range(n):
        k = np.nonzero(down[f])[0]
        mid = ank[f].mean()
        tgt[f] = mid if len(k) != 1 else mid + REACH * (ank[f, k[0]] - mid)
    # A target that steps when the support does is not walkable; the body has to be
    # moving there before the foot leaves the ground, which is what the smoothing
    # buys. It is also what makes the shift a sway rather than a lurch.
    tgt = gaussian_filter1d(tgt, SIGMA, mode="nearest")

    ease = np.ones(n)
    ease[:PROTECT] = 0.0
    ease[PROTECT:PROTECT + RAMP] = 0.5 * (
        1.0 - np.cos(np.pi * np.arange(RAMP) / RAMP)
    )
    want = com0 + ease * (tgt - com0)

    log(f"clip {n} frames; {down.sum(axis=1).tolist().count(1)} single-support frames")
    log(f"CoM now sits {np.abs(com0 - tgt).mean()*1000:.0f} mm from where the loaded feet"
        f" want it, worst {np.abs(com0 - tgt).max()*1000:.0f} mm")

    # ---- find the root shift that puts it there ------------------------------
    # The CoM follows the pelvis at less than 1:1 because the legs stay pinned, so
    # the gain is measured once and then iterated rather than assumed.
    shift = np.zeros(n)
    for it in range(4):
        err = np.zeros(n)
        for f in range(n):
            root = root_pos[f] + np.array([0.0, shift[f], 0.0])
            tg = {b: feet0[f][b] for b in FEET}
            sol = leg_ik(legs, dof[f].copy(), jn, root, quat_wxyz_to_mat(root_quat[f]),
                         tg, lim)
            _, _, acc, tot = state(f, sol, root)
            m = mass_of(f)
            cy = acc[1] + (BOX_M * (box_pos[f, 1] + shift[f]) if box_pos[f, 2] > BOX_DOWN else 0.0)
            err[f] = want[f] - cy / m
        if it == 0:
            gain = 0.85
        shift = shift + err / gain
        shift = gaussian_filter1d(shift, 2.0, mode="nearest")
        shift[:PROTECT] = 0.0
        log(f"  pass {it+1}: worst CoM error {np.abs(err).max()*1000:5.1f} mm,"
            f" shift now spans {shift.min()*1000:+.0f} .. {shift.max()*1000:+.0f} mm")

    # ---- cap and smooth BEFORE asking whether the leg can do it --------------
    # Order matters. Smoothing after the feasibility check is what broke the first
    # attempt: the filter raised frames back above the fraction they had just been
    # found unable to do. Cap first, smooth second, check last, and let the check
    # only ever reduce.
    shift = np.clip(shift, -CAP, CAP)
    shift = gaussian_filter1d(shift, SHIFT_SIGMA, mode="nearest")
    shift[:PROTECT] = 0.0
    log(f"  capped and smoothed: {shift.min()*1000:+.0f} .. {shift.max()*1000:+.0f} mm")

    # ---- apply it, backing off wherever the leg cannot follow ----------------
    keep = np.ones(n)
    for f in range(PROTECT, n):
        for frac in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            root = root_pos[f] + np.array([0.0, frac * shift[f], 0.0])
            tg = {b: feet0[f][b] for b in FEET}
            sol = leg_ik(legs, dof[f].copy(), jn, root, quat_wxyz_to_mat(root_quat[f]),
                         tg, lim)
            fe, h, _, _ = state(f, sol, root)
            res = max(np.linalg.norm(fe[b][0] - feet0[f][b][0]) for b in FEET)
            if res < RESID and np.abs(h - h0[f]).max() < 0.002:
                keep[f] = frac
                break
        else:
            keep[f] = 0.0
    # Only ever reduce: erode the fractions so a frame is never handed back more than
    # it was found able to do, then round the corners off the plateau that leaves.
    keep = gaussian_filter1d(minimum_filter1d(keep, 9, mode="nearest"), 3.0, mode="nearest")
    shift = shift * keep

    worst, reverted = 0.0, 0
    for f in range(n):
        if abs(shift[f]) < 1e-6:
            continue
        root = root_pos[f] + np.array([0.0, shift[f], 0.0])
        tg = {b: feet0[f][b] for b in FEET}
        sol = leg_ik(legs, dof[f].copy(), jn, root, quat_wxyz_to_mat(root_quat[f]), tg, lim)
        fe, h, _, _ = state(f, sol, root)
        res = max(np.linalg.norm(fe[b][0] - feet0[f][b][0]) for b in FEET)
        # Last line of defence: a frame that still cannot hold its feet keeps the pose
        # it already had. Better a shift that stops early than a foot that slides.
        if res > RESID or np.abs(h - h0[f]).max() > 0.002:
            shift[f] = 0.0
            reverted += 1
            continue
        worst = max(worst, res)
        root_pos[f] = root
        dof[f] = sol
        if box_pos[f, 2] > BOX_DOWN:  # welded to the hands, so it sways with them
            box_pos[f, 1] += shift[f]

    log(f"applied: shift {shift.min()*1000:+.0f} .. {shift.max()*1000:+.0f} mm,"
        f" worst foot residual {worst*1000:.2f} mm, {reverted} frames reverted")

    # ---- did it work? --------------------------------------------------------
    mom_before, mom_after, ar_before, ar_after = [], [], [], []
    for f in range(n):
        k = np.nonzero(down[f])[0]
        if len(k) != 1:
            continue
        m = mass_of(f)
        mom_before.append(m * G * abs(com0[f] - ank[f, k[0]]))
        _, _, acc, tot = state(f)
        cy = (acc[1] + (BOX_M * box_pos[f, 1] if box_pos[f, 2] > BOX_DOWN else 0.0)) / m
        mom_after.append(m * G * abs(cy - ank[f, k[0]]))
    mb, ma = np.array(mom_before), np.array(mom_after)
    log(f"\nankle_roll restoring moment in single support:")
    log(f"   before  mean {mb.mean():5.1f}  max {mb.max():6.1f} N-m,"
        f" over {ROLL_LIMIT:.0f} on {100*(mb>ROLL_LIMIT).mean():3.0f}% of frames")
    log(f"   after   mean {ma.mean():5.1f}  max {ma.max():6.1f} N-m,"
        f" over {ROLL_LIMIT:.0f} on {100*(ma>ROLL_LIMIT).mean():3.0f}% of frames")

    q0 = np.asarray(np.load(CLIP, allow_pickle=True)["joint_pos"])[:, 7:]
    for s in ("left", "right"):
        i = jn.index(f"{s}_ankle_roll_joint")
        b4 = np.abs(q0[:, i]) * R2D
        af = np.abs(dof[:, i]) * R2D
        log(f"   {s:5s} ankle_roll: peak {b4.max():4.1f} -> {af.max():4.1f} deg,"
            f" frames within 0.5 deg of the stop {int((b4>14.5).sum()):3d} -> {int((af>14.5).sum()):3d}")

    hi = np.array([max(state(f)[1]) for f in range(n)])
    lo = np.array([min(state(f)[1]) for f in range(n)])
    log(f"\n   feet still on the floor: soles {(lo.min()-0.011)*1000:+.1f} to"
        f" {(hi.max()-0.011)*1000:+.1f} mm about it")
    log(f"   peak joint jerk {np.abs(np.diff(dof, 3, axis=0)).max()*125000:.0f} rad/s3"
        f" (was {np.abs(np.diff(q0, 3, axis=0)).max()*125000:.0f})")
    log(f"   root lateral acceleration peak"
        f" {np.abs(np.diff(root_pos[:,1],2)*2500).max():.2f} m/s2")
    viol = 0
    for j in jn:
        if j in robot.lim:
            a, b = robot.lim[j]
            viol += int(((dof[:, jn.index(j)] < a - 1e-6) | (dof[:, jn.index(j)] > b + 1e-6)).sum())
    log(f"   joint limit violations {viol}")

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
