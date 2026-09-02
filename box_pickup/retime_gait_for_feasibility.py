"""Retime the swings so single support is short AND lands where the body already is.

The walk asks for up to 127 N-m of ankle_roll restoring moment against a foot that
can transmit 21, which is why both ankles are pinned at their limit for most of a
rollout and why the robot has to be held up. The moment is m*g times the lateral
distance from the CoM to the stance ankle, so there are only two ways down: move the
CoM, or change which frames are single support.

Moving the CoM does not work. With the sole flat the +-15 deg ankle caps the pelvis
at ~161 mm to the side of the foot it stands on, and the CoM starts on the WRONG side
for 98 of the 199 single-support frames, so getting it across needs more travel than
the leg has. Two attempts at it either dragged the feet 112 mm or bought 11 of the 67
percentage points at the cost of 4.6 g of lateral acceleration.

So this changes the frames instead, and it is cheap because the sway is already there:
78 mm of it against the 99 mm wanted. It is simply not in time with the feet
(correlation +0.29). Each swing is therefore

  - shortened, which turns single-support frames into double-support ones, where the
    polygon spans both feet and the load can be shared instead of a single ankle
    carrying all of it; and
  - slid to the sub-window where the CoM is genuinely nearest that stance foot, which
    is what the existing sway is good for once nobody insists on the original timing.

The footfalls do not move. Each foot lands exactly where it landed, the swing between
is the same path resampled onto a shorter window, and the root is untouched -- so the
box, the grasp and the pickup are all exactly as they were.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_reference_motion import FIXED_FRAMES, URDF, LegChain, Robot, leg_ik, load_masses
from urdf_fk import quat_wxyz_to_mat

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
CONTACT = 0.020
FRACTION = 0.55     # how much of each swing window to keep
MIN_SWING = 14      # never shorter than 0.28 s -- the foot still has to clear
SLACK = 12          # frames the window may slide either way
GAP = 3             # frames of double support to keep between two swings
RESID = 0.002
BOX_DOWN = 0.30
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
    box_pos = np.asarray(d["object_pos_w"], float)
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

    def poses_at(f, src=None, root=None):
        src = dof[f] if src is None else src
        root = root_pos[f] if root is None else root
        return robot.chain.fk({nm: src[jn.index(nm)] for nm in jn}, root, root_quat[f])

    # ---- current geometry ----------------------------------------------------
    foot = [dict() for _ in range(n)]
    h0 = np.zeros((n, 2))
    com0 = np.zeros(n)
    for f in range(n):
        p = poses_at(f)
        acc, tot = np.zeros(3), 0.0
        for nm, (mk, c) in mass.items():
            if nm in p:
                pp, RR = p[nm]
                acc += mk * (pp + RR @ np.asarray(c))
                tot += mk
        m = tot + (BOX_M if box_pos[f, 2] > BOX_DOWN else 0.0)
        com0[f] = (acc[1] + (BOX_M * box_pos[f, 1] if box_pos[f, 2] > BOX_DOWN else 0.0)) / m
        for i, b in enumerate(FEET):
            pp, RR = p[b]
            foot[f][b] = (pp.copy(), RR.copy())
            h0[f, i] = min((pp + RR @ s)[2] for s in sole[b])
    mass_of = lambda f: m_robot + (BOX_M if box_pos[f, 2] > BOX_DOWN else 0.0)
    down0 = h0 < CONTACT

    # ---- find the swings -----------------------------------------------------
    swings = []
    for i, b in enumerate(FEET):
        sw = ~down0[:, i]
        st = None
        for f in range(n + 1):
            if f < n and sw[f] and st is None:
                st = f
            elif (f == n or not sw[f]) and st is not None:
                if f - st >= 6:
                    swings.append([st, f - 1, i, b])
                st = None
    swings.sort(key=lambda s: s[0])
    log(f"{len(swings)} swings, double support {100*(down0.sum(1)==2).mean():.0f}% of the clip")

    # ---- choose a shorter, better-placed window for each ----------------------
    plan = []
    for si, (a, b_, i, name) in enumerate(swings):
        dur = b_ - a + 1
        new = max(MIN_SWING, int(round(FRACTION * dur)))
        other = 1 - i
        lo = a - SLACK
        hi = b_ + SLACK - new + 1
        if plan:
            lo = max(lo, plan[-1][1] + 1 + GAP)
        if si + 1 < len(swings):
            hi = min(hi, swings[si + 1][0] + SLACK - new - GAP)
        lo, hi = max(0, lo), max(lo, min(n - new, hi))
        # Pick the placement whose frames have the CoM nearest the foot that will be
        # carrying: that is the whole point, and it is the sway that already exists
        # doing the work.
        best, bestcost = None, None
        for s in range(lo, hi + 1):
            ay = np.array([foot[f][FEET[other]][0][1] for f in range(s, s + new)])
            cost = np.abs(com0[s:s + new] - ay).mean()
            if bestcost is None or cost < bestcost:
                best, bestcost = s, cost
        old = np.abs(com0[a:b_ + 1]
                     - np.array([foot[f][FEET[other]][0][1] for f in range(a, b_ + 1)])).mean()
        plan.append((best, best + new - 1, i, name, a, b_))
        log(f"  {name.split('_')[0]:5s} f{a:3d}-{b_:3d} ({dur:2d}f) -> f{best:3d}-{best+new-1:3d}"
            f" ({new:2d}f);  mean |CoM-stance| {old*1000:5.1f} -> {bestcost*1000:5.1f} mm")

    # ---- rebuild the foot targets --------------------------------------------
    # Outside its window the foot is planted: at the pose it took off from before, at
    # the pose it lands on after. Inside, the original swing path resampled.
    tgt = [dict() for _ in range(n)]
    for f in range(n):
        for b in FEET:
            tgt[f][b] = foot[f][b]
    for (s, e, i, name, a, b_) in plan:
        take = foot[max(a - 1, 0)][name]
        land = foot[min(b_ + 1, n - 1)][name]
        src_p = np.array([foot[f][name][0] for f in range(a, b_ + 1)])
        src_R = Rot.from_matrix(np.array([foot[f][name][1] for f in range(a, b_ + 1)]))
        u = np.linspace(0.0, 1.0, len(src_p))
        sl = Slerp(u, src_R)
        for f in range(n):
            if f < s and f >= min(a, s):
                tgt[f][name] = take
            elif f > e and f <= max(b_, e):
                tgt[f][name] = land
        for j, f in enumerate(range(s, e + 1)):
            t = j / max(e - s, 1)
            k = t * (len(src_p) - 1)
            k0, frac = int(np.floor(k)), k - int(np.floor(k))
            k1 = min(k0 + 1, len(src_p) - 1)
            tgt[f][name] = ((1 - frac) * src_p[k0] + frac * src_p[k1], sl(t).as_matrix())

    # ---- solve ----------------------------------------------------------------
    # A frame the leg cannot reach is not skipped -- skipping it leaves the original
    # pose sitting between two retimed neighbours, and 83 such holes are what put
    # 37601 rad/s3 into the first attempt. Instead the target is walked back towards
    # where the foot already was until the leg can hold it, so the trajectory stays
    # continuous and only gives up depth where the step is too long to span.
    # Those long steps are the real limit here: 746 and 812 mm on a 600 mm leg cannot
    # both be planted at once, so double support simply cannot be extended across them.
    worst, partial = 0.0, 0
    newdof = dof.copy()
    achieved = np.ones(n)
    for f in range(n):
        if all(np.allclose(tgt[f][b][0], foot[f][b][0]) for b in FEET):
            continue
        for frac in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0):
            blend = {}
            for b in FEET:
                p0, R0 = foot[f][b]
                p1, R1 = tgt[f][b]
                sl = Slerp([0.0, 1.0], Rot.from_matrix(np.array([R0, R1])))
                blend[b] = ((1 - frac) * p0 + frac * p1, sl(frac).as_matrix())
            sol = leg_ik(legs, dof[f].copy(), jn, root_pos[f],
                         quat_wxyz_to_mat(root_quat[f]), blend, lim)
            p = poses_at(f, sol)
            res = max(np.linalg.norm(p[b][0] - blend[b][0]) for b in FEET)
            if res < RESID:
                achieved[f] = frac
                worst = max(worst, res)
                newdof[f] = sol
                if frac < 1.0:
                    partial += 1
                break
    leg_idx = [jn.index(nm) for lg in legs.values() for nm in lg.names]
    newdof[:, leg_idx] = gaussian_filter1d(newdof[:, leg_idx], 1.6, axis=0, mode="nearest")
    dof = newdof
    log(f"  {partial} frames took a partial target (mean depth"
        f" {achieved.mean()*100:.0f}% of what was asked)")
    failed = int((achieved < 0.05).sum())

    # ---- verify ----------------------------------------------------------------
    h1 = np.zeros((n, 2))
    for f in range(n):
        p = poses_at(f)
        for i, b in enumerate(FEET):
            pp, RR = p[b]
            h1[f, i] = min((pp + RR @ s)[2] for s in sole[b])
    down1 = h1 < CONTACT

    def moments(down, heights):
        out = []
        for f in range(n):
            k = np.nonzero(down[f])[0]
            if len(k) != 1:
                continue
            ay = poses_at(f)[FEET[k[0]]][0][1]
            out.append(mass_of(f) * G * abs(com0[f] - ay))
        return np.array(out) if out else np.array([0.0])

    q0 = np.asarray(np.load(CLIP, allow_pickle=True)["joint_pos"])[:, 7:]
    mb = moments(down0, h0)
    ma = moments(down1, h1)
    log(f"\nIK: worst residual {worst*1000:.2f} mm, {failed} frames left alone")
    log(f"double support {100*(down0.sum(1)==2).mean():.0f}% -> {100*(down1.sum(1)==2).mean():.0f}%"
        f"   single {int((down0.sum(1)==1).sum())} -> {int((down1.sum(1)==1).sum())} frames")
    log(f"ankle_roll moment in single support:")
    log(f"   before  mean {mb.mean():5.1f}  max {mb.max():6.1f} N-m, over {ROLL_LIMIT:.0f}"
        f" on {100*(mb>ROLL_LIMIT).mean():3.0f}%  ({len(mb)} frames)")
    log(f"   after   mean {ma.mean():5.1f}  max {ma.max():6.1f} N-m, over {ROLL_LIMIT:.0f}"
        f" on {100*(ma>ROLL_LIMIT).mean():3.0f}%  ({len(ma)} frames)")
    log(f"   frames over the limit: {int((mb>ROLL_LIMIT).sum())} -> {int((ma>ROLL_LIMIT).sum())}")
    for s in ("left", "right"):
        i = jn.index(f"{s}_ankle_roll_joint")
        log(f"   {s:5s} ankle_roll peak {np.abs(q0[:,i]).max()*R2D:4.1f} ->"
            f" {np.abs(dof[:,i]).max()*R2D:4.1f} deg, at the stop"
            f" {int((np.abs(q0[:,i])*R2D>14.5).sum()):3d} -> {int((np.abs(dof[:,i])*R2D>14.5).sum()):3d} frames")
    log(f"   soles {(h1.min()-0.011)*1000:+.1f} to {(h1.max()-0.011)*1000:+.1f} mm about the floor"
        f" (was {(h0.min()-0.011)*1000:+.1f} to {(h0.max()-0.011)*1000:+.1f})")
    log(f"   peak joint jerk {np.abs(np.diff(dof,3,axis=0)).max()*125000:.0f} rad/s3"
        f" (was {np.abs(np.diff(q0,3,axis=0)).max()*125000:.0f})")
    viol = sum(int(((dof[:, jn.index(j)] < robot.lim[j][0] - 1e-6)
                    | (dof[:, jn.index(j)] > robot.lim[j][1] + 1e-6)).sum())
               for j in jn if j in robot.lim)
    log(f"   joint limit violations {viol}")

    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos,
                        np.asarray(d["object_quat_w"], float)], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
