"""Move the grasp from across the middle of the box to its near corners.

The refined walk clip is sound everywhere except in where it puts the hands. The
OmniRetarget grip lands at box-local y = +0.06 -- past the centre of a 460 mm deep
box -- so the palms end up about 290 mm beyond the robot's own toes. The arm spans
540 mm, so all of that budget goes on reaching forward and none is left to reach
down, the torso has to fold to make up the difference, and folding is the one thing
the waist cannot afford: it puts waist_pitch at 94-102% of its 48 N-m limit, and the
only posture under the limit is a 60% squat the policy cannot track. It dies on
bad_tracking at 2.5 s of a 10.2 s clip, long before it ever reaches the box.

Taking the near corners instead -- which is how a person picks up a box this size --
shortens the payload's lever on the waist without touching anything else. This pass
deliberately re-solves the ARMS ONLY. Legs, pelvis, feet, balance and timing are
left exactly as the refine pass left them, because those are already correct and
every attempt to regenerate them alongside the grip broke the stance detection.

The box itself is not moved. Its aim point is corrected, though: the retargeted rest
pose has it tilted 9.4 deg with its lowest corner 69 mm under the floor, while the
simulator spawns it and lets it settle flat, so aiming by the retargeted pose misses
by that much.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_static_torque import G, effort_limits, subtrees
from rebuild_reference_motion import FIXED_FRAMES, URDF, LegChain, Robot, ik_reach
from urdf_fk import quat_wxyz_to_mat

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
BOX_MIN = (-0.234, -0.230, -0.198)  # the box collision mesh's own bounding box, metres
BOX_MAX = (0.237, 0.229, 0.210)
GRIP_X = 0.25  # m out from the box centre to each palm, across the faces it squeezes.
# The face is at 0.235 and the palm sphere is ~15 mm, so this is a contact rather than
# a hand buried in the box, and it is the offset hands_to_object_distance_exp expects.
GRIP_Y = 0.18  # m back from the box centre toward the robot: the near corners. The
# near face is at 0.230, so the palms sit just inside the edge closest to the robot.
GRIP_Z = 0.28  # m. Palm height at the grasp, a little under mid-height of the box side.
BLEND = 26  # frames over which the new grip eases in, so the arm is not stepped across
PAYLOAD = 1.0  # kg, the real box


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
    eff = effort_limits(URDF)
    sub = subtrees(robot.chain)
    jinfo = {j.name: j for j in robot.chain.joints}
    wp_lim = eff["waist_pitch_joint"]
    arms = {
        s: LegChain(
            robot.chain,
            FIXED_FRAMES[f"{s}_hand_contact_link"][0],
            FIXED_FRAMES[f"{s}_hand_contact_link"][1],
        )
        for s in ("left", "right")
    }
    # Wrists stay put: they are what the ankle_roll action cap is protecting, and the
    # grip does not need them to move.
    free = {
        s: [nm for nm in arms[s].names if nm.startswith(s) and "wrist" not in nm]
        for s in ("left", "right")
    }

    def palms(f, dofs=None):
        R0 = quat_wxyz_to_mat(root_quat[f])
        src = dof[f] if dofs is None else dofs
        return {
            s: arms[s].fk({nm: src[jn.index(nm)] for nm in arms[s].names}, root_pos[f], R0)[0]
            for s in ("left", "right")
        }

    def waist_frac(f):
        o = robot.fk(dof[f], jn, root_pos[f], root_quat[f])
        j = jinfo["waist_pitch_joint"]
        jp, jR = o[j.child]
        tau = np.zeros(3)
        for link in sub["waist_pitch_joint"]:
            if link in o and link in robot.mass:
                m, c = robot.mass[link]
                p, Rm = o[link]
                tau += np.cross(p + Rm @ c - jp, m * G)
        if carry[f]:
            tau += np.cross(box_pos[f] - jp, PAYLOAD * G)
        return abs(float((jR @ j.axis) @ tau)) / wp_lim

    # The frames where the box is off the floor are the ones the hands own.
    lifted = box_pos[:, 2] > box_pos[0, 2] + 0.02
    carry = lifted.copy()
    idx = np.nonzero(carry)[0]
    if not len(idx):
        raise SystemExit("no carry phase found in the clip")
    g0, g1 = int(idx[0]), int(idx[-1])
    log(f"carry runs t={g0/50:.2f}s to t={g1/50:.2f}s ({g1-g0+1} frames)")

    # Where the box actually is, once it settles flat. Level it by standing its own
    # vertical axis upright rather than reading a yaw off the quaternion: it rests
    # flipped about 174 deg, so euler yaw is 180 deg from the true face orientation.
    Rb = quat_wxyz_to_mat(box_quat[0])
    v = Rb[:, 2]
    up = np.array([0.0, 0.0, np.sign(v[2]) or 1.0])
    ax = np.cross(v, up)
    s = float(np.linalg.norm(ax))
    Rrest = Rb if s < 1e-9 else Rot.from_rotvec(ax / s * np.arctan2(s, float(v @ up))).as_matrix() @ Rb
    corners = np.array(np.meshgrid(*zip(BOX_MIN, BOX_MAX))).reshape(3, -1).T
    seat = box_pos[0].copy()
    seat[2] = -(corners @ Rrest.T)[:, 2].min()
    log(f"box settles {(seat[2]-box_pos[0,2])*1000:+.0f} mm off its retargeted rest height")

    o = robot.fk(dof[g0], jn, root_pos[g0], root_quat[g0])
    mid = np.mean([o[f"{s}_ankle_roll_link"][0] for s in ("left", "right")], axis=0)
    near = -np.sign((Rrest.T @ (seat - mid))[1])
    # Put each hand on its own face at the point nearest its OWN shoulder, rather than
    # both on a fixed pair of corners. The robot meets the box 24 deg off square, so a
    # fixed corner pair sits 200 mm from one shoulder and 450 mm from the other, and
    # the far arm simply cannot get there -- it gives up 215 mm short and rides up over
    # the top of the box, which is the very thing this pass exists to stop.
    sx = {"left": +1.0, "right": -1.0}
    if np.linalg.norm(palms(g0)["left"] - (seat + Rrest @ np.array([GRIP_X, 0, 0]))) > \
       np.linalg.norm(palms(g0)["left"] - (seat + Rrest @ np.array([-GRIP_X, 0, 0]))):
        sx = {"left": -1.0, "right": +1.0}  # do not cross the arms over
    # Slide each hand along its own face to the nearest point that arm can actually
    # reach. The robot meets the box about 24 deg off square, so the near corner on one
    # side ends up across the robot's chest, and that shoulder cannot roll inwards (its
    # roll range stops at 3 deg) -- it gives up 88 mm short with the elbow locked
    # straight and the hand rides up over the box. Better an asymmetric grip that both
    # arms can hold than a symmetric one that only one of them reaches.
    R0g = quat_wxyz_to_mat(root_quat[g0])
    off = {}
    for s in ("left", "right"):
        # Keep the lateral placement the retarget found -- the shoulders are only
        # 290 mm apart, so splaying the palms to the 470 mm faces while reaching
        # forward and down is outside the arm's workspace entirely. What costs the
        # waist is the FORWARD reach, and that is what this pass shortens.
        lx = float((Rrest.T @ (palms(g0)[s] - seat))[0])
        for y in near * np.linspace(GRIP_Y, -GRIP_Y, 19):
            cand = Rrest @ np.array([lx, y, 0.0])
            tgt = seat + cand
            tgt[2] = GRIP_Z
            sol = ik_reach(
                arms[s], {nm: dof[g0, jn.index(nm)] for nm in arms[s].names}, free[s], tgt,
                {nm: robot.lim[nm] for nm in free[s]},
                {nm: dof[g0, jn.index(nm)] for nm in free[s]}, root_pos[g0], R0g,
            )
            res = float(np.linalg.norm(arms[s].fk(sol, root_pos[g0], R0g)[0] - tgt))
            if res < 0.015:
                off[s] = cand
                log(f"  {s:5s} grips box-local y {y:+.3f} (near edge {near*0.230:+.3f},"
                    f" centre 0.000), residual {res*1000:.0f} mm")
                break
        else:
            raise SystemExit(f"{s} arm cannot reach anywhere along its face")

    # One rigid transform for the whole grasp -- both hands AND the box. Shifting each
    # hand by its own vector is not a grasp: it stretches the pair, so the box has no
    # single pose consistent with both palms, and re-deriving one frame by frame threw
    # 148 mm jumps and 9.8 g into a box trajectory that had been smooth.
    grab = palms(g0)
    tgt0 = {}
    for s in ("left", "right"):
        tgt0[s] = seat + off[s]
        tgt0[s][2] = GRIP_Z
    a = (grab["right"] - grab["left"])[:2]
    b = (tgt0["right"] - tgt0["left"])[:2]
    theta = float(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    c0 = 0.5 * (grab["left"] + grab["right"])
    cn = 0.5 * (tgt0["left"] + tgt0["right"])
    # Leave the grip height exactly where the refine pass put it. Dropping it drags
    # the box down with it for the whole carry and sets it down 36 mm into the floor,
    # and height was never the problem -- the forward reach was.
    cn[2] = c0[2]

    def rigid(p, weight, pivot):
        """The grasp transform, eased in by `weight`.

        The turn is taken about the hands' own centre at that instant, not about
        where they were at the grasp: pivoting the whole carry around a point down
        by the floor swings the hands a third of a metre sideways at the top of the
        lift, well past anything the arm can reach.
        """
        th = weight * theta
        c, s_ = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return pivot + Rz @ (p - pivot) + weight * (cn - c0)

    log(f"grasp turns {np.degrees(theta):+.1f} deg and moves"
        f" {np.linalg.norm(cn - c0)*1000:.0f} mm; palms end up"
        f" {np.linalg.norm(tgt0['right']-tgt0['left'])*1000:.0f} mm apart"
        f" (were {np.linalg.norm(grab['right']-grab['left'])*1000:.0f})")

    # Ease the displacement in before the grasp and back out after the set-down so the
    # arm is never stepped across, and so the clip's own reach-out and stand-away are
    # left alone.
    w = np.zeros(n)
    w[g0:g1 + 1] = 1.0
    w[max(g0 - BLEND, 0):g0] = np.linspace(0, 1, min(BLEND, g0), endpoint=False)
    tail = min(BLEND, n - 1 - g1)
    if tail:
        w[g1 + 1:g1 + 1 + tail] = np.linspace(1, 0, tail, endpoint=False)
    w = gaussian_filter1d(w, 3.0, mode="nearest")

    before_hand = {s: np.array([palms(f)[s] for f in range(n)]) for s in ("left", "right")}
    before_waist = np.array([waist_frac(f) for f in range(n)])
    pivot = 0.5 * (before_hand["left"] + before_hand["right"])

    def solve(f, weight, commit):
        """Both arms to the transformed grip. Returns the worst residual."""
        R0 = quat_wxyz_to_mat(root_quat[f])
        res = 0.0
        for s in ("left", "right"):
            tgt = rigid(before_hand[s][f], weight, pivot[f])
            sol = ik_reach(
                arms[s], {nm: dof[f, jn.index(nm)] for nm in arms[s].names}, free[s], tgt,
                {nm: robot.lim[nm] for nm in free[s]},
                {nm: dof0[f, jn.index(nm)] for nm in free[s]}, root_pos[f], R0,
            )
            res = max(res, float(np.linalg.norm(arms[s].fk(sol, root_pos[f], R0)[0] - tgt)))
            if commit:
                for nm, val in sol.items():
                    dof[f, jn.index(nm)] = val
        return res

    # Back the transform off wherever the arm cannot follow it, and back the BOX off by
    # the same amount at that frame. Letting the box take the full turn while the hands
    # fall 240 mm short is what stops it being a grasp at all.
    dof0 = dof.copy()
    keep = w.copy()
    for f in range(n):
        if w[f] < 1e-3:
            continue
        for cand in w[f] * np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
            if solve(f, cand, commit=False) < 0.015:
                keep[f] = cand
                break
        else:
            keep[f] = 0.0
    # A weight that jumps frame to frame is itself a source of jitter.
    keep = gaussian_filter1d(np.minimum(keep, w), 4.0, mode="nearest")
    w = keep
    worst = max((solve(f, w[f], commit=True) for f in range(n) if w[f] >= 1e-3), default=0.0)
    log(f"arm IK: worst residual {worst*1000:.1f} mm;"
        f" transform held at {w[g0:g1+1].mean()*100:.0f}% of full across the carry")

    # Smooth only the joints we touched, and only where we touched them.
    touched = sorted({nm for s in free for nm in free[s]})
    cols = [jn.index(nm) for nm in touched]
    span = slice(max(g0 - BLEND - 4, 0), min(g1 + BLEND + 5, n))
    for c in cols:
        dof[span, c] = gaussian_filter1d(dof[span, c], 1.5, mode="nearest")

    # The box goes through exactly the same transform, so the grasp stays rigid and the
    # box keeps the smooth trajectory the refine pass gave it.
    for f in range(n):
        if w[f] < 1e-6:
            continue
        box_pos[f] = rigid(box_pos[f], w[f], pivot[f])
        box_quat[f] = (
            Rot.from_rotvec([0.0, 0.0, w[f] * theta]) * Rot.from_quat(box_quat[f][[1, 2, 3, 0]])
        ).as_quat()[[3, 0, 1, 2]]

    after_waist = np.array([waist_frac(f) for f in range(n)])
    hold = slice(max(g0 - 10, 0), min(g1 + 11, n))
    log("")
    log(f"waist_pitch over the lift: peak {before_waist[hold].max()*100:.0f}%"
        f" -> {after_waist[hold].max()*100:.0f}% of 48 N-m at {PAYLOAD} kg")
    log(f"waist_pitch over the whole clip: peak {before_waist.max()*100:.0f}%"
        f" -> {after_waist.max()*100:.0f}%")
    log(f"frames over 90%: {(before_waist>0.90).sum()} -> {(after_waist>0.90).sum()}")

    after_hand = {s: np.array([palms(f, dof[f])[s] for f in range(n)]) for s in ("left", "right")}
    for s in ("left", "right"):
        loc = Rrest.T @ (after_hand[s][g0] - seat)
        log(f"{s:5s} palm at the grasp: box-local {loc.round(3)}  world z {after_hand[s][g0][2]:.3f}"
            f"  (face at {'%.3f' % BOX_MAX[0]}, near edge at {'%.3f' % BOX_MIN[1]})")
    for s in ("left", "right"):
        j0 = np.abs(np.diff(before_hand[s], 3, axis=0)).max() * 50**3
        j1 = np.abs(np.diff(after_hand[s], 3, axis=0)).max() * 50**3
        log(f"peak {s:5s} hand jerk {j0:.0f} -> {j1:.0f} m/s^3")

    # Same writer the rest of the clips go through, so the body frames and the velocity
    # conventions stay identical to what the trainer already reads.
    from cut_walking import to_training_npz

    to_training_npz(
        np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1), 50.0, CLIP
    )
    log(f"\nwrote {CLIP}  ({n} frames)")


if __name__ == "__main__":
    main()
