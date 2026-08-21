"""Make the full OmniRetarget box clip -- walk included -- feasible on the X2.

This is a repair pass, not a re-authoring.  diagnose_clip.py was run on the raw
retarget first, and only the things it flagged as beyond the hardware are touched:

  1. palms reach 6.9 cm, so the grasp is effectively off the floor  -> lift the grip
     onto the side faces with arm IK
  2. the loaded foot slides (peak 2.0 m/s) and sits on an edge (up to 25 deg of sole
     tilt, ankle roll pinned at its +/-15 deg limit) -> pin and level each foot for
     the span it is carrying load, leaving the swing between steps untouched
  3. the pelvis asks for 0.639 m of leg against a 0.618 m leg -> let it ride lower
  4. wrist pitch runs at 174 % of its velocity limit, joint jerk peaks at 5011 rad/s^3
     -> bounded smoothing, heavier on the wrists
  5. planar CoM acceleration peaks at 9.8 m/s^2 and throws the ZMP far outside the
     feet -> stretch the clock only where that happens
  6. the box is welded 13 cm above the palms -> hang it off the palms instead

Deliberately NOT touched: where the box sits on the floor and which way it faces,
the walk, the step timing pattern, and the overall pose sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.path.insert(0, str(Path(__file__).parent))
from rebuild_reference_motion import (
    FIXED_FRAMES,
    LegChain,
    Robot,
    ik_reach,
    leg_ik,
    pull_inside,
    support_margin,
)
from urdf_fk import mat_to_quat_wxyz, quat_wxyz_to_mat

WS = Path("/home/baaqer/baaqer_ws")
MOTIONS = WS / "holosoma/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking"
SRC = MOTIONS / "sub3_largebox_003_walk.npz"  # full retarget, walk included
DST = MOTIONS / "sub3_largebox_003_walk_feasible.npz"

FPS = 50
GRASP_Z = 0.22  # m. Palms never drop below this, so the hands work the box, not the floor.
CONTACT_Z = 0.010  # m above the floor at which a sole may be carrying load
CONTACT_V = 0.20  # m/s. The swings only clear 2-6 cm, so height alone cannot tell
# stance from swing -- a foot crossing 1 cm at 1.5 m/s is mid-step, not planted.
BLEND_S = 0.12  # s trimmed off each end of a stance before taking its hold pose
PIN_SIGMA = 8.0  # frames. The foot correction is smoothed, never ramped, so that no
# stance edge leaves a corner in the CoM path for the ZMP check to trip over.
SOLE_DZ = 0.068  # ankle_roll_link sits this far above its contact spheres
SMOOTH_SIGMA = 2.5  # frames. Light: the retarget pose sequence is the thing worth keeping.
WRIST_SIGMA = 7.0  # frames. The wrists are the only joints over their velocity limit.
ARM_SIGMA = 4.0
BAL_MARGIN = 0.025  # m the CoM is kept inside the feet while both are down
BAL_MARGIN_SS = -0.030  # m allowed outside during a step: a walking CoM is meant to
# travel towards the swing foot, and forcing it over the stance foot every instant
# both destroys the gait and injects the acceleration it was supposed to remove
STAND_H = 0.640  # m from the soles to the pelvis in the final stance. A straight leg
# is 0.670, so this leaves a gentle knee bend and keeps the IK off that singularity.
SETTLE_S = 1.2  # s to rise out of the set-down into that stance
HOLD_S = 0.4  # s held upright, so the episode ends settled rather than mid-move
BOX_SIGMA = 6.0  # frames; a carried box is low-passed by its own inertia, it does
# not copy every tremor in the hands
ZMP_MARGIN = 0.010  # m the ZMP is kept inside the feet while both are down
ZMP_MARGIN_SS = -0.060  # m allowed outside mid-step
BAL_SIGMA = 12.0  # frames; a weight transfer is a lean, not a twitch
ACCEL_MAX = 7.0  # m/s^2 planar CoM. A side-step legitimately needs this; the raw clip
# sits at 9.8, and squeezing it much below 7 would slow the walk out of recognition.
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
SPHERES = [f"{s}_ankle_roll_sphere_{i}_link" for s in ("left", "right") for i in range(1, 6)]


def log(msg):
    print(msg, flush=True)


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def level(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def intervals(mask, min_len=5, close=6):
    """Contiguous True runs in a boolean mask, as [start, stop) pairs.

    Gaps up to `close` frames are filled first: a single frame dipping out of the
    contact test would otherwise split one stance into two pins at different hold
    positions, and the foot gets yanked between them.
    """
    mask = mask.copy()
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if 0 < i and j < n and j - i <= close:
                mask[i:j] = True
            i = j
        else:
            i += 1
    out, i = [], 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def resample(src_idx, dof, root_pos, root_quat, extra):
    """Put every channel on a new (fractional) source-index clock."""
    idx = np.arange(len(dof))
    dof_n = CubicSpline(idx, dof, axis=0)(src_idx)
    rp_n = CubicSpline(idx, root_pos, axis=0)(src_idx)
    key = Rot.from_quat(root_quat[:, [1, 2, 3, 0]])
    rq_n = Slerp(idx, key)(np.clip(src_idx, 0, idx[-1])).as_quat()[:, [3, 0, 1, 2]]
    ex_n = {k: CubicSpline(idx, v, axis=0)(src_idx) for k, v in extra.items()}
    return dof_n, rp_n, rq_n, ex_n


def main():
    d = np.load(SRC, allow_pickle=True)
    jn = [str(x) for x in d["joint_names"]]
    bn = [str(x) for x in d["body_names"]]
    qp = np.asarray(d["joint_pos"], float)
    root_pos, root_quat, dof = qp[:, 0:3].copy(), qp[:, 3:7].copy(), qp[:, 7:].copy()
    box_pos = np.asarray(d["object_pos_w"], float).copy()
    box_quat = np.asarray(d["object_quat_w"], float).copy()
    # when the retarget has the box off the floor is when the hands are on it
    carry_m = (box_pos[:, 2] > box_pos[:, 2].min() + 0.03).astype(float)
    n = len(dof)
    robot = Robot()
    legs = {b: LegChain(robot.chain, b) for b in FEET}
    arms = {
        s: LegChain(robot.chain, FIXED_FRAMES[f"{s}_hand_contact_link"][0],
                    FIXED_FRAMES[f"{s}_hand_contact_link"][1])
        for s in ("left", "right")
    }
    # Shoulder and elbow only.  Freeing the wrists as well gives the position-only
    # solver four redundant degrees of freedom and it spends them driving the wrists
    # into their stops -- the retarget already sits on the +/-32 deg pitch stop for
    # most of the clip, and the IK made it worse.  The hand keeps its retargeted pose.
    armfree = {
        s: [nm for nm in arms[s].names if nm.startswith(s) and "wrist" not in nm]
        for s in ("left", "right")
    }
    wrist_idx = [i for i, nm in enumerate(jn) if "wrist" in nm]
    arm_idx = [i for i, nm in enumerate(jn) if any(k in nm for k in ("shoulder", "elbow", "wrist"))]
    dt = 1.0 / FPS

    log(f"source: {SRC.name}  {n} frames, {n/FPS:.2f}s  (walk kept)")

    # ------------------------------------------- 0. finish standing up after the drop
    # The retarget stops 22 deg short of upright, still rising out of the set-down, so
    # the episode would end with the robot folded over.  Ease to a neutral stance on
    # the feet it is already standing on.  Done here, before everything else, so the
    # foot pinning, the balance pass and the ZMP retiming all cover the new frames too.
    log("\n[0] finish the clip standing upright")
    R_end = quat_wxyz_to_mat(root_quat[-1])
    qL = {nm: dof[-1, jn.index(nm)] for leg in legs.values() for nm in leg.names}
    foot_end = {b: legs[b].fk(qL, root_pos[-1], R_end) for b in FEET}
    floor_end = min(
        (p + Rm @ np.asarray(FIXED_FRAMES[s][1]))[2]
        for b, (p, Rm) in foot_end.items()
        for s in SPHERES if s.startswith(b.split("_")[0])
    )
    stance_end = {
        b: (np.array([p[0], p[1], floor_end + SOLE_DZ]), level(yaw_of(Rm)))
        for b, (p, Rm) in foot_end.items()
    }
    yaw_end = yaw_of(R_end)
    tgt_rp = np.array([
        np.mean([stance_end[b][0][0] for b in FEET]),
        np.mean([stance_end[b][0][1] for b in FEET]),
        floor_end + STAND_H,
    ])
    tgt_rq = mat_to_quat_wxyz(level(yaw_end))
    tgt_dof = leg_ik(  # neutral upper body, legs solved to stay on those two feet
        legs, np.zeros(len(jn)), jn, tgt_rp, level(yaw_end), stance_end, robot.lim
    )
    ease = int(SETTLE_S * FPS)
    hold = int(HOLD_S * FPS)
    w = 0.5 * (1 - np.cos(np.pi * np.arange(1, ease + 1) / ease))  # C1 at both ends
    key = Rot.from_quat(np.stack([root_quat[-1], tgt_rq])[:, [1, 2, 3, 0]])
    add_q = Slerp([0.0, 1.0], key)(w).as_quat()[:, [3, 0, 1, 2]]
    add_dof = dof[-1] + w[:, None] * (tgt_dof - dof[-1])
    add_rp = root_pos[-1] + w[:, None] * (tgt_rp - root_pos[-1])
    dof = np.concatenate([dof, add_dof, np.repeat(tgt_dof[None], hold, 0)])
    root_pos = np.concatenate([root_pos, add_rp, np.repeat(tgt_rp[None], hold, 0)])
    root_quat = np.concatenate([root_quat, add_q, np.repeat(tgt_rq[None], hold, 0)])
    pad = ease + hold
    box_pos = np.concatenate([box_pos, np.repeat(box_pos[-1][None], pad, 0)])
    box_quat = np.concatenate([box_quat, np.repeat(box_quat[-1][None], pad, 0)])
    carry_m = np.concatenate([carry_m, np.zeros(pad)])
    n = len(dof)
    tp = np.degrees(np.arccos(np.clip(
        robot.fk(dof[-1], jn, root_pos[-1], root_quat[-1])["torso_link"][1][2, 2], -1, 1)))
    log(f"    appended {SETTLE_S:.1f}s rise + {HOLD_S:.1f}s hold on the final stance:"
        f" trunk pitch ends at {tp:.1f} deg, pelvis at {root_pos[-1,2]:.3f} m")

    def stage(tag, dofs=None, rp=None, rq=None):
        """Planar CoM acceleration after a stage, so any kink is attributed correctly."""
        dofs = dof if dofs is None else dofs
        rp = root_pos if rp is None else rp
        rq = root_quat if rq is None else rq
        c = np.array([robot.com(robot.fk(dofs[f], jn, rp[f], rq[f]))[0] for f in range(len(dofs))])
        a = np.linalg.norm(np.gradient(np.gradient(c, dt, axis=0), dt, axis=0)[:, :2], axis=1)
        log(f"      CoM accel after {tag}: peak {a.max():5.1f} m/s^2 at t={a.argmax()/FPS:.2f}s,"
            f" 99th {np.percentile(a,99):5.1f}, median {np.median(a):.1f}")

    stage("nothing (raw)")

    def feet_now(dofs, rp, rq):
        """World pose of both ankles, and the lowest contact sphere under each."""
        P, R, LOW = {}, {}, {}
        for b in FEET:
            P[b] = np.empty((len(dofs), 3))
            R[b] = np.empty((len(dofs), 3, 3))
            LOW[b] = np.empty(len(dofs))
        for f in range(len(dofs)):
            R0 = quat_wxyz_to_mat(rq[f])
            q = {nm: dofs[f, jn.index(nm)] for leg in legs.values() for nm in leg.names}
            for b in FEET:
                p, Rm = legs[b].fk(q, rp[f], R0)
                P[b][f], R[b][f] = p, Rm
                LOW[b][f] = min(
                    (p + Rm @ np.asarray(FIXED_FRAMES[s][1]))[2]
                    for s in SPHERES
                    if s.startswith(b.split("_")[0])
                )
        return P, R, LOW

    def palms(dofs, rp, rq):
        out = {}
        for s in ("left", "right"):
            out[s] = np.array([
                arms[s].fk(
                    {nm: dofs[f, jn.index(nm)] for nm in arms[s].names},
                    rp[f], quat_wxyz_to_mat(rq[f]),
                )[0]
                for f in range(len(dofs))
            ])
        return out["left"], out["right"]

    # -------------------------------------------------- 1. bounded-jerk smoothing
    log("\n[1] smooth (light everywhere, heavier on the wrists)")
    j0 = np.abs(np.diff(dof, 3, axis=0)).max() / dt**3
    dof = gaussian_filter1d(dof, SMOOTH_SIGMA, axis=0, mode="nearest")
    dof[:, arm_idx] = gaussian_filter1d(dof[:, arm_idx], ARM_SIGMA, axis=0, mode="nearest")
    dof[:, wrist_idx] = gaussian_filter1d(dof[:, wrist_idx], WRIST_SIGMA, axis=0, mode="nearest")
    root_pos = gaussian_filter1d(root_pos, SMOOTH_SIGMA, axis=0, mode="nearest")
    log(f"    peak joint jerk {j0:.0f} -> {np.abs(np.diff(dof,3,axis=0)).max()/dt**3:.0f} rad/s^3")
    stage("smoothing")

    # ------------------------------------- 2. pin + level each foot while it is loaded
    log("\n[2] pin and level every stance phase (swing left alone)")
    P, R, LOW = feet_now(dof, root_pos, root_quat)
    floor = min(LOW[b].min() for b in FEET)
    tgt_p, tgt_R = {}, {}
    blend = max(1, int(BLEND_S * FPS))
    for b in FEET:
        spd = np.r_[0.0, np.linalg.norm(np.diff(P[b][:, :2], axis=0), axis=1)] * FPS
        runs = intervals((LOW[b] < floor + CONTACT_Z) & (spd < CONTACT_V))
        log(f"    {b.split('_')[0]:5s} {len(runs)} stance phases: "
            + ", ".join(f"{a/FPS:.2f}-{c/FPS:.2f}s" for a, c in runs))
        # Build the correction as a field over the whole clip and then smooth it.
        # Fading in and out of each stance with a ramp leaves a corner at every
        # stance edge, and a corner in position is an acceleration spike once it is
        # differentiated twice -- that alone took the CoM from 5 to 18 m/s^2.
        dp = np.zeros((n, 3))
        dr = np.zeros((n, 3))
        for a, c in runs:
            pad = min(blend // 2, (c - a - 1) // 2)  # short stances have no room to trim
            core = slice(a + pad, c - pad)
            hold_xy = np.median(P[b][core, :2], axis=0)
            hold_yaw = np.median(np.unwrap([yaw_of(R[b][f]) for f in range(a, c)]))
            hp = np.array([hold_xy[0], hold_xy[1], floor + SOLE_DZ])
            dp[a:c] = hp - P[b][a:c]
            dr[a:c] = Rot.from_matrix(
                level(hold_yaw) @ R[b][a:c].transpose(0, 2, 1)
            ).as_rotvec()
        log(f"          pin moves the foot by up to {np.abs(dp).max()*1000:4.0f} mm and"
            f" levels up to {np.degrees(np.linalg.norm(dr,axis=1)).max():4.1f} deg")
        dp = gaussian_filter1d(dp, PIN_SIGMA, axis=0, mode="nearest")
        dr = gaussian_filter1d(dr, PIN_SIGMA, axis=0, mode="nearest")
        tgt_p[b] = P[b] + dp
        tgt_R[b] = Rot.from_rotvec(dr).as_matrix() @ R[b]

    # -------------------------- 3+4. leg IK, letting the pelvis go where the legs can
    def solve_legs(dofs, rp, passes=4, tag=""):
        """Put the feet on their targets, letting the pelvis go where the legs reach.

        The retarget buys effective leg length by standing on the edge of the foot.
        Once the sole is flat that length is gone and the pelvis has to give it back.
        Rather than guess a reach limit, drive the pelvis by the IK's own shortfall:
        if the foot lands short by e, moving the pelvis by e makes the target reachable.
        """
        dofs, rp = dofs.copy(), rp.copy()
        for it in range(passes):
            err = np.zeros((len(dofs), 3))
            for f in range(len(dofs)):
                R0 = quat_wxyz_to_mat(root_quat[f])
                dofs[f] = leg_ik(
                    legs, dofs[f], jn, rp[f], R0,
                    {b: (tgt_p[b][f], tgt_R[b][f]) for b in FEET}, robot.lim,
                )
                q = {nm: dofs[f, jn.index(nm)] for leg in legs.values() for nm in leg.names}
                e = [tgt_p[b][f] - legs[b].fk(q, rp[f], R0)[0] for b in FEET]
                err[f] = e[int(np.argmax([np.linalg.norm(x) for x in e]))]
            worst = np.linalg.norm(err, axis=1).max()
            if worst < 0.005:
                break
            rp = rp + gaussian_filter1d(err, 5.0, axis=0, mode="nearest")
        if tag:
            log(f"    {tag}: worst foot shortfall {worst*1000:.1f} mm,"
                f" {(np.linalg.norm(err,axis=1)>0.005).sum()} frames over 5 mm")
        return dofs, rp

    log("\n[3] leg IK, with the pelvis following wherever the legs can actually reach")
    pel0 = root_pos.copy()
    dof, root_pos = solve_legs(dof, root_pos, passes=5, tag="converged")
    shift = np.linalg.norm(root_pos - pel0, axis=1)
    log(f"    pelvis moved by {shift.mean()*1000:.0f} mm on average,"
        f" {shift.max()*1000:.0f} mm at worst (t={shift.argmax()/FPS:.2f}s)")
    stage("foot pinning")

    # ------------------------------------------------------ 5. lift the grasp height
    log("\n[5] raise the grip off the floor with the arms")
    LP, RP = palms(dof, root_pos, root_quat)
    lo0 = ((LP[:, 2] + RP[:, 2]) / 2).min()
    lift = gaussian_filter1d(
        np.maximum(GRASP_Z - (LP[:, 2] + RP[:, 2]) / 2, 0.0), 5.0, mode="nearest"
    )
    for f in range(n):
        if lift[f] < 1e-4:
            continue
        R0 = quat_wxyz_to_mat(root_quat[f])
        q = {nm: dof[f, i] for i, nm in enumerate(jn)}
        up = np.array([0.0, 0.0, lift[f]])
        for s, Pm in (("left", LP), ("right", RP)):
            q = ik_reach(arms[s], q, armfree[s], Pm[f] + up, robot.lim, q, root_pos[f], R0)
        for i, nm in enumerate(jn):
            dof[f, i] = q[nm]
    dof[:, arm_idx] = gaussian_filter1d(dof[:, arm_idx], ARM_SIGMA, axis=0, mode="nearest")
    LP, RP = palms(dof, root_pos, root_quat)
    log(f"    lowest palm {lo0*100:.1f} -> {((LP[:,2]+RP[:,2])/2).min()*100:.1f} cm"
        f"   (lifted by up to {lift.max()*1000:.0f} mm)")

    stage("palm raise")

    # ------------------------------------- 6. shift the weight so the CoM stays over the feet
    log("\n[6] shift the pelvis so the CoM stays over the feet")
    # This is the "cannot get up out of the squat" failure.  The retarget stands up
    # and starts the side-step at the same instant without shifting its weight first,
    # so the CoM leaves the support polygon.  No amount of slowing the clip fixes
    # that -- as the speed drops the ZMP converges on the CoM, and the CoM is what is
    # outside.  The weight transfer has to be put back in.
    def com_and_contacts(dofs, rp):
        c = np.empty((len(dofs), 3))
        ct = np.empty((len(dofs), len(SPHERES), 3))
        for f in range(len(dofs)):
            out = robot.fk(dofs[f], jn, rp[f], root_quat[f])
            c[f] = robot.com(out)[0]
            pos, _ = robot.frames(out, SPHERES)
            ct[f] = [pos[s] for s in SPHERES]
        return c, ct

    # both soles carrying => a static weight shift is meaningful; one sole => a step
    _, ct0 = com_and_contacts(dof, root_pos)
    dsup = np.array([
        float(sum(ct0[f, k * 5:(k + 1) * 5, 2].min() < ct0[f, :, 2].min() + 0.03
                  for k in (0, 1)) == 2)
        for f in range(n)
    ])
    log(f"    double support on {int(dsup.sum())}/{n} frames,"
        f" stepping on {n-int(dsup.sum())}")
    bal0 = root_pos.copy()
    for it in range(6):
        com, ct = com_and_contacts(dof, root_pos)
        marg = np.array([support_margin(ct[f], com[f, :2])[0] for f in range(n)])
        want = np.where(dsup > 0.5, BAL_MARGIN, BAL_MARGIN_SS)
        bad = marg < want
        log(f"    pass {it}: CoM outside the feet on {(marg<0).sum():3d} frames,"
            f" worst margin {marg.min()*1000:+5.0f} mm, short of target on {bad.sum():3d}")
        if not bad.any():
            break
        corr = np.zeros((n, 2))
        for f in np.where(bad)[0]:
            corr[f] = pull_inside(ct[f], com[f, :2], want[f]) - com[f, :2]
        corr = gaussian_filter1d(corr, BAL_SIGMA, axis=0, mode="nearest")
        corr = np.clip(corr, -0.05, 0.05)  # never yank; let it converge over passes
        root_pos[:, :2] += corr
        dof, root_pos = solve_legs(dof, root_pos, passes=3)
    shift = np.linalg.norm(root_pos[:, :2] - bal0[:, :2], axis=1)
    log(f"    pelvis shifted laterally by {shift.mean()*1000:.0f} mm on average,"
        f" {shift.max()*1000:.0f} mm at worst")
    for i, nm in enumerate(jn):
        lo, hi = robot.lim.get(nm, (-np.inf, np.inf))
        dof[:, i] = np.clip(dof[:, i], lo, hi)
    stage("balance")

    # ----------------------------- 7. stretch the clock where the body cannot follow
    log("\n[7] stretch the clock where CoM accel / joint speed exceed the hardware")
    best, n0 = np.inf, n
    snap = None  # the best-scoring clip seen, restored on the way out: a warp that
    # makes things worse must not be the one we keep just because it was the last
    for it in range(6):
        com = np.array([
            robot.com(robot.fk(dof[f], jn, root_pos[f], root_quat[f]))[0] for f in range(len(dof))
        ])
        ca = np.linalg.norm(np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)[:, :2], axis=1)
        log(f"      accel peak {ca.max():.1f} at t={ca.argmax()/FPS:.2f}s, "
            f"99th pct {np.percentile(ca,99):.1f}, median {np.median(ca):.1f} m/s^2")
        # The binding constraint is not acceleration itself but where it puts the CoP:
        # at a CoM height of 0.7 m, even 6.7 m/s^2 drags the ZMP 47 cm off the CoM,
        # which is several times the length of a foot.  Target the ZMP directly.
        cv = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
        zmp = com[:, :2] - (com[:, 2:3] / 9.81) * cv[:, :2]
        ct = np.empty((len(dof), len(SPHERES), 3))
        for f in range(len(dof)):
            pos, _ = robot.frames(robot.fk(dof[f], jn, root_pos[f], root_quat[f]), SPHERES)
            ct[f] = [pos[s] for s in SPHERES]
        zm = np.array([support_margin(ct[f], zmp[f])[0] for f in range(len(dof))])
        zoff = np.linalg.norm(zmp - com[:, :2], axis=1)
        want = np.where(dsup > 0.5, ZMP_MARGIN, ZMP_MARGIN_SS)
        # halving the speed quarters the acceleration, so the speed scale is the sqrt
        desired = np.maximum(zoff - np.maximum(want - zm, 0.0), 0.02)
        over_zmp = np.sqrt(zoff / desired)
        log(f"      ZMP outside on {(zm<0).sum()} frames, worst {zm.min()*1000:+.0f} mm")
        dv = np.abs(np.gradient(dof, dt, axis=0))
        vr = np.max([dv[:, i] / robot.vlim[nm] for i, nm in enumerate(jn) if nm in robot.vlim], axis=0)
        over = np.maximum(np.maximum(np.sqrt(np.maximum(ca / ACCEL_MAX, 1e-9)), vr), over_zmp)
        log(f"    pass {it}: peak CoM accel {ca.max():5.2f} m/s^2, peak joint speed "
            f"{vr.max()*100:3.0f}% of limit, {len(dof)/FPS:.2f}s")
        score = float(over.max())
        if score < best - 0.02:
            best = score
            snap = (dof.copy(), root_pos.copy(), root_quat.copy(),
                    box_pos.copy(), box_quat.copy(), carry_m.copy(), dsup.copy())
        else:  # a posture problem will not yield to more time
            log("    no further gain from stretching; the residual is spatial, not timing")
            break
        if score <= 1.02:
            break
        if len(dof) > 1.25 * n0:
            log("    stopping: any more and the walk stops looking like the original")
            break
        speed = gaussian_filter1d(1.0 / np.maximum(over, 1.0), 6.0, mode="nearest")
        s = np.concatenate([[0.0], np.cumsum(1.0 / (FPS * speed[:-1]))])
        src = np.interp(np.arange(int(s[-1] * FPS) + 1) / FPS, s, np.arange(len(speed)))
        dof, root_pos, root_quat, ex = resample(
            src, dof, root_pos, root_quat,
            {"bp": box_pos, "bq": box_quat,
             "cm": carry_m[:, None], "ds": dsup[:, None]},
        )
        box_pos, box_quat = ex["bp"], ex["bq"]
        carry_m, dsup = ex["cm"][:, 0], ex["ds"][:, 0]
        box_quat /= np.linalg.norm(box_quat, axis=1, keepdims=True)
        for b in FEET:
            tgt_p[b] = CubicSpline(np.arange(n), tgt_p[b], axis=0)(src)
            tgt_R[b] = Slerp(np.arange(n), Rot.from_matrix(tgt_R[b]))(
                np.clip(src, 0, n - 1)
            ).as_matrix()
        n = len(dof)
        for f in range(n):  # the resample drifts the pinned feet by a millimetre or two
            dof[f] = leg_ik(
                legs, dof[f], jn, root_pos[f], quat_wxyz_to_mat(root_quat[f]),
                {b: (tgt_p[b][f], tgt_R[b][f]) for b in FEET}, robot.lim,
            )

    if snap is not None and len(snap[0]) != n:
        dof, root_pos, root_quat, box_pos, box_quat, carry_m, dsup = snap
        n = len(dof)
        log(f"    rolled back to the best pass ({n/FPS:.2f}s)")
    log(f"    clip length {n0/FPS:.2f}s -> {n/FPS:.2f}s"
        f"  ({100*(n/n0-1):+.0f}% to stay inside the feet)")

    # ------------------------------------------------- 8. hang the box off the palms
    log("\n[8] carry the box with the hands (pickup spot left exactly as retargeted)")
    LP, RP = palms(dof, root_pos, root_quat)
    carry = np.where(carry_m > 0.5)[0]
    g, r = int(carry[0]), int(carry[-1])
    rest = box_pos[0].copy()  # exactly where the retarget put it, position AND pose
    rest_rot = Rot.from_quat(box_quat[0][[1, 2, 3, 0]])

    # A real box does not follow every tremor in the hands -- its own inertia low
    # passes it.  Welding it straight onto the raw palm path made it the single
    # jerkiest body in the clip, so drive it from a smoothed hand path instead.
    mid = gaussian_filter1d((LP + RP) / 2, BOX_SIGMA, axis=0, mode="nearest")
    pyaw = gaussian_filter1d(
        np.unwrap(np.arctan2(*(RP - LP)[:, [1, 0]].T)), BOX_SIGMA, mode="nearest"
    )

    def rotz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    # One constant offset in the hand frame, fixed so that at the instant of the
    # grasp the box is exactly on its retargeted spot.  After that it can only move
    # where the hands move it, which is what stops it drifting or teleporting.
    # The orientation is carried the same way: the retargeted resting pose turned by
    # however far the hands have turned since the grasp, so frame 0 is untouched.
    off = rotz(-pyaw[g]) @ (rest - mid[g])
    track = np.array([mid[f] + rotz(pyaw[f]) @ off for f in range(n)])
    trot = Rot.from_rotvec(np.outer(pyaw - pyaw[g], [0.0, 0.0, 1.0])) * rest_rot

    # Two cosine ramps: the box takes up the hands' motion at the grasp, and hands it
    # back to the floor at the release.  Cosine so its velocity is zero at both ends
    # and it never inherits a step.
    ramp = min(max(1, int(0.30 * FPS)), (r - g) // 2)
    k = np.arange(n)
    s_up = np.clip((k - g) / ramp, 0.0, 1.0)
    s_dn = np.clip((k - (r - ramp)) / ramp, 0.0, 1.0)
    s_up, s_dn = 0.5 * (1 - np.cos(np.pi * s_up)), 0.5 * (1 - np.cos(np.pi * s_dn))
    box_pos = rest + s_up[:, None] * (track - rest)  # rest -> riding with the hands
    box_pos = box_pos + s_dn[:, None] * (track[r] - box_pos)  # -> parked where left
    # slerp the same two ramps in orientation, so frame 0 is the retargeted pose exactly
    rel = (rest_rot.inv() * trot).as_rotvec()
    rel = s_up[:, None] * rel
    rel = rel + s_dn[:, None] * ((rest_rot.inv() * trot[r]).as_rotvec() - rel)
    box_quat = (rest_rot * Rot.from_rotvec(rel)).as_quat()[:, [3, 0, 1, 2]]

    drop = np.linalg.norm(track[r][:2] - np.asarray(d["object_pos_w"])[-1, :2])
    log(f"    carried from t={g/FPS:.2f}s to t={r/FPS:.2f}s, rigid to the hands throughout")
    log(f"    pickup spot ({rest[0]:+.3f}, {rest[1]:+.3f}, {rest[2]:.3f}) -- exactly as"
        f" retargeted, and its resting orientation is carried through untouched")
    log(f"    set-down lands {drop*100:.1f} cm from where the retarget put it,"
        f" at z {track[r][2]:.3f} m")
    log(f"    carried height {box_pos[:,2].max():.3f} m (retarget had"
        f" {np.asarray(d['object_pos_w'])[:,2].max():.3f} m with the box floating"
        f" above the hands)")

    # --------------------------------------------------------------- 9. write it out
    log("\n[9] replay through MuJoCo and write the training clip")
    # Reuse the same writer the other clips went through, so the body frames and the
    # velocity conventions are identical to what the trainer already reads.
    from cut_walking import to_training_npz

    q = np.concatenate([root_pos, root_quat, dof, box_pos, box_quat], axis=1)
    to_training_npz(q, FPS, DST)
    log(f"\nwrote {DST}  ({n} frames, {n/FPS:.2f}s)")


if __name__ == "__main__":
    main()
