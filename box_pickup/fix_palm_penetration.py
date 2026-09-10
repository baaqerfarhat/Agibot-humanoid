"""Seat the palms on the box instead of inside it.

The carry phase of the clip was authored around `hand_contact_link`, a point frame at
wrist + (0.02, 0, -0.13), on the assumption of a roughly 15 mm palm. The geometry the
physics engine actually collides is a different thing: a 50 mm sphere at
wrist + (0.01, 0, -0.10), from holosoma's halfspherehand asset. The two are 32 mm
apart and the radii differ by 35 mm, so a grip that looks like a firm contact on the
authored frame is a hand buried two thirds of a palm inside the box.

That is not cosmetic. The box is a physical body in both simulators, and a reference
that puts the hands inside it asks the policy to do two incompatible things: track the
reference, or keep the box. Isaac's own rollout has the box driven up to 1.53 m off
the reference, which is the "box randomly goes up and is never picked up properly"
report this work started from.

So the palms move out along the box surface normal until the collision sphere rests on
the face with a small deliberate overlap, and the arms are re-solved to put them there.
The box is not touched: it keeps its authored position, orientation and path. Only the
hands move, and only during the carry.

    .venv/bin/python fix_palm_penetration.py [out.npz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rebuild_reference_motion import FIXED_FRAMES, URDF, LegChain, Robot, ik_reach  # noqa: E402
from urdf_fk import quat_wxyz_to_mat  # noqa: E402

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
DEFAULT_OUT = CLIP.with_name("sub3_largebox_003_walk_grip.npz")

# The collision sphere holosoma welds to each wrist, and the box's own bounding box.
PALM_OFFSET = np.array([0.01, 0.0, -0.10])
PALM_RADIUS = 0.05
BOX_MIN = np.array([-0.2341, -0.2301, -0.1982])
BOX_MAX = np.array([0.2371, 0.2286, 0.2097])
BOX_HALF = (BOX_MAX - BOX_MIN) / 2
BOX_CENTRE = (BOX_MAX + BOX_MIN) / 2

GRIP = 0.005  # m of deliberate overlap left in: a squeeze, not a gap
BLEND = 20  # frames to ease the correction in and out at the carry boundaries
SIGMA = 2.0  # smoothing on the correction, in frames
LIFT = 0.02  # m above its rest height before the box counts as carried


def log(m):
    print(m, flush=True)


def box_sdf(local: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed distance from a point to the box, and the outward normal there.

    Negative inside. The normal inside a box is the axis of least penetration, which
    is what pushes a buried palm out through the nearest face rather than diagonally
    through an edge.
    """
    p = local - BOX_CENTRE
    q = np.abs(p) - BOX_HALF
    if (q > 0).any():  # outside
        qc = np.maximum(q, 0.0)
        d = float(np.linalg.norm(qc))
        n = np.sign(p) * qc
        nn = np.linalg.norm(n)
        return d, (n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0]))
    i = int(np.argmax(q))  # inside: least-penetrating axis
    n = np.zeros(3)
    n[i] = np.sign(p[i]) or 1.0
    return float(q[i]), n


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    d = dict(np.load(CLIP, allow_pickle=True))
    jn = [str(x) for x in d["joint_names"]]
    q = d["joint_pos"].copy()
    root_pos, root_quat, dof = q[:, 0:3].copy(), q[:, 3:7].copy(), q[:, 7:].copy()
    box_pos = np.asarray(d["object_pos_w"], float)
    box_quat = np.asarray(d["object_quat_w"], float)
    n = len(dof)

    robot = Robot()
    # Chains whose tip is the collision sphere, not the authored contact frame.
    arms = {s: LegChain(robot.chain, f"{s}_wrist_roll_link", tuple(PALM_OFFSET))
            for s in ("left", "right")}
    # Wrists stay where they are: they carry the ankle/wrist action caps and the grip
    # does not need them. Shoulders and elbow do the reaching.
    free = {s: [nm for nm in arms[s].names if nm.startswith(s) and "wrist" not in nm]
            for s in ("left", "right")}
    lim = robot.chain.limits

    def sphere(f, src=None):
        R0 = quat_wxyz_to_mat(root_quat[f])
        s_ = dof[f] if src is None else src
        return {
            side: arms[side].fk(
                {nm: s_[jn.index(nm)] for nm in arms[side].names}, root_pos[f], R0
            )[0]
            for side in ("left", "right")
        }

    carry = box_pos[:, 2] > box_pos[0, 2] + LIFT
    idx = np.nonzero(carry)[0]
    if not len(idx):
        raise SystemExit("no carry phase in the clip")
    g0, g1 = int(idx[0]), int(idx[-1])
    log(f"carry t={g0/50:.2f}..{g1/50:.2f} s ({g1-g0+1} frames)")

    # ---- measure ----------------------------------------------------------------
    before = {s: np.zeros(n) for s in ("left", "right")}
    for f in range(n):
        Rb = quat_wxyz_to_mat(box_quat[f])
        p = sphere(f)
        for s in ("left", "right"):
            sd, _ = box_sdf(Rb.T @ (p[s] - box_pos[f]))
            before[s][f] = sd - PALM_RADIUS

    for s in ("left", "right"):
        b = before[s][carry]
        log(f"  {s} palm gap before: min {b.min()*1000:+.0f} mm  "
            f"mean {b.mean()*1000:+.0f} mm  max {b.max()*1000:+.0f} mm")

    # ---- pick the face each hand squeezes, once ---------------------------------
    # Pushing out along the box's own least-penetrating axis looks principled and is
    # not: a palm buried near a corner gets sent out through whichever face happens to
    # be marginally closer, which flips between frames and can point somewhere the arm
    # cannot follow. A carried box is held by two opposing side faces, so each hand is
    # assigned one horizontal face for the whole carry and stays on it.
    FACES = [(0, +1.0), (0, -1.0), (1, +1.0), (1, -1.0)]
    face: dict[str, tuple[int, float]] = {}
    Rb0 = quat_wxyz_to_mat(box_quat[g0])
    p0 = sphere(g0)
    for s in ("left", "right"):
        loc = Rb0.T @ (p0[s] - box_pos[g0]) - BOX_CENTRE
        ax, sg = max(FACES, key=lambda f_: f_[1] * loc[f_[0]] / BOX_HALF[f_[0]])
        face[s] = (ax, sg)
        log(f"  {s} hand assigned the {'xy'[ax]}{'+' if sg > 0 else '-'} face")
    if face["left"] == face["right"]:
        raise SystemExit("both hands picked the same face; the grip is not a squeeze")

    def target_local(s, loc):
        """Palm on its face, tangential coords kept but held inside the face."""
        ax, sg = face[s]
        t = loc.copy()
        t[ax] = BOX_CENTRE[ax] + sg * (BOX_HALF[ax] + PALM_RADIUS - GRIP)
        for k in range(3):
            if k != ax:
                lim_ = BOX_HALF[k] * 0.8  # stay off the edges
                t[k] = np.clip(t[k], BOX_CENTRE[k] - lim_, BOX_CENTRE[k] + lim_)
        return t

    targets: dict[int, dict[str, np.ndarray]] = {}
    for f in np.nonzero(carry)[0]:
        Rb = quat_wxyz_to_mat(box_quat[f])
        p = sphere(f)
        for s in ("left", "right"):
            loc = Rb.T @ (p[s] - box_pos[f])
            targets.setdefault(int(f), {})[s] = box_pos[f] + Rb @ target_local(s, loc)

    # ---- ease the correction in and out ----------------------------------------
    # Stepping the arm across on the first carried frame would put a spike into the
    # joint trace far worse than the penetration it fixes.
    ramp = np.zeros(n)
    ramp[g0 : g1 + 1] = 1.0
    for i in range(BLEND):
        w = (i + 1) / (BLEND + 1)
        if g0 - BLEND + i >= 0:
            ramp[g0 - BLEND + i] = w
        if g1 + BLEND - i < n:
            ramp[g1 + BLEND - i] = w
    ramp = gaussian_filter1d(ramp, SIGMA, mode="nearest")

    # ---- solve ------------------------------------------------------------------
    newdof = dof.copy()
    worst, backed = 0.0, 0
    RESID = 0.02  # m: past this the arm is not tracking, it is diverging
    for f in sorted(targets):
        if ramp[f] <= 1e-3:
            continue
        R0 = quat_wxyz_to_mat(root_quat[f])
        cur = newdof[f].copy()
        for s, tgt in targets[f].items():
            p0 = sphere(f, cur)[s]
            # Back off toward the pose the arm already had rather than let the solver
            # run away. A hand that only comes half way out is still better than the
            # 550 mm excursion an unguarded solve produces near a corner.
            for frac in (1.0, 0.8, 0.6, 0.45, 0.3, 0.15, 0.0):
                if frac == 0.0:
                    break
                aim = p0 + ramp[f] * frac * (tgt - p0)
                qd = {nm: cur[jn.index(nm)] for nm in arms[s].names}
                rest = {nm: dof[f][jn.index(nm)] for nm in arms[s].names}
                sol = ik_reach(arms[s], qd, free[s], aim, lim, rest, root_pos[f], R0)
                trial = cur.copy()
                for nm, v in sol.items():
                    trial[jn.index(nm)] = v
                got = arms[s].fk({nm: trial[jn.index(nm)] for nm in arms[s].names},
                                 root_pos[f], R0)[0]
                res = float(np.linalg.norm(got - aim))
                if res < RESID:
                    cur = trial
                    worst = max(worst, res)
                    if frac < 1.0:
                        backed += 1
                    break
        newdof[f] = cur
    log(f"  arm IK worst residual {worst*1000:.1f} mm, {backed} frame-hands backed off")

    # Take the high-frequency edge off the joints the pass touched, and only those.
    touched = sorted({nm for s in ("left", "right") for nm in free[s]})
    ti = [jn.index(nm) for nm in touched]
    newdof[:, ti] = gaussian_filter1d(newdof[:, ti], 1.2, axis=0, mode="nearest")

    # ---- verify -----------------------------------------------------------------
    dof[:] = newdof
    after = {s: np.zeros(n) for s in ("left", "right")}
    for f in range(n):
        Rb = quat_wxyz_to_mat(box_quat[f])
        p = sphere(f)
        for s in ("left", "right"):
            sd, _ = box_sdf(Rb.T @ (p[s] - box_pos[f]))
            after[s][f] = sd - PALM_RADIUS
    for s in ("left", "right"):
        a = after[s][carry]
        log(f"  {s} palm gap after:  min {a.min()*1000:+.0f} mm  "
            f"mean {a.mean()*1000:+.0f} mm  max {a.max()*1000:+.0f} mm")

    # Joint limits and smoothness: a fix that trades penetration for a jerk spike or a
    # joint parked on its stop is not a fix.
    viol = 0
    for i, nm in enumerate(jn):
        lo, hi = lim.get(nm, (-np.inf, np.inf))
        viol += int(((newdof[:, i] < lo - 1e-6) | (newdof[:, i] > hi + 1e-6)).sum())
    def jerk(a):
        return np.sqrt((np.diff(a, 3, axis=0) ** 2).mean()) * (50.0**3)
    log(f"  joint-limit violations: {viol}")
    log(f"  arm jerk before {jerk(q[:, 7:][:, ti]):.0f} -> after {jerk(newdof[:, ti]):.0f} rad/s^3")
    log(f"  max joint change: {np.degrees(np.abs(newdof - q[:, 7:]).max()):.1f} deg")

    # ---- write ------------------------------------------------------------------
    d["joint_pos"] = np.concatenate([root_pos, root_quat, newdof], axis=1).astype(np.float32)
    jv = d["joint_vel"].copy()
    jv[:, 6:] = np.gradient(newdof, 1.0 / 50.0, axis=0)
    d["joint_vel"] = jv.astype(np.float32)

    # body_pos_w / body_quat_w have to be rebuilt or the tracking targets still point
    # at the old arms.
    bn = [str(x) for x in d["body_names"]]
    bp = np.asarray(d["body_pos_w"], float).copy()
    bq = np.asarray(d["body_quat_w"], float).copy()
    from urdf_fk import mat_to_quat_wxyz

    for f in range(n):
        o = robot.fk(newdof[f], jn, root_pos[f], root_quat[f])
        pos, rot = robot.frames(o, bn)
        for i, b in enumerate(bn):
            if b in pos:
                bp[f, i] = pos[b]
                bq[f, i] = mat_to_quat_wxyz(rot[b])
    d["body_pos_w"] = bp.astype(np.float32)
    d["body_quat_w"] = bq.astype(np.float32)
    d["body_lin_vel_w"] = np.gradient(bp, 1.0 / 50.0, axis=0).astype(np.float32)

    np.savez(out_path, **d)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
