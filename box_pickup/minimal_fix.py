"""Make the OmniRetarget pickup usable in MJLab by changing as little as possible.

Measuring the untouched clip against the mjlab X2 shows it is almost already right:

    palms      67..98 mm INSIDE the box for the whole carry -- the hands reach the
               box easily, they just interpenetrate it, which a rigid-body sim
               resolves by firing the box across the room
    feet       8 mm into the box while it sits on the floor
    waist      45.1 Nm of a 48 Nm limit to hold the bow against gravity, but 240 Nm
               once the clip's own accelerations are included

Earlier attempts re-authored the bow into a squat to fix the waist. That traded away
the reach the bow was providing -- the arms ended up 0.4..1.0 m short of the box, and
a policy trained on it never grasped anything in 20k iterations. The pose is fine.
Only the overlaps and the timing are wrong.

So: push the palms out onto the box surface along the face they are already behind,
slide the whole robot back far enough to clear the feet, and leave every other joint
exactly as retargeted. Timing is handled separately by retime_feasible.py.

    cd ~/baaqer_ws/mjlab && uv run python <this>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

SRC = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
DST = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_minimal.npz"
)

BOX_SIZE = np.array([0.4712, 0.4587, 0.4079])
BOX_OFFSET = np.array([0.0015, -0.0007, 0.0058])
PALM_OFFSET = np.array([0.01, 0.0, -0.10])
PALM_RADIUS = 0.05
GRIP = 0.005          # let the palm sit this far inside the face, so contact is real
GRIP_RAMP = 45        # frames to bring the arms from their default onto the box
SQUAT_RAMP = 50       # frames to drop into the squat before the hug starts
# Exact centre of each side face: vertical midpoint, mid-depth.
GRIP_Z = 0.0
GRIP_DEPTH = 0.0
P_DEEP = 0.50
P_STAND = 0.67
PELVIS_FLOOR = 0.28
FOOT_GAP = 0.055
ARM_PAT = ("shoulder", "elbow", "wrist")
HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class Fixer:
    def __init__(self, clip):
        from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg

        self.m = get_x2_robot_cfg().spec_fn().compile()
        self.d = mujoco.MjData(self.m)
        cj = [str(x) for x in clip["joint_names"]]
        self.cj = cj
        self.cb = [str(x) for x in clip["body_names"]]
        mj = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(self.m.njnt)]
        self.hinge = [j for j in mj if j in cj]
        self.cj_idx = [cj.index(j) for j in self.hinge]
        self.qadr = np.array([self.m.jnt_qposadr[mj.index(j)] for j in self.hinge])
        self.vadr = np.array([self.m.jnt_dofadr[mj.index(j)] for j in self.hinge])
        rng = self.m.jnt_range[[mj.index(j) for j in self.hinge]]
        self.lo, self.hi = rng[:, 0], rng[:, 1]
        is_arm = np.array([any(p in j for p in ARM_PAT) for j in self.hinge])
        self.arm_v = self.vadr[is_arm]

        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.hand_id = [bid(h) for h in HANDS]
        self.foot_id = [bid(f) for f in FEET]
        self.pelvis_id = bid("pelvis")
        self.torso_id = bid("torso_link")
        self.foot_geoms = [
            g for f in FEET
            for g in range(self.m.body_geomadr[bid(f)],
                           self.m.body_geomadr[bid(f)] + self.m.body_geomnum[bid(f)])
            if self.m.geom_contype[g] or self.m.geom_conaffinity[g]
        ]
        # Visual + collision geoms on the legs. The X2 only collides at the feet, so
        # a knee can sit inside the box with zero contact and still look (and be)
        # planted through it.
        LEG_TOKENS = ("hip", "knee", "ankle", "foot", "thigh", "shin", "pelvis")
        self.leg_geoms = []
        for b in range(self.m.nbody):
            bn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if any(k in bn for k in ("wrist", "palm", "hand", "elbow", "shoulder")):
                continue
            if not any(k in bn for k in LEG_TOKENS):
                continue
            for g in range(self.m.body_geomadr[b],
                           self.m.body_geomadr[b] + self.m.body_geomnum[b]):
                self.leg_geoms.append(g)
        self.obj = np.asarray(clip["object_pos_w"], float)
        self.oq = np.asarray(clip["object_quat_w"], float)
        self.half = BOX_SIZE / 2
        z = self.obj[:, 2]
        self.floor = np.flatnonzero(z < z.min() + 0.010)
        off = np.flatnonzero(z > z.min() + 0.010)
        self.grip = (int(off[0]), int(off[-1])) if len(off) else (0, -1)

        init = get_x2_robot_cfg().init_state.joint_pos or {}
        self.q_default = np.zeros(len(self.hinge))
        for i, j in enumerate(self.hinge):
            for pat, v in init.items():
                if re.fullmatch(pat, j):
                    self.q_default[i] = v
        self.is_arm = is_arm
        self.arm_q = self.qadr[is_arm]
        self.arm_default = self.q_default[is_arm]

        jp = np.asarray(clip["joint_pos"], float)
        self.q = np.zeros((len(jp), self.m.nq))
        self.q[:, 0:7] = jp[:, 0:7]
        self.q[:, self.qadr] = jp[:, 7:][:, self.cj_idx]
        self.q[:, self.qadr] = np.clip(self.q[:, self.qadr], self.lo, self.hi)
        self.T = len(jp)

    def box_frame(self, t):
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.oq[t])
        Rb = Rb.reshape(3, 3)
        return Rb, self.obj[t] + Rb @ BOX_OFFSET

    def sdf(self, p, t, r, frame=None):
        Rb, cen = frame if frame else self.box_frame(t)
        d = np.abs(Rb.T @ (p - cen)) - self.half
        return float(np.linalg.norm(np.maximum(d, 0.0)) + min(d.max(), 0.0) - r)

    def geom_radius(self, gi):
        """Half-extent used for clearance. Prefer geom size over the bounding sphere.

        Visual foot meshes have rbound 0.15 m against a real half-extent of ~0.11 m;
        using rbound parks the robot too far to reach the side-face centres.
        """
        if self.m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
            return float(np.max(self.m.geom_size[gi]))
        return float(self.m.geom_rbound[gi])

    def worst_leg_gap(self, t):
        frame = self.box_frame(t)
        worst, at = np.inf, None
        for gi in self.leg_geoms:
            g = self.sdf(self.d.geom_xpos[gi], t, self.geom_radius(gi), frame)
            if g < worst:
                worst, at = g, self.d.geom_xpos[gi].copy()
        return worst, at

    def square_to_box(self):
        """Turn the robot so it meets the box face-on instead of corner-on.

        The demo walks up to the box 32.7 deg off square, so the robot ends up facing
        a corner: the two faces it has to squeeze are skewed across its chest and the
        box visibly sits at an angle in front of it. Rotating the robot *about the box*
        fixes the heading without moving the box a millimetre, and keeps the box
        centred straight ahead because robot and heading turn together.
        """
        Rb, cen = self.box_frame(0)
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, self.q[0, 3:7])
        fwd = Rr.reshape(3, 3)[:, 0].copy()
        fwd[2] = 0.0
        fwd /= np.linalg.norm(fwd)
        best = None
        for a in (0, 1):
            for s in (1.0, -1.0):
                n = s * Rb[:, a].copy()
                n[2] = 0.0
                n /= np.linalg.norm(n)
                # The face the robot should look at has its outward normal pointing
                # back at the robot, so the target heading is -n.
                ang = np.arctan2(fwd[0] * -n[1] - fwd[1] * -n[0], fwd[:2] @ -n[:2])
                if best is None or abs(ang) < abs(best):
                    best = ang
        theta = float(best)
        print(f"squaring robot to box: rotating {np.degrees(theta):+.1f} deg "
              f"about the box (box does not move)")
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
        for t in range(self.T):
            self.q[t, 0:2] = cen[:2] + R @ (self.q[t, 0:2] - cen[:2])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, qz, self.q[t, 3:7])
            self.q[t, 3:7] = out
        return theta

    def grip_weight(self, t):
        """0 while the arms hang at their default, 1 while they hold the box.

        The demo starts swinging the arms toward the box in the first half second,
        which on the robot reads as the hands snapping out before it has even gone
        down. Hold them at the default pose through the descent and bring them onto
        the box over a ramp that finishes exactly as the box leaves the floor.
        """
        g0, g1 = self.grip
        if g1 < g0:
            return 0.0
        if g0 <= t <= g1:
            return 1.0
        u = ((t - (g0 - GRIP_RAMP)) / GRIP_RAMP if t < g0
             else ((g1 + GRIP_RAMP) - t) / GRIP_RAMP)
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def squat_weight(self, t):
        """1 on the floor grasp/set-down, 0 while standing or carrying high.

        The hug must not start until this is already 1 -- squat first, arms still
        hanging, then close the hands on the sides.
        """
        g0, g1 = self.grip
        floor = float(self.obj[:, 2].min())
        lift = float(np.clip((self.obj[t, 2] - floor) / 0.35, 0.0, 1.0))
        if t < g0:
            u = float(np.clip((t - (g0 - GRIP_RAMP - SQUAT_RAMP)) / SQUAT_RAMP, 0.0, 1.0))
            return u * u * (3.0 - 2.0 * u)
        if t > g1:
            u = float(np.clip(((g1 + GRIP_RAMP + SQUAT_RAMP) - t) / SQUAT_RAMP, 0.0, 1.0))
            return u * u * (3.0 - 2.0 * u)
        return 1.0 - lift

    def foot_shift(self):
        """Smallest horizontal slide that lifts the feet clear of the box.

        Only a centimetre or two is needed, so it is applied to the whole clip: a
        time-varying shift would slide planted feet, and these feet never leave the
        floor during the approach.
        """
        worst, at = np.inf, None
        for t in self.floor:
            self.d.qpos[:] = self.q[t]
            mujoco.mj_forward(self.m, self.d)
            g, p = self.worst_leg_gap(t)
            if g < worst:
                worst, at = g, (t, p)
        if worst >= FOOT_GAP:
            print(f"legs already clear by {worst * 1e3:.0f} mm")
            return np.zeros(2)
        t, p = at
        Rb, cen = self.box_frame(t)
        away = p - cen
        away[2] = 0.0
        away /= np.linalg.norm(away)
        shift = away[:2] * (FOOT_GAP - worst)
        print(f"worst leg-box gap {worst * 1e3:+.0f} mm at frame {t}; "
              f"sliding robot {np.linalg.norm(shift) * 1e3:.0f} mm "
              f"to ({shift[0]:+.3f}, {shift[1]:+.3f})")
        return shift

    def assign_faces(self):
        """Give each hand the side face on its own side of the body.

        Fixed once, from the robot's heading at the grip: letting each hand chase its
        nearest face lets the assignment flip mid-carry and drag a hand across the
        box. Both hands press the same axis, because the grasp is a squeeze.
        """
        g0 = self.grip[0]
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, self.q[g0, 3:7])
        left = Rr.reshape(3, 3)[:, 1]
        Rb, _ = self.box_frame(g0)
        ax = int(np.argmax([abs(Rb[:, i] @ left) for i in range(2)]))
        s = float(np.sign(Rb[:, ax] @ left) or 1.0)
        self.face = [(ax, s), (ax, -s)]
        self.depth_ax = 1 - ax
        Rb2, cen = self.box_frame(g0)
        root = self.q[g0, 0:3]
        self.depth_sign = float(np.sign(Rb2[:, self.depth_ax] @ (root - cen)) or 1.0)
        print(f"grip faces: left {'xy'[ax]}{'+' if s > 0 else '-'}   "
              f"right {'xy'[ax]}{'-' if s > 0 else '+'}   "
              f"near-edge {'xy'[self.depth_ax]}{'+' if self.depth_sign > 0 else '-'}")

    def palm_target(self, t, k, frame):
        """Centre of the side face this hand squeezes.

        The hands are driven onto the box rather than merely pushed out of it: they
        now start from the default pose, so there is nothing to push out. Aiming at
        the face centre is the hug -- palms flat on the middle of each side.
        """
        Rb, cen = frame
        R = self.d.xmat[self.hand_id[k]].reshape(3, 3)
        p = self.d.xpos[self.hand_id[k]] + R @ PALM_OFFSET
        ax, sign = self.face[k]
        want = np.zeros(3)
        want[ax] = sign * (self.half[ax] + PALM_RADIUS - GRIP)
        want[2] = GRIP_Z   # lower on the side, never the lid
        lim = self.half[self.depth_ax] - PALM_RADIUS - 0.02
        want[self.depth_ax] = self.depth_sign * float(np.clip(GRIP_DEPTH, 0.0, max(lim, 0.0)))
        return cen + Rb @ want, p

    def _quat_err(self, q_cur, q_des):
        conj = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
        d = np.zeros(4)
        mujoco.mju_mulQuat(d, q_des, conj)
        if d[0] < 0:
            d = -d
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, d, 1.0)
        return v

    def snapshot_feet(self):
        """Planted foot poses after the stance is finalized, before the squat."""
        self.foot_p = np.zeros((self.T, 2, 3))
        self.foot_q = np.zeros((self.T, 2, 4))
        self.torso_q = np.zeros((self.T, 4))
        for t in range(self.T):
            self.d.qpos[:] = self.q[t]
            mujoco.mj_forward(self.m, self.d)
            for k in range(2):
                self.foot_p[t, k] = self.d.xpos[self.foot_id[k]]
                self.foot_q[t, k] = self.d.xquat[self.foot_id[k]]
            self.torso_q[t] = self.d.xquat[self.torso_id]

    def fix_body(self, t, prev_body, iters=50):
        """Drop the pelvis into a squat. Arms stay frozen at whatever they currently are.

        The original clip is a standing bow: pelvis never leaves 0.66 m, so the
        shoulders stay too high for a side hug and the left hand lands on the lid.
        Feet stay where the stance put them; only the legs, waist and root Z move.
        """
        sw = self.squat_weight(t)
        if sw <= 0.0:
            return
        z_des = P_DEEP * sw + P_STAND * (1.0 - sw)
        nv = self.m.nv
        jp_ = np.zeros((3, nv))
        jr_ = np.zeros((3, nv))
        if prev_body is not None:
            self.d.qpos[:] = prev_body
            self.d.qpos[self.arm_q] = self.q[t, self.arm_q]
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            rows, res, wts = [], [], []

            def add(J, r, wt):
                rows.append(J)
                res.append(np.asarray(r, float))
                wts.append(np.full(len(r), wt))

            for k in range(2):
                mujoco.mj_jacBody(self.m, self.d, jp_, jr_, self.foot_id[k])
                add(jp_.copy(), self.foot_p[t, k] - self.d.xpos[self.foot_id[k]], 600.0)
                add(jr_.copy(),
                    self._quat_err(self.d.xquat[self.foot_id[k]], self.foot_q[t, k]),
                    200.0)

            com = np.zeros((3, nv))
            mujoco.mj_jacSubtreeCom(self.m, self.d, com, 0)
            mid = 0.5 * (self.foot_p[t, 0, :2] + self.foot_p[t, 1, :2])
            add(com[:2].copy(), mid - self.d.subtree_com[0, :2], 20.0)

            mujoco.mj_jacBody(self.m, self.d, jp_, None, self.pelvis_id)
            add(jp_[2:3].copy(), [z_des - self.d.xpos[self.pelvis_id][2]], 80.0)

            # Keep the demo's torso lean. Dropping into a squat without this
            # straightens the back, the shoulders stay over the heels, and the
            # palms cannot reach the side-face centres.
            mujoco.mj_jacBody(self.m, self.d, None, jr_, self.torso_id)
            add(jr_.copy(),
                self._quat_err(self.d.xquat[self.torso_id], self.torso_q[t]), 25.0)

            J = np.vstack(rows)
            J[:, self.arm_v] = 0.0
            r = np.concatenate(res)
            wt = np.concatenate(wts)
            Jw = J * wt[:, None]
            dq = np.linalg.solve(Jw.T @ J + 1e-4 * np.eye(nv), Jw.T @ r)
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            mujoco.mj_integratePos(self.m, self.d.qpos, dq, 1.0)
            self.d.qpos[self.qadr] = np.clip(self.d.qpos[self.qadr], self.lo, self.hi)
            self.d.qpos[2] = max(self.d.qpos[2], PELVIS_FLOOR)
            if n < 1e-5:
                break
        mujoco.mj_forward(self.m, self.d)

    def fix_arms(self, t, w, prev, iters=60):
        """Arms-only IK onto the box. Legs, waist and root are never touched.

        The reach is blended at the *target*, not at the joints. Interpolating joint
        angles toward a solution for a face the arm cannot yet reach makes the solver
        thrash against its limits -- that showed up as a 49 rad/s shoulder snap
        halfway through the descent. Walking the target out from where the hands
        already are keeps every intermediate pose reachable.
        """
        frame = self.box_frame(t)
        nv = self.m.nv
        self.d.qpos[self.arm_q] = self.arm_default
        mujoco.mj_forward(self.m, self.d)
        Rb, cen = frame
        goal = []
        for k in range(2):
            tgt, p_def = self.palm_target(t, k, frame)
            g = (1.0 - w) * p_def + w * tgt
            # The straight line from the hanging hand to the face centre can cut the
            # corner of the box, which asks the arm to pass through it.
            local = Rb.T @ (g - cen)
            over = np.abs(local) - (self.half + PALM_RADIUS - GRIP)
            if over.max() < 0.0:
                a = int(np.argmax(over))
                local[a] = np.sign(local[a] or 1.0) * (self.half[a] + PALM_RADIUS - GRIP)
            # Never slide onto the lid or under the box -- those are the two
            # faces the user already rejected.
            local[2] = float(np.clip(local[2],
                                     -(self.half[2] - 0.03),
                                     self.half[2] - 0.03))
            g = cen + Rb @ local
            goal.append(g)
        if prev is not None:
            self.d.qpos[self.arm_q] = prev

        na = len(self.arm_v)
        Jq = np.zeros((na, nv))
        Jq[np.arange(na), self.arm_v] = 1.0
        jp_ = np.zeros((3, nv))
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            rows, res, wts = [], [], []

            def add(J, r, wt):
                rows.append(J)
                res.append(np.asarray(r, float))
                wts.append(np.full(len(r), wt))

            for k in range(2):
                R = self.d.xmat[self.hand_id[k]].reshape(3, 3)
                p = self.d.xpos[self.hand_id[k]] + R @ PALM_OFFSET
                mujoco.mj_jac(self.m, self.d, jp_, None, p, self.hand_id[k])
                add(jp_.copy(), goal[k] - p, 120.0)

            # Seven arm joints chasing a three-axis target leaves four unconstrained
            # directions, and per-frame IK picks a different corner of that nullspace
            # every frame. Unregularized that reads as a 50 rad/s shoulder snap.
            cur = self.d.qpos[self.arm_q]
            add(Jq.copy(), self.arm_default - cur, 0.15)
            if prev is not None:
                add(Jq.copy(), prev - cur, 4.0)

            J = np.vstack(rows)
            mask = np.ones(nv, bool)
            mask[self.arm_v] = False
            J[:, mask] = 0.0
            r = np.concatenate(res)
            wt = np.concatenate(wts)
            Jw = J * wt[:, None]
            dq = np.linalg.solve(Jw.T @ J + 1e-4 * np.eye(nv), Jw.T @ r)
            n = np.linalg.norm(dq)
            if n > 0.20:
                dq *= 0.20 / n
            mujoco.mj_integratePos(self.m, self.d.qpos, dq, 1.0)
            self.d.qpos[self.qadr] = np.clip(self.d.qpos[self.qadr], self.lo, self.hi)
            if n < 1e-5:
                break

    def run(self):
        theta = self.square_to_box()
        self.assign_faces()
        # Sliding the robot changes which geom is closest and in which direction, so
        # one step undershoots. Repeat until the gap actually holds.
        shift = np.zeros(2)
        for _ in range(8):
            s = self.foot_shift()
            if not s.any():
                break
            shift += s
            self.q[:, 0:2] += s
        # The standing clearance misses the squat: knees travel forward over the
        # toes and go through the box. Probe one deep frame and back up if needed.
        g0 = self.grip[0]
        self.snapshot_feet()
        self.d.qpos[:] = self.q[g0]
        self.d.qpos[self.arm_q] = self.arm_default
        self.fix_body(g0, None)
        kg, kp = self.worst_leg_gap(g0)
        if kg < FOOT_GAP and kp is not None:
            Rb, cen = self.box_frame(g0)
            away = kp - cen
            away[2] = 0.0
            nrm = np.linalg.norm(away)
            extra = away[:2] / nrm * (FOOT_GAP - kg) if nrm > 1e-6 else np.zeros(2)
            self.q[:, 0:2] += extra
            shift = shift + extra
            print(f"squat knee clearance {kg * 1e3:+.0f} mm; extra slide "
                  f"{np.linalg.norm(extra) * 1e3:.0f} mm")
        self.snapshot_feet()
        stats, prev_arm, prev_body = [], None, None
        for t in range(self.T):
            self.d.qpos[:] = self.q[t]
            self.d.qpos[self.arm_q] = self.arm_default
            self.fix_body(t, prev_body)
            prev_body = self.d.qpos.copy()
            w = self.grip_weight(t)
            if w > 0.0:
                self.fix_arms(t, w, prev_arm)
                prev_arm = self.d.qpos[self.arm_q].copy()
            self.q[t] = self.d.qpos.copy()
            mujoco.mj_forward(self.m, self.d)
            frame = self.box_frame(t)
            pl = [self.sdf(self.d.xpos[h] + self.d.xmat[h].reshape(3, 3) @ PALM_OFFSET,
                           t, PALM_RADIUS, frame) for h in self.hand_id]
            stats.append(min(pl))
        # The solver changes branch where the box leaves the floor and where it lands,
        # which leaves a few-frame elbow flick at each end of the grip. Filtering the
        # arm columns costs a millimetre of palm contact and removes it.
        from scipy.signal import savgol_filter

        arm_t = savgol_filter(self.q[:, self.arm_q], 15, 3, axis=0)
        self.q[:, self.arm_q] = np.clip(arm_t, self.lo[self.is_arm],
                                        self.hi[self.is_arm])
        stats = []
        for t in range(self.T):
            self.d.qpos[:] = self.q[t]
            mujoco.mj_forward(self.m, self.d)
            frame = self.box_frame(t)
            stats.append(min(
                self.sdf(self.d.xpos[h] + self.d.xmat[h].reshape(3, 3) @ PALM_OFFSET,
                         t, PALM_RADIUS, frame) for h in self.hand_id))

        g0, g1 = self.grip
        a = np.array(stats[g0:g1 + 1]) * 1e3
        print(f"palm penetration while carrying: worst {a.min():+.1f} mm  "
              f"median {np.median(a):+.1f} mm   (was -98 mm)")
        pz = self.q[:, 2]
        print(f"root z: {pz.min():.3f} .. {pz.max():.3f} m   "
              f"(squat target {P_DEEP:.2f}, grip-z {GRIP_Z:+.2f} on side face)")
        arm = self.q[:, self.arm_q]
        v = np.abs(np.diff(arm, axis=0)).max() * 50.0
        print(f"peak arm joint speed: {v:.2f} rad/s   "
              f"(arms held at default until frame {g0 - GRIP_RAMP})")
        worst = np.inf
        for t in list(self.floor[:: max(len(self.floor) // 8, 1)]) + [g0]:
            self.d.qpos[:] = self.q[t]
            mujoco.mj_forward(self.m, self.d)
            g, _ = self.worst_leg_gap(t)
            worst = min(worst, g)
        print(f"leg-box gap after squat:    {worst * 1e3:+.0f} mm")
        self.d.qpos[:] = self.q[g0]
        mujoco.mj_forward(self.m, self.d)
        Rb, cen = self.box_frame(g0)
        for k, name in enumerate(("left", "right")):
            p = self.d.xpos[self.hand_id[k]] + self.d.xmat[self.hand_id[k]].reshape(3, 3) @ PALM_OFFSET
            loc = Rb.T @ (p - cen)
            print(f"  {name} palm at pickup  local {loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}  "
                  f"(side-center z should be 0.00)")
        return shift, theta, self.box_frame(0)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DST))
    args = ap.parse_args()

    clip = np.load(SRC, allow_pickle=True)
    fx = Fixer(clip)
    shift, theta, pivot = fx.run()

    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    T, q = fx.T, fx.q
    jp_out = np.asarray(clip["joint_pos"], float).copy()
    jp_out[:, 0:7] = q[:, 0:7]
    jp_out[:, 7:][:, fx.cj_idx] = q[:, fx.qadr]

    data = mujoco.MjData(fx.m)
    cb = fx.cb
    bpos = np.asarray(clip["body_pos_w"], float).copy()
    bquat = np.asarray(clip["body_quat_w"], float).copy()
    # Bodies mjlab does not carry (ankle spheres, hand contact links, the box link)
    # are not reachable by FK, so move them by the same rigid transform as the robot.
    c_, s_ = np.cos(theta), np.sin(theta)
    Rz = np.array([[c_, -s_], [s_, c_]])
    bpos[:, :, 0:2] = (bpos[:, :, 0:2] - pivot[:2]) @ Rz.T + pivot[:2] + shift
    qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    for t in range(bquat.shape[0]):
        for k in range(bquat.shape[1]):
            o = np.zeros(4)
            mujoco.mju_mulQuat(o, qz, bquat[t, k])
            bquat[t, k] = o
    have = {b: mujoco.mj_name2id(fx.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in cb}
    for t in range(T):
        data.qpos[:] = q[t]
        mujoco.mj_forward(fx.m, data)
        for k, b in enumerate(cb):
            if have[b] >= 0:
                bpos[t, k] = data.xpos[have[b]]
                bquat[t, k] = data.xquat[have[b]]

    def d1(a):
        v = np.zeros_like(a)
        v[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
        v[0], v[-1] = v[1], v[-2]
        return v

    jv = np.asarray(clip["joint_vel"], float).copy()
    jv[:, 0:3] = d1(q[:, 0:3])
    jv[:, 6:][:, fx.cj_idx] = d1(q[:, fx.qadr])

    o = {
        "joint_pos": jp_out.astype(np.float32),
        "joint_vel": jv.astype(np.float32),
        "joint_names": np.array(fx.cj),
        "body_names": np.array(cb),
        "body_pos_w": bpos.astype(np.float32),
        "body_quat_w": bquat.astype(np.float32),
        "body_lin_vel_w": d1(bpos).astype(np.float32),
        "body_ang_vel_w": np.asarray(clip["body_ang_vel_w"], np.float32),
        "fps": np.array([fps], np.int64),
        "stance_shift": np.array([shift[0], shift[1], theta], np.float32),
    }
    for k in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
        o[k] = np.asarray(clip[k], np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **o)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
