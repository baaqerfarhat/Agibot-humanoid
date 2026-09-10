"""Re-author the OmniRetarget pickup as a squat lift the X2 can actually hold.

The retargeted clip reproduces a human spine bow: the pelvis pitches 71 deg forward
over straight legs, the torso goes horizontal, and the waist is asked for 105 Nm
against a 48 Nm limit. A bow like that also buys about half a metre of forward reach,
which is the only reason the hands get to a box sitting 0.41 m from the feet. Take the
bow away and the box is out of reach -- so the stance has to come closer.

The box is never touched. Two things change:

  stance   the whole robot is translated/yawed to a standoff from which the grip
           points are reachable from a squat. Solved once, then held for the clip,
           so the feet stay planted exactly as they were relative to each other.
  posture  per-frame whole-body IK trades the bow for knee flexion, keeps the CoM
           over the feet, and puts the palms on the box faces instead of 100 mm
           inside them.

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
    "/whole_body_tracking/sub3_largebox_003_squat_lift.npz"
)

BOX_SIZE = np.array([0.4712, 0.4587, 0.4079])
BOX_OFFSET = np.array([0.0015, -0.0007, 0.0058])
PALM_OFFSET = np.array([0.01, 0.0, -0.10])
PALM_RADIUS = 0.05
GRIP = 0.005
# The hand geom on wrist_roll_link is a slab, half-extents (0.033, 0.056, 0.107):
# thin in local x, long in local z. So the gripping faces are the ones normal to
# local x, and that is the axis that has to face the box.
PALM_AXIS = 0

HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO = "torso_link"
PITCH_CAP = np.radians(32.0)
ARM_PAT = ("shoulder", "elbow", "wrist")
# Frames that define the hard part of the reach: first contact, deep grip, set-down.
PROBE = (110, 150, 300, 400)
# Grip faces are read off the floor-grasp frame, where the hands squeeze opposite
# sides of the box. Reading them mid-carry instead picks up a corner hug that cannot
# hold the box by friction.
FACE_FRAME = 110
# Frames over which the arms hand off between their default posture and the hug.
GRIP_RAMP = 70
# Pelvis follows the grip down and back up: standing when the hands are at carry
# height, squatting to the proven depth when they are at the floor. Reach analysis
# says the floor grasp needs the pelvis at or below ~0.33 m.
P_DEEP, P_STAND = 0.30, 0.67
GRIP_Z_FLOOR, GRIP_Z_CARRY = 0.27, 0.69
PELVIS_FLOOR = 0.26
# Gap the feet must keep from the box. The box is a free body: any overlap at reset
# is resolved by launching it, which is what "the robot kicks the box away" was.
FOOT_BOX_GAP = 0.05
RAMP = 25  # frames to ease the hands onto / off the box faces


def quat_err(q_cur, q_des):
    conj = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    d = np.zeros(4)
    mujoco.mju_mulQuat(d, q_des, conj)
    if d[0] < 0:
        d = -d
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, d, 1.0)
    return v


def cap_pitch(q, cap):
    w, x, y, z = q
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    if abs(pitch) <= cap:
        return q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = np.zeros(4)
    mujoco.mju_euler2Quat(out, np.array([roll, np.clip(pitch, -cap, cap), yaw]), "xyz")
    return out


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def qz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


class Retargeter:
    def __init__(self, model, clip):
        self.model = model
        self.data = mujoco.MjData(model)
        self.clip = clip
        cj = [str(x) for x in clip["joint_names"]]
        cb = [str(x) for x in clip["body_names"]]
        self.cj, self.cb = cj, cb
        mj_j = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(model.njnt)]
        self.hinge = [j for j in mj_j if j in cj]
        self.qadr = np.array([model.jnt_qposadr[mj_j.index(j)] for j in self.hinge])
        self.vadr = np.array([model.jnt_dofadr[mj_j.index(j)] for j in self.hinge])
        self.lo = model.jnt_range[[mj_j.index(j) for j in self.hinge], 0]
        self.hi = model.jnt_range[[mj_j.index(j) for j in self.hinge], 1]
        self.is_arm = np.array([any(p in j for p in ARM_PAT) for j in self.hinge])
        self.arm_v = self.vadr[self.is_arm]

        from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg

        init = get_x2_robot_cfg().init_state.joint_pos or {}
        self.q_default = np.zeros(len(self.hinge))
        for i, j in enumerate(self.hinge):
            for pat, v in init.items():
                if re.fullmatch(pat, j):
                    self.q_default[i] = v

        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        self.foot_id = [bid(f) for f in FEET]
        self.hand_id = [bid(h) for h in HANDS]
        self.torso_id = bid(TORSO)
        self.pelvis_id = bid("pelvis")

        jp = np.asarray(clip["joint_pos"], np.float64)
        self.qj_ref = jp[:, 7:][:, [cj.index(j) for j in self.hinge]]
        self.root_pos, self.root_quat = jp[:, 0:3], jp[:, 3:7]
        self.bpos = np.asarray(clip["body_pos_w"], np.float64)
        self.bquat = np.asarray(clip["body_quat_w"], np.float64)
        self.obj = np.asarray(clip["object_pos_w"], np.float64)
        self.oq = np.asarray(clip["object_quat_w"], np.float64)
        self.cbi = {b: cb.index(b) for b in (*FEET, *HANDS, TORSO, "pelvis")}
        self.T = jp.shape[0]
        self.half = BOX_SIZE / 2
        # Pivot for the stance transform: where the feet stand in the raw clip.
        self.pivot = 0.5 * (self.bpos[0][self.cbi[FEET[0]]]
                            + self.bpos[0][self.cbi[FEET[1]]])
        self.pivot[2] = 0.0
        # Face assignment is fixed so it cannot flip mid-carry and drag a hand
        # across the box. Both hands use the same axis: it is a squeeze. Pick the
        # box axis that lies across the robot rather than reading it off the demo
        # hands, so each palm gets the face already on its own side.
        self.grip = self._grip_window()
        gf = self.grip[0] if self.grip[1] >= self.grip[0] else FACE_FRAME
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, self.root_quat[gf])
        left_dir = Rr.reshape(3, 3)[:, 1]
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.oq[gf])
        Rb = Rb.reshape(3, 3)
        ax = int(np.argmax([abs(Rb[:, i] @ left_dir) for i in range(2)]))
        s = float(np.sign(Rb[:, ax] @ left_dir) or 1.0)
        self.face = [(ax, s), (ax, -s)]
        # Depth axis of the face: the horizontal one the hands do NOT press. Aiming
        # at the middle of a 0.46 m deep face puts the target 0.24 m past the near
        # edge, and that overhang alone is most of the arm's 0.56 m reach. Gripping
        # near the edge closest to the robot is the same friction squeeze for a
        # fraction of the reach.
        self.depth_ax = 1 - ax
        self.gf = gf
        self.depth_sign = 1.0          # replaced by set_stance once the shift is known
        # Collidable geoms only. Each foot also carries a visual mesh whose bounding
        # radius is 0.150 m against the contact spheres' 0.012 m, and counting it
        # charges the stance 0.14 m of clearance it does not owe.
        self.foot_geoms_local = [
            [g for g in range(model.body_geomadr[fid],
                              model.body_geomadr[fid] + model.body_geomnum[fid])
             if model.geom_contype[g] or model.geom_conaffinity[g]]
            for fid in self.foot_id
        ]
        self.foot_geoms = [g for gs in self.foot_geoms_local for g in gs]
        z = self.obj[:, 2]
        self.floor_frames = np.flatnonzero(z < z.min() + 0.010)[::10]
        self.q_prev = None
        self.palm_align = 0.0
        self.grip_faces = False
        self.grip_z = 0.0
        self.grip_depth = 0.0
        self.pitch_cap = PITCH_CAP

    def set_stance(self, dx, dy, dyaw):
        """Pick which edge of the gripped faces to aim at: the one on the robot's side.

        This has to be read off the stance-shifted root rather than the raw clip.
        The stance moves the robot up to a quarter of a metre, which is enough to
        put it on the other side of the box's depth axis, and gripping the far edge
        then means reaching straight across the box -- 0.25 m of pure overhang that
        the arm does not have.
        """
        gf = self.gf
        root = rz(dyaw) @ (self.root_pos[gf] - self.pivot) + self.pivot \
            + np.array([dx, dy, 0.0])
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.oq[gf])
        Rb = Rb.reshape(3, 3)
        centre = self.obj[gf] + Rb @ BOX_OFFSET
        self.depth_sign = float(np.sign(Rb[:, self.depth_ax] @ (root - centre)) or 1.0)

    def _palm_local(self, t, k):
        ci = self.cbi[HANDS[k]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.bquat[t][ci])
        p = self.bpos[t][ci] + R.reshape(3, 3) @ PALM_OFFSET
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.oq[t])
        Rb = Rb.reshape(3, 3)
        centre = self.obj[t] + Rb @ BOX_OFFSET
        return Rb.T @ (p - centre), Rb, centre

    def _nearest_face(self, t, k):
        local, _, _ = self._palm_local(t, k)
        d = np.abs(local) - self.half
        ax = int(np.argmax(d[:2]))  # horizontal faces only: never grip the lid
        return ax, float(np.sign(local[ax]) or 1.0)

    def _grip_window(self):
        """Frames where the box is off the floor, i.e. the hands are actually carrying.

        The old test -- any frame where a demo hand is inside the box -- was true for
        343 of 584 frames, so the arms were being hauled toward the box all the way
        down and the right one swung across the squat. The box leaving the floor is
        the only unambiguous evidence that a hand is bearing load.
        """
        z = self.obj[:, 2]
        off = np.flatnonzero(z > z.min() + 0.010)
        return (int(off[0]), int(off[-1])) if len(off) else (0, -1)

    def grip_phase(self, t):
        """0 while the arms hang at their default, 1 while they hold the box."""
        g0, g1 = self.grip
        if g1 < g0:
            return 0.0
        if g0 <= t <= g1:
            return 1.0
        u = ((t - (g0 - GRIP_RAMP)) / GRIP_RAMP if t < g0
             else ((g1 + GRIP_RAMP) - t) / GRIP_RAMP)
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)   # smoothstep: no velocity jump at the handoff

    def palm_target(self, t, k):
        """Where the palm should be: nowhere while descending, on the face when gripping.

        Squat first with the arms where they naturally hang, then hug. Giving the
        arms a target during the descent is what put the right hand in the way of
        the squat.
        """
        if self.grip_phase(t) <= 0.0:
            return None
        _, Rb, centre = self._palm_local(t, k)
        ax, sign = self.face[k]
        want = np.zeros(3)
        want[ax] = sign * (self.half[ax] + PALM_RADIUS - GRIP)
        want[2] = self.grip_z                 # height on the face, box-local
        # Along the face, toward the robot. Clamped to keep the palm on the face
        # rather than wrapped around the near corner, where it would slide off.
        lim = self.half[self.depth_ax] - PALM_RADIUS - 0.02
        want[self.depth_ax] = self.depth_sign * float(
            np.clip(self.grip_depth, 0.0, max(lim, 0.0))
        )
        return centre + Rb @ want

    def palm_normal(self, t, k):
        """Inward normal of the face this hand grips, in world.

        The hand slab is symmetric about its palm axis, so either sign presses the
        same face. Which one is picked has to be decided here, from the reference
        pose, and then held fixed: choosing it inside the IK loop lets the target
        flip between iterations whenever the alignment passes through 90 degrees,
        and the solver tears the rest of the body apart chasing both.
        """
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.oq[t])
        ax, sign = self.face[k]
        n = -sign * Rb.reshape(3, 3)[:, ax]
        Rh = np.zeros(9)
        mujoco.mju_quat2Mat(Rh, self.bquat[t][self.cbi[HANDS[k]]])
        return n if Rh.reshape(3, 3)[:, PALM_AXIS] @ n >= 0.0 else -n

    def pelvis_height(self, t, palm):
        """Squat depth implied by how low the hands have to go this frame."""
        zs = [p[2] for p in palm if p is not None]
        if not zs:
            zs = []
            for k in range(2):
                ci = self.cbi[HANDS[k]]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, self.bquat[t][ci])
                zs.append((self.bpos[t][ci] + R.reshape(3, 3) @ PALM_OFFSET)[2])
        gz = float(np.mean(zs))
        s = np.clip((gz - GRIP_Z_FLOOR) / (GRIP_Z_CARRY - GRIP_Z_FLOOR), 0.0, 1.0)
        return P_DEEP + (P_STAND - P_DEEP) * s

    def targets(self, t, dx, dy, dyaw):
        """Reference-derived IK targets, with the robot side moved by the stance."""
        R = rz(dyaw)
        shift = np.array([dx, dy, 0.0])
        xf = lambda p: R @ (p - self.pivot) + self.pivot + shift
        qf = lambda q: (lambda o: (mujoco.mju_mulQuat(o, qz(dyaw), q), o)[1])(np.zeros(4))
        fp = [xf(self.bpos[t][self.cbi[f]]) for f in FEET]
        fq = [qf(self.bquat[t][self.cbi[f]]) for f in FEET]
        palm = [self.palm_target(t, k) for k in range(2)]
        return {
            "grip": self.grip_phase(t),
            "palm_n": [self.palm_normal(t, k) for k in range(2)],
            "foot_p": fp,
            "foot_q": fq,
            "com_xy": 0.5 * (fp[0][:2] + fp[1][:2]),
            "torso_q": cap_pitch(qf(self.bquat[t][self.cbi[TORSO]]), self.pitch_cap),
            "palm": palm,
            "pelvis_z": self.pelvis_height(t, palm),
            "q_ref": self.qj_ref[t],
        }

    def seed(self, t, dx, dy, dyaw):
        R = rz(dyaw)
        self.data.qpos[0:3] = R @ (self.root_pos[t] - self.pivot) + self.pivot \
            + np.array([dx, dy, 0.0])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz(dyaw), self.root_quat[t])
        self.data.qpos[3:7] = out
        self.data.qpos[self.qadr] = self.qj_ref[t]

    def solve(self, tg, iters):
        model, data, nv = self.model, self.data, self.model.nv
        jp_ = np.zeros((3, nv))
        jr_ = np.zeros((3, nv))
        for _ in range(iters):
            mujoco.mj_forward(model, data)
            rows, res, wts = [], [], []

            def add(J, r, w):
                rows.append(J)
                res.append(np.asarray(r, float))
                wts.append(np.full(len(r), w))

            for k in range(2):
                mujoco.mj_jacBody(model, data, jp_, jr_, self.foot_id[k])
                add(jp_.copy(), tg["foot_p"][k] - data.xpos[self.foot_id[k]], 600.0)
                add(jr_.copy(),
                    quat_err(data.xquat[self.foot_id[k]], tg["foot_q"][k]), 200.0)

            jc = np.zeros((3, nv))
            mujoco.mj_jacSubtreeCom(model, data, jc, 0)
            add(jc[:2].copy(), tg["com_xy"] - data.subtree_com[0][:2], 15.0)

            mujoco.mj_jacBody(model, data, None, jr_, self.torso_id)
            add(jr_.copy(), quat_err(data.xquat[self.torso_id], tg["torso_q"]), 15.0)

            # Scheduled squat. Without this the solver has no reason to bend the
            # knees and instead drives the root through the floor chasing the hands.
            mujoco.mj_jacBody(model, data, jp_, None, self.pelvis_id)
            add(jp_[2:3].copy(),
                [tg["pelvis_z"] - data.xpos[self.pelvis_id][2]], 60.0)

            Jp = np.zeros((len(self.hinge), nv))
            Jp[np.arange(len(self.hinge)), self.vadr] = 1.0
            # The clip's arm angles come from a human shoulder and retarget into
            # contorted X2 poses, so regularize the arms toward the robot's own
            # default instead. The palm position and normal tasks still pull them
            # onto the box; away from the box they simply hang naturally.
            post = np.where(self.is_arm, self.q_default, tg["q_ref"])
            add(Jp, post - data.qpos[self.qadr], 0.0)
            wts[-1] = np.where(self.is_arm, 8.0, 0.5)

            # Per-frame IK has no memory: neighbouring frames can satisfy the same
            # tasks from different nullspace corners, which reads as jitter. Filtering
            # it out afterwards drags the feet off their targets, so damp it here
            # instead by pulling toward the previous frame's solution.
            if self.q_prev is not None:
                add(Jp.copy(), self.q_prev - data.qpos[self.qadr], 4.0)

            J = np.vstack(rows)
            J[:, self.arm_v] = 0.0    # stage 1 owns the squat; arms are frozen
            r = np.concatenate(res)
            w = np.concatenate(wts)
            Jw = J * w[:, None]
            dq = np.linalg.solve(Jw.T @ J + 1e-4 * np.eye(nv), Jw.T @ r)
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
            data.qpos[self.qadr] = np.clip(data.qpos[self.qadr], self.lo, self.hi)
            data.qpos[2] = max(data.qpos[2], PELVIS_FLOOR)
            if n < 1e-5:
                break
        mujoco.mj_forward(model, data)
        self._solve_arms(tg, iters)

    def _solve_arms(self, tg, iters):
        """Close the hands onto the box using the arms only.

        Stage 1 has already placed the feet, pelvis and torso. Letting the palm
        tasks back into that solve is what threw the CoM half a metre and lifted a
        foot off the floor: reaching for the far face is cheaper for the solver if
        it leans the whole body. Freezing everything below the shoulders means the
        arms either reach the box or they do not, and the squat is untouched either
        way.
        """
        if tg["grip"] <= 0.0:
            return
        model, data, nv = self.model, self.data, self.model.nv
        jp_ = np.zeros((3, nv))
        jr_ = np.zeros((3, nv))
        for _ in range(iters):
            mujoco.mj_forward(model, data)
            rows, res, wts = [], [], []

            def add(J, r, w):
                rows.append(J)
                res.append(np.asarray(r, float))
                wts.append(np.full(len(r), w))

            for k in range(2):
                if tg["palm"][k] is None:
                    continue
                R = data.xmat[self.hand_id[k]].reshape(3, 3)
                pt = data.xpos[self.hand_id[k]] + R @ PALM_OFFSET
                mujoco.mj_jac(model, data, jp_, None, pt, self.hand_id[k])
                add(jp_.copy(), tg["palm"][k] - pt, 40.0 * tg["grip"])

                # Point the flat of the hand at the face it grips. Position alone
                # leaves the wrist free to spin about the arm, and the solver spends
                # that freedom wherever the nullspace is cheapest, which is what
                # turns the hand edge-on to the box.
                mujoco.mj_jacBody(model, data, None, jr_, self.hand_id[k])
                add(jr_.copy(), np.cross(R[:, PALM_AXIS], tg["palm_n"][k]),
                    self.palm_align * tg["grip"])

            Jp = np.zeros((len(self.hinge), nv))
            Jp[np.arange(len(self.hinge)), self.vadr] = 1.0
            add(Jp, self.q_default - data.qpos[self.qadr], 0.6)
            if self.q_prev is not None:
                add(Jp.copy(), self.q_prev - data.qpos[self.qadr], 4.0)

            J = np.vstack(rows)
            mask = np.ones(nv, bool)
            mask[self.arm_v] = False
            J[:, mask] = 0.0          # body is fixed; only the arms may move
            r = np.concatenate(res)
            w = np.concatenate(wts)
            Jw = J * w[:, None]
            dq = np.linalg.solve(Jw.T @ J + 1e-4 * np.eye(nv), Jw.T @ r)
            n = np.linalg.norm(dq)
            if n > 0.25:
                dq *= 0.25 / n
            mujoco.mj_integratePos(model, data.qpos, dq, 1.0)
            data.qpos[self.qadr] = np.clip(data.qpos[self.qadr], self.lo, self.hi)
            if n < 1e-5:
                break
        mujoco.mj_forward(model, data)

    def residuals(self, tg):
        d = self.data
        foot = max(float(np.linalg.norm(tg["foot_p"][k] - d.xpos[self.foot_id[k]]))
                   for k in range(2))
        palm = 0.0
        for k in range(2):
            if tg["palm"][k] is None:
                continue
            R = d.xmat[self.hand_id[k]].reshape(3, 3)
            pt = d.xpos[self.hand_id[k]] + R @ PALM_OFFSET
            palm = max(palm, float(np.linalg.norm(tg["palm"][k] - pt)))
        com = float(np.linalg.norm(d.subtree_com[0][:2] - tg["com_xy"]))
        q = d.xquat[self.torso_id]
        pitch = float(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))
        return foot, palm, com, pitch

    def floor_clearance(self, dx, dy, dyaw):
        """Worst foot-box gap over the frames where the box is sitting on the floor.

        Those are the only frames where the feet can hit it, and they are exactly the
        ones PROBE misses -- every probe frame has the box already in the air, so a
        clearance term evaluated there is always satisfied and the optimizer happily
        parks the feet inside the box at reset.

        The feet track their stance-shifted targets to about 2 mm, so this reads them
        off the clip directly instead of paying for an IK solve per frame.
        """
        R = rz(dyaw)
        shift = np.array([dx, dy, 0.0])
        worst = np.inf
        for t in self.floor_frames:
            Rb = np.zeros(9)
            mujoco.mju_quat2Mat(Rb, self.oq[t])
            Rb = Rb.reshape(3, 3)
            centre = self.obj[t] + Rb @ BOX_OFFSET
            for k in range(2):
                ci = self.cbi[FEET[k]]
                pos = R @ (self.bpos[t][ci] - self.pivot) + self.pivot + shift
                fq = np.zeros(4)
                mujoco.mju_mulQuat(fq, qz(dyaw), self.bquat[t][ci])
                Rf = np.zeros(9)
                mujoco.mju_quat2Mat(Rf, fq)
                Rf = Rf.reshape(3, 3)
                for gi in self.foot_geoms_local[k]:
                    p = pos + Rf @ self.model.geom_pos[gi]
                    d = np.abs(Rb.T @ (p - centre)) - self.half
                    worst = min(worst, float(
                        np.linalg.norm(np.maximum(d, 0.0)) + min(d.max(), 0.0)
                        - self.model.geom_rbound[gi]))
        return worst

    def score_stance(self, p):
        dx, dy, dyaw = p
        self.set_stance(dx, dy, dyaw)
        tot = 0.0
        for t in PROBE:
            self.seed(t, dx, dy, dyaw)
            tg = self.targets(t, dx, dy, dyaw)
            self.solve(tg, 40)
            foot, palm, com, _ = self.residuals(tg)
            tot += palm + 0.5 * foot + 0.3 * com
        return tot + 40.0 * max(0.0, FOOT_BOX_GAP
                                - self.floor_clearance(dx, dy, dyaw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--out", default=str(DST))
    ap.add_argument("--stance", nargs=3, type=float, default=None,
                    metavar=("DX", "DY", "DYAW_DEG"),
                    help="fix the stance instead of solving for it")
    ap.add_argument("--fix-yaw", type=float, default=None, metavar="DEG",
                    help="hold the stance yaw at DEG and solve only the translation. "
                         "The score rewards reach, not squareness, so left free it "
                         "keeps the clip's corner-first approach.")
    ap.add_argument("--grip-z", type=float, default=0.0,
                    help="palm height on the side face, metres above the box "
                         "centre; 0 is the vertical middle")
    ap.add_argument("--pitch-cap", type=float, default=np.degrees(PITCH_CAP),
                    metavar="DEG",
                    help="torso pitch ceiling. The demo bow is 89.6 deg and needs "
                         "105 Nm at a 48 Nm waist; some lean has to come back to "
                         "reach the box, so this trades waist torque for reach")
    ap.add_argument("--grip-depth", type=float, default=0.0,
                    help="palm position along the side face toward the robot, "
                         "metres from the box centre; 0 is the mid-depth grip that "
                         "the arms cannot reach")
    ap.add_argument("--grip-faces", action="store_true",
                    help="aim both palms at the centres of the two opposing "
                         "faces instead of nudging the demo grip out of the box")
    ap.add_argument("--palm-align", type=float, default=0.0,
                    help="weight on aligning the palm normal with the gripped "
                         "box face; 0 leaves the wrist free")
    ap.add_argument("--smooth", type=int, default=9,
                    help="Savitzky-Golay window (frames) over the solved trajectory")
    args = ap.parse_args()

    from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg

    model = get_x2_robot_cfg().spec_fn().compile()
    clip = np.load(SRC, allow_pickle=True)
    rt = Retargeter(model, clip)
    rt.palm_align = args.palm_align
    rt.grip_faces = args.grip_faces
    rt.grip_z = args.grip_z
    rt.grip_depth = args.grip_depth
    rt.pitch_cap = np.radians(args.pitch_cap)
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    print(f"grip faces: left {'xy'[rt.face[0][0]]}{'+' if rt.face[0][1] > 0 else '-'}"
          f"   right {'xy'[rt.face[1][0]]}{'+' if rt.face[1][1] > 0 else '-'}")

    from scipy.optimize import minimize

    if args.stance is not None:
        dx, dy, dyaw = args.stance[0], args.stance[1], np.radians(args.stance[2])
        print(f"\nstance fixed: dx {dx:+.3f} m  dy {dy:+.3f} m  "
              f"yaw {np.degrees(dyaw):+.1f} deg  "
              f"(score {rt.score_stance([dx, dy, dyaw]):.4f})")
        return finish(rt, model, fps, args, dx, dy, dyaw)

    if args.fix_yaw is not None:
        yfix = np.radians(args.fix_yaw)
        print(f"\nsolving stance with yaw held at {args.fix_yaw:+.2f} deg "
              f"(robot moves, box does not)...")
        grid, best_g = None, np.inf
        for dxg in np.arange(-0.35, 0.45, 0.05):
            for dyg in np.arange(-0.55, 0.30, 0.05):
                s = rt.score_stance([dxg, dyg, yfix])
                if s < best_g:
                    best_g, grid = s, np.array([dxg, dyg])
        print(f"  grid best dx {grid[0]:+.2f} dy {grid[1]:+.2f}  (score {best_g:.4f})")
        simplex = np.vstack([grid, grid + [0.04, 0.0], grid + [0.0, 0.04]])
        best = minimize(lambda p: rt.score_stance([p[0], p[1], yfix]), grid,
                        method="Nelder-Mead",
                        options={"xatol": 5e-3, "fatol": 2e-3, "maxfev": 100,
                                 "initial_simplex": simplex})
        dx, dy = best.x
        print(f"  stance shift dx {dx:+.3f} m  dy {dy:+.3f} m  "
              f"yaw {args.fix_yaw:+.2f} deg   (score {best.fun:.4f})")
        return finish(rt, model, fps, args, dx, dy, yfix)

    print("\nsolving stance (robot moves, box does not)...")
    # The squeeze axis has to line up with the shoulders, which is a rotation of
    # tens of degrees away from the clip's facing. That is a different basin, not a
    # local correction, so sweep coarsely before refining.
    grid, best_g = None, np.inf
    for dxg in np.arange(-0.20, 0.45, 0.10):
        for dyg in np.arange(-0.50, 0.25, 0.10):
            for yg in np.radians(np.arange(-60, 75, 15)):
                s = rt.score_stance([dxg, dyg, yg])
                if s < best_g:
                    best_g, grid = s, np.array([dxg, dyg, yg])
    print(f"  grid best dx {grid[0]:+.2f} dy {grid[1]:+.2f} "
          f"yaw {np.degrees(grid[2]):+.0f} deg  (score {best_g:.4f})")
    simplex = np.vstack([grid, grid + [0.06, 0, 0], grid + [0, 0.06, 0],
                         grid + [0, 0, np.radians(6)]])
    best = minimize(rt.score_stance, grid, method="Nelder-Mead",
                    options={"xatol": 5e-3, "fatol": 2e-3, "maxfev": 120,
                             "initial_simplex": simplex})
    dx, dy, dyaw = best.x
    print(f"  stance shift dx {dx:+.3f} m  dy {dy:+.3f} m  yaw {np.degrees(dyaw):+.1f} deg"
          f"   (score {best.fun:.4f})")
    return finish(rt, model, fps, args, dx, dy, dyaw)


def finish(rt, model, fps, args, dx, dy, dyaw):
    rt.set_stance(dx, dy, dyaw)
    T = rt.T
    qpos = np.zeros((T, model.nq))
    st = {"foot": [], "palm": [], "com": [], "pitch": []}
    rt.seed(0, dx, dy, dyaw)
    print("\nper-frame IK:")
    for t in range(T):
        tg = rt.targets(t, dx, dy, dyaw)
        rt.solve(tg, args.iters)
        qpos[t] = rt.data.qpos.copy()
        rt.q_prev = rt.data.qpos[rt.qadr].copy()
        foot, palm, com, pitch = rt.residuals(tg)
        st["foot"].append(foot)
        st["palm"].append(palm)
        st["com"].append(com)
        st["pitch"].append(pitch)
        if t % 100 == 0:
            print(f"  t={t / fps:5.2f}s  foot {foot * 1e3:5.1f}  palm {palm * 1e3:5.1f}"
                  f"  com {com * 1e3:5.1f} mm  torso {np.degrees(pitch):+5.1f} deg")

    print()
    for k, unit in (("foot", "mm"), ("palm", "mm"), ("com", "mm")):
        a = np.array(st[k]) * 1e3
        print(f"{k:5} residual  median {np.median(a):6.1f}  p90 {np.percentile(a, 90):6.1f}"
              f"  max {a.max():6.1f} {unit}")
    print(f"torso pitch    max {np.degrees(max(st['pitch'])):6.1f} deg   (was 89.6)")

    # Per-frame IK has no memory, so neighbouring solves can sit in slightly
    # different nullspace corners. That reads as jitter and blows up the
    # finite-difference accelerations the torque check runs on.
    if args.smooth and args.smooth >= 5:
        from scipy.signal import savgol_filter

        w = args.smooth | 1
        acc_before = _peak_acc(qpos, rt, fps)
        qpos[:, 0:3] = savgol_filter(qpos[:, 0:3], w, 3, axis=0)
        qpos[:, rt.qadr] = savgol_filter(qpos[:, rt.qadr], w, 3, axis=0)
        qpos[:, rt.qadr] = np.clip(qpos[:, rt.qadr], rt.lo, rt.hi)
        n = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
        qpos[:, 3:7] /= np.where(n > 0, n, 1.0)
        print(f"smoothing (window {w}): peak joint accel "
              f"{acc_before:.0f} -> {_peak_acc(qpos, rt, fps):.0f} rad/s^2")

    write(model, qpos, rt, fps, Path(args.out), dx, dy, dyaw)


def _peak_acc(qpos, rt, fps):
    q = qpos[:, rt.qadr]
    return float(np.abs(np.diff(q, 2, axis=0)).max() * fps * fps)


def write(model, qpos, rt, fps, out: Path, dx, dy, dyaw):
    T = qpos.shape[0]
    data = mujoco.MjData(model)
    dt = 1.0 / fps
    cb, cj = rt.cb, rt.cj

    have = {}
    for b in cb:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if i >= 0:
            have[b] = i
    bpos = np.zeros((T, len(cb), 3))
    bquat = np.zeros((T, len(cb), 4))
    bquat[..., 0] = 1.0
    # Bodies the mjlab model does not carry (world, ankle spheres, box link) keep
    # their clip values, moved by the same stance transform as the robot.
    R = rz(dyaw)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for k, b in enumerate(cb):
            if b in have:
                bpos[t, k] = data.xpos[have[b]]
                bquat[t, k] = data.xquat[have[b]]
            else:
                bpos[t, k] = R @ (rt.bpos[t, k] - rt.pivot) + rt.pivot \
                    + np.array([dx, dy, 0.0])
                o = np.zeros(4)
                mujoco.mju_mulQuat(o, qz(dyaw), rt.bquat[t, k])
                bquat[t, k] = o

    def d1(a):
        v = np.zeros_like(a)
        v[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
        v[0], v[-1] = v[1], v[-2]
        return v

    qj = qpos[:, rt.qadr]
    jorder = np.array([cj.index(j) for j in rt.hinge])
    jp_out = np.zeros((T, 38))
    jp_out[:, 0:3] = qpos[:, 0:3]
    jp_out[:, 3:7] = qpos[:, 3:7]
    jp_out[:, 7 + jorder] = qj
    jv_out = np.zeros((T, 37))
    jv_out[:, 0:3] = d1(qpos[:, 0:3])
    jv_out[:, 6 + jorder] = d1(qj)

    ang = np.zeros((T, len(cb), 3))
    for k in range(len(cb)):
        for t in range(1, T - 1):
            ang[t, k] = quat_err(bquat[t - 1, k], bquat[t + 1, k]) / (2 * dt)

    o = {
        "joint_pos": jp_out.astype(np.float32),
        "joint_vel": jv_out.astype(np.float32),
        "joint_names": np.array(cj),
        "body_names": np.array(cb),
        "body_pos_w": bpos.astype(np.float32),
        "body_quat_w": bquat.astype(np.float32),
        "body_lin_vel_w": d1(bpos).astype(np.float32),
        "body_ang_vel_w": ang.astype(np.float32),
        "fps": np.array([fps], np.int64),
        "stance_shift": np.array([dx, dy, dyaw], np.float32),
    }
    for k in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w"):
        o[k] = np.asarray(rt.clip[k], np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **o)
    pi = cb.index("pelvis")
    print(f"\nwrote {out}")
    print(f"  pelvis z {bpos[:, pi, 2].min():.3f} .. {bpos[:, pi, 2].max():.3f} m")


if __name__ == "__main__":
    main()
