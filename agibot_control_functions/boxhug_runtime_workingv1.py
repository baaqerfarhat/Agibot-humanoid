#!/usr/bin/env python3
"""Numpy-only runtime for the mjlab X2 box-hug policy (task Mjlab-BoxHug-Ref-X2-Timed).

Shared by the robot-side deploy script (`deploy_x2_box_hug.py`) and by the mjlab-side
export/verification script (`scripts/x2_box_hug/export_boxhug_policy_npz.py`), so
the command the robot feeds the policy is the very same code that was checked in
simulation. Only numpy is needed here.

What the policy consumes (153 floats, see DEPLOY_NOTES.md in the handoff folder):

    [ base_ang_vel(3)        pelvis gyro, pelvis frame
      projected_gravity(3)   unit gravity in the pelvis frame (upright ~ [0, 0, -1])
      joint_pos - default(31)
      joint_vel(31)
      previous action(31)    raw policy output, clipped to +-10
      command(54) ]

    command = [ phase one-hot(9), phase progress(1), target pelvis height(1),
                box centre rel. pelvis in the yaw-aligned pelvis frame(3),
                sin/cos of box yaw rel. pelvis yaw(2),
                palm target - palm position, yaw-aligned pelvis frame, L then R(6),
                bilateral-grasp flag(1),
                reference joint pos - default(31) ]

In training the phases were driven by TIME (the clip's timeline), so on the robot the
phase, its progress, the reference joints, the target height and the nominal box
pose all come from one clock started at engage. The palm positions come from forward
kinematics of the measured joints (the same MJCF tree the policy was trained on,
shipped inside the .npz). The box is not perceived: its pose relative to the pelvis
is taken from the clip (``box_source="clip"``) or, while it is being held, attached
to the measured hands (``box_source="hands"``).
"""

from __future__ import annotations

import json
import math

import numpy as np

PHASE_NAMES = (
    "STAND", "LOWER", "GRASP", "LIFT", "HOLD", "LOWER_BOX", "SET_DOWN", "RELEASE", "DONE",
)
NUM_PHASES = len(PHASE_NAMES)
COMMAND_DIM = NUM_PHASES + 14  # 23 without the reference joints
HOLDING_PHASES = (2, 3, 4, 5, 6)  # GRASP, LIFT, HOLD, LOWER_BOX, SET_DOWN (hand inset)


# ------------------------------------------------------------------ rotations
def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        np.float64,
    )


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return quat_xyzw_to_mat((x, y, z, w))


def yaw_of_mat(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def roll_of_xyzw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return float(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float64)


def axis_angle_mat(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3)
    a = a / n
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def projected_gravity(pelvis_quat_xyzw) -> np.ndarray:
    """Unit gravity vector in the pelvis frame: R^T [0, 0, -1]."""
    R = quat_xyzw_to_mat(pelvis_quat_xyzw)
    return (R.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)


# ------------------------------------------------------------------ kinematics
class TreeFK:
    """Forward kinematics of the robot's MJCF body tree, relative to the pelvis.

    Arrays (from the .npz, extracted from the compiled mjlab model):
      fk_parent (B,) int      parent body index, -1 for the pelvis (body 0)
      fk_body_pos (B,3)       body frame position in the parent frame
      fk_body_quat (B,4)      body frame orientation in the parent frame (w,x,y,z)
      fk_joint_idx (B,) int   index into the policy joint order, -1 = no hinge
      fk_joint_axis (B,3)     hinge axis in the body frame
      fk_joint_pos (B,3)      hinge anchor in the body frame
      fk_site_body (S,) int, fk_site_pos (S,3): sites of interest
    """

    def __init__(self, d):
        self.parent = np.asarray(d["fk_parent"], np.int64)
        self.body_pos = np.asarray(d["fk_body_pos"], np.float64)
        self.body_rot = np.stack([quat_wxyz_to_mat(q) for q in d["fk_body_quat"]])
        self.joint_idx = np.asarray(d["fk_joint_idx"], np.int64)
        self.joint_axis = np.asarray(d["fk_joint_axis"], np.float64)
        self.joint_pos = np.asarray(d["fk_joint_pos"], np.float64)
        self.body_names = [str(s) for s in d["fk_body_names"]]
        self.site_body = np.asarray(d["fk_site_body"], np.int64)
        self.site_pos = np.asarray(d["fk_site_pos"], np.float64)
        self.site_names = [str(s) for s in d["fk_site_names"]]
        self.nb = len(self.parent)

    def body_poses(self, q: np.ndarray):
        """Returns (pos (B,3), rot (B,3,3)) of every body in the pelvis frame."""
        q = np.asarray(q, np.float64)
        pos = np.zeros((self.nb, 3))
        rot = np.zeros((self.nb, 3, 3))
        for b in range(self.nb):
            p = self.parent[b]
            if p < 0:
                pos[b] = 0.0
                rot[b] = np.eye(3)
                continue
            R = rot[p] @ self.body_rot[b]
            x = pos[p] + rot[p] @ self.body_pos[b]
            j = self.joint_idx[b]
            if j >= 0:
                Rj = axis_angle_mat(self.joint_axis[b], q[j])
                anchor = self.joint_pos[b]
                x = x + R @ (anchor - Rj @ anchor)
                R = R @ Rj
            pos[b] = x
            rot[b] = R
        return pos, rot

    def site_positions(self, q: np.ndarray) -> np.ndarray:
        pos, rot = self.body_poses(q)
        out = np.zeros((len(self.site_body), 3))
        for i, (b, sp) in enumerate(zip(self.site_body, self.site_pos)):
            out[i] = pos[b] + rot[b] @ sp
        return out

    def site(self, name: str) -> int:
        return self.site_names.index(name)

    def body(self, name: str) -> int:
        return self.body_names.index(name)


# ------------------------------------------------------------------ policy
class NumpyPolicy:
    """rsl_rl MLP actor (ELU) with the observation normaliser baked in."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.d = d
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))
        self.action_clip = float(self.meta.get("action_clip", 10.0))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs (153,) or (N,153) -> raw action (31,) or (N,31), clipped like training."""
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)  # ELU(1)
        x = x @ self.W[-1].T + self.b[-1]
        return np.clip(x, -self.action_clip, self.action_clip)


# ------------------------------------------------------------------ timeline + command
class BoxHugTimeline:
    """Phase / progress / clip frame from the seconds since engage (training's rule)."""

    def __init__(self, meta: dict):
        self.seg_start_s = np.asarray(meta["seg_start_s"], np.float64)  # (9,)
        self.seg_start = np.asarray(meta["seg_start_frame"], np.int64)
        self.seg_len = np.asarray(meta["seg_len_frames"], np.int64)
        self.durations = np.asarray(meta["phase_durations_s"], np.float64)
        self.fps = float(meta["motion_fps"])
        self.n_frames = int(meta["motion_frames"])
        self.clip_end_s = self.n_frames / self.fps
        self.grasp_on_s = float(meta["grasp_flag_on_s"])
        self.grasp_off_s = float(meta["grasp_flag_off_s"])

    def phase(self, t: float) -> int:
        return int(np.clip(np.sum(t >= self.seg_start_s) - 1, 0, NUM_PHASES - 1))

    def state(self, t: float):
        """(phase, progress, frame) exactly as BoxHugCommand computes them."""
        ph = self.phase(t)
        pt = max(0.0, t - self.seg_start_s[ph])
        progress = float(np.clip(pt / self.durations[ph], 0.0, 1.0))
        k = int(round(pt * self.fps))
        frame = int(self.seg_start[ph] + min(k, self.seg_len[ph] - 1))
        return ph, progress, frame

    def grasp_flag(self, t: float) -> float:
        return 1.0 if self.grasp_on_s <= t < self.grasp_off_s else 0.0


class BoxHugCommandBuilder:
    """Builds the 54-float command from the clock, the pelvis attitude and the joints."""

    def __init__(self, policy: NumpyPolicy, box_source: str = "clip",
                 grasp_flag: str = "timeline"):
        d, meta = policy.d, policy.meta
        if box_source not in ("clip", "hands"):
            raise ValueError(box_source)
        if grasp_flag not in ("timeline", "zero"):
            raise ValueError(grasp_flag)
        self.box_source = box_source
        self.grasp_mode = grasp_flag
        self.meta = meta
        self.timeline = BoxHugTimeline(meta)
        self.fk = TreeFK(d)
        self.default = np.asarray(meta["default_joint_pos"], np.float32)
        self.ref_q = d["ref_joint_pos"].astype(np.float32)        # (T,31)
        self.ref_pelvis_z = d["ref_pelvis_z"].astype(np.float32)  # (T,)
        self.ref_box_b = d["ref_box_pos_b"].astype(np.float64)    # (T,3) box centre, yaw frame
        self.ref_box_rot_b = d["ref_box_rot_b"].astype(np.float64)  # (T,3,3)
        self.ref_palm_b = d["ref_palm_pos_b"].astype(np.float64)  # (T,2,3) palm sites, yaw frame
        self.box_half = np.asarray(meta["box_half"], np.float64)
        self.standoff = float(meta["palm_standoff"])
        self.patch_x = tuple(meta["grasp_patch_x"])
        self.patch_z = tuple(meta["grasp_patch_z"])
        self.palm_sites = [self.fk.site(n) for n in meta["palm_site_names"]]
        self.hand_side_sign = np.array([1.0, -1.0])
        # Diagnostics of the last call.
        self.last = {}

    # -- helpers -------------------------------------------------------------
    def palm_positions_b(self, pelvis_quat_xyzw, q):
        """Palm-centre sites in the yaw-aligned pelvis frame, (2,3), plus R_rp."""
        R = quat_xyzw_to_mat(pelvis_quat_xyzw)
        R_rp = rot_z(-yaw_of_mat(R)) @ R  # roll/pitch only: pelvis -> yaw frame
        sites = self.fk.site_positions(q)
        return np.stack([R_rp @ sites[i] for i in self.palm_sites]), R_rp

    def palm_targets(self, box_c, R_box, palm_b):
        """Replicates BoxHugCommand._compute_features' palm_target_w in the yaw frame."""
        rel = palm_b - box_c[None, :]
        palm_box = rel @ R_box  # R^T rel per hand
        box_y = R_box[:, 1]
        left_sign = np.sign(box_y[1]) or 1.0  # robot-left vector in the yaw frame is +y
        face_sign = left_sign * self.hand_side_sign  # (2,)
        pelvis_box = R_box.T @ (-box_c)
        near_sign = np.sign(pelvis_box[0]) or -1.0
        x0, x1 = self.patch_x
        if near_sign < 0:
            lo, hi = x0, x1
        else:
            lo, hi = -x1, -x0
        hy = self.box_half[1] + self.standoff
        target_box = np.stack(
            [
                np.clip(palm_box[:, 0], lo, hi),
                face_sign * hy,
                np.clip(palm_box[:, 2], self.patch_z[0], self.patch_z[1]),
            ],
            axis=-1,
        )
        return box_c[None, :] + target_box @ R_box.T

    # -- main -----------------------------------------------------------------
    def build(self, t: float, pelvis_quat_xyzw, q) -> np.ndarray:
        ph, progress, frame = self.timeline.state(t)
        q = np.asarray(q, np.float64)
        palm_b, R_rp = self.palm_positions_b(pelvis_quat_xyzw, q)

        R_box = self.ref_box_rot_b[frame]
        if self.box_source == "hands" and ph in HOLDING_PHASES:
            # Box rides with the real hands: clip offset from the clip's palms.
            box_c = palm_b.mean(0) + (self.ref_box_b[frame] - self.ref_palm_b[frame].mean(0))
        else:
            box_c = self.ref_box_b[frame].copy()
        targets = self.palm_targets(box_c, R_box, palm_b)
        rel_palm = targets - palm_b
        dyaw = yaw_of_mat(R_box)

        cmd = np.zeros(COMMAND_DIM + len(self.default), np.float32)
        cmd[ph] = 1.0
        cmd[NUM_PHASES] = progress
        cmd[NUM_PHASES + 1] = self.ref_pelvis_z[frame]
        cmd[NUM_PHASES + 2:NUM_PHASES + 5] = box_c
        cmd[NUM_PHASES + 5] = math.sin(dyaw)
        cmd[NUM_PHASES + 6] = math.cos(dyaw)
        cmd[NUM_PHASES + 7:NUM_PHASES + 13] = rel_palm.reshape(-1)
        cmd[NUM_PHASES + 13] = self.timeline.grasp_flag(t) if self.grasp_mode == "timeline" else 0.0
        cmd[COMMAND_DIM:] = self.ref_q[frame] - self.default

        self.last = {
            "phase": ph, "progress": progress, "frame": frame,
            "target_height": float(self.ref_pelvis_z[frame]),
            "box_b": box_c, "palm_b": palm_b, "palm_err": np.linalg.norm(rel_palm, axis=-1),
            "grasp_flag": float(cmd[NUM_PHASES + 13]), "R_rp": R_rp,
        }
        return cmd


class BoxHugObservation:
    """Assembles the 153-float actor observation."""

    def __init__(self, policy: NumpyPolicy, command: BoxHugCommandBuilder):
        self.default = np.asarray(policy.meta["default_joint_pos"], np.float32)
        self.command = command
        self.obs_dim = int(policy.meta["obs_dim"])
        self.last_action = np.zeros(len(self.default), np.float32)

    def build(self, t: float, ang_vel_pelvis, pelvis_quat_xyzw, q, dq) -> np.ndarray:
        cmd = self.command.build(t, pelvis_quat_xyzw, q)
        obs = np.concatenate(
            [
                np.asarray(ang_vel_pelvis, np.float32),
                projected_gravity(pelvis_quat_xyzw),
                np.asarray(q, np.float32) - self.default,
                np.asarray(dq, np.float32),
                self.last_action,
                cmd,
            ]
        ).astype(np.float32)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"obs dim {obs.shape[0]} != {self.obs_dim}")
        return obs


def pelvis_height_from_feet(fk: TreeFK, R_rp: np.ndarray, q, foot_body_names,
                            ankle_height: float) -> float:
    """Pelvis height above the floor assuming the lower foot is flat on it."""
    pos, _ = fk.body_poses(q)
    z = [float((R_rp @ pos[fk.body(n)])[2]) for n in foot_body_names]
    return ankle_height - min(z)
