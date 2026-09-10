"""Make the OmniRetarget pickup dynamically feasible by slowing it down, not by
changing the pose.

The clip was rejected earlier because the waist wants 240 Nm against a 48 Nm limit,
and the response was to replace the human's 89.6 deg bow with a squat. That was the
wrong call. Holding the full bow against gravity costs 45.1 Nm -- it fits. Essentially
all of the 240 Nm is acceleration: the motion is played too fast and carries
finite-difference noise from retargeting. Removing the bow also removed the reach that
put the hands on the box, which is why the trained policy never grasped anything.

So the pose is left alone and the timing is fixed instead:

  smooth   Savitzky-Golay over root and joints, with the root quaternion sign
           unwrapped first. Filtering a quaternion that flips sign halfway through
           produces garbage, and the garbage shows up as torque spikes.
  retime   a per-frame slowdown, solved so every actuator stays inside its limit
           with margin. Uniform 3x works but stretches an 11.7 s clip to 35 s; the
           torque is only over budget during the bow and the lift, so the slowdown
           is spent there and the rest runs near real time.

Torque is evaluated with the box carried: the robot-only model has no box in it, and
a 1 kg payload at arm's length is worth several Nm at the waist.

    cd ~/baaqer_ws/mjlab && uv run python <this>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

SRC = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
DST = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_retimed.npz"
)

HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")
BOX_MASS = 1.0
GRAV = np.array([0.0, 0.0, -9.81])


def unwrap(q):
    """Remove sign flips so the quaternion track is continuous enough to filter."""
    q = q.copy()
    for t in range(1, len(q)):
        if q[t] @ q[t - 1] < 0:
            q[t] = -q[t]
    return q


def slerp_series(q, idx):
    """Sample a wxyz quaternion track at fractional frame indices."""
    from scipy.spatial.transform import Rotation, Slerp

    q = unwrap(q)
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    out = Slerp(np.arange(len(q)), rot)(np.clip(idx, 0, len(q) - 1)).as_quat()
    return out[:, [3, 0, 1, 2]]


def lerp_series(a, idx):
    i0 = np.clip(np.floor(idx).astype(int), 0, len(a) - 1)
    i1 = np.clip(i0 + 1, 0, len(a) - 1)
    w = (idx - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[i0] * (1 - w) + a[i1] * w


class Model:
    def __init__(self, clip):
        from mjlab.asset_zoo.robots.x2.x2_constants import (
            X2_ARTICULATION,
            get_x2_robot_cfg,
        )

        self.m = get_x2_robot_cfg().spec_fn().compile()
        self.d = mujoco.MjData(self.m)
        cj = [str(x) for x in clip["joint_names"]]
        mj = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(self.m.njnt)]
        self.hinge = [j for j in mj if j in cj]
        self.cj_idx = [cj.index(j) for j in self.hinge]
        self.qadr = np.array([self.m.jnt_qposadr[mj.index(j)] for j in self.hinge])
        self.vadr = np.array([self.m.jnt_dofadr[mj.index(j)] for j in self.hinge])
        self.lim = np.zeros(len(self.hinge))
        for a in X2_ARTICULATION.actuators:
            for pat in a.target_names_expr:
                for i, j in enumerate(self.hinge):
                    if re.fullmatch(pat, j):
                        self.lim[i] = a.effort_limit
        self.hand_id = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, h)
                        for h in HANDS]
        rng = self.m.jnt_range[[mj.index(j) for j in self.hinge]]
        self.lo, self.hi = rng[:, 0], rng[:, 1]

    def qpos(self, jp):
        q = np.zeros((len(jp), self.m.nq))
        q[:, 0:7] = jp[:, 0:7]
        q[:, self.qadr] = jp[:, 7:][:, self.cj_idx]
        return q

    def torque_ratio(self, q, dt, carried, box_acc, worst=None):
        """Peak |tau|/limit per frame, with the box hung off the hands while carried."""
        T = len(q)
        out = np.zeros(T)
        vp, vn = np.zeros(self.m.nv), np.zeros(self.m.nv)
        for t in range(1, T - 1):
            self.d.qpos[:] = q[t]
            mujoco.mj_differentiatePos(self.m, vp, dt, q[t - 1], q[t])
            mujoco.mj_differentiatePos(self.m, vn, dt, q[t], q[t + 1])
            self.d.qvel[:] = 0.5 * (vp + vn)
            self.d.qacc[:] = (vn - vp) / dt
            self.d.xfrc_applied[:] = 0.0
            if carried[t]:
                # Reaction of the payload on each hand: half the box, accelerating.
                f = 0.5 * BOX_MASS * (GRAV - box_acc[t])
                for hid in self.hand_id:
                    self.d.xfrc_applied[hid, 0:3] = f
            mujoco.mj_inverse(self.m, self.d)
            ratio = np.abs(self.d.qfrc_inverse[self.vadr]) / self.lim
            out[t] = ratio.max()
            if worst is not None:
                worst[:] = np.maximum(worst, ratio)
        out[0], out[-1] = out[1], out[-2]
        return out

    def clamp(self, q):
        """Keep the filtered/resampled pose inside the joint limits.

        Savitzky-Golay overshoots at the turning points, and a joint parked past its
        stop makes mj_inverse report the limit constraint force -- tens of Nm on a
        2 Nm wrist. That reads as an infeasible motion when the motion is fine.
        """
        q[:, self.qadr] = np.clip(q[:, self.qadr], self.lo, self.hi)
        return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(DST))
    ap.add_argument("--smooth", type=int, default=11,
                    help="Savitzky-Golay window over the source clip")
    ap.add_argument("--margin", type=float, default=0.75,
                    help="fraction of each actuator limit the retimed clip may use")
    ap.add_argument("--max-slow", type=float, default=6.0)
    ap.add_argument("--sched-window", type=int, default=21,
                    help="smoothing over the slowdown schedule. Too wide and a\n"
                         "narrow torque peak gets averaged away faster than the\n"
                         "solver can build it up, and the schedule stalls.")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--uniform", action="store_true",
                    help="one global slowdown instead of a per-frame schedule")
    args = ap.parse_args()

    clip = np.load(args.src, allow_pickle=True)
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    M = Model(clip)

    jp = np.asarray(clip["joint_pos"], float)
    obj = np.asarray(clip["object_pos_w"], float)
    T0 = len(jp)

    # Smooth in the source timeline, before any resampling.
    q = M.qpos(jp)
    w = args.smooth | 1
    if w >= 5:
        q[:, 0:3] = savgol_filter(q[:, 0:3], w, 3, axis=0)
        q[:, M.qadr] = savgol_filter(q[:, M.qadr], w, 3, axis=0)
        qt = savgol_filter(unwrap(q[:, 3:7]), w, 3, axis=0)
        q[:, 3:7] = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    M.clamp(q)
    jp_s = jp.copy()
    jp_s[:, 0:7] = q[:, 0:7]
    jp_s[:, 7:][:, M.cj_idx] = q[:, M.qadr]

    carried = obj[:, 2] > obj[:, 2].min() + 0.010
    print(f"source {T0} frames @ {fps:g} fps = {T0 / fps:.1f} s, "
          f"carried for {int(carried.sum())} frames")

    # Solve a slowdown schedule. Accelerations fall as 1/s^2, so a frame over budget
    # by a factor r needs about sqrt(r) more time; iterate because stretching one
    # region changes the derivatives everywhere.
    s = np.ones(T0)
    for it in range(args.iters):
        idx = warp_index(s, T0)
        qw = resample_q(M, jp_s, idx)
        ow = lerp_series(obj, idx)
        cw = lerp_series(carried.astype(float), idx) > 0.5
        acc = np.gradient(np.gradient(ow, dt, axis=0), dt, axis=0)
        worst = np.zeros(len(M.hinge))
        r = M.torque_ratio(qw, dt, cw, acc, worst) / args.margin
        # Ratios live on the warped timeline; pull them back to the source frames.
        r_src = np.interp(np.arange(T0), idx, r)
        peak = r.max()
        if args.uniform:
            s = np.full(T0, min(float(np.clip(s[0] * np.sqrt(max(peak, 1e-6)),
                                              1.0, args.max_slow)), args.max_slow))
        else:
            s = np.clip(s * np.sqrt(np.maximum(r_src, 1e-6)) ** 1.0, 1.0, args.max_slow)
            # A slowdown that itself changes abruptly injects acceleration, so the
            # schedule has to be at least as smooth as the motion it is fixing.
            s = savgol_filter(s, max(args.sched_window | 1, 5), 2)
            s = np.clip(s, 1.0, args.max_slow)
        print(f"  iter {it:2d}: peak torque ratio {peak * args.margin:5.2f} of limit   "
              f"slowdown {s.min():.2f}..{s.max():.2f}   "
              f"duration {len(idx) / fps:5.1f} s   "
              f"binding {M.hinge[int(np.argmax(worst))]}")
        if peak <= 1.0:
            break

    idx = warp_index(s, T0)
    write(M, clip, jp_s, idx, dt, fps, Path(args.out), s)


def warp_index(s, T0):
    """Source-frame index for each output frame, given a per-frame slowdown."""
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]))])
    K = int(np.floor(tau[-1])) + 1
    return np.interp(np.arange(K), tau, np.arange(T0))


def resample_q(M, jp_s, idx):
    q = np.zeros((len(idx), M.m.nq))
    q[:, 0:3] = lerp_series(jp_s[:, 0:3], idx)
    q[:, 3:7] = slerp_series(jp_s[:, 3:7], idx)
    q[:, M.qadr] = lerp_series(jp_s[:, 7:][:, M.cj_idx], idx)
    return M.clamp(q)


def write(M, clip, jp_s, idx, dt, fps, out: Path, s):
    K = len(idx)
    cb = [str(x) for x in clip["body_names"]]
    q = resample_q(M, jp_s, idx)

    jp_out = np.zeros((K, jp_s.shape[1]))
    jp_out[:, 0:7] = q[:, 0:7]
    jp_out[:, 7:][:, M.cj_idx] = q[:, M.qadr]

    def d1(a):
        v = np.zeros_like(a)
        v[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
        v[0], v[-1] = v[1], v[-2]
        return v

    # Re-derive every body pose from the resampled joints so the body tracks and the
    # joint track cannot disagree.
    data = mujoco.MjData(M.m)
    bpos = lerp_series(np.asarray(clip["body_pos_w"], float), idx)
    bquat = np.zeros((K, len(cb), 4))
    bquat[..., 0] = 1.0
    for k, b in enumerate(cb):
        i = mujoco.mj_name2id(M.m, mujoco.mjtObj.mjOBJ_BODY, b)
        if i < 0:
            bquat[:, k] = slerp_series(np.asarray(clip["body_quat_w"], float)[:, k], idx)
    have = {b: mujoco.mj_name2id(M.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in cb}
    for t in range(K):
        data.qpos[:] = q[t]
        mujoco.mj_forward(M.m, data)
        for k, b in enumerate(cb):
            if have[b] >= 0:
                bpos[t, k] = data.xpos[have[b]]
                bquat[t, k] = data.xquat[have[b]]

    jv = np.zeros((K, jp_s.shape[1] - 1))
    jv[:, 0:3] = d1(q[:, 0:3])
    jv[:, 6:][:, M.cj_idx] = d1(q[:, M.qadr])

    ang = np.zeros((K, len(cb), 3))
    for k in range(len(cb)):
        for t in range(1, K - 1):
            dq = np.zeros(4)
            conj = bquat[t - 1, k] * np.array([1.0, -1, -1, -1])
            mujoco.mju_mulQuat(dq, bquat[t + 1, k], conj)
            if dq[0] < 0:
                dq = -dq
            v = np.zeros(3)
            mujoco.mju_quat2Vel(v, dq, 1.0)
            ang[t, k] = v / (2 * dt)

    o = {
        "joint_pos": jp_out.astype(np.float32),
        "joint_vel": jv.astype(np.float32),
        "joint_names": np.array([str(x) for x in clip["joint_names"]]),
        "body_names": np.array(cb),
        "body_pos_w": bpos.astype(np.float32),
        "body_quat_w": bquat.astype(np.float32),
        "body_lin_vel_w": d1(bpos).astype(np.float32),
        "body_ang_vel_w": ang.astype(np.float32),
        "fps": np.array([fps], np.int64),
        "slowdown": s.astype(np.float32),
    }
    op = lerp_series(np.asarray(clip["object_pos_w"], float), idx)
    oq = slerp_series(np.asarray(clip["object_quat_w"], float), idx)
    o["object_pos_w"] = op.astype(np.float32)
    o["object_quat_w"] = oq.astype(np.float32)
    o["object_lin_vel_w"] = d1(op).astype(np.float32)
    oa = np.zeros((K, 3))
    for t in range(1, K - 1):
        dq = np.zeros(4)
        conj = oq[t - 1] * np.array([1.0, -1, -1, -1])
        mujoco.mju_mulQuat(dq, oq[t + 1], conj)
        if dq[0] < 0:
            dq = -dq
        v = np.zeros(3)
        mujoco.mju_quat2Vel(v, dq, 1.0)
        oa[t] = v / (2 * dt)
    o["object_ang_vel_w"] = oa.astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **o)
    print(f"\nwrote {out}")
    print(f"  {K} frames @ {fps:g} fps = {K / fps:.1f} s "
          f"(source {len(s) / fps:.1f} s)")


if __name__ == "__main__":
    main()
