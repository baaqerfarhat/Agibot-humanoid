"""Cut the walking phase out of the raw OmniRetarget box clip.

The retarget output (196 frames @30 Hz) is: stand, squat, grasp, lift, then five
side-steps carrying the box 1.3 m, then squat, set down, stand, and one more step.
For an in-place pickup demo everything between the lift and the set-down is dead
weight, and the steps are what drag the feet and ankles around.

This keeps the lift (frames 0-51) and the set-down (136-185), rigidly re-anchors
the set-down onto the end of the lift so the robot never travels, bridges the
seam, and eases the tail back to the exact opening pose so the clip starts and
ends standing upright.  Output is in the training .npz format.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

WS = Path("/home/baaqer/baaqer_ws")
RETARGET = WS / "holosoma/src/holosoma_retargeting/holosoma_retargeting"
SRC = RETARGET / "demo_results/x2/object_interaction/omomo/sub3_largebox_003_original.npz.bak_notrim"
XML = RETARGET / "models/x2/x2_31dof_w_largebox.xml"
OUT = WS / "holosoma/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking/sub3_largebox_003_nowalk.npz"

IN_FPS = 30
OUT_FPS = 50
LIFT = (0, 52)  # stand -> squat -> grasp -> lift, before the first side-step
SETDOWN = (136, 186)  # both feet planted again -> squat -> set down -> stand
BRIDGE_S = 0.50  # seam between the two, held at carry height
TAIL_S = 0.60  # ease back onto the opening stance


def log(m=""):
    print(m, flush=True)


def ease(t):
    return t * t * (3.0 - 2.0 * t)


def yaw_of(quat_wxyz):
    return Rot.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler("zyx")[:, 0]


def rotate_about(pos, quat, dyaw, pivot_xy, target_xy):
    """Rigid SE(2) move: spin by dyaw about pivot, then drop onto target."""
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s], [s, c]])
    out_pos = pos.copy()
    out_pos[:, :2] = (pos[:, :2] - pivot_xy) @ R.T + target_xy
    spin = Rot.from_euler("z", dyaw)
    out_quat = (spin * Rot.from_quat(quat[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    return out_pos, out_quat


_FK = {}


def foot_pair(frame):
    """World (left, right) ankle positions for a single qpos frame."""
    if not _FK:
        model = mujoco.MjModel.from_xml_path(str(XML))
        _FK["m"], _FK["d"] = model, mujoco.MjData(model)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        _FK["i"] = [names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
    _FK["d"].qpos[:] = frame
    mujoco.mj_forward(_FK["m"], _FK["d"])
    return [_FK["d"].xpos[i].copy() for i in _FK["i"]]


def blend(a, b, w):
    """Blend two full qpos frames; quaternions slerp, everything else lerps."""
    out = np.zeros((len(w), 45))
    for lo, hi in ((0, 3), (7, 38), (38, 41)):
        out[:, lo:hi] = a[lo:hi] + (b[lo:hi] - a[lo:hi]) * w[:, None]
    for lo in (3, 41):
        key = Rot.from_quat(np.stack([a[lo : lo + 4], b[lo : lo + 4]])[:, [1, 2, 3, 0]])
        out[:, lo : lo + 4] = Slerp([0, 1], key)(w).as_quat()[:, [3, 0, 1, 2]]
    return out


def resample(q, in_fps, out_fps):
    dur = (len(q) - 1) / in_fps
    t_in = np.arange(len(q)) / in_fps
    t_out = np.arange(0, dur, 1.0 / out_fps)
    out = np.zeros((len(t_out), 45))
    for lo, hi in ((0, 3), (7, 38), (38, 41)):
        for c in range(lo, hi):
            out[:, c] = np.interp(t_out, t_in, q[:, c])
    for lo in (3, 41):
        rots = Rot.from_quat(q[:, lo : lo + 4][:, [1, 2, 3, 0]])
        out[:, lo : lo + 4] = Slerp(t_in, rots)(t_out).as_quat()[:, [3, 0, 1, 2]]
    return out


def so3_derivative(quat_wxyz, dt):
    r = Rot.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    rel = (r[2:] * r[:-2].inv()).as_rotvec() / (2.0 * dt)
    return np.concatenate([rel[:1], rel, rel[-1:]], axis=0)


def to_training_npz(q, fps, out_path):
    """Replay the qpos through MuJoCo and log the frames the trainer expects."""
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    dt = 1.0 / fps

    lin = np.gradient(q[:, 0:3], dt, axis=0)
    ang = so3_derivative(q[:, 3:7], dt)
    dofv = np.gradient(q[:, 7:38], dt, axis=0)
    olin = np.gradient(q[:, 38:41], dt, axis=0)
    oang = so3_derivative(q[:, 41:45], dt)

    n = len(q)
    nb = model.nbody
    out = {
        "joint_pos": np.zeros((n, 38)),
        "joint_vel": np.zeros((n, 37)),
        "body_pos_w": np.zeros((n, nb, 3)),
        "body_quat_w": np.zeros((n, nb, 4)),
        "body_lin_vel_w": np.zeros((n, nb, 3)),
        "body_ang_vel_w": np.zeros((n, nb, 3)),
        "object_pos_w": np.zeros((n, 3)),
        "object_quat_w": np.zeros((n, 4)),
        "object_lin_vel_w": np.zeros((n, 3)),
        "object_ang_vel_w": np.zeros((n, 3)),
    }
    v = np.zeros(6)
    for f in range(n):
        data.qpos[:] = q[f]
        data.qvel[:] = np.concatenate([lin[f], ang[f], dofv[f], olin[f], oang[f]])
        mujoco.mj_forward(model, data)
        out["joint_pos"][f] = data.qpos[:-7]
        out["joint_vel"][f] = data.qvel[:-6]
        out["body_pos_w"][f] = data.xpos
        out["body_quat_w"][f] = data.xquat
        for b in range(nb):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            out["body_ang_vel_w"][f, b] = v[0:3]
            out["body_lin_vel_w"][f, b] = v[3:6]
        out["object_pos_w"][f] = data.qpos[-7:-4]
        out["object_quat_w"][f] = data.qpos[-4:]
        out["object_lin_vel_w"][f] = data.qvel[-6:-3]
        out["object_ang_vel_w"][f] = data.qvel[-3:]

    out["fps"] = np.array([fps])
    out["joint_names"] = np.array(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)][1:-1]
    )
    out["body_names"] = np.array([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nb)])
    np.savez(out_path, **out)
    return out


def main():
    q = np.load(SRC, allow_pickle=True)["qpos"].astype(np.float64)
    log(f"source {SRC.name}: {len(q)} frames @ {IN_FPS} Hz ({len(q)/IN_FPS:.2f}s)")

    a = q[LIFT[0] : LIFT[1]].copy()
    b = q[SETDOWN[0] : SETDOWN[1]].copy()
    log(f"  keep  lift    frames {LIFT[0]:3d}-{LIFT[1]-1:3d}  ({len(a)/IN_FPS:.2f}s)")
    log(f"  keep  setdown frames {SETDOWN[0]:3d}-{SETDOWN[1]-1:3d}  ({len(b)/IN_FPS:.2f}s)")
    log(f"  drop  walk    frames {LIFT[1]:3d}-{SETDOWN[0]-1:3d}"
        f"  ({(SETDOWN[0]-LIFT[1])/IN_FPS:.2f}s, 5 side-steps)")
    log(f"  drop  tail    frames {SETDOWN[1]:3d}-{len(q)-1:3d}  (final step)\n")

    # Re-anchor the set-down onto the end of the lift so the robot stays put.
    # Match the feet, not the pelvis: the feet are what touch the ground, and a
    # pelvis-matched seam leaves the two stances 17 cm apart, which no amount of
    # leg IK can then hide.
    fa, fb = foot_pair(a[-1]), foot_pair(b[0])
    dyaw = np.arctan2(*(fa[0] - fa[1])[[1, 0]]) - np.arctan2(*(fb[0] - fb[1])[[1, 0]])
    pivot = (fb[0] + fb[1])[:2] / 2
    target = (fa[0] + fa[1])[:2] / 2
    travel = np.linalg.norm(pivot - target)
    b[:, 0:3], b[:, 3:7] = rotate_about(b[:, 0:3], b[:, 3:7], dyaw, pivot, target)
    b[:, 38:41], b[:, 41:45] = rotate_about(b[:, 38:41], b[:, 41:45], dyaw, pivot, target)
    res = max(np.linalg.norm(foot_pair(b[0])[i][:2] - fa[i][:2]) for i in (0, 1))
    log(f"[1] re-anchor set-down on the feet: removed {travel:.2f} m of travel and"
        f" {np.degrees(dyaw):+.1f} deg of turn")
    log(f"    stance mismatch left at the seam: {res*100:.1f} cm (stance widths differ)")

    nb_ = int(BRIDGE_S * IN_FPS)
    bridge = blend(a[-1], b[0], ease(np.linspace(0, 1, nb_ + 2)[1:-1]))
    log(f"[2] bridge the seam with {nb_} frames ({BRIDGE_S:.2f}s),"
        f" box {a[-1,40]:.2f} -> {b[0,40]:.2f} m")

    stitched = np.concatenate([a, bridge, b], axis=0)

    # ease the robot (not the box, it is already down) back onto the opening pose
    nt = int(TAIL_S * IN_FPS)
    w = ease(np.linspace(0, 1, nt + 1)[1:])
    tail = blend(stitched[-1], np.concatenate([q[0, :38], stitched[-1, 38:]]), w)
    tail[:, 38:45] = stitched[-1, 38:45]
    tail[:, 0:2] = stitched[-1, 0:2]  # stay where we are, only the pose settles
    out = np.concatenate([stitched, tail], axis=0)
    log(f"[3] settle onto the opening upright pose over {nt} frames ({TAIL_S:.2f}s)")

    log(f"\n[4] resample {IN_FPS} -> {OUT_FPS} Hz")
    out = resample(out, IN_FPS, OUT_FPS)
    log(f"    {len(out)} frames ({len(out)/OUT_FPS:.2f}s)")

    log("\n[5] replay through MuJoCo and write the training format")
    d = to_training_npz(out, OUT_FPS, OUT)
    bn = list(d["body_names"])
    lf, rf = bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    fp = d["body_pos_w"]
    step = np.linalg.norm(fp[:, lf, :2] - fp[0, lf, :2], axis=1).max()
    step_r = np.linalg.norm(fp[:, rf, :2] - fp[0, rf, :2], axis=1).max()
    log(f"    foot travel: left {step*100:.1f} cm, right {step_r*100:.1f} cm  (was 130 cm)")
    log(f"    box height {d['object_pos_w'][:,2].min():.2f} .. {d['object_pos_w'][:,2].max():.2f} m")
    log(f"    root height {d['joint_pos'][0,2]:.2f} (start) .. {d['joint_pos'][-1,2]:.2f} (end) m")
    log(f"\nwrote {OUT}  ({len(out)} frames, {len(out)/OUT_FPS:.2f}s)")


if __name__ == "__main__":
    main()
