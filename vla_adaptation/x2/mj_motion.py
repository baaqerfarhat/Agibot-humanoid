"""Drive the X2 MJCF through the real box-pickup reference motion.

This is where "real task poses" come from for anything that renders X2: the retargeted
clip holosoma trains the box policy against. Rest-pose renders are actively misleading --
at rest the box sits at the robot's feet and the head is upright, so an egocentric camera
sees only floor no matter how it is mounted.

The clip carries the box too, which is what makes it usable on its own: `object_pos_w` /
`object_quat_w` place `largebox_link` per frame, so the whole scene is determined without
running physics. That matters because this MJCF has nu=0 -- there is nothing to actuate.

Layout, verified rather than assumed (see `check_order`):
    joint_pos[:, 0:3]   pelvis position, world      -> qpos[0:3]
    joint_pos[:, 3:7]   pelvis quaternion, wxyz     -> qpos[3:7]
    joint_pos[:, 7:38]  31 joints, in MJCF order    -> qpos[7:38]
    object_pos/quat_w   largebox free joint         -> qpos[38:45]
"""
from __future__ import annotations

import numpy as np
import mujoco

MOTION = ("/home/mtaheri/ws_AgibotX2/holosoma/src/holosoma/holosoma/data/motions/"
          "x2_31dof/whole_body_tracking/box_multispeed/box_speed100.npz")


def load_motion(path: str = MOTION) -> dict:
    """Load the reference clip. 734 frames at 50 fps for box_speed100."""
    d = np.load(path, allow_pickle=True)
    return dict(jp=d["joint_pos"], obj_p=d["object_pos_w"], obj_q=d["object_quat_w"],
                body_p=d["body_pos_w"], fps=int(d["fps"][0]),
                joint_names=[str(x) for x in d["joint_names"]],
                body_names=[str(x) for x in d["body_names"]])


def check_order(m, mo: dict) -> None:
    """Assert the clip's joints line up with the model's, instead of trusting they do."""
    mj = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    mj = [x for x in mj if x not in (None, "floating_base_joint")]  # None = box free joint
    if mj != mo["joint_names"]:
        bad = [(a, b) for a, b in zip(mj, mo["joint_names"]) if a != b]
        raise AssertionError(f"joint order mismatch, first few: {bad[:5]}")
    if m.nq != mo["jp"].shape[1] + 7:
        raise AssertionError(f"nq {m.nq} != motion {mo['jp'].shape[1]} + box 7")


def qpos_at(mo: dict, t: int) -> np.ndarray:
    """Full qpos (robot + box) for frame t."""
    return np.concatenate([mo["jp"][t], mo["obj_p"][t], mo["obj_q"][t]])


def phases(mo: dict, rise: float = 0.02) -> dict:
    """Landmark frames, derived from the box height rather than hand-picked indices.

    `carry` is taken from the middle of the plateau where the box is held high, and
    `place` from where it has come back down and settled -- not from the last frame it
    is still airborne, which is mid-descent and looks like a drop.
    """
    z = mo["obj_p"][:, 2]
    z0 = float(z[0])
    up = z > z0 + rise
    start = int(np.argmax(up))
    peak = int(np.argmax(z))
    high = np.nonzero(z > z0 + 0.9 * (z.max() - z0))[0]
    settled = np.nonzero(~up)[0]
    settled = settled[settled > peak]
    return {"approach": 0, "grasp": start, "lift": (start + peak) // 2,
            "carry": int(high[len(high) // 2]), "peak": peak,
            "place": int(settled[0]) if len(settled) else len(z) - 1,
            "released": int(settled[min(50, len(settled) - 1)]) if len(settled) else len(z) - 1}


def episode(mo: dict, stride: int = 1):
    """Yield (t, qpos) over the clip -- the pose stream a renderer consumes."""
    for t in range(0, len(mo["jp"]), stride):
        yield t, qpos_at(mo, t)


if __name__ == "__main__":
    from mj_cameras import load_with_cameras

    mo = load_motion()
    m = load_with_cameras()
    d = mujoco.MjData(m)
    check_order(m, mo)
    print(f"OK: {len(mo['jp'])} frames @ {mo['fps']} fps, nq={m.nq}")
    ip = mo["body_names"].index("pelvis")
    print("root == pelvis body:", np.allclose(mo["jp"][:, :3], mo["body_p"][:, ip, :], atol=1e-5))
    z = mo["obj_p"][:, 2]
    print(f"box z {z.min():.3f} -> {z.max():.3f}")
    for name, t in phases(mo).items():
        d.qpos[:] = qpos_at(mo, t)
        mujoco.mj_forward(m, d)
        head = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")]
        print(f"  {name:<9} t={t:<4} box_z={mo['obj_p'][t, 2]:.3f} head={np.round(head, 3)}")
