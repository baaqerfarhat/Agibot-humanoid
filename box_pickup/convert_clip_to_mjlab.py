"""Convert a holosoma WBT clip into the motion file mjlab's tracking task reads.

The two schemas are the same BeyondMimic npz, so this is a reindex rather than a
translation. Three things differ:

  * holosoma's `joint_pos` carries the floating base in the leading 7 columns and
    `joint_vel` in the leading 6; mjlab takes the root from the anchor body instead,
    so those columns come off.
  * the body arrays must be ordered by the *robot entity's* body list, because
    mjlab's MotionLoader slices them with indices from `robot.find_bodies(...)`. Our
    clip carries 46 bodies -- `world`, ten ankle spheres, two hand contact links and
    `largebox_link` -- against the entity's 32.
  * the box rides along in `object_*`. mjlab's stock tracking task ignores those keys;
    they are preserved here so the box-aware task can read them.

    .venv/bin/python convert_clip_to_mjlab.py [in.npz] [out.npz]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_IN = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
DEFAULT_OUT = Path(
    "/home/baaqer/baaqer_ws/mjlab/motions/x2_box_walk_feasible.npz"
)

BODY_ARRAYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
OBJECT_ARRAYS = ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg
    from mjlab.entity.entity import Entity

    robot = Entity(get_x2_robot_cfg())
    want_bodies = list(robot.body_names)
    want_joints = list(robot.joint_names)

    c = np.load(src, allow_pickle=True)
    have_bodies = [str(x) for x in c["body_names"]]
    have_joints = [str(x) for x in c["joint_names"]]
    fps = float(np.asarray(c["fps"]).reshape(-1)[0])

    missing_b = [b for b in want_bodies if b not in have_bodies]
    missing_j = [j for j in want_joints if j not in have_joints]
    if missing_b or missing_j:
        raise SystemExit(f"clip is missing bodies {missing_b} / joints {missing_j}")

    bperm = [have_bodies.index(b) for b in want_bodies]
    jperm = [have_joints.index(j) for j in want_joints]

    jp = np.asarray(c["joint_pos"], np.float32)
    jv = np.asarray(c["joint_vel"], np.float32)
    n_j = len(have_joints)
    # Strip the floating base. Guard on width rather than assuming: a clip that
    # already carries bare joints would otherwise lose seven real columns.
    if jp.shape[1] == n_j + 7:
        root_pos, root_quat_wxyz = jp[:, 0:3], jp[:, 3:7]
        jp = jp[:, 7:]
    elif jp.shape[1] == n_j:
        root_pos = root_quat_wxyz = None
    else:
        raise SystemExit(f"unexpected joint_pos width {jp.shape[1]} for {n_j} joints")
    if jv.shape[1] == n_j + 6:
        jv = jv[:, 6:]
    elif jv.shape[1] != n_j:
        raise SystemExit(f"unexpected joint_vel width {jv.shape[1]} for {n_j} joints")

    out = {
        "joint_pos": jp[:, jperm].astype(np.float32),
        "joint_vel": jv[:, jperm].astype(np.float32),
        "fps": np.array([fps], np.float32),
        "joint_names": np.array(want_joints),
        "body_names": np.array(want_bodies),
    }
    for k in BODY_ARRAYS:
        out[k] = np.asarray(c[k], np.float32)[:, bperm]
    for k in OBJECT_ARRAYS:
        if k in c.files:
            out[k] = np.asarray(c[k], np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)

    T = out["joint_pos"].shape[0]
    print(f"{src.name} -> {dst}")
    print(f"  {T} frames @ {fps:g} Hz ({T/fps:.2f} s)")
    print(f"  joints {jp.shape[1]} -> {out['joint_pos'].shape[1]}, "
          f"bodies {len(have_bodies)} -> {len(want_bodies)}")
    if root_pos is not None:
        # The anchor body is what mjlab uses for the root, so it has to agree with the
        # root the clip was authored around, or the whole motion sits in the wrong place.
        ai = want_bodies.index("torso_link")
        print(f"  dropped root: pos[0] {np.round(root_pos[0], 4)} "
              f"quat[0] {np.round(root_quat_wxyz[0], 4)}")
        print(f"  anchor torso_link pos[0] {np.round(out['body_pos_w'][0, ai], 4)}")
    pi = want_bodies.index("pelvis")
    print(f"  pelvis z: min {out['body_pos_w'][:, pi, 2].min():.3f} "
          f"max {out['body_pos_w'][:, pi, 2].max():.3f}")
    if "object_pos_w" in out:
        print(f"  box carried through: z {out['object_pos_w'][:, 2].min():.3f} .. "
              f"{out['object_pos_w'][:, 2].max():.3f}")


if __name__ == "__main__":
    main()
