#!/usr/bin/env python3
"""Convert v31's reference clip into the recorded-rollout schema so it can be
rendered next to the actual Isaac rollout with the same camera.

The clip stores 38 joints and 46 bodies; the policy uses 31 joints. Names are
matched explicitly rather than by position.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MOTION = Path("/home/baaqer/baaqer_ws/Agibot-humanoid/adaptation/FOR_MENTOR/"
              "v31_reference_box_speed100.npz")
ROLLOUT = Path("/home/baaqer/baaqer_ws/Agibot-humanoid/adaptation/FOR_MENTOR/"
               "isaac_v31_rollout.npz")
OUT = Path("/home/baaqer/baaqer_ws/Agibot-humanoid/adaptation/FOR_MENTOR/"
           "v31_reference_as_rollout.npz")


def main() -> None:
    m = np.load(MOTION, allow_pickle=True)
    r = np.load(ROLLOUT, allow_pickle=True)
    meta_r = json.loads(str(r["_metadata_json"]))
    dof_names = list(meta_r["dof_names"])

    # `joint_pos` is (T, 38) = 7 floating-base DoF (pos 3 + quat wxyz 4) followed by the
    # 31 joints. `joint_names` has 31 entries and names the JOINT part only, so
    # joint_names[i] is column 7+i. Verified against the policy npz's exported
    # ref_joint_pos: joint_pos[:, 7:38] matches it to 0.0 exactly.
    # Do NOT zip joint_names against columns 0..30 -- every joint ends up shifted by 7.
    clip_joints = [str(s) for s in m["joint_names"]]
    missing = [n for n in dof_names if n not in clip_joints]
    if missing:
        raise SystemExit(f"clip is missing policy joints: {missing}")
    idx = np.array([7 + clip_joints.index(n) for n in dof_names], dtype=int)

    qpos = np.asarray(m["joint_pos"])
    joint_pos = qpos[:, idx]
    base_pos = qpos[:, :3]
    q_wxyz = qpos[:, 3:7]
    base_quat_xyzw = np.stack(
        [q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], axis=1
    )
    print(f"mapped {len(idx)} joints from columns 7..37; base from columns 0..6")

    # `body_pos_w`/`body_quat_w` are PELVIS-RELATIVE despite the _w suffix (pelvis is
    # exactly 0 / identity at every frame), so the base has to come from qpos.
    ref_delta = np.linalg.norm(base_pos[0] - np.asarray(r["root_pos"])[0])
    print(f"clip base vs Isaac spawn at frame 0: {ref_delta * 1000:.1f} mm apart")

    payload = {
        "root_pos": base_pos,
        "root_quat_xyzw": base_quat_xyzw,
        "dof_pos": joint_pos,
        "object_pos": np.asarray(m["object_pos_w"]),
        "object_quat_wxyz": np.asarray(m["object_quat_w"]),
    }
    meta = {
        "dt": 1.0 / float(np.asarray(m["fps"]).ravel()[0]),
        "fps": int(np.asarray(m["fps"]).ravel()[0]),
        "dof_names": dof_names,
        "source": str(MOTION),
        "note": "retargeted reference motion rendered in the rollout schema",
    }
    payload["_metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(OUT, **payload)
    print(f"wrote {OUT}  ({len(joint_pos)} frames @ {meta['fps']} fps)")

    box = payload["object_pos"]
    dist = np.linalg.norm(box[:, :2] - base_pos[:, :2], axis=1)
    print(f"reference box: z {box[0, 2]:.3f} -> max {box[:, 2].max():.3f}, "
          f"pelvis distance {dist[0]:.3f} -> {dist.min():.3f} min")


if __name__ == "__main__":
    main()
