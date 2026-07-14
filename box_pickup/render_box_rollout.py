"""Render a recorded eval rollout (robot + box) to an MP4 using MuJoCo offscreen (EGL)."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import json
import sys

import mujoco
import numpy as np

XML = "/home/baaqer/baaqer_ws/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml"
NPZ = sys.argv[1] if len(sys.argv) > 1 else "/home/baaqer/baaqer_ws/x2_box_eval_rollout.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/baaqer/baaqer_ws/x2_box_carry_progress.mp4"

d = np.load(NPZ, allow_pickle=True)
meta = json.loads(str(d["_metadata_json"]))
fps = int(meta.get("fps", 50))
dof_names = list(meta["dof_names"])

root_pos = d["root_pos"]
root_quat_xyzw = d["root_quat_xyzw"]
dof_pos = d["dof_pos"]
obj_pos = d.get("object_pos")
obj_quat_wxyz = d.get("object_quat_wxyz")
N = root_pos.shape[0]

H, W = 720, 1280
spec = mujoco.MjSpec.from_file(XML)
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H
model = spec.compile()
data = mujoco.MjData(model)

base_adr = model.joint("floating_base_joint").qposadr[0]
dof_adr = np.array([model.joint(n).qposadr[0] for n in dof_names], dtype=int)

box_body = model.body("largebox_link")
box_jnt = int(box_body.jntadr[0])
box_adr = int(model.jnt_qposadr[box_jnt])

renderer = mujoco.Renderer(model, height=H, width=W)

cam = mujoco.MjvCamera()
cam.azimuth = 135.0
cam.elevation = -18.0
cam.distance = 2.8

opt = mujoco.MjvOption()

frames = []
for i in range(N):
    q = data.qpos
    q[base_adr : base_adr + 3] = root_pos[i]
    x, y, z, w = root_quat_xyzw[i]
    q[base_adr + 3 : base_adr + 7] = [w, x, y, z]
    q[dof_adr] = dof_pos[i]
    if obj_pos is not None:
        q[box_adr : box_adr + 3] = obj_pos[i]
        q[box_adr + 3 : box_adr + 7] = obj_quat_wxyz[i]
    mujoco.mj_forward(model, data)

    cam.lookat[0] = float(root_pos[i, 0])
    cam.lookat[1] = float(root_pos[i, 1])
    cam.lookat[2] = 0.45
    renderer.update_scene(data, camera=cam, scene_option=opt)
    frames.append(renderer.render().copy())

print(f"Rendered {len(frames)} frames at {fps} fps -> {OUT}")

try:
    import imageio.v2 as imageio

    imageio.mimwrite(OUT, frames, fps=fps, quality=8, macro_block_size=8)
    print("Wrote MP4 via imageio")
except Exception as e:  # noqa: BLE001
    print(f"imageio failed ({e}); trying opencv")
    import cv2

    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print("Wrote MP4 via opencv")
