"""Render a recorded eval rollout of the slope-crawl policy to an MP4
(MuJoCo offscreen, EGL), with the slope terrain mesh in the scene.

Usage: python render_crawl_rollout.py <rollout.npz> <out.mp4>
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import json
import subprocess
import sys

import mujoco
import numpy as np

XML = "/home/baaqer/baaqer_ws/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/x2/x2_31dof.xml"
TERRAIN_OBJ = (
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
    "g1_29dof/whole_body_tracking/terrain_slope.obj"
)
NPZ = sys.argv[1]
OUT = sys.argv[2]

d = np.load(NPZ, allow_pickle=True)
meta = json.loads(str(d["_metadata_json"]))
fps = int(meta.get("fps", 50))
dof_names = list(meta["dof_names"])

root_pos = d["root_pos"]
root_quat_xyzw = d["root_quat_xyzw"]
dof_pos = d["dof_pos"]
N = root_pos.shape[0]

H, W = 720, 1280
spec = mujoco.MjSpec.from_file(XML)
spec.visual.global_.offwidth = W
spec.visual.global_.offheight = H
mesh = spec.add_mesh()
mesh.name = "terrain"
mesh.file = TERRAIN_OBJ
spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH, meshname="terrain",
    rgba=[0.55, 0.5, 0.45, 1.0], contype=0, conaffinity=0,
)
model = spec.compile()
data = mujoco.MjData(model)

base_adr = model.joint("floating_base_joint").qposadr[0]
dof_adr = np.array([model.joint(n).qposadr[0] for n in dof_names], dtype=int)

renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.azimuth = 125.0
cam.elevation = -14.0
cam.distance = 3.2
opt = mujoco.MjvOption()

frames = []
for i in range(N):
    q = data.qpos
    q[base_adr : base_adr + 3] = root_pos[i]
    x, y, z, w = root_quat_xyzw[i]
    q[base_adr + 3 : base_adr + 7] = [w, x, y, z]
    q[dof_adr] = dof_pos[i]
    mujoco.mj_forward(model, data)

    cam.lookat[:] = [float(root_pos[i, 0]), float(root_pos[i, 1]), float(root_pos[i, 2]) - 0.1]
    renderer.update_scene(data, camera=cam, scene_option=opt)
    frames.append(renderer.render().copy())

proc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
     "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUT],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in frames:
    proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
proc.stdin.close()
proc.wait()
print(f"Rendered {len(frames)} frames @ {fps}fps -> {OUT}")
