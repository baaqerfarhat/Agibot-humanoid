"""Render two recorded rollouts side by side into one MP4 (MuJoCo offscreen / EGL).

  python render_side_by_side.py left.npz "LEFT LABEL" right.npz "RIGHT LABEL" out.mp4

The shorter clip holds its last frame (dimmed, marked FALLEN) so both panels stay in
sync on the wall clock.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import json
import sys

import mujoco
import numpy as np

XML = "/home/baaqer/baaqer_ws/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml"

L_NPZ, L_LABEL, R_NPZ, R_LABEL, OUT = sys.argv[1:6]

PANEL_W, PANEL_H = 640, 720


def load(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["_metadata_json"]))
    return {
        "meta": meta,
        "fps": int(meta.get("fps", 50)),
        "dof_names": list(meta["dof_names"]),
        "root_pos": d["root_pos"],
        "root_quat_xyzw": d["root_quat_xyzw"],
        "dof_pos": d["dof_pos"],
        "obj_pos": d["object_pos"] if "object_pos" in d else None,
        "obj_quat": d["object_quat_wxyz"] if "object_quat_wxyz" in d else None,
    }


left, right = load(L_NPZ), load(R_NPZ)
fps = left["fps"]
n_frames = max(len(left["root_pos"]), len(right["root_pos"]))

spec = mujoco.MjSpec.from_file(XML)
spec.visual.global_.offwidth = PANEL_W
spec.visual.global_.offheight = PANEL_H
model = spec.compile()
data = mujoco.MjData(model)

base_adr = model.joint("floating_base_joint").qposadr[0]
box_jnt = int(model.body("largebox_link").jntadr[0])
box_adr = int(model.jnt_qposadr[box_jnt])

renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)
opt = mujoco.MjvOption()


def render_panel(clip, i):
    n = len(clip["root_pos"])
    fallen = i >= n
    j = min(i, n - 1)

    dof_adr = np.array([model.joint(nm).qposadr[0] for nm in clip["dof_names"]], dtype=int)
    q = data.qpos
    q[base_adr : base_adr + 3] = clip["root_pos"][j]
    x, y, z, w = clip["root_quat_xyzw"][j]
    q[base_adr + 3 : base_adr + 7] = [w, x, y, z]
    q[dof_adr] = clip["dof_pos"][j]
    if clip["obj_pos"] is not None:
        q[box_adr : box_adr + 3] = clip["obj_pos"][j]
        q[box_adr + 3 : box_adr + 7] = clip["obj_quat"][j]
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135.0, -18.0, 2.8
    cam.lookat[0] = float(clip["root_pos"][j, 0])
    cam.lookat[1] = float(clip["root_pos"][j, 1])
    cam.lookat[2] = 0.45
    renderer.update_scene(data, camera=cam, scene_option=opt)
    img = renderer.render().copy()
    if fallen:
        img = (img * 0.45).astype(np.uint8)
    return img, fallen


try:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = small = ImageFont.load_default()
except ImportError:
    Image = None


def annotate(img, label, t, fallen, n_steps):
    if Image is None:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, PANEL_W, 78], fill=(15, 15, 20))
    dr.text((14, 8), label, font=font, fill=(255, 255, 255))
    status = f"FALLEN at {n_steps / fps:.2f}s" if fallen else f"t = {t:.2f}s"
    dr.text((14, 44), status, font=small,
            fill=(255, 90, 90) if fallen else (140, 230, 140))
    return np.asarray(im)


frames = []
for i in range(n_frames):
    li, lf = render_panel(left, i)
    ri, rf = render_panel(right, i)
    t = i / fps
    li = annotate(li, L_LABEL, t, lf, len(left["root_pos"]))
    ri = annotate(ri, R_LABEL, t, rf, len(right["root_pos"]))
    frames.append(np.hstack([li, ri]))

print(f"Rendered {len(frames)} frames at {fps} fps -> {OUT}")

import imageio.v2 as imageio

imageio.mimwrite(OUT, frames, fps=fps, quality=8, macro_block_size=8)
print("Wrote MP4")
