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

# The recording runs one frame past the end of the episode, so the last sample is
# the reset pose in a different part of the arena. Left in, it shows up as a 2 m
# teleport that swamps every acceleration-based measure below.
jump = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
tele = np.nonzero(jump > 0.1)[0]
if len(tele):
    N = int(tele[0]) + 1
    print(f"dropping {root_pos.shape[0] - N} frame(s) after the episode reset at t={N/fps:.2f}s")

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

renderer = mujoco.Renderer(model, height=H, width=W, max_geom=20000)

cam = mujoco.MjvCamera()
cam.azimuth = 135.0
cam.elevation = -18.0
cam.distance = 2.8

opt = mujoco.MjvOption()

# ---- support polygon overlay -------------------------------------------------
# Every sphere the feet can actually stand on: the six foot collision spheres plus
# the five ankle spheres per side. A foot counts towards the polygon when its
# lowest sphere is on the floor, and the polygon is the convex hull of whichever
# spheres are down -- so it collapses to one foot during a step, exactly as the
# real support does.
CONTACT_H = 0.006  # m: how close a sphere's underside must be to count as down
GROUND_Z = 0.004  # m: draw the polygon just above the floor so it is not z-fought
BOX_CARRIED = 0.30  # m: above this the box's mass rides on the robot

foot_geoms = {"left": [], "right": []}
for g in range(model.ngeom):
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
        continue
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
    if "ankle_roll" not in body:
        continue
    for side in foot_geoms:
        if body.startswith(side):
            foot_geoms[side].append(g)

box_bid = model.body("largebox_link").id
robot_bids = np.array(
    [b for b in range(1, model.nbody) if b != box_bid and "largebox" not in
     (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
)
robot_m = model.body_mass[robot_bids]
box_m = float(model.body_mass[box_bid])


def hull_2d(pts):
    """Counter-clockwise convex hull (monotone chain), so edge normals point out."""
    p = sorted(map(tuple, np.round(pts, 6)))
    if len(p) < 3:
        return np.array(p)

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def support_margin(hull, com_xy):
    """Distance from the CoM to the nearest edge; negative once it is outside."""
    if len(hull) < 3:
        return -np.inf
    best = np.inf
    for k in range(len(hull)):
        a, b = hull[k], hull[(k + 1) % len(hull)]
        e = b - a
        nrm = np.array([e[1], -e[0]])
        n = np.linalg.norm(nrm)
        if n < 1e-9:
            continue
        best = min(best, -float(nrm @ (com_xy - a)) / n)
    return best


def add_line(scene, p0, p1, rgba, r=0.006):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, p0, p1)
    scene.ngeom += 1


def add_marker(scene, pos, rgba, r=0.022):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), np.asarray(pos, float),
        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def pose(i):
    q = data.qpos
    q[base_adr : base_adr + 3] = root_pos[i]
    x, y, z, w = root_quat_xyzw[i]
    q[base_adr + 3 : base_adr + 7] = [w, x, y, z]
    q[dof_adr] = dof_pos[i]
    if obj_pos is not None:
        q[box_adr : box_adr + 3] = obj_pos[i]
        q[box_adr + 3 : box_adr + 7] = obj_quat_wxyz[i]
    mujoco.mj_forward(model, data)


# Pass one: the geometry. The ZMP needs the CoM's acceleration, so nothing can be
# drawn until the whole CoM path is known.
coms = np.zeros((N, 3))
hulls = []
for i in range(N):
    pose(i)
    down = []
    for gs in foot_geoms.values():
        if min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in gs) < CONTACT_H:
            down += [data.geom_xpos[g][:2] for g in gs]
    hulls.append(hull_2d(np.array(down)) if len(down) >= 3 else np.empty((0, 2)))

    c = (data.xipos[robot_bids] * robot_m[:, None]).sum(0)
    mass = robot_m.sum()
    if obj_pos is not None and obj_pos[i][2] > BOX_CARRIED:
        c = c + data.xipos[box_bid] * box_m  # the box only loads the feet once lifted
        mass += box_m
    coms[i] = c / mass

# The cart-table ZMP. Differencing raw positions at 50 Hz is mostly noise, so the
# CoM path is smoothed first -- the ZMP is a trend, not a per-frame reading.
try:
    from scipy.ndimage import gaussian_filter1d

    sm = gaussian_filter1d(coms, 3.0, axis=0, mode="nearest")
except ImportError:  # the fallback interpreter has no scipy
    k = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    k /= k.sum()
    pad = np.pad(coms, ((9, 9), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, c], k, "valid") for c in range(3)], axis=1)
acc = np.gradient(np.gradient(sm, 1.0 / fps, axis=0), 1.0 / fps, axis=0)
zmp = sm[:, :2] - sm[:, 2:3] * acc[:, :2] / np.maximum(acc[:, 2:3] + 9.81, 1.0)

com_margins = np.array([support_margin(hulls[i], coms[i, :2]) for i in range(N)])
zmp_margins = np.array([support_margin(hulls[i], zmp[i]) for i in range(N)])

frames = []
for i in range(N):
    pose(i)
    m = zmp_margins[i]
    if m > 0.04:
        rgba = (0.15, 0.90, 0.25, 0.85)  # comfortably supported
    elif m > 0.0:
        rgba = (1.00, 0.75, 0.10, 0.90)  # on the edge
    else:
        rgba = (1.00, 0.15, 0.15, 0.95)  # ZMP outside: tipping

    cam.lookat[0] = float(root_pos[i, 0])
    cam.lookat[1] = float(root_pos[i, 1])
    cam.lookat[2] = 0.45
    renderer.update_scene(data, camera=cam, scene_option=opt)
    sc = renderer.scene
    hull = hulls[i]
    for k in range(len(hull)):
        a = np.array([hull[k][0], hull[k][1], GROUND_Z])
        b = np.array([hull[(k + 1) % len(hull)][0], hull[(k + 1) % len(hull)][1], GROUND_Z])
        add_line(sc, a, b, rgba)
    com = coms[i]
    add_marker(sc, com, (0.25, 0.55, 1.0, 0.9))
    add_line(sc, com, np.array([com[0], com[1], GROUND_Z]), (0.25, 0.55, 1.0, 0.5), r=0.004)
    add_marker(sc, np.array([com[0], com[1], GROUND_Z]), (0.25, 0.55, 1.0, 0.9), r=0.026)
    add_marker(sc, np.array([zmp[i, 0], zmp[i, 1], GROUND_Z]), rgba, r=0.032)
    frames.append(renderer.render().copy())

for nm, mg in (("CoM (static)", com_margins), ("ZMP (dynamic)", zmp_margins)):
    fin = mg[np.isfinite(mg)]
    print(f"{nm:14s} margin: min {fin.min()*1000:6.0f} mm, outside on {int((mg <= 0).sum()):3d}/{N} frames")
print(f"both feet off the ground on {int((~np.isfinite(zmp_margins)).sum())} frames")

bad = zmp_margins <= 0
edges = np.diff(np.r_[0, bad.astype(int), 0])
print("\nspells where the ZMP leaves the polygon:")
for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
    print(f"  t {s/fps:5.2f} - {e/fps:5.2f} s  ({(e-s)/fps:4.2f} s)  worst {zmp_margins[s:e].min()*1000:5.0f} mm")

try:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = small = ImageFont.load_default()
    for i, f in enumerate(frames):
        m, cm = zmp_margins[i], com_margins[i]
        if not np.isfinite(m):
            txt, col = "AIRBORNE - no support polygon", (255, 60, 60)
        elif m > 0.04:
            txt, col = f"ZMP inside support  +{m*1000:.0f} mm", (60, 230, 60)
        elif m > 0:
            txt, col = f"ZMP near edge  +{m*1000:.0f} mm", (255, 190, 30)
        else:
            txt, col = f"ZMP OUTSIDE support  {m*1000:.0f} mm", (255, 60, 60)
        img = Image.fromarray(f)
        dr = ImageDraw.Draw(img)
        dr.text((24, 22), txt, fill=col, font=font)
        dr.text((24, 58), f"t = {i/fps:5.2f} s     CoM margin "
                          f"{'--' if not np.isfinite(cm) else f'{cm*1000:+.0f} mm'}",
                fill=(220, 220, 220), font=small)
        dr.text((24, 84), "big dot = ZMP (coloured)     blue dot = CoM",
                fill=(170, 170, 170), font=small)
        frames[i] = np.asarray(img)
except Exception as e:  # noqa: BLE001
    print(f"(no text overlay: {e})")

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
