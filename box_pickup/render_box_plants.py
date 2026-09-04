"""Three panels of the same box policy: the reference, Isaac Sim, and mjlab.

All three are drawn kinematically in one MuJoCo scene so the renderer, camera and
lighting are shared and the only thing that differs between panels is the trajectory.
The Isaac panel is Isaac's own recorded state, not a re-simulation -- Kit's RTX camera
will not run headless on this box, so its motion is replayed here.

    MUJOCO_GL=egl .venv/bin/python render_box_plants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_box_mjlab import CLIP, DEFAULT_POLICY, Policy, build_scene  # noqa: E402

ROLL = HERE / "sim_rollouts"
OUT = HERE / "videos" / "box_v19_reference_vs_isaac_vs_mjlab.mp4"
W, H = 480, 560


def font(sz):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def label(img, title, sub, colour):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 52], fill=(20, 20, 24))
    d.text((10, 5), title, font=font(22), fill=colour)
    d.text((10, 30), sub, font=font(15), fill=(205, 205, 210))
    return np.asarray(im)


def main() -> None:
    pol = Policy(DEFAULT_POLICY)
    jn = list(pol.meta["joint_names"])
    kp = np.asarray(pol.meta["joint_stiffness"])
    kd = np.asarray(pol.meta["joint_damping"])
    eff = np.asarray(pol.meta["joint_effort_limit"])
    model = build_scene(jn, kp, kd, eff, actuators="position")
    model.vis.global_.offwidth = W  # default offscreen buffer is smaller than a panel
    model.vis.global_.offheight = H
    data = mujoco.MjData(model)
    qadr = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in jn
    ])
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # ---- reference ----------------------------------------------------------
    clip = np.load(CLIP, allow_pickle=True)
    cq = np.asarray(clip["joint_pos"])
    cjn = [str(x) for x in clip["joint_names"]]
    perm = [cjn.index(n) for n in jn]
    ref = {
        "pos": cq[:, 0:3],
        "quat_wxyz": cq[:, 3:7],
        "q": cq[:, 7:][:, perm],
    }

    # ---- Isaac --------------------------------------------------------------
    I = np.load(ROLL / "x2_box_walk_retimed_v19_iter85500_rollout.npz", allow_pickle=True)
    iq = I["root_quat_xyzw"]
    isaac = {
        "pos": I["root_pos"],
        "quat_wxyz": np.stack([iq[:, 3], iq[:, 0], iq[:, 1], iq[:, 2]], 1),
        "q": I["dof_pos"],
    }

    # ---- mjlab --------------------------------------------------------------
    M = np.load(ROLL / "mjlab_box_v19_pos_rollout.npz", allow_pickle=True)
    mjn = list(M["joint_names"])
    mp_ = [mjn.index(n) for n in jn]
    mq = M["root_quat_xyzw"]
    mjlab = {
        "pos": M["root_pos"],
        "quat_wxyz": np.stack([mq[:, 3], mq[:, 0], mq[:, 1], mq[:, 2]], 1),
        "q": M["joint_pos"][:, mp_],
    }

    panels = [
        (ref, "REFERENCE", "the clip the policy tracks", (140, 200, 255)),
        (isaac, "ISAAC SIM", "trained here - completes", (130, 235, 150)),
        (mjlab, "MJLAB", "same policy - falls", (255, 120, 120)),
    ]
    n = min(len(p[0]["q"]) for p in panels)

    # Nothing but the feet collides in this plant, so once the robot is down the
    # body keeps sinking through the floor and the panel goes empty. Freeze it at
    # the fall instead: the fall is the result, the sinking is just bookkeeping.
    zt = mjlab["pos"][:, 2]
    fell = int(np.argmax(zt < 0.25)) if (zt < 0.25).any() else None
    if fell:
        print(f"mjlab fell at frame {fell} (t={fell/50:.2f} s); panel frozen there")

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.4, -14, 128
    frames = []
    for f in range(n):
        row = []
        for traj, title, sub, colour in panels:
            g = f
            note = ""
            if title == "MJLAB" and fell and f >= fell:
                g = fell
                note = f"   FELL at {fell/50:.1f} s"
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = traj["pos"][g]
            data.qpos[3:7] = traj["quat_wxyz"][g]
            data.qpos[qadr] = traj["q"][g]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [traj["pos"][g][0], traj["pos"][g][1], 0.55]
            renderer.update_scene(data, camera=cam)
            img = renderer.render().copy()
            z = data.xpos[pelvis][2]
            row.append(label(img, title, f"{sub}   pelvis {z:+.2f} m{note}", colour))
        frames.append(np.concatenate(row, axis=1))
        if f % 100 == 0:
            print(f"  frame {f}/{n}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(OUT, frames, fps=50, quality=8)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
