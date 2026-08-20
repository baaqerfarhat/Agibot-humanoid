"""Render reference motion npz files to MP4, optionally two side by side.

usage: render_reference_compare.py OUT.mp4 LABEL=motion.npz [LABEL=motion.npz]
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import sys

import mujoco
import numpy as np

XML = "/home/baaqer/baaqer_ws/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml"
H, W = 720, 900
SHARED = {}  # camera framing, shared across panes so clips are comparable


def load(path):
    d = np.load(path, allow_pickle=True)
    qp = d["joint_pos"]
    return {
        "fps": int(np.atleast_1d(d["fps"])[0]),
        "root_pos": qp[:, 0:3],
        "root_quat": qp[:, 3:7],  # wxyz
        "dof": qp[:, 7:],
        "names": [str(x) for x in d["joint_names"]],
        "obj_pos": d["object_pos_w"],
        "obj_quat": d["object_quat_w"],
        "n": len(qp),
    }


def render(m, label):
    spec = mujoco.MjSpec.from_file(XML)
    spec.visual.global_.offwidth = W
    spec.visual.global_.offheight = H
    model = spec.compile()
    data = mujoco.MjData(model)
    base = model.joint("floating_base_joint").qposadr[0]
    adr = np.array([model.joint(n).qposadr[0] for n in m["names"]], dtype=int)
    bjnt = int(model.body("largebox_link").jntadr[0])
    badr = int(model.jnt_qposadr[bjnt])

    r = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    # side-on: the robot faces -y, so looking down +x puts the squat, the reach and
    # the lift all in the image plane instead of hiding them behind the box
    cam.azimuth = float(os.environ.get("CAM_AZ", 2.0))
    cam.elevation = float(os.environ.get("CAM_EL", -8.0))
    cam.distance = float(os.environ.get("CAM_D", 3.4))
    opt = mujoco.MjvOption()
    out = []
    for i in range(m["n"]):
        q = data.qpos
        q[base : base + 3] = m["root_pos"][i]
        q[base + 3 : base + 7] = m["root_quat"][i]
        q[adr] = m["dof"][i]
        q[badr : badr + 3] = m["obj_pos"][i]
        q[badr + 3 : badr + 7] = m["obj_quat"][i]
        mujoco.mj_forward(model, data)
        if os.environ.get("CAM_STATIC"):
            # The robot side-steps ~1.5 m, so panning with it hides the walk entirely.
            # One lookat shared by every pane, or the two clips get framed differently
            # and it is impossible to compare them.
            SHARED.setdefault(
                "lookat",
                [float(m["root_pos"][:, 0].mean()), float(m["root_pos"][:, 1].mean()), 0.50],
            )
            cam.lookat[:] = SHARED["lookat"]
        else:
            cam.lookat[:] = [
                float(m["root_pos"][i, 0]),
                float((m["root_pos"][i, 1] + m["obj_pos"][i, 1]) / 2),
                0.50,
            ]
        r.update_scene(data, camera=cam, scene_option=opt)
        out.append(r.render().copy())
    print(f"  {label}: {len(out)} frames @ {m['fps']} fps ({m['n']/m['fps']:.2f}s)")
    return out


def banner(img, text, sub):
    try:
        import cv2

        cv2.rectangle(img, (0, 0), (W, 62), (18, 18, 18), -1)
        cv2.putText(img, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, sub, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 200, 255), 1)
    except Exception:
        pass
    return img


def main():
    out_path = sys.argv[1]
    clips = []
    for a in sys.argv[2:]:
        label, path = a.split("=", 1)
        m = load(path)
        clips.append((label, m, render(m, label)))

    fps = clips[0][1]["fps"]
    n = max(len(c[2]) for c in clips)
    frames = []
    for i in range(n):
        panes = []
        for label, m, fr in clips:
            img = fr[min(i, len(fr) - 1)].copy()
            done = "" if i < len(fr) else "  [end]"
            panes.append(banner(img, label + done, f"t = {min(i,len(fr)-1)/fps:5.2f} s"))
        frames.append(np.concatenate(panes, axis=1) if len(panes) > 1 else panes[0])

    import imageio.v2 as imageio

    imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=8)
    print(f"wrote {out_path}  ({len(frames)} frames, {len(frames)/fps:.2f}s)")


if __name__ == "__main__":
    main()
