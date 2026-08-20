"""VLA-facing cameras for the X2 MuJoCo model.

Isaac cannot render on this machine (Omniverse misparses the driver and refuses the RTX
renderer; see docs/PLAN_X2_DEPLOYMENT.md), so image observations come from MuJoCo, whose
offscreen EGL path works here -- the same path that rendered 500 LIBERO episodes.

Cameras are added programmatically via MjSpec so the tracked MJCF is never edited:

    agentview  world-fixed third person, auto-aimed at torso_link. VERIFIED good:
               shows the whole robot and the box, the role agentview plays in LIBERO.
    ego        on head_pitch_link, pitched down toward the workspace. Renders, but at
               the REST pose it sees only floor -- the box starts at the robot's feet
               and the head is upright. Needs tuning against real task poses, which
               requires the task loop; do not tune it against the rest pose.

Note the MJCF here (`holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml`) has nu=0 --
it is a kinematic/retargeting model, not actuated. It is the right thing for RENDERING;
driving the task needs holosoma's MuJoCo backend.
"""
from __future__ import annotations

import numpy as np
import mujoco

X2_LARGEBOX = ("/home/mtaheri/ws_AgibotX2/holosoma/src/holosoma_retargeting/"
               "holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml")


def _look_quat(pitch_deg: float) -> list[float]:
    """Quaternion for a camera looking along +x, pitched down by pitch_deg.

    MuJoCo cameras look down their own -z with +y up, which is why this builds the
    rotation explicitly rather than guessing a literal.
    """
    t = np.deg2rad(pitch_deg)
    fwd = np.array([np.cos(t), 0.0, -np.sin(t)])
    up = np.array([np.sin(t), 0.0, np.cos(t)])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    R = np.column_stack([right, np.cross(right, fwd), -fwd])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q.tolist()


def add_vla_cameras(spec, ego_pitch_deg: float = 35.0, target: str = "torso_link"):
    """Add `agentview` and `ego` to an MjSpec in place; returns it for chaining."""
    world = next(b for b in spec.bodies if b.name == "world")
    a = world.add_camera()
    a.name = "agentview"
    a.pos = [1.6, -1.6, 1.5]
    a.fovy = 55.0
    a.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    a.targetbody = target

    head = next(b for b in spec.bodies if b.name == "head_pitch_link")
    e = head.add_camera()
    e.name = "ego"
    e.pos = [0.12, 0.0, 0.06]
    e.fovy = 80.0
    e.quat = _look_quat(ego_pitch_deg)
    return spec


def load_with_cameras(xml_path: str = X2_LARGEBOX, **kw):
    """Compile the X2 model with both cameras attached."""
    return add_vla_cameras(mujoco.MjSpec.from_file(xml_path), **kw).compile()


if __name__ == "__main__":
    import imageio.v2 as iio
    m = load_with_cameras()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print("cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                       for i in range(m.ncam)])
    with mujoco.Renderer(m, height=224, width=224) as r:
        for nm in ("agentview", "ego"):
            r.update_scene(d, camera=nm)
            img = np.asarray(r.render())
            print(f"  {nm:<10} {img.shape} mean {img.mean():.1f}")
            iio.imwrite(f"x2_{nm}.png", img)
