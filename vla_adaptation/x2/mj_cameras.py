"""VLA-facing cameras for the X2 MuJoCo model.

Isaac cannot render on this machine (Omniverse misparses the driver and refuses the RTX
renderer; see docs/PLAN_X2_DEPLOYMENT.md), so image observations come from MuJoCo, whose
offscreen EGL path works here -- the same path that rendered 500 LIBERO episodes.

Cameras are added programmatically via MjSpec so the tracked MJCF is never edited:

    agentview  world-fixed third person, auto-aimed at torso_link. Shows the whole robot
               and the box -- the role agentview plays in LIBERO.
    ego        on head_pitch_link, 3 cm forward of the face at eye height, pitched 50 deg
               down. Sees the box in 100% of task frames with both grippers on its edges.

Both are TUNED AND VERIFIED against the real box-pickup reference motion (`mj_motion.py`),
not against the rest pose -- at rest the box lies at the robot's feet and the head is
upright, so any rest-pose judgement of the ego view is meaningless. `tune_ego_camera.py`
reproduces the search.

Why these ego numbers, since two cheaper criteria both pick badly here:
  * frustum-only scoring likes pitch 50-55 at z=+0.10, which mounts the camera ON TOP OF
    THE SKULL -- the bottom 40% of every frame is then the robot's own head.
  * "keep both hands in view" is unachievable, not a tuning failure: the hands grip the
    box being carried, so the box occludes them for ~85% of the task. Ray casting shows
    this; frustum tests wrongly score it 100%.
  Scoring on segmentation renders instead -- what actually lands on the image -- gives
  pitch 50 / pos [0.10, 0, 0.04]: box visible in 31/31 sampled frames at 32% of pixels,
  centroid 0.18 from image centre, only 5% of pixels the robot's own body. The previous
  pitch-35 mount also always saw the box, but pushed to the frame edge (offset 0.31) at
  23% of pixels, and showed the grippers far less.

Requires mujoco, which is NOT in the hssim conda env; use the standalone install:
    PYTHONPATH=/home/mtaheri/.holosoma_deps/mjrender MUJOCO_GL=egl python3 ...

Note the MJCF here (`holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml`) has nu=0 --
it is a kinematic/retargeting model, not actuated. It is the right thing for RENDERING
and for replaying recorded motion; driving a closed-loop task needs holosoma's MuJoCo
backend.
"""
from __future__ import annotations

import numpy as np
import mujoco

X2_LARGEBOX = ("/home/mtaheri/ws_AgibotX2/holosoma/src/holosoma_retargeting/"
               "holosoma_retargeting/models/x2/x2_31dof_w_largebox.xml")

# Tuned on the box-pickup reference motion; see the module docstring and tune_ego_camera.py.
EGO_POS = (0.10, 0.0, 0.04)   # head_pitch_link frame: 3 cm clear of the face mesh, eye height
EGO_PITCH_DEG = 50.0
EGO_FOVY = 80.0


def _look_quat(pitch_deg: float) -> list[float]:
    """Quaternion for a camera looking along +x, pitched down by pitch_deg.

    MuJoCo cameras look down their own -z with +y up, which is why this builds the
    rotation explicitly rather than guessing a literal. head_pitch_link's local +x is
    the robot's forward direction (verified: +x . head->box = +0.64 over the task).
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


def add_vla_cameras(spec, ego_pitch_deg: float = EGO_PITCH_DEG, ego_pos=EGO_POS,
                    ego_fovy: float = EGO_FOVY, target: str = "torso_link"):
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
    e.pos = list(ego_pos)
    e.fovy = ego_fovy
    e.quat = _look_quat(ego_pitch_deg)
    return spec


def load_with_cameras(xml_path: str = X2_LARGEBOX, **kw):
    """Compile the X2 model with both cameras attached."""
    return add_vla_cameras(mujoco.MjSpec.from_file(xml_path), **kw).compile()


if __name__ == "__main__":
    import imageio.v2 as iio
    from mj_motion import load_motion, qpos_at, phases

    m = load_with_cameras()
    d = mujoco.MjData(m)
    mo = load_motion()
    ph = phases(mo)
    print("cameras:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                       for i in range(m.ncam)])
    print("task phases:", ph)
    with mujoco.Renderer(m, height=224, width=224) as r:
        for nm in ("agentview", "ego"):
            tiles = []
            for t in ph.values():
                d.qpos[:] = qpos_at(mo, t)
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=nm)
                tiles.append(np.asarray(r.render()))
            sheet = np.concatenate(tiles, axis=1)
            print(f"  {nm:<10} {sheet.shape} mean {sheet.mean():.1f}")
            iio.imwrite(f"x2_{nm}.png", sheet)
