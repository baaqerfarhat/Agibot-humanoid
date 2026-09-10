"""Check the OmniRetarget clip against the mjlab X2 model before any RL runs.

Two separate questions get answered here, and they are easy to conflate:

  1. Did the conversion land? Every body pose in the clip was authored by the
     retargeter's own kinematics. Driving MuJoCo from the clip's joint angles has to
     reproduce those same poses. If it does not, the joint order, the quaternion
     order or the root frame is wrong, and nothing downstream is trustworthy.
  2. Is the motion physically sane for this robot? Joint limits, inverse-dynamics
     torque against the actuator limits, foot penetration, palm placement relative
     to the box.

The first is a bug hunt with a right answer. The second is a survey -- RL is expected
to fix up some of it, so violations here are reported, not repaired.

    cd ~/baaqer_ws/mjlab && uv run python <this> [--render]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
BOX_SIZE = np.array([0.4712, 0.4587, 0.4079])
BOX_OFFSET = np.array([0.0015, -0.0007, 0.0058])
PALM_OFFSET = np.array([0.01, 0.0, -0.10])
PALM_RADIUS = 0.05
HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz -> 3x3."""
    m = np.zeros((3, 3))
    mujoco.mju_quat2Mat(m.reshape(-1), q)
    return m


def main() -> None:
    global CLIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--clip", default=str(CLIP))
    ap.add_argument("--out", default="videos/reference_in_mjlab.mp4")
    args = ap.parse_args()
    CLIP = Path(args.clip)

    from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg

    spec = get_x2_robot_cfg().spec_fn()
    model = spec.compile()
    data = mujoco.MjData(model)

    clip = np.load(CLIP, allow_pickle=True)
    cj = [str(x) for x in clip["joint_names"]]
    cb = [str(x) for x in clip["body_names"]]
    fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    T = clip["joint_pos"].shape[0]

    mj_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    hinge = [j for j in mj_joints if j in cj]
    print("=" * 78)
    print("1. CONVERSION")
    print("=" * 78)
    print(f"clip {CLIP.name}: {T} frames @ {fps:g} Hz = {T / fps:.2f} s")
    print(f"clip joints {len(cj)}, model hinge joints {len(hinge)}, "
          f"order identical: {hinge == cj}")

    jperm = [cj.index(j) for j in hinge]
    qadr = [model.jnt_qposadr[mj_joints.index(j)] for j in hinge]
    vadr = [model.jnt_dofadr[mj_joints.index(j)] for j in hinge]

    jp_all = np.asarray(clip["joint_pos"], np.float64)
    qj = jp_all[:, 7:][:, jperm]              # 31 hinge angles, model order
    root_pos = jp_all[:, 0:3]
    root_quat = jp_all[:, 3:7]                # convention TBD
    bpos = np.asarray(clip["body_pos_w"], np.float64)
    bquat = np.asarray(clip["body_quat_w"], np.float64)

    # Which body does the freejoint belong to, and is the stored root quaternion
    # wxyz or xyzw? Settle both by driving FK and scoring against the clip's own
    # body positions -- the retargeter and MuJoCo share a kinematic tree, so the
    # correct choice reproduces every body to numerical precision.
    shared = [b for b in cb if b in [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)]]
    mb_idx = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in shared]
    cb_idx = [cb.index(b) for b in shared]

    def fk_error(quat_mode: str, n: int = 40) -> float:
        errs = []
        for t in np.linspace(0, T - 1, n).astype(int):
            data.qpos[:] = 0.0
            data.qpos[0:3] = root_pos[t]
            q = root_quat[t]
            data.qpos[3:7] = q if quat_mode == "wxyz" else np.roll(q, 1)
            data.qpos[qadr] = qj[t]
            mujoco.mj_forward(model, data)
            errs.append(np.abs(data.xpos[mb_idx] - bpos[t][cb_idx]).max())
        return float(np.mean(errs))

    e_wxyz, e_xyzw = fk_error("wxyz"), fk_error("xyzw")
    quat_mode = "wxyz" if e_wxyz < e_xyzw else "xyzw"
    print(f"FK vs clip body_pos_w:  root quat as wxyz -> {e_wxyz * 1e3:8.3f} mm   "
          f"as xyzw -> {e_xyzw * 1e3:8.3f} mm")
    print(f"=> root quaternion is {quat_mode}, "
          f"reproduction error {min(e_wxyz, e_xyzw) * 1e3:.4f} mm")
    if min(e_wxyz, e_xyzw) > 1e-3:
        print("!! FK does not reproduce the clip; conversion is wrong, stop here")

    # Same question for the body quaternions, scored on orientation.
    def quat_err(mode: str, n: int = 40) -> float:
        errs = []
        for t in np.linspace(0, T - 1, n).astype(int):
            data.qpos[:] = 0.0
            data.qpos[0:3] = root_pos[t]
            data.qpos[3:7] = (root_quat[t] if quat_mode == "wxyz"
                              else np.roll(root_quat[t], 1))
            data.qpos[qadr] = qj[t]
            mujoco.mj_forward(model, data)
            for mi, ci in zip(mb_idx, cb_idx):
                ref = bquat[t][ci]
                ref = ref if mode == "wxyz" else np.roll(ref, 1)
                d = np.zeros(4)
                mujoco.mju_mulQuat(d, data.xquat[mi], np.array(
                    [ref[0], -ref[1], -ref[2], -ref[3]]))
                errs.append(2 * np.arcsin(min(1.0, np.linalg.norm(d[1:]))))
        return float(np.mean(errs))

    b_wxyz, b_xyzw = quat_err("wxyz"), quat_err("xyzw")
    bmode = "wxyz" if b_wxyz < b_xyzw else "xyzw"
    print(f"body_quat_w as wxyz -> {np.degrees(b_wxyz):.4f} deg   "
          f"as xyzw -> {np.degrees(b_xyzw):.4f} deg   => {bmode}")

    # ---------------------------------------------------------------- physics
    print()
    print("=" * 78)
    print("2. PHYSICAL FEASIBILITY ON THE MJLAB MODEL")
    print("=" * 78)

    lo = model.jnt_range[[mj_joints.index(j) for j in hinge], 0]
    hi = model.jnt_range[[mj_joints.index(j) for j in hinge], 1]
    over = np.maximum(qj - hi, 0.0) + np.maximum(lo - qj, 0.0)
    nviol = int((over > 1e-4).any(axis=1).sum())
    print(f"joint limits: {nviol}/{T} frames violate, "
          f"worst {np.degrees(over.max()):.2f} deg")
    if nviol:
        for k in np.argsort(-over.max(axis=0))[:5]:
            if over[:, k].max() > 1e-4:
                print(f"    {hinge[k]:<28} worst {np.degrees(over[:, k].max()):6.2f} deg")

    # Inverse dynamics: what torque would the reference demand of each actuator?
    qpos = np.zeros((T, model.nq))
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat if quat_mode == "wxyz" else np.roll(root_quat, 1, axis=1)
    qpos[:, qadr] = qj
    qvel = np.zeros((T, model.nv))
    qvel[1:-1, vadr] = (qj[2:] - qj[:-2]) / (2 * dt)
    qvel[1:-1, 0:3] = (root_pos[2:] - root_pos[:-2]) / (2 * dt)
    qacc = np.zeros((T, model.nv))
    qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)

    tau = np.zeros((T, len(hinge)))
    for t in range(T):
        data.qpos[:] = qpos[t]
        data.qvel[:] = qvel[t]
        data.qacc[:] = qacc[t]
        mujoco.mj_inverse(model, data)
        tau[t] = data.qfrc_inverse[vadr]

    from mjlab.asset_zoo.robots.x2.x2_constants import X2_ARTICULATION
    import re
    eff = np.zeros(len(hinge))
    for a in X2_ARTICULATION.actuators:
        for pat in a.target_names_expr:
            for k, j in enumerate(hinge):
                if re.fullmatch(pat, j):
                    eff[k] = a.effort_limit
    frac = np.abs(tau).max(axis=0) / eff
    print(f"inverse-dynamics torque vs mjlab effort limits "
          f"(free-flying root, so legs read high):")
    for k in np.argsort(-frac)[:8]:
        flag = "  OVER" if frac[k] > 1.0 else ""
        print(f"    {hinge[k]:<28} peak {np.abs(tau[:, k]).max():7.1f} Nm / "
              f"{eff[k]:5.1f} = {frac[k]:5.2f}{flag}")

    # Feet, pelvis, palms, box.
    pi = cb.index("pelvis")
    print()
    print(f"pelvis height: min {bpos[:, pi, 2].min():.3f} "
          f"max {bpos[:, pi, 2].max():.3f} m")
    for f in FEET:
        fi = cb.index(f)
        print(f"{f:<26} z min {bpos[:, fi, 2].min():+.4f}  max {bpos[:, fi, 2].max():+.4f}")

    obj = np.asarray(clip["object_pos_w"], np.float64)
    oq = np.asarray(clip["object_quat_w"], np.float64)
    print(f"box centre z: {obj[:, 2].min():.3f} .. {obj[:, 2].max():.3f} m "
          f"(lift {obj[:, 2].max() - obj[:, 2].min():.3f} m)")
    print(f"box xy travel: {np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1).max():.3f} m")

    # Palm sphere centre, and its signed distance to the box surface.
    half = BOX_SIZE / 2
    print()
    print("palm (wrist + offset, r=50 mm) vs box surface; negative = inside:")
    for h in HANDS:
        hi_ = cb.index(h)
        gap = np.zeros(T)
        pz = np.zeros(T)
        for t in range(T):
            R = quat_to_mat(bquat[t][hi_] if bmode == "wxyz"
                            else np.roll(bquat[t][hi_], 1))
            p = bpos[t][hi_] + R @ PALM_OFFSET
            pz[t] = p[2]
            Rb = quat_to_mat(oq[t] if bmode == "wxyz" else np.roll(oq[t], 1))
            local = Rb.T @ (p - (obj[t] + Rb @ BOX_OFFSET))
            d = np.abs(local) - half
            outside = np.linalg.norm(np.maximum(d, 0.0))
            gap[t] = (outside if outside > 0 else d.max()) - PALM_RADIUS
        carry = gap < 0.02
        print(f"  {h:<26} floor clearance min {pz.min() - PALM_RADIUS:+.3f} m")
        print(f"  {'':<26} surface gap: min {gap.min() * 1e3:+7.1f} mm  "
              f"frames touching/inside {int(carry.sum())}/{T}")

    if args.render:
        render(get_x2_robot_cfg().spec_fn(), qpos, obj, oq, Path(args.out), fps)


def add_scenery(spec) -> None:
    """Floor plus a kinematically-driven box, so the render shows the grasp."""
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300, height=300, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
    )
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5],
                      texuniform=True, reflectance=0.2)
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05], material="grid", contype=0, conaffinity=0,
    )
    box = spec.worldbody.add_body(name="ref_box")
    box.add_freejoint()
    box.add_geom(
        name="ref_box_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BOX_SIZE / 2, pos=BOX_OFFSET, mass=1.0,
        contype=0, conaffinity=0, rgba=[0.80, 0.60, 0.35, 0.85],
    )


def render(spec, qpos, obj, oq, out: Path, fps: float) -> None:
    import imageio.v2 as imageio

    add_scenery(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    nq_robot = qpos.shape[1]

    W, H = 640, 480
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    r = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.4, -12.0, 135.0
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as w:
        for t in range(qpos.shape[0]):
            data.qpos[:nq_robot] = qpos[t]
            data.qpos[nq_robot:nq_robot + 3] = obj[t]
            data.qpos[nq_robot + 3:nq_robot + 7] = oq[t]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [qpos[t, 0], qpos[t, 1], 0.5]
            r.update_scene(data, cam)
            w.append_data(r.render())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
