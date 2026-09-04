"""Roll the exported box policy on the mjlab X2 plant, through the deploy obs pipeline.

The point is a controlled plant swap. The squat comparison in squat/MJLAB_VS_ISAAC.md
showed the same npz standing up in mjlab and falling over in Isaac, and pinned the
difference on the collision model: holosoma's X2 carries full-body URDF colliders with
self-collision on and reports multi-kN torso/wrist/knee forces at t=0 while the robot
is merely standing, which mjlab never sees. This runs the box policy through the same
test.

Everything except the plant is held to what the robot actually does. Observations are
built by the deploy script's own layout -- actions, base_ang_vel, dof_pos, dof_vel,
motion_command, motion_ref_ori_b, alphabetical, 164 wide -- and the torque is
holosoma's explicit PD clipped to the effort limit, not MuJoCo's implicit position
servo. So the actuator model matches Isaac and only the contact model differs.

Two honest gaps. There is no box: the policy is blind to it, so the observations are
unaffected, but the carried mass is not simulated, and the hardware runs this is
compared against were also the no-box safety trials. And base_ang_vel is the true
pelvis rate rather than the deploy-time reconstruction from the torso IMU, which is
what training saw; feeding the reconstruction here would confound the plant question
with the estimator.

    MUJOCO_GL=egl .venv/bin/python run_box_mjlab.py [policy.npz] [out_prefix]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
X2_XML = Path(
    "/home/baaqer/baaqer_ws/mjlab/src/mjlab/asset_zoo/robots/x2/xmls/x2.xml"
)
CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
DEFAULT_POLICY = HERE / "policy" / "x2_box_policy_walk_retimed_v19_iter85500.npz"
CONTROL_HZ = 50.0
SETTLE = 0  # Isaac starts the clip at once, so any settle here would be a confound


def log(m):
    print(m, flush=True)


# ---- quaternion helpers, xyzw, copied in behaviour from the deploy script -------
def quat_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        np.float32,
    )


def quat_inv(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], np.float32)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        np.float32,
    )


def yaw_quat(q):
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32)


class Policy:
    """The exported npz: a 4-layer ELU MLP with its own obs normalisation."""

    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.meta = json.loads(str(d["meta_json"]))
        self.mean = d["mean"].astype(np.float64)
        self.std = d["std"].astype(np.float64)
        self.n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float64) for i in range(self.n)]
        self.b = [d[f"b{i}"].astype(np.float64) for i in range(self.n)]
        self.ref_q = d["ref_joint_pos"].astype(np.float32)
        self.ref_dq = d["ref_joint_vel"].astype(np.float32)
        self.ref_quat = d["ref_quat_xyzw"].astype(np.float32)

    def __call__(self, obs):
        x = (np.asarray(obs, np.float64) - self.mean) / self.std
        for i in range(self.n - 1):
            x = self.W[i] @ x + self.b[i]
            x = np.where(x > 0, x, np.expm1(np.clip(x, -30, 0)))  # ELU
        return (self.W[-1] @ x + self.b[-1]).astype(np.float32)


def build_scene(
    jn=None, kp=None, kd=None, eff=None, actuators: str = "torque"
) -> mujoco.MjModel:
    """x2.xml plus a floor, with the drive model under test.

    `actuators="position"` installs MuJoCo native position servos, which is what
    mjlab trains against and what the implicitfast integrator handles implicitly.
    `actuators="torque"` leaves the model passive so the caller can apply
    holosoma's explicit clipped PD as a generalised force, which is what Isaac and
    the robot do. Gains are holosoma's either way, so the drive model is the only
    thing that moves between the two.
    """
    spec = mujoco.MjSpec.from_file(str(X2_XML))
    for a in list(spec.actuators):  # the raw MJCF ships <motor>; mjlab strips these
        a.delete()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, 0],
        rgba=[0.55, 0.55, 0.58, 1],
        contype=1,
        conaffinity=1,
        friction=[0.6, 0.005, 0.0001],
    )
    if not spec.lights:
        spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], directional=True)

    if actuators == "position":
        for n, p, d_, e in zip(jn, kp, kd, eff):
            act = spec.add_actuator(name=f"{n}_pos", target=n)
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = p
            act.biasprm[1] = -p
            act.biasprm[2] = -d_
            act.forcerange = [-e, e]
            act.ctrlrange = [-6.28, 6.28]

    model = spec.compile()
    # mjlab's tracking sim settings: 5 ms implicitfast Newton, 4x decimation -> 50 Hz.
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 100
    model.opt.ls_iterations = 50
    return model


def main():
    pol_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_POLICY
    prefix = sys.argv[2] if len(sys.argv) > 2 else "mjlab_box_v19"
    pol = Policy(pol_path)
    meta = pol.meta
    jn = list(meta["joint_names"])
    default = np.asarray(meta["default_joint_pos"], np.float32)
    scale = np.asarray(meta["action_scale"], np.float32)
    kp = np.asarray(meta["joint_stiffness"], np.float64)
    kd = np.asarray(meta["joint_damping"], np.float64)
    eff = np.asarray(meta["joint_effort_limit"], np.float64)
    log(f"policy {pol_path.name}: obs {meta['obs_dim']}, {len(jn)} joints, "
        f"{meta['motion_frames']} frames @ {meta['motion_fps']} Hz")

    mode = sys.argv[3] if len(sys.argv) > 3 else "position"
    model = build_scene(jn, kp, kd, eff, actuators=mode)
    data = mujoco.MjData(model)
    sub = int(round((1.0 / CONTROL_HZ) / model.opt.timestep))
    log(f"mjlab plant: {model.nbody} bodies, dt {model.opt.timestep}, {sub} substeps/tick, "
        f"drive={mode} ({'MuJoCo implicit position servo' if mode == 'position' else 'holosoma explicit clipped torque'})")

    # Map the policy's joint order onto the model's qpos/qvel addresses.
    qadr, vadr = [], []
    for n in jn:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0:
            raise SystemExit(f"joint {n} not in the mjlab model")
        qadr.append(model.jnt_qposadr[j])
        vadr.append(model.jnt_dofadr[j])
    qadr, vadr = np.array(qadr), np.array(vadr)

    ncon_col = sum(
        1 for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    )
    log(f"contact-enabled geoms: {ncon_col} "
        f"({', '.join(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom) if (model.geom_contype[i] or model.geom_conaffinity[i]))[:120]}...)")

    # ---- start where the clip starts -------------------------------------------
    clip = np.load(CLIP, allow_pickle=True)
    cq = np.asarray(clip["joint_pos"])
    cjn = [str(x) for x in clip["joint_names"]]
    perm = [cjn.index(n) for n in jn]
    root0, rquat0 = cq[0, 0:3].copy(), cq[0, 3:7].copy()  # wxyz
    dof0 = cq[0, 7:][perm]

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = root0
    data.qpos[3:7] = rquat0
    data.qpos[qadr] = dof0
    mujoco.mj_forward(model, data)

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    # Align the reference heading to the robot's, exactly as deploy does at engage.
    tq = data.xquat[torso]
    yaw_off = quat_mul(
        yaw_quat(np.array([tq[1], tq[2], tq[3], tq[0]], np.float32)),
        quat_inv(yaw_quat(pol.ref_quat[0])),
    )

    T = pol.ref_q.shape[0]
    last_action = np.zeros(31, np.float32)
    logs = {k: [] for k in (
        "t", "action", "target", "joint_pos", "joint_vel", "torque",
        "root_pos", "root_quat_xyzw", "pelvis_h", "torso_h", "base_ang_vel",
        "roll", "pitch", "contact_force", "ncon")}

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 135
    frames = []

    total = SETTLE + T
    for step in range(total):
        frame = max(0, step - SETTLE)
        frame = min(frame, T - 1)

        q = data.qpos[qadr].astype(np.float32)
        dq = data.qvel[vadr].astype(np.float32)
        # base_ang_vel: pelvis rate in the pelvis frame, which is the training signal.
        R_p = data.xmat[pelvis].reshape(3, 3)
        w_pel = (R_p.T @ data.cvel[pelvis][:3]).astype(np.float32)
        tq = data.xquat[torso]
        q_torso = np.array([tq[1], tq[2], tq[3], tq[0]], np.float32)
        q_ref = quat_mul(yaw_off, pol.ref_quat[frame])
        ori6 = quat_to_mat(quat_mul(quat_inv(q_torso), q_ref))[:, :2].reshape(-1)

        obs = np.concatenate([
            last_action, w_pel, q - default, dq,
            pol.ref_q[frame], pol.ref_dq[frame], ori6,
        ]).astype(np.float32)

        action = pol(obs)
        last_action = action
        target = default + scale * action

        if mode == "position":
            data.ctrl[:] = target
            for _ in range(sub):
                mujoco.mj_step(model, data)
            tau = data.qfrc_actuator[vadr].copy()
        else:
            for _ in range(sub):
                tau = np.clip(
                    kp * (target - data.qpos[qadr]) - kd * data.qvel[vadr], -eff, eff
                )
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[vadr] = tau
                mujoco.mj_step(model, data)

        pq = data.xquat[pelvis]
        pquat = np.array([pq[1], pq[2], pq[3], pq[0]], np.float32)
        R = quat_to_mat(pquat)
        roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        pitch = float(np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1))))
        cf = float(np.abs(data.cfrc_ext).max())

        logs["t"].append(step / CONTROL_HZ)
        logs["action"].append(action)
        logs["target"].append(target.astype(np.float32))
        logs["joint_pos"].append(data.qpos[qadr].astype(np.float32))
        logs["joint_vel"].append(data.qvel[vadr].astype(np.float32))
        logs["torque"].append(tau.astype(np.float32))
        logs["root_pos"].append(data.qpos[0:3].astype(np.float32).copy())
        logs["root_quat_xyzw"].append(pquat)
        logs["pelvis_h"].append(float(data.xpos[pelvis][2]))
        logs["torso_h"].append(float(data.xpos[torso][2]))
        logs["base_ang_vel"].append(w_pel)
        logs["roll"].append(roll)
        logs["pitch"].append(pitch)
        logs["contact_force"].append(cf)
        logs["ncon"].append(int(data.ncon))

        cam.lookat[:] = data.xpos[pelvis]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

        if step % 100 == 0:
            log(f"  step {step:3d}/{total}  pelvis {data.xpos[pelvis][2]:.3f} m  "
                f"roll {roll:+6.1f}  pitch {pitch:+6.1f}  ncon {data.ncon}")

    out = {k: np.asarray(v) for k, v in logs.items()}
    out["joint_names"] = np.array(jn)
    out["settle"] = np.array(SETTLE)
    npz = HERE / "sim_rollouts" / f"{prefix}_rollout.npz"
    np.savez_compressed(npz, **out)

    try:
        import imageio.v2 as imageio

        mp4 = HERE / "videos" / f"{prefix}.mp4"
        imageio.mimwrite(mp4, frames, fps=int(CONTROL_HZ), quality=8)
        log(f"wrote {mp4}")
    except Exception as e:  # noqa: BLE001
        log(f"video failed: {e}")

    z = out["pelvis_h"]
    log("")
    log(f"RESULT  pelvis min {z.min():.3f} m, final {z[-1]:.3f} m")
    log(f"        max |roll| {np.abs(out['roll']).max():.1f} deg, "
        f"max |pitch| {np.abs(out['pitch']).max():.1f} deg")
    log(f"        xy drift {np.linalg.norm(out['root_pos'][:, :2] - out['root_pos'][0, :2], axis=1).max():.3f} m")
    log(f"        fell: {'YES' if z.min() < 0.35 else 'no'}")
    log(f"        peak |cfrc_ext| at t=0: {out['contact_force'][0]:.1f} N "
        f"(Isaac reports multi-kN here while merely standing)")
    log(f"wrote {npz}")


if __name__ == "__main__":
    main()
