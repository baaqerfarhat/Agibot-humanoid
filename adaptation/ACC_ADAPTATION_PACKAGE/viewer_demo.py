"""Watch the X2 box-pickup policy in the MuJoCo viewer, with and without ACE layer adaptation.

    python view_box.py            # frozen policy   -> topples ~1 s in
    python view_box.py --adapt    # ACC-2026 Lyapunov layer adaptation -> keeps tracking
    python view_box.py --adapt --seed 7 --speed 0.5

Runs natively on Windows so the GPU drives the real MuJoCo viewer. Physics 500 Hz, policy 50 Hz,
matching the deployment loop (leg target filter 0.8, gain scale 1.2, 0.15 rad/step rate limit).

Adaptation (only with --adapt), the confirmed configuration:
    Wdot_2 = Gamma * delta_2 z_1^T - gamma W_2 ,  Gamma = 3e-4, leak = 1e-2
    delta_L = -P * action_scale * Kp * (q - q_ref), error restricted to legs+waist
Held-out result: tracked steps 42.8 -> 81.5 (31/32 seeds, p < 1e-4), leg tracking error -37%.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import mujoco
import mujoco.viewer

XML = r"C:\SimuAgibot\acf_x2\robot_full_flat_ground_excl.xml"
POLICY = r"C:\SimuAgibot\box_run\x2_box_policy_v31.npz"

GAIN_SCALE, LEG_FILTER, MAX_JOINT_STEP = 1.2, 0.8, 0.15
T0, GAMMA, LEAK, P_GAIN, LAYER = 20, 3e-4, 1e-2, 1.0, 2
FALL_HEIGHT = 0.35


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def yaw_quat(q):
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def elu_jac(a):
    return np.where(a > 0, 1.0, np.exp(np.minimum(a, 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt", action="store_true", help="enable ACE-selected layer adaptation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="restart automatically after a fall")
    args = ap.parse_args()

    d = np.load(POLICY, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    names = meta["joint_names"]
    W = [d[f"W{i}"].astype(np.float64) for i in range(int(d["n_layers"]))]
    b = [d[f"b{i}"].astype(np.float64) for i in range(int(d["n_layers"]))]
    mean, std = d["mean"].astype(np.float64), d["std"].astype(np.float64)
    ref_pos, ref_vel = d["ref_joint_pos"].astype(np.float64), d["ref_joint_vel"].astype(np.float64)
    ref_quat = d["ref_quat_xyzw"].astype(np.float64)

    default_q = np.array(meta["default_joint_pos"])
    scale = np.array(meta["action_scale"])
    kp = np.array(meta["joint_stiffness"]) * GAIN_SCALE
    kd = np.array(meta["joint_damping"]) * GAIN_SCALE
    leg = np.array([any(k in n for k in ("hip", "knee", "ankle")) for n in names])
    emask = np.array([1.0 if any(k in n for k in ("hip", "knee", "ankle", "waist")) else 0.0
                      for n in names])

    model = mujoco.MjModel.from_xml_path(XML)
    for i in range(31):
        model.actuator_gainprm[i, 0] = kp[i]
        model.actuator_biasprm[i, 1] = -kp[i]
        model.actuator_biasprm[i, 2] = -kd[i]
    dec = int(round((1.0 / meta["control_hz"]) / model.opt.timestep))
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, meta["ref_body"])
    spheres = [i for i in range(model.ngeom)
               if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
               and (model.geom_contype[i] or model.geom_conaffinity[i])
               and model.geom_bodyid[i] != 0]

    data = mujoco.MjData(model)

    def reset(seed):
        rng = np.random.default_rng(seed)
        data.qpos[7:] = ref_pos[0] + rng.normal(0, args.noise, size=31)
        data.qpos[3:7] = np.array([1.0, 0, 0, 0])
        data.qpos[2] = 1.0
        mujoco.mj_forward(model, data)
        data.qpos[2] -= min(data.geom_xpos[i][2] - model.geom_size[i][0] for i in spheres)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        rq = ref_quat[0]
        yoff = quat_mul(yaw_quat(data.xquat[torso].copy()),
                        quat_conj(yaw_quat(np.array([rq[3], rq[0], rq[1], rq[2]]))))
        return yoff, [w.copy() for w in W], np.zeros(31), default_q.copy()

    yaw_off, Wa, last_action, prev_target = reset(args.seed)
    step = 0
    ctrl_dt = dec * model.opt.timestep
    vel6 = np.zeros(6)
    mode = "ADAPTED (ACE layer 2)" if args.adapt else "FROZEN"
    print(f"[{mode}]  seed {args.seed}   physics {1/model.opt.timestep:.0f} Hz / "
          f"policy {meta['control_hz']} Hz   speed {args.speed}x")
    print("close the viewer window to quit")

    def draw_balance(v, foot_pts, com):
        """CoM ground projection + support-polygon extent.

        This is what decides the fall: the reference is statically feasible but only by ~1.7 cm
        of margin, so seeing the CoM relative to the feet explains the topple far better than
        watching the body. Red marker = CoM projected to the floor, green = support extent.
        """
        n = 0
        scn = v.user_scn

        def add(pos, size, rgba):
            nonlocal n
            if n >= len(scn.geoms):
                return
            mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([size, 0, 0]), np.asarray(pos, dtype=float),
                                np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
            n += 1

        inside = (foot_pts[:, 0].min() <= com[0] <= foot_pts[:, 0].max()
                  and foot_pts[:, 1].min() <= com[1] <= foot_pts[:, 1].max())
        add([com[0], com[1], 0.005], 0.035,
            [0.2, 0.9, 0.2, 0.9] if inside else [0.95, 0.15, 0.15, 0.95])
        for sx in (foot_pts[:, 0].min(), foot_pts[:, 0].max()):
            for sy in (foot_pts[:, 1].min(), foot_pts[:, 1].max()):
                add([sx, sy, 0.004], 0.018, [0.25, 0.5, 1.0, 0.75])
        scn.ngeom = n

    foot_sph = [i for i in spheres
                if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                      int(model.geom_bodyid[i])) or "")]
    if not foot_sph:
        foot_sph = spheres

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 135, -15
        v.cam.lookat[:] = [0.2, 0.0, 0.6]
        wall = time.time()
        last_print = 0.0
        while v.is_running():
            frame = min(step, len(ref_pos) - 1)

            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, torso, vel6, 0)
            ang = data.xmat[torso].reshape(3, 3).T @ vel6[:3]
            rqf = ref_quat[frame]
            q_ref = quat_mul(yaw_off, np.array([rqf[3], rqf[0], rqf[1], rqf[2]]))
            q_rel = quat_mul(quat_conj(data.xquat[torso].copy()), q_ref)
            m9 = np.zeros(9)
            mujoco.mju_quat2Mat(m9, q_rel)
            ori6 = m9.reshape(3, 3)[:, :2].reshape(-1)

            obs = np.concatenate([last_action, ang, data.qpos[7:] - default_q, data.qvel[6:],
                                  ref_pos[frame], ref_vel[frame], ori6])

            x = (obs - mean) / std
            acts, zs = [], [x]
            for i in range(len(Wa)):
                ai = Wa[i] @ x + b[i]
                acts.append(ai)
                x = ai if i == len(Wa) - 1 else elu(ai)
                zs.append(x)
            action = acts[-1]
            last_action = action.copy()

            tgt = action * scale + default_q
            tgt = np.where(leg, (1 - LEG_FILTER) * tgt + LEG_FILTER * prev_target, tgt)
            tgt = prev_target + np.clip(tgt - prev_target, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            prev_target = tgt.copy()

            for _ in range(dec):
                data.ctrl[:] = tgt
                mujoco.mj_step(model, data)
            step += 1

            if args.adapt and step > T0:
                e = (data.qpos[7:] - ref_pos[min(step, len(ref_pos) - 1)]) * emask
                dl = -P_GAIN * (scale * kp * e)
                for l in range(len(Wa) - 1, LAYER, -1):
                    dl = elu_jac(acts[l - 1]) * (Wa[l].T @ dl)
                Wa[LAYER] += ctrl_dt * (GAMMA * np.outer(dl, zs[LAYER]) - LEAK * Wa[LAYER])

            fp = np.array([data.geom_xpos[i][:2] for i in foot_sph])
            com = data.subtree_com[0]
            margin = min(fp[:, 0].max() - com[0], com[0] - fp[:, 0].min(),
                         fp[:, 1].max() - com[1], com[1] - fp[:, 1].min())
            draw_balance(v, fp, com)

            t_now = step * ctrl_dt
            if t_now - last_print >= 0.25:
                last_print = t_now
                err = np.degrees(np.abs(data.qpos[7:] - ref_pos[frame]))[
                    [i for i, n in enumerate(names)
                     if any(k in n for k in ("hip", "knee", "ankle"))]].mean()
                print(f"  t={t_now:5.2f}s  legErr={err:5.1f} deg  CoM margin="
                      f"{margin*100:+6.1f} cm  pelvis={data.qpos[2]:.2f} m"
                      f"{'   <-- CoM OUTSIDE feet' if margin < 0 else ''}")

            v.sync()
            wall += ctrl_dt / max(args.speed, 1e-6)
            sleep = wall - time.time()
            if sleep > 0:
                time.sleep(sleep)

            fallen = data.qpos[2] < FALL_HEIGHT
            if fallen or step >= len(ref_pos):
                print(f"  {'FELL' if fallen else 'motion complete'} at t = {step*ctrl_dt:.2f} s"
                      f"  (pelvis {data.qpos[2]:.2f} m)")
                if not args.loop:
                    while v.is_running():
                        v.sync()
                        time.sleep(0.05)
                    break
                time.sleep(1.0)
                yaw_off, Wa, last_action, prev_target = reset(args.seed)
                step = 0
                wall = time.time()


if __name__ == "__main__":
    main()
