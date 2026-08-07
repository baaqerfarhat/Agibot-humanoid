"""Reference integration: the ACC layer adaptation on the X2 box-pickup policy, in MuJoCo.

    python run_mujoco_demo.py                 # frozen
    python run_mujoco_demo.py --adapt         # with adaptation
    python run_mujoco_demo.py --adapt --view  # interactive MuJoCo viewer
    python run_mujoco_demo.py --adapt --seeds 32   # batch, prints the summary table

This file is the WORKED EXAMPLE of the deployment loop. If you are wiring the adapter into your
own environment, the parts you must reproduce are marked  ### CONTRACT ###  below — everything
else is specific to this robot.

The deployment loop is transcribed from the vendor's `deploy_x2_box_pickup.py` and validated
against real-robot logs: replaying the robot's logged state through this pipeline reproduces its
own commanded targets to 0.84 deg on the leg joints across all 742 logged control steps.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

from ace_adapt import AdaptConfig, ExportedPolicy, LayerAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "assets", "robot_full_flat_ground_excl.xml")
POLICY = os.path.join(HERE, "assets", "x2_box_policy_v31.npz")

# Deployment constants (vendor script + run metadata). Do not change casually.
GAIN_SCALE = 1.2          # commanded PD gains are scaled by this on the real robot
LEG_FILTER = 0.8          # low-pass on LEG targets only; arms unfiltered
MAX_JOINT_STEP = 0.15     # rad per control step, all joints
FALL_HEIGHT = 0.35        # pelvis height below which the episode is a fall
TRACK_LIMIT_DEG = 20.0    # leg tracking error above which tracking is considered lost


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


class BoxPickupEnv:
    """X2 box-pickup, deployment-faithful: 500 Hz physics / 50 Hz policy."""

    def __init__(self, policy: ExportedPolicy):
        self.pol = policy
        meta = policy.meta
        self.names = meta["joint_names"]
        self.default = np.array(meta["default_joint_pos"])
        self.scale = np.array(meta["action_scale"])
        self.kp = np.array(meta["joint_stiffness"]) * GAIN_SCALE
        self.kd = np.array(meta["joint_damping"]) * GAIN_SCALE
        self.leg = np.array([any(k in n for k in ("hip", "knee", "ankle")) for n in self.names])
        self.leg_idx = [i for i, n in enumerate(self.names)
                        if any(k in n for k in ("hip", "knee", "ankle"))]

        self.model = mujoco.MjModel.from_xml_path(XML)
        for i in range(31):                      # write commanded gains into the position servos
            self.model.actuator_gainprm[i, 0] = self.kp[i]
            self.model.actuator_biasprm[i, 1] = -self.kp[i]
            self.model.actuator_biasprm[i, 2] = -self.kd[i]
        self.dec = int(round((1.0 / meta["control_hz"]) / self.model.opt.timestep))
        self.ctrl_dt = self.dec * self.model.opt.timestep
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, meta["ref_body"])
        self.spheres = [i for i in range(self.model.ngeom)
                        if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                        and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
                        and self.model.geom_bodyid[i] != 0]
        self.data = mujoco.MjData(self.model)

    def reset(self, seed=0, noise=0.01):
        d, m = self.data, self.model
        rng = np.random.default_rng(seed)
        d.qpos[7:] = self.pol.ref_pos[0] + rng.normal(0, noise, size=31)
        d.qpos[3:7] = np.array([1.0, 0, 0, 0])
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        d.qpos[2] -= min(d.geom_xpos[i][2] - m.geom_size[i][0] for i in self.spheres)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        rq = self.pol.ref_quat[0]
        self.yaw_off = quat_mul(yaw_quat(d.xquat[self.torso].copy()),
                                quat_conj(yaw_quat(np.array([rq[3], rq[0], rq[1], rq[2]]))))
        self.step_i = 0
        self.prev_target = self.default.copy()
        self.last_action = np.zeros(31)

    ### CONTRACT ### the observation the policy expects (164-d, ALPHABETICAL term order)
    def observation(self):
        d, m = self.data, self.model
        f = min(self.step_i, len(self.pol.ref_pos) - 1)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.torso, vel6, 0)
        ang = d.xmat[self.torso].reshape(3, 3).T @ vel6[:3]     # torso IMU gyro, BODY frame
        rqf = self.pol.ref_quat[f]
        q_ref = quat_mul(self.yaw_off, np.array([rqf[3], rqf[0], rqf[1], rqf[2]]))
        q_rel = quat_mul(quat_conj(d.xquat[self.torso].copy()), q_ref)
        m9 = np.zeros(9)
        mujoco.mju_quat2Mat(m9, q_rel)
        ori6 = m9.reshape(3, 3)[:, :2].reshape(-1)
        return np.concatenate([self.last_action, ang, d.qpos[7:] - self.default, d.qvel[6:],
                               self.pol.ref_pos[f], self.pol.ref_vel[f], ori6])

    ### CONTRACT ### action -> joint target, with the leg filter and rate limit
    def apply(self, action):
        self.last_action = action.copy()
        tgt = action * self.scale + self.default
        tgt = np.where(self.leg, (1 - LEG_FILTER) * tgt + LEG_FILTER * self.prev_target, tgt)
        tgt = self.prev_target + np.clip(tgt - self.prev_target, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        self.prev_target = tgt.copy()
        for _ in range(self.dec):
            self.data.ctrl[:] = tgt
            mujoco.mj_step(self.model, self.data)
        self.step_i += 1

    ### CONTRACT ### the tracking error the adapter consumes
    def joint_error(self):
        f = min(self.step_i, len(self.pol.ref_pos) - 1)
        return self.data.qpos[7:] - self.pol.ref_pos[f]

    def leg_error_deg(self):
        return float(np.degrees(np.abs(self.joint_error()))[self.leg_idx].mean())

    def fallen(self):
        return bool(self.data.qpos[2] < FALL_HEIGHT)


def rollout(env, adapter, max_steps, view=False):
    """Returns (survival_steps, tracked_steps, mean_leg_error_deg)."""
    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 135, -15
        viewer.cam.lookat[:] = [0.2, 0.0, 0.6]

    errs, tracked, survival = [], None, max_steps
    try:
        for k in range(max_steps):
            action = adapter.act(env.observation()) if adapter else \
                env.pol.forward(env.observation())[0]
            env.apply(action)
            if adapter:
                adapter.update(env.joint_error(), env.ctrl_dt)

            e = env.leg_error_deg()
            errs.append(e)
            if viewer is not None:
                viewer.sync()
            if tracked is None and (env.fallen() or e > TRACK_LIMIT_DEG):
                tracked = k + 1
            if env.fallen():
                survival = k + 1
                break
    finally:
        if viewer is not None:
            viewer.close()
    return survival, (tracked if tracked is not None else max_steps), float(np.mean(errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt", action="store_true")
    ap.add_argument("--view", action="store_true", help="open the interactive viewer")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (batch mode if >1)")
    ap.add_argument("--seed0", type=int, default=600)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    pol = ExportedPolicy(POLICY)
    env = BoxPickupEnv(pol)
    cfg = AdaptConfig()
    if args.gain is not None:
        cfg.gain = args.gain
    if args.layer is not None:
        cfg.layer = args.layer
    print(f"policy: {pol.meta['task']}  layers {pol.layer_shapes()}")
    print(f"control {pol.meta['control_hz']} Hz, physics {1/env.model.opt.timestep:.0f} Hz")
    print(f"mode: {'ADAPTED (layer %d, Gamma %g)' % (cfg.layer, cfg.gain) if args.adapt else 'FROZEN'}")
    print()

    rows = []
    for s in range(args.seed0, args.seed0 + args.seeds):
        env.reset(seed=s)
        adapter = None
        if args.adapt:
            adapter = LayerAdapter(pol, cfg, joint_names=pol.meta["joint_names"])
            adapter.reset()
        surv, trk, err = rollout(env, adapter, args.max_steps, view=args.view)
        drift = adapter.weight_drift if adapter else 0.0
        div = adapter.diverged if adapter else False
        rows.append((s, surv, trk, err, drift, div))
        if args.seeds <= 8:
            print(f"  seed {s}: survival {surv:4d} ({surv*env.ctrl_dt:5.2f}s)  "
                  f"tracked {trk:4d}  legErr {err:6.2f} deg  |dW| {drift:.3f}"
                  f"{'  DIVERGED' if div else ''}")

    if args.seeds > 1:
        a = np.array([(r[1], r[2], r[3]) for r in rows], dtype=float)
        ndiv = sum(1 for r in rows if r[5])
        print()
        print(f"=== {args.seeds} seeds ===")
        print(f"  survival  mean {a[:,0].mean():6.1f} steps ({a[:,0].mean()*env.ctrl_dt:.2f} s)"
              f"   never fell: {int((a[:,0] >= args.max_steps).sum())}/{args.seeds}")
        print(f"  tracked   mean {a[:,1].mean():6.1f} steps")
        print(f"  legErr    mean {a[:,2].mean():6.2f} deg")
        print(f"  diverged  {ndiv}/{args.seeds}")


if __name__ == "__main__":
    main()
