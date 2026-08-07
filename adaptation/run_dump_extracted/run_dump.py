"""Capture everything needed to diff two environments running the same policy.

Run this in the WORKING environment and send back the .npz it writes. It records the per-step
observation/action/target/state of one rollout plus a full description of the model, the control
loop, and the policy file, so the two setups can be compared field by field.

Two ways to use it:

  (a) Drop-in for a MuJoCo env: import and call `dump_rollout(...)` with your own step function.
  (b) Standalone reference: run as-is against this package's own env to produce the file that
      the other side compares against.

      python run_dump.py --out mine.npz --seed 600 --steps 200

Nothing here is specific to the adaptation — it is purely a description of the environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform

import numpy as np


def sha256(path, blocks=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(blocks):
            h.update(chunk)
    return h.hexdigest()


def describe_model(model):
    """Everything about the plant that could differ between two setups."""
    import mujoco
    nu = model.nu
    return dict(
        nq=int(model.nq), nv=int(model.nv), nu=int(nu), nbody=int(model.nbody),
        total_mass=float(model.body_mass.sum()),
        timestep=float(model.opt.timestep),
        gravity=model.opt.gravity.tolist(),
        solver=str(mujoco.mjtSolver(model.opt.solver).name),
        iterations=int(model.opt.iterations),
        integrator=str(mujoco.mjtIntegrator(model.opt.integrator).name),
        cone=str(mujoco.mjtCone(model.opt.cone).name),
        impratio=float(model.opt.impratio),
        o_solref=model.opt.o_solref.tolist(),
        dof_armature=model.dof_armature.tolist(),
        dof_damping=model.dof_damping.tolist(),
        dof_frictionloss=model.dof_frictionloss.tolist(),
        actuator_gaintype=[int(x) for x in model.actuator_gaintype[:nu]],
        actuator_biastype=[int(x) for x in model.actuator_biastype[:nu]],
        actuator_gainprm=model.actuator_gainprm[:nu, :3].tolist(),
        actuator_biasprm=model.actuator_biasprm[:nu, :3].tolist(),
        actuator_forcerange=model.actuator_forcerange[:nu].tolist(),
        joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                     for j in range(model.njnt)],
        geom_friction_plane=[model.geom_friction[i].tolist() for i in range(model.ngeom)
                             if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE],
        n_collidable_geoms=int(sum(1 for i in range(model.ngeom)
                                   if model.geom_contype[i] or model.geom_conaffinity[i])),
    )


def dump_rollout(out_path, model, data, step_fn, n_steps, meta=None, extra_files=None):
    """Record a rollout.

    step_fn(k) must advance ONE control step and return a dict with any of:
        obs (164,), action (31,), target (31,), frame (int)
    Positions/velocities are read from `data` automatically.
    """
    rec = {k: [] for k in ("obs", "action", "target", "qpos", "qvel", "frame")}
    for k in range(n_steps):
        out = step_fn(k) or {}
        for key in ("obs", "action", "target"):
            if key in out:
                rec[key].append(np.asarray(out[key], dtype=np.float64).copy())
        rec["frame"].append(int(out.get("frame", k)))
        rec["qpos"].append(np.array(data.qpos, dtype=np.float64).copy())
        rec["qvel"].append(np.array(data.qvel, dtype=np.float64).copy())
        if out.get("done"):
            break

    payload = {k: np.asarray(v) for k, v in rec.items() if len(v)}
    payload["model_description_json"] = json.dumps(describe_model(model), indent=1)
    payload["meta_json"] = json.dumps(meta or {}, indent=1)
    payload["platform_json"] = json.dumps(dict(
        python=platform.python_version(), system=platform.system(),
        machine=platform.machine(),
    ))
    try:
        import mujoco
        payload["mujoco_version"] = str(mujoco.__version__)
    except Exception:
        pass
    if extra_files:
        payload["file_hashes_json"] = json.dumps(
            {p: {"sha256": sha256(p), "bytes": os.path.getsize(p)}
             for p in extra_files if os.path.exists(p)}, indent=1)

    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}")
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.dtype != object and v.ndim:
            print(f"   {k:<12} {v.shape}")
    print("\nSend this file back. It contains no proprietary weights — only observations,")
    print("actions, states, and a description of the model and loop.")


# ---------------------------------------------------------------------------------------
# Reference usage against this package's own environment
# ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="env_dump.npz")
    ap.add_argument("--seed", type=int, default=600)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    from ace_adapt import ExportedPolicy
    from run_mujoco_demo import POLICY, XML, BoxPickupEnv, GAIN_SCALE, LEG_FILTER, MAX_JOINT_STEP

    pol = ExportedPolicy(POLICY)
    env = BoxPickupEnv(pol)
    env.reset(seed=args.seed)

    def step_fn(k):
        obs = env.observation()
        action, _, _ = pol.forward(obs)
        frame = min(env.step_i, len(pol.ref_pos) - 1)
        env.apply(action)
        return dict(obs=obs, action=action, target=env.prev_target, frame=frame,
                    done=env.fallen())

    meta = dict(
        source="ACC_ADAPTATION_PACKAGE reference environment",
        seed=args.seed,
        control_hz=pol.meta["control_hz"],
        decimation=env.dec,
        gain_scale=GAIN_SCALE,
        leg_filter=LEG_FILTER,
        max_joint_step=MAX_JOINT_STEP,
        obs_term_order=["actions(31)", "base_ang_vel(3)", "dof_pos(31)", "dof_vel(31)",
                        "motion_command(62)", "motion_ref_ori_b(6)"],
        base_ang_vel_frame="torso body frame (xmat[torso].T @ omega_world)",
        dof_pos_convention="q - default_joint_pos",
        reference_frames=int(len(pol.ref_pos)),
        reference_indexing="one frame per control step, no interpolation",
        yaw_alignment="yaw_offset = yaw(q_torso_0) x inv(yaw(q_ref_0)), applied to every ref quat",
        policy_meta=pol.meta,
        box_present=False,
    )
    dump_rollout(args.out, env.model, env.data, step_fn, args.steps,
                 meta=meta, extra_files=[POLICY, XML])


if __name__ == "__main__":
    main()
