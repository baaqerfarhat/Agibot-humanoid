#!/usr/bin/env python3
"""Does the deployment's base_ang_vel frame error explain the hardware failure?

holosoma builds `base_ang_vel` from the articulation ROOT (the `pelvis` freejoint
body) in the pelvis frame. `deploy_x2_box_pickup.py` had no pelvis IMU and fed
the TORSO IMU gyro into that slot instead, and the torso is three waist joints
above the pelvis. This script puts that exact substitution into Isaac, where the
pelvis signal is ground truth, and runs three variants back to back:

    pelvis       the correct observation -- reproduces the training condition
    torso        the deployment's observation -- torso gyro in the torso frame
    torso_fixed  the deployment's observation passed through
                 `agibot_control_functions/base_frame.PelvisEstimator`

`pelvis` vs `torso` says whether the defect is sufficient to cause the failure.
`torso_fixed` vs `pelvis` says whether the reconstruction is a correct fix, both
numerically (residual against the ground-truth pelvis signal) and behaviourally.

Usage (hssim env, from the repo root):
  OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
    /home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    adaptation/obs_frame_isaac.py --seeds 5 --steps 500
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ACC_ADAPTATION_PACKAGE"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agibot_control_functions"))

import base_frame as bf  # noqa: E402
import eval_adapt_isaac as base  # noqa: E402
import paths  # noqa: E402
from ace_adapt import ExportedPolicy  # noqa: E402
from adapt_experiments_isaac import box_metrics  # noqa: E402

# Alphabetical term order, so base_ang_vel is the 3 dims right after actions(31).
ANG_VEL_SLICE = slice(31, 34)
MODES = ("pelvis", "torso", "torso_fixed")


class ObsFrameHook:
    """Rewrites base_ang_vel to whatever the deployment would have measured."""

    def __init__(self, task, mode: str, dof_names: list[str]):
        import torch

        self.torch = torch
        self.task = task
        self.mode = mode
        sim = task.simulator
        body_list = list(sim._body_list)
        self.torso_idx = body_list.index("torso_link")
        self.waist_idx = [dof_names.index(n) for n in bf.WAIST_JOINTS]
        self.est = bf.PelvisEstimator()
        self.log = {"true": [], "used": [], "torso_raw": []}

    def _torso_gyro_body(self):
        """What an IMU rigidly mounted on torso_link would report."""
        sim = self.task.simulator
        q = sim._rigid_body_rot[0, self.torso_idx].detach().cpu().numpy().astype(float)
        w_world = sim._rigid_body_ang_vel[0, self.torso_idx].detach().cpu().numpy().astype(float)
        R = bf._quat_xyzw_to_mat(q)
        return q, R.T @ w_world

    def __call__(self, actor_obs):
        true = actor_obs[0, ANG_VEL_SLICE].detach().cpu().numpy().astype(float)
        q_torso, w_torso = self._torso_gyro_body()

        if self.mode == "pelvis":
            used = true
        elif self.mode == "torso":
            used = w_torso
        else:
            sim = self.task.simulator
            qj = sim.dof_pos[0].detach().cpu().numpy().astype(float)
            dqj = sim.dof_vel[0].detach().cpu().numpy().astype(float)

            class J:
                __slots__ = ("position", "velocity")

                def __init__(self, p, v):
                    self.position, self.velocity = p, v

            jmap = {n: J(qj[i], dqj[i]) for n, i in zip(bf.WAIST_JOINTS, self.waist_idx)}
            used, _ = self.est.update(q_torso, w_torso, jmap)
            used = np.asarray(used, float)

        self.log["true"].append(true.copy())
        self.log["used"].append(np.asarray(used, float).copy())
        self.log["torso_raw"].append(w_torso.copy())

        if self.mode != "pelvis":
            actor_obs = actor_obs.clone()
            actor_obs[0, ANG_VEL_SLICE] = self.torch.as_tensor(
                used, device=actor_obs.device, dtype=actor_obs.dtype
            )
        return actor_obs


def lateral_lean_deg(root_quat_xyzw: np.ndarray) -> np.ndarray:
    """Signed sideways tilt of the pelvis, in degrees.

    Euler roll is useless here: the task pitches past 70 deg, where roll and yaw
    become degenerate and arctan2 snaps to +/-180. This uses the world-z
    component of the pelvis y-axis instead, which stays well behaved through the
    whole bend and is still positive/negative for left/right lean.
    """
    q = np.asarray(root_quat_xyzw, float)
    x, y, z, w = q.T
    y_axis_z = 2.0 * (y * z + w * x)  # R[2, 1]
    return np.degrees(np.arcsin(np.clip(y_axis_z, -1.0, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.resolve_ckpt()))
    ap.add_argument("--policy-npz", default=str(paths.POLICY_NPZ))
    ap.add_argument("--motion", default=str(paths.MOTION))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=600)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out-dir", default=str(HERE / "isaac_runs" / "obs_frame"))
    ap.add_argument("--record-seed", type=int, default=None)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        assert m in MODES, f"unknown mode {m}"

    paths.enter_holosoma()

    from holosoma.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from holosoma.utils.common import seeding  # noqa: E402
    from holosoma.utils.config_utils import CONFIG_NAME  # noqa: E402
    from holosoma.utils.eval_utils import (  # noqa: E402
        CheckpointConfig,
        init_eval_logging,
        load_checkpoint,
        load_saved_experiment_config,
    )
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
    from holosoma.utils.helpers import get_class  # noqa: E402
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment  # noqa: E402

    init_eval_logging()
    checkpoint_cfg = CheckpointConfig(checkpoint=args.ckpt)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # Same demo conditions as every other adaptation experiment: start at t=0,
    # no initial-pose noise, no bad_tracking termination.
    motion_term = saved_cfg.command.setup_terms["motion_command"]
    mc = motion_term.params["motion_config"]
    if isinstance(mc, dict):
        mc = dict(mc)
        mc["use_adaptive_timesteps_sampler"] = False
        mc["start_at_timestep_zero_prob"] = 1.0
        mc["freeze_at_timestep_zero_prob"] = 0.0
        noise = dict(mc.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        mc["noise_to_initial_pose"] = noise
        mc["motion_file"] = args.motion
        mc["motion_dir"] = ""
        motion_term.params["motion_config"] = mc
    else:
        motion_term.params["motion_config"] = dataclasses.replace(
            mc,
            use_adaptive_timesteps_sampler=False,
            start_at_timestep_zero_prob=1.0,
            freeze_at_timestep_zero_prob=0.0,
            noise_to_initial_pose=dataclasses.replace(mc.noise_to_initial_pose, overall_noise_scale=0.0),
            motion_file=args.motion,
            motion_dir="",
        )
    saved_cfg.termination.terms.pop("bad_tracking", None)

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", args.steps)

    env, device, simulation_app = setup_simulation_environment(eval_cfg)
    eval_log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.save_config(str(eval_log_dir / CONFIG_NAME))
    object.__setattr__(eval_cfg.algo.config, "eval_callbacks", {})

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    algo: BaseAlgo = get_class(eval_cfg.algo._target_)(
        device=device, env=env, config=eval_cfg.algo.config, log_dir=str(eval_log_dir), multi_gpu_cfg=None
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(checkpoint))
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()

    task = algo._unwrap_env()
    ctrl_dt = float(task.dt)
    pol = ExportedPolicy(args.policy_npz)
    ref_pos = pol.ref_pos
    assert ref_pos is not None
    dof_names = list(task.simulator.dof_names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    summary: dict[str, dict] = {}

    for mode in modes:
        print(f"\n=== mode {mode} ({len(seeds)} seeds) ===")
        rows = []
        for s in seeds:
            seeding(s, torch_deterministic=False)
            hook = ObsFrameHook(task, mode, dof_names)
            r = base._rollout(algo, task, None, args.steps, ref_pos, ctrl_dt, obs_hook=hook)
            r.update(box_metrics(r["records"], ctrl_dt))

            roll = lateral_lean_deg(np.asarray(r["records"]["root_quat_xyzw"]))
            height = np.asarray(r["records"]["root_pos"])[:, 2]
            true = np.asarray(hook.log["true"])
            used = np.asarray(hook.log["used"])
            n = min(len(true), len(used))
            obs_err = float(np.abs(used[:n] - true[:n]).mean())
            raw_err = float(np.abs(np.asarray(hook.log["torso_raw"])[:n] - true[:n]).mean())

            rows.append(
                {
                    "seed": s,
                    "survival": r["survival"],
                    "leg_err": r["leg_err"],
                    "picked": bool(r.get("picked", False)),
                    "success": bool(r.get("success", False)),
                    "peak_lean": float(np.abs(roll).max()),
                    "lean_at_peak": float(roll[np.argmax(np.abs(roll))]),
                    "min_height": float(height.min()),
                    "obs_err": obs_err,
                    "raw_err": raw_err,
                }
            )
            print(
                f"  seed {s}: survival {r['survival']:4d} ({r['survival']*ctrl_dt:5.2f}s)  "
                f"leg_err {r['leg_err']:5.2f} deg  peak lean {rows[-1]['lean_at_peak']:+6.1f} deg  "
                f"min height {rows[-1]['min_height']:5.2f} m  "
                f"picked={rows[-1]['picked']} success={rows[-1]['success']}  "
                f"obs_err {obs_err:.4f} rad/s"
            )

            if args.record_seed is not None and s == args.record_seed:
                base._save_npz(
                    out_dir / f"isaac_obsframe_{mode}_seed{s}.npz",
                    r,
                    {"mode": mode, "seed": s, "ckpt": args.ckpt, "obs_err": obs_err},
                )

        summary[mode] = {
            "rows": rows,
            "survival_mean": float(np.mean([x["survival"] for x in rows])),
            "survival_std": float(np.std([x["survival"] for x in rows])),
            "leg_err_mean": float(np.mean([x["leg_err"] for x in rows])),
            "peak_lean_mean": float(np.mean([x["peak_lean"] for x in rows])),
            "min_height_mean": float(np.mean([x["min_height"] for x in rows])),
            "success_rate": float(np.mean([x["success"] for x in rows])),
            "picked_rate": float(np.mean([x["picked"] for x in rows])),
            "obs_err_mean": float(np.mean([x["obs_err"] for x in rows])),
            "raw_err_mean": float(np.mean([x["raw_err"] for x in rows])),
        }

    print(f"\n=== SUMMARY over {len(seeds)} seeds ===")
    print(f"  {'mode':12s} {'survival(s)':>14s} {'leg_err':>8s} {'peak lean':>10s} "
          f"{'min h':>7s} {'picked':>7s} {'success':>8s} {'obs_err':>9s}")
    for mode in modes:
        v = summary[mode]
        print(
            f"  {mode:12s} {v['survival_mean']*ctrl_dt:6.2f} +/- {v['survival_std']*ctrl_dt:5.2f} "
            f"{v['leg_err_mean']:8.2f} {v['peak_lean_mean']:10.1f} "
            f"{v['min_height_mean']:7.2f} "
            f"{v['picked_rate']*100:6.0f}% {v['success_rate']*100:7.0f}% "
            f"{v['obs_err_mean']:9.4f}"
        )
    if "torso" in summary and "torso_fixed" in summary:
        raw = summary["torso"]["obs_err_mean"]
        fix = summary["torso_fixed"]["obs_err_mean"]
        print(f"\n  observation error vs the ground-truth pelvis signal: "
              f"{raw:.4f} -> {fix:.4f} rad/s ({100*(1-fix/max(raw,1e-9)):.1f}% removed)")

    (out_dir / "summary.json").write_text(
        json.dumps({"seeds": seeds, "steps": args.steps, "ctrl_dt": ctrl_dt,
                    "ckpt": args.ckpt, "modes": summary}, indent=2)
    )
    print(f"\nwrote {out_dir / 'summary.json'}")
    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
