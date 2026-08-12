#!/usr/bin/env python3
"""What do the deployment's command filters cost, measured in Isaac?

`deploy_x2_box_pickup.py` does not publish the policy's target directly. It runs
two post-processing steps that have NO counterpart in training:

    leg EMA        tgt = (1-f)*tgt + f*prev_tgt      on hip/knee/ankle only
    step clamp     |tgt - prev_tgt| <= max_joint_step  on every joint

Both were added to damp leg jitter on hardware. This puts the exact same
arithmetic in front of the Isaac action, so the cost of each can be read off
against the unfiltered baseline the policy was trained and validated with.

Isaac's joint action term is target = action * action_scale + default_joint_pos,
so a filter on targets is applied here by mapping action -> target, filtering,
and mapping back.

Usage (hssim env, from the repo root):
  OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
    /home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    adaptation/deploy_artifacts_isaac.py --seeds 5 --steps 500
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

import eval_adapt_isaac as base  # noqa: E402
import paths  # noqa: E402
from ace_adapt import ExportedPolicy  # noqa: E402
from adapt_experiments_isaac import box_metrics  # noqa: E402
from obs_frame_isaac import lateral_lean_deg  # noqa: E402

LEG_KEYS = ("hip", "knee", "ankle")


class DeployFilterHook:
    """Reproduces deploy_x2_box_pickup.py's target post-processing."""

    def __init__(self, task, meta, leg_filter: float, max_joint_step: float):
        import torch

        self.torch = torch
        names = list(task.simulator.dof_names)
        idx = [meta["joint_names"].index(n) for n in names]
        self.scale = np.asarray(meta["action_scale"], float)[idx]
        self.default = np.asarray(meta["default_joint_pos"], float)[idx]
        self.is_leg = np.array([any(k in n for k in LEG_KEYS) for n in names])
        self.leg_filter = float(leg_filter)
        self.max_step = float(max_joint_step)
        self.prev_target = None
        self.clamped_ticks = 0
        self.n = 0
        self.lag = []  # |published - raw| target deviation, deg

    def __call__(self, actions):
        a = actions[0].detach().cpu().numpy().astype(float)
        raw = a * self.scale + self.default
        if self.prev_target is None:
            # the deploy loop enters the policy phase holding the start pose
            self.prev_target = raw.copy()

        tgt = raw.copy()
        if self.leg_filter > 0.0:
            tgt[self.is_leg] = ((1.0 - self.leg_filter) * tgt[self.is_leg]
                                + self.leg_filter * self.prev_target[self.is_leg])
        if np.isfinite(self.max_step):
            step = np.clip(tgt - self.prev_target, -self.max_step, self.max_step)
            if np.any(np.abs(tgt - self.prev_target) > self.max_step + 1e-12):
                self.clamped_ticks += 1
            tgt = self.prev_target + step

        self.lag.append(float(np.degrees(np.abs(tgt - raw)).mean()))
        self.prev_target = tgt
        self.n += 1

        a_eff = (tgt - self.default) / self.scale
        return self.torch.as_tensor(a_eff, device=actions.device,
                                    dtype=actions.dtype).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.resolve_ckpt()))
    ap.add_argument("--policy-npz", default=str(paths.POLICY_NPZ))
    ap.add_argument("--motion", default=str(paths.MOTION))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=600)
    ap.add_argument("--variants",
                    default="none:0:inf,f0.2:0.2:inf,f0.5:0.5:inf,f0.9:0.9:inf,"
                            "step:0:0.15,deploy0.2:0.2:0.15,deploy0.9:0.9:0.15",
                    help="comma list of name:leg_filter:max_joint_step")
    ap.add_argument("--out-dir", default=str(HERE / "isaac_runs" / "deploy_artifacts"))
    args = ap.parse_args()

    variants = []
    for spec in args.variants.split(","):
        nm, f, s = spec.split(":")
        variants.append((nm, float(f), float(s)))

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
    meta = json.loads(str(np.load(args.policy_npz, allow_pickle=True)["meta_json"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    summary = {}

    for nm, f, s in variants:
        print(f"\n=== {nm}  (leg_filter={f}, max_joint_step={s})  {len(seeds)} seeds ===")
        rows = []
        for sd in seeds:
            seeding(sd, torch_deterministic=False)
            hook = (None if (f == 0.0 and not np.isfinite(s))
                    else DeployFilterHook(task, meta, f, s))
            r = base._rollout(algo, task, None, args.steps, ref_pos, ctrl_dt, action_hook=hook)
            r.update(box_metrics(r["records"], ctrl_dt))
            lean = lateral_lean_deg(np.asarray(r["records"]["root_quat_xyzw"]))
            rows.append({
                "seed": sd,
                "survival": r["survival"],
                "leg_err": r["leg_err"],
                "success": bool(r.get("success", False)),
                "peak_lean": float(np.abs(lean).max()),
                "cmd_lag": float(np.mean(hook.lag)) if hook else 0.0,
                "clamp_pct": (100.0 * hook.clamped_ticks / max(hook.n, 1)) if hook else 0.0,
            })
            print(f"  seed {sd}: survival {r['survival']:4d} ({r['survival']*ctrl_dt:5.2f}s)  "
                  f"leg_err {r['leg_err']:5.2f}  peak lean {rows[-1]['peak_lean']:5.1f} deg  "
                  f"success={rows[-1]['success']}  cmd offset {rows[-1]['cmd_lag']:5.2f} deg  "
                  f"clamped {rows[-1]['clamp_pct']:4.1f}%")
        summary[nm] = {
            "leg_filter": f, "max_joint_step": s, "rows": rows,
            "survival_mean": float(np.mean([x["survival"] for x in rows])),
            "survival_std": float(np.std([x["survival"] for x in rows])),
            "leg_err_mean": float(np.mean([x["leg_err"] for x in rows])),
            "success_rate": float(np.mean([x["success"] for x in rows])),
            "peak_lean_mean": float(np.mean([x["peak_lean"] for x in rows])),
            "cmd_lag_mean": float(np.mean([x["cmd_lag"] for x in rows])),
            "clamp_pct_mean": float(np.mean([x["clamp_pct"] for x in rows])),
        }

    print(f"\n=== SUMMARY over {len(seeds)} seeds ===")
    print(f"  {'variant':11s} {'filt':>5s} {'step':>5s} {'survival(s)':>14s} {'leg_err':>8s} "
          f"{'success':>8s} {'lean':>6s} {'cmd off':>8s} {'clamp':>7s}")
    for nm, _, _ in variants:
        v = summary[nm]
        print(f"  {nm:11s} {v['leg_filter']:5.1f} {v['max_joint_step']:5.2f} "
              f"{v['survival_mean']*ctrl_dt:6.2f} +/- {v['survival_std']*ctrl_dt:5.2f} "
              f"{v['leg_err_mean']:8.2f} {v['success_rate']*100:7.0f}% "
              f"{v['peak_lean_mean']:6.1f} {v['cmd_lag_mean']:8.2f} {v['clamp_pct_mean']:6.1f}%")

    (out_dir / "summary.json").write_text(json.dumps(
        {"seeds": seeds, "steps": args.steps, "ctrl_dt": ctrl_dt,
         "ckpt": args.ckpt, "variants": summary}, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")
    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
