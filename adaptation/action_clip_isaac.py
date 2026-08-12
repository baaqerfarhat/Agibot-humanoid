#!/usr/bin/env python3
"""Can the v33 policy still do the task if its actions are bounded?

The policy emits actions of |a| up to 34 in sim and 61 on hardware. Because
`action_scale = 0.25 * effort_limit / kp`, |a| = 4 already commands the full
effort limit, so everything past that is a request for torque the actuator cannot
produce. Training let it grow unchecked (`action_clip_value: 100`).

In sim this is harmless: the saturated torque is absorbed by contact -- the ankle
rolls press into the ground, the wrists into the box -- and those joints stay near
the reference (right_ankle_roll sits at +0.03 rad). On hardware the same torque
drives them to their mechanical stops (right_ankle_roll pins at +0.34 rad for 96%
of the motion, left_wrist_roll at -1.56), the feet end up on their edges and the
robot topples sideways at ~2.9 s. See run_logs/_sim_vs_real.py.

So the question is whether the saturation is load-bearing for the task or just an
unpenalised artifact. If v33 still completes the pickup with actions clipped to
around the effort limit, the clip can go straight into the deployment and the
runaway stops with no retraining. If it does not, the policy genuinely depends on
contact-absorbed saturation and v34 has to be retrained with the actions bounded.

Variants are `name:clip:joint_substrings`, clip in action units (4 = the effort
limit), `inf` = unbounded, empty substrings = every joint.

Usage (hssim env, from the repo root):
  OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
    /home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    adaptation/action_clip_isaac.py --seeds 3 --steps 500
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
from ace_adapt import AdaptConfig, ExportedPolicy  # noqa: E402
from adapt_experiments_isaac import W0LeakAdapter, box_metrics  # noqa: E402
from obs_frame_isaac import lateral_lean_deg  # noqa: E402

# the joints the hardware logs show running to their stops
RUNAWAY = ("ankle_roll", "wrist")


class ActionClipHook:
    """Clips the policy action, optionally only on selected joints."""

    def __init__(self, task, meta, clip: float, subs: tuple[str, ...]):
        import torch

        self.torch = torch
        names = list(task.simulator.dof_names)
        self.clip = float(clip)
        self.mask = np.array([(not subs) or any(s in n for s in subs) for n in names])
        self.names = names
        self.n = 0
        self.clipped_ticks = 0
        self.clipped_joint_ticks = 0
        self.per_joint = np.zeros(len(names))
        self.max_abs = np.zeros(len(names))
        # per joint, ticks whose |action| exceeds the effort limit (|a| = 4),
        # measured on the action the policy actually produced this run
        self.over_any = np.zeros(len(names))

    def __call__(self, actions):
        a = actions[0].detach().cpu().numpy().astype(float)
        self.max_abs = np.maximum(self.max_abs, np.abs(a))
        self.over_any += (np.abs(a) > self.clip if np.isfinite(self.clip)
                          else np.abs(a) > 4.0)
        self.n += 1
        if not np.isfinite(self.clip):
            return actions
        over = (np.abs(a) > self.clip) & self.mask
        if over.any():
            self.clipped_ticks += 1
            self.clipped_joint_ticks += int(over.sum())
            self.per_joint += over
        out = a.copy()
        out[self.mask] = np.clip(a[self.mask], -self.clip, self.clip)
        return self.torch.as_tensor(out, device=actions.device,
                                    dtype=actions.dtype).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(paths.resolve_ckpt()))
    ap.add_argument("--policy-npz", default=str(paths.POLICY_NPZ))
    ap.add_argument("--motion", default=str(paths.MOTION))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=600)
    ap.add_argument("--variants",
                    default="none:inf:,all8:8:,all6:6:,all4:4:,"
                            "runaway8:8:ankle_roll|wrist,"
                            "runaway4:4:ankle_roll|wrist,"
                            "ankle4:4:ankle_roll",
                    help="comma list of name:clip:substrings ('|'-separated, empty=all)")
    ap.add_argument("--out-dir", default=str(HERE / "isaac_runs" / "action_clip"))
    ap.add_argument("--record-seed", type=int, default=None,
                    help="Save this seed's rollout per variant, for rendering with "
                         "box_pickup/render_side_by_side.py.")
    ap.add_argument("--obs-noise", default="off", choices=["off", "on"])
    ap.add_argument("--dr", default="no-push", choices=["no-push", "none", "all"],
                    help="Matches adapt_experiments_isaac.py. The push randomizer shoves "
                         "the robot up to 0.7 m/s every 1-2.5 s, which changes the "
                         "trajectory enough that the ankle-roll saturation seen on "
                         "hardware stops appearing; keep it off to stay comparable to "
                         "the archived baseline and the adaptation study.")
    ap.add_argument("--policy-source", default="npz", choices=["npz", "torch"],
                    help="'npz' runs the exported numpy policy at adaptation gain 0 -- "
                         "the same forward pass the robot runs, and the same control arm "
                         "the adaptation study used. 'torch' runs holosoma's checkpoint; "
                         "the task is chaotic enough that the two take different "
                         "trajectories, and only 'npz' is comparable to hardware.")
    args = ap.parse_args()

    variants = []
    for spec in args.variants.split(","):
        nm, c, subs = spec.split(":")
        variants.append((nm, float(c), tuple(s for s in subs.split("|") if s)))

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
            noise_to_initial_pose=dataclasses.replace(mc.noise_to_initial_pose,
                                                      overall_noise_scale=0.0),
            motion_file=args.motion,
            motion_dir="",
        )
    saved_cfg.termination.terms.pop("bad_tracking", None)

    if args.obs_noise == "off":
        for group_name, group in saved_cfg.observation.groups.items():
            if getattr(group, "enable_noise", False):
                object.__setattr__(group, "enable_noise", False)
                print(f"[obs] disabled observation noise on '{group_name}'")

    drop = []
    if args.dr != "all":
        drop.append("push")
    if args.dr == "none":
        drop += ["actuator", "randomize_action_delay", "com", "bias", "friction", "mass"]
    if drop:
        for bucket in ("setup_terms", "reset_terms", "step_terms"):
            terms = getattr(saved_cfg.randomization, bucket, None)
            if not terms:
                continue
            for key in [k for k in terms if any(d in k.lower() for d in drop)]:
                terms.pop(key)
                print(f"[dr] dropped randomization term '{key}'")

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", args.steps)

    env, device, simulation_app = setup_simulation_environment(eval_cfg)
    eval_log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(),
                                      task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.save_config(str(eval_log_dir / CONFIG_NAME))
    object.__setattr__(eval_cfg.algo.config, "eval_callbacks", {})

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    algo: BaseAlgo = get_class(eval_cfg.algo._target_)(
        device=device, env=env, config=eval_cfg.algo.config, log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
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
    dof_names = list(task.simulator.dof_names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    summary = {}

    # which joints does the hardware pin to a stop? track them in sim for comparison
    watch = [n for n in dof_names if "ankle_roll" in n or "wrist_roll" in n
             or "wrist_pitch" in n]
    watch_idx = [dof_names.index(n) for n in watch]

    for nm, c, subs in variants:
        print(f"\n=== {nm}  (clip={c}, joints={'all' if not subs else '|'.join(subs)})  "
              f"{len(seeds)} seeds ===")
        rows = []
        for sd in seeds:
            seeding(sd, torch_deterministic=False)
            hook = ActionClipHook(task, meta, c, subs)
            adapter = None
            if args.policy_source == "npz":
                # gain 0 makes the adaptation update a no-op, so this is exactly the
                # exported numpy policy -- the one deployed on the robot.
                adapter = W0LeakAdapter(
                    pol, AdaptConfig(layer=2, gain=0.0, leak=1e-2, gx_level=1,
                                     error_joints=("hip", "knee", "ankle", "waist"),
                                     engage_step=0),
                    joint_names=pol.meta["joint_names"], mass_matrix_fn=None,
                )
            r = base._rollout(algo, task, adapter, args.steps, ref_pos, ctrl_dt,
                              action_hook=hook)
            r.update(box_metrics(r["records"], ctrl_dt))
            lean = lateral_lean_deg(np.asarray(r["records"]["root_quat_xyzw"]))
            dofp = np.asarray(r["records"]["dof_pos"])
            rows.append({
                "seed": sd,
                "survival": r["survival"],
                "leg_err": r["leg_err"],
                "success": bool(r.get("success", False)),
                "peak_lean": float(np.abs(lean).max()),
                "clip_pct": 100.0 * hook.clipped_ticks / max(hook.n, 1),
                "max_abs_action": float(hook.max_abs.max()),
                "watch_absmax": {n: float(np.abs(dofp[:, i]).max())
                                 for n, i in zip(watch, watch_idx)},
            })
            if args.record_seed is not None and sd == args.record_seed:
                base._save_npz(out_dir / f"isaac_{nm}_seed{sd}.npz", r,
                               {"mode": nm, "clip": c, "clip_joints": list(subs),
                                "ckpt": args.ckpt, "policy_npz": args.policy_npz,
                                "policy_source": args.policy_source})
            w = rows[-1]["watch_absmax"]
            print(f"  seed {sd}: survival {r['survival']:4d} "
                  f"({r['survival']*ctrl_dt:5.2f}s)  leg_err {r['leg_err']:5.2f}  "
                  f"success={rows[-1]['success']}  lean {rows[-1]['peak_lean']:5.1f}deg  "
                  f"clipped {rows[-1]['clip_pct']:5.1f}%  |a|max {rows[-1]['max_abs_action']:6.1f}")
            print("            max|q|: " + "  ".join(
                f"{n.replace('_joint','').replace('right_','R_').replace('left_','L_')}"
                f"={w[n]:.2f}" for n in watch))
            if sd == seeds[0]:
                top = np.argsort(-hook.over_any)[:6]
                print("            |a|>4 (past the effort limit): " + "  ".join(
                    f"{dof_names[i].replace('_joint','').replace('right_','R_').replace('left_','L_')}"
                    f"={100*hook.over_any[i]/max(hook.n,1):.0f}%" for i in top))
        summary[nm] = {
            "clip": c, "joints": list(subs), "rows": rows,
            "survival_mean": float(np.mean([x["survival"] for x in rows])),
            "survival_std": float(np.std([x["survival"] for x in rows])),
            "leg_err_mean": float(np.mean([x["leg_err"] for x in rows])),
            "success_rate": float(np.mean([x["success"] for x in rows])),
            "peak_lean_mean": float(np.mean([x["peak_lean"] for x in rows])),
            "clip_pct_mean": float(np.mean([x["clip_pct"] for x in rows])),
        }

    print(f"\n=== SUMMARY over {len(seeds)} seeds ===")
    print(f"  {'variant':11s} {'clip':>5s} {'joints':>18s} {'survival(s)':>15s} "
          f"{'leg_err':>8s} {'success':>8s} {'lean':>6s} {'clipped':>8s}")
    for nm, _, _ in variants:
        v = summary[nm]
        print(f"  {nm:11s} {v['clip']:5.1f} {('all' if not v['joints'] else '|'.join(v['joints'])):>18s} "
              f"{v['survival_mean']*ctrl_dt:6.2f} +/- {v['survival_std']*ctrl_dt:5.2f} "
              f"{v['leg_err_mean']:8.2f} {v['success_rate']*100:7.0f}% "
              f"{v['peak_lean_mean']:6.1f} {v['clip_pct_mean']:7.1f}%")

    (out_dir / "summary.json").write_text(json.dumps(
        {"seeds": seeds, "steps": args.steps, "ctrl_dt": ctrl_dt,
         "ckpt": args.ckpt, "dr": args.dr, "obs_noise": args.obs_noise,
         "policy_source": args.policy_source, "variants": summary}, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")
    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
