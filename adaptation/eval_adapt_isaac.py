#!/usr/bin/env python3
"""Deploy mentor ACE layer-adaptation into our Isaac / holosoma box-pickup eval.

Runs FROZEN then ADAPTED in one Isaac session (one startup). Records NPZs that
`box_pickup/render_box_rollout.py` can turn into videos.

Usage (hssim env):
  OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \\
    /home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python \\
    adaptation/eval_adapt_isaac.py \\
      [--ckpt model_202500.pt] [--steps 400] [--out-dir /tmp]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Mentor adapter lives next to this file.
HERE = Path(__file__).resolve().parent
PKG = HERE / "ACC_ADAPTATION_PACKAGE"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from ace_adapt import AdaptConfig, ExportedPolicy, LayerAdapter  # noqa: E402

DEFAULT_CKPT = str(paths.resolve_ckpt())
DEFAULT_POLICY_NPZ = str(paths.POLICY_NPZ)
DEFAULT_MOTION = str(paths.MOTION)

FALL_HEIGHT = 0.35
TRACK_LIMIT_DEG = 20.0


def _leg_idx(names: list[str]) -> list[int]:
    return [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]


def _rollout(algo, task, adapter, max_steps: int, ref_pos: np.ndarray, ctrl_dt: float,
             on_reset=None):
    """One closed-loop Isaac rollout. adapter=None => frozen torch policy.

    `on_reset` runs immediately after the reset, before the first action. Fault
    injection needs it because resetting restores the nominal actuator scales.
    """
    eval_policy = algo.get_inference_policy()
    obs_dict = task.reset_all()
    if on_reset is not None:
        on_reset()
    actor_state = {"done_indices": [], "stop": False, "obs": obs_dict}
    if adapter is not None:
        adapter.reset()

    cmd = task.command_manager.get_state("motion_command")
    names = list(algo._unwrap_env().simulator.dof_names)
    legs = _leg_idx(names)

    errs: list[float] = []
    tracked = None
    survival = max_steps
    records = {
        "dof_pos": [],
        "dof_vel": [],
        "actions": [],
        "root_pos": [],
        "root_quat_xyzw": [],
        "object_pos": [],
        "object_quat_wxyz": [],
        "leg_err_deg": [],
        "frame": [],
        "weight_drift": [],
    }

    # Optional box handle (same detection as EvalRecordingCallback).
    obj = None
    try:
        scene = getattr(task.simulator, "scene", None)
        rigid_objects = getattr(scene, "rigid_objects", None) if scene is not None else None
        if rigid_objects:
            for oname, o in rigid_objects.items():
                if oname in ("usd_scene_objects",):
                    continue
                obj = o
                break
    except Exception:
        obj = None

    import torch

    for step in range(max_steps):
        actor_obs = torch.cat([actor_state["obs"][k] for k in algo.actor_obs_keys], dim=1)
        if adapter is None:
            actions = eval_policy({"actor_obs": actor_obs})
        else:
            obs_np = actor_obs[0].detach().cpu().numpy().astype(np.float64)
            a_np = adapter.act(obs_np)
            actions = torch.as_tensor(a_np, device=actor_obs.device, dtype=actor_obs.dtype).unsqueeze(0)

        actor_state["actions"] = actions
        actor_state["step"] = step
        for c in algo.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)

        actor_state = algo.env_step(actor_state)

        for c in algo.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)

        sim = task.simulator
        q = sim.dof_pos[0].detach().cpu().numpy()
        frame = int(cmd.time_steps[0].item()) if hasattr(cmd, "time_steps") else step
        frame = min(frame, len(ref_pos) - 1)
        err_vec = q - ref_pos[frame]
        leg_err = float(np.degrees(np.abs(err_vec)[legs]).mean())
        errs.append(leg_err)

        if adapter is not None:
            adapter.update(err_vec, ctrl_dt)

        root = sim.robot_root_states[0].detach().cpu().numpy()
        records["dof_pos"].append(q.copy())
        records["dof_vel"].append(sim.dof_vel[0].detach().cpu().numpy().copy())
        records["actions"].append(actions[0].detach().cpu().numpy().copy())
        records["root_pos"].append(root[:3].copy())
        records["root_quat_xyzw"].append(root[3:7].copy())
        records["leg_err_deg"].append(leg_err)
        records["frame"].append(frame)
        records["weight_drift"].append(adapter.weight_drift if adapter is not None else 0.0)

        if obj is not None:
            try:
                st = obj.data.root_state_w[0].detach().cpu().numpy()
                records["object_pos"].append(st[:3].copy())
                records["object_quat_wxyz"].append(st[3:7].copy())
            except Exception:
                pass

        fallen = bool(root[2] < FALL_HEIGHT)
        if tracked is None and (fallen or leg_err > TRACK_LIMIT_DEG):
            tracked = step + 1
        if fallen:
            survival = step + 1
            break

        actor_state["obs"] = actor_state.get("obs", actor_state["obs"])

    mean_err = float(np.mean(errs)) if errs else float("nan")
    drift = adapter.weight_drift if adapter is not None else 0.0
    diverged = bool(adapter.diverged) if adapter is not None else False
    return {
        "survival": survival,
        "tracked": tracked if tracked is not None else survival,
        "leg_err": mean_err,
        "drift": drift,
        "diverged": diverged,
        "records": records,
        "steps": len(errs),
        "ctrl_dt": ctrl_dt,
        "dof_names": names,
        "box_present": obj is not None,
    }


def _save_npz(path: Path, result: dict, meta_extra: dict) -> None:
    rec = result["records"]
    payload = {k: np.asarray(v) for k, v in rec.items() if len(v)}
    meta = {
        "dt": result["ctrl_dt"],
        "fps": round(1.0 / result["ctrl_dt"]),
        "dof_names": result["dof_names"],
        "survival_steps": result["survival"],
        "tracked_steps": result["tracked"],
        "mean_leg_err_deg": result["leg_err"],
        "weight_drift": result["drift"],
        "diverged": result["diverged"],
        "box_present": result["box_present"],
        **meta_extra,
    }
    payload["_metadata_json"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"wrote {path}  ({result['steps']} steps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--policy-npz", default=DEFAULT_POLICY_NPZ)
    ap.add_argument("--motion", default=DEFAULT_MOTION)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--skip-adapt", action="store_true")
    ap.add_argument("--gains", default="3e-4",
                    help="comma-separated Gamma values to sweep")
    ap.add_argument("--leak-mode", default="zero", choices=["zero", "w0"],
                    help="sigma-modification target: -gamma*W (zero) or -gamma*(W-W0)")
    args = ap.parse_args()

    # Holosoma imports must happen after Isaac / omni bootstrap inside helpers.
    paths.enter_holosoma()

    from holosoma.utils.eval_utils import (  # noqa: E402
        CheckpointConfig,
        init_eval_logging,
        load_checkpoint,
        load_saved_experiment_config,
    )
    from holosoma.utils.helpers import get_class  # noqa: E402
    from holosoma.utils.sim_utils import (  # noqa: E402
        close_simulation_app,
        setup_simulation_environment,
    )
    from holosoma.utils.config_utils import CONFIG_NAME  # noqa: E402
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
    from holosoma.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    import dataclasses  # noqa: E402

    init_eval_logging()

    checkpoint_cfg = CheckpointConfig(checkpoint=args.ckpt)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    # Demo mode: start at t=0, no noise, no bad_tracking termination.
    motion_term = saved_cfg.command.setup_terms["motion_command"]
    motion_config = motion_term.params["motion_config"]
    if isinstance(motion_config, dict):
        motion_config = dict(motion_config)
        motion_config["use_adaptive_timesteps_sampler"] = False
        motion_config["start_at_timestep_zero_prob"] = 1.0
        motion_config["freeze_at_timestep_zero_prob"] = 0.0
        noise = dict(motion_config.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        motion_config["noise_to_initial_pose"] = noise
        motion_config["motion_file"] = args.motion
        motion_config["motion_dir"] = ""
        motion_term.params["motion_config"] = motion_config
    else:
        motion_term.params["motion_config"] = dataclasses.replace(
            motion_config,
            use_adaptive_timesteps_sampler=False,
            start_at_timestep_zero_prob=1.0,
            freeze_at_timestep_zero_prob=0.0,
            noise_to_initial_pose=dataclasses.replace(
                motion_config.noise_to_initial_pose, overall_noise_scale=0.0
            ),
            motion_file=args.motion,
            motion_dir="",
        )
    saved_cfg.termination.terms.pop("bad_tracking", None)

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", args.steps)

    print(f"[adapt-isaac] ckpt={args.ckpt}")
    print(f"[adapt-isaac] policy_npz={args.policy_npz}")
    print(f"[adapt-isaac] motion={args.motion}")
    print(f"[adapt-isaac] steps={args.steps}")

    env, device, simulation_app = setup_simulation_environment(eval_cfg)
    eval_log_dir = get_experiment_dir(eval_cfg.logger, eval_cfg.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.save_config(str(eval_log_dir / CONFIG_NAME))

    # Recording disabled here — we write our own NPZs with metrics.
    object.__setattr__(eval_cfg.algo.config, "eval_callbacks", {})

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    algo_class = get_class(eval_cfg.algo._target_)
    algo: BaseAlgo = algo_class(
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

    out_dir = Path(args.out_dir)
    summary = []

    if not args.skip_frozen:
        print("\n=== FROZEN (torch policy, Isaac + box) ===")
        r = _rollout(algo, task, None, args.steps, ref_pos, ctrl_dt)
        print(
            f"  survival {r['survival']:4d} ({r['survival']*ctrl_dt:5.2f}s)  "
            f"tracked {r['tracked']:4d}  legErr {r['leg_err']:6.2f} deg  "
            f"box={r['box_present']}"
        )
        _save_npz(
            out_dir / "isaac_box_frozen.npz",
            r,
            {"mode": "frozen", "ckpt": args.ckpt, "policy_npz": args.policy_npz},
        )
        summary.append(("frozen", r))

    # NPZ frozen control: same ExportedPolicy path as adaptation, no weight updates.
    # Isolates ".npz vs torch" from "adaptation helps/hurts".
    class _FrozenNpz:
        diverged = False
        weight_drift = 0.0

        def __init__(self, policy: ExportedPolicy):
            self.pol = policy

        def reset(self):
            pass

        def act(self, obs):
            return self.pol.forward(obs)[0]

        def update(self, *_a, **_k):
            pass

    print("\n=== FROZEN-NPZ (exported .npz, no adapt, Isaac + box) ===")
    r = _rollout(algo, task, _FrozenNpz(pol), args.steps, ref_pos, ctrl_dt)
    print(
        f"  survival {r['survival']:4d} ({r['survival']*ctrl_dt:5.2f}s)  "
        f"tracked {r['tracked']:4d}  legErr {r['leg_err']:6.2f} deg  box={r['box_present']}"
    )
    _save_npz(
        out_dir / "isaac_box_frozen_npz.npz",
        r,
        {"mode": "frozen_npz", "ckpt": args.ckpt, "policy_npz": args.policy_npz},
    )
    summary.append(("frozen_npz", r))

    if not args.skip_adapt:
        # Isaac applies target = a*scale + default with no leg filter and no rate
        # limit, so a given weight change moves the legs ~5x more than in the
        # reference MuJoCo loop the default Gamma was tuned against. Sweep Gamma.
        class _Adapter(LayerAdapter):
            def update(self, joint_error, dt):
                if args.leak_mode == "zero":
                    return super().update(joint_error, dt)
                from ace_adapt import _elu_jacobian

                self.step += 1
                if self.diverged or self.step <= self.cfg.engage_step or self._cache is None:
                    return
                a, z = self._cache
                L, layer = self.pol.n_layers, self.cfg.layer
                d = self.delta_L(joint_error)
                for l in range(L - 1, layer, -1):
                    d = _elu_jacobian(a[l - 1]) * (self.W[l].T @ d)
                Wdot = (self.cfg.gain * np.outer(d, z[layer])
                        - self.cfg.leak * (self.W[layer] - self.pol.W0[layer]))
                self.W[layer] = self.W[layer] + dt * Wdot
                if (not np.isfinite(self.W[layer]).all()
                        or self.weight_drift > self.cfg.max_weight_drift):
                    self.diverged = True
                    self.W[layer] = self.pol.W0[layer].copy()

        for gain in [float(g) for g in args.gains.split(",")]:
            tag = f"adapt_g{gain:g}_{args.leak_mode}"
            print(f"\n=== ADAPTED  Gamma={gain:g}  leak->{args.leak_mode}  (Isaac + box) ===")
            cfg = AdaptConfig(layer=2, gain=gain, leak=1e-2, gx_level=1,
                              error_joints=("hip", "knee", "ankle", "waist"), engage_step=0)
            adapter = _Adapter(pol, cfg, joint_names=pol.meta["joint_names"])
            r = _rollout(algo, task, adapter, args.steps, ref_pos, ctrl_dt)
            print(
                f"  survival {r['survival']:4d} ({r['survival']*ctrl_dt:5.2f}s)  "
                f"tracked {r['tracked']:4d}  legErr {r['leg_err']:6.2f} deg  "
                f"|dW| {r['drift']:.3f}  diverged={r['diverged']}  box={r['box_present']}"
            )
            _save_npz(
                out_dir / f"isaac_box_{tag}.npz",
                r,
                {
                    "mode": tag,
                    "ckpt": args.ckpt,
                    "policy_npz": args.policy_npz,
                    "adapt": {"layer": cfg.layer, "gain": cfg.gain, "leak": cfg.leak,
                              "gx_level": cfg.gx_level, "leak_mode": args.leak_mode},
                },
            )
            summary.append((tag, r))

    print("\n=== SUMMARY ===")
    for name, r in summary:
        print(
            f"  {name:8s}  surv={r['survival']*ctrl_dt:5.2f}s  "
            f"legErr={r['leg_err']:6.2f}deg  drift={r['drift']:.3f}  "
            f"box={r['box_present']}  diverged={r['diverged']}"
        )

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
