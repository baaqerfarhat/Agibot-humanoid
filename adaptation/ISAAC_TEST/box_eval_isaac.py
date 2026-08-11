"""Evaluate the frozen box-pickup policy in IsaacSim -- the simulator it was trained in.

This is the reproduction control the MuJoCo backend could not provide.  Nothing about the
simulator, robot, task or reference motion is overridden: the checkpoint's saved config
already pins IsaacSim, which supports the scene object, so the box is physically present and
the reference motion is the trained `_w_obj` clip.

Only three things are changed, none of which touch a frozen policy's actions:

  * demo mode -- start at motion t=0 with no init noise, and drop the `bad_tracking`
    termination so the rollout plays the motion end to end rather than resetting early;
  * reward terms cleared -- the training config references reward code the public overlay does
    not ship (`penalty_foot_slip` exists in no commit; `motion_relative_body_position_error_exp`
    lacks `exclude_body_names`).  Reward is consumed by the learning update, which never runs
    here, so clearing is exactly neutral and avoids inventing signatures;
  * observation noise off -- the actor group trains with noise (dof_vel 0.5 rad/s), which has
    no place in a deterministic nominal check.

Usage: python box_eval_isaac.py <checkpoint.pt> <output.npz> [max_steps]
"""
from __future__ import annotations

import dataclasses
import sys

from holosoma.config_types.eval_callback import (
    EvalCallbacksConfig,
    RecordingCallbackConfig,
    RecordingConfig,
)
from holosoma.eval_agent import run_eval_with_tyro
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)


def main() -> None:
    init_eval_logging()
    checkpoint_path = sys.argv[1]
    output_npz = sys.argv[2]
    # 434 frames at 50 Hz control (simulator fps=200, control_decimation=4) = the full clip.
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 434

    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_path)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    print(f"[box-eval] simulator -> {saved_cfg.simulator._target_} "
          f"(name={saved_cfg.simulator.config.name})")

    motion_term = saved_cfg.command.setup_terms["motion_command"]
    motion_config = motion_term.params["motion_config"]
    if isinstance(motion_config, dict):
        motion_config = dict(motion_config)
        motion_config["use_adaptive_timesteps_sampler"] = False
        motion_config["start_at_timestep_zero_prob"] = 1.0
        noise = dict(motion_config.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        motion_config["noise_to_initial_pose"] = noise
        motion_term.params["motion_config"] = motion_config
        print(f"[box-eval] motion -> {str(motion_config['motion_file']).rsplit('/', 1)[-1]}")
    else:
        motion_term.params["motion_config"] = dataclasses.replace(
            motion_config,
            use_adaptive_timesteps_sampler=False,
            start_at_timestep_zero_prob=1.0,
            noise_to_initial_pose=dataclasses.replace(
                motion_config.noise_to_initial_pose, overall_noise_scale=0.0),
        )
    saved_cfg.termination.terms.pop("bad_tracking", None)

    n_rew = len(saved_cfg.reward.terms)
    saved_cfg.reward.terms.clear()
    print(f"[box-eval] cleared all {n_rew} reward terms (no effect on frozen-policy actions)")

    for group_name, group in saved_cfg.observation.groups.items():
        if getattr(group, "enable_noise", False):
            object.__setattr__(group, "enable_noise", False)
            print(f"[box-eval] disabled observation noise on '{group_name}'")

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", max_steps)

    # Same per-term observation trace as the MuJoCo driver, so the two backends can be diffed
    # term by term.  The MuJoCo run collapses ~1.1 s in while the reference timing, initial
    # pose, gains and DOF order all check out; whichever term first disagrees with IsaacSim
    # localises the remaining backend defect instead of leaving it to inspection.
    if "--dump-obs" in sys.argv:
        import collections

        import numpy as np

        from holosoma.managers.observation import manager as obs_manager

        trace = collections.defaultdict(list)
        original_compute_term = obs_manager.ObservationManager._compute_term

        def recording_compute_term(self, group_name, term_name, term_cfg):
            out = original_compute_term(self, group_name, term_name, term_cfg)
            if group_name == "actor_obs":
                trace[term_name].append(out.detach().cpu().numpy().copy())
            return out

        obs_manager.ObservationManager._compute_term = recording_compute_term

        import atexit

        def save_trace():
            if not trace:
                return
            arrays = {k: np.concatenate(v, axis=0) for k, v in trace.items()}
            out_path = output_npz.replace(".npz", "_obs.npz")
            np.savez(out_path, **arrays)
            print(f"[box-eval] wrote per-term observation trace -> {out_path}")
            for k, v in arrays.items():
                print(f"    {k:<22} {str(v.shape):<14} "
                      f"|.|max {np.abs(v).max():9.3f}  mean {v.mean():+8.4f}")

        atexit.register(save_trace)

    eval_cbs_cfg = EvalCallbacksConfig(
        recording=RecordingCallbackConfig(
            config=RecordingConfig(enabled=True, output_path=output_npz, env_id=0),
        ),
    )
    run_eval_with_tyro(eval_cfg, checkpoint_cfg, saved_cfg, saved_wandb_path,
                       eval_cbs_cfg=eval_cbs_cfg)


if __name__ == "__main__":
    main()
