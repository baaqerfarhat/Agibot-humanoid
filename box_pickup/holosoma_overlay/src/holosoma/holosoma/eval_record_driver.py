"""Headless policy rollout + trajectory recording driver.

Bypasses the tyro subcommand CLI (which conflicts with --training.* overrides)
by constructing the configs in code. Records robot + object states to an NPZ
for offscreen MuJoCo rendering.

Usage:
    python eval_record_driver.py <checkpoint.pt> <output.npz> [max_steps]
"""

from __future__ import annotations

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
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    # 4th arg "demo": start every episode at motion t=0 with zero init noise,
    # producing a clean beginning-to-end rollout instead of the training-style
    # random mid-motion starts.
    demo_mode = len(sys.argv) > 4 and sys.argv[4] == "demo"
    # Optional 5th arg: pin the demo to one specific motion .npz. Needed for
    # multi-motion runs (motion_dir), where eval otherwise assigns the recorded
    # env a RANDOM clip.
    motion_file_override = sys.argv[5] if len(sys.argv) > 5 else None

    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_path)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)

    if demo_mode:
        import dataclasses

        motion_term = saved_cfg.command.setup_terms["motion_command"]
        motion_config = motion_term.params["motion_config"]
        # The reloaded config may deserialize nested configs as plain dicts.
        if isinstance(motion_config, dict):
            motion_config = dict(motion_config)
            motion_config["use_adaptive_timesteps_sampler"] = False
            motion_config["start_at_timestep_zero_prob"] = 1.0
            # No random start-pose freezing in a demo take.
            motion_config["freeze_at_timestep_zero_prob"] = 0.0
            noise = dict(motion_config.get("noise_to_initial_pose") or {})
            noise["overall_noise_scale"] = 0.0
            motion_config["noise_to_initial_pose"] = noise
            if motion_file_override:
                motion_config["motion_file"] = motion_file_override
                motion_config["motion_dir"] = ""
            motion_term.params["motion_config"] = motion_config
        else:
            replacements = dict(
                use_adaptive_timesteps_sampler=False,
                start_at_timestep_zero_prob=1.0,
                freeze_at_timestep_zero_prob=0.0,
                noise_to_initial_pose=dataclasses.replace(
                    motion_config.noise_to_initial_pose, overall_noise_scale=0.0
                ),
            )
            if motion_file_override:
                replacements["motion_file"] = motion_file_override
                replacements["motion_dir"] = ""
            motion_term.params["motion_config"] = dataclasses.replace(motion_config, **replacements)
        # Drop early-termination so the rollout plays the full motion in one
        # continuous take (only the timeout term remains).
        saved_cfg.termination.terms.pop("bad_tracking", None)
        # Same intent, for terms whose implementation is not in this checkout:
        # the v17/v18 configs reference
        # holosoma.managers.termination.terms.wbt:HandGroundSupport, which is not in
        # the overlay, so the env cannot be built at all without dropping it. The
        # comment above already says only the timeout term should remain, so this
        # completes that rather than changing it -- but it is printed so a missing
        # implementation never passes unnoticed.
        import importlib as _il
        for _n in list(saved_cfg.termination.terms):
            _t = saved_cfg.termination.terms[_n]
            _f = _t.get("func") if isinstance(_t, dict) else getattr(_t, "func", "")
            if not isinstance(_f, str) or ":" not in _f:
                continue
            _m, _a = _f.split(":", 1)
            try:
                _ok = hasattr(_il.import_module(_m), _a)
            except Exception:
                _ok = True
            if not _ok:
                print(f"[record] DROPPING termination term {_n!r} -> {_f} "
                      f"(not available in this checkout)", flush=True)
                saved_cfg.termination.terms.pop(_n, None)

    eval_cfg = saved_cfg.get_eval_config()
    # Force headless (no RTX viewer -> no crash on this GPU), single env, finite rollout.
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", max_steps)

    eval_cbs_cfg = EvalCallbacksConfig(
        recording=RecordingCallbackConfig(
            config=RecordingConfig(enabled=True, output_path=output_npz, env_id=0),
        ),
    )

    run_eval_with_tyro(eval_cfg, checkpoint_cfg, saved_cfg, saved_wandb_path, eval_cbs_cfg=eval_cbs_cfg)


if __name__ == "__main__":
    main()
