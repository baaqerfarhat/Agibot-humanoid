#!/usr/bin/env python3
"""Produce the rollout log requested in WHAT_I_NEED_FROM_ISAAC.md section 1.

Based on his `ISAAC_TEST/box_eval_isaac.py`, with three additions:

  * the motion clip is PINNED. v31's config sets `motion_dir` (box_multispeed),
    which takes precedence over `motion_file`, so eval otherwise hands the
    recorded env a RANDOM one of the three clips. His script does not pin it.
  * per-term termination logging, so the log says which term ended the episode.
  * everything is merged into one npz, including the per-term observation trace.

Base pose was already recorded by EvalRecordingCallback (`root_pos`,
`root_quat_xyzw`); his copy of the overlay predates that.

  OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \\
    /home/baaqer/.holosoma_deps/miniconda3/envs/hssim/bin/python \\
    adaptation/dump_for_mentor.py
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

CKPT = (
    "/home/baaqer/baaqer_ws/holosoma/logs/WholeBodyTracking/"
    "20260730_215012-x2_box_v31_flatfoot-locomotion/model_202500.pt"
)
# v31's actual reference: 734 frames, matches the policy npz's exported reference
# exactly, and carries `object_pos_w` (the true box trajectory).
MOTION = (
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
    "x2_31dof/whole_body_tracking/box_multispeed/box_speed100.npz"
)

TERM_LOG: list[tuple[int, str]] = []
STEP = [0]


def patch_termination_logging():
    """Record which termination term fired, without re-evaluating stateful terms."""
    from holosoma.managers.termination.manager import TerminationManager

    original_init_terms = TerminationManager._initialize_terms
    original_check = TerminationManager.check

    def wrap(name, fn):
        def recorder(env, **kwargs):
            out = fn(env, **kwargs)
            try:
                if bool(out.any()):
                    TERM_LOG.append((STEP[0], name))
            except Exception:
                pass
            return out

        return recorder

    def patched_init_terms(self):
        original_init_terms(self)
        for name in list(self._term_instances):
            self._term_instances[name] = wrap(name, self._term_instances[name])
        for name in list(self._term_funcs):
            self._term_funcs[name] = wrap(name, self._term_funcs[name])

    def patched_check(self):
        STEP[0] += 1
        return original_check(self)

    TerminationManager._initialize_terms = patched_init_terms
    TerminationManager.check = patched_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--motion", default=MOTION)
    ap.add_argument("--steps", type=int, default=734)
    ap.add_argument("--out", default="/home/baaqer/baaqer_ws/Agibot-humanoid/"
                                     "adaptation/FOR_MENTOR/isaac_v31_rollout.npz")
    args = ap.parse_args()

    os.chdir("/home/baaqer/baaqer_ws/holosoma")
    sys.path.insert(0, "/home/baaqer/baaqer_ws/holosoma/src/holosoma")

    from holosoma.config_types.eval_callback import (
        EvalCallbacksConfig,
        RecordingCallbackConfig,
        RecordingConfig,
    )
    from holosoma.eval_agent import run_eval_with_tyro
    from holosoma.managers.observation import manager as obs_manager
    from holosoma.utils.eval_utils import (
        CheckpointConfig,
        init_eval_logging,
        load_saved_experiment_config,
    )

    init_eval_logging()
    patch_termination_logging()

    checkpoint_cfg = CheckpointConfig(checkpoint=args.ckpt)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    print(f"[dump] simulator -> {saved_cfg.simulator._target_} "
          f"(name={saved_cfg.simulator.config.name})")

    motion_term = saved_cfg.command.setup_terms["motion_command"]
    mc = motion_term.params["motion_config"]
    pins = dict(
        use_adaptive_timesteps_sampler=False,
        start_at_timestep_zero_prob=1.0,
        freeze_at_timestep_zero_prob=0.0,
        motion_file=args.motion,
        motion_dir="",
    )
    if isinstance(mc, dict):
        mc = dict(mc)
        mc.update(pins)
        noise = dict(mc.get("noise_to_initial_pose") or {})
        noise["overall_noise_scale"] = 0.0
        mc["noise_to_initial_pose"] = noise
        motion_term.params["motion_config"] = mc
    else:
        motion_term.params["motion_config"] = dataclasses.replace(
            mc, **pins,
            noise_to_initial_pose=dataclasses.replace(
                mc.noise_to_initial_pose, overall_noise_scale=0.0),
        )
    print(f"[dump] motion PINNED -> {Path(args.motion).name} (motion_dir cleared)")

    saved_cfg.termination.terms.pop("bad_tracking", None)
    print("[dump] dropped bad_tracking so the clip plays end to end")

    for group_name, group in saved_cfg.observation.groups.items():
        if getattr(group, "enable_noise", False):
            object.__setattr__(group, "enable_noise", False)
            print(f"[dump] disabled observation noise on '{group_name}'")

    eval_cfg = saved_cfg.get_eval_config()
    object.__setattr__(eval_cfg.training, "headless", True)
    object.__setattr__(eval_cfg.training, "num_envs", 1)
    object.__setattr__(eval_cfg.training, "max_eval_steps", args.steps)
    object.__setattr__(eval_cfg.training, "seed", 42)

    trace = collections.defaultdict(list)
    original_compute_term = obs_manager.ObservationManager._compute_term

    def recording_compute_term(self, group_name, term_name, term_cfg):
        out = original_compute_term(self, group_name, term_name, term_cfg)
        if group_name == "actor_obs":
            trace[term_name].append(out.detach().cpu().numpy().copy())
        return out

    obs_manager.ObservationManager._compute_term = recording_compute_term

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_cbs_cfg = EvalCallbacksConfig(
        recording=RecordingCallbackConfig(
            config=RecordingConfig(enabled=True, output_path=str(out_path), env_id=0),
        ),
    )

    def merge():
        if not out_path.exists():
            print(f"[dump] {out_path} was never written; nothing to merge")
            return
        base = dict(np.load(out_path, allow_pickle=True))
        meta = json.loads(str(base.get("_metadata_json", "{}")))

        for term, vals in trace.items():
            base[f"obs__{term}"] = np.concatenate(vals, axis=0)

        order = sorted(trace.keys())
        meta["actor_obs_term_order_alphabetical"] = order
        meta["actor_obs_term_dims"] = {
            k: int(np.concatenate(trace[k], axis=0).shape[-1]) for k in order
        }
        meta["motion_file"] = args.motion
        meta["motion_frames"] = 734
        meta["checkpoint"] = args.ckpt
        meta["termination_events"] = [{"step": s, "term": t} for s, t in TERM_LOG]
        meta["termination_step"] = TERM_LOG[0][0] if TERM_LOG else None
        meta["termination_reason"] = TERM_LOG[0][1] if TERM_LOG else "none (ran to max_steps)"
        meta["obs_noise"] = False
        meta["seed"] = 42
        base["_metadata_json"] = np.array(json.dumps(meta))

        np.savez_compressed(out_path, **base)
        print(f"\n[dump] merged per-term obs + termination into {out_path}")
        for k in sorted(base):
            a = np.asarray(base[k])
            if a.ndim >= 1 and a.dtype != object and a.size > 1:
                print(f"    {k:<32} {str(a.shape):<14}")
        print(f"[dump] termination: {meta['termination_reason']} at step {meta['termination_step']}")

    # Isaac's shutdown path skips atexit, so chain the merge onto the recorder's own
    # save instead of registering it at exit.
    from holosoma.agents.callbacks.recording import EvalRecordingCallback

    original_save = EvalRecordingCallback._save

    def patched_save(self):
        original_save(self)
        merge()

    EvalRecordingCallback._save = patched_save

    run_eval_with_tyro(eval_cfg, checkpoint_cfg, saved_cfg, saved_wandb_path,
                       eval_cbs_cfg=eval_cbs_cfg)


if __name__ == "__main__":
    main()
