#!/usr/bin/env python3
"""Export the mjlab X2 squat PPO checkpoint to a numpy-only .npz for the robot.

Run inside the mjlab uv environment (needs torch + mjlab, not ROS):

    cd ~/baaqer_ws/mjlab
    uv run python ~/baaqer_ws/Agibot-humanoid/agibot_control_functions/export_squat_policy_npz.py \
        --checkpoint logs/rsl_rl/x2_squat/2026-08-26_23-04-28_deeper/model_16499.pt \
        --out ~/baaqer_ws/Agibot-humanoid/agibot_control_functions/policies/x2_squat_policy_40pct_iter16499.npz
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.squat.checkpoint import DEFAULT_CHECKPOINT, TASK_ID
from mjlab.tasks.squat.mdp.commands import (
  SquatCommandCfg,
  cycle_hold_time_s,
)
from mjlab.utils.torch import configure_torch_backends


def _extract_actor(ckpt: dict) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
  sd = ckpt["actor_state_dict"]
  mean = sd["obs_normalizer._mean"].detach().cpu().numpy().reshape(-1).astype(np.float32)
  # EmpiricalNormalization: (x - mean) / (std + eps) with eps=1e-2. Bake eps into
  # the saved std so deploy_x2_walk-style numpy inference matches torch.
  std = sd["obs_normalizer._std"].detach().cpu().numpy().reshape(-1).astype(np.float32) + 1e-2
  layer_ids = sorted(
    int(k.split(".")[1])
    for k in sd
    if k.startswith("mlp.") and k.endswith(".weight")
  )
  weights = [sd[f"mlp.{i}.weight"].detach().cpu().numpy().astype(np.float32) for i in layer_ids]
  biases = [sd[f"mlp.{i}.bias"].detach().cpu().numpy().astype(np.float32) for i in layer_ids]
  return mean, std, weights, biases


def _numpy_forward(mean, std, weights, biases, obs: np.ndarray) -> np.ndarray:
  x = (obs.astype(np.float32) - mean) / std
  for i in range(len(weights) - 1):
    x = x @ weights[i].T + biases[i]
    x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)
  return x @ weights[-1].T + biases[-1]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
  ap.add_argument("--out", required=True)
  ap.add_argument("--task-id", default=TASK_ID)
  ap.add_argument("--device", default=None)
  ap.add_argument("--squat-height-frac", type=float, default=0.40)
  args = ap.parse_args()

  configure_torch_backends()
  device = args.device or ("cuda:1" if torch.cuda.is_available() else "cpu")
  ckpt_path = Path(args.checkpoint)
  if not ckpt_path.exists():
    raise FileNotFoundError(ckpt_path)

  env_cfg = load_env_cfg(args.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = False
  squat_cmd = env_cfg.commands["squat"]
  assert isinstance(squat_cmd, SquatCommandCfg)
  squat_cmd.squat_height_frac = args.squat_height_frac
  squat_cmd.wrap_cycle = False

  agent_cfg = load_rl_cfg(args.task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task_id)
  runner = runner_cls(wrapped, copy.deepcopy(asdict(agent_cfg)), device=device)
  runner.load(
    str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  torch_policy = runner.get_inference_policy(device=device)

  ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
  mean, std, weights, biases = _extract_actor(ckpt)
  meta = get_base_metadata(env, str(ckpt_path))
  hold_t = cycle_hold_time_s(
    squat_cmd.cycle_time_s,
    squat_cmd.stand_duration_s,
    squat_cmd.descend_duration_s,
    squat_cmd.bottom_duration_s,
    squat_cmd.ascend_duration_s,
  )
  extra = {
    "task": "x2_squat",
    "control_hz": 50,
    "standing_height": float(squat_cmd.standing_height),
    "squat_height_frac": float(squat_cmd.squat_height_frac),
    "cycle_time_s": float(squat_cmd.cycle_time_s),
    "stand_duration_s": float(squat_cmd.stand_duration_s),
    "descend_duration_s": float(squat_cmd.descend_duration_s),
    "bottom_duration_s": float(squat_cmd.bottom_duration_s),
    "ascend_duration_s": float(squat_cmd.ascend_duration_s),
    "wrap_cycle": bool(squat_cmd.wrap_cycle),
    "hold_time_s": float(hold_t),
    "obs_dim": int(mean.shape[0]),
    "action_dim": int(weights[-1].shape[0]),
    "iteration": int(ckpt.get("iter", -1)),
    "notes": (
      "In-place 40% squat (pelvis 0.69 m -> 0.276 m -> stand). One 5 s cycle, "
      "then hold standing. Actor obs has no base_lin_vel. Training IMU is the "
      "pelvis; deploy should reconstruct pelvis gyro/attitude from the torso IMU."
    ),
  }
  meta.update(extra)
  njoints = len(meta["joint_names"])
  for key in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping"):
    if len(meta[key]) != njoints:
      raise RuntimeError(f"{key} length {len(meta[key])} != njoints {njoints}")
  if meta["obs_dim"] != 102 or meta["action_dim"] != 31:
    raise RuntimeError(
      f"unexpected shapes obs_dim={meta['obs_dim']} action_dim={meta['action_dim']}"
    )

  obs_td = wrapped.get_observations()
  actor_obs = obs_td["actor"]
  with torch.inference_mode():
    torch_act = torch_policy(obs_td).detach().cpu().numpy().reshape(-1)
  np_act = _numpy_forward(
    mean, std, weights, biases, actor_obs.detach().cpu().numpy().reshape(-1)
  )
  err = float(np.max(np.abs(torch_act - np_act)))
  print(f"[export] numpy vs torch max |action| err = {err:.3e}")
  if err > 1e-4:
    raise RuntimeError(f"exported MLP does not match the inference policy (err={err})")

  class _Enc(json.JSONEncoder):
    def default(self, o):
      if isinstance(o, np.ndarray):
        return o.tolist()
      if isinstance(o, (np.floating, np.integer)):
        return o.item()
      return super().default(o)

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  save = {
    "mean": mean,
    "std": std,
    "n_layers": np.array(len(weights), dtype=np.int64),
    "meta_json": np.array(json.dumps(meta, cls=_Enc)),
  }
  for i, (w, b) in enumerate(zip(weights, biases)):
    save[f"W{i}"] = w
    save[f"b{i}"] = b
  np.savez(out, **save)

  print(f"[export] wrote {out}")
  print(f"[export] obs_dim={meta['obs_dim']}  action_dim={meta['action_dim']}  "
        f"layers={[w.shape for w in weights]}")
  print(f"[export] observation_names = {meta['observation_names']}")
  print(f"[export] squat {meta['squat_height_frac']:.2f} of {meta['standing_height']:.3f} m  "
        f"cycle={meta['cycle_time_s']}s  hold_t={meta['hold_time_s']}s")


if __name__ == "__main__":
  main()
