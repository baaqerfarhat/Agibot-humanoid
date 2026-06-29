#!/usr/bin/env python3
"""Convert an mjlab/rsl_rl exported policy (.onnx) into a self-contained .npz.

Run this in the mjlab `uv` environment (it has `onnx` + `numpy`):

    cd ~/baaqer_ws/mjlab
    uv run python ~/baaqer_ws/agibot_control_functions/export_policy_npz.py \
        --onnx logs/rsl_rl/x2_velocity_deploy/<run>/<run>.onnx \
        --out  ~/baaqer_ws/agibot_control_functions/x2_policy.npz

The resulting .npz holds the normalizer (mean/std), the MLP weights, AND all of
the deployment metadata (joint order, default pose, action scale, PD gains,
observation/command term order). The on-robot script `deploy_x2_walk.py` then
needs **only numpy** at runtime -- no onnxruntime, no torch, no onnx.

The network is the standard rsl_rl actor:
    x = (obs - mean) / std
    for W, b in hidden_layers:  x = elu(x @ W.T + b)
    action = x @ W_out.T + b_out
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnx
from onnx import numpy_helper


def _csv_floats(s: str) -> list[float]:
  return [float(x) for x in s.split(",") if x != ""]


def _csv_strs(s: str) -> list[str]:
  return [x for x in s.split(",") if x != ""]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True, help="Path to the exported .onnx policy.")
  ap.add_argument("--out", required=True, help="Output .npz path.")
  args = ap.parse_args()

  model = onnx.load(args.onnx)
  graph = model.graph

  inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

  # --- normalizer: find the Sub (mean) and Div (std) nodes ---
  mean = None
  std = None
  for node in graph.node:
    if node.op_type == "Sub" and node.input[1] in inits:
      mean = inits[node.input[1]].reshape(-1).astype(np.float32)
    elif node.op_type == "Div" and node.input[1] in inits:
      std = inits[node.input[1]].reshape(-1).astype(np.float32)
  if mean is None or std is None:
    raise RuntimeError("Could not locate obs normalizer (Sub/Div) in the graph.")

  # --- MLP layers: Gemm nodes in graph order ---
  weights: list[np.ndarray] = []
  biases: list[np.ndarray] = []
  for node in graph.node:
    if node.op_type == "Gemm":
      w = inits[node.input[1]].astype(np.float32)  # [out, in]
      b = inits[node.input[2]].astype(np.float32)  # [out]
      weights.append(w)
      biases.append(b)
  if not weights:
    raise RuntimeError("No Gemm layers found in the graph.")

  # --- metadata attached by mjlab's exporter ---
  md = {p.key: p.value for p in model.metadata_props}
  meta = {
    "joint_names": _csv_strs(md["joint_names"]),
    "default_joint_pos": _csv_floats(md["default_joint_pos"]),
    "action_scale": _csv_floats(md["action_scale"]),
    "joint_stiffness": _csv_floats(md["joint_stiffness"]),
    "joint_damping": _csv_floats(md["joint_damping"]),
    "observation_names": _csv_strs(md["observation_names"]),
    "command_names": _csv_strs(md.get("command_names", "")),
    "run_path": md.get("run_path", ""),
    "obs_dim": int(mean.shape[0]),
    "action_dim": int(weights[-1].shape[0]),
  }

  njoints = len(meta["joint_names"])
  for key in ("default_joint_pos", "action_scale", "joint_stiffness", "joint_damping"):
    if len(meta[key]) != njoints:
      raise RuntimeError(f"{key} length {len(meta[key])} != njoints {njoints}")

  save = {
    "mean": mean,
    "std": std,
    "n_layers": np.array(len(weights), dtype=np.int64),
    "meta_json": np.array(json.dumps(meta)),
  }
  for i, (w, b) in enumerate(zip(weights, biases)):
    save[f"W{i}"] = w
    save[f"b{i}"] = b

  np.savez(args.out, **save)

  print(f"[export] wrote {args.out}")
  print(f"[export] obs_dim={meta['obs_dim']}  action_dim={meta['action_dim']}  "
        f"layers={[w.shape for w in weights]}")
  print(f"[export] observation_names = {meta['observation_names']}")
  print(f"[export] command_names     = {meta['command_names']}")
  print(f"[export] njoints           = {njoints}")


if __name__ == "__main__":
  main()
