#!/usr/bin/env python3
"""Dump the policy's joint mapping + default pose and cross-check names against
the real robot_model, so we can verify we command the right joints."""
import json
import numpy as np
from robot_states_control import robot_model, JointArea

d = np.load("policies/x2_policy_original.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))

names = meta["joint_names"]
default = meta["default_joint_pos"]
scale = meta["action_scale"]
if np.isscalar(scale):
    scale = [float(scale)] * len(names)

# real robot joint names (in HEAD, WAIST, ARM, LEG order)
real = []
for area in (JointArea.HEAD, JointArea.WAIST, JointArea.ARM, JointArea.LEG):
    real += [j.name for j in robot_model[area]]
real_set = set(real)

print(f"policy joints: {len(names)}   real joints (head+waist+arm+leg): {len(real)}")
print("=" * 86)
print(f"{'idx':>3}  {'policy joint name':<28} {'default(rad)':>12} {'act_scale':>10}  in_robot?")
print("-" * 86)
missing = []
for i, n in enumerate(names):
    ok = "OK" if n in real_set else "** NOT FOUND **"
    if n not in real_set:
        missing.append(n)
    sc = scale[i] if i < len(scale) else scale[0]
    print(f"{i:>3}  {n:<28} {default[i]:>12.4f} {sc:>10.4f}  {ok}")

print("=" * 86)
print("policy joints NOT in robot_model:", missing or "(none)")
extra = [n for n in real if n not in set(names)]
print("robot joints the policy does NOT command:", extra or "(none)")

# Highlight the leg default pose (most relevant to 'stance')
print("\n--- LEG default pose (the stance it ramps into) ---")
for i, n in enumerate(names):
    if "hip" in n or "knee" in n or "ankle" in n:
        print(f"  {n:<26} {default[i]:+.4f} rad  ({np.degrees(default[i]):+6.1f} deg)")
