#!/usr/bin/env python3
"""Print the PD gains (kp/kd) + action_scale the deploy script actually uses,
straight from the policy metadata, and the effective values after --gain-scale."""
import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "policies/x2_policy.npz"
gain_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

d = np.load(path, allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
jn = meta["joint_names"]
kp = meta["joint_stiffness"]
kd = meta["joint_damping"]
sc = meta["action_scale"]
if np.isscalar(sc):
    sc = [float(sc)] * len(jn)

print(f"policy: {path}    gain_scale = {gain_scale}")
print("=" * 92)
print(f"{'joint':<28} {'kp':>8} {'kd':>8} {'act_scale':>10} | "
      f"{'kp*scale':>9} {'kd*scale':>9}")
print("-" * 92)
for i, n in enumerate(jn):
    print(f"{n:<28} {kp[i]:>8.2f} {kd[i]:>8.2f} {sc[i]:>10.3f} | "
          f"{kp[i]*gain_scale:>9.2f} {kd[i]*gain_scale:>9.2f}")
print("=" * 92)
print(f"kp range: {min(kp):.1f}..{max(kp):.1f}   kd range: {min(kd):.1f}..{max(kd):.1f}")
