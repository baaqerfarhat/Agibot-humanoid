#!/usr/bin/env python3
"""Print obs terms / dims for a policy npz so we can confirm it's the deploy model."""
import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "policies/x2_policy.npz"
d = np.load(path, allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
print("file:           ", path)
print("observation_names:", meta["observation_names"])
print("obs_dim:        ", meta["obs_dim"])
print("action_dim:     ", meta["action_dim"])
print("run_path:       ", meta.get("run_path", "?"))
print("base_lin_vel in obs?", "base_lin_vel" in meta["observation_names"],
      "  <-- should be FALSE for the deploy policy")
