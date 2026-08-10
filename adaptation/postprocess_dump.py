#!/usr/bin/env python3
"""Add causally-aligned observation arrays to the rollout dump.

The raw per-term trace has 738 rows against 734 recorded steps: the observation
manager also runs during reset/warmup. The offset was measured, not assumed —
`obs__actions[4+i] == actions[i]` to 0.0 and `obs__dof_pos[4+i] - dof_pos[i]` is a
constant (residual std 6e-7, i.e. exactly default_joint_pos).

Row r of the trace therefore holds last_action = actions[r-4], so the observation
that PRODUCED actions[j] is row j+3. That gives obs__*[3:737] as the 734-row
aligned block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / "FOR_MENTOR" / "isaac_v31_rollout.npz"
ALPHABETICAL = ["actions", "base_ang_vel", "dof_pos", "dof_vel",
                "motion_command", "motion_ref_ori_b"]
LAG = 3


def main() -> None:
    d = dict(np.load(NPZ, allow_pickle=True))
    meta = json.loads(str(d["_metadata_json"]))
    n_steps = int(d["actions"].shape[0])

    # Re-verify the offset rather than trusting the constant.
    check = np.abs(d["obs__actions"][LAG + 1: LAG + 1 + n_steps] - d["actions"][:n_steps]).mean()
    assert check == 0.0, f"alignment check failed: mean|diff| = {check}"
    print(f"alignment verified: obs__actions[{LAG + 1}+i] == actions[i] exactly")

    blocks = []
    for term in ALPHABETICAL:
        a = d[f"obs__{term}"][LAG: LAG + n_steps]
        assert a.shape[0] == n_steps, f"{term}: got {a.shape[0]} rows, want {n_steps}"
        d[f"obs_aligned__{term}"] = a
        blocks.append(a)
        print(f"  obs_aligned__{term:<18} {a.shape}")

    actor_obs = np.concatenate(blocks, axis=1)
    d["actor_obs_aligned"] = actor_obs
    print(f"  actor_obs_aligned          {actor_obs.shape}")
    assert actor_obs.shape[1] == 164, actor_obs.shape

    meta["alignment"] = {
        "raw_trace_rows": int(d["obs__actions"].shape[0]),
        "recorded_steps": n_steps,
        "rule": "obs_aligned__X[j] is the observation that produced actions[j]",
        "raw_index_of_aligned_j": f"obs__X[{LAG} + j]",
        "note": ("the raw trace has 4 extra leading rows from reset/warmup; verified via "
                 "obs__actions[4+i] == actions[i] (exact) and obs__dof_pos[4+i] - dof_pos[i] "
                 "== default_joint_pos (residual std 6e-7)"),
        "state_causality": ("obs_aligned__dof_pos[j] is built from the state BEFORE step j, "
                            "i.e. dof_pos[j-1]; dof_pos[j] is post-step"),
    }
    meta["actor_obs_concat_order"] = ALPHABETICAL
    d["_metadata_json"] = np.array(json.dumps(meta))

    np.savez_compressed(NPZ, **d)
    print(f"\nrewrote {NPZ}")


if __name__ == "__main__":
    main()
