"""Check the mjlab harness's observation builder against Isaac's own rollout.

The harness is only worth anything if it feeds the policy the same 164 numbers
holosoma does. Isaac logged the state and the action it produced, so replaying the
state through this builder and comparing the predicted action to the logged one is a
direct test, with no simulator in the loop. A deterministic MLP means a correct
builder has to reproduce the action to numerical precision.

Several conventions are ambiguous from the outside -- whether the logged arrays are
pre- or post-step, which frame base_ang_vel lives in -- so this greps the small grid
of plausible readings and reports which one lands.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_box_mjlab import (  # noqa: E402
    DEFAULT_POLICY,
    Policy,
    quat_inv,
    quat_mul,
    quat_to_mat,
)

ISAAC = HERE / "sim_rollouts" / "x2_box_walk_retimed_v19_iter85500_rollout.npz"


def main() -> None:
    pol = Policy(DEFAULT_POLICY)
    default = np.asarray(pol.meta["default_joint_pos"], np.float32)

    d = np.load(ISAAC, allow_pickle=True)
    md = json.loads(str(d["_metadata_json"]))
    bodies = list(md["body_names"])
    ti = bodies.index("torso_link")

    q = d["dof_pos"]
    dq = d["dof_vel"]
    act = d["actions"]
    wq = d["root_ang_vel"]
    rquat = d["root_quat_xyzw"]
    bquat = d["body_quat_xyzw"]
    n = len(act)
    ks = np.arange(5, min(n - 2, 260))  # before any large divergence

    def rot(qxyzw):
        return quat_to_mat(np.asarray(qxyzw, np.float32))

    best = None
    for state_lag in (0, 1):        # is dof_pos[k] the state that produced act[k]?
        for ref_off in (0, 1):      # which clip frame the command points at
            for wframe in ("raw", "body"):
                errs = []
                for k in ks:
                    si = k - state_lag
                    if si < 1:
                        continue
                    w = np.asarray(wq[si], np.float32)
                    if wframe == "body":
                        w = rot(rquat[si]).T @ w
                    f = min(si + ref_off, pol.ref_q.shape[0] - 1)
                    q_ref = pol.ref_quat[f]
                    q_tor = np.asarray(bquat[si, ti], np.float32)
                    ori6 = quat_to_mat(quat_mul(quat_inv(q_tor), q_ref))[:, :2].reshape(-1)
                    obs = np.concatenate([
                        act[si - 1], w, q[si] - default, dq[si],
                        pol.ref_q[f], pol.ref_dq[f], ori6,
                    ]).astype(np.float32)
                    errs.append(np.abs(pol(obs) - act[k]).max())
                e = float(np.mean(errs))
                tag = f"state_lag={state_lag} ref_off={ref_off} w={wframe:4s}"
                print(f"  {tag}: mean max|a_pred - a_logged| = {e:.4f}")
                if best is None or e < best[0]:
                    best = (e, tag)

    print()
    print(f"best: {best[1]}  ->  {best[0]:.4f}")
    if best[0] < 1e-3:
        print("VERDICT: observation builder reproduces Isaac exactly. Harness is sound;")
        print("         any divergence in the mjlab rollout is the plant.")
    else:
        print("VERDICT: builder does NOT reproduce Isaac. The mjlab rollout is not a")
        print("         controlled plant swap until this is closed.")


if __name__ == "__main__":
    main()
