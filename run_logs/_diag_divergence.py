#!/usr/bin/env python3
"""Split the reference-tracking error into the two things it can mean.

    q - ref  =  (q - tgt)  +  (tgt - ref)
                 servo        policy

`q - tgt` is the actuator failing to reach the commanded position (torque limit,
friction, a hard stop, a bad limit table). `tgt - ref` is the policy choosing to
command something other than the reference, which is legitimate feedback control
unless it runs away.

Confusing the two sends you fixing the wrong layer, so report them separately per
joint, plus the joints whose measured position sits on a mechanical limit.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analyze_clip import LIMITS  # noqa: E402
from _analyze_ff_runs import analyse  # noqa: E402
from _replay_deploy import HERE, REPO, Policy  # noqa: E402


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    jn = policy.meta["joint_names"]
    lo = np.array([LIMITS[n][0] for n in jn])
    hi = np.array([LIMITS[n][1] for n in jn])

    files = sys.argv[1:] or ["20260812_132056", "20260812_132139", "20260812_132219"]
    for f in files:
        p = os.path.join(HERE, f + "_box_pickup_x2_box_policy_v33_iter253000.csv")
        R = analyse(p, policy, verbose=False)
        if R is None:
            continue
        frame, q, tgt = R["frame"], R["q"], R["tgt"]
        T = policy.ref_joint_pos.shape[0]
        ref = policy.ref_joint_pos[np.minimum(frame, T - 1)]
        servo = q - tgt
        devia = tgt - ref
        total = q - ref
        tau, lim = R["tau_cmd"], R["eff_lim"]

        print("=" * 104)
        print(f"{R['name']}   frames 0-{frame.max()} ({frame.max()/50:.2f}s)")
        print("=" * 104)
        print(f"  {'joint':26s}{'q-ref':>8s}{'q-tgt':>8s}{'tgt-ref':>9s}"
              f"{'at lim':>8s}{'tau/lim':>9s}  verdict")
        for i in np.argsort(-np.abs(total).mean(axis=0))[:12]:
            n = jn[i]
            atlim = 100.0 * ((q[:, i] <= lo[i] + 0.02) | (q[:, i] >= hi[i] - 0.02)).mean()
            tsat = 100.0 * (np.abs(tau[:, i]) > lim[i] - 1e-3).mean()
            s, d = np.abs(servo[:, i]).mean(), np.abs(devia[:, i]).mean()
            if s > 2 * d and tsat > 20:
                v = "SERVO, torque-saturated -> actuator cannot reach it"
            elif s > 2 * d:
                v = "SERVO, torque spare -> hard stop / wrong limit / friction"
            elif d > 2 * s:
                v = "policy deviating from reference (feedback)"
            else:
                v = "mixed"
            print(f"  {n:26s}{np.abs(total[:,i]).mean():8.2f}{s:8.2f}{d:9.2f}"
                  f"{atlim:7.0f}%{tsat:8.0f}%  {v}")

        # joints parked on a mechanical limit -- these poison the observation
        print("  measured position sitting on a mechanical limit >20% of the run:")
        any_ = False
        for i in range(len(jn)):
            f_lo = (q[:, i] <= lo[i] + 0.02).mean()
            f_hi = (q[:, i] >= hi[i] - 0.02).mean()
            if max(f_lo, f_hi) > 0.2:
                any_ = True
                side, fr = ("lower", f_lo) if f_lo > f_hi else ("upper", f_hi)
                print(f"      {jn[i]:26s} {side} limit "
                      f"{lo[i] if side=='lower' else hi[i]:+.3f}  {100*fr:5.1f}% of ticks  "
                      f"ref wants {ref[:,i].min():+.2f}..{ref[:,i].max():+.2f}")
        if not any_:
            print("      none")
        print()


if __name__ == "__main__":
    main()
