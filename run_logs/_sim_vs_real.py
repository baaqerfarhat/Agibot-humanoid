#!/usr/bin/env python3
"""Compare the v33 policy's joint trajectories in Isaac against the real robot.

Both sides run the same network on the same reference motion, so any joint whose
sim and hardware traces disagree marks a remaining sim-to-real gap. Reporting
`sim - ref` alongside `real - ref` matters: a joint that sits on its mechanical
limit in BOTH is just how the policy behaves and is not a hardware problem,
whereas one that only saturates on hardware is.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analyze_clip import LIMITS  # noqa: E402
from _analyze_ff_runs import analyse  # noqa: E402
from _replay_deploy import HERE, REPO, Policy  # noqa: E402

SIM = os.path.join(REPO, "adaptation/isaac_runs/v33/isaac_frozen_npz_seed600.npz")


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    jn = policy.meta["joint_names"]
    lo = np.array([LIMITS[n][0] for n in jn])
    hi = np.array([LIMITS[n][1] for n in jn])
    ascale = np.array(policy.meta["action_scale"], np.float64)
    kp = np.array(policy.meta["joint_stiffness"], np.float64)
    default = np.array(policy.meta["default_joint_pos"], np.float64)

    d = np.load(SIM, allow_pickle=True)
    meta = json.loads(str(d["_metadata_json"]))
    print("sim rollout:", {k: meta[k] for k in list(meta)[:6] if k in meta})
    sim_q = d["dof_pos"].astype(np.float64)
    sim_a = d["actions"].astype(np.float64)
    sim_frame = d["frame"]
    sim_tgt = sim_a * ascale + default

    real_name = sys.argv[1] if len(sys.argv) > 1 else "20260812_132056"
    R = analyse(os.path.join(HERE, real_name + "_box_pickup_x2_box_policy_v33_iter253000.csv"),
                policy, verbose=False)
    frame, q, tgt = R["frame"], R["q"], R["tgt"]
    T = policy.ref_joint_pos.shape[0]
    ref = policy.ref_joint_pos[np.minimum(frame, T - 1)]

    # align on the reference frame index, only where both have data
    n = min(sim_frame.max(), frame.max())
    m_real = frame <= n
    sim_ref = policy.ref_joint_pos[np.minimum(sim_frame, T - 1)]
    m_sim = sim_frame <= n

    print(f"comparing frames 0-{n}  (sim {m_sim.sum()} ticks, real {m_real.sum()} ticks)")
    print()
    print(f"  {'joint':26s}{'sim-ref':>9s}{'real-ref':>10s}{'gap':>8s}"
          f"{'sim@lim':>9s}{'real@lim':>10s}  note")
    sd = sim_q[m_sim] - sim_ref[m_sim]
    rd = q[m_real] - ref[m_real]
    gap = np.abs(rd).mean(axis=0) - np.abs(sd).mean(axis=0)
    for i in np.argsort(-gap)[:14]:
        s_lim = 100.0 * ((sim_q[m_sim, i] <= lo[i] + 0.02) |
                         (sim_q[m_sim, i] >= hi[i] - 0.02)).mean()
        r_lim = 100.0 * ((q[m_real, i] <= lo[i] + 0.02) |
                         (q[m_real, i] >= hi[i] - 0.02)).mean()
        note = ""
        if r_lim > 40 and s_lim < 10:
            note = "REAL-ONLY saturation -> hardware gap"
        elif r_lim > 40 and s_lim > 40:
            note = "saturates in sim too -> policy behaviour, not a defect"
        elif gap[i] > 0.3:
            note = "diverges on hardware"
        print(f"  {jn[i]:26s}{np.abs(sd[:,i]).mean():9.2f}{np.abs(rd[:,i]).mean():10.2f}"
              f"{gap[i]:+8.2f}{s_lim:8.0f}%{r_lim:9.0f}%  {note}")

    print()
    print("  action magnitude, sim vs real (the policy's own output):")
    Rfull = R
    # recover the real action from the logged target: tgt = a*scale + default
    real_a = (tgt[m_real] - default) / ascale
    print(f"      sim  |a| mean {np.abs(sim_a[m_sim]).mean():6.2f}  "
          f"max {np.abs(sim_a[m_sim]).max():7.1f}")
    print(f"      real |a| mean {np.abs(real_a).mean():6.2f}  max {np.abs(real_a).max():7.1f}")
    print("      joints where the real action is much larger (policy working harder):")
    da = np.abs(real_a).mean(axis=0) - np.abs(sim_a[m_sim]).mean(axis=0)
    for i in np.argsort(-da)[:8]:
        print(f"        {jn[i]:26s} sim {np.abs(sim_a[m_sim,i]).mean():7.2f}  "
              f"real {np.abs(real_a[:,i]).mean():7.2f}   x{np.abs(real_a[:,i]).mean()/max(1e-6,np.abs(sim_a[m_sim,i]).mean()):5.1f}")

    print()
    print("  per-joint sim vs real measured position at a few frames:")
    watch = ["right_ankle_roll_joint", "left_ankle_roll_joint", "waist_pitch_joint",
             "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
             "left_knee_joint", "right_knee_joint", "left_hip_pitch_joint"]
    fr_pts = [f for f in (0, 40, 80, 120, 160, 200, 260, 320, 420, 520) if f <= n]
    print(f"      {'joint':26s}" + "".join(f"{f:>13d}" for f in fr_pts))
    for nm in watch:
        i = jn.index(nm)
        cells = ""
        for f in fr_pts:
            ks = int(np.argmin(np.abs(sim_frame - f)))
            kr = int(np.argmin(np.abs(frame - f)))
            cells += f"{sim_q[ks,i]:+6.2f}/{q[kr,i]:+6.2f}"
        print(f"      {nm:26s}{cells}")
    print("      (sim/real, rad)")


if __name__ == "__main__":
    main()
