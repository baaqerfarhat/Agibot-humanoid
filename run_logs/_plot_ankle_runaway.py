#!/usr/bin/env python3
"""Plot the mechanism behind the 2026-08-12 falls: the ankle roll runs away.

Three panels, all against the reference frame index so sim and hardware line up:

  1. right ankle roll -- Isaac stays at the reference under the same commanded
     torque; hardware saturates at frame ~40 and stays pinned past its URDF stop
  2. the commanded action on that joint, against |a| = 4, the effort limit
  3. pelvis roll -- the lateral error that grows for ~100 frames and topples the
     robot, versus the same quantity in Isaac
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analyze_ff_runs import analyse  # noqa: E402
from _replay_deploy import HERE, REPO, Policy  # noqa: E402

JOINT = "right_ankle_roll_joint"
LIMIT = 0.2625
RUNS = [("20260812_132056", "hardware, supported (completed)", "tab:green"),
        ("20260812_132139", "hardware 13:21, unsupported (fell)", "tab:red"),
        ("20260812_132219", "hardware 13:22, unsupported (fell)", "tab:orange")]


def main():
    pol = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    jn = pol.meta["joint_names"]
    i = jn.index(JOINT)
    ascale = np.array(pol.meta["action_scale"], np.float64)
    default = np.array(pol.meta["default_joint_pos"], np.float64)

    d = np.load(os.path.join(REPO, "adaptation/isaac_runs/v33/isaac_frozen_npz_seed600.npz"),
                allow_pickle=True)
    sfr = d["frame"]
    sq = d["dof_pos"].astype(np.float64)
    sa = d["actions"].astype(np.float64)
    quat = d["root_quat_xyzw"].astype(np.float64)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sroll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    ax[0].axhspan(-LIMIT, LIMIT, color="0.92", zorder=0,
                  label=f"URDF range +-{LIMIT:.3f} rad")
    ax[0].plot(sfr, sq[:, i], color="tab:blue", lw=2.2, label="Isaac (same policy)")
    ax[1].plot(sfr, np.abs(sa[:, i]), color="tab:blue", lw=2.2, label="Isaac")
    ax[2].plot(sfr, sroll, color="tab:blue", lw=2.2, label="Isaac")

    for name, label, col in RUNS:
        R = analyse(os.path.join(HERE, name + "_box_pickup_x2_box_policy_v33_iter253000.csv"),
                    pol, verbose=False)
        fr = R["frame"]
        ax[0].plot(fr, R["q"][:, i], color=col, lw=1.6, label=label)
        a = (R["tgt"][:, i] - default[i]) / ascale[i]
        ax[1].plot(fr, np.abs(a), color=col, lw=1.3)
        ax[2].plot(fr, R["roll"], color=col, lw=1.4)
        if fr.max() < 500:  # aborted -> mark where
            for a_ in ax:
                a_.axvline(fr.max(), color=col, ls=":", lw=1.2)

    ax[0].set_ylabel("right ankle roll (rad)")
    ax[0].set_title("The right ankle roll runs to its stop on hardware and stays there.\n"
                    "Isaac holds it at the reference under the same commanded torque, "
                    "because the ground pushes back.", fontsize=11)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_ylim(-0.30, 0.40)

    ax[1].axhline(4.0, color="k", ls="--", lw=1.4,
                  label="|a| = 4, the effort limit (everything above is unachievable)")
    ax[1].set_ylabel("|action| on that joint")
    ax[1].set_ylim(0, 45)
    ax[1].legend(fontsize=8, loc="upper left")

    ax[2].axhline(0.7, color="k", ls="--", lw=1.0, label="deploy abort threshold")
    ax[2].axhline(-0.7, color="k", ls="--", lw=1.0)
    ax[2].set_ylabel("pelvis roll (rad)")
    ax[2].set_xlabel("reference motion frame (50 Hz)")
    ax[2].legend(fontsize=8, loc="lower left")

    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "ankle_roll_runaway.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
