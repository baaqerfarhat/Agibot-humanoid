#!/usr/bin/env python3
"""Figures for the crossed command-stream replay (results/fya_crossed_replay_v1/figures/).

  crossed_cells.pdf    four-cell costs per key (J00, J10, J01, J11; lines connect the cells of one key), medians marked
  crossed_effects.pdf  per-key D0, R1, T and I (dots) with the equal-task mean and task-clustered 95 % interval; the
                       registered margin +-delta_I shaded for I
  crossed_by_task.pdf  I per task (two states each) and the leave-one-task-out means
"""
import argparse, csv, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = ["J00", "J10", "J01", "J11"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/fya_crossed_replay_v1"))
    ap.add_argument("--analysis", default="analysis"); ap.add_argument("--tag", default=""); a = ap.parse_args()
    R = a.root; F = R / "figures"; F.mkdir(exist_ok=True); tag = a.tag
    eff = list(csv.DictReader(open(R / a.analysis / "source_effects.csv"))); iv = json.load(open(R / a.analysis / "source_intervals.json"))
    # cells
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for r in eff:
        ax.plot(range(4), [float(r[c]) * 1e5 for c in CELLS], color="#999999", lw=.7, alpha=.8)
    med = [iv["cells"][c]["J_median"] * 1e5 for c in CELLS]; ax.plot(range(4), med, "ko-", ms=5, label="median")
    ax.set_xticks(range(4)); ax.set_xticklabels(["J00\nM0, off", "J10\nM0, NT", "J01\nM1, off", "J11\nM1, NT"]); ax.set_ylabel("cost $J$ ($10^{-5}$ m$^2$ s)"); ax.legend(fontsize=8)
    ax.set_title(f"Four-cell costs, {iv['n_keys_valid']} keys (lines = one key)", fontsize=10); fig.tight_layout(); fig.savefig(F / f"crossed_cells{tag}.pdf", bbox_inches="tight"); plt.close(fig)
    # effects
    fig, ax = plt.subplots(figsize=(5.6, 3.4)); names = ["D0", "R1", "T", "I"]; rng = np.random.default_rng(0)
    for i, q in enumerate(names):
        v = np.array([float(r[q]) for r in eff]) * 1e5; ax.scatter(np.full(len(v), i) + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color="#1f77b4")
        e = iv["descriptive"][q]["equal_task"] if q != "I" else iv["primary_I"]["equal_task"]
        ax.errorbar([i], [e["estimate"] * 1e5], yerr=[[(e["estimate"] - e["ci95"][0]) * 1e5], [(e["ci95"][1] - e["estimate"]) * 1e5]], fmt="_", color="k", capsize=4, ms=14, lw=1.5)
    d = iv["delta_I"] * 1e5; ax.axhspan(-d, d, xmin=.8, xmax=1.0, color="#dddddd", alpha=.6); ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(["$D_0=J_{00}-J_{10}$", "$R_1=J_{10}-J_{11}$", "$T=J_{00}-J_{11}$", "$I=D_1-D_0$"], fontsize=8); ax.set_ylabel("$10^{-5}$ m$^2$ s")
    ax.set_title(f"Decomposition per key; equal-task mean, task-clustered 95 % CI; I decision: {iv['interaction_decision']}", fontsize=8)
    fig.tight_layout(); fig.savefig(F / f"crossed_effects{tag}.pdf", bbox_inches="tight"); plt.close(fig)
    # by task
    fig, ax = plt.subplots(figsize=(5.6, 3.0)); tasks = sorted({int(r["task"]) for r in eff})
    for t in tasks:
        v = [float(r["I"]) * 1e5 for r in eff if int(r["task"]) == t]; ax.scatter([t] * len(v), v, s=16, color="#1f77b4")
    loo = iv["leave_one_task_out_I"]; ax.plot(tasks, [loo[str(t)] * 1e5 for t in tasks], "r_", ms=12, label="leave-one-task-out mean of I")
    ax.axhline(iv["primary_I"]["equal_task"]["estimate"] * 1e5, color="k", lw=.8, label="equal-task mean"); ax.axhline(0, color="k", lw=.4)
    ax.set_xlabel("task"); ax.set_ylabel("$I$ ($10^{-5}$ m$^2$ s)"); ax.legend(fontsize=7); ax.set_title("Interaction by task (two states each)", fontsize=10)
    fig.tight_layout(); fig.savefig(F / f"crossed_by_task{tag}.pdf", bbox_inches="tight"); plt.close(fig); print("figures written to", F)


if __name__ == "__main__":
    main()
