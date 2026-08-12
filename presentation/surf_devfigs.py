#!/usr/bin/env python3
"""Development figures for the SURF deck, built from the hardware run logs.

Two figures, both entirely from files in this repo. Nothing here is synthetic.

  fig_joint_error.png   where the v31 hardware error actually lived, and what
                        v33 did about it. Sources: run_logs/_irl_202500_analysis.json
                        (per-joint RMSE over the 15 Aug 5 runs) and the raw
                        Aug 5 / Aug 11 CSVs for the waist traces.

  fig_foot.png          the foot contact story. Sources: run_logs/_run_analysis.json
                        (sim foot tilt per v30 checkpoint) and the reward term
                        weights recorded in the holosoma overlay config.

Run:  python3 presentation/surf_devfigs.py
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "run_logs"
POL = ROOT / "box_pickup" / "policy"
OUT = HERE / "surf_assets"

NAVY = "#12263f"
ACCENT = "#0b6bcb"
RED = "#c0392b"
GREEN = "#1e8449"
AMBER = "#b7791f"
GRAY = "#6b7a8f"
RULE = "#d5dce5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": RULE,
    "axes.labelcolor": NAVY,
    "axes.titlecolor": NAVY,
    "text.color": NAVY,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

PRETTY = {
    "waist_pitch": "waist pitch", "waist_roll": "waist roll", "waist_yaw": "waist yaw",
    "left_shoulder_roll": "L shoulder roll", "right_shoulder_roll": "R shoulder roll",
    "left_shoulder_pitch": "L shoulder pitch", "right_shoulder_pitch": "R shoulder pitch",
    "left_elbow": "L elbow", "right_elbow": "R elbow",
    "left_hip_pitch": "L hip pitch", "right_hip_pitch": "R hip pitch",
    "left_hip_roll": "L hip roll", "right_hip_roll": "R hip roll",
    "left_hip_yaw": "L hip yaw", "right_hip_yaw": "R hip yaw",
    "left_knee": "L knee", "right_knee": "R knee",
    "left_ankle_pitch": "L ankle pitch", "right_ankle_pitch": "R ankle pitch",
    "left_ankle_roll": "L ankle roll", "right_ankle_roll": "R ankle roll",
}


def _policy_phase(path, cols):
    with open(path) as fh:
        r = csv.reader(fh)
        head = next(r)
        idx = {c: i for i, c in enumerate(head)}
        rows = [x for x in r if x[idx["phase"]] == "policy"]
    if len(rows) < 40:
        return None
    out = {"frame": np.array([int(x[idx["frame"]]) for x in rows])}
    for c in cols:
        out[c] = np.degrees(np.array([float(x[idx[c]]) for x in rows]))
    return out


def _waist_traces(pattern, policy_npz):
    """Mean measured and commanded waist pitch against the reference clip."""
    z = np.load(POL / policy_npz, allow_pickle=True)
    jn = json.loads(str(z["meta_json"]))["joint_names"]
    ref = np.degrees(z["ref_joint_pos"][:, jn.index("waist_pitch_joint")])

    cols = ["waist_pitch_joint__pos_meas", "waist_pitch_joint__tgt"]
    grid = np.arange(len(ref))
    meas, cmd = [], []
    for f in sorted(glob.glob(str(LOGS / pattern))):
        d = _policy_phase(f, cols)
        if d is None:
            continue
        fr = d["frame"]
        keep = fr < len(ref)
        if keep.sum() < 40:
            continue
        meas.append(np.interp(grid, fr[keep], d[cols[0]][keep],
                              left=np.nan, right=np.nan))
        cmd.append(np.interp(grid, fr[keep], d[cols[1]][keep],
                             left=np.nan, right=np.nan))
    return grid / 50.0, ref, np.nanmean(meas, axis=0), np.nanmean(cmd, axis=0), len(meas)


def fig_joint_error():
    an = json.loads((LOGS / "_irl_202500_analysis.json").read_text())
    runs = an["runs"]

    names = list(runs[0]["per_joint_rmse"])
    stats = []
    for k in names:
        v = [r["per_joint_rmse"][k] for r in runs if k in r["per_joint_rmse"]]
        stats.append((float(np.mean(v)), float(np.min(v)), float(np.max(v)), k))
    stats.sort()
    stats = stats[-12:]

    fig = plt.figure(figsize=(13.2, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.20, 1.0, 1.0], wspace=0.30,
                          left=0.115, right=0.985, top=0.845, bottom=0.155)

    # ---- panel 1: which joint carried the error
    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(stats))
    vals = [s[0] for s in stats]
    cols = [RED if "waist_pitch" in s[3] else
            (ACCENT if s[3].startswith("waist") else "#b8c4d2") for s in stats]
    ax.barh(ys, vals, color=cols, height=0.72, zorder=3)
    for y, s in zip(ys, stats):
        ax.plot([s[1], s[2]], [y, y], color=NAVY, lw=0.9, alpha=0.45, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(s[3], s[3]) for s in stats], fontsize=8.2)
    ax.set_xlabel("tracking error vs commanded target, RMSE (deg)", fontsize=8.4)
    ax.set_title("v31 on hardware: one joint holds the error",
                 fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(stats[-1][0] - 3, ys[-1], f"{stats[-1][0]:.0f}\u00b0", va="center",
            ha="right", fontsize=9.5, fontweight="bold", color="white", zorder=5)
    ax.text(0.985, 0.06, "15 runs  ·  bars mean, line min to max",
            transform=ax.transAxes, ha="right", fontsize=7.4, color=GRAY)

    # ---- panels 2 and 3: the waist command before and after
    panels = [
        (gs[0, 1], "20260805_*box_pickup_x2_box_policy.csv", "x2_box_policy_v31.npz",
         "v31", "commanded far outside the clip", RED),
        (gs[0, 2], "20260811_*box_pickup_*v33_iter253000.csv",
         "x2_box_policy_v33_iter253000.npz", "v33",
         "waist tracking terms added", GREEN),
    ]
    for cell, pat, npz, tag, sub, col in panels:
        t, ref, meas, cmd, n = _waist_traces(pat, npz)
        ax = fig.add_subplot(cell)
        ax.axhline(0, color=RULE, lw=0.8)
        ax.plot(t, ref, color=NAVY, lw=1.9, ls=(0, (4, 2)), label="reference clip",
                zorder=4)
        ax.plot(t, cmd, color=col, lw=1.5, label="policy command", zorder=3)
        ax.plot(t, meas, color=ACCENT, lw=1.7, label="measured", zorder=5)
        ax.set_ylim(-195, 45)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time in motion (s)", fontsize=8.4)
        if tag == "v31":
            ax.set_ylabel("waist pitch (deg)", fontsize=8.4)
        ax.set_title(f"{tag}: {sub}", fontsize=10, fontweight="bold",
                     loc="left", pad=8, color=col)
        ax.grid(color=RULE, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo = np.nanmin(cmd)
        ax.annotate(f"min command {lo:.0f}\u00b0",
                    xy=(t[int(np.nanargmin(cmd))], lo), xytext=(0.34, 0.10),
                    textcoords="axes fraction", fontsize=8, color=col,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.text(0.985, 0.955, f"{n} hardware runs", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.4, color=GRAY)
        ax.text(0.015, 0.045, "clip peaks at +18\u00b0", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.4, color=GRAY)
        if tag == "v33":
            ax.legend(loc="center right", fontsize=7.6, frameon=True,
                      facecolor="white", edgecolor=RULE, borderpad=0.5,
                      bbox_to_anchor=(1.0, 0.42))

    p = OUT / "fig_joint_error.png"
    fig.savefig(p, dpi=210)
    plt.close(fig)
    print("wrote", p.name)


def fig_foot():
    an = json.loads((LOGS / "_run_analysis.json").read_text())
    ck = an["simCheckpoints"]

    fig = plt.figure(figsize=(13.2, 2.70))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.22,
                          left=0.062, right=0.985, top=0.800, bottom=0.190)

    # ---- foot tilt per checkpoint: the right foot rides its edge
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(ck))
    ax.bar(xs - 0.19, [c["footTiltL"] for c in ck], 0.36, color="#b8c4d2",
           label="left foot", zorder=3)
    ax.bar(xs + 0.19, [c["footTiltR"] for c in ck], 0.36, color=RED,
           label="right foot", zorder=3)
    ax.axhline(10, color=NAVY, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.text(len(ck) - 0.42, 11.2, "flat contact limit  10\u00b0", fontsize=7.6,
            color=NAVY, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c['iter'] // 1000}k" for c in ck], fontsize=8.2)
    ax.set_xlabel("v30 training checkpoint", fontsize=8.4)
    ax.set_ylabel("foot tilt in sim (deg)", fontsize=8.4)
    ax.set_title("The reward passed a foot standing on its edge",
                 fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylim(0, 34)
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left", ncol=2)

    # ---- how the contact penalties escalated
    ax = fig.add_subplot(gs[0, 1])
    vers = ["v25", "v27", "v28", "v30", "v31", "v33"]
    xs = np.arange(len(vers))
    series = [
        ("foot slip", [0, 1.0, 2.0, 3.0, 3.0, 6.0], RED, "o"),
        ("contact loss", [0, 0, 0, 2.0, 2.0, 4.0], ACCENT, "s"),
        ("not flat", [0, 0, 0, 0, 3.0, 3.0], GREEN, "^"),
    ]
    for lbl, v, c, m in series:
        ax.step(xs, v, where="post", color=c, lw=1.8, zorder=3)
        ax.plot(xs, v, m, color=c, ms=4.6, zorder=4, label=lbl)
    ax.set_xticks(xs)
    ax.set_xticklabels(vers, fontsize=8.4)
    ax.set_xlabel("policy version", fontsize=8.4)
    ax.set_ylabel("penalty weight", fontsize=8.4)
    ax.set_title("Each hardware trial added or raised a contact cost",
                 fontsize=10, fontweight="bold", loc="left", pad=8)
    ax.set_ylim(-0.35, 7.9)
    ax.set_xlim(-0.25, len(vers) - 0.30)
    ax.grid(axis="y", color=RULE, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left", ncol=3)
    for x, ha, lab in [(1, "left", "feet skated\n28 cm \u2192 0.6 cm"),
                       (3, "left", "feet stepped\ninstead of slid"),
                       (5, "right", "waist terms\nout-competed slip")]:
        ax.annotate(lab, xy=(x, 0.35), xytext=(x + (0.06 if ha == "left" else -0.12), 5.4),
                    fontsize=7.2, color=GRAY, ha=ha, va="top",
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.9))

    p = OUT / "fig_foot.png"
    fig.savefig(p, dpi=210)
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_joint_error()
    fig_foot()
