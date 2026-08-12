#!/usr/bin/env python3
"""Data charts and equation images for the SURF final presentation.

Block diagrams are built as native PowerPoint shapes in build_surf_slides.py;
this file produces only things matplotlib is actually better at.

Every number comes from the repository:
  adaptation/isaac_runs/fault_knee03/adapt_experiments_summary.json
  adaptation/isaac_runs/adapt_experiments_summary.json
  presentation/Humanoid Weekly Updates (5).pdf   (VLM latency table)

Output: presentation/surf_assets/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "surf_assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#152240"
ACCENT = "#2E86C1"
RED = "#C0392B"
GREEN = "#1E8449"
GRAY = "#5C6670"

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#C5CDD6",
        "axes.labelcolor": GRAY,
        "text.color": NAVY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

DPI = 240


def save(fig, name, transparent=False):
    p = OUT / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.05, transparent=transparent)
    plt.close(fig)
    print("wrote", p.relative_to(ROOT))


def _fault_json():
    return json.load(
        open(ROOT / "adaptation/isaac_runs/fault_knee03/adapt_experiments_summary.json")
    )


# --------------------------------------------------------------------------
# Headline chart: survival under the right-knee fault
# --------------------------------------------------------------------------
def fig_fault_results():
    d = _fault_json()
    dt = d["ctrl_dt"]
    frozen = np.sort([r["survival"] for r in d["results"]["frozen_npz"]]) * dt
    waist = np.sort([r["survival"] for r in d["results"]["w0_g3e-4_waistonly"]]) * dt
    legs = np.sort([r["survival"] for r in d["results"]["w0_g3e-4_gx1"]]) * dt
    NOMINAL = 14.68

    fig, ax = plt.subplots(figsize=(7.1, 4.35))
    series = [
        ("Frozen\nbaseline", frozen, RED),
        ("Waist-only\nadaptation", waist, GREEN),
        ("Paper mask\nlegs + waist", legs, GRAY),
    ]

    ax.axhline(NOMINAL, color=ACCENT, lw=1.7, ls=(0, (5, 3)), zorder=1)
    ax.text(
        -0.52, NOMINAL + 0.30, f"no fault:  {NOMINAL:.2f} s",
        color=ACCENT, fontsize=11, fontweight="bold", ha="left", va="bottom",
    )

    rng = np.random.default_rng(7)
    for i, (label, vals, c) in enumerate(series):
        m, sd = vals.mean(), vals.std(ddof=0)
        ax.bar(i, m, width=0.52, color=c, alpha=0.15, edgecolor=c, linewidth=1.7, zorder=2)
        ax.errorbar(i, m, yerr=sd, fmt="none", ecolor=c, elinewidth=1.9,
                    capsize=8, capthick=1.9, zorder=4)
        ax.scatter(i + rng.uniform(-0.115, 0.115, len(vals)), vals, s=44,
                   color="white", edgecolor=c, linewidth=1.8, zorder=5, clip_on=False)
        ax.text(i, max(m + sd, vals.max()) + 0.55, f"{m:.2f} s", ha="center", va="bottom",
                fontsize=14.5, fontweight="bold", color=c, zorder=6)

    # complete separation between the two arms that matter
    lo, hi = frozen.max(), waist.min()
    ax.annotate("", xy=(0.30, lo), xytext=(0.30, hi),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.4))
    ax.text(0.355, (lo + hi) / 2, "every adapted seed\nbeats every frozen seed",
            fontsize=9.4, color=NAVY, va="center", ha="left", linespacing=1.45)

    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([s[0] for s in series], fontsize=11.5, color=NAVY)
    ax.set_ylabel("Survival time (s)", fontsize=11.5)
    ax.set_ylim(0, 16.6)
    ax.set_xlim(-0.58, 2.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E6EBF0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Right knee at 30% nominal PD stiffness  ·  6 seeds  ·  circles are seeds",
                 fontsize=10.4, color=GRAY, pad=9, loc="left")
    save(fig, "fig_fault_results.png")


# --------------------------------------------------------------------------
# Backup chart: the mask decides, on a healthy robot and under the fault
# --------------------------------------------------------------------------
def fig_healthy_vs_fault():
    d_h = json.load(open(ROOT / "adaptation/isaac_runs/adapt_experiments_summary.json"))
    d_f = _fault_json()

    def mean(d, k):
        return float(np.mean([r["survival"] for r in d["results"][k]]) * d["ctrl_dt"])

    keys = [
        ("frozen_npz", "Frozen"),
        ("w0_g3e-4_waistonly", "Waist only\ngain 3e-4"),
        ("w0_g3e-4_gx1", "Legs + waist\ngain 3e-4"),
        ("w0_g1e-5_gx1", "Legs + waist\ngain 1e-5"),
    ]
    healthy = [mean(d_h, k) for k, _ in keys]
    fault = [mean(d_f, k) for k, _ in keys]

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    x = np.arange(len(keys))
    ax.bar(x - 0.19, healthy, 0.36, label="Healthy robot", color=ACCENT, alpha=0.9)
    ax.bar(x + 0.19, fault, 0.36, label="Right knee at 30%", color=RED, alpha=0.9)
    for xi, (a, b) in enumerate(zip(healthy, fault)):
        ax.text(xi - 0.19, a + 0.28, f"{a:.1f}", ha="center", fontsize=10,
                color=ACCENT, fontweight="bold")
        ax.text(xi + 0.19, b + 0.28, f"{b:.1f}", ha="center", fontsize=10,
                color=RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in keys], fontsize=10.2, color=NAVY)
    ax.set_ylabel("Mean survival (s)", fontsize=10.5)
    ax.set_ylim(0, 17.6)
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#E6EBF0", lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "fig_healthy_vs_fault.png")


# --------------------------------------------------------------------------
# VLM measured latency (RTX 5070 8 GB, 12 frames from one 15 s clip)
# --------------------------------------------------------------------------
def fig_vlm_latency():
    fig, ax = plt.subplots(figsize=(5.5, 2.75))
    models = ["Moondream\n~1.8 B", "Qwen2.5-VL\n3 B", "Qwen2.5-VL\n7 B"]
    lat = [0.61, 2.45, 2.87]
    rate = [1.65, 0.41, 0.35]
    cols = [GREEN, ACCENT, NAVY]
    ax.barh(range(3), lat, color=cols, alpha=0.88, height=0.56)
    for i, (l, r) in enumerate(zip(lat, rate)):
        ax.text(l + 0.10, i, f"{l:.2f} s  ·  {r:.2f} Hz", va="center", ha="left",
                fontsize=10, color=cols[i], fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(models, fontsize=10, color=NAVY)
    ax.set_xlim(0, 4.6)
    ax.set_xlabel("Mean latency per frame (s)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#E6EBF0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5)
    save(fig, "fig_vlm_latency.png")


# --------------------------------------------------------------------------
# Equation images (transparent, dropped onto slides)
# --------------------------------------------------------------------------
def equations():
    eqs = {
        "eq_update.png": r"$\dot{W} \;=\; \Gamma\,\delta\, z^{\top} \;-\; \gamma\,(W - W_0)$",
        "eq_delta.png": r"$\delta_L \;=\; g(x)^{\top} P\, e ,\qquad "
                        r"\delta_l \;=\; \Psi_l(a_l)\, W_{l+1}^{\top}\, \delta_{l+1}$",
        "eq_error.png": r"$e \;=\; q - q_{\mathrm{ref}}$",
    }
    for name, tex in eqs.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=22, color=NAVY)
        save(fig, name, transparent=True)


if __name__ == "__main__":
    fig_fault_results()
    fig_healthy_vs_fault()
    fig_vlm_latency()
    equations()
    print("done")
