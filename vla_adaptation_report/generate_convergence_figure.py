#!/usr/bin/env python3
"""Reproduce the historical LIBERO/ALOHA/GR1 estimator trajectories.

No simulator is used. The source hashes fix the exact historical cohorts; this
figure neither establishes a convergence rate nor adds new evaluation trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "libero": ("results/suites/libero_spatial_rotonly_paired.json",
               "8198c7a50db48ef36eee3df5af641754dcc6a8a5c23d149cd9e3d5b5f776de1e"),
    "aloha": ("results/aloha/off002_identify1_hold.json",
              "071b7425c91f7f2e268707aac85132f41779774cf37ca32f5000fa476039a1f3"),
    "gr1": ("results/gr1/p2p_right010_hold.json",
            "66a65c0684ed30f721c76c3126075095c6e87ba75a7c79b643c5399308d1245e"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sources():
    data, receipt = {}, {}
    for name, (relative, expected) in SOURCES.items():
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError(f"Historical source changed: {relative}")
        data[name] = json.loads(path.read_text())
        receipt[name] = {"path": relative, "sha256": expected,
                         "bytes": path.stat().st_size}
    return data, receipt


def trajectory(arm, episode, dimension):
    value = np.asarray(arm["traj"][episode], float)
    if value.ndim != 2 or value.shape[1] != dimension or not np.isfinite(value).all():
        raise ValueError("Invalid recorded trajectory")
    return value


def prepare(data):
    arm = data["libero"]["arms"]["adaptive"]
    if arm["n"] != 20 or len(arm["traj"]) != 20:
        raise ValueError("LIBERO panel requires its complete 20-episode cohort")
    traces = [trajectory(arm, i, 6) for i in range(20)]
    if min(map(len, traces)) < 60:
        raise ValueError("Do not pad or impute missing LIBERO samples")
    first60 = np.stack([trace[:60, 3:6] for trace in traces])
    # NumPy's explicit linear quantile convention, independently per step/channel.
    q25, median, q75 = np.quantile(first60, [.25, .5, .75], axis=0, interpolation="linear")
    curves = {"libero": median}
    summaries = {"libero": {
        "episodes": 20, "channels": [3, 4, 5], "sample_indices": [0, 59],
        "episode_lengths": list(map(len, traces)),
        "aggregate": "Median and pointwise interquartile range; linear quantiles; no padding",
        "median_at_logged_index_59": median[-1].tolist(),
        "median": median.tolist(), "q25": q25.tolist(), "q75": q75.tolist(),
        "fault_value": .05,
        "protocol_provenance": "docs/ADAPTIVE_CONTROL_VLA.md sections10/13: uniform +0.05 Cartesian command fault, rotation-only correction. This older JSON lacks embedded arguments/f_true; the dashed target uses the documented protocol, not a fitted estimate.",
        "cohort": arm["per_ep"],
    }}
    for name, dimension, active, fault in [
        ("aloha", 14, list(range(6)), .02),
        ("gr1", 29, list(range(7, 14)), .10),
    ]:
        run = data[name]
        arm = run["arms"]["adaptive"]
        trace = trajectory(arm, 0, dimension)
        truth = np.asarray(arm["f_true"][0], float)
        if truth.shape != trace.shape or not np.all(truth[:, active] == fault):
            raise ValueError(f"Unexpected fault on {name}")
        if run["args"]["identify_episodes"] != 1:
            raise ValueError("Panel must show the identification episode")
        statistic = "last" if name == "aloha" else "mean50"
        if run["args"].get("hold_stat", "last") != statistic:
            raise ValueError("Unexpected hold statistic")
        held = trace[-1, active] if statistic == "last" else trace[-50:, active].mean(0)
        for i in range(1, arm["n"]):
            subsequent = trajectory(arm, i, dimension)[:, active]
            if not np.allclose(subsequent, held, rtol=0, atol=1e-14):
                raise ValueError(f"Subsequent {name} episode does not apply recorded held estimate")
        curves[name] = trace[:, active]
        summaries[name] = {
            "episode_index": 0, "seed": run["args"]["seed"],
            "channels": active, "steps": len(trace), "trace": trace[:, active].tolist(),
            "fault_value": fault, "held_statistic": statistic,
            "held_values_rad": held.tolist(), "held_range_rad": [float(held.min()), float(held.max())],
            "held_percent_of_fault": (100 * held / fault).tolist(),
            "held_in_all_subsequent_recorded_episodes": arm["n"] - 1,
            "episode_outcome": arm["per_ep"][0],
            "run_arguments": run["args"],
        }
        if name == "gr1":
            summaries[name]["at_logged_index_100_rad"] = trace[100, active].tolist()
            summaries[name]["final50_sample_indices"] = [len(trace)-50, len(trace)-1]
            summaries[name]["scope"] = "Historical unpaired-scene study; predates the GR1 environment-generator reset fix. The held average does not establish sustained settling by step100."
    return curves, (q25, q75), summaries


def draw(curves, interval):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.titlesize": 7.7, "axes.labelsize": 7,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
        "legend.fontsize": 6.3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .65,
    })
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.28))
    fig.subplots_adjust(left=.090, right=.985, top=.75, bottom=.31, wspace=.56)
    palettes = {
        "libero": ["#c0392b", "#2874a6", "#1e8449"],
        "aloha": plt.cm.viridis(np.linspace(0, .86, 6)),
        "gr1": plt.cm.plasma(np.linspace(.04, .9, 7)),
    }
    titles = [r"LIBERO · $\pi_{0.5}$"+"\nuniform +0.05", r"ALOHA · $\pi_0$"+"\nleft arm, +0.02 rad",
              "GR1 · GR00T N1.5\nright arm, +0.10 rad"]
    for index, (name, ax, title) in enumerate(zip(curves, axes, titles)):
        a = curves[name]
        x = np.arange(len(a))
        labels = ([r"$r_x$", r"$r_y$", r"$r_z$"] if name == "libero"
                  else [f"j{j}" for j in (range(6) if name == "aloha" else range(7, 14))])
        for j, (color, label) in enumerate(zip(palettes[name], labels)):
            if name == "libero":
                ax.fill_between(x, interval[0][:, j], interval[1][:, j], color=color,
                                alpha=.14, linewidth=0)
            ax.plot(x, a[:, j], color=color, lw=1.05, label=label)
        fault = [.05, .02, .10][index]
        ax.axhline(fault, color="#263746", ls=(0, (3, 2)), lw=.8)
        ax.set_title(title, pad=7, linespacing=1.4)
        ax.set_xlabel("Control step", labelpad=3)
        ax.set_ylabel("Estimate (action units)" if name == "libero" else "Estimate (rad)", labelpad=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#81909a")
        ax.tick_params(length=2.5, color="#81909a", pad=2)
        ax.grid(axis="y", color="#e7ebef", lw=.5)
        ax.set_axisbelow(True)
        ax.set_xlim(0, [60, 300, 164][index])
        ax.set_ylim([(-.008, .07), (-.004, .034), (-.018, .155)][index])
        ax.set_xticks([[0, 20, 40, 60], [0, 100, 200, 300], [0, 50, 100, 150]][index])
        ax.set_yticks([[0, .02, .04, .06], [0, .01, .02, .03], [0, .05, .10, .15]][index])
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.30), ncol=3 if name != "gr1" else 4,
                  frameon=False, handlelength=1.1, handletextpad=.3,
                  columnspacing=.65, borderaxespad=0, labelspacing=.4)
    fig.text(.5, .019, "Dashed: injected offset   ·   LIBERO: median + IQR   ·   ALOHA/GR1: identification episode",
             ha="center", color="#536475", fontsize=6.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    failures = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < bounds.x0-1 or box.x1 > bounds.x1+1
                                        or box.y0 < bounds.y0-1 or box.y1 > bounds.y1+1):
            failures.append(artist.get_text())
    if failures:
        raise ValueError(f"Figure text leaves canvas: {failures}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"paper/fig_convergence.pdf")
    parser.add_argument("--receipt", type=Path, default=ROOT/"paper/convergence_figure_receipt.json")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    data, sources = read_sources()
    curves, interval, summaries = prepare(data)
    fig = draw(curves, interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Title": "Historical estimator trajectories on three robots",
        "Subject": "Reproduced from pinned stored trajectories; no convergence guarantee",
        "Creator": "paper/generate_convergence_figure.py", "CreationDate": None, "ModDate": None})
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=220)
    plt.close(fig)
    receipt = {"schema_version": 1, "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": sha(__file__), "sources": sources, "panels": summaries,
        "step_convention": "Zero-based logged action/update index; each sample is the estimate AFTER that step. No initial zeros are prepended and no endpoint values are extrapolated.",
        "scope": "Historical illustrative estimator traces, not matched three-robot efficacy evidence or a convergence-time guarantee. Each panel has its own units/protocol.",
        "layout": {"width_inches": 5.5, "height_inches": 2.28, "all_text_inside_canvas": True},
        "output": {"path": str(args.output.resolve()), "sha256": sha(args.output)},
        "libraries": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "reproduce": "python paper/generate_convergence_figure.py"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "receipt": str(args.receipt),
                      "sources_verified": len(sources), "gr1_steps": summaries["gr1"]["steps"],
                      "gr1_final50_range": summaries["gr1"]["held_range_rad"]}))


if __name__ == "__main__":
    main()
