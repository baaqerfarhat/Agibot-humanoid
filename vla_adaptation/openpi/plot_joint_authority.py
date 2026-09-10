"""Export paired local physical-error diagnostics; these are not task outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    record = json.loads(raw)
    if record.get("status") != "complete" or record.get("n_probes") != 252:
        raise ValueError("Expected the complete 252-probe physical grid")
    rows = [row for row in record["per_probe"] if row["metadata"]["checkpoint"] > 0]
    if len(rows) != 168:
        raise ValueError("Expected 168 probes after initialization")
    all_ratios = [row["nonlinear_replays"][name+"_"+support]["error_ratio"]
                  for row in rows for name in ("legacy", "weighted")
                  for support in ("translation", "full")]
    extent = [min(all_ratios)/1.4, max(all_ratios)*1.4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    for ax, support in zip(axes, ("translation", "full")):
        ratios = np.array([[row["nonlinear_replays"][name+"_"+support]["error_ratio"]
                            for name in ("legacy", "weighted")] for row in rows])
        if not np.isfinite(ratios).all() or (ratios <= 0).any():
            raise ValueError("Log-scale error ratios must be finite and positive")
        for joint in range(7):
            selected = [i for i, row in enumerate(rows) if row["metadata"]["joint"] == joint]
            ax.scatter(ratios[selected, 0], ratios[selected, 1], s=26, color=colors(joint),
                       alpha=.72, edgecolors="none", label=f"Joint {joint}")
        ax.plot(extent, extent, color=".5", lw=1, ls="--", zorder=0)
        ax.axhline(1, color="#a13b3b", lw=1)
        ax.axvline(1, color="#a13b3b", lw=1)
        ax.set(xscale="log", yscale="log", xlim=extent, ylim=extent,
               xlabel="Legacy physical error / correction off",
               title="Translation inputs" if support == "translation" else "All six Cartesian inputs")
        worse = int((ratios[:, 1] > 1+1e-9).sum())
        better = int((ratios[:, 1] < ratios[:, 0]-1e-9).sum())
        ax.text(.04, .96, f"Weighted worsens {worse}/168 vs off\n"
                f"Weighted lowers error in {better}/168 vs legacy",
                va="top", transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        ax.grid(alpha=.12, which="both")
    axes[0].set_ylabel("Weighted physical error / correction off")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=7,
               bbox_to_anchor=(.5, .06), frameon=False)
    fig.suptitle("Better than legacy can still be worse than correction off", fontsize=13)
    fig.text(.5, .035, "Panda, three LIBERO suites; translation / 0.05 m and rotation / 0.5 rad.",
             ha="center", fontsize=9)
    fig.text(.5, .005, "All points shown; large ratios can reflect tiny off-error. These are not independent task episodes.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .95))
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = args.out_prefix.with_suffix("."+extension)
        if path.exists():
            raise ValueError(f"Preserve the earlier figure: {path}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    args.out_prefix.with_suffix(".json").write_text(json.dumps(dict(
        source=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest(),
        plotted_probes=168, inclusion="checkpoint > 0", interpretation="local physical error, not task success"), indent=2)+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
