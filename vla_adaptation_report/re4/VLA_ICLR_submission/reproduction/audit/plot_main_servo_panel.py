"""Plot a compact main-text view of the existing verified servo trajectories.

No simulation is run, and no source data or historical VLA outcome is changed.
Run from any directory: python audit/plot_main_servo_panel.py
"""
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audit" / "contraction_tracking_trajectories.npz"
EXPECTED_SHA256 = "07ce5f5ea0b49716a1b9c29db54c61a013a782a5352b9670b5dd92b02c8be6ed"
DT = 0.02
WINDOW = (0.8, 4.5)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert before == EXPECTED_SHA256, "The verified source trajectories changed."
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = {"frozen": "#62666c", "attenuated": "#bf6433", "innovation": "#2465a4"}
    fig, ax = plt.subplots(figsize=(6.8, 1.65))
    fig.subplots_adjust(left=.087, right=.755, bottom=.29, top=.975)
    plotted_differences = []
    with np.load(SOURCE) as data:
        t = DT * np.arange(len(data["frozen__metric_error"]))
        for family in ("frozen", "attenuated", "innovation"):
            quantities = ("metric_error",) if family == "frozen" else ("metric_error", "envelope")
            for quantity in quantities:
                y = data[f"{family}__{quantity}"] * 1000.0
                line, = ax.plot(t, y, color=colors[family],
                                ls="--" if quantity == "envelope" else "-",
                                lw=1.15 if quantity == "envelope" else 1.5)
                plotted_differences.append(float(np.max(np.abs(line.get_ydata() - y))))
                assert np.array_equal(line.get_xdata(), t)
        ax.axvspan(1.3, 2.0, color="#888888", alpha=.13, lw=0, zorder=0)
        ax.axvline(1.0, color="#999999", ls=":", lw=.8)
        ax.text(1.65, 104.0, "Update hold", ha="center", va="center", fontsize=7.5, color="#555555")
        ax.set(xlim=WINDOW, ylim=(0, 113), xlabel="Time (s)", ylabel="Metric error (mm)",
               yticks=[0, 25, 50, 75, 100], xticks=[1, 2, 3, 4])
        ax.grid(alpha=.15, linewidth=.6)
        handles = [Line2D([0], [0], color=colors[k], lw=1.5)
                   for k in ("frozen", "attenuated", "innovation")]
        fig.legend(handles, ["Frozen", "Observation attenuation", "Innovation normalization"],
                   loc="upper left", bbox_to_anchor=(.765, .975), frameon=False,
                   borderaxespad=0, handlelength=1.5, labelspacing=.65, handletextpad=.6,
                   fontsize=7.25)
        fig.text(.774, .33, "Solid: error $X_k$\nDashed: bound $R_k$",
                 ha="left", va="bottom", fontsize=7.8, linespacing=1.5)
        fig.savefig(ROOT / "paper" / "fig_servo_main.pdf")
        fig.savefig(ROOT / "audit" / "fig_servo_main_preview.png", dpi=240)
    plt.close(fig)
    after = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert before == after
    assert max(plotted_differences) == 0.0
    print(f"Source SHA256: {after}")
    print(f"Maximum plotted-curve difference from source, after m-to-mm conversion: {max(plotted_differences)}")
    print("State and bound timestamps: t_k = 0.02 k s; window [0.8, 4.5] s.")


if __name__ == "__main__":
    main()
