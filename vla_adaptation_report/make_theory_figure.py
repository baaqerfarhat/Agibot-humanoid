"""Visualize two constructed examples from the paper; no robot measurements.

The numerical values are checked against the existing theory verification receipt.
Axes in the authority panel use equal physical scales. The stability curve is the
spectral radius of the constant, unclipped scalar example, not a robot gain bound.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

PAPER = Path(__file__).resolve().parent
INK, TEAL, PURPLE, RED = "#24384B", "#247B78", "#745689", "#AA4D3B"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checked = json.loads((PAPER / "interface_theory_receipt.json").read_text())
    local_sources = {p.name: p for p in [PAPER / "verify_interface_theory.py",
                     PAPER.parent / "openpi/composite_observer.py",
                     PAPER / "authority_appendix.tex", PAPER / "theory_appendix.tex"]}
    for recorded_path, expected in checked["sources_sha256"].items():
        path = local_sources[Path(recorded_path).name]
        if sha(path) != expected:
            raise ValueError(f"Theory source changed: {path}")
    m = np.array([[1., 1.], [0., .1]])
    d = np.array([0., .02])
    c = np.linalg.solve(m, d)[0]
    response = m[:, 0] * c
    ratio = np.linalg.norm(d-response) / np.linalg.norm(d)
    assert np.isclose(ratio, checked["checks"]["authority_counterexamples"]
                      ["inverse_then_mask"]["physical_error_ratio"])
    rates = np.linspace(0., 22., 1101)

    def radius(eta):
        matrix = np.array([[.8, -1.], [.16*eta, .8-.2*eta]])
        return float(np.max(np.abs(np.linalg.eigvals(matrix))))

    radii = np.array([radius(x) for x in rates])
    for case in checked["checks"]["scalar_tracking"]["cases"]:
        assert np.isclose(radius(case["rate"]), max(abs(x) for x in case["eigenvalues"]))
    critical = 81/5
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3,
                         "axes.labelsize": 7.3, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 6.8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(5.5, 2.15), facecolor="white")
    left = fig.add_axes([.075, .23, .395, .58])
    right = fig.add_axes([.615, .23, .36, .58])
    fig.text(.055, .955, "A  Correction authority", color=INK, weight="bold", va="top", fontsize=8)
    fig.text(.56, .955, "B  Estimator–controller interaction", color=INK,
             weight="bold", va="top", fontsize=8)
    left.set(xlim=(-.235, .045), ylim=(-.025, .13), xlabel=r"motion coordinate $d_1$",
             ylabel=r"$d_2$")
    left.set_aspect("equal", adjustable="box")
    left.axhline(0, color="#BFCACD", lw=3, zorder=0)
    left.axvline(0, color="#DEE4E6", lw=.7, zorder=0)
    left.set_xticks([-.2, -.1, 0])
    left.set_yticks([0, .1])

    def arrow(ax, start, end, color, lw=1.5):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                    linewidth=lw, color=color, shrinkA=0, shrinkB=0))

    arrow(left, (0, 0), d, TEAL)
    arrow(left, (0, 0), response, PURPLE)
    arrow(left, response, d, RED)
    left.scatter([0], [0], s=13, color=INK, zorder=5)
    left.text(.008, .027, r"$d$", color=TEAL, fontsize=9)
    left.text(-.205, -.016, r"$Dc$", color=PURPLE, fontsize=8, va="top")
    left.text(-.117, .021, r"$d-Dc$", color=RED, fontsize=8, ha="center")
    left.text(-.229, .117, r"$d=(0,.02),\quad c=-.2$", fontsize=7.3, va="top", color=INK)
    left.text(-.229, .084, r"$\|d-Dc\|/\|d\|=\sqrt{101}$", color=RED, fontsize=8, va="top")
    left.text(-.229, .049, r"$\mathrm{range}(D)$", color="#62747D", fontsize=7.3)
    left.annotate("", xy=(-.218, .002), xytext=(-.218, .045),
                  arrowprops={"arrowstyle": "-", "color": "#89989E", "lw": .7})

    right.axvspan(0, critical, color="#EFF7F5", zorder=0)
    right.axvspan(critical, 22, color="#F9EEE9", zorder=0)
    right.axhline(1, color="#536571", lw=.85, ls="--")
    right.axvline(critical, color=RED, lw=.8, ls=":")
    right.plot(rates, radii, color=PURPLE, lw=1.8)
    right.scatter([0], [radius(0)], color=INK, s=15, zorder=4, clip_on=False)
    right.scatter([20], [radius(20)], color=RED, s=17, zorder=4)
    right.set(xlim=(0, 22), ylim=(0, 2.85), xlabel=r"tracking gain $\eta$",
              ylabel=r"spectral radius $\rho(\mathcal{A}(\eta))$")
    right.set_xticks([0, 10, critical, 22], ["0", "10", "16.2", "22"])
    right.set_yticks([0, 1, 2])
    right.text(4.2, 1.36, r"$\rho<1$", color=TEAL, fontsize=9)
    right.text(18.4, 2.51, r"$\rho>1$", color=RED, fontsize=9, ha="center")
    right.text(1.1, .48, r"$\eta=0$: Kalman", color=INK, fontsize=7)
    right.annotate(r"$\eta=20$", xy=(20, radius(20)), xytext=(12.2, 2.1),
                   fontsize=7, color=RED, arrowprops={"arrowstyle": "-", "lw": .7, "color": RED})
    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#89989E")
            ax.spines[side].set_linewidth(.6)
        ax.tick_params(length=2.5, width=.6, colors="#536571")
    fig.text(.5, .025, "Constructed examples  •  exact local maps  •  not robot measurements",
             ha="center", fontsize=7, color="#5D6B77")
    outputs = [PAPER / "fig_theory_analysis.pdf", PAPER / "fig_theory_analysis.png"]
    fig.savefig(outputs[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], dpi=240)
    plt.close(fig)
    receipt = {
        "status": "passed", "kind": "constructed_theoretical_examples_not_robot_results",
        "source_hashes": {p.name: sha(p) for p in [Path(__file__), PAPER / "interface_theory_receipt.json",
                                                   PAPER / "authority_appendix.tex", PAPER / "theory_appendix.tex"]},
        "output_hashes": {p.name: sha(p) for p in outputs},
        "authority": {"M": m.tolist(), "d": d.tolist(), "permitted_columns": [0],
                      "inverse_then_mask_c": float(c), "physical_response": response.tolist(),
                      "physical_error_ratio": ratio, "equal_coordinate_scales": True},
        "stability": {"A": .8, "D": 1, "M": 1, "L": 1, "K": .2,
                      "P_posterior": .2, "dt": 1, "damping": 0,
                      "constant_gain_clipping_inactive": True,
                      "nonnegative_gain_stable_iff_eta_less_than": critical,
                      "eta": rates.tolist(), "spectral_radius": radii.tolist(),
                      "scope": "Threshold applies only to the stated scalar example, not ALOHA tuning."}}
    (PAPER / "theory_figure_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"status": "passed", "ratio": ratio, "critical_eta": critical}))


if __name__ == "__main__":
    main()
