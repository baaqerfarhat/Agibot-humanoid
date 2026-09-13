"""Plot complete confirmation outcomes; never select cells or tune configurations.

Run from any directory with Python, NumPy and Matplotlib. The receipt preserves
all plotted numbers and input/output hashes. No simulation or result files change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
SOURCES: dict[str, str] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative: str, expected: str | None = None) -> dict:
    path = ROOT / relative
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Source hash mismatch: {relative}")
    SOURCES[relative] = digest
    return json.loads(path.read_text())


def episode_map(arm: dict) -> dict[tuple, bool]:
    episodes = arm["per_ep"]
    result = {}
    for ep in episodes:
        key = (ep["task"], ep["init"], ep["episode"])
        if key in result or type(ep["ok"]) is not bool:
            raise ValueError("Duplicate scenario or nonbinary outcome")
        result[key] = ep["ok"]
    assert len(result) == arm["n"] == 20
    assert sum(result.values()) == arm["successes"]
    return result


def panda_data() -> dict:
    base = "results/joint_followup/confirmation_plan"
    cross = read("results/joint_followup/confirmation_statistics_crosscheck.json")
    assert cross["status"] == "passed"
    check = next(x for x in cross["studies"] if x["plan"] == base)
    scores = read(f"{base}/scores.json", check["scores_sha256"])
    assert scores["stage"] == "confirmation" and scores["n_scenarios"] == 20
    suites = ["libero_spatial", "libero_object", "libero_goal"]
    conditions = ["healthy"] + [f"joint{i}" for i in range(7)]
    methods = ["legacy_translation", "weighted_full"]
    artifacts = {a["id"]: a for a in scores["artifacts"]}
    assert len(artifacts) == len(scores["artifacts"]) == 48
    comparisons = {c["id"]: c for c in scores["comparisons"]}
    paired = {}
    values = {}
    for suite in suites:
        for condition in conditions:
            cell = f"{suite}_{condition}"
            values[cell] = {}
            controls = []
            for method in methods:
                record = artifacts[f"{cell}_{method}"]
                relative = "results/" + record["path"].split("/results/", 1)[1]
                data = read(relative, record["sha256"])
                assert data["status"] == "complete" and data["scoreable"]
                assert data["pairing"]["valid_so_far"]
                assert data["args"]["suite"] == suite
                assert data["args"]["torque"] == (0.0 if condition == "healthy" else 5.0)
                if condition != "healthy":
                    assert data["args"]["joint"] == int(condition.removeprefix("joint"))
                off = episode_map(data["arms"]["frozen_faulted"])
                on = episode_map(data["arms"]["adaptive"])
                assert off.keys() == on.keys()
                controls.append((data["shared_control_id"], off))
                fixed = sum(on[k] and not off[k] for k in off)
                broken = sum(off[k] and not on[k] for k in off)
                delta = 100 * (fixed - broken) / len(off)
                comp_suffix = "legacy_vs_off" if method == methods[0] else "candidate_vs_off"
                comparison = comparisons[f"{cell}_{comp_suffix}"]
                assert (sum(off.values()), sum(on.values()), fixed, broken) == (
                    comparison["left_successes"], comparison["right_successes"],
                    comparison["right_only"], comparison["left_only"],
                )
                assert delta == comparison["gain_points"]
                values[cell][method] = {
                    "n": len(off), "off_successes": sum(off.values()),
                    "successes": sum(on.values()), "observed_fixed": fixed,
                    "observed_broken": broken, "change_vs_off_pp": delta,
                }
                paired[(suite, condition, method)] = on
            assert controls[0] == controls[1], "Methods must share the same off cohort"
    # Cross-check the registered pooled joint-5 comparison without plotting a
    # selectively pooled joint in place of the full grid.
    primary = {"off": 0, "legacy": 0, "weighted": 0, "fixed": 0, "broken": 0, "n": 0}
    for suite in suites:
        cell = values[f"{suite}_joint5"]
        primary["off"] += cell[methods[0]]["off_successes"]
        legacy = paired[(suite, "joint5", methods[0])]
        weighted = paired[(suite, "joint5", methods[1])]
        primary["legacy"] += sum(legacy.values())
        primary["weighted"] += sum(weighted.values())
        primary["n"] += len(legacy)
        primary["fixed"] += sum(weighted[k] and not legacy[k] for k in legacy)
        primary["broken"] += sum(legacy[k] and not weighted[k] for k in legacy)
    return {"suites": suites, "conditions": conditions, "methods": methods,
            "cells": values, "joint5_primary": primary,
            "unique_rollouts": len(suites) * len(conditions) * 3 * scores["n_scenarios"]}


def aloha_data() -> dict:
    audit = "results/composite_tuning/audit/confirmation_independent_statistics_v1"
    cross = read(f"{audit}/crosscheck.json")
    assert cross["status"] == "verified" and not cross["mismatches"]
    path = "results/composite_tuning/confirmation/analysis/analysis.json"
    report = read(path, cross["checked_hashes"][path])
    raw = read(f"{audit}/extracted_outcomes.json", cross["extracted_outcomes_file_sha256"])
    assert report["kind"] == "confirmation_analysis" and report["validated_outcomes"] == 1920
    assert raw["selected"] == report["selected"]
    assert raw["seeds"] == list(range(3400, 3430))
    assert raw["conditions"] == ["healthy", "offset"] + [f"joint{i}" for i in range(6)]
    families = ["legacy", "dob", "rls", "kalman", "composite", "integral_calibrated"]
    rows = []
    arrays = {}
    for family in families + ["off", "oracle"]:
        candidate = report["selected"].get(family, family)
        array = np.asarray(raw["outcomes"][candidate], dtype=int)
        assert array.shape == (30, 8) and np.isin(array, [0, 1]).all()
        arrays[candidate] = array
        counts = array.sum(axis=0)
        for condition, count in zip(raw["conditions"], counts):
            assert int(count) == report["table"][candidate]["conditions"][condition]["successes"]
        balanced = 0.5 * array[:, 0].mean() + 0.5 * array[:, 1:].mean()
        assert abs(balanced - report["table"][candidate]["balanced_score"]) < 1e-12
        rows.append({"family": family, "candidate": candidate,
                     "healthy_successes": int(counts[0]), "healthy_n": len(array),
                     "faulted_successes": int(counts[1:].sum()), "faulted_n": array[:, 1:].size,
                     "balanced_score_percent": 100 * balanced,
                     "condition_successes": dict(zip(raw["conditions"], counts.tolist()))})
    primary = report["primary"]
    difference = arrays[primary["right"]] - arrays[primary["left"]]
    cluster = 0.5 * difference[:, 0] + 0.5 * difference[:, 1:].mean(axis=1)
    assert np.allclose(cluster, primary["cluster_differences"], atol=1e-14)
    assert abs(cluster.mean() - primary["effect"]) < 1e-14
    assert primary["p_two_sided"] == cross["primary"]["p_two_sided"]
    for key in ["lower", "upper"]:
        assert abs(primary["confidence_interval"][key] - cross["primary"]["confidence_interval"][key]) < 1e-14
    return {"rows": rows, "primary": primary, "unique_rollouts": report["validated_outcomes"]}


def draw(panda: dict, aloha: dict) -> tuple[Path, Path, list[dict]]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.titlesize": 7.2, "axes.labelsize": 6.7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    ink, muted = "#24384B", "#64717D"
    fig = plt.figure(figsize=(5.5, 3.1), facecolor="white")
    fig.text(.025, .965, "A  Panda: joint-dependent repair", fontsize=8, weight="bold", color=ink)
    fig.text(.620, .965, "B  ALOHA: tuned confirmation", fontsize=8, weight="bold", color=ink)
    fig.text(.130, .899, "Paired change vs off; n=20 per arm/cell", fontsize=6.8, color=muted)
    fig.text(.620, .899, "30 seeds × 8 conditions per arm", fontsize=6.8, color=muted)
    cmap = LinearSegmentedColormap.from_list("repair", ["#A34436", "#FAFAF8", "#247B78"])
    norm = Normalize(-100, 100)
    color_checks = []
    for method, y, title in [("legacy_translation", .605, "Legacy translation − off"),
                              ("weighted_full", .305, "Weighted full − off")]:
        ax = fig.add_axes([.13, y, .42, .215])
        grid = np.array([[panda["cells"][f"{s}_{c}"][method]["change_vs_off_pp"]
                          for c in panda["conditions"]] for s in panda["suites"]])
        # Explicit vector cells avoid the indexed-image palette corruption seen
        # when this Matplotlib version embeds a low-bit-depth imshow in a PDF.
        ax.set(xlim=(-.5, 7.5), ylim=(2.5, -.5), aspect="auto")
        for row in range(3):
            for col in range(8):
                color = cmap(norm(grid[row, col]))
                ax.add_patch(Rectangle((col-.5, row-.5), 1, 1,
                                       facecolor=color, edgecolor="none", linewidth=0))
                color_checks.append({
                    "kind": "heatmap_cell", "method": method, "row": row, "column": col,
                    "value_pp": float(grid[row, col]),
                    "figure_xy": [.13 + .42 * (col+.23)/8, y + .215 * (1-(row+.23)/3)],
                    "expected_rgb": [round(255*c) for c in color[:3]],
                })
        ax.set_title(title, loc="left", pad=5, fontweight="bold", color=ink)
        ax.set_yticks(range(3), ["Spatial", "Object", "Goal"])
        ax.set_xticks(range(8), ["H"] + [str(i) for i in range(7)])
        if method == "legacy_translation":
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(length=0, pad=4, colors=ink)
        ax.axvline(.5, color=ink, linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(3):
            for col in range(8):
                v = grid[row, col]
                ax.text(col, row, f"{v:+.0f}" if v else "0", ha="center", va="center",
                        color="white" if abs(v) >= 65 else ink, fontsize=6.8)
    fig.text(.34, .218, "Joint (H: healthy; 0–6: +5 N m)", ha="center", fontsize=6.7, color=ink)
    cax = fig.add_axes([.215, .130, .255, .025])
    # Gouraud shading is a native vector PDF gradient. Separate flat tiles can
    # show white antialiasing seams; the default Colorbar instead rasterizes.
    gradient_x = np.linspace(-100, 100, 257)
    cax.pcolormesh(gradient_x, [0, 1], np.tile(gradient_x, (2, 1)),
                  cmap=cmap, norm=norm, shading="gouraud", rasterized=False)
    cax.set(xlim=(-100, 100), ylim=(0, 1), yticks=[], xticks=[-100, -50, 0, 50, 100])
    for spine in cax.spines.values():
        spine.set_linewidth(.4)
    cax.tick_params(length=2, pad=2, labelsize=6)
    for fraction in [.1, .25, .5, .75, .9]:
        color_checks.append({"kind": "colorbar", "fraction": fraction,
                             "figure_xy": [.215 + .255*fraction, .130 + .025/2],
                             "expected_rgb": [round(255*c) for c in cmap(fraction)[:3]]})
    fig.text(.342, .048, "Success change (percentage points)", ha="center", fontsize=6.7, color=ink)

    ax = fig.add_axes([.756, .451, .218, .386])
    names = {"legacy": "Original", "dob": "DOB", "rls": "RLS", "kalman": "Kalman",
             "composite": "Composite", "integral_calibrated": "Cal. integral",
             "off": "Off", "oracle": "Oracle*"}
    values = [r["balanced_score_percent"] for r in aloha["rows"]]
    colors = ["#A7B6C2"] * 8
    colors[3], colors[4] = "#247B78", "#745689"
    bars = ax.barh(range(8), values, height=.60, color=colors, edgecolor="none", zorder=3)
    for index in [6, 7]:
        bars[index].set(facecolor="white", edgecolor=muted, linewidth=.8,
                        hatch="//" if index == 7 else None)
    ax.set_yticks(range(8), [names[r["family"]] for r in aloha["rows"]])
    ax.set_xlim(0, 63)
    ax.set_xticks([0, 25, 50], ["0", "25", "50"])
    ax.set_ylim(7.7, -.7)
    ax.set_xlabel("Balanced score (%)", labelpad=2)
    ax.axhline(5.55, color="#CFD6DC", linewidth=.6)
    ax.grid(axis="x", color="#E8ECEF", linewidth=.6, zorder=0)
    ax.tick_params(length=0, pad=3, colors=ink)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(values):
        ax.text(value + 1.5, i, f"{value:.1f}", va="center", fontsize=6.4, color=ink)
    primary = aloha["primary"]
    ci = primary["confidence_interval"]
    effect, lower, upper = 100 * primary["effect"], 100 * ci["lower"], 100 * ci["upper"]
    fig.text(.620, .327, "Composite − Kalman", fontsize=7.1, color=ink, weight="bold")
    fig.text(.620, .283, f"{effect:+.2f} pp; exact p={primary['p_two_sided']:.3f}", fontsize=6.8, color=ink)
    ci_ax = fig.add_axes([.702, .132, .267, .116])
    ci_ax.axvline(0, color="#A8B2BB", linestyle="--", linewidth=.8)
    ci_ax.errorbar(effect, 0, xerr=[[effect-lower], [upper-effect]], fmt="o",
                   color="#745689", markersize=3.4, linewidth=1.6, capsize=3)
    ci_ax.set(xlim=(-12, 12), ylim=(-1, 1), yticks=[], xticks=[-10, 0, 10])
    ci_ax.set_xticklabels(["−10", "0", "+10"])
    ci_ax.tick_params(length=2, pad=2, labelsize=6.3, colors=ink)
    for side in ["top", "left", "right"]:
        ci_ax.spines[side].set_visible(False)
    ci_ax.spines["bottom"].set_color("#CDD4DA")
    fig.text(.835, .050, "95% paired-seed CI (pp)", ha="center", fontsize=6.6, color=ink)
    fig.text(.620, .009, "* Privileged diagnostic, not an upper bound", fontsize=5.9, color=muted)
    pdf, png = OUT / "fig_results_analysis.pdf", OUT / "fig_results_analysis.png"
    fig.savefig(pdf, metadata={"Title": "Complete confirmation outcomes: joint effects and tuned estimators",
                              "Creator": "paper/make_results_figure.py", "CreationDate": None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png, color_checks


def verify_pdf_colors(pdf: Path, checks: list[dict]) -> dict:
    """Check the PDF renderer, not merely Matplotlib's separate PNG backend."""
    images = subprocess.run(["pdfimages", "-list", str(pdf)], check=True,
                            capture_output=True, text=True).stdout
    image_rows = [line for line in images.splitlines() if re.match(r"\s*\d+\s+\d+\s+\w", line)]
    if image_rows:
        raise ValueError("Result figure must contain vector cells, not embedded raster images")
    with tempfile.TemporaryDirectory(prefix="vla-results-pdf-render-") as tmp:
        prefix = Path(tmp) / "render"
        subprocess.run(["pdftoppm", "-r", "300", "-singlefile", "-png", str(pdf), str(prefix)],
                       check=True, capture_output=True)
        render_path = prefix.with_suffix(".png")
        rendered_hash = sha256(render_path)
        pixels = np.asarray(Image.open(render_path).convert("RGB"))
        height, width, _ = pixels.shape
        results = []
        for check in checks:
            fx, fy = check["figure_xy"]
            x, y = round(fx*width), round((1-fy)*height)
            observed = pixels[y, x].astype(int)
            difference = int(np.abs(observed - check["expected_rgb"]).max())
            tolerance = 2 if check["kind"] == "heatmap_cell" else 3
            if difference > tolerance:
                raise ValueError(f"PDF cell/gradient color mismatch: {check}, observed={observed.tolist()}")
            results.append({**check, "pdf_pixel_rgb": observed.tolist(),
                            "max_channel_error": difference, "tolerance": tolerance})
    version = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True, check=True)
    return {"status": "passed", "renderer": (version.stdout + version.stderr).splitlines()[0],
            "dpi": 300, "pdf_sha256": sha256(pdf), "rendered_png_sha256": rendered_hash,
            "rendered_dimensions": [width, height], "embedded_raster_images": len(image_rows),
            "heatmap_cells_checked": sum(c["kind"] == "heatmap_cell" for c in checks),
            "colorbar_samples_checked": sum(c["kind"] == "colorbar" for c in checks),
            "max_channel_error": max(r["max_channel_error"] for r in results), "samples": results}


def main() -> None:
    panda, aloha = panda_data(), aloha_data()
    pdf, png, checks = draw(panda, aloha)
    files = (pdf, png)
    rendering = verify_pdf_colors(pdf, checks)
    receipt = {
        "schema_version": 1, "script_sha256": sha256(Path(__file__)),
        "sources": SOURCES, "outputs": {p.name: sha256(p) for p in files},
        "figure_size_inches": [5.5, 3.1],
        "pdf_render_verification": rendering,
        "panda_metric": "100*(observed fixed - observed broken)/20, each method versus shared off cohort",
        "aloha_metric": "50*healthy success fraction + 50*mean of seven fault success fractions",
        "uncertainty": "Only the stored primary composite-minus-Kalman paired whole-seed percentile bootstrap95% CI is plotted; no per-arm independent CIs",
        "interpretation": "Complete confirmation cells; descriptive scores and changes do not establish a total ranking or equivalence. Policy sampling is independent across arms.",
        "panda": panda, "aloha": aloha,
    }
    (OUT / "results_figure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"panda_rollouts": panda["unique_rollouts"], "aloha_rollouts": aloha["unique_rollouts"],
                      "joint5_primary": panda["joint5_primary"], "outputs": receipt["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
