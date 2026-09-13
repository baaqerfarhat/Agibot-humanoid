"""Render complete-confirmation tracking diagnostics with explicit RMS semantics."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
BASE = "results/composite_tuning/confirmation/diagnosis_v1"
SOURCES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: str, expected: str | None = None) -> dict:
    value = digest(ROOT / path)
    if expected is not None and value != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    SOURCES[path] = value
    return json.loads((ROOT / path).read_text())


def load_data() -> dict:
    manifest = read(f"{BASE}/artifact_manifest.json")
    assert manifest["status"] == "complete"
    read("results/composite_tuning/audit/diagnosis_confirmation_v1_run_receipt.json",
         manifest["source_run_receipt_sha256"])
    summary_path = f"{BASE}/summary_compact.json"
    summary = read(summary_path, manifest["local_outputs"][summary_path]["sha256"])
    comparison_path = f"{BASE}/diagnostic_comparison.json"
    comparison = read(comparison_path, manifest["additional_outputs"][comparison_path]["sha256"])
    assert summary["status"] == "complete" and summary["phase"] == "confirmation"
    assert summary["correction_coordinates"] == summary["tracking_position_coordinates"] == list(range(6))
    conditions = ["healthy", "offset"] + [f"joint{i}" for i in range(6)]
    candidates = {family: summary["family_to_candidate"][family][0] for family in ["kalman", "composite"]}
    keys, total_steps = set(), 0
    episode_counts = {}
    accumulators = {}
    ep_path = f"{BASE}/per_episode.jsonl.gz"
    SOURCES[ep_path] = digest(ROOT / ep_path)
    assert SOURCES[ep_path] == manifest["local_outputs"][ep_path]["sha256"]
    uncompressed_hash = hashlib.sha256()
    with gzip.open(ROOT / ep_path, "rb") as stream:
        for line in stream:
            uncompressed_hash.update(line)
            ep = json.loads(line)
            key = (ep["condition"], ep["candidate"], ep["seed"])
            assert key not in keys
            keys.add(key)
            total_steps += ep["actual_steps"]
            assert ep["seed"] in range(3400, 3430)
            if ep["candidate"] not in candidates.values():
                continue
            assert ep["condition"] in conditions
            group = (ep["condition"], ep["candidate"])
            episode_counts[group] = episode_counts.get(group, 0) + 1
            assert ep["correction_coordinates"] == ep["tracking_position_coordinates"] == list(range(6))
            late = ep["windows"]["last100"]
            start, end = ep["window_intervals"]["last100"]
            assert end == ep["actual_steps"] and end - start == late["samples"] == 100
            assert start >= ep["window_intervals"]["first20"][1]
            for metric in ["position_reference_error", "diagnostic_parameter_error", "active_tracking_increment"]:
                moments = late["metrics"][metric]
                rms = np.asarray(moments["rms"], dtype=float)
                assert rms.shape == (6,) and np.isfinite(rms).all()
                assert abs(float(np.linalg.norm(rms)) - moments["rms_norm"]) < 1e-12
                # Pool squared coordinate errors, not arithmetic episode RMS.
                key_metric = (*group, metric)
                accumulators[key_metric] = accumulators.get(key_metric, np.zeros(6)) + rms**2 * late["samples"]
    expected_keys = {(condition, candidate, seed) for condition in conditions
                     for candidate in summary["selected_configurations"] for seed in range(3400, 3430)}
    assert keys == expected_keys
    assert len(keys) == summary["episodes"] == manifest["episodes"]
    assert len(keys) == comparison["confirmation_episode_totals"]["episodes"]
    assert total_steps == comparison["confirmation_episode_totals"]["total_steps"]
    authoritative = [v for k, v in manifest["authoritative_outputs"].items() if k.endswith("per_episode.jsonl")]
    assert len(authoritative) == 1 and uncompressed_hash.hexdigest() == authoritative[0]["sha256"]
    rows = []
    concentration = []
    for condition in conditions:
        row = {"condition": condition, "arms": {}}
        for family, candidate in candidates.items():
            group = (condition, candidate)
            episodes = episode_counts[group]
            assert episodes == 30
            n = 100 * episodes
            data = {"candidate": candidate, "episodes": episodes, "samples": n}
            for metric in ["position_reference_error", "diagnostic_parameter_error", "active_tracking_increment"]:
                energy = accumulators[(*group, metric)]
                rms = float(np.sqrt(energy.sum() / n))
                stored = summary["windows"][condition][candidate]["last100"]
                assert stored["episodes"] == episodes and stored["samples"] == n
                assert abs(rms - stored["metrics"][metric]["rms_norm"]) < 1e-12
                data[metric + "_rms_rad"] = rms
                if family == "composite" and metric == "active_tracking_increment":
                    fractions = energy / energy.sum()
                    assert np.allclose(fractions, stored["metrics"][metric]["coordinate_squared_energy_fraction"], atol=1e-12)
                    data["tracking_increment_energy_fraction"] = fractions.tolist()
                    concentration.append(float(fractions[1]))
            compared = [r for r in comparison["comparison"] if r["phase"] == "confirmation"
                        and r["condition"] == condition and r["candidate"] == candidate and r["window"] == "last100"]
            assert len(compared) == 1
            assert abs(data["position_reference_error_rms_rad"] - compared[0]["position_reference_error_rms"]) < 1e-12
            assert abs(data["diagnostic_parameter_error_rms_rad"] - compared[0]["parameter_error_rms"]) < 1e-12
            row["arms"][family] = data
        rows.append(row)
    offset = next(r for r in rows if r["condition"] == "offset")
    tradeoff = []
    for metric, label in [("position_reference_error", "Reference"), ("diagnostic_parameter_error", "Parameter")]:
        kf = offset["arms"]["kalman"][metric + "_rms_rad"]
        composite = offset["arms"]["composite"][metric + "_rms_rad"]
        tradeoff.append({"metric": metric, "label": label, "kalman_mrad": kf * 1000,
                         "composite_mrad": composite * 1000, "relative_change_percent": 100 * (composite / kf - 1)})
    return {"rows": rows, "offset_tradeoff": tradeoff,
            "joint1_active_tracking_energy_fraction_range": [min(concentration), max(concentration)],
            "total_confirmation_episodes_checked": len(keys), "total_confirmation_steps_checked": total_steps,
            "plotted_episodes": sum(episode_counts.values()), "uncompressed_episode_sha256": uncompressed_hash.hexdigest()}


def draw(data: dict) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "xtick.labelsize": 6.6, "ytick.labelsize": 6.6})
    ink, muted = "#24384B", "#64717D"
    kalman_color, composite_color = "#247B78", "#745689"
    fig = plt.figure(figsize=(5.5, 1.9), facecolor="white")
    fig.text(.025, .942, "A  Reference tracking across all conditions", fontsize=7.8, weight="bold", color=ink)
    fig.text(.620, .942, "B  Exact command offset", fontsize=7.8, weight="bold", color=ink)
    ax = fig.add_axes([.102, .298, .432, .427])
    kalman = np.array([r["arms"]["kalman"]["position_reference_error_rms_rad"] for r in data["rows"]]) * 1000
    composite = np.array([r["arms"]["composite"]["position_reference_error_rms_rad"] for r in data["rows"]]) * 1000
    x = np.arange(len(data["rows"]))
    ax.vlines(x, composite, kalman, color="#B8C3CC", linewidth=1.0, zorder=2)
    ax.scatter(x, kalman, s=17, marker="o", color=kalman_color, label="Kalman", zorder=3)
    ax.scatter(x, composite, s=16, marker="s", color=composite_color, label="Composite", zorder=3)
    ax.set(xlim=(-.5, len(x) - .5), ylim=(0, 12), yticks=[0, 4, 8, 12])
    ax.set_xticks(x, ["H", "O"] + [str(i) for i in range(6)])
    ax.set_ylabel("Reference RMS (mrad)", fontsize=6.9, labelpad=3)
    ax.tick_params(length=2, pad=2, colors=ink)
    ax.grid(axis="y", linewidth=.5, color="#E3E8EC", zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.03), ncol=2, frameon=False,
              borderaxespad=0, fontsize=6.6, handletextpad=.2, columnspacing=1.4)
    fig.text(.318, .187, "H: healthy; O: offset; 0–5: torque joint", ha="center", fontsize=6.4, color=muted)

    right = fig.add_axes([.727, .298, .241, .427])
    change = [r["relative_change_percent"] for r in data["offset_tradeoff"]]
    right.bar([0, 1], change, width=.55, color=[kalman_color, "#AA4D3B"], alpha=.90, zorder=3)
    right.axhline(0, linewidth=.7, color="#9EABB5")
    right.set(xlim=(-.6, 1.6), ylim=(-30, 22), yticks=[-20, 0, 20])
    right.set_xticks([0, 1], [r["label"] for r in data["offset_tradeoff"]])
    right.set_ylabel("RMS change (%)", fontsize=6.9, labelpad=2)
    right.tick_params(length=2, pad=2, colors=ink)
    right.grid(axis="y", linewidth=.5, color="#E3E8EC", zorder=0)
    for i, row in enumerate(data["offset_tradeoff"]):
        v = row["relative_change_percent"]
        right.text(i, v + (1.2 if v > 0 else -1.3), f"{v:+.1f}%",
                   va="bottom" if v > 0 else "top", ha="center", fontsize=6.8, color=ink)
    fig.text(.848, .796, "Composite relative to Kalman", ha="center", fontsize=6.5, color=muted)
    ref, param = data["offset_tradeoff"]
    fig.text(.848, .185, f"{ref['kalman_mrad']:.2f}→{ref['composite_mrad']:.2f}  |  "
                        f"{param['kalman_mrad']:.2f}→{param['composite_mrad']:.2f} mrad",
             ha="center", fontsize=6.3, color=muted)
    for a in [ax, right]:
        for side in ["top", "right"]:
            a.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            a.spines[side].set_color("#CDD4DA")
            a.spines[side].set_linewidth(.6)
    fig.text(.025, .076, "Last 100 steps × 30 episodes per arm/condition; pooled six-joint vector RMS.", fontsize=6.5, color=ink)
    fig.text(.025, .015, "Each rollout uses its own raw-command reference; selected controllers differ beyond tracking feedback.", fontsize=6.2, color=muted)
    pdf, png = OUT / "fig_tracking_analysis.pdf", OUT / "fig_tracking_analysis.png"
    fig.savefig(pdf, metadata={"Title": "Tracking and parameter estimation in completed ALOHA confirmation",
                              "Creator": "paper/make_tracking_figure.py", "CreationDate": None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def main() -> None:
    data = load_data()
    outputs = draw(data)
    receipt = {"schema_version": 1, "script_sha256": digest(Path(__file__)), "sources": SOURCES,
               "outputs": {p.name: digest(p) for p in outputs}, "figure_size_inches": [5.5, 1.9],
               "rms_definition": "sqrt((1/30)*sum_episode((1/100)*sum_last100steps(sum_joint0to5(error_j^2))))",
               "aggregation": "Equal 100-step late windows from each of 30 episodes; pool squared errors before final root, not mean episode RMS and not per-coordinate RMS.",
               "offset_relative_change": "100*(composite_RMS/Kalman_RMS-1); exact injected command offset is the parameter target.",
               "scope": "All eight confirmation conditions; each controller follows its own raw-policy-command reference. No common counterfactual reference, causal isolation, confidence intervals or statistical ranking is claimed.",
               "tracking_concentration_scope": "Unplotted diagnostic: fraction of last100 pre-projection tracking-increment energy on corrected left joint1, normalized over six active coordinates separately per condition.",
               "data": data}
    (OUT / "tracking_figure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"episodes": data["total_confirmation_episodes_checked"],
                      "steps": data["total_confirmation_steps_checked"],
                      "offset_tradeoff": data["offset_tradeoff"],
                      "tracking_concentration_range": data["joint1_active_tracking_energy_fraction_range"],
                      "outputs": receipt["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
