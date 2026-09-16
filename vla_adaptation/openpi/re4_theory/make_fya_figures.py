#!/usr/bin/env python3
"""Figures for the FrozenYet Adaptive recovery campaign (results/frozen_yet_adaptive_deadline_v1/figures/).

  fya_benefits.pdf     per scenario: source-level benefit B = J_off - J_A (m^2 s) for nt / innovation / reference on the
                       locked test, every source shown (negative sources visible), task-clustered mean and 95 % interval
  fya_delay_cap.pdf    adapted cost J_A by scenario for each law (delay / cap / sign effects) plus the off cost
  fya_forecast.pdf     frozen point forecast of B versus measured B on the locked test, sign agreement in the title;
                       the interval labels (all inconclusive if so) are stated in the caption text
  fya_healthy.pdf      healthy harm: J_p of healthy nt / innovation per source (false updates), test and qualification
  fya_stage2.pdf       reacting-policy bridge: success counts per arm and condition with paired wins/losses
"""
import argparse, csv, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCEN = ["reference", "delay10", "cap_half", "sign_reverse"]; LAWS = ["nt", "innovation", "reference"]
COL = {"nt": "#1f77b4", "innovation": "#ff7f0e", "reference": "#2ca02c", "off": "#7f7f7f"}


def load_benefits(path):
    return list(csv.DictReader(open(path)))


def fig_benefits(rows, iv, out):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, sc in zip(axes, SCEN):
        for i, law in enumerate(LAWS):
            B = np.array([float(r["B"]) for r in rows if r["scenario"] == sc and r["law"] == law]) * 1e5
            ax.scatter(np.full(len(B), i) + np.random.default_rng(0).uniform(-.15, .15, len(B)), B, s=14, color=COL[law], alpha=.7)
            agg = iv["benefits"][sc][law]["B_task_clustered"]
            ax.errorbar([i], [agg["mean"] * 1e5], yerr=[[(agg["mean"] - agg["ci95"][0]) * 1e5], [(agg["ci95"][1] - agg["mean"]) * 1e5]], fmt="_", color="k", capsize=4, ms=14, lw=1.5)
        ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(3)); ax.set_xticklabels(["NT", "innov.", "ref."]); ax.set_title(sc)
    axes[0].set_ylabel("benefit  $J_{off}-J_A$  ($10^{-5}$ m$^2$ s)")
    fig.suptitle("Locked test: source-level physical benefit (dots) with task-clustered mean and 95 % interval", fontsize=10)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_delay_cap(metrics, out):
    by = {}
    for r in metrics:
        by.setdefault(r["branch"], []).append(float(r["J_p"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(len(SCEN)); w = .2
    for i, arm in enumerate(["off", "nt", "innovation", "reference"]):
        med = [np.median(by[f"{sc}__{arm}"]) * 1e5 for sc in SCEN]
        q1 = [np.percentile(by[f"{sc}__{arm}"], 25) * 1e5 for sc in SCEN]; q3 = [np.percentile(by[f"{sc}__{arm}"], 75) * 1e5 for sc in SCEN]
        ax.bar(x + (i - 1.5) * w, med, w, color=COL[arm], label=arm, yerr=[np.subtract(med, q1), np.subtract(q3, med)], capsize=2, error_kw=dict(lw=.8))
    ax.set_xticks(x); ax.set_xticklabels(SCEN); ax.set_ylabel("$J_p$ median [IQR]  ($10^{-5}$ m$^2$ s)"); ax.legend(fontsize=8, ncol=4)
    ax.set_title("Adapted cost by scenario: delay, halved cap, reversed sign", fontsize=10)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_forecast(eval_rows, summary, out):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for law in LAWS:
        rr = [r for r in eval_rows if r["law"] == law]
        pred = np.array([float(r["Jhat_off"]) - float(r["Jhat_A"]) for r in rr]) * 1e5; meas = np.array([float(r["B"]) for r in rr]) * 1e5
        ax.scatter(meas, pred, s=12, color=COL[law], alpha=.7, label=law)
    lim = ax.get_xlim(); ax.plot([min(lim[0], 0), lim[1]], [min(lim[0], 0), lim[1]], "k--", lw=.7); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
    ax.set_xlabel("measured $B$ ($10^{-5}$ m$^2$ s)"); ax.set_ylabel("frozen point forecast of $B$")
    sa = summary.get("point_forecast_sign_agreement", {})
    ax.set_title("Forecast vs measured; sign agreement " + ", ".join(f"{k} {v:.2f}" for k, v in sa.items()), fontsize=8); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_healthy(metrics_test, metrics_qual, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for j, (lab, metrics) in enumerate((("test", metrics_test), ("qualification", metrics_qual))):
        for i, br in enumerate(("healthy_nt", "healthy_innovation")):
            v = np.array([float(r["J_p"]) for r in metrics if r["branch"] == br]) * 1e5
            ax.scatter(np.full(len(v), 2 * j + i) + np.random.default_rng(1).uniform(-.15, .15, len(v)), v, s=12, color=COL[br.split("_")[1]], alpha=.7)
            ax.plot([2 * j + i - .25, 2 * j + i + .25], [np.median(v)] * 2, color="k")
    ax.set_xticks(range(4)); ax.set_xticklabels(["NT\ntest", "innov.\ntest", "NT\nqual.", "innov.\nqual."], fontsize=8)
    ax.set_ylabel("healthy false-update cost $J_p$ ($10^{-5}$ m$^2$ s)"); ax.set_title("Healthy continuations: deviation caused by false updates", fontsize=10)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_stage2(s2, out):
    conds = [("healthy", ["healthy_off", "healthy_nt", "healthy_innovation"]), ("faulted", ["fault_off", "fault_nt", "fault_innovation"]), ("delayed", ["fault_off", "delay_nt", "delay_innovation"])]
    fig, ax = plt.subplots(figsize=(6.0, 3.0)); x = np.arange(3); w = .25
    for i, arm in enumerate(["off", "nt", "innovation"]):
        vals = [s2["arms"][names[i]]["successes"] for _, names in conds]
        ax.bar(x + (i - 1) * w, vals, w, color=COL[arm], label=arm)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + .3, str(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([c for c, _ in conds]); ax.set_ylabel("successes / 20"); ax.set_ylim(0, 22); ax.legend(fontsize=8)
    ax.set_title("Stage 2 reacting-policy bridge (states 13, 14; onset at step 30; delay 10 steps)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("results/frozen_yet_adaptive_deadline_v1")); a = ap.parse_args()
    R = a.root; F = R / "figures"; F.mkdir(exist_ok=True)
    ben = load_benefits(R / "analysis/test_score/source_benefits.csv"); iv = json.load(open(R / "analysis/test_score/source_intervals.json"))
    met = list(csv.DictReader(open(R / "analysis/test_score/source_metrics.csv"))); metq = list(csv.DictReader(open(R / "qualification/score/source_metrics.csv")))
    fig_benefits(ben, iv, F / "fya_benefits.pdf"); fig_delay_cap(met, F / "fya_delay_cap.pdf"); fig_healthy(met, metq, F / "fya_healthy.pdf")
    ev = list(csv.DictReader(open(R / "analysis/test_evaluation/evaluation_rows.csv"))); summ = json.load(open(R / "analysis/test_evaluation/evaluation_summary.json"))
    fig_forecast(ev, summ, F / "fya_forecast.pdf")
    if (R / "analysis/stage2_summary.json").exists():
        fig_stage2(json.load(open(R / "analysis/stage2_summary.json")), F / "fya_stage2.pdf")
    print("figures written to", F)


if __name__ == "__main__":
    main()
