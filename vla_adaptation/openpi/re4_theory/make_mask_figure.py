#!/usr/bin/env python3
"""Figure: what the correction mask costs and buys on libero_10 (the E2 keys, 60 per arm). Faulted success for
off / U / C / oracle under the rotation-only mask and under the six-channel mask, with the healthy-adaptive
arms alongside (healthy off = 56). Bars with exact counts; paired task-clustered intervals are in the scores."""
import json, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parents[2]
rot = json.loads((HERE / "results/iclr_unified_v1/E2_core/score_libero_10.json").read_text())["success"]
six = json.loads((HERE / "results/six_channel/score_libero_10_six.json").read_text())["success"]
arms = [("faulted_off", "off"), ("faulted_U", "U"), ("faulted_C", "C"), ("faulted_oracle", "oracle"), ("healthy_U", "healthy U"), ("healthy_C", "healthy C")]
fig, ax = plt.subplots(figsize=(4.6, 2.6), constrained_layout=True); x = range(len(arms)); w = 0.38
for i, (src, lab, col) in enumerate(((rot, "rotation-only mask {3,4,5}", "#ef8a62"), (six, "six-channel mask {0..5}", "#2166ac"))):
    vals = [src[a][0] for a, _ in arms]; bars = ax.bar([xx + (i - 0.5) * w for xx in x], vals, w, color=col, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", va="bottom", fontsize=7)
ax.axhline(56, color="k", lw=0.8, ls=":"); ax.text(len(arms) - 0.5, 56.8, "healthy, no correction: 56", ha="right", fontsize=6.5)
ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in arms], fontsize=8); ax.set_ylabel("successes / 60 (libero_10, E2 keys)", fontsize=8); ax.set_ylim(0, 64); ax.tick_params(labelsize=7)
ax.axvline(3.5, color="0.7", lw=0.8); ax.text(1.5, 62, "faulted, +0.05 on six channels", ha="center", fontsize=7); ax.text(4.5, 62, "healthy", ha="center", fontsize=7)
ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.grid(axis="y", alpha=0.25)
out = HERE / "results/iclr_unified_v1/figures/libero10_mask.pdf"; fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200); print("wrote", out)
