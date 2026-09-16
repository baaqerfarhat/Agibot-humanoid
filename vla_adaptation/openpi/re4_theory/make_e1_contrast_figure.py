#!/usr/bin/env python3
"""Figure: execution memory as a contrast between two interfaces. Left: Panda (E1 v2, ee deviation, cm);
right: ALOHA (E1 on ALOHA, joint deviation on the left arm, rad). Median over locked checkpoints per branch,
with the remaining-disturbance fraction as a dashed line on a twin axis. Matplotlib, vector PDF."""
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parents[2]
BR = [("faulted_off", "faulted, no correction", "#b2182b"), ("legacy_from_zero", "legacy law from zero", "#ef8a62"), ("innovation_from_zero", "innovation law from zero", "#2166ac"),
      ("hold_supplied", "held supplied estimate", "#67a9cf"), ("exact_cancellation", "exact cancellation", "#4d4d4d")]


def traces(run, space, joints=(0, 1, 2, 3, 4, 5)):
    d = json.loads(pathlib.Path(run).read_text()); out = {}
    for name, _, _ in BR:
        dev, rem = [], []
        for ep in d["episodes"]:
            for cp in ep["checkpoints"]:
                if cp.get("status") != "ok" or name not in cp["branches"]:
                    continue
                h = cp["branches"]["healthy"]; br = cp["branches"][name]; L = min(len(h), len(br))
                if space == "ee":
                    dev.append([100 * np.linalg.norm(np.array(s["ee_pos"]) - np.array(hh["ee_pos"])) for s, hh in zip(br[:L], h[:L])])
                    rem.append([np.linalg.norm(np.array(s["remaining_disturbance"])[3:6]) / 0.05 / np.sqrt(3) for s in br[:L]])
                else:
                    dev.append([np.linalg.norm(np.array(s["q"])[list(joints)] - np.array(hh["q"])[list(joints)]) for s, hh in zip(br[:L], h[:L])])
                    rem.append([np.linalg.norm(np.array(s["remaining_disturbance"])[list(joints)]) / (0.02 * np.sqrt(6)) for s in br[:L]])
        L = min(len(x) for x in dev); out[name] = (np.median(np.array([x[:L] for x in dev]), axis=0), np.median(np.array([x[:L] for x in rem]), axis=0))
    return out


def main():
    panda = traces(HERE / "results/iclr_unified_v1/E1_v2/pass2_test.json", "ee"); aloha = traces(HERE / "results/iclr_unified_v1/E1_aloha_v2/pass2_test.json", "joint")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
    for ax, data, title, ylab, dt in ((axes[0], panda, "Panda, OSC_POSE (integrating)", "end-effector deviation from healthy (cm)", 0.05), (axes[1], aloha, "ALOHA, position servos (forgetting)", "left-arm joint deviation (rad)", 0.02)):
        for name, label, col in BR:
            dev, rem = data[name]; t = np.arange(len(dev)) * dt
            ax.plot(t, dev, color=col, lw=1.6, label=label)
        ax.set_title(title, fontsize=9); ax.set_xlabel("time after checkpoint (s)", fontsize=8); ax.set_ylabel(ylab, fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        for name in ("legacy_from_zero", "innovation_from_zero"):
            dev, rem = data[name]; t = np.arange(len(rem)) * dt; ax2.plot(t, rem, color="#2166ac" if "innov" in name else "#ef8a62", lw=1.0, ls="--", alpha=0.8)
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("remaining disturbance / fault (dashed)", fontsize=7); ax2.tick_params(labelsize=7)
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    out = HERE / "results/iclr_unified_v1/figures/e1_contrast_panda_aloha.pdf"; fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200); print("wrote", out)
    print("Panda endpoints (cm):", {n: round(float(panda[n][0][-1]), 3) for n, _, _ in BR}); print("ALOHA endpoints (rad):", {n: round(float(aloha[n][0][-1]), 4) for n, _, _ in BR})


if __name__ == "__main__":
    main()
