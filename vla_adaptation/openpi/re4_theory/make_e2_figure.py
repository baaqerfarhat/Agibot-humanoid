#!/usr/bin/env python3
"""Figure: the E2 predictor intervention as a mechanism. For each suite, the r_y estimate over time
(median across episodes, IQR band) for U and C on the faulted arms, the truth line, and on a second row
the signed remaining r_y disturbance; the outcome counts are annotated. From the E2 telemetry (gzipped)."""
import gzip, json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parents[2]; R = HERE / "results/iclr_unified_v1/E2_core"


def trajectories(arm, ch=4, T=150):
    per = {}
    with gzip.open(R / f"{arm}/telemetry.jsonl.gz", "rt") as fh:
        for line in fh:
            s = json.loads(line)
            if s.get("type") != "step" or s.get("phase") != "rollout" or s.get("arm") != "adaptive":
                continue
            per.setdefault(int(s["episode"]), []).append((int(s["t"]), float(s["f_hat"][ch]), float(s["correction"][ch])))
    F = np.full((len(per), T), np.nan); C = np.full((len(per), T), np.nan)
    for i, (ep, steps) in enumerate(sorted(per.items())):
        for t, f, c in steps:
            if t < T:
                F[i, t] = f; C[i, t] = c
    return F, C


def main():
    scores = {s: json.loads((R / f"score_{s}.json").read_text()) for s in ("libero_spatial", "libero_10")}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), constrained_layout=True, sharex="col")
    for j, suite in enumerate(("libero_spatial", "libero_10")):
        for arm, col, lab in (("faulted_U", "#ef8a62", "U (unconstrained)"), ("faulted_C", "#2166ac", "C (r_y response 0.254)")):
            F, C = trajectories(f"{suite}_{arm}"); t = np.arange(F.shape[1]) * 0.05
            med = np.nanmedian(F, axis=0); q1, q3 = np.nanpercentile(F, 25, axis=0), np.nanpercentile(F, 75, axis=0)
            axes[0, j].plot(t, med / 0.05, color=col, lw=1.6, label=lab); axes[0, j].fill_between(t, q1 / 0.05, q3 / 0.05, color=col, alpha=0.15)
            rem = np.nanmedian(0.05 + C, axis=0); axes[1, j].plot(t, rem / 0.05, color=col, lw=1.6)
        axes[0, j].axhline(1.0, color="k", lw=0.8, ls=":"); axes[0, j].set_ylim(-0.1, 1.3); axes[1, j].axhline(0.0, color="k", lw=0.8, ls=":"); axes[1, j].set_ylim(-0.3, 1.1)
        sc = scores[suite]["success"]; o = scores[suite]["observation"]
        axes[0, j].set_title(f"{suite}: off {sc['faulted_off'][0]}, U {sc['faulted_U'][0]}, C {sc['faulted_C'][0]}, oracle {sc['faulted_oracle'][0]} of 60", fontsize=8.5)
        axes[1, j].text(0.98, 0.95, f"mean |r_y obs. bias|: U {o['abs_obs_bias_ry']['U']:.3f}, C {o['abs_obs_bias_ry']['C']:.3f}\nsigned remaining (last 50): U {o['remaining_ry']['U']:+.3f}, C {o['remaining_ry']['C']:+.3f}",
                        transform=axes[1, j].transAxes, ha="right", va="top", fontsize=6.8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        axes[1, j].set_xlabel("time in episode (s)", fontsize=8)
        for ax in axes[:, j]:
            ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("r_y estimate / fault (median, IQR)", fontsize=8); axes[1, 0].set_ylabel("remaining r_y disturbance / fault", fontsize=8); axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    out = HERE / "results/iclr_unified_v1/figures/e2_mechanism.pdf"; fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200); print("wrote", out)


if __name__ == "__main__":
    main()
