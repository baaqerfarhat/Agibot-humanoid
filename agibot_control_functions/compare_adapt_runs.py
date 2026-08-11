#!/usr/bin/env python3
"""Compare frozen vs adapted hardware runs from `deploy_x2_box_adapt.py` logs.

    python compare_adapt_runs.py run_logs/*_box_adapt_*.csv

Runs are grouped automatically by the adaptation config recorded in each run's
`.meta.json`, so you just point it at every log and it sorts out which are the frozen
control arm and which are adapted.

Two things it does that a hand comparison gets wrong:

  * **Common window only.** Every error average is taken over the frame range that
    ALL runs reached. A run that falls early otherwise looks like the most accurate
    one, because it only ever attempted the easy standing part of the motion.
  * **No verdict from one pair.** With fewer than 3 runs per arm it reports the
    numbers and explicitly refuses to call a winner. This task is chaotic; in sim a
    1e-6 action perturbation moved leg tracking error 1.7 deg over 2.4 s.

Reads only the CSVs the deploy script writes -- numpy + stdlib, no ROS, no torch.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

import numpy as np


def load_run(csv_path: str) -> dict | None:
    base = csv_path[:-4]
    adapt_path = base + "_adapt.csv"
    meta_path = base + ".meta.json"
    if not os.path.exists(adapt_path):
        print(f"[skip] {os.path.basename(csv_path)}: no _adapt.csv sidecar "
              "(not a deploy_x2_box_adapt.py run?)")
        return None

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    ad = meta.get("adapt", {})

    with open(adapt_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("phase") == "policy"]
    if not rows:
        print(f"[skip] {os.path.basename(csv_path)}: never reached the policy phase")
        return None

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.asarray(out)

    with open(csv_path) as f:
        roll = np.asarray([float(r["roll"]) for r in csv.DictReader(f)
                           if r.get("phase") == "policy" and r.get("roll")])

    gain = float(ad.get("gain", 0.0))
    # "20260811_131013_box_adapt_<policy>_<tag>" -> "131013_<tag>"
    parts = os.path.basename(base).split("_")
    return {
        "name": f"{parts[1]}_{parts[-1]}" if len(parts) > 2 else os.path.basename(base),
        "arm": "frozen" if gain == 0.0 else f"adapted g{gain:g} {ad.get('mask', '?')}",
        "gain": gain,
        "engaged": bool(meta.get("engage", False)),
        "policy": os.path.basename(str(meta.get("policy", "?"))),
        "frame": col("frame"),
        "err_masked": col("err_masked_deg"),
        "err_leg": col("err_leg_deg"),
        "err_all": col("err_all_deg"),
        "drift": col("drift"),
        "loop_ms": col("loop_ms"),
        "dev_clamped": col("dev_clamped"),
        "max_roll": float(np.nanmax(np.abs(roll))) if roll.size else float("nan"),
        "mask_joints": ad.get("mask_joints", []),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="Main run CSVs (globs are fine).")
    ap.add_argument("--window", type=int, default=0,
                   help="Force the comparison window to end at this frame instead of "
                        "the min frame every run reached.")
    args = ap.parse_args()

    # `*.csv` also matches the sidecars; they are read via their main run, not directly.
    paths = sorted({p for pat in args.logs for p in glob.glob(pat)
                    if not p.endswith("_adapt.csv")})
    runs = [r for r in (load_run(p) for p in paths) if r]
    if not runs:
        raise SystemExit("no usable runs found")

    dry = [r["name"] for r in runs if not r["engaged"]]
    if dry:
        print(f"[warn] {len(dry)} run(s) were DRY (nothing published); the robot never "
              "moved, so their tracking error is not a control result.\n")

    last = min(int(r["frame"].max()) for r in runs)
    window = args.window or last
    print("=" * 78)
    print(f"  runs: {len(runs)}   policy: {sorted({r['policy'] for r in runs})}")
    print(f"  comparison window: frames 0-{window} "
          f"(shortest run reached {last})")
    if runs[0]["mask_joints"]:
        print(f"  adapted joints: {', '.join(runs[0]['mask_joints'])}")
    print("=" * 78)

    print(f"\n{'run':38s} {'reach':>6s} {'mask°':>7s} {'leg°':>7s} {'all°':>7s} "
          f"{'|roll|':>7s} {'drift':>7s} {'loop':>6s}")
    for r in sorted(runs, key=lambda x: (x["gain"], x["name"])):
        m = r["frame"] <= window
        r["w_masked"] = float(np.nanmean(r["err_masked"][m]))
        r["w_leg"] = float(np.nanmean(r["err_leg"][m]))
        r["w_all"] = float(np.nanmean(r["err_all"][m]))
        print(f"{r['name'][-38:]:38s} {int(r['frame'].max()):6d} "
              f"{r['w_masked']:7.2f} {r['w_leg']:7.2f} {r['w_all']:7.2f} "
              f"{r['max_roll']:7.3f} {np.nanmax(r['drift']):7.3f} "
              f"{np.nanmedian(r['loop_ms']):6.1f}")

    arms: dict[str, list] = {}
    for r in runs:
        arms.setdefault(r["arm"], []).append(r)

    print(f"\n{'arm':30s} {'n':>3s} {'reach':>7s} {'mask deg':>16s} {'leg deg':>16s}")
    for arm, rs in sorted(arms.items(), key=lambda kv: kv[1][0]["gain"]):
        reach = statistics.median([int(x["frame"].max()) for x in rs])
        mask_v = [x["w_masked"] for x in rs]
        leg_v = [x["w_leg"] for x in rs]
        print(f"{arm:30s} {len(rs):3d} {reach:7.0f} "
              f"{statistics.median(mask_v):10.2f} +-{np.std(mask_v):4.2f} "
              f"{statistics.median(leg_v):10.2f} +-{np.std(leg_v):4.2f}")

    frozen = arms.get("frozen", [])
    adapted = [rs for arm, rs in arms.items() if arm != "frozen"]
    if not frozen or not adapted:
        print("\n[verdict] need both a frozen arm (--gain 0) and an adapted arm.")
        return

    print()
    for rs in adapted:
        arm = rs[0]["arm"]
        n = min(len(frozen), len(rs))
        f_mask = statistics.median([x["w_masked"] for x in frozen])
        a_mask = statistics.median([x["w_masked"] for x in rs])
        f_leg = statistics.median([x["w_leg"] for x in frozen])
        a_leg = statistics.median([x["w_leg"] for x in rs])
        f_reach = statistics.median([int(x["frame"].max()) for x in frozen])
        a_reach = statistics.median([int(x["frame"].max()) for x in rs])
        print(f"[{arm}] vs frozen, medians:")
        print(f"    adapted-joint error {f_mask:6.2f} -> {a_mask:6.2f} deg "
              f"({(a_mask - f_mask) / max(1e-9, f_mask) * 100:+.0f}%)")
        print(f"    leg error           {f_leg:6.2f} -> {a_leg:6.2f} deg "
              f"({(a_leg - f_leg) / max(1e-9, f_leg) * 100:+.0f}%)")
        print(f"    reached frame       {f_reach:6d} -> {a_reach:6d}")
        if n < 3:
            print(f"    [verdict] NO CONCLUSION: {len(frozen)} frozen vs {len(rs)} "
                  "adapted run(s). Need >=3 of each, alternated, before this means "
                  "anything.")
        elif a_reach < f_reach:
            print("    [verdict] adaptation got LESS FAR through the motion. That is a "
                  "loss regardless of the error numbers.")
        elif a_mask < f_mask and a_reach >= f_reach:
            print("    [verdict] adaptation reduced the error it targets without losing "
                  "motion completion. This is the outcome you are looking for.")
        else:
            print("    [verdict] no improvement on the targeted error.")


if __name__ == "__main__":
    main()
