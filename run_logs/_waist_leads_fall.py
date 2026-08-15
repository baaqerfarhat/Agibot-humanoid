#!/usr/bin/env python3
"""Does the waist droop cause the fall, or just record it?

Across the 2026-08-14 runs the association is hard to miss: every run that stayed
up held mean |waist_pitch - target| at or under 0.136 rad, and every run above
0.17 went down. That is not enough to act on. If the waist error only grows once
the robot is already toppling, it is a symptom, and a waist adaptation term would
be chasing it. If it grows first, while the torso is still upright, then the waist
is losing the pose and the fall follows -- and an adaptive term that closes a
standing offset is aimed at the right joint.

So this asks about order in time. Everything is compared at matched motion frames
against the runs that survived, which is what makes the comparison mean anything:
the motion bends the torso 80 deg forward on purpose, and a run that fails early
spends most of its short life in that bend, so any average taken over wall-clock
or over a whole run is really just measuring how long the run lasted. Frame
alignment removes that -- both the survivors and the failures are doing the same
thing at frame 140.

Usage: ~/baaqer_ws/mjlab/.venv/bin/python run_logs/_waist_leads_fall.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def tilt_of(r):
    x, y, z, w = (float(r[f"base_quat_{k}"]) for k in "xyzw")
    return np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))


def load(p):
    rows = [r for r in csv.DictReader(open(p)) if r["phase"] == "policy"]
    if not rows:
        return None
    fr = np.array([int(r["frame"]) for r in rows])
    t = np.array([float(r["t_s"]) for r in rows])
    tilt = np.array([tilt_of(r) for r in rows])
    werr = np.array([float(r["waist_pitch_joint__pos_meas"])
                     - float(r["waist_pitch_joint__tgt"]) for r in rows])
    return fr, t, tilt, werr


def main() -> None:
    runs = {}
    for p in sorted(glob.glob(os.path.join(HERE, "20260814_*box_pickup*.csv"))):
        d = load(p)
        if d is not None:
            runs[os.path.basename(p)[9:15]] = d

    survivors = [k for k, (fr, _, _, _) in runs.items() if fr.max() >= 584]
    print(f"survivors: {', '.join(survivors)}\n")

    grid = np.arange(0, 584)

    def envelope(col, pad):
        band = [np.interp(grid, runs[k][0], np.abs(runs[k][col]))
                for k in survivors]
        return np.max(band, axis=0) + pad

    tilt_band = envelope(2, 0.15)
    werr_band = envelope(3, 0.05)

    def first_out(fr, sig, band):
        over = np.flatnonzero(np.abs(sig) > np.interp(fr, grid, band))
        return int(fr[over[0]]) if len(over) else None

    print(f"{'run':>7s}{'outcome':>9s}{'tilt out':>10s}{'waist out':>11s}"
          f"{'which first':>13s}")
    for k, (fr, t, tilt, werr) in runs.items():
        out = "up" if fr.max() >= 584 else "fell"
        ft = first_out(fr, tilt, tilt_band)
        fw = first_out(fr, werr, werr_band)
        if ft is None and fw is None:
            verdict = "neither"
        elif fw is None:
            verdict = "tilt only"
        elif ft is None:
            verdict = "waist only"
        else:
            verdict = ("waist, %+d frames" % (ft - fw) if fw < ft else
                       "tilt, %+d frames" % (fw - ft) if ft < fw else "same frame")
        print(f"{k:>7s}{out:>9s}{str(ft or '-'):>10s}{str(fw or '-'):>11s}"
              f"{verdict:>13s}")

    print("\nEach column is the first motion frame at which that signal left the")
    print("band the surviving runs stayed inside at the same frame. 'which first'")
    print("names the one that broke earlier, and by how many frames (50 per second).")

    # The frame-matched version of the association, over the frames where every
    # run is still alive, so run length cannot drive it.
    lo, hi = 100, 160
    print(f"\nmean |waist_pitch - target| over frames {lo}-{hi}, "
          f"where every run is still running:")
    for k, (fr, t, tilt, werr) in sorted(
            runs.items(), key=lambda kv: kv[0] not in survivors):
        m = (fr >= lo) & (fr <= hi)
        if m.sum() < 5:
            continue
        print(f"    {k}  {'up  ' if k in survivors else 'fell'}"
              f"  {np.abs(werr[m]).mean():.3f} rad")


if __name__ == "__main__":
    main()
