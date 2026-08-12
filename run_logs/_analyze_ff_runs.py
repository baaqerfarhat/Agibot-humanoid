#!/usr/bin/env python3
"""Analyse the first runs recorded with the torque feed-forward enabled.

Answers, in order:

1. Did the firmware honour the `effort` field? The log now carries the commanded
   feed-forward (`__eff_cmd`) and the measured actuator torque (`__eff_meas`), so
   the total commanded torque can be compared against what the joint reported.
2. Did the previously-starved joints (ankle roll, wrist pitch, shoulder roll)
   actually receive their saturated torque this time?
3. Where does the robot stop following the reference, and which joints lead the
   divergence?
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _analyze_clip import LIMITS  # noqa: E402
from _replay_deploy import HERE, REPO, Policy  # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    meta = json.load(open(path.replace(".csv", ".meta.json")))
    return rows, meta


def cols(rows, jn, suffix):
    return np.array([[float(r[f"{n}__{suffix}"]) for n in jn] for r in rows], np.float64)


def analyse(path, policy, verbose=True):
    jn = policy.meta["joint_names"]
    kp = np.array(policy.meta["joint_stiffness"], np.float64)
    kd = np.array(policy.meta["joint_damping"], np.float64)
    ascale = np.array(policy.meta["action_scale"], np.float64)
    eff_lim = 4.0 * ascale * kp
    lo = np.array([LIMITS[n][0] for n in jn])
    hi = np.array([LIMITS[n][1] for n in jn])

    rows, meta = load(path)
    pol = [r for r in rows if r["phase"] == "policy"]
    if len(pol) < 10:
        return None
    frame = np.array([int(r["frame"]) for r in pol])
    q = cols(pol, jn, "pos_meas")
    dq = cols(pol, jn, "vel_meas")
    tgt = cols(pol, jn, "tgt")
    e_meas = cols(pol, jn, "eff_meas")
    e_cmd = cols(pol, jn, "eff_cmd")
    roll = np.array([float(r["pelvis_roll"]) for r in pol])

    pos_sent = np.clip(tgt, lo, hi)
    tau_cmd = e_cmd + kp * (pos_sent - q) - kd * dq          # what the low level is asked for
    tau_train = np.clip(kp * (tgt - q) - kd * dq, -eff_lim, eff_lim)
    tau_oldpath = np.clip(kp * (pos_sent - q) - kd * dq, -eff_lim, eff_lim)

    out = dict(name=os.path.basename(path)[9:15], frame=frame, roll=roll, q=q, dq=dq,
               tgt=tgt, e_meas=e_meas, e_cmd=e_cmd, tau_cmd=tau_cmd,
               tau_train=tau_train, tau_oldpath=tau_oldpath, jn=jn, meta=meta,
               eff_lim=eff_lim)
    if not verbose:
        return out

    print("=" * 104)
    print(f"{os.path.basename(path)}   ff={meta.get('torque_ff')} gain={meta.get('gain_scale')} "
          f"frames 0-{frame.max()} ({frame.max()/50:.2f}s)")
    print("=" * 104)

    # ---- 1. did the effort field do anything? ----
    print("  [1] commanded feed-forward vs measured actuator torque")
    print(f"      |eff_cmd| : mean {np.abs(e_cmd).mean():6.3f}  max {np.abs(e_cmd).max():7.2f} Nm")
    print(f"      |eff_meas|: mean {np.abs(e_meas).mean():6.3f}  max {np.abs(e_meas).max():7.2f} Nm")
    # correlate the commanded total torque with the reported one, per joint
    print("      joints with the largest feed-forward, and whether the joint reports it:")
    print(f"        {'joint':26s}{'|ff|mean':>10s}{'tau_cmd':>10s}{'tau_meas':>10s}"
          f"{'corr':>7s}{'ratio':>8s}")
    for i in np.argsort(-np.abs(e_cmd).mean(axis=0))[:8]:
        a, b = tau_cmd[:, i], e_meas[:, i]
        c = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else np.nan
        print(f"        {jn[i]:26s}{np.abs(e_cmd[:,i]).mean():10.2f}"
              f"{np.abs(a).mean():10.2f}{np.abs(b).mean():10.2f}{c:7.2f}"
              f"{np.abs(b).mean()/max(1e-6,np.abs(a).mean()):8.2f}")

    # ---- 2. is the torque now the training torque? ----
    gain = 0.0 if np.abs(tau_train - tau_oldpath).mean() < 1e-9 else \
        1.0 - np.abs(tau_cmd - tau_train).mean() / np.abs(tau_oldpath - tau_train).mean()
    print(f"  [2] |tau_cmd - tau_train| mean {np.abs(tau_cmd-tau_train).mean():.4f} Nm  "
          f"(old path would be {np.abs(tau_oldpath-tau_train).mean():.3f})  "
          f"-> {100*gain:.1f}% of the gap closed")

    # ---- 3. where does tracking break down? ----
    ref = policy.ref_joint_pos[np.minimum(frame, policy.ref_joint_pos.shape[0] - 1)]
    terr = q - ref
    print("  [3] reference tracking (measured - reference), policy phase")
    win = 25
    print(f"        {'frame':>6s}{'roll':>8s}{'|terr|mean':>12s}{'worst joint':>30s}{'err':>8s}")
    for a in range(0, frame.max() + 1, max(win, (frame.max() + 1) // 12)):
        m = (frame >= a) & (frame < a + win)
        if m.sum() < 3:
            continue
        e = np.abs(terr[m]).mean(axis=0)
        i = int(np.argmax(e))
        print(f"        {a:6d}{roll[m].mean():8.2f}{np.abs(terr[m]).mean():12.3f}"
              f"{jn[i]:>30s}{terr[m][:,i].mean():8.2f}")
    print("      worst-tracking joints over the whole run:")
    for i in np.argsort(-np.abs(terr).mean(axis=0))[:8]:
        k = int(np.argmax(np.abs(terr[:, i])))
        sat = 100.0 * (np.abs(tau_cmd[:, i]) > eff_lim[i] - 1e-3).mean()
        print(f"        {jn[i]:26s} mean |err| {np.abs(terr[:,i]).mean():5.2f}  "
              f"max {terr[k,i]:+6.2f} @frame {frame[k]:3d}  "
              f"tau at limit {sat:5.1f}% of ticks")
    print()
    return out


def main():
    policy = Policy(os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"))
    files = sys.argv[1:] or ["20260812_132056", "20260812_132139", "20260812_132219"]
    for f in files:
        p = f if f.endswith(".csv") else os.path.join(
            HERE, f + "_box_pickup_x2_box_policy_v33_iter253000.csv")
        analyse(p, policy)


if __name__ == "__main__":
    main()
