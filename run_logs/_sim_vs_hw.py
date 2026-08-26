#!/usr/bin/env python3
"""Sim rollout vs hardware run, same policy, same metrics, same clip.

The point is to separate two explanations for the chatter that look identical in
a hardware log on its own:

  the policy chatters       -- then sim shows it too, and the fix is in training
  the robot drives it off   -- then sim is smooth, and the fix is in the
                               observation pipeline or the plant

Everything below is computed identically on both sides. Targets are the
POSITION TARGETS the policy asked for (action * scale + default), which is what
the joint sees, not the raw action.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
POLICY = HERE.parent / "box_pickup" / "policy" / "x2_box_policy_walk_feasible_v16_iter30500.npz"
SIM = Path("/tmp/x2_box_walk_feasible_v16_iter30000_rollout.npz")
HW = HERE / "20260825_173138_box_pickup_x2_box_policy_walk_feasible_v16_iter30500.csv"
CLIP = Path("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
            "x2_31dof/whole_body_tracking/sub3_largebox_003_walk_feasible.npz")
LEG = ("hip", "knee", "ankle")


def stats(name, tgt, pos, vel, w, ref, leg):
    dt = np.diff(tgt, axis=0)
    rev = np.sign(dt[1:]) * np.sign(dt[:-1]) < 0
    dv = np.diff(vel, axis=0)
    vrev = np.sign(dv[1:]) * np.sign(dv[:-1]) < 0
    nw = np.linalg.norm(w, axis=1)
    out = {
        "target |d| leg (mrad/tick)": np.abs(dt[:, leg]).mean() * 1000,
        "target |d| leg p99": np.percentile(np.abs(dt[:, leg]), 99) * 1000,
        "target reversals leg (%)": 100 * rev[:, leg].mean(),
        "measured |vel| leg (rad/s)": np.abs(vel[:, leg]).mean(),
        "measured |vel| leg max": np.abs(vel[:, leg]).max(),
        "accel reversals leg (%)": 100 * vrev[:, leg].mean(),
        "base |w| mean (rad/s)": nw.mean(),
        "base |w| max": nw.max(),
        "base w roughness (rad/s/tick)": np.abs(np.diff(w, axis=0)).mean(),
        "|q-ref| leg (mrad)": np.abs(pos[:, leg] - ref[:, leg]).mean() * 1000,
        "|q-ref| leg max": np.abs(pos[:, leg] - ref[:, leg]).max() * 1000,
    }
    return out


def main():
    d = np.load(POLICY, allow_pickle=True)
    pmeta = json.loads(str(d["meta_json"]))
    jn = pmeta["joint_names"]
    ref_q = d["ref_joint_pos"]
    leg = [i for i, n in enumerate(jn) if any(k in n for k in LEG)]

    # ---------------- sim ----------------
    s = np.load(SIM, allow_pickle=True)
    n = len(s["dof_pos"])
    sim_w_world = s["root_ang_vel"]
    Rr = Rot.from_quat(s["root_quat_xyzw"])
    # training's base_ang_vel is body frame; try both and keep the one that is not
    # obviously the world copy
    sim_w = Rr.inv().apply(sim_w_world)
    sim = stats("sim", s["dof_pos_target"], s["dof_pos"], s["dof_vel"], sim_w,
                ref_q[:n], leg)

    # ---------------- hardware ----------------
    lines = HW.read_text().splitlines()
    hdr = lines[0].split(",")
    rows = [ln for ln in lines[1:] if ln.count(",") == len(hdr) - 1]
    raw = np.genfromtxt(rows, delimiter=",", dtype=float)
    col = {c: i for i, c in enumerate(hdr)}
    ph = np.array([r.split(",")[col["phase"]] for r in rows])
    m = ph == "policy"
    blk = lambda f: np.stack([raw[m, col[f"{j}__{f}"]] for j in jn], axis=1)
    fr = raw[m, col["frame"]].astype(int).clip(0, len(ref_q) - 1)
    hw_w = np.stack([raw[m, col[f"obs_ang_vel_{c}"]] for c in "xyz"], axis=1)
    hw = stats("hw", blk("tgt"), blk("pos_meas"), blk("vel_meas"), hw_w, ref_q[fr], leg)

    # ---------------- reference, as a floor ----------------
    c = np.load(CLIP, allow_pickle=True)
    names = [str(x) for x in c["body_names"]]
    bi = names.index("pelvis")
    Rp = Rot.from_quat(np.c_[np.asarray(c["body_quat_w"])[:, bi, 1:],
                             np.asarray(c["body_quat_w"])[:, bi, 0]])
    ref_w = Rp.inv().apply(np.asarray(c["body_ang_vel_w"])[:, bi])
    rf = stats("ref", ref_q, ref_q, np.asarray(c["joint_vel"]), ref_w, ref_q, leg)

    print(f"{'metric':32s} {'reference':>12} {'SIM (30000)':>13} {'HARDWARE':>12}   {'hw/sim':>7}")
    print("-" * 82)
    for k in sim:
        a, b, r = sim[k], hw[k], rf[k]
        ratio = b / a if abs(a) > 1e-9 else float("inf")
        print(f"{k:32s} {r:12.2f} {a:13.2f} {b:12.2f}   {ratio:6.1f}x")
    print(f"\nsim ran {n} steps, hardware {int(m.sum())} of the clip's {len(ref_q)} frames")


if __name__ == "__main__":
    main()
