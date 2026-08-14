#!/usr/bin/env python3
"""Does the policy push off the ground with its hands to stand back up?

The hypothesis is that Isaac lets the robot lever itself up through the hands and
the real robot cannot, because its hands are soft. Before planning around that,
check it: run forward kinematics on the recorded Isaac rollout and on the hardware
logs, and report how low the wrists actually get, and what torque the arms carry
while the legs are extending.

If the hands are on the floor during the rise AND the arm joints are loaded, the
robot is pushing. If they are up holding the box, the support story is wrong and
the stand-up failure is somewhere else.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOLO = os.environ.get("HOLOSOMA_ROOT", "/home/baaqer/baaqer_ws/holosoma")
XML = os.path.join(HOLO, "src/holosoma_retargeting/holosoma_retargeting"
                         "/models/x2/x2_31dof_w_largebox.xml")
WRISTS = ("left_wrist_roll_link", "right_wrist_roll_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
RISE = (120, 165)


def fk_heights(model, data, dof_names, mj_order, root_pos, root_quat_xyzw, dof_pos):
    """Return per-frame world height of the wrist and foot bodies."""
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in WRISTS + FEET}
    out = {n: [] for n in bid}
    for t in range(dof_pos.shape[0]):
        data.qpos[:3] = root_pos[t]
        x, y, z, w = root_quat_xyzw[t]
        data.qpos[3:7] = [w, x, y, z]
        for j, name in enumerate(dof_names):
            k = mj_order.get(name)
            if k is not None:
                data.qpos[7 + k] = dof_pos[t, j]
        mujoco.mj_forward(model, data)
        for n, i in bid.items():
            out[n].append(float(data.xpos[i][2]))
    return {n: np.asarray(v) for n, v in out.items()}


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mj_order = {}
    for k in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, k)
        if nm and model.jnt_type[k] == mujoco.mjtJoint.mjJNT_HINGE:
            mj_order[nm] = model.jnt_qposadr[k] - 7

    d = np.load(os.path.join(REPO, "adaptation/isaac_runs/v33/isaac_frozen_npz_seed600.npz"),
                allow_pickle=True)
    meta = json.loads(str(d["_metadata_json"]))
    jn = meta["dof_names"]
    h = fk_heights(model, data, jn, mj_order, d["root_pos"], d["root_quat_xyzw"],
                   d["dof_pos"].astype(np.float64))
    box_z = d["object_pos"][:, 2]
    frame = d["frame"]

    print("ISAAC -- body height above the floor (m)")
    print(f"  {'frame':>6s}{'L wrist':>10s}{'R wrist':>10s}{'L foot':>9s}"
          f"{'R foot':>9s}{'box z':>8s}  phase")
    for f in [0, 40, 80, 100, 120, 130, 140, 150, 160, 200, 420, 480, 560]:
        k = int(np.argmin(np.abs(frame - f)))
        ph = "RISE" if RISE[0] <= f <= RISE[1] else ""
        print(f"  {f:6d}{h[WRISTS[0]][k]:10.3f}{h[WRISTS[1]][k]:10.3f}"
              f"{h[FEET[0]][k]:9.3f}{h[FEET[1]][k]:9.3f}{box_z[k]:8.3f}  {ph}")

    lo = min(h[WRISTS[0]][RISE[0]:RISE[1]].min(), h[WRISTS[1]][RISE[0]:RISE[1]].min())
    foot = float(np.median(np.concatenate([h[FEET[0]], h[FEET[1]]])))
    print(f"\n  lowest wrist during the rise: {lo:.3f} m   (foot centre sits at "
          f"{foot:.3f} m, so the floor is ~{foot - 0.068:.3f} m below it)")
    print(f"  wrist clearance above the floor at its lowest: "
          f"{lo - (foot - 0.068):.3f} m")

    # what the arms carry while the legs extend, sim vs hardware
    print("\nARM LOADING DURING THE RISE (frames 120-165)")
    kp = np.array(json.loads(str(np.load(
        os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"),
        allow_pickle=True)["meta_json"]))["joint_stiffness"])
    arms = [i for i, n in enumerate(jn)
            if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    legs = [i for i, n in enumerate(jn) if any(k in n for k in ("hip", "knee", "ankle"))]
    sq = d["dof_pos"].astype(np.float64)
    sa = d["actions"].astype(np.float64)
    asc = np.array(json.loads(str(np.load(
        os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"),
        allow_pickle=True)["meta_json"]))["action_scale"])
    dflt = np.array(json.loads(str(np.load(
        os.path.join(REPO, "box_pickup/policy/x2_box_policy_v33_iter253000.npz"),
        allow_pickle=True)["meta_json"]))["default_joint_pos"])
    eff = 4.0 * asc * kp
    m = (frame >= RISE[0]) & (frame <= RISE[1])
    tau_sim = np.clip(kp * ((sa * asc + dflt) - sq), -eff, eff)
    print(f"  Isaac    arms |tau| {np.abs(tau_sim[m][:, arms]).mean():6.2f} Nm    "
          f"legs |tau| {np.abs(tau_sim[m][:, legs]).mean():6.2f} Nm")

    for nm in ("20260812_132139", "20260812_132219", "20260812_132056"):
        p = os.path.join(HERE, nm + "_box_pickup_x2_box_policy_v33_iter253000.csv")
        rows = [r for r in csv.DictReader(open(p)) if r["phase"] == "policy"]
        fr = np.array([int(r["frame"]) for r in rows])
        sel = (fr >= RISE[0]) & (fr <= RISE[1])
        if sel.sum() < 3:
            continue
        e = np.array([[float(r[f"{n}__eff_meas"]) for n in jn] for r in rows])[sel]
        print(f"  {nm[9:]}   arms |tau| {np.abs(e[:, arms]).mean():6.2f} Nm    "
              f"legs |tau| {np.abs(e[:, legs]).mean():6.2f} Nm   (measured)")


if __name__ == "__main__":
    main()
