"""Side-by-side of the three robot interfaces, read out of the live code.

MJLab walking/squat is the canonical implementation, holosoma is where the box task
currently lives, and the deploy script is what the robot actually runs. Anything that
disagrees between the first and the third is a sim-to-real bug waiting to happen;
anything the second does differently is a candidate for being dropped rather than
ported. Written out rather than eyeballed because most of these numbers are per-joint
and the interesting mismatches are in a handful of entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POLICY = Path(
    "/home/baaqer/baaqer_ws/Agibot-humanoid/box_pickup/policy"
    "/x2_box_policy_walk_retimed_v19_iter85500.npz"
)
SQUAT = Path(
    "/home/baaqer/baaqer_ws/Agibot-humanoid/agibot_control_functions/policies"
    "/x2_squat_policy_40pct_iter16499.npz"
)


def meta_of(p: Path) -> dict:
    return json.loads(str(np.load(p, allow_pickle=True)["meta_json"]))


def main() -> None:
    from mjlab.asset_zoo.robots.x2.x2_constants import (
        X2_ACTION_SCALE,
        X2_ARTICULATION,
        get_x2_robot_cfg,
    )
    from mjlab.entity.entity import Entity

    robot = Entity(get_x2_robot_cfg())
    jn = list(robot.joint_names)

    # mjlab: gains/effort live on the actuator configs, keyed by regex.
    import re

    mj_kp, mj_kd, mj_eff, mj_scale = {}, {}, {}, {}
    for a in X2_ARTICULATION.actuators:
        for pat in a.target_names_expr:
            for j in jn:
                if re.fullmatch(pat, j):
                    mj_kp[j], mj_kd[j] = a.stiffness, a.damping
                    mj_eff[j] = a.effort_limit
                    mj_scale[j] = X2_ACTION_SCALE[pat]

    box = meta_of(POLICY)
    hs_jn = list(box["joint_names"])
    hs = {
        k: dict(zip(hs_jn, box[k]))
        for k in ("joint_stiffness", "joint_damping", "joint_effort_limit",
                  "action_scale", "default_joint_pos")
    }
    sq = meta_of(SQUAT) if SQUAT.exists() else None

    print("=" * 100)
    print("PER-JOINT INTERFACE: mjlab (canonical) vs holosoma box (to be replaced)")
    print("=" * 100)
    print(f"{'joint':<28} {'kp mj/hs':>14} {'kd mj/hs':>12} {'effort mj/hs':>15} "
          f"{'scale mj/hs':>16}")
    print("-" * 100)
    diffs = {"kp": 0, "kd": 0, "eff": 0, "scale": 0, "default": 0}
    for j in jn:
        kp1, kp2 = mj_kp[j], hs["joint_stiffness"][j]
        kd1, kd2 = mj_kd[j], hs["joint_damping"][j]
        e1, e2 = mj_eff[j], hs["joint_effort_limit"][j]
        s1, s2 = mj_scale[j], hs["action_scale"][j]
        for key, a, b in (("kp", kp1, kp2), ("kd", kd1, kd2), ("eff", e1, e2),
                          ("scale", s1, s2)):
            if abs(a - b) > 1e-6:
                diffs[key] += 1
        mark = "  <<" if abs(s1 - s2) > 1e-6 or abs(kp1 - kp2) > 1e-6 else ""
        print(f"{j.replace('_joint',''):<28} {kp1:6.0f}/{kp2:<7.0f} "
              f"{kd1:5.1f}/{kd2:<6.1f} {e1:7.1f}/{e2:<7.1f} "
              f"{s1:7.4f}/{s2:<8.4f}{mark}")

    init = get_x2_robot_cfg().init_state.joint_pos or {}
    dmj = np.zeros(len(jn))
    for i, j in enumerate(jn):
        for pat, v in init.items():
            if re.fullmatch(pat, j):
                dmj[i] = v
    dhs = np.array([hs["default_joint_pos"][j] for j in jn])
    diffs["default"] = int((np.abs(dmj - dhs) > 1e-6).sum())
    for i, j in enumerate(jn):
        if abs(dmj[i] - dhs[i]) > 1e-6:
            print(f"  default differs: {j:<28} mjlab {dmj[i]:+.3f}  holosoma {dhs[i]:+.3f}")

    print("-" * 100)
    print(f"joints differing:  kp {diffs['kp']}/31   kd {diffs['kd']}/31   "
          f"effort {diffs['eff']}/31   action_scale {diffs['scale']}/31   "
          f"default_pos {diffs['default']}/31")
    print()
    print(f"joint ORDER identical: {jn == hs_jn}")
    print()

    print("=" * 100)
    print("OBSERVATIONS")
    print("=" * 100)
    print(f"holosoma box actor : {box['observation_names']}  -> {box['obs_dim']} dims")
    if sq:
        print(f"mjlab squat actor  : {sq['observation_names']}  -> {sq['obs_dim']} dims")
        print(f"mjlab squat scale == mjlab constants: "
              f"{np.allclose([mj_scale[j] for j in jn], sq['action_scale'])}")
        print(f"mjlab squat kp     == mjlab constants: "
              f"{np.allclose([mj_kp[j] for j in jn], sq['joint_stiffness'])}")
    print()
    print("=" * 100)
    print("CONTROL")
    print("=" * 100)
    print(f"{'':<22}{'mjlab squat/walk':<22}{'holosoma box':<22}{'deployment'}")
    rows = [
        ("sim timestep", "0.005 s", "0.005 s", "-"),
        ("decimation", "4", "4", "-"),
        ("control rate", "50 Hz", "50 Hz", "50 Hz"),
        ("action def", "q_des = default + s*a", "q_des = default + s*a",
         "q_des = default + s*a"),
        ("torque", "MuJoCo position servo", "explicit PD, clipped",
         "onboard PD, kp/kd sent"),
        ("integrator", "implicitfast", "PhysX", "-"),
        ("self-collision", "off (feet only)", "ON, full meshes", "real"),
        ("quat convention", "wxyz", "xyzw", "xyzw"),
        ("base frame", "pelvis", "pelvis", "pelvis (reconstructed)"),
    ]
    for r in rows:
        print(f"{r[0]:<22}{r[1]:<22}{r[2]:<22}{r[3]}")


if __name__ == "__main__":
    main()
