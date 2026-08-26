#!/usr/bin/env python3
"""Run the deploy-time pelvis reconstruction against sim, where the answer is known.

On hardware there is no pelvis gyro, so the reconstruction cannot be checked. In
the sim rollout both ends exist: root_ang_vel IS the pelvis rate the policy was
trained on, and the torso link's attitude stream gives the torso rate the robot's
IMU would have measured. Feeding the latter through base_frame.PelvisEstimator
should return the former.

If it does, the reconstruction is sound and the hardware discrepancy is the robot
genuinely moving. If it does not, the observation fed to the policy on hardware
was never the quantity it was trained on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agibot_control_functions"))
from base_frame import pelvis_from_torso_rot, waist_ang_vel_in_pelvis  # noqa: E402

SIM = Path("/tmp/x2_box_walk_feasible_v16_iter30000_rollout.npz")
FS = 50.0


def main():
    d = np.load(SIM, allow_pickle=True)
    meta = json.loads(str(d["_metadata_json"]))
    bn, dn = meta["body_names"], meta["dof_names"]
    ti = bn.index("torso_link")

    # truth: the pelvis rate the policy actually observed in training, body frame
    Rp = Rot.from_quat(d["root_quat_xyzw"])
    w_pelvis_true = Rp.inv().apply(d["root_ang_vel"])

    # what the robot's torso IMU would have read, in the torso body frame
    Rt = Rot.from_quat(d["body_quat_xyzw"][:, ti])
    w_torso = (Rt[:-1].inv() * Rt[1:]).as_rotvec() * FS
    w_torso = np.r_[w_torso[:1], w_torso]

    wy, wp, wr = (dn.index(f"waist_{a}_joint") for a in ("yaw", "pitch", "roll"))
    q, dq = d["dof_pos"], d["dof_vel"]

    est_sub, est_add = [], []
    for k in range(len(q)):
        yaw, pitch, roll = q[k, wy], q[k, wp], q[k, wr]
        yr, pr, rr = dq[k, wy], dq[k, wp], dq[k, wr]
        R_pt = pelvis_from_torso_rot(yaw, pitch, roll)
        corr = waist_ang_vel_in_pelvis(yaw, pitch, yr, pr, rr)
        est_sub.append(R_pt @ w_torso[k] - corr)   # what deploy does
        est_add.append(R_pt @ w_torso[k] + corr)   # the opposite sign, as a control
    est_sub, est_add = np.array(est_sub), np.array(est_add)

    def rep(tag, v):
        n = np.linalg.norm(v, axis=1)
        e = np.linalg.norm(v - w_pelvis_true, axis=1)
        cs = [np.corrcoef(v[:, i], w_pelvis_true[:, i])[0, 1] for i in range(3)]
        print(f"  {tag:34s} |w| mean {n.mean():.3f}  err vs truth {e.mean():.3f} rad/s"
              f"  corr xyz {cs[0]:+.2f}/{cs[1]:+.2f}/{cs[2]:+.2f}")

    print("Validating the deploy pelvis reconstruction inside sim\n" + "=" * 74)
    nt = np.linalg.norm(w_pelvis_true, axis=1)
    print(f"  {'TRUE pelvis rate (training obs)':34s} |w| mean {nt.mean():.3f}")
    print(f"  {'torso rate (what the IMU sees)':34s} |w| mean "
          f"{np.linalg.norm(w_torso, axis=1).mean():.3f}")
    print()
    rep("reconstruction, deploy's sign", est_sub)
    rep("reconstruction, opposite sign", est_add)
    rep("no correction (raw torso)", w_torso)
    print()
    ns, nt2 = np.linalg.norm(est_sub, axis=1), np.linalg.norm(w_torso, axis=1)
    print(f"  in SIM the reconstruction is larger than the torso on "
          f"{100*(ns>nt2).mean():.0f}% of ticks (ratio {ns.mean()/nt2.mean():.2f})")
    print(f"  on HARDWARE the same code was larger on 91% of ticks (ratio 1.52)")


if __name__ == "__main__":
    main()
