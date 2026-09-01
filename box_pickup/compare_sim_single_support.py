"""Does the policy follow the reference into single support in simulation?

That is the question WHAT_WE_NEED_FROM_YOU.md wants the v18 checkpoint in order to
answer. The checkpoint is being handed over regardless, but the sim rollouts behind
the duty-cycle numbers in 2cd5b81 are already on disk, and they answer it without
re-running anything.

The hypothesis under test: on hardware the runs that survive are the ones that refuse
to follow the reference into single support, measured as foot height asymmetry over
frames 179-250 (17.5 mm in the run that completed, 29.7 and 43.3 mm in two that fell).

So the same numbers are computed here on the sim rollouts, in the same window and with
the same definition, next to the reference the policy was tracking. If sim also refuses,
sim and hardware are doing the same thing and the gap is disturbance rejection. If sim
follows the reference in and survives, sim is making restoring moment the real foot
cannot, and the contact model is the gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_reference_motion import FIXED_FRAMES, URDF, Robot, load_masses
from urdf_fk import quat_wxyz_to_mat

CLIP = ("/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
        "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz")
RUNS = {
    "v18 i81499": "/tmp/x2_box_ankle_scale_v18_iter81499_rollout.npz",
    "v17 i49000": "/tmp/x2_box_walk_feasible_v17_iter49000_rollout.npz",
}
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
CONTACT = 0.020            # sole below this is carrying load
WINDOW = (179, 250)        # the window the hardware asymmetry was measured over
ROLL_LIMIT = 24.0
G = 9.81
BOX_M = 1.0


def sole_offsets():
    return {b: [np.asarray(FIXED_FRAMES[k][1]) for k in FIXED_FRAMES
                if "sphere" in k and FIXED_FRAMES[k][0] == b] for b in FEET}


def support_of(h):
    """h is [N, 2] sole heights -> counts of double / single / airborne."""
    down = h < CONTACT
    k = down.sum(axis=1)
    return (k == 2).sum(), (k == 1).sum(), (k == 0).sum()


def main():
    off = sole_offsets()
    mass = load_masses(URDF)
    robot = Robot()
    total = sum(m for m, _ in mass.values())

    # ---- reference -----------------------------------------------------------
    d = np.load(CLIP, allow_pickle=True)
    bn = [str(x) for x in d["body_names"]]
    bp, bq = np.asarray(d["body_pos_w"]), np.asarray(d["body_quat_w"])
    n = len(bp)
    ref_h = np.zeros((n, 2))
    for i, b in enumerate(FEET):
        j = bn.index(b)
        for f in range(n):
            R = quat_wxyz_to_mat(bq[f, j])
            ref_h[f, i] = min((bp[f, j] + R @ s)[2] for s in off[b])

    print(f"{'':14s} {'double':>8s} {'single':>8s} {'air':>6s}   "
          f"{'foot asym in 179-250':>22s}   {'max sole lift':>14s}")
    dd, ss, aa = support_of(ref_h)
    a, b_ = WINDOW
    print(f"{'REFERENCE':14s} {dd:8d} {ss:8d} {aa:6d}   "
          f"{np.abs(ref_h[a:b_, 0] - ref_h[a:b_, 1]).mean()*1000:19.1f} mm   "
          f"{ref_h.max()*1000:11.1f} mm")

    out = {}
    for tag, path in RUNS.items():
        r = np.load(path, allow_pickle=True)
        m = json.loads(str(r["_metadata_json"]))
        rbn, jn = list(m["body_names"]), list(m["dof_names"])
        eff = np.asarray(m["effort_limits"], float)
        rbp = np.asarray(r["body_pos_w"])
        rbq = np.asarray(r["body_quat_xyzw"])
        N = len(rbp)
        h = np.zeros((N, 2))
        ay = np.zeros((N, 2))
        for i, b in enumerate(FEET):
            j = rbn.index(b)
            for f in range(N):
                q = rbq[f, j]
                R = quat_wxyz_to_mat(np.r_[q[3], q[:3]])
                h[f, i] = min((rbp[f, j] + R @ s)[2] for s in off[b])
                ay[f, i] = rbp[f, j][1]
        dd, ss, aa = support_of(h)
        print(f"{tag:14s} {dd:8d} {ss:8d} {aa:6d}   "
              f"{np.abs(h[a:b_, 0] - h[a:b_, 1]).mean()*1000:19.1f} mm   "
              f"{h.max()*1000:11.1f} mm")
        out[tag] = dict(h=h, ay=ay, bp=rbp, bn=rbn, jn=jn, eff=eff,
                        tau=np.asarray(r["torques_substep"]).reshape(-1, len(jn)),
                        obj=np.asarray(r["object_pos"]), N=N)

    print("\n(hardware, frames 179-250: 17.5 mm in the run that completed,"
          " 29.7 and 43.3 mm in two that fell)")

    # ---- does sim enter single support WHERE the reference does? --------------
    ref_single = (ref_h < CONTACT).sum(axis=1) == 1
    print("\nagreement with the reference's own support schedule:")
    for tag, s in out.items():
        k = (s["h"] < CONTACT).sum(axis=1)
        sim_single = k == 1
        both = min(len(ref_single), len(sim_single))
        rs, ss_ = ref_single[:both], sim_single[:both]
        follow = (rs & ss_).sum()
        refuse = (rs & ~ss_).sum()
        print(f"  {tag}: reference is in single support on {rs.sum()} frames;"
              f" sim follows it there on {follow} ({100*follow/max(rs.sum(),1):.0f}%),"
              f" keeps both feet down on {refuse} ({100*refuse/max(rs.sum(),1):.0f}%)")

    # ---- lateral CoM offset from the stance foot, in sim -----------------------
    print("\nlateral CoM offset from the stance ankle while in single support")
    print(f"(the {ROLL_LIMIT:.0f} N-m bound is {ROLL_LIMIT/((total+BOX_M)*G)*1000:.0f} mm;"
          " the reference asks up to 300 mm)")
    for tag, s in out.items():
        idx = [s["bn"].index(nm) for nm in s["bn"]]
        # mass-weighted CoM from the logged link poses
        com = np.zeros((s["N"], 3))
        tot = 0.0
        for nm, (mk, c) in mass.items():
            if nm not in s["bn"]:
                continue
            j = s["bn"].index(nm)
            for f in range(s["N"]):
                q = s["bq"][f, j] if "bq" in s else None
            tot += mk
        # vectorised: rotate each link's own CoM offset into the world
        com = np.zeros((s["N"], 3))
        tot = 0.0
        rbq = np.asarray(np.load(RUNS[tag], allow_pickle=True)["body_quat_xyzw"])
        for nm, (mk, c) in mass.items():
            if nm not in s["bn"]:
                continue
            j = s["bn"].index(nm)
            c = np.asarray(c)
            for f in range(s["N"]):
                q = rbq[f, j]
                com[f] += mk * (s["bp"][f, j] + quat_wxyz_to_mat(np.r_[q[3], q[:3]]) @ c)
            tot += mk
        carried = s["obj"][:, 2] > 0.30
        mtot = np.where(carried, tot + BOX_M, tot)
        com = com + np.where(carried[:, None], BOX_M * s["obj"], 0.0)
        com /= mtot[:, None]

        k = (s["h"] < CONTACT).sum(axis=1)
        sel = np.nonzero(k == 1)[0]
        if not len(sel):
            print(f"  {tag}: never enters single support")
            continue
        stance = np.argmin(s["h"][sel], axis=1)
        d_lat = np.abs(com[sel, 1] - s["ay"][sel, stance])
        moment = mtot[sel] * G * d_lat
        print(f"  {tag}: {len(sel)} single-support frames, offset mean"
              f" {d_lat.mean()*1000:5.1f} mm max {d_lat.max()*1000:5.1f} mm"
              f" -> moment mean {moment.mean():5.1f} max {moment.max():5.1f} N-m,"
              f" over {ROLL_LIMIT:.0f} on {100*(moment>ROLL_LIMIT).mean():.0f}%")

    # ---- ankle duty cycle, restated so the md can quote one source -------------
    print("\nankle duty cycle at the effort limit (substeps >= 99%)")
    for tag, s in out.items():
        row = f"  {tag}: "
        for base in ("ankle_roll", "ankle_pitch"):
            v = [100.0 * (np.abs(s["tau"][:, s["jn"].index(f"{sd}_{base}_joint")])
                          >= 0.99 * s["eff"][s["jn"].index(f"{sd}_{base}_joint")]).mean()
                 for sd in ("left", "right")]
            row += f"{base} {v[0]:.0f}%/{v[1]:.0f}%   "
        print(row)


if __name__ == "__main__":
    main()
