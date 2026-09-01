"""Can a 24 N-m ankle_roll hold up the reference pose at all?

The ankle_roll actuator is pinned at its limit for 64-70% of a v18 rollout while
the foot is on the ground, and raising the action scale 3x did not change that.
Scale sets how far the joint can be COMMANDED; it says nothing about the force
needed to hold a pose once the foot is loaded. This checks the force.

In single support the whole weight passes through one foot, so the moment the
ankle_roll has to carry is m*g times the lateral distance from the centre of mass
to that foot's roll axis. Turned around, 24 N-m buys a fixed lateral offset and no
more. If the reference parks the CoM further out than that while standing on one
leg, no policy and no action scale can track it: the joint is asked for a torque
that does not exist, the PD term pins, and there is nothing left over to balance
with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_reference_motion import FIXED_FRAMES, URDF, Robot, load_masses
from urdf_fk import quat_wxyz_to_mat

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_walk_feasible.npz"
)
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
ROLL_LIMIT = 24.0   # N-m, ankle_roll effort limit
PITCH_LIMIT = 36.0  # N-m, ankle_pitch
CONTACT = 0.020     # sole below this counts as carrying load
G = 9.81


def main():
    d = np.load(CLIP, allow_pickle=True)
    jn = [str(x) for x in d["joint_names"]]
    q = np.asarray(d["joint_pos"])
    root_pos, root_quat, dof = q[:, 0:3], q[:, 3:7], q[:, 7:]
    box_pos = np.asarray(d["object_pos_w"], float)
    n = len(dof)

    robot = Robot()
    mass = load_masses(URDF)
    spheres = {
        b: [np.asarray(FIXED_FRAMES[k][1]) for k in FIXED_FRAMES
            if "sphere" in k and FIXED_FRAMES[k][0] == b]
        for b in FEET
    }

    total = sum(m for m, _ in mass.values())
    # The box is carried for most of the clip and its weight goes through the ankles
    # too. 1 kg is what the real one weighs.
    box_m = 1.0
    print(f"robot {total:.1f} kg, box {box_m:.1f} kg")
    print(f"ankle_roll limit {ROLL_LIMIT:.0f} N-m -> in single support that is a lateral")
    print(f"CoM offset of {ROLL_LIMIT/((total+box_m)*G)*1000:.0f} mm and no more\n")

    rows = []
    for f in range(n):
        poses = robot.chain.fk(
            {nm: dof[f, jn.index(nm)] for nm in jn}, root_pos[f], root_quat[f]
        )
        acc = np.zeros(3)
        for name, (mk, c) in mass.items():
            if name in poses:
                p, Rm = poses[name]
                acc += mk * (p + Rm @ np.asarray(c))
        m_tot = total
        com = acc.copy()
        if box_pos[f, 2] > 0.30:  # box is up, so it is being carried
            com = com + box_m * box_pos[f]
            m_tot = total + box_m
        com = com / m_tot

        down, ankle_y = [], {}
        for b in FEET:
            p, Rm = poses[b]
            h = min((p + Rm @ s)[2] for s in spheres[b])
            ankle_y[b] = p[1]
            if h < CONTACT:
                down.append(b)
        rows.append((f, com[1], down, ankle_y, m_tot))

    single = [r for r in rows if len(r[2]) == 1]
    double = [r for r in rows if len(r[2]) == 2]
    air = [r for r in rows if not r[2]]
    print(f"{len(double)} double-support frames, {len(single)} single, {len(air)} airborne\n")

    def report(tag, sel, use_stance_foot):
        if not sel:
            print(f"{tag}: none")
            return
        tq = []
        for f, comy, down, ay, m_tot in sel:
            if use_stance_foot:
                off = abs(comy - ay[down[0]])
            else:
                off = abs(comy - 0.5 * (ay[FEET[0]] + ay[FEET[1]]))
            tq.append(m_tot * G * off)
        tq = np.array(tq)
        over = (tq > ROLL_LIMIT).sum()
        print(f"{tag}: {len(sel)} frames")
        print(f"   ankle_roll moment needed: mean {tq.mean():5.1f}, median {np.median(tq):5.1f},"
              f" max {tq.max():5.1f} N-m")
        print(f"   over the {ROLL_LIMIT:.0f} N-m limit on {over}/{len(sel)} frames"
              f" ({100*over/len(sel):.0f}%), worst {tq.max()/ROLL_LIMIT:.1f}x")

    report("SINGLE SUPPORT (whole moment on the stance ankle)", single, True)
    print()
    report("DOUBLE SUPPORT (shared, so this is the optimistic half each)", double, False)

    # The moment above is what would be needed to HOLD the pose. The ankle cannot
    # actually apply it however strong it is: the ground pushes up through the sole,
    # so the moment arm can never exceed half the foot's width before the foot tips.
    half = 0.05  # sole spheres span y = -0.05 .. +0.05
    tip = (total + box_m) * G * half
    print(f"\nWhat the foot can physically transmit, whatever the actuator: the ground")
    print(f"reaction acts inside the sole, so the arm cannot exceed its half width")
    print(f"({half*1000:.0f} mm) -> {tip:.1f} N-m. The actuator limit is {ROLL_LIMIT:.0f} N-m, so")
    print(f"ankle_roll is correctly sized for this foot; it can already saturate the")
    print(f"tipping bound. The shortfall is not the actuator, it is that the reference")
    print(f"asks for {single and max(r[4]*G*abs(r[1]-r[3][r[2][0]]) for r in single) or 0:.0f} N-m of")
    print(f"restoring moment from a foot that can only ever deliver {tip:.1f}.")

    print("\nwhere single support happens:")
    runs, start = [], None
    idx = {r[0] for r in single}
    for f in range(n + 1):
        if f in idx and start is None:
            start = f
        elif f not in idx and start is not None:
            runs.append((start, f - 1))
            start = None
    for a, b in runs:
        sel = [r for r in single if a <= r[0] <= b]
        tq = max(r[4] * G * abs(r[1] - r[3][r[2][0]]) for r in sel)
        print(f"   t {a/50:5.2f} - {b/50:5.2f} s  ({(b-a+1)/50:.2f} s)  worst"
              f" {tq:5.1f} N-m = {tq/ROLL_LIMIT:4.1f}x the limit")


if __name__ == "__main__":
    main()
