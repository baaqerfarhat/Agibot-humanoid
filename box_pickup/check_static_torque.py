"""Gravity-torque feasibility of a reference clip against the X2 actuator limits.

A reference the robot cannot hold statically will never track well, however good
the RL policy is, so this reports the gravity load each joint has to carry in the
reference pose (plus the payload) against the URDF effort limits.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rebuild_reference_motion import URDF, Robot

G = np.array([0.0, 0.0, -9.81])


def effort_limits(urdf):
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("effort") is not None:
            out[j.get("name")] = float(lim.get("effort"))
    return out


def subtrees(chain):
    kids = {}
    for j in chain.joints:
        kids.setdefault(j.parent, []).append(j.child)

    def collect(link):
        out, stack = [], [link]
        while stack:
            l = stack.pop()
            out.append(l)
            stack.extend(kids.get(l, []))
        return out

    return {j.name: collect(j.child) for j in chain.joints if j.jtype in ("revolute", "continuous")}


def main():
    path = sys.argv[1]
    payload = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    d = np.load(path, allow_pickle=True)
    jn = [str(x) for x in d["joint_names"]]
    qp = d["joint_pos"]
    op = d["object_pos_w"]
    n = len(qp)
    robot = Robot()
    eff = effort_limits(URDF)
    sub = subtrees(robot.chain)
    jinfo = {j.name: j for j in robot.chain.joints}
    held = np.linalg.norm(np.diff(op, axis=0, prepend=op[:1]), axis=1) > 1e-9
    held |= np.roll(held, 1)

    peak = {nm: 0.0 for nm in jn}
    peak_t = {nm: 0.0 for nm in jn}
    for f in range(n):
        out = robot.fk(qp[f, 7:], jn, qp[f, 0:3], qp[f, 3:7])
        # world CoM and mass of every link
        lm = {}
        for link, (m, c) in robot.mass.items():
            if link in out:
                p, R = out[link]
                lm[link] = (m, p + R @ c)
        for nm in jn:
            if nm not in sub or nm not in eff:
                continue
            j = jinfo[nm]
            jp, jR = out[j.child]
            axis = jR @ j.axis
            tau = np.zeros(3)
            for link in sub[nm]:
                if link not in lm:
                    continue
                m, c = lm[link]
                tau += np.cross(c - jp, m * G)
            if held[f]:  # payload rides with the hands, so it loads the arms too
                tau += np.cross(op[f] - jp, payload * G)
            t = abs(float(axis @ tau))
            if t > peak[nm]:
                peak[nm], peak_t[nm] = t, f / 50.0

    print(f"{path.split('/')[-1]}   payload {payload:.1f} kg")
    print(f"{'joint':28s} {'peak |tau|':>11s} {'limit':>8s} {'use':>7s}   at")
    rows = sorted(
        ((nm, peak[nm], eff[nm]) for nm in jn if nm in eff), key=lambda r: -r[1] / r[2]
    )
    over = 0
    for nm, t, e in rows[:12]:
        flag = "  <-- OVER" if t > e else ""
        over += t > e
        print(f"{nm:28s} {t:9.1f} Nm {e:7.1f} {100*t/e:6.0f} %   t={peak_t[nm]:5.2f}s{flag}")
    print(f"\njoints over their effort limit: {sum(1 for nm,t,e in rows if t>e)} / {len(rows)}")


if __name__ == "__main__":
    main()
