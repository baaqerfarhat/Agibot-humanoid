"""Rotation increments done properly.

Everything upstream measured orientation change as `axis_angle(q1) - axis_angle(q0)`: a
difference of two axis-angle VECTORS. That is not a metric on SO(3). Axis-angle is a chart,
not a vector space -- the same physical rotation has representations differing by 2*pi in
magnitude, the chart is singular at 0 and pi, and subtracting two charts only approximates
the true increment when both rotations are small and nearly parallel. Neither holds along a
manipulation trajectory.

The consequences were visible and were blamed on the plant: the FIR model fitted rotation at
R^2 = 0.11-0.49 against 0.98 for translation, and the measured sensitivity matrix came out
with its rotation block SWAPPED and sign-flipped (dry <- drz at +0.383, drz <- dry at -0.424,
own diagonals ~0.01). Those are the fingerprints of a bad chart, not of bad dynamics.

The correct increment is the RELATIVE rotation, mapped to the tangent space at identity:

    q_rel = q1 (x) conj(q0)          then     omega = axis_angle(q_rel)

robosuite stores quaternions xyzw.
"""
from __future__ import annotations

import numpy as np


def qmul(a, b):
    """Hamilton product, xyzw convention."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def axis_angle(q):
    """Axis-angle of a single quaternion, with the shortest-arc branch taken."""
    q = np.asarray(q, float)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[3] < 0:                    # q and -q are the same rotation; pick |angle| <= pi
        q = -q
    w = np.clip(q[3], -1.0, 1.0)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3)
    return (q[:3] / s) * (2.0 * np.arccos(w))


def rot_delta(q0, q1):
    """The rotation increment from q0 to q1, as a tangent vector at identity.

    This is what `axis_angle(q1) - axis_angle(q0)` was standing in for, and unlike that
    expression it is exact, singularity-free away from pi, and sign-consistent.
    """
    return axis_angle(qmul(np.asarray(q1, float), qconj(np.asarray(q0, float))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # a small increment applied to a random base rotation must be recovered exactly
    worst_new = worst_old = 0.0
    for _ in range(2000):
        q0 = rng.normal(size=4); q0 /= np.linalg.norm(q0)
        w = rng.normal(size=3) * 0.05                       # true increment
        th = np.linalg.norm(w)
        dq = np.concatenate([w / th * np.sin(th / 2), [np.cos(th / 2)]])
        q1 = qmul(dq, q0)
        worst_new = max(worst_new, np.linalg.norm(rot_delta(q0, q1) - w))
        worst_old = max(worst_old, np.linalg.norm((axis_angle(q1) - axis_angle(q0)) - w))
    print(f"max error over 2000 random cases, true increment |w| ~ 0.05 rad")
    print(f"  rot_delta (relative rotation) : {worst_new:.2e}")
    print(f"  axis_angle(q1) - axis_angle(q0): {worst_old:.2e}   <- what was used before")
