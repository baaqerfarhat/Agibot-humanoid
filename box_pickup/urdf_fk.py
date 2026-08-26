"""Minimal forward kinematics for the X2 URDF.

Written because neither mujoco nor pinocchio is available in the hssim
environment, and the retargeted motion files need to be checked (and
regenerated) against the robot's real kinematic tree.

Only revolute/continuous/fixed joints are supported, which covers the X2.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), the convention Isaac uses."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray
    jtype: str
    lower: float
    upper: float


class UrdfChain:
    def __init__(self, urdf_path: str):
        root = ET.parse(urdf_path).getroot()
        self.links = [l.get("name") for l in root.iter("link")]
        self.joints: list[Joint] = []
        for j in root.iter("joint"):
            o = j.find("origin")
            xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            a = j.find("axis")
            axis = np.fromstring(a.get("xyz", "1 0 0"), sep=" ") if a is not None else np.array([1.0, 0, 0])
            lim = j.find("limit")
            lo = float(lim.get("lower")) if lim is not None and lim.get("lower") is not None else -np.inf
            hi = float(lim.get("upper")) if lim is not None and lim.get("upper") is not None else np.inf
            self.joints.append(
                Joint(
                    name=j.get("name"),
                    parent=j.find("parent").get("link"),
                    child=j.find("child").get("link"),
                    origin_xyz=xyz,
                    origin_rot=rpy_to_mat(rpy),
                    axis=axis,
                    jtype=j.get("type"),
                    lower=lo,
                    upper=hi,
                )
            )
        self.by_child = {j.child: j for j in self.joints}
        children = {j.child for j in self.joints}
        roots = [l for l in self.links if l not in children]
        self.root = roots[0]
        self.actuated = [j.name for j in self.joints if j.jtype in ("revolute", "continuous")]
        self.limits = {j.name: (j.lower, j.upper) for j in self.joints if j.jtype == "revolute"}
        self._order = self._topo_order()

    def _topo_order(self) -> list[str]:
        order, seen = [], {self.root}
        kids: dict[str, list[str]] = {}
        for j in self.joints:
            kids.setdefault(j.parent, []).append(j.child)
        stack = [self.root]
        while stack:
            n = stack.pop()
            order.append(n)
            for c in kids.get(n, []):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return order

    def fk(self, q: dict[str, float], root_pos=None, root_quat=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return {link: (position, rotation_matrix)} in world frame."""
        R0 = quat_wxyz_to_mat(np.asarray(root_quat)) if root_quat is not None else np.eye(3)
        p0 = np.asarray(root_pos, dtype=float) if root_pos is not None else np.zeros(3)
        out = {self.root: (p0, R0)}
        for link in self._order:
            if link == self.root:
                continue
            j = self.by_child[link]
            pp, pR = out[j.parent]
            R = pR @ j.origin_rot
            p = pp + pR @ j.origin_xyz
            if j.jtype in ("revolute", "continuous"):
                R = R @ axis_angle_to_mat(j.axis, float(q.get(j.name, 0.0)))
            out[link] = (p, R)
        return out
