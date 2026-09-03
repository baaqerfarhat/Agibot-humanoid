"""Shared squat policy + command used by both the mjlab and Isaac dumps.

Must stay numpy-only so the Isaac (hssim) interpreter can import it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


DEFAULT_POLICY = (
    Path(__file__).resolve().parents[1]
    / "agibot_control_functions"
    / "policies"
    / "x2_squat_policy_40pct_iter16499.npz"
)
CONTROL_DT = 0.02
CYCLE_S = 5.0
HOLD_S = 1.5
NUM_STEPS = int(round((CYCLE_S + HOLD_S) / CONTROL_DT))  # 325


class NumpyPolicy:
    def __init__(self, npz_path: str | Path):
        d = np.load(npz_path, allow_pickle=True)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.meta = json.loads(str(d["meta_json"]))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = x @ self.W[i].T + self.b[i]
            x = np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)
        return x @ self.W[-1].T + self.b[-1]


def cosine_smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return 0.5 * (1.0 - math.cos(math.pi * s))


class SquatCommand:
    """Same stand-squat-stand trajectory as mjlab SquatCommand (play mode)."""

    def __init__(self, meta: dict):
        self.standing = float(meta["standing_height"])
        self.frac = float(meta["squat_height_frac"])
        self.cycle = float(meta["cycle_time_s"])
        self.t_stand = float(meta["stand_duration_s"])
        self.t_down = float(meta["descend_duration_s"])
        self.t_bottom = float(meta["bottom_duration_s"])
        self.t_up = float(meta["ascend_duration_s"])
        self.wrap = bool(meta.get("wrap_cycle", False))
        self.hold_t = float(meta.get("hold_time_s", self.cycle * 0.9))
        self.h_squat = self.standing * self.frac

    def elapsed(self, t: float) -> float:
        if self.wrap:
            return t % self.cycle
        return min(t, self.hold_t)

    def target_height(self, t: float) -> float:
        t0 = self.t_stand
        t1 = t0 + self.t_down
        t2 = t1 + self.t_bottom
        t3 = t2 + self.t_up
        if t0 <= t < t1:
            a = cosine_smoothstep((t - t0) / max(self.t_down, 1e-6))
            return self.standing + (self.h_squat - self.standing) * a
        if t1 <= t < t2:
            return self.h_squat
        if t2 <= t < t3:
            a = cosine_smoothstep((t - t2) / max(self.t_up, 1e-6))
            return self.h_squat + (self.standing - self.h_squat) * a
        return self.standing

    def command(self, t: float) -> np.ndarray:
        te = self.elapsed(t)
        phase = te / self.cycle
        h = self.target_height(te)
        two_pi = 2.0 * math.pi
        return np.array(
            [math.sin(two_pi * phase), math.cos(two_pi * phase), h], np.float32
        )


def quat_rotate_inverse_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return (a - b + c).astype(np.float32)


def rpy_from_xyzw(q: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = [float(v) for v in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def build_obs(
    meta: dict,
    ang_vel: np.ndarray,
    proj_g: np.ndarray,
    q_rel: np.ndarray,
    dq: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    parts = []
    for name in meta["observation_names"]:
        if name == "base_lin_vel":
            parts.append(np.zeros(3, np.float32))
        elif name == "base_ang_vel":
            parts.append(np.asarray(ang_vel, np.float32).reshape(-1))
        elif name == "projected_gravity":
            parts.append(np.asarray(proj_g, np.float32).reshape(-1))
        elif name == "joint_pos":
            parts.append(np.asarray(q_rel, np.float32).reshape(-1))
        elif name == "joint_vel":
            parts.append(np.asarray(dq, np.float32).reshape(-1))
        elif name == "actions":
            parts.append(np.asarray(last_action, np.float32).reshape(-1))
        elif name == "command":
            parts.append(np.asarray(command, np.float32).reshape(-1))
        else:
            raise ValueError(f"Unhandled observation term: {name!r}")
    return np.concatenate(parts).astype(np.float32)


def name_index(src_names: list[str], dst_names: list[str]) -> np.ndarray:
    """Index array `idx` such that `dst[i] = src[idx[i]]` when names match."""
    lookup = {n: i for i, n in enumerate(src_names)}
    missing = [n for n in dst_names if n not in lookup]
    if missing:
        raise ValueError(f"Joint names missing from source: {missing}")
    return np.array([lookup[n] for n in dst_names], dtype=np.int64)


def save_rollout(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[dump] wrote {path}")
