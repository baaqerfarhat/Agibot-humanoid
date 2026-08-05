#!/usr/bin/env python3
"""Minimal numpy inference for the X2 box-pickup WBT policy.

No ROS / torch required. Drop this into your sim and call BoxPolicy.step() at 50 Hz.

Example (open-loop demo on the bundled reference motion):

    python infer_policy_numpy.py --policy ../policy/x2_box_policy_v31.npz --demo

Wire to your simulator:

    pol = BoxPolicy("x2_box_policy_v31.npz")
    pol.align_yaw(torso_quat_xyzw)          # once at engage
    for frame in range(pol.n_frames):
        target_q, action = pol.step(
            joint_pos=...,                  # (31,) rad, policy joint order
            joint_vel=...,                  # (31,) rad/s
            base_ang_vel=...,               # (3,) torso gyro
            torso_quat_xyzw=...,            # (4,)
            frame=frame,
        )
        # apply PD: kp/kd from pol.stiffness / pol.damping toward target_q
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(np.clip(x, -30.0, 0.0)) - 1.0)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        np.float32,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        np.float32,
    )


def yaw_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32)


class BoxPolicy:
    def __init__(self, npz_path: str | Path):
        d = np.load(npz_path, allow_pickle=True)
        self.meta = json.loads(str(d["meta_json"]))
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(n)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(n)]
        self.ref_joint_pos = d["ref_joint_pos"].astype(np.float32)
        self.ref_joint_vel = d["ref_joint_vel"].astype(np.float32)
        self.ref_quat_xyzw = d["ref_quat_xyzw"].astype(np.float32)

        self.joint_names = list(self.meta["joint_names"])
        self.default = np.array(self.meta["default_joint_pos"], np.float32)
        self.action_scale = np.array(self.meta["action_scale"], np.float32)
        self.stiffness = np.array(self.meta["joint_stiffness"], np.float32)
        self.damping = np.array(self.meta["joint_damping"], np.float32)
        self.n_frames = int(self.ref_joint_pos.shape[0])
        self.action_dim = int(self.meta["action_dim"])
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.yaw_offset = np.array([0, 0, 0, 1], np.float32)

    def align_yaw(self, torso_quat_xyzw: np.ndarray) -> None:
        """Call once at engage so the motion heading matches the robot."""
        q_robot_yaw = yaw_quat(np.asarray(torso_quat_xyzw, np.float32))
        q_ref0_yaw = yaw_quat(self.ref_quat_xyzw[0])
        self.yaw_offset = quat_mul(q_robot_yaw, quat_inv(q_ref0_yaw))

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def act_raw(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / self.std
        for i in range(len(self.W) - 1):
            x = elu(x @ self.W[i].T + self.b[i])
        return x @ self.W[-1].T + self.b[-1]

    def build_obs(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        torso_quat_xyzw: np.ndarray,
        frame: int,
    ) -> np.ndarray:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        q_ref = quat_mul(self.yaw_offset, self.ref_quat_xyzw[frame])
        q_rel = quat_mul(quat_inv(np.asarray(torso_quat_xyzw, np.float32)), q_ref)
        ori6 = quat_to_mat(q_rel)[:, :2].reshape(-1)
        # Alphabetical holosoma order:
        # actions, base_ang_vel, dof_pos, dof_vel, motion_command, motion_ref_ori_b
        return np.concatenate(
            [
                self.last_action,
                np.asarray(base_ang_vel, np.float32).reshape(3),
                np.asarray(joint_pos, np.float32).reshape(31) - self.default,
                np.asarray(joint_vel, np.float32).reshape(31),
                self.ref_joint_pos[frame],
                self.ref_joint_vel[frame],
                ori6,
            ]
        ).astype(np.float32)

    def step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        torso_quat_xyzw: np.ndarray,
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (target_q, action). Updates internal prev-action."""
        obs = self.build_obs(joint_pos, joint_vel, base_ang_vel, torso_quat_xyzw, frame)
        action = self.act_raw(obs)
        self.last_action = action.copy()
        target_q = action * self.action_scale + self.default
        return target_q.astype(np.float32), action.astype(np.float32)


def demo(policy_path: Path) -> None:
    pol = BoxPolicy(policy_path)
    print(f"loaded {policy_path.name}")
    print(f"  frames={pol.n_frames}  obs={pol.meta['obs_dim']}  act={pol.action_dim}")
    print(f"  control_hz={pol.meta['control_hz']}  task={pol.meta['task']}")
    print(f"  hold_frame_range={pol.meta.get('hold_frame_range')}")
    print(f"  joints[{len(pol.joint_names)}]={pol.joint_names[:4]} ...")

    # Open-loop: pretend the robot tracks the reference perfectly.
    pol.align_yaw(pol.ref_quat_xyzw[0])
    pol.reset()
    errs = []
    for f in range(0, pol.n_frames, 10):
        q = pol.ref_joint_pos[f]
        dq = pol.ref_joint_vel[f]
        tgt, _ = pol.step(q, dq, np.zeros(3, np.float32), pol.ref_quat_xyzw[f], f)
        errs.append(float(np.linalg.norm(tgt - q)))
    print(f"  open-loop |target - ref| L2  mean={np.mean(errs):.3f}  max={np.max(errs):.3f} rad")
    print("  (non-zero is expected: policy corrects; this just smoke-tests the forward pass)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "policy" / "x2_box_policy_v31.npz",
    )
    ap.add_argument("--demo", action="store_true", help="Smoke-test forward pass on the ref motion")
    args = ap.parse_args()
    if not args.policy.is_file():
        raise SystemExit(f"policy not found: {args.policy}")
    demo(args.policy)


if __name__ == "__main__":
    main()
