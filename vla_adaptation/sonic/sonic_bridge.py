"""SONIC Python bridge — robot-state side (Terminal-2 replacement, CPU).

Phase 1 (this file): subscribe the sim's rt/lowstate over DDS, maintain the 10-frame
history buffers, build the decoder observation obs[994] per SONIC_BRIDGE_SPEC.md §2,
and run an OPEN-LOOP sanity check (no commands sent):

    dims exact; gravity_dir ~ [0,0,-1] at rest; q ~ near defaults; ang_vel ~ 0.

Conventions pinned from the C++ deploy (g1_deploy_onnx_ref.cpp):
    ang_vel     = lowstate.imu_state.gyroscope         (PELVIS imu — verified line 2896)
    gravity_dir = rotate([0,0,-1]) by conj(imu quaternion (w,x,y,z))
    histories   = 10 frames, sampled every control_dt (50 Hz), OLDEST first
    last_actions= bridge-side ring buffer (zeros until a policy runs)
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from g1_params import DEFAULT_ANGLES  # noqa: F401  (used by later phases too)

NJ = 29
HIST = 10
TOKEN_DIM = 64
OBS_DIM = TOKEN_DIM + 30 + 290 * 3 + 30      # 994


def quat_rotate_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by the conjugate of quaternion q (w,x,y,z) — vector into base frame."""
    w, x, y, z = q_wxyz
    q_vec = np.array([x, y, z])
    # conj(q) applied to v:  v' = v*(2w^2-1) - 2w*(q_vec x v) + 2*q_vec*(q_vec . v)
    return v * (2 * w * w - 1) - 2 * w * np.cross(q_vec, v) + 2 * q_vec * np.dot(q_vec, v)


class StateBuffer:
    def __init__(self):
        self.q = deque(maxlen=HIST)
        self.dq = deque(maxlen=HIST)
        self.ang = deque(maxlen=HIST)
        self.grav = deque(maxlen=HIST)
        self.act = deque(maxlen=HIST)
        for _ in range(HIST):
            self.act.append(np.zeros(NJ))

    def push_state(self, q, dq, ang, grav):
        self.q.append(q); self.dq.append(dq); self.ang.append(ang); self.grav.append(grav)

    def push_action(self, a):
        self.act.append(np.asarray(a, float))

    def ready(self) -> bool:
        return len(self.q) == HIST

    def obs(self, token: np.ndarray) -> np.ndarray:
        """Assemble obs[994]: token | ang(10x3) | q(10x29) | dq(10x29) | act(10x29) | grav(10x3),
        each history OLDEST first (deque iterates oldest->newest)."""
        parts = [np.asarray(token, float).reshape(TOKEN_DIM),
                 np.concatenate(list(self.ang)),
                 np.concatenate(list(self.q)),
                 np.concatenate(list(self.dq)),
                 np.concatenate(list(self.act)),
                 np.concatenate(list(self.grav))]
        o = np.concatenate(parts)
        assert o.shape == (OBS_DIM,), o.shape
        return o.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-id", type=int, default=1)
    ap.add_argument("--iface", type=str, default="lo")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    try:
        ChannelFactoryInitialize(a.domain_id, a.iface)
    except Exception:
        ChannelFactoryInitialize(a.domain_id)

    buf = StateBuffer()
    n_msg = 0

    def on_lowstate(msg: LowState_):
        nonlocal n_msg
        n_msg += 1
        q = np.array([msg.motor_state[i].q for i in range(NJ)])
        dq = np.array([msg.motor_state[i].dq for i in range(NJ)])
        quat = np.array(msg.imu_state.quaternion)          # w, x, y, z
        ang = np.array(msg.imu_state.gyroscope)
        grav = quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        buf.push_state(q, dq, ang, grav)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_lowstate, 10)

    t0 = time.time()
    while time.time() - t0 < a.seconds:
        time.sleep(0.2)

    print(f"messages received: {n_msg}")
    if not buf.ready():
        print("NOT READY — insufficient lowstate messages (is run_sim_loop running? "
              "domain/iface right?)")
        return
    o = buf.obs(np.zeros(TOKEN_DIM))
    q_now = list(buf.q)[-1]
    print(f"obs dims OK: {o.shape}")
    print(f"gravity_dir (latest): {np.round(list(buf.grav)[-1], 3)}   (rest ~ [0,0,-1])")
    print(f"|ang_vel| (latest): {np.linalg.norm(list(buf.ang)[-1]):.4f} rad/s")
    print(f"mean |q - defaults|: {np.abs(q_now - DEFAULT_ANGLES).mean():.3f} rad")
    print("OPEN-LOOP STATE BRIDGE OK")


if __name__ == "__main__":
    main()
