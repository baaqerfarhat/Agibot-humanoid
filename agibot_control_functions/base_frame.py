#!/usr/bin/env python3
"""Reconstruct the PELVIS base state from the torso IMU and the waist joints.

Why this exists
---------------
holosoma builds the WBT `base_ang_vel` observation from the articulation ROOT,
which for x2 is the `pelvis` freejoint body, expressed in the pelvis frame:

    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(base_quat, ang_vel_world)      # base_quat = root quat

The deployment has no pelvis IMU, so it fed the torso IMU gyro straight into
that slot. The torso is three waist joints above the pelvis, so that substitutes
a different body AND a different frame:

  * body  -- omega_torso = omega_pelvis + omega_waist. On the v33 runs the waist
             rate is ~80% of the total gyro magnitude through the deep bend, so
             most of what the policy read as "my base is rotating" was the waist
             bending.
  * frame -- the gyro is resolved in the torso frame, which sits at the +20 deg
             waist-pitch limit with a steady ~+9 deg waist roll during the bend.
             A pitch offset rotates roll into yaw and vice versa, so even a
             perfect pelvis rate lands in the wrong channels.

`motion_ref_ori_b` is NOT affected: holosoma's x2 command config sets
`body_name_ref=["torso_link"]`, so the reference orientation really is the
torso's, and the raw IMU quat is the right input for that term.

Kinematics (from x2_31dof_w_largebox.xml, all joint origins coincident, no
fixed rotation offsets between the links):

    pelvis --[waist_yaw, axis z]--> waist_yaw_link
           --[waist_pitch, axis y]--> waist_pitch_link
           --[waist_roll, axis x]--> torso_link

so R_pelvis_from_torso = Rz(yaw) @ Ry(pitch) @ Rx(roll), and

    omega_pelvis_in_pelvis = R_pelvis_from_torso @ omega_torso_in_torso
                             - omega_waist_in_pelvis

    omega_waist_in_pelvis = z * yaw_rate
                            + Rz(yaw) @ y * pitch_rate
                            + Rz(yaw) @ Ry(pitch) @ x * roll_rate

Pure numpy, no ROS, so it can be unit-tested and replayed against recorded logs.
"""

from __future__ import annotations

import numpy as np

WAIST_JOINTS = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Shepperd's method: pick the largest denominator for numerical stability."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def pelvis_from_torso_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R such that v_pelvis = R @ v_torso."""
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def waist_ang_vel_in_pelvis(
    yaw: float, pitch: float, yaw_rate: float, pitch_rate: float, roll_rate: float
) -> np.ndarray:
    """Angular velocity of the torso relative to the pelvis, in pelvis coords."""
    Rz = _rot_z(yaw)
    return _EZ * yaw_rate + (Rz @ _EY) * pitch_rate + (Rz @ _rot_y(pitch) @ _EX) * roll_rate


class PelvisEstimator:
    """Turns (torso IMU, waist joint state) into the pelvis base state the policy expects.

    Usage in the control loop::

        est = PelvisEstimator()
        ang_vel, pelvis_quat = est.update(imu.quat, imu.ang_vel, jmap)

    `ang_vel` goes into the `base_ang_vel` observation slot. `pelvis_quat` is
    exposed for the roll-abort safety check, which otherwise trips on waist roll
    rather than on an actual pelvis lean. The raw IMU quat should still be used
    for `motion_ref_ori_b` (that term tracks torso_link by design).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.last_correction = np.zeros(3)

    def waist_state(self, jmap):
        q = np.array([float(jmap[n].position) for n in WAIST_JOINTS])
        dq = np.array([float(jmap[n].velocity) for n in WAIST_JOINTS])
        return q, dq

    def update(self, imu_quat_xyzw, imu_ang_vel, jmap):
        """Return (base_ang_vel_for_obs, pelvis_quat_xyzw)."""
        w_torso = np.asarray(imu_ang_vel, float)
        q_torso = np.asarray(imu_quat_xyzw, float)
        if not self.enabled:
            self.last_correction[:] = 0.0
            return w_torso.astype(np.float32), q_torso.astype(np.float32)

        (yaw, pitch, roll), (yaw_r, pitch_r, roll_r) = self.waist_state(jmap)
        R_pt = pelvis_from_torso_rot(yaw, pitch, roll)
        w_pelvis = R_pt @ w_torso - waist_ang_vel_in_pelvis(yaw, pitch, yaw_r, pitch_r, roll_r)

        # pelvis attitude: q_world_pelvis = q_world_torso * inv(q_pelvis_torso)
        R_world_pelvis = _quat_xyzw_to_mat(q_torso) @ R_pt.T
        q_pelvis = _mat_to_quat_xyzw(R_world_pelvis)

        self.last_correction = w_pelvis - w_torso
        return w_pelvis.astype(np.float32), q_pelvis.astype(np.float32)


def _self_test() -> None:
    """Consistency checks that do not need the robot."""
    rng = np.random.default_rng(0)

    class J:
        def __init__(self, p, v):
            self.position, self.velocity = p, v

    # 1. zero waist state must be the identity transform
    est = PelvisEstimator()
    jmap = {n: J(0.0, 0.0) for n in WAIST_JOINTS}
    w = np.array([0.1, -0.2, 0.3])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    wp, qp = est.update(q, w, jmap)
    assert np.allclose(wp, w, atol=1e-9), wp
    assert np.allclose(qp, q, atol=1e-9), qp

    # 2. pure waist motion with a stationary pelvis must reconstruct to zero rate
    for _ in range(200):
        ang = rng.uniform(-0.5, 0.5, 3)
        rate = rng.uniform(-2.0, 2.0, 3)
        yaw, pitch, roll = ang
        jmap = {n: J(a, r) for n, a, r in zip(WAIST_JOINTS, ang, rate)}
        # a stationary pelvis sees the torso spinning at exactly the waist rate,
        # measured in the torso frame
        R_pt = pelvis_from_torso_rot(yaw, pitch, roll)
        w_rel_p = waist_ang_vel_in_pelvis(yaw, pitch, rate[0], rate[1], rate[2])
        w_torso_meas = R_pt.T @ w_rel_p
        est = PelvisEstimator()
        wp, _ = est.update(np.array([0.0, 0.0, 0.0, 1.0]), w_torso_meas, jmap)
        assert np.abs(wp).max() < 1e-9, (ang, rate, wp)

    # 3. quat round-trip: reconstructed pelvis quat composed back gives the torso
    #    quat. Tolerance is float32 eps because update() returns obs-ready float32.
    for _ in range(200):
        ang = rng.uniform(-0.5, 0.5, 3)
        jmap = {n: J(a, 0.0) for n, a in zip(WAIST_JOINTS, ang)}
        v = rng.normal(size=4)
        q_torso = v / np.linalg.norm(v)
        est = PelvisEstimator()
        _, q_pelvis = est.update(q_torso, np.zeros(3), jmap)
        R_back = _quat_xyzw_to_mat(q_pelvis) @ pelvis_from_torso_rot(*ang)
        assert np.allclose(R_back, _quat_xyzw_to_mat(q_torso), atol=1e-6)

    # 4. disabled estimator is a pass-through
    est = PelvisEstimator(enabled=False)
    jmap = {n: J(0.3, 1.0) for n in WAIST_JOINTS}
    wp, qp = est.update(q, w, jmap)
    assert np.allclose(wp, w) and np.allclose(qp, q)

    print("base_frame self-test OK (identity, pure-waist cancellation, quat round-trip, bypass)")


if __name__ == "__main__":
    _self_test()
