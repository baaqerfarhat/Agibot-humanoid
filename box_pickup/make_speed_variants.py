"""Generate time-scaled variants of the box-pickup reference motion.

v29 trains on multiple playback speeds of the same pickup so the policy is
robust to executing the motion slower (longer squat hold, sustained grip
under load) or faster (more dynamic lift) than the nominal demonstration.
Positions/orientations are resampled along the time axis (linear / slerp)
and all velocities are scaled by the speed factor. Output .npz files keep
the holosoma motion format and the original 50 fps playback rate, so a
speed-0.8 clip simply has more frames (longer duration).

Usage: python make_speed_variants.py
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SRC = (
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
    "x2_31dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
OUT_DIR = (
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/"
    "x2_31dof/whole_body_tracking/box_multispeed"
)
SPEEDS = [0.8, 1.0, 1.25]

# v30: append this many seconds of the clip's final frame (static, zero
# velocity) to every variant. The clip ends at the default upright pose, so
# this trains the post-set-down HOLD -- the exact segment where the robot fell
# backward on hardware: in sim episodes used to reset the instant the clip
# ended, so balancing after releasing the box was never practiced. Deployment
# also plays several seconds past set-down before ramping out.
END_HOLD_SECONDS = 3.0


def lerp(arr: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0 for arrays of shape (T, ...)."""
    flat = arr.reshape(arr.shape[0], -1)
    out = np.stack([np.interp(t_new, t_orig, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((len(t_new),) + arr.shape[1:])


def slerp_wxyz(quats_wxyz: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Slerp for (T, 4) quaternions stored wxyz (holosoma npz convention)."""
    q_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    rots = Rotation.from_quat(q_xyzw)
    out_xyzw = Slerp(t_orig, rots)(t_new).as_quat()
    return out_xyzw[:, [3, 0, 1, 2]]


def append_end_hold(out: dict, fps: float) -> dict:
    """Repeat the final frame for END_HOLD_SECONDS with all velocities zeroed."""
    n = int(round(END_HOLD_SECONDS * fps))
    if n <= 0:
        return out
    for key in ["joint_pos", "body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w"]:
        last = out[key][-1:]
        out[key] = np.concatenate([out[key], np.repeat(last, n, axis=0)], axis=0)
    for key in ["joint_vel", "body_lin_vel_w", "body_ang_vel_w", "object_lin_vel_w", "object_ang_vel_w"]:
        zeros = np.zeros((n,) + out[key].shape[1:], dtype=out[key].dtype)
        out[key] = np.concatenate([out[key], zeros], axis=0)
    return out


def main() -> None:
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    data = dict(np.load(SRC))
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    T = data["joint_pos"].shape[0]
    duration = (T - 1) / fps
    t_orig = np.arange(T) / fps

    for s in SPEEDS:
        name = f"box_speed{int(round(s * 100)):03d}.npz"
        path = os.path.join(OUT_DIR, name)
        if s == 1.0:
            held = append_end_hold(dict(data), fps)
            np.savez(path, **held)
            print(f"{name}: copied + {END_HOLD_SECONDS}s end hold ({held['joint_pos'].shape[0]} frames)")
            continue

        new_duration = duration / s
        N = int(np.floor(new_duration * fps)) + 1
        # timeline in ORIGINAL motion time: frame i of the new clip shows the
        # original motion at time i/fps * s
        t_new = np.minimum(np.arange(N) / fps * s, t_orig[-1])

        out = {
            "fps": data["fps"],
            "joint_names": data["joint_names"],
            "body_names": data["body_names"],
        }

        # joint_pos = [root xyz (3), root quat wxyz (4), joints (31)]
        jp = data["joint_pos"]
        jp_new = np.empty((N, jp.shape[1]))
        jp_new[:, 0:3] = lerp(jp[:, 0:3], t_orig, t_new)
        jp_new[:, 3:7] = slerp_wxyz(jp[:, 3:7], t_orig, t_new)
        jp_new[:, 7:] = lerp(jp[:, 7:], t_orig, t_new)
        out["joint_pos"] = jp_new

        # joint_vel = [root lin (3), root ang (3), joints (31)] -- scale by s
        out["joint_vel"] = lerp(data["joint_vel"], t_orig, t_new) * s

        out["body_pos_w"] = lerp(data["body_pos_w"], t_orig, t_new)
        bq = data["body_quat_w"]
        bq_new = np.empty((N, bq.shape[1], 4))
        for b in range(bq.shape[1]):
            bq_new[:, b] = slerp_wxyz(bq[:, b], t_orig, t_new)
        out["body_quat_w"] = bq_new
        out["body_lin_vel_w"] = lerp(data["body_lin_vel_w"], t_orig, t_new) * s
        out["body_ang_vel_w"] = lerp(data["body_ang_vel_w"], t_orig, t_new) * s

        out["object_pos_w"] = lerp(data["object_pos_w"], t_orig, t_new)
        out["object_quat_w"] = slerp_wxyz(data["object_quat_w"], t_orig, t_new)
        out["object_lin_vel_w"] = lerp(data["object_lin_vel_w"], t_orig, t_new) * s
        out["object_ang_vel_w"] = lerp(data["object_ang_vel_w"], t_orig, t_new) * s

        out = append_end_hold(out, fps)
        np.savez(path, **out)
        print(f"{name}: {out['joint_pos'].shape[0]} frames ({new_duration:.2f}s motion + {END_HOLD_SECONDS}s hold, speed {s}x)")


if __name__ == "__main__":
    main()
