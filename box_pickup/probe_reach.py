"""How far can the X2 arm actually reach, and where would it have to stand?

The retarget IK keeps failing to put the palms on the box, and there are two very
different reasons that could happen: the arm is too short for the grip points, or the
balance/foot constraints are stealing the solution. This separates them. First measure
true arm reach off the model, then -- for a grid of stance offsets -- ask only
"is the grip within reach of the shoulder", ignoring balance entirely.

    cd ~/baaqer_ws/mjlab && uv run python <this>
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

CLIP = Path(
    "/home/baaqer/baaqer_ws/holosoma/src/holosoma/holosoma/data/motions/x2_31dof"
    "/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
BOX_SIZE = np.array([0.4712, 0.4587, 0.4079])
BOX_OFFSET = np.array([0.0015, -0.0007, 0.0058])
PALM_OFFSET = np.array([0.01, 0.0, -0.10])
PALM_RADIUS = 0.05
GRIP = 0.005
HANDS = ("left_wrist_roll_link", "right_wrist_roll_link")
SHOULDERS = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
import sys
GRIP_FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main() -> None:
    from mjlab.asset_zoo.robots.x2.x2_constants import get_x2_robot_cfg

    model = get_x2_robot_cfg().spec_fn().compile()
    data = mujoco.MjData(model)
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    mj_j = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)]

    # --- true arm reach: sweep the arm joints, take the farthest palm from shoulder.
    print("=" * 72)
    print("ARM REACH (shoulder origin -> palm sphere centre)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    arm_j = [j for j in mj_j if j.startswith("left_") and
             any(p in j for p in ("shoulder", "elbow", "wrist"))]
    adr = [model.jnt_qposadr[mj_j.index(j)] for j in arm_j]
    lo = np.array([model.jnt_range[mj_j.index(j), 0] for j in arm_j])
    hi = np.array([model.jnt_range[mj_j.index(j), 1] for j in arm_j])
    best = 0.0
    for _ in range(20000):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)
        R = data.xmat[bid(HANDS[0])].reshape(3, 3)
        p = data.xpos[bid(HANDS[0])] + R @ PALM_OFFSET
        best = max(best, float(np.linalg.norm(p - data.xpos[bid(SHOULDERS[0])])))
    print(f"max shoulder->palm distance: {best:.3f} m")
    reach = best

    # --- grip points demanded by the clip.
    clip = np.load(CLIP, allow_pickle=True)
    cb = [str(x) for x in clip["body_names"]]
    bp = np.asarray(clip["body_pos_w"], np.float64)
    bq = np.asarray(clip["body_quat_w"], np.float64)
    obj = np.asarray(clip["object_pos_w"], np.float64)
    oq = np.asarray(clip["object_quat_w"], np.float64)
    half = BOX_SIZE / 2

    def mat(q):
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, q)
        return m.reshape(3, 3)

    Rb = mat(oq[GRIP_FRAME])
    centre = obj[GRIP_FRAME] + Rb @ BOX_OFFSET
    grips, faces = [], []
    for k, h in enumerate(HANDS):
        ci = cb.index(h)
        p = bp[GRIP_FRAME][ci] + mat(bq[GRIP_FRAME][ci]) @ PALM_OFFSET
        local = Rb.T @ (p - centre)
        d = np.abs(local) - half
        ax = int(np.argmax(d[:2]))
        want = local.copy()
        want[ax] = np.sign(local[ax]) * (half[ax] + PALM_RADIUS - GRIP)
        grips.append(centre + Rb @ want)
        faces.append(f"{'xy'[ax]}{'+' if local[ax] > 0 else '-'}")
    print(f"\ngrip points at t={GRIP_FRAME / 50:.2f}s  (faces L={faces[0]} R={faces[1]})")
    for k in range(2):
        print(f"  {HANDS[k]:<24} target {np.round(grips[k], 3)}")

    foot_mid = 0.5 * (bp[GRIP_FRAME][cb.index(FEET[0])]
                      + bp[GRIP_FRAME][cb.index(FEET[1])])
    foot_mid[2] = 0.0
    print(f"  foot midpoint {np.round(foot_mid, 3)}   box z {obj[GRIP_FRAME, 2]:.3f} m")
    # The clip's own facing, so the search result can be read as a delta.
    qp = bq[GRIP_FRAME][cb.index("pelvis")]
    clip_yaw = np.arctan2(2 * (qp[0] * qp[3] + qp[1] * qp[2]),
                          1 - 2 * (qp[2] ** 2 + qp[3] ** 2))
    print(f"  clip pelvis yaw {np.degrees(clip_yaw):+.1f} deg")

    # --- where do the shoulders sit in a squat? Measure on the model directly.
    print()
    print("=" * 72)
    print("STANCE SEARCH: shoulder->grip distance vs reach, for a squat posture")
    print("=" * 72)
    # Squat posture: hips/knees/ankles flexed to put the pelvis at a target height,
    # torso upright. Take shoulder offsets relative to the foot midpoint from it.
    for pelvis_h in (0.65, 0.50, 0.40, 0.30):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        # crude squat: knee = 2*theta, hip = -theta, ankle = -theta
        th = np.arccos(np.clip(pelvis_h / 0.72, -1, 1))
        for side in ("left", "right"):
            for jn, val in ((f"{side}_hip_pitch_joint", -th),
                            (f"{side}_knee_joint", 2 * th),
                            (f"{side}_ankle_pitch_joint", -th)):
                if jn in mj_j:
                    data.qpos[model.jnt_qposadr[mj_j.index(jn)]] = val
        mujoco.mj_forward(model, data)
        fm = 0.5 * (data.xpos[bid(FEET[0])] + data.xpos[bid(FEET[1])])
        sh = [data.xpos[bid(s)] - fm for s in SHOULDERS]
        h = data.xpos[bid("pelvis")][2] - data.xpos[bid(FEET[0])][2]
        print(f"\npelvis {h:.3f} m above feet; shoulder offsets from foot midpoint:")
        for k in range(2):
            print(f"    {SHOULDERS[k]:<28} {np.round(sh[k], 3)}")

        # Grid over stance offsets: how far must each shoulder stretch?
        best = None
        for dx in np.arange(-0.1, 0.85, 0.05):
            for dy in np.arange(-0.85, 0.35, 0.05):
                for yaw in np.radians(np.arange(-90, 95, 10)):
                    c, s = np.cos(yaw), np.sin(yaw)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    origin = foot_mid + np.array([dx, dy, 0.0])
                    need = [float(np.linalg.norm(grips[k] - (origin + Rz @ sh[k])))
                            for k in range(2)]
                    worst = max(need)
                    if best is None or worst < best[0]:
                        best = (worst, dx, dy, np.degrees(yaw), need)
        worst, dx, dy, yaw, need = best
        ok = "REACHABLE" if worst <= reach else f"SHORT by {(worst - reach) * 1e3:.0f} mm"
        print(f"    best stance dx {dx:+.2f} dy {dy:+.2f} yaw {yaw:+.0f} deg abs "
              f"({yaw - np.degrees(clip_yaw):+.0f} vs clip) -> "
              f"need L {need[0]:.3f} R {need[1]:.3f} m vs reach {reach:.3f}  [{ok}]")


if __name__ == "__main__":
    main()
