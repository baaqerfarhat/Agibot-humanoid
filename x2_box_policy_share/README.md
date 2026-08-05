# X2 box-pickup WBT policy share pack

Self-contained package for replaying / testing the AgiBot X2 **box pickup**
whole-body tracking policy (31 DoF) in another sim or on hardware.

## What’s inside

| Path | Contents |
|------|----------|
| `policy/x2_box_policy_v31.npz` | **Primary deployable policy** (numpy MLP + motion refs). Same bytes as `x2_box_policy.npz`. |
| `policy/x2_box_policy_v31_{slow080,fast125}.npz` | Time-scaled motion variants |
| `checkpoint/model_202500.pt` | Holosoma / rsl_rl training checkpoint (iter 202500) |
| `checkpoint/holosoma_config.yaml` | Training config from that run |
| `deploy/deploy_x2_box_pickup.py` | Hardware deploy (ROS 2 + numpy). Documents obs/action pipeline. |
| `deploy/deploy_x2_box_hybrid.py` | Hybrid: WBT pickup then freeze-at-hold + walk |
| `deploy/export_box_policy_npz.py` | How the `.npz` was produced from `.pt` |
| `deploy/run_logger.py`, `robot_states_control.py` | Deploy dependencies |
| `sim/infer_policy_numpy.py` | Minimal numpy loader + one-step / open-loop helper for **your sim** |
| `run_logs/` | Real-robot CSVs + `.meta.json` from box pickup / hybrid deploys |

## Recommended policy

Use **`policy/x2_box_policy_v31.npz`**.

- Source checkpoint: `model_202500.pt` (`x2_box_v31_flatfoot`)
- Control rate: **50 Hz**
- Obs dim: **164**, action dim: **31**
- Blind WBT: policy does **not** see the box; place the box at the reference location
- Motion (~14.7 s / 734 frames @ 50 Hz): stand → bend → grasp/lift → hold (~2 s) → set-down → stand

## Using it in your sim (numpy path)

```bash
cd sim
python infer_policy_numpy.py --policy ../policy/x2_box_policy_v31.npz --demo
```

Wire your sim to call `BoxPolicy.act(...)` each 20 ms with:

1. `joint_pos`, `joint_vel` in the **exact joint order** in `meta["joint_names"]`
2. torso IMU angular velocity (`base_ang_vel`)
3. torso orientation quaternion (xyzw) relative to world / yaw-invariant frame used in training
4. motion clock frame `t` (0 … `motion_frames-1`)

Then apply:

```text
target_q = action * action_scale + default_joint_pos
```

with the per-joint PD gains in `meta` (`joint_stiffness`, `joint_damping`).

Observation layout (holosoma alphabetical concat):

```text
[ prev_action(31),
  base_ang_vel(3),
  joint_pos - default(31),
  joint_vel(31),
  ref_joint_pos(31),
  ref_joint_vel(31),
  motion_ref_ori_b(6) ]   # first two columns of R_ref_in_body, row-major
```

See `sim/infer_policy_numpy.py` and comments at the top of
`deploy/deploy_x2_box_pickup.py` for the full pipeline.

## Using the torch checkpoint

If you already run Holosoma WBT:

```text
checkpoint/model_202500.pt
checkpoint/holosoma_config.yaml
```

Load with your usual evaluator / play script. The exported `.npz` is preferred if you
only need inference and a motion clock.

## Hardware deploy (optional)

`deploy/deploy_x2_box_pickup.py` expects ROS 2 (`rclpy`, `aimdk_msgs`) on the
AgiBot stack. For sim-only testing you do **not** need that — use `sim/`.

## Run logs

Each run is a pair:

- `YYYYMMDD_HHMMSS_box_pickup_*.csv` — 50 Hz commanded vs measured joints, base IMU, phase/frame
- matching `.meta.json` — joint names, gains, policy path, checkpoint

Aug 5 logs (`20260805_*`) are IRL tests of **this** v31 / 202500 policy.

## Notes / known issues (IRL)

On the real robot this checkpoint often **pitch-collapses early in the squat**
(~1–1.5 s after policy engage) even though sim stay upright. Waist pitch hard
limits are only ±18°. Useful context if you compare your sim vs these CSVs.
