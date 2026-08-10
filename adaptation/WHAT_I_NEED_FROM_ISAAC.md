# What I need from the Isaac Sim setup — x2_box_pickup v31

Short version: **one rollout log with per-step observations** would probably close this. The rest
is nice-to-have.

Context in one paragraph: I'm reproducing the v31 box-pickup policy in MuJoCo to study online
adaptation on top of it. The policy works in your Isaac Sim and on the real robot (the logged
hardware run ends 2.9° from vertical after the bend). In my MuJoCo it ends up collapsed. I've
verified every observation term against holosoma source, all three joint orderings, the PD gains,
armature, effort limits, link masses, foot geometry, ground friction, the contact model, four
integrators and three timesteps — and it still diverges. I've run out of things to check by
elimination, so I need to diff against a working rollout.

---

## 1. One rollout log with per-step observations — the one that matters

A single episode of v31 in your environment, logging **every control step**:

| field | shape | note |
|---|---|---|
| `actor_obs` **per term**, not just the concatenated vector | 734 × 164 | the important part — I need to compare term by term |
| `action` | 734 × 31 | raw policy output, before `action_scale` |
| joint position | 734 × 31 | |
| joint velocity | 734 × 31 | |
| **base position + quaternion** | 734 × 7 | **this is where my sim diverges** |
| object pose, if the scene has the box | 734 × 7 | |
| termination step + reason | — | did it finish, or terminate early? |

`box_eval_isaac.py --dump-obs` (attached, in `ISAAC_TEST/`) already produces the per-term
observation trace — it monkey-patches `ObservationManager._compute_term` and writes an npz. It
does **not** currently log the base pose; that's one extra line, and it's the field I most need.

Any format is fine — npz, CSV, pickle. Deterministic seed, no observation noise, start at motion
frame 0 if that's easy.

**Why this is decisive:** every term I compute matches holosoma's source as far as I can read it,
yet the closed loop still diverges within 0.5 s. Diffing your observations against mine at matched
states finds the discrepancy in one pass instead of by continued elimination.

## 2. v31's motion file

The reference exported in the policy npz is **734 frames**. It matches none of the clips I have —
`sub3_largebox_003_mj_w_obj.npz` is 434 frames, and the best alignment I can find is still ~103°
off, so v31's reference appears to have been re-authored.

Without it I also don't have the true object trajectory, so the box's start pose in my scene is
inferred: 45 cm cube, 0.40 m in front of the initial pelvis, validated only to ~5 cm against the
reference hand kinematics.

## 3. The `holosoma_config.yaml` from run `20260730_215012`

The config shipped beside the policy says `name: x2_box_v27_planted_feet_dr`, which looks like a
different run. Its gains are certainly right — I recovered them independently from
`action_scale = 0.25 · effort_limit / kp` to 0.0 error — but the motion file it names may not be
v31's.

The run directory is `20260730_215012-x2_box_v31_flatfoot-locomotion`. **"flatfoot"** suggests a
configuration detail I know nothing about, and it may matter.

## 4. A short video or screen capture of it working

Cheapest of all, and it settles a question I can't answer from here: what does "it works" look
like? Specifically — does the robot **lift the box and stand back up**, or does it complete the
motion without falling but without really picking the box up? My MuJoCo does the second: it goes
through the gestures, barely lifts the box (0.19 m against a reference of 0.70 m), and ends
collapsed.

---

## What I do NOT need

- the checkpoint — the exported npz evaluates correctly (replaying the real robot's logged state
  through it reproduces the robot's own commanded targets to 0.84° across 742 steps)
- the robot model or meshes
- the training code or any retraining

---

## Separately: a bug in `deploy_x2_box_pickup.py` worth fixing regardless

`base_ang_vel` is built from the **torso** IMU (`--base-imu`, default `torso`). holosoma trains it
from the **articulation root**:

```python
# holosoma/managers/observation/terms/wbt.py
def get_base_ang_vel(env):
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]      # ROOT body
    return quat_rotate_inverse(_base_quat(env), ang_vel_world)     # in the ROOT frame
# simulator/isaacsim/isaacsim.py:792
self.base_quat = self.robot_root_states[:, 3:7]                    # = root_state_w[3:7]
```

For `x2_31dof_w_object_halfspherehand` the root is **`pelvis`**. With a 3-DoF waist between pelvis
and torso the two are not interchangeable — I measure `|ω_torso − ω_pelvis|` at mean 2.3 rad/s,
max 8.4 rad/s, worst during the deep bend. Changing only that body in my sim takes survival from
148.8 to 472.2 steps.

Independent check against your own hardware logs: comparing my simulated angular velocity to the
logged IMU stream, the **pelvis** convention matches better than the torso one (RMS 0.50 vs 0.67
rad/s, correlation 0.83 vs 0.62 on the dominant axis).

**Fix:** add the pelvis IMU to `IMU_TOPICS` in `robot_states_control.py` and default `--base-imu`
to `pelvis`. Worth confirming the pelvis IMU's mounting frame matches the URDF `pelvis` link — a
fixed frame offset would reintroduce the same class of error.

**Do not change `motion_ref_ori_b`** — that term genuinely *is* torso-relative (it uses
`robot_ref_quat_w`, the ref body). The two observation terms reference different bodies, and that
is correct.

This is consistent with the robot having needed physical support during the hardware runs.

Full writeup: `BUG_REPORT_deploy_x2_box_pickup.md`.

---

Happy to share the MuJoCo reproduction, per-seed data, or anything else useful. Thanks!
