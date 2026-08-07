# Matching two environments exactly — what to capture from the working setup

Context: the frozen box-pickup policy reportedly completes the pickup on setup **B** (the working
one) but falls at ~2 s on setup **A** (this package). Something differs. This lists exactly what
is needed from B to find it, ordered by how much each item narrows the search.

---

## If you send only ONE thing

**A per-step log of one rollout from B**, containing for every control step:

| field | shape | why |
|---|---|---|
| `obs` | (T, 164) | the policy input — the single most diagnostic item |
| `action` | (T, 31) | raw network output, before filtering |
| `target` | (T, 31) | joint target actually commanded |
| `qpos` | (T, nq) | full state incl. root position + quaternion |
| `qvel` | (T, nv) | full velocity |
| `frame` | (T,) | which reference frame index was used at that step |

`run_dump.py` in this folder produces exactly this from a MuJoCo-based env.

With that I can replay B's observations through the policy here. Two outcomes, both decisive:

- **Same actions from same obs** → the policy code matches; the difference is in how the
  observation is *built from state*, or in the plant. The first step where `obs` diverges from
  mine, given matching `qpos`, localises the bug to one observation block.
- **Different actions from same obs** → different weights, different normaliser, or a different
  policy file entirely.

Even **the first 50 steps** is enough — the divergence starts well before the fall.

---

## Tier 1 — identity (cheap, rules out whole classes of difference)

1. **Repo + commit + branch** for the code that runs the rollout.
2. **Exact command line**, including task id / config name if applicable.
3. **Policy file**: absolute path, `sha256`, file size. Also its embedded `meta_json`
   (`run_path`, `motion_frames`, `control_hz`, `action_scale`, `joint_stiffness`,
   `default_joint_pos`).
   → This package uses `x2_box_policy_v31.npz`, whose `meta_json.run_path` is
   `20260730_215012-x2_box_v31_flatfoot-locomotion/model_202500.pt`.
   **If B's policy differs, nothing else matters — start here.**
4. **Reference motion file**: path, `sha256`, number of frames, fps.
   → This package uses the 734-frame reference baked into the `.npz`. The repo also contains a
   *different* 434-frame clip (`sub3_largebox_003_mj_w_obj.npz`). If B tracks a different clip,
   that alone explains everything.
5. **Is the box present in B's scene?** Yes/no, and is it a physical body or only a reward target?

## Tier 2 — the plant

6. **Robot model file** (MJCF/URDF/USD) + `sha256`.
7. **Model summary**: `nq`, `nv`, `nu`, total mass, `opt.timestep`, solver, iterations, gravity.
8. **Joint dynamics arrays**: `dof_armature`, `dof_damping`, `dof_frictionloss` (all DoF).
   → A differs from mjlab here: armature 0.000 vs 0.030, frictionloss 0.000 vs 0.300.
9. **Actuator setup**: type (motor / position / general), `gaintype`/`biastype`,
   `gainprm`, `biasprm`, `forcerange` per actuator.
   → Position-servo vs torque-motor semantics are easy to get wrong: writing a joint *angle*
   into a torque actuator's `ctrl` silently produces near-zero torque.
10. **Contact**: ground friction, foot geom count / radii / positions, `impratio`, `cone`,
    `solref`/`solimp`.

## Tier 3 — the control loop

11. **Control rate** and **decimation** (physics steps per control step).
12. **kp / kd actually applied**, including any gain scale (this package uses ×1.2).
13. **Action → target mapping**, verbatim. Specifically:
    - is there a low-pass filter on targets, on which joints, with what coefficient?
      (this package: 0.8 on leg joints only)
    - is there a per-step rate limit? (this package: 0.15 rad/step, all joints)
    - `target = action * action_scale + default_joint_pos`, or something else?
14. **Torque clipping** limits, if any.

## Tier 4 — observation construction (the leading suspect)

15. **Term order** in the 164-vector. This package uses ALPHABETICAL:
    `actions(31) | base_ang_vel(3) | dof_pos(31) | dof_vel(31) | motion_command(62) | motion_ref_ori_b(6)`
16. **`base_ang_vel`**: which body, and in which frame? This package uses the **torso** IMU gyro
    in the **torso body frame**. Note `mj_objectVelocity(..., flg_local=1)` returns the *inertial*
    frame, which is **not** the body frame — an easy and silent error.
17. **`motion_ref_ori_b`**: how is it formed? This package uses the first two columns of the
    rotation matrix of `conj(q_torso) ⊗ q_ref`, row-major.
18. **Yaw alignment**: is the reference rotated into the robot's heading at engage
    (`yaw_offset = yaw(q_torso_0) ⊗ inv(yaw(q_ref_0))`), or is the robot spawned at the
    reference's heading, or neither?
19. **Reference time indexing**: does the frame index advance one per control step? Is there
    interpolation? Any hold/pause phase before the motion starts?
20. **`dof_pos` convention**: `q − default_joint_pos`, or raw `q`?

## Tier 5 — initial state

21. **Full `qpos` and `qvel` at reset** (one array each). This removes all ambiguity about
    spawn height, orientation, and initial pose.
22. **How the robot is placed**: dropped to contact? fixed height? feet pinned?
23. **Initialisation noise**: distribution and magnitude, and whether B uses any.

## Tier 6 — outcome, for calibration

24. Does B's frozen policy complete the **whole** motion, or just get further?
25. Over how many seeds/trials, and what fraction succeed?
26. A video, if one exists.

---

## Why this matters beyond a bug hunt

If B's frozen policy succeeds, then the baseline in this package is failing for a reason unrelated
to the policy, and every adaptation number here was measured against a broken baseline. The
adaptation study would need re-running before any of its results are used. So it is worth
resolving before building anything further on these numbers.
