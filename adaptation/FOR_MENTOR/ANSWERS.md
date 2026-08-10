# Answers to WHAT_I_NEED_FROM_ISAAC.md — v31 box pickup

All four items are here, plus confirmation of your `base_ang_vel` bug report.

**The single most likely cause of your divergence is item 2: you are simulating against the
wrong reference clip.** v31's config sets `motion_dir`, not `motion_file`, and `motion_dir`
takes precedence.

| file | what it is |
|---|---|
| `isaac_v31_rollout.npz` | the rollout log, 734 steps, per-term observations + base pose |
| `isaac_v31_rollout.mp4` | video of exactly that rollout |
| `isaac_v31_reference_vs_actual.mp4` | reference motion and the policy side by side, same camera |
| `v31_reference_box_speed100.npz` | **v31's actual reference motion, 734 frames** |
| `v31_reference_as_rollout.npz` | same clip in the rollout schema (base + 31 joints unpacked) |
| `holosoma_config_v31_20260730_215012.yaml` | the real config for the run |
| `dump_for_mentor.py`, `postprocess_dump.py`, `make_reference_rollout.py` | what produced the above |

---

## 1. The rollout log

`isaac_v31_rollout.npz`. Deterministic (seed 42), observation noise off, starts at motion
frame 0, `bad_tracking` dropped. **It ran all 734 steps with no termination of any kind.**

Per-term observations, exactly as you asked, all `(734, d)`:

| array | dim |
|---|---|
| `obs_aligned__actions` | 31 |
| `obs_aligned__base_ang_vel` | 3 |
| `obs_aligned__dof_pos` | 31 |
| `obs_aligned__dof_vel` | 31 |
| `obs_aligned__motion_command` | 62 |
| `obs_aligned__motion_ref_ori_b` | 6 |
| `actor_obs_aligned` | 164 (the six above concatenated alphabetically) |

State channels, all `(734, ...)`: `actions`, `dof_pos`, `dof_vel`, `dof_pos_target`, `torques`,
**`root_pos`**, **`root_quat_xyzw`**, `root_lin_vel`, `root_ang_vel`, `body_pos_w` (32 bodies),
`body_quat_xyzw`, `object_pos`, `object_quat_wxyz`, and `*_substep` variants at the 200 Hz
physics rate.

Base pose was already being recorded by `EvalRecordingCallback` — your copy of the overlay
predates that. No extra line was needed.

### Alignment — please read before diffing

The raw trace has **738** rows against 734 steps, because the observation manager also runs
during reset/warmup. The offset was measured, not assumed:

- `obs__actions[4+i] == actions[i]` exactly (mean |diff| = 0.0)
- `obs__dof_pos[4+i] - dof_pos[i]` is constant, residual std 6e-7 (= `default_joint_pos`)

Row `r` therefore carries `last_action = actions[r-4]`, so the observation that **produced**
`actions[j]` is row `j+3`. The `obs_aligned__*` arrays already have this applied:

```
obs_aligned__X[j]  is the policy input that produced  actions[j]
                   and is built from the state BEFORE step j (i.e. dof_pos[j-1])
```

The raw `obs__*` arrays are kept alongside if you want to check the trimming yourself.

### One gotcha your script would have hit

`box_eval_isaac.py` does not pin the motion clip. Because v31 sets `motion_dir`, eval hands the
recorded env a **random one of three clips**, so the log would not have been reproducible. The
driver here sets `motion_file` and clears `motion_dir`.

---

## 2. v31's motion file — this is the one that matters

**`box_multispeed/box_speed100.npz`, 734 frames, 50 fps.** Attached as
`v31_reference_box_speed100.npz`. It matches the 734-frame reference exported in the policy npz.

Why you could not find it: the config has both fields set, and

```
motion_dir: .../whole_body_tracking/box_multispeed     <-- this one wins
motion_file: .../whole_body_tracking/sub3_largebox_003_mj.npz
```

From `config_types/command.py`: *"Directory (or comma-separated directories) of .npz motion
files. When non-empty, takes precedence over motion_file."* Confirmed in
`managers/command/terms/wbt.py`, which builds a `MultiMotionLoader` whenever `motion_dir` is set
and ignores `motion_file` entirely. The `motion_file` line is a stale leftover.

v31 therefore trained on all three clips in that directory:

| clip | frames |
|---|---|
| `box_speed080.npz` | 879 |
| **`box_speed100.npz`** | **734** |
| `box_speed125.npz` | 617 |

`sub3_largebox_003_mj_w_obj.npz` is 584 frames in our tree today (404 in the `_v25_backup`), and
is not what v31 used. Your 434 number matches neither, so that clip has been re-authored more
than once — another reason to switch to the attached file.

### Two array-layout traps in this clip — please read, one of these may be your bug

**1. `joint_pos` is `(734, 38)`, not 31 wide. The first 7 columns are the floating base**
(position 3 + quaternion wxyz 4), and the 31 joints follow. `joint_names` has 31 entries and
names the **joint part only**, so `joint_names[i]` is column `7+i`.

Verified: `joint_pos[:, 7:38]` equals the policy npz's exported `ref_joint_pos` **exactly**
(max abs diff 0.0 across all 734 frames). Zipping `joint_names` against columns 0..30 instead
shifts every joint by seven and gives max abs diff 2.44 rad — a robot doing confidently wrong
things, which is roughly what you describe seeing. `joint_vel` is `(734, 37)` = 6 base + 31,
same idea.

**2. `body_pos_w` and `body_quat_w` are PELVIS-RELATIVE despite the `_w` suffix.** Body 0
(pelvis) is exactly `[0,0,0]` / identity at every frame. The world base pose lives in
`joint_pos[:, :7]`, and it agrees with Isaac's actual spawn at frame 0 to **6.4 mm**.

`make_reference_rollout.py` unpacks all of this correctly if you want a worked example, and
`v31_reference_as_rollout.npz` is the result.

### Box placement — your inference was right

Box **center** in the pelvis frame at frame 0:

| source | x (forward) | y (lateral) | z | horizontal |
|---|---|---|---|---|
| Isaac rollout | 0.3915 | 0.0269 | -0.4853 | **0.3924 m** |
| reference clip | 0.3989 | 0.0208 | -0.4855 | **0.3995 m** |

Your 0.40 m is correct to within 1 cm, and the clip's `object_pos_w` agrees — it **is** usable,
in the same world frame as `joint_pos[:, :3]`. The clip's box goes 0.184 → 0.830 m → 0.184 m,
staying 0.36–0.45 m from the pelvis throughout.

---

## 3. The config

`holosoma_config_v31_20260730_215012.yaml`, copied from the run directory. Yours is a different
run — it says `name: x2_box_v27_planted_feet_dr`; the real one says `name: x2_box_v31_flatfoot`.
Your gains were right anyway, but the motion fields in your copy were not.

On **"flatfoot"**: it is the ankle/foot-contact reward and termination shaping that keeps both
feet planted through the bend. It is in the attached config; nothing about it is hidden in code.

Also worth noting from the config: `control_decimation: 4`, `fps: 200` (so 50 Hz control, matching
your assumption) and `max_episode_length_s: 20.0`.

---

## 4. What "it works" looks like

**It genuinely lifts the box and stands back up.** Two videos:

- `isaac_v31_rollout.mp4` — the rollout on its own, exactly matching the npz.
- `isaac_v31_reference_vs_actual.mp4` — the retargeted reference on the left, the policy on the
  right, same camera and same clock. This is probably the more useful one: it shows the policy
  tracking with a visible forward lean and a lower box carry than the reference, but completing
  the whole motion.

From the log:

| | value |
|---|---|
| box z at start | 0.184 m |
| box z at peak | **0.751 m** (step 224, 4.5 s) |
| carried at 0.68–0.75 m | steps 180–360 (3.6–7.2 s) |
| box set down, under control | ~step 420 (8.4 s), back to 0.181 m |
| pelvis height | 0.663 → 0.506 (bend) → 0.667 (standing again) |
| box-to-pelvis distance | 0.400 m at start, 0.402 m at end — never leaves the hands |
| termination | **none, ran all 734 steps** |

So the answer to your question is the first option, not the second: it reaches 0.75 m against a
reference of 0.70, holds it, sets it down deliberately, and returns to standing. Your MuJoCo's
0.19 m lift and collapse is a real behavioural gap, not a metric artifact.

### One thing that surprised us and may matter to you

Observation noise materially changes the outcome. Our earlier adaptation runs left the training
observation noise **on** during eval, and there the frozen policy dropped the box on 3 of 5 seeds
and sometimes fell. With noise off — as in this log, and as in your `box_eval_isaac.py` — it
completes the task cleanly every time. If you are comparing against any earlier numbers we sent,
that flag is the difference.

---

## 5. Your `base_ang_vel` bug report — confirmed

You are right, and it reproduces in our source.

```python
# holosoma/managers/observation/terms/wbt.py
def get_base_ang_vel(env):
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)
```

and `body_names[0] == 'pelvis'` in the recorded metadata, so `robot_root_states` is the **pelvis**.
Training used the pelvis. Meanwhile `deploy_x2_box_pickup.py` has
`--base-imu default="torso", choices=["torso","chest"]`, and its own docstring mispositions it as
*"training ref body = torso_link"*. That comment is wrong for this term.

Your point about `motion_ref_ori_b` being genuinely torso-relative also holds — it uses the ref
body, which is `torso_link`. The two terms reference different bodies and that is correct.

**One blocker on the fix as written.** `IMU_TOPICS` in `robot_states_control.py` currently has
only `chest` and `torso`:

```python
IMU_TOPICS = {
    "chest": "/aima/hal/imu/chest/state",
    "torso": "/aima/hal/imu/torso/state",
}
```

There is no pelvis entry, and we have not confirmed the robot publishes a pelvis IMU at all. So
"add the pelvis IMU and default to it" needs one prior step: check whether a pelvis IMU topic
exists on the hardware. If it does not, the options are to derive pelvis angular velocity from
the torso IMU through the 3-DoF waist joint states, or to retrain the term against the torso.
Happy to check the topic list on the next hardware session.

---

## Reproducing this

```bash
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 python adaptation/dump_for_mentor.py
python adaptation/postprocess_dump.py
python adaptation/make_reference_rollout.py
python box_pickup/render_side_by_side.py \
  adaptation/FOR_MENTOR/v31_reference_as_rollout.npz "REFERENCE (retargeted motion)" \
  adaptation/FOR_MENTOR/isaac_v31_rollout.npz "ISAAC ACTUAL (frozen v31 policy)" \
  adaptation/FOR_MENTOR/isaac_v31_reference_vs_actual.mp4
```
