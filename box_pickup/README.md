# X2 Box Pickup — retargeted whole-body tracking policy

A whole-body loco-manipulation policy for the AgiBot X2: starting from the
robot's exact default upright pose it bends to a 45 cm box, squeeze-grasps it
with both (fingerless) palms, lifts it to chest height, carries it ~1.4 m
**straight ahead with feet pointing forward**, sets it down directly in front
over planted feet, and returns to the default upright pose — 8.7 s end to end.
Current model: **v24, iteration 89500** (`policy/model_89500.pt`).
See `videos/x2_box_v24_one_motion_drop_iter89000.mp4` for the latest rollout
and `videos/reference_motion_STRAIGHT_walk.mp4` for the reference it tracks.

Built on [amazon-far/holosoma](https://github.com/amazon-far/holosoma)
(IsaacLab whole-body tracking, PPO) with a human demonstration from the
[OmniRetarget dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)
(`sub3_largebox_003`) retargeted to the X2's 31-DOF skeleton.

## Contents

```
box_pickup/
├── policy/
│   ├── x2_box_policy.npz      # DEPLOYABLE: numpy-only policy + motion reference + metadata
│   ├── model_89500.pt         # raw holosoma checkpoint (v24 straight walk + clean set-down)
│   └── holosoma_config.yaml   # full training config snapshot of the v24 run
├── holosoma_overlay/          # every file added/changed in holosoma for this task
├── setup_holosoma_x2.sh       # clone holosoma + apply overlay + install meshes
├── render_box_rollout.py      # render recorded eval .npz rollouts to .mp4 (MuJoCo)
└── videos/                    # training progression, v4 (broken) -> final success
```

## Deploy on the real robot

Same pattern as the walking policy — see `../agibot_control_functions/`:

```bash
cd ../agibot_control_functions
# DRY RUN first (no commands published):
python3 deploy_x2_box_pickup.py --policy ../box_pickup/policy/x2_box_policy.npz
# Then escalate per the safety ladder printed by the script.
```

The policy is **blind** (no cameras): actor obs = reference-motion clock +
torso IMU + joint state + previous action. The box must be placed at the
reference start pose: 45 cm cube, on the floor, its **center ~0.40 m in front
of the robot** (near edge ~0.17 m from the feet), centered on its heading.
Training randomized box mass 0.4–1.6 kg and friction, so a LIGHT
(empty or lightly-filled) cardboard box is the right target — the X2's wrist
actuators cannot squeeze-hold a heavy box.

**Start the robot standing upright.** The reference motion begins AND ends at
the robot's exact default standing pose at zero velocity: the script ramps the
robot from wherever it is to that pose, holds it, then engages the policy.
Timeline of the 8.7 s motion once engaged: hold upright (0–1.0 s) → bend down
(1.0–2.5 s) → grasp + lift (2.5–3.0 s) → carry ~1.4 m STRAIGHT AHEAD at chest
height, feet pointing forward (3.0–5.8 s) → one continuous set-down directly
in front with both feet planted (5.8–8.0 s, the set-down is the pickup played
in reverse) → hold the default upright pose (8.0–8.7 s). Leave ~2 m of clear
floor in front of the robot for the carry.

## Retrain / evaluate in simulation

```bash
./setup_holosoma_x2.sh ~/holosoma      # clone + overlay + meshes
# install holosoma per its README (Isaac Sim + IsaacLab), then:
cd ~/holosoma
python src/holosoma/holosoma/train_agent.py exp:x2-31dof-wbt-w-object \
    logger:disabled --training.num-envs 4096 --training.name x2_box \
    --training.checkpoint <path>/model_89500.pt   # optional warm start

# record a clean demo rollout from a checkpoint (headless), then render it:
python src/holosoma/holosoma/eval_record_driver.py <ckpt.pt> /tmp/demo.npz 450 demo
python <this_repo>/box_pickup/render_box_rollout.py   # edit paths at top
```

## What was changed vs. upstream holosoma (the overlay)

- **X2 robot integration**: 31-DOF `RobotConfig` (joint limits, SDK PD gains,
  effort limits), URDF/XML variants incl. `x2_31dof_w_object_halfspherehand.urdf`
  (rigid half-sphere palms — the X2 has no articulated fingers, matching the
  real hardware and the G1 recipe upstream uses).
- **Task configs** (`config_values/wbt/x2/`): command/reward/termination/
  randomization/observation presets; experiment `exp:x2-31dof-wbt-w-object`.
- **Adaptive sampler fix** (`managers/command/terms/wbt.py`): the uniform
  mixing ratio was applied to raw failure counts (which scale with num_envs),
  making it a no-op; it now mixes normalized probabilities. Without this the
  sampler collapsed onto the grasp bin (top1 ≈ 0.95) and the policy
  catastrophically forgot the walk-up phase.
- **Grasp shaping reward** (`managers/reward/terms/wbt.py`):
  `hands_to_object_distance_exp` — product of per-hand proximity terms, so BOTH
  palms must reach the box surface. The object-tracking rewards alone are
  flat-zero once the box is left behind, which let the policy "pantomime" the
  carry; a mean over hands let one touching palm mask the other hovering away.
  This term is what finally made the pickup happen.
- **Retargeter joint-limit fix** (`interaction_mesh_retargeter.py`): X2's
  named freejoint caused an off-by-one in the joint-limit array — every joint
  was clamped to the PREVIOUS joint's limits (frozen left hip, impossible
  waist bends, the awkward left arm). Limits are now indexed by joint type.
- **Retargeter warm-up** (`examples/robot_retarget.py`): the per-frame
  optimizer needs ~15 frames to converge onto the human, which used to bake a
  violent fake transient into the clip start (root spinning at 4.6 rad/s at
  frame 0). The first human frame is now repeated ~20x as a warm-up and those
  frames discarded, so the clip starts at the true upright standing pose with
  zero velocity.
- **`eval_record_driver.py`**: headless rollout recorder with a `demo` mode
  (start at t=0, no init noise, no early termination) for clean full-motion
  videos and hardware-comparable trajectories.

## Training history (see videos/)

| Run | What happened |
|-----|---------------|
| v4–v5 | GPU contention fixed; basic tracking learned |
| v6–v7 | physics/DR fixes; box friction/mass made liftable; sampler collapse diagnosed |
| v8 | sampler fix → full-motion balance, but "pantomimes" the lift |
| v9 | hand-to-box proximity reward → fights for the box, one-handed touch |
| v10 | two-hand requirement → full pickup + carry + set-down (first success in sim) |
| v11 | hardware-robustness DR (PD gain/delay/encoder noise), smoother actions |
| v12–v13 | retargeter joint-limit bug found and fixed; retrained on corrected reference |
| v14 | loosened termination thresholds → survives imperfect grasps, learns stand-up |
| v15 | trimmed optimizer transient from clip start (fixed sideways fall at spawn) |
| v16 | retarget warm-up → reference starts fully upright at zero velocity |
| v17 | reference starts at the exact default pose; joint-space tracking reward vs carry crab-walk |
| v18 | low-pass filtered reference (carry jitter −31%); stronger action-rate penalty |
| v19 | reference ends back at the default upright pose (`model_79000.pt`, first hardware tests) |
| v20 | carry straightened: the human's ~65° turn removed, hip-yaw splay compressed so feet point forward |
| v21 | feet bolted during set-down/stand-up (mocap human shuffled feet 30 cm while placing the box) |
| v22–v23 | set-down replaced with the time-reversed pickup: box lowered straight ahead, quiet waist |
| v24 | set-down made one continuous motion (no pause / re-lift) → **`model_89500.pt`** (current) |
