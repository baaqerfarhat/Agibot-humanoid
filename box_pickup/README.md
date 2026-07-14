# X2 Box Pickup — retargeted whole-body tracking policy

A whole-body loco-manipulation policy for the AgiBot X2: from a standing start it
walks to a 45 cm box, bends, squeeze-grasps it with both (fingerless) palms,
lifts it to chest height, carries it a few steps, and sets it back down — 6.5 s
end to end. See `videos/x2_box_FINAL_model30000.mp4`.

Built on [amazon-far/holosoma](https://github.com/amazon-far/holosoma)
(IsaacLab whole-body tracking, PPO) with a human demonstration from the
[OmniRetarget dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)
(`sub3_largebox_003`) retargeted to the X2's 31-DOF skeleton.

## Contents

```
box_pickup/
├── policy/
│   ├── x2_box_policy.npz      # DEPLOYABLE: numpy-only policy + motion reference + metadata
│   ├── model_30000.pt         # raw holosoma checkpoint (30k iterations, final)
│   ├── model_30000.onnx       # holosoma's ONNX export of the same checkpoint
│   └── holosoma_config.yaml   # full training config snapshot of the final run
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
reference start pose: 45 cm cube, on the floor, ~0.55 m in front of the robot,
centered on its heading. Training randomized box mass 2.4–12 kg and friction,
so a mid-weight cardboard box is the easiest first target.

## Retrain / evaluate in simulation

```bash
./setup_holosoma_x2.sh ~/holosoma      # clone + overlay + meshes
# install holosoma per its README (Isaac Sim + IsaacLab), then:
cd ~/holosoma
python src/holosoma/holosoma/train_agent.py exp:x2-31dof-wbt-w-object \
    logger:disabled --training.num-envs 4096 --training.name x2_box \
    --training.checkpoint <path>/model_30000.pt   # optional warm start

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
| v10 | two-hand requirement → **full pickup + carry + set-down** (`model_30000.pt`) |
