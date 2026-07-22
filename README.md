# Agibot-humanoid

End-to-end stack for training and deploying policies on the **AgiBot X2** humanoid.
Two tasks are included, each with simulation training setup, a trained policy, and an
on-robot deployment script over the ROS 2 control interface:

1. **Velocity walking** — trained in MuJoCo / [mjlab](https://github.com/mujocolab/mjlab).
2. **Box pickup (retargeted whole-body tracking)** — from an upright standing start,
   bends to a 45 cm box, two-handed squeeze grasp, lifts it to chest height, carries
   it ~1.4 m straight ahead (feet pointing forward), sets it down directly in front
   over planted feet, and returns upright (8.7 s). Trained in IsaacLab /
   [holosoma](https://github.com/amazon-far/holosoma) by tracking a human
   demonstration from the OmniRetarget dataset retargeted onto the X2. See
   `box_pickup/README.md` and `box_pickup/videos/x2_box_v24_one_motion_drop_iter89000.mp4`.

```
Agibot-humanoid/
├── mjlab/                      # Walking training framework (MuJoCo-Warp + rsl_rl PPO)
│   └── src/mjlab/
│       ├── asset_zoo/robots/x2/            # X2 MJCF, meshes, actuators, keyframe
│       └── tasks/velocity/config/x2/       # X2 velocity-walk task + deploy variant
├── box_pickup/                 # Box-pickup task (holosoma whole-body tracking)
│   ├── policy/x2_box_policy.npz            # deployable policy (numpy-only inference)
│   ├── policy/model_89500.pt               # current checkpoint (v24 straight walk + clean set-down)
│   ├── holosoma_overlay/                   # all holosoma changes for the X2 task
│   ├── setup_holosoma_x2.sh                # recreate the training setup anywhere
│   └── videos/                             # training progression -> final success
└── agibot_control_functions/   # Real-robot deployment (run this on the robot)
    ├── deploy_x2_walk.py       # Walking: loads policy, reads state, walks (50 Hz)
    ├── deploy_x2_box_pickup.py # Box pickup: plays the 8.7 s pickup motion (50 Hz)
    ├── release_joints.py       # Ramp all joint gains to zero so the robot can be moved by hand
    ├── export_policy_npz.py    # Walking: trained .onnx -> self-contained .npz
    ├── export_box_policy_npz.py# Box pickup: holosoma .pt -> self-contained .npz
    ├── robot_states_control.py # ROS 2 RobotStateClient + WholeBodyCommander (provided API)
    ├── robot_states_control.cpp
    ├── README.md               # The robot's state/command API reference
    └── policies/
        ├── x2_policy_original.npz   # Trained walking policy (numpy-only inference)
        └── x2_policy_original.onnx  # Same policy in ONNX form
```

---

## 1. Run on the real robot (the part you do over Ethernet)

The deployment script runs **inside the robot's ROS 2 environment** and needs only
`numpy` plus the ROS packages the robot already provides (`rclpy`, `aimdk_msgs`,
`sensor_msgs`). No torch / onnxruntime required — inference is a tiny numpy MLP.

```bash
# On the machine connected to the robot (sourced ROS 2 workspace):
cd agibot_control_functions

# 1) DRY RUN — computes & prints commands but does NOT move the robot:
python3 deploy_x2_walk.py --policy policies/x2_policy.npz

# 2) Once the dry-run output looks sane AND the robot is safe, engage:
python3 deploy_x2_walk.py --policy policies/x2_policy.npz \
        --engage --vx 0.3 --run-seconds 8

# (policies/x2_policy_original.npz is the older model that needs base_lin_vel;
#  prefer policies/x2_policy.npz for ground walking.)
```

What the script does each 50 Hz tick: reads IMU + joint state → builds the exact
observation the policy was trained on → runs the policy → `target_q = action ·
action_scale + default_q` → publishes per-area position commands with the trained PD
gains. The observation/joint/scale/gain layout is read from the policy metadata, so the
script stays correct if you retrain.

### ⚠️ Safety — read before `--engage`
1. **Robot fully SUSPENDED** (feet off the ground) for the first runs.
2. **Stop the motion controller first:** `aima em stop-app mc` (on `10.0.1.40`).
3. Default is **dry-run**; `--engage` is required to actually publish commands.
4. Escalate: dry-run → suspended `--engage` → harness/gantry on ground → free walking.
5. Keep the **e-stop** in hand. `Ctrl+C` ramps back to the default pose and exits.
6. When done: `aima em start-app mc`.

Useful flags: `--base-imu {torso,chest}`, `--vx/--vy/--wz` (command), `--gain-scale`
(lower = gentler PD), `--run-seconds`, `--max-joint-step`, `--tilt-abort`.

---

## 2. About the policies

| Policy | Obs | Status | Notes |
|--------|-----|--------|-------|
| `x2_policy_original.npz` | 105-dim (**includes `base_lin_vel`**) | trained 10k iters | Ready to run. The robot cannot measure base linear velocity, so the script feeds **zeros** for it — valid while suspended/near-stationary, degraded for free ground walking. |
| `x2_policy.npz` (deploy) | 102-dim (**no `base_lin_vel`**) | trained 10k iters | **Recommended for ground walking.** Depends only on on-board sensing (ang vel, projected gravity, joint state, command). |

The deploy variant removes `base_lin_vel` from the *actor* observation (keeping it in the
critic), so the policy depends only on on-board sensing — the standard sim-to-real fix.

Other known sim-to-real gaps to expect: training IMU was on the **pelvis** while the
hardware IMUs are on **chest/torso**; flat-ground, feet-only-collision training with light
randomization. Tune `--gain-scale` and `--vx` conservatively on first ground tests.

---

## 3. Train / retrain in simulation (mjlab)

Requires an NVIDIA GPU. mjlab uses [uv](https://docs.astral.sh/uv/) for dependencies.

```bash
cd mjlab
uv sync   # creates .venv and installs deps (first run downloads CUDA torch + mujoco-warp)

# Train the deployment-friendly policy (no base_lin_vel in actor obs):
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-X2-Deploy \
    --env.scene.num-envs 4096 --agent.max-iterations 10000 --agent.logger tensorboard

# Resume from a checkpoint (additional iterations):
uv run train Mjlab-Velocity-Flat-X2-Deploy --env.scene.num-envs 4096 \
    --agent.max-iterations 7400 --agent.resume True \
    --agent.load-run "<run_dir>" --agent.load-checkpoint "model_2600.pt"

# Visualize a checkpoint in the browser (Viser): open the Twist folder -> Enable -> set lin_vel_x:
LATEST=$(ls -t logs/rsl_rl/x2_velocity_deploy/*/model_*.pt | head -1)
uv run play Mjlab-Velocity-Flat-X2-Deploy --viewer viser --num-envs 1 --checkpoint-file "$LATEST"
```

Registered X2 tasks: `Mjlab-Velocity-Flat-X2`, `Mjlab-Velocity-Rough-X2`,
`Mjlab-Velocity-Flat-X2-Deploy`.

### Convert a trained policy for the robot
mjlab exports an `.onnx` next to the checkpoints. Convert it to the numpy `.npz`:

```bash
cd mjlab
RUN=$(ls -td logs/rsl_rl/x2_velocity_deploy/*/ | head -1)
uv run python ../agibot_control_functions/export_policy_npz.py \
    --onnx "$RUN"/$(basename "$RUN").onnx \
    --out  ../agibot_control_functions/policies/x2_policy.npz
```

Then deploy with `--policy policies/x2_policy.npz`.

---

## 4. Box pickup (retargeted whole-body tracking)

The second task: a single policy that tracks a retargeted human demonstration of a full
box pickup — walk up, deep bend, two-handed squeeze grasp (the X2 has rigid fingerless
palms, so it hugs the box between them), lift to chest, short carry, controlled set-down.
Trained with holosoma's whole-body tracking PPO on IsaacLab, 30k iterations, with domain
randomization on box mass (0.4–1.6 kg), friction, and initial pose.

```bash
# Deploy (inside the robot's ROS 2 env; numpy-only, same pattern as walking):
cd agibot_control_functions
python3 deploy_x2_box_pickup.py --policy ../box_pickup/policy/x2_box_policy.npz          # dry run
python3 deploy_x2_box_pickup.py --policy ../box_pickup/policy/x2_box_policy.npz --engage # for real
```

Key differences from the walking deployment:
- The policy is **blind** — it tracks a fixed 6.5 s reference motion (embedded in the
  `.npz`) and needs the box placed at the reference start: 45 cm cube on the floor,
  ~0.55 m in front of the robot, centered. First runs: **suspended, no box**.
- The safety abort watches **roll only** (`--roll-abort`); pitch is part of the motion
  (deep forward bend), unlike walking where any large tilt is a failure.
- Full training/retraining instructions, the holosoma overlay, and the training story
  are in **`box_pickup/README.md`**.

---

## Attribution
- `mjlab/` is derived from [mujocolab/mjlab](https://github.com/mujocolab/mjlab) (training
  outputs, virtualenv, and W&B logs are intentionally not committed — see `.gitignore`).
- `agibot_control_functions/` builds on the AgiBot control API from
  [YujinAnn/agibot_control_functions](https://github.com/YujinAnn/agibot_control_functions).
- `box_pickup/` builds on [amazon-far/holosoma](https://github.com/amazon-far/holosoma)
  (whole-body tracking framework) and the
  [OmniRetarget dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)
  (`sub3_largebox_003` demonstration).
