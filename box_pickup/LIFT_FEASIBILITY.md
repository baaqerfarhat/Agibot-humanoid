# Why the lift fails on hardware, and how to fix it

The v33 box pickup completes in Isaac and topples on the robot. This is not a tuning
problem and not an adaptation problem: **the policy commands hip torque the actuator
cannot deliver, nothing in the reward ever charged it for that, and simulation's contact
covers the shortfall so it never had to learn otherwise.**

Everything below is measured. Reproduce commands are at the end.

---


> **Correction (2026-08-18).** The premise below — that the hips command torque the
> actuator cannot deliver — is wrong, and the fix it motivated has been replaced.
> Reconstructing delivered torque from the recorded rollouts
> (`tau = kp*(default + a*scale - q) - kd*qd`, clipped as `clip_torques=True` does)
> shows **hip_pitch never exceeds 0.95x of its 120 N-m limit and the knees never pass
> 0.29x**. `|a| = 4` equals the effort limit only if the joint stays pinned at its
> default; the hip reaches `|a| = 9.12` while delivering 34.8 N-m, because it tracks
> its target to within 0.19 rad. **Action magnitude is not a torque criterion.**
>
> The joints that actually saturate are the three small ones: waist_pitch (48 N-m,
> 1.82x demand / 11% of the baseline episode), ankle_pitch (36) and ankle_roll (24,
> 1.99x / 22%). Worse, ankle_roll degrades monotonically as the hip-targeted penalty
> trains — 1.99x -> 4.17x -> 7.61x — because the policy sheds hip command and the
> load lands on the smallest actuator on the robot.
>
> `penalty_action_over_effort` is therefore replaced by
> `penalty_joint_torque_saturation`, which charges on delivered torque against each
> joint's own limit, masked to `waist_pitch,ankle_pitch,ankle_roll`. Full audit and
> seed-averaged results: `box_pickup_results/README.md`.


## 1. The chain

| # | measurement | value |
|---|---|---|
| 1 | reference pelvis travel over the whole pickup | **18 mm** |
| 2 | pelvis travel the policy actually uses | **100 mm** |
| 3 | hip-pitch command during the rise (frames 110–150) | mean **6.2**, peak **9.2** |
| 3 | …against the effort limit | **4.0** |
| 4 | ankle-roll deflection under the same capped command | sim **0.036 rad**, hardware **0.30 rad** |

**The reference is an arm-dominated human lift.** The demonstrator kept their hips high —
the pelvis moves 18 mm across the entire motion while the hands travel 0.36 → 0.70 m and
the box goes 0.184 → 0.826 m. The X2's arms cannot do that, so the learned policy
substitutes leg work: it squats 100 mm, **8 cm below the reference** (0.579 m vs 0.658 m).

**That substitution is what saturates the hips.** Squatting deeper pitches the torso mass
ahead of the hip axis and lengthens the moment arm. `action_scale = cfg_scale · effort / kp`,
so `|a| = 4` *is* the effort limit by construction; the policy asks for up to **9.2**, i.e.
2.3× the available torque, for **52–75%** of the rise — in simulation and on hardware alike.

**Simulation hides the consequence.** Replaying the robot's own recorded actions in Isaac,
sim tracks its own targets *worse* than hardware does (8.08° vs 4.84°) because its contact
resists harder. Sim's floor supplies what the hips cannot; the real floor yields instead,
the feet roll onto their edges, and the motion fails.

Two things this is **not**:

- **Not payload.** The box is empty. It is also held slightly *closer* than the reference
  asks (0.444 m vs 0.469 m from the pelvis), so box placement is not the lever.
- **Not timing.** Running the 0.8× speed clip changes peak hip demand by **−3%**
  (10.48 → 10.14). The demand is gravity-dominated, not inertial, so no retiming helps.

---

## 2. The gap in the reward

The v31/v33 reward set contains:

```
action_rate_l2   -1.0     # penalises action CHANGES
limits_dof_pos  -10.0     # penalises POSITION limits
```

There is **no effort or action-magnitude penalty anywhere in holosoma**. A saturated
command is free. Combined with contact that covers the shortfall, the policy has no
gradient telling it the torque is unavailable.

Clipping the action at deploy (`--action-clip`) treats the symptom: the policy still
*requests* 9.2 and simply receives 4.0. It does not change the learned strategy.

---

## 3. The fix

### 3.1 Add the effort penalty

`penalty_action_over_effort` is in the overlay at
`src/holosoma/holosoma/managers/reward/terms/wbt.py`, and is wired into
`x2_31dof_wbt_reward_w_object` — the preset the box-pickup experiment uses. It
penalises only the **excess** beyond the limit, so the policy keeps the full actuator
envelope and only the impossible part becomes expensive.

```python
"penalty_action_over_effort": RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:penalty_action_over_effort",
    params={"limit": 4.0, "joints": "hip_pitch", "ramp_steps": 400_000},
    weight=-0.5,
),
```

`joints` restricts it to the measured bottleneck; penalising every joint taxes ones
that legitimately run near their limit.

`ramp_steps` fades the penalty in linearly over that many environment steps. This
matters when **warm-starting**: a policy already sitting at `|a| ~ 9` receives
`(9.2 - 4)^2 ~ 27` per hip the instant the term switches on, which is a large negative
advantage on a previously-optimal policy. PPO handles that badly and can collapse a
good policy in a few hundred iterations. 400k steps is ~100 iterations at 4096 envs.

### 3.2 Keep training under the deploy bound

Keep the action clip enabled during training so the policy experiences the same command
path it deploys with. Clip and penalty do different jobs: the clip makes training and
deployment consistent, the penalty changes what the policy learns to ask for.

### 3.3 If the pelvis stays low, weight the reference position term up

The policy squats 8 cm below the reference. `motion_global_ref_position_error_exp` is at
**0.5**, low against the 1.0–3.0 on the other tracking terms. Raising it pulls the pelvis
back toward the demonstrated height, shortening the hip moment arm.

---

## 4. Running the retrain

Warm-start rather than training from scratch. The task has not changed — grasping,
balance and tracking are all intact; what has to change is one postural choice inside a
40-step window. Relearning the rest would waste the 200k+ iterations already spent.

```bash
cd ../holosoma
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 \
  python src/holosoma/holosoma/train_agent.py exp:x2-31dof-wbt-w-object logger:disabled \
  --training.num-envs 4096 --training.name x2_box_effort_ft \
  --training.checkpoint <path>/model_202500.pt
```

**Host caveat — cuDNN.** On a driver-535 / CUDA-12.2 host the cuDNN 9.2 bundled with the
torch cu128 wheels raises `CUDNN_STATUS_NOT_INITIALIZED` on any cuDNN call. Training hits
it through a 1-D smoothing `conv1d` in the adaptive timestep sampler; evaluation never
does, because the eval harness pins `use_adaptive_timesteps_sampler=False`. Setting
`torch.backends.cudnn.enabled = False` before importing the trainer fixes it at no cost —
that conv is tiny and the fallback kernel is fine.

## 5. Pass criterion

**Hip-pitch `|a| < 4.0` through frames 110–150.** It currently peaks at 9.2. One
measurement in simulation decides whether the retrain worked — before any hardware time.

```python
import numpy as np, json
z = np.load("adaptation/isaac_runs/<run>/isaac_frozen_npz_seed600.npz", allow_pickle=True)
m = json.loads(str(np.load("box_pickup/policy/<policy>.npz", allow_pickle=True)["meta_json"]))
HIP = [m["joint_names"].index(f"{s}_hip_pitch_joint") for s in ("left", "right")]
a, f = z["actions"], z["frame"]
rise = (f >= 110) & (f < 150)
print("mean", np.abs(a[rise][:, HIP]).mean(), "peak", np.abs(a[rise][:, HIP]).max())
```

---

## 6. What not to do

| approach | why not |
|---|---|
| slow the motion | −3% hip demand; the demand is gravity-dominated |
| hold the box closer | already closer than the reference; the box is empty |
| adaptation / online compensation | cannot add torque to a saturated actuator; and driving the policy toward the reference measurably **worsens** the task — −39.5% tracking error but 0/6 box lifts, because the reference is infeasible for this robot |
| more hardware runs on the current policy | the failure is reproducible and understood |

On the last row specifically: §II-C of the ACC-2026 paper requires the desired trajectory
to be reachable through the input, `ẋ_d − f(x) − Ke ∈ Im(g(x))`. A motion retargeted from
a demonstrator with different limb strength does not satisfy that, so driving `e → 0`
steers toward a point the robot cannot reach.

---

## 7. Reproducing the measurements

The hip-demand measurement runs on the harness as it stands:

```bash
PY=~/.holosoma_deps/miniconda3/envs/hssim/bin/python
M=../holosoma/src/holosoma/holosoma/data/motions/x2_31dof/whole_body_tracking/box_multispeed

OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 $PY adaptation/adapt_experiments_isaac.py \
  --only frozen_npz --seeds 3 --steps 734 --obs-noise off --dr no-push \
  --motion $M/box_speed100.npz \
  --record-seed 600 --record-dir adaptation/isaac_runs/hipcheck \
  --out-dir adaptation/isaac_runs/hipcheck
```

Then apply the pass criterion in section 5 to the recorded `actions`.

**Use `--dr no-push`, not `--dr none`.** `--dr none` drops every randomisation term whose
key matches `mass`, which includes `randomize_object_rigid_body_mass_startup`. Training
sampled the box as `0.1 + U(0.3, 2.0)` kg; `--dr none` evaluates on the 0.1 kg URDF base
mass, below the trained range and with the arms effectively unloaded. `--dr no-push`
keeps the object mass randomisation while still dropping the push randomiser.

The sim-vs-hardware contact numbers (8.08° vs 4.84° target tracking, ankle roll 0.036 rad
vs 0.30 rad) come from replaying a hardware run's recorded actions through the simulator
so both sides execute identical commands. That needs a small addition to the harness and
is not reproducible with the flags in this branch.

## 8. Where adaptation does help

Not with this. But once the lift works, the ACC layer adaptation is established for
**actuator degradation**, at gain `1e-4` (the inherited `3e-4` is what made it harmful):

| condition | frozen | adapted | placement |
|---|---|---|---|
| healthy | 734 | 734 | 10/10 → 10/10 |
| right knee 0.3 | 328 | 668 | 1/10 → **7/10** (p = 0.0099) |
| left knee 0.3 | 204 | 150 | 1/10 → 0/10 |

It helps on the fault it was tuned for, is free when the robot is healthy, and harms the
mirror-image fault — which the paper's own ACE attribution predicts, having found no
causal handle on the left knee.
