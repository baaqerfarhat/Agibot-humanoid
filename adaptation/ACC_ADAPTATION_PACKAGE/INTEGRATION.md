# Wiring the adapter into your own environment

`ace_adapt.py` has no simulator dependency. It needs four things from you, once at setup and two
per control step.

## Setup

```python
from ace_adapt import ExportedPolicy, AdaptConfig, LayerAdapter

policy  = ExportedPolicy("assets/x2_box_policy_v31.npz")
adapter = LayerAdapter(
    policy,
    AdaptConfig(layer=2, gain=3e-4, leak=1e-2, gx_level=1,
                error_joints=("hip", "knee", "ankle", "waist"), engage_step=0),
    joint_names=policy.meta["joint_names"],   # ordering must match your joint vector
)
```

`action_scale` and `joint_stiffness` default to the values in the policy metadata. Pass them
explicitly if your controller uses different ones — **`gx_level=1` multiplies by `Kp`, so a wrong
`Kp` silently rescales the correction per joint.**

## Per control step

```python
adapter.reset()                       # at the start of EVERY episode

while not done:
    obs    = env.observation()        # (1) 164-d, see contract below
    action = adapter.act(obs)         # forward pass through the ADAPTED weights
    env.apply(action)                 # (2) your action -> target -> plant
    adapter.update(env.joint_error(), control_dt)   # (3) error, (4) control period
```

Order matters: `act` then step the plant then `update`, so the update sees the error the action
produced. `control_dt` is the **control** period (0.02 s at 50 Hz), not the physics timestep.

---

## The four contracts

### 1. Observation — 164-d, ALPHABETICAL term order

```
actions(31) | base_ang_vel(3) | dof_pos(31) | dof_vel(31) | motion_command(62) | motion_ref_ori_b(6)
```

- `actions` — the previous **raw** action (before any filtering or rate limiting)
- `base_ang_vel` — torso IMU gyro, expressed in the **torso body frame**
  (in MuJoCo: `xmat[torso].T @ omega_world`; note `mj_objectVelocity(..., flg_local=1)` returns
  the *inertial* frame, which is **not** the body frame)
- `dof_pos` — `q - default_joint_pos`
- `dof_vel` — joint velocities
- `motion_command` — `[ref_joint_pos(31), ref_joint_vel(31)]` at the current motion frame
- `motion_ref_ori_b` — first two columns of the rotation matrix of
  `conj(q_torso) ⊗ q_ref`, row-major → `[m00, m01, m10, m11, m20, m21]`.
  The reference is yaw-aligned to the robot at engage:
  `yaw_offset = yaw(q_torso_0) ⊗ inv(yaw(q_ref_0))`, applied to every reference quaternion.

The alphabetical ordering is not cosmetic — it is what the exporting framework does and what the
weights expect.

### 2. Action → plant

```
target = action * action_scale + default_joint_pos
target = (1 - 0.8) * target + 0.8 * prev_target     # LEG joints only; arms unfiltered
target = prev_target + clip(target - prev_target, -0.15, +0.15)   # all joints, rad/step
```
then a PD servo with the policy's `joint_stiffness`/`joint_damping` scaled by **1.2**.

### 3. Joint error

`q - q_ref` for the 31 actuated joints in radians, same ordering as `joint_names`. The adapter
applies the `error_joints` mask itself — pass the full vector.

### 4. Control period

Seconds per control step (0.02 at 50 Hz). The update integrates `Wdot` with this.

---

## If your plant is not position-controlled

`gx_level=1` assumes the policy output becomes a joint target through a PD servo, which makes
`d(tau)/d(a) = Kp diag(action_scale)` exact. If your action maps to torque differently, replace
`LayerAdapter.delta_L` with your own `g(x)^T P e`. That is the only place the plant model enters.

If you have the mass matrix and want to try the full mapping, pass `mass_matrix_fn` (returning
the actuated 31×31 block) and set `gx_level=2` — but note this measured **worse** than level 1
here, for the reason given in README §4.

---

## Sanity checks before trusting any result

1. **`adapter.diverged` must be False.** A diverged run scores as a perfect one unless you check.
2. **`adapter.weight_drift`** should land around 0.2–0.5 for the confirmed configuration. Orders
   of magnitude larger means the gain is too high for your plant.
3. **Frozen baseline first.** Run with `adapter=None`. If your frozen numbers differ from
   §2 of the README, the observation contract is wrong somewhere — fix that before evaluating
   adaptation.
4. **Score every arm identically.** Do not compare an arm run from a restored simulator snapshot
   against one run continuously (see README §6).
5. **Run a matched random null.** Perturb the same layer with a random `dW` of the same Frobenius
   norm, ~20 draws. If your gain does not clearly beat that distribution, you are measuring
   "perturbing this layer helps", not the law. `evaluate.py --null` does this.
