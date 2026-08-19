# SONIC harness spec — verified edit sites, exact obs layout, and the bridge design

19 Aug 2026. Everything below is extracted from the shipped artifacts (ONNX graphs,
`observation_config.yaml`, `g1_deploy_onnx_ref` C++ source) and verified where marked.

## 1. Edit sites — VERIFIED on the shipped decoder (core/sonic_edit.py, 3/3 PASS)

```
decoder obs[1,994] -> action[1,29]
  module.decoders.g1_dyn.module.12.bias   [29]       b6 site (output bias)
  onnx::MatMul_142                        [512,29]   w6 site (output weights)
```
Both are graph initializers. `sonic_edit.edit_decoder(db, theta)` writes a variant ONNX;
verified exact at float32: b6 shifts the action by exactly `db`; w6 scales it by exactly
`(1+theta)` (final layer affine). Any ONNX consumer — onnxruntime AND the C++ TensorRT
deploy (it parses this file) — executes the edit. **The adaptation method needs nothing
else from the model.**

## 2. Decoder observation layout — exact (dims from the C++ registry; sums to 994 ✓)

| slice | term | dim | source |
|---|---|---|---|
| 0:64 | token_state | 64 | encoder output (see §4) |
| 64:94 | his_base_angular_velocity_10frame_step1 | 30 | 10 frames × 3, oldest-first |
| 94:384 | his_body_joint_positions_10frame_step1 | 290 | 10 × 29 |
| 384:674 | his_body_joint_velocities_10frame_step1 | 290 | 10 × 29 |
| 674:964 | his_last_actions_10frame_step1 | 290 | 10 × 29 |
| 964:994 | his_gravity_dir_10frame_step1 | 30 | 10 × 3 |

Order = the `observations:` list order in `observation_config.yaml` (offsets assigned in
config order). Histories: `newest_first=false` (OLDEST first), sampled every
`control_dt × step` (control 50 Hz). The yaml's "436" header comment is stale — the
registry dims are authoritative and match the ONNX input exactly.

## 3. Action mapping (policy_parameters.hpp)

```
joint_target = action × g1_action_scale + default_angle        (per joint, 29)
action_scale = 0.25 × effort_limit / stiffness;  stiffness = armature × (2π·10Hz)²
kps, kds     = const arrays in policy_parameters.hpp
```
Same PD-target structure as the X2 walker — all fault machinery concepts (torque limit,
kp scale, gain, friction) transfer directly, applied on the sim side (MuJoCo model).

## 4. Encoder — only needed OFFLINE

`encoder: obs[1,1762] -> encoded_tokens[1,64]`. Its inputs are REFERENCE-MOTION features
(motion joint pos/vel @10frame step5, root z, anchor orientation, mode flags) — not robot
state. **For a fixed task/motion the token sequence can be precomputed once and replayed**,
so the online loop needs only the decoder. This collapses the bridge to: robot-state
histories → obs[994] with cached tokens → decoder ONNX → action → PD targets.

## 5. Two ways to run experiments

- **Official (GPU machine):** MuJoCo sim (`run_sim_loop.py`, .venv_sim) + C++ deploy
  (`deploy.sh sim`; needs TensorRT 10.13 exact). Adaptation = point the deploy at
  `model_decoder_edited.onnx` per episode; automate start/motion/stop over the ZMQ
  keyboard/state interfaces (`g1_debug` topic, keyboard publisher protocol).
- **Bridge (laptop, CPU, fallback):** replace the C++ deploy with a Python loop:
  unitree_sdk2py DDS (lowstate ← sim, lowcmd → sim, loopback) + §2 obs builder +
  cached tokens + ORT decoder + §3 action mapping. Risk concentrates in the obs builder;
  §2/§3 pin everything except the state→lowstate field mapping, which `run_sim_loop.py`'s
  own publisher defines (read it when building).

## 6. Episodic adaptation loop (either backend)

```
per candidate theta:  edit_decoder(db=theta_b6 or theta=theta_w6) -> variant.onnx
                      run N fresh episodes (motion task from sample_data)
                      score = realised metric (tracking completion / distance / uprightness)
core/episodic_search.py drives it: seeds_per_gen >= 2, refit-off control, ACE arm,
settled = elite-mean of final generation, held-out episodes disjoint.
```
Fault suite (sim side, MuJoCo model): torque_limit scale, kp scale, joint gain/offset,
friction, payload — the walker suite on the G1. Gates before any search: headroom +
(for clipped classes) envelope ceiling — remembering the walker verdict: the contract
selects the class.
