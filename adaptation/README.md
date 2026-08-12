# Causality-guided layer adaptation on the X2 box-pickup policy

Porting the ACC-2026 adaptation law (M. Taheri, S.-J. Chung, F. Y. Hadaegh, *"Closing the
Loop Inside Neural Networks: Causality-Guided Layer Adaptation for Fault Recovery
Control"*) onto the v31 box-pickup policy in **our** Isaac Sim / holosoma environment,
rather than the MuJoCo testbed it shipped with.

**Headline.** On a healthy policy the adaptation reproduces its published tracking result
(−40% leg tracking error, versus −37% in his testbed) but costs the task: the baseline
completes the pickup on 6/6 seeds, every run at his default gain and mask falls within
2.5 s on 6/6. **Under an actual actuator fault — the regime the method is actually for —
adaptation wins decisively**, extending survival 53% with complete seed separation
(p = 0.0011), *provided the error mask excludes the leg joints*.

---

## 1. Is the adaptation online?

Yes, fully online, and cheap enough to deploy. Concretely:

| | |
|---|---|
| when it runs | every control step, 50 Hz (`ctrl_dt` = 0.02 s) |
| what drives it | the live joint tracking error `q − q_ref` measured that step |
| what it changes | one weight matrix, layer 2 of 4 (128×256 = 32,768 of 251,776 params) |
| cost per step | forward 1.93 ms + update 3.31 ms = **5.25 ms, 26% of the 20 ms budget** |
| dependencies | numpy only, no torch, no GPU |
| training involved | none — no gradients of a learned objective, no replay, no offline phase |
| model of the plant | none — no counterfactual model, no system identification, no probing |

Each control step does exactly one forward pass with the current weights and one backward
pass through the layers *above* the adapted one:

```
delta_L   = g(x)^T P e                            error signal, in ACTION space
delta_l   = Psi_l(a_l) . W_{l+1}^T delta_{l+1}    backprop through activation Jacobians
Wdot      = Gamma . delta z^T  -  gamma . W       Lyapunov update + leakage
W        <- W + dt * Wdot
```

Two properties worth being explicit about, because "online" can mean several things:

- **It adapts from the first control step** (`engage_step=0`), not after a trigger. There
  is no detector deciding that something has gone wrong; it is always running.
- **It does not persist across episodes.** `reset()` restores the frozen weights `W0` at
  every episode start, so nothing is carried over or accumulated between runs. It is
  online *within* an episode, not lifelong learning. On hardware this means each engage
  starts from the trained policy.

There is also **no performance monitor and no reversion**. The only safety net is a norm
bound, `‖W − W0‖_F < 5`, which trips a hard reset to `W0`. Nothing checks whether the
robot is doing better or worse. This is why adaptation is *not* guaranteed to be at least
as good as the frozen policy, and in our healthy-robot runs it is much worse.

---

## 2. How it is wired into our environment

The adapter is environment-agnostic; the integration supplies four things per step.

| contract item | our source |
|---|---|
| observation, 164-dim | `torch.cat([obs[k] for k in algo.actor_obs_keys])`, holosoma's own ordering |
| joint error `e` | `sim.dof_pos − ref_pos[frame]`, 31 joints, policy order |
| `action_scale`, `Kp` | from the exported policy metadata (nominal values) |
| mass matrix (`gx_level=2`) | PhysX `get_generalized_mass_matrices()`, Schur complement |

Correctness checks that gate every number below:

- **The numpy export is the same function as the torch checkpoint.** Feeding both the
  identical observation each step, max action difference is **4.2e-6** over 200 steps —
  float32 rounding. Run it yourself with `--check-export 200`.
- **The mass matrix is unscrambled.** PhysX reports 37×37 with the 6 root DoFs first and
  in its own joint ordering; both the root offset and the permutation to policy order are
  undone, then reduced by the Schur complement `M_jj − M_jb M_bb⁻¹ M_bj`.
- **This task is chaotic.** A 1e-6 action perturbation moves leg tracking error by ~1.7°
  over 2.4 s. Single-seed comparisons are meaningless here; everything is multi-seed, and
  the baseline is `frozen_npz` (the numpy policy at gain 0) rather than the torch policy,
  so the only difference from an adapted run is the adaptation itself.

---

## 3. Results on the healthy policy

6 seeds × 734 steps, observation noise off, push randomizer dropped, motion clip pinned to
`box_speed100`. `his_*` runs his shipped leak `−γW`; `w0_*` runs `−γ(W−W0)`.

| variant | survival (s) | never fell | box handled |
|---|---|---|---|
| `frozen_npz` (control) | 14.68 ± 0.00 | 6/6 | 6/6 |
| `frozen` (torch) | 13.81 ± 1.46 | 4/6 | 6/6 |
| `his_g3e-4_gx1` (his defaults) | 2.49 ± 0.27 | 0/6 | 0/6 |
| `w0_g3e-4_gx1` | 2.45 ± 0.54 | 0/6 | 0/6 |
| `his_g1e-5_gx1` | 4.86 ± 2.09 | 0/6 | 5/6 |
| `w0_g1e-5_gx1` | 4.91 ± 1.43 | 0/6 | 5/6 |
| `w0_g3e-4_gx2_schur` | 3.09 ± 1.45 | 0/6 | 0/6 (diverged 6/6) |
| `w0_g1e-5_gx2_schur` | 2.76 ± 0.23 | 0/6 | 0/6 |
| `w0_g3e-4_waistonly` | 13.10 ± 3.29 | 4/6 | 6/6 |
| `w0_g3e-4_noankle` | 2.26 ± 0.24 | 0/6 | 0/6 |

**The tracking claim reproduces.** Comparing on the 145-step window every variant survived
(seed 600), leg error goes 14.82° frozen → 8.88° his law → 8.01° corrected leak. That is
−40%, against −37% in his MuJoCo testbed from a nearly identical 14.69° baseline. The law
does what it says it does, and it transfers across simulators.

**What does not transfer is the implication.** His frozen baseline falls at 1.94 s and
never completes the motion (0/32 seeds), so there was no balance margin to lose and
adaptation could only add. Ours survives 14.68 s and completes the task 6/6, and the same
law spends that margin buying tracking accuracy the task did not need.

`gx_level=2` being worse is a reproduction, not a new finding — he measured the same and
explains it: inverse-inertia weighting aims the correction at the lightest joints (wrists,
head) whose tracking errors are irrelevant. Our contribution is only that it holds with
the *true* floating-base inertia from PhysX, not a diagonal surrogate.

Video: `../box_pickup/videos/isaac_adapt_frozen_vs_adapted_seed600.mp4`

---

## 4. Results under an actuator fault

The paper is about **fault recovery**, and applying a recovery controller to a healthy
policy is out of its domain. `--fault` scales the PD stiffness of matching joints;
holosoma computes torque in python as `kp_scale * p_gains * (target − q) − ...`, so this
is a genuinely weak actuator that still moves but sags under load. Neither the policy nor
the adapter is told the fault happened — both keep using the nominal `Kp`.

Right knee at 30% stiffness, 6 seeds, all randomization off:

| variant | survival (s) | reached set-down | lifted box |
|---|---|---|---|
| `frozen_npz` | 7.43 ± 2.76 | 4/6 | 5/6 |
| **`w0_g3e-4_waistonly`** | **11.36 ± 0.42** | **6/6** | 6/6 |
| `his_g1e-5_gx1` | 3.77 ± 0.12 | 0/6 | 4/6 |
| `w0_g1e-5_gx1` | 4.12 ± 0.15 | 0/6 | 6/6 |
| `his_g3e-4_gx1` (his defaults) | 2.23 ± 0.10 | 0/6 | 0/6 |
| `w0_g3e-4_gx1` | 2.63 ± 0.87 | 0/6 | 0/6 |

Per-seed survival separates **completely**:

```
frozen     [ 66, 393, 430, 433, 452, 456]   max 456
waistonly  [522, 568, 578, 579, 580, 581]   min 522
```

Exact one-sided Mann-Whitney p = **0.0011**, the floor achievable with 6 versus 6. Median
survival +34%, mean +53%, variance collapses from ±2.76 s to ±0.42 s, and the seed where
frozen crumples at 1.3 s is rescued to 10.4 s.

**The error mask, not the gain, decides everything here.** `waistonly` runs the *same*
3e-4 gain that is catastrophic with his default legs+waist mask. Same law, same gain, same
fault: include the leg joints in the regulated error and the robot is down in 2.2 s having
never lifted the box; exclude them and you get the result above. The legs are where
balance lives on a floating base, so regulating their tracking error fights the balance
the policy is maintaining; the waist carries error the policy can afford to correct.

This is where our environment disagrees with his most sharply. `RESULTS.md` measures the
mask as worth ~+0.8 tracked steps and calls it a stability enabler rather than a source of
performance. Here it is the entire result.

Video: `../box_pickup/videos/isaac_fault_knee03_frozen_vs_waistadapt.mp4`

---

## 5. Defects found

**The leak decays toward zero, not toward the trained weights.** The shipped law uses
`−γW`, so the sigma-modification pulls layer 2 toward the origin. Driving the adapter with
*perfect* tracking (zero error) for one 734-step episode still erodes **13.7% of the
layer's weight norm**, independent of gain. `−γ(W−W0)` leaves it at exactly 0.0. At his
low gain essentially all the weight change is this decay rather than anything
error-driven. Fixed in `W0LeakAdapter`. Honest caveat: **fixing it does not change the
outcome** — `w0_g3e-4_gx1` fails as hard as `his_g3e-4_gx1`. It is a real bug, but it is
not why the adaptation fails on a healthy robot.

**A metric trap in our own scoring.** The box-success criterion credits any run that ends
while the box is still up, so the low-gain fault variants scored 4–5/6 "box handled" while
reaching the set-down phase **0/6** — they died at ~200 steps mid-carry. This is the same
composite-metric trap his README warns about. Always report "reached set-down" or survival
alongside box success.

**Two environment bugs, fixed here.** `--dr none` dropped the action-delay *setup* term,
leaving `env._randomize_ctrl_delay` undefined so `joint_control.reset()` threw. And output
paths were resolved after the `chdir` into holosoma, so results silently landed there.

---

## 6. Reproducing

Environment setup is in [`../SETUP_ISAAC.md`](../SETUP_ISAAC.md). Then:

```bash
PY=~/.holosoma_deps/miniconda3/envs/hssim/bin/python
cd <repo root>

# Sanity: is the numpy export the same function as the torch checkpoint?
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 $PY adaptation/adapt_experiments_isaac.py \
    --check-export 200

# Healthy policy, all variants (~2 h)
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 $PY adaptation/adapt_experiments_isaac.py \
    --seeds 6 --steps 734 --obs-noise off --dr no-push \
    --out-dir adaptation/isaac_runs

# Under a knee fault (~1 h)
OMNI_KIT_ACCEPT_EULA=1 CUDA_VISIBLE_DEVICES=0 $PY adaptation/adapt_experiments_isaac.py \
    --only frozen_npz,his_g3e-4_gx1,w0_g3e-4_gx1,w0_g1e-5_gx1,w0_g3e-4_waistonly \
    --seeds 6 --steps 734 --obs-noise off --dr none --fault "right_knee:0.3" \
    --out-dir adaptation/isaac_runs/fault_knee03

# Render any two recorded rollouts side by side (MuJoCo, needs the mjlab venv)
mjlab/.venv/bin/python box_pickup/render_side_by_side.py \
    a.npz "LEFT" b.npz "RIGHT" out.mp4
```

Useful flags: `--only`, `--seeds/--seed0`, `--fault SUBSTR:SCALE`, `--dr no-push|none|all`,
`--obs-noise on|off`, `--record-seed N`.

---

## 7. Running it on the real X2

Hardware port of the same law, in `../agibot_control_functions/`:

| file | what |
|---|---|
| `layer_adapt.py` | the adapter, numpy only, no ROS. Bit-identical to the Isaac `W0LeakAdapter`: over 400 steps on an identical obs/error stream, max weight difference **0.0e+00**. |
| `deploy_x2_box_adapt.py` | the deploy loop. Imports the observation builder, PD gains, filters and safety ladder from `deploy_x2_box_pickup.py` verbatim, so the ONLY difference from the working frozen deploy is where the action comes from. |
| `compare_adapt_runs.py` | turns the run logs into a frozen-vs-adapted verdict. |

**Get `deploy_x2_box_pickup.py` working first.** This script is that script plus an
adapter; if the frozen policy is not already running on the robot, nothing here is
interpretable.

### The protocol

Adaptation is compared against a control arm produced by the *same code path* with the
update disabled, so the only difference between the two arms is the adaptation:

```bash
cd agibot_control_functions

# 0. On the robot, with no motion at all: does the numpy adapter reproduce the
#    deployed policy, and does the update fit in the 20 ms tick on THIS CPU?
python deploy_x2_box_adapt.py --self-check

# 1. Frozen control arm.  gain 0 makes the update a no-op.
python deploy_x2_box_adapt.py --engage --gain 0    --tag frozen

# 2. Adapted arm.
python deploy_x2_box_adapt.py --engage --gain 3e-4 --tag adapted

# ... alternate 1 and 2 at least three times each, then:
python compare_adapt_runs.py 'run_logs/*_box_adapt_*.csv'
```

`compare_adapt_runs.py` averages error only over the frame window every run reached (a
run that falls early otherwise looks like the most accurate one) and refuses to call a
winner with fewer than 3 runs per arm. That is not pedantry: in sim a 1e-6 action
perturbation moved leg tracking error 1.7 deg over 2.4 s, so a single A/B pair on this
task is noise.

### What is different from the sim runs, and what it costs

- **`--mask waist` is the default here**, not the paper's legs+waist. Section 4 is why:
  waist-only is the only configuration that ever beat the frozen policy. The leg EMA
  filter in the deploy loop (`--leg-filter 0.9`) is a second reason — it attenuates leg
  commands, so adapting leg joints partly fights the filter.
- **`--gain 3e-4` is inherited from v31, but the sim evidence is v31 and the deploy
  default is v33.** One thing does transfer exactly: the adapter's input map is
  `action_scale * Kp`, and since `action_scale = cfg_scale * effort / Kp`, the v33 waist
  retune (kp 20 -> 60, scale 0.6 -> 0.2) leaves that product **identical** (12.0 for
  waist pitch/roll, 30.0 for yaw). The error signal the adapter sees on the waist is
  unchanged. The policy weights are not, so this is a defensible starting point, not a
  validated one. Re-running section 4's fault experiment on v33 would settle it.
- **Hardware-only guards**, none of which exist in the paper: a weight-drift bound that
  reverts to the trained weights and latches adaptation off for the rest of the run
  (`--max-drift 1.0`, tighter than sim's 5.0; a healthy waist-only run sits near 0.04);
  a clamp on how far the adapted action may deviate from the frozen action
  (`--max-action-dev`, costs one extra forward pass); and a loop-deadline watchdog that
  disables adaptation if the update starts starving the 50 Hz loop.
- **Adaptation still never persists.** `reset()` runs at every engage, so each take
  starts from the trained weights.

### Expected outcome, stated in advance

On a *healthy* robot this should do nothing good — that is what section 3 measured, and
it is what the method is for: **fault recovery**. The honest test is to degrade the
robot first (the hardware analogue of `--fault`: a weakened joint, added payload, a
worn actuator) and check whether adaptation extends the motion the frozen policy can no
longer complete. Running only the healthy comparison and finding no improvement
reproduces a result we already have.

The per-tick trace lands in `run_logs/<stamp>_..._adapt.csv`: drift, adapted-joint and
leg tracking error, action deviation, clamp count and loop time, one row per 20 ms.

---

## 7b. Why the hardware failure is not the fault adaptation recovers

Worth stating plainly, because it is the first thing to check before reaching for the
adapter. The 2026-08-12 runs fail because the policy commands **torque it cannot get**
on joints whose saturation is absorbed by contact in training. `action_scale =
0.25*effort_limit/kp`, so |a| = 4 is already the effort limit, and the policy runs
|a| = 10-40 on the ankle rolls and wrists. In Isaac the ground and the box push back and
those joints stay at the reference (`right_ankle_roll` at +0.03 rad); on hardware they go
to their stops (+0.34 rad, 96% of the motion) and the robot stands on the edges of its
feet until it topples at 2.9 s. `../run_logs/_sim_vs_real.py` is the comparison.

That is a *command* defect, not a degraded actuator, and it is outside this method's
domain twice over: the adapter has no term for "this request is unachievable", and the
joints involved are ankles, which section 4 shows must be excluded from the error mask or
the robot is down in 2.2 s. The fix is to bound the action -- `action_clip_isaac.py`,
7 seeds, survival unchanged from baseline on 7/7 with the ankle rolls and wrists clipped
to |a| = 4, against 0/9 when every joint is clipped. That is now
`deploy_x2_box_pickup.py --action-clip 4`.

Adaptation stays what section 4 measured it to be: worth running once the robot completes
the task unaided, as insurance against actuator degradation, at `--mask waist`.

---

## 8. Files

| path | what |
|---|---|
| `ACC_ADAPTATION_PACKAGE/` | his package, unmodified. `ace_adapt.py` is the method. |
| `adapt_experiments_isaac.py` | the experiment harness: variants, fault injection, metrics |
| `action_clip_isaac.py` | does bounding the action to the effort limit cost the task? |
| `eval_adapt_isaac.py` | single-run Isaac rollout loop and the adapter wiring |
| `paths.py` | resolves checkpoint / motion / policy, all env-overridable |
| `isaac_runs/` | per-seed JSON summaries and recorded rollouts |
| `isaac_runs/fault_knee03/` | the fault experiment |
| `FOR_MENTOR/` | rollout log, reference clip, config and videos sent to him |
| `dump_for_mentor.py` | produces the per-step observation log in `FOR_MENTOR/` |
| `../agibot_control_functions/layer_adapt.py` | the adapter, hardware build |
| `../agibot_control_functions/deploy_x2_box_adapt.py` | on-robot deploy with adaptation |
| `../agibot_control_functions/compare_adapt_runs.py` | frozen-vs-adapted analysis of run logs |

---

## 9. Open questions

1. **Confirm the waist-only result on other faults and held-out seeds.** Everything above
   is one fault (right knee, 30%) on seeds 600–605, which is the pool his configuration
   was tuned on. A hip or ankle fault, and a fresh seed pool, would make it airtight.
2. **A balance-targeted error signal.** His own open question, and our results are the
   evidence for it: the adapter regulates joint error and hopes balance follows, and on a
   floating base it does not. Defining `delta_L` on a CoM/ZMP error via the centroidal
   mapping targets what actually fails.
3. **Gating.** Engaging only once tracking error exceeds a threshold would restore an "at
   least as good as frozen" floor, which the law currently has no mechanism for.
